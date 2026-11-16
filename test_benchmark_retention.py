import copy
from contextlib import closing
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import portalocker

from benchmarking.download import PROJECT
from benchmarking.retention import checked_path, cleanup_memories, run_lock_path, touch_run
from benchmarking.runner import code_version, resume_identity_matches


class RetentionTests(unittest.TestCase):
    def setUp(self):
        parent = PROJECT / "benchmarks" / "runs" / "_offline_test_tmp"
        parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "runs"
        self.root.mkdir()

    def make_run(self, number, *, mode="mem0", cases=1, nested=False):
        folder = self.root / ("group" if nested else "") / f"run{number:02}"
        folder.mkdir(parents=True)
        (folder / "run_manifest.json").write_text(json.dumps({"identity": {"settings": {"mode": mode}}}), "utf-8")
        (folder / "memory_retention.json").write_text(json.dumps({"last_used_ns": number}), "utf-8")
        (folder / "predictions.jsonl").write_text('{"hypothesis":"Boston"}\n', "utf-8")
        (folder / "metrics_deepseek.json").write_text('{"accuracy":1}', "utf-8")
        (folder / "completed").mkdir()
        (folder / "completed" / "case.json").write_text('{"checkpoint":true}', "utf-8")
        if mode == "mem0":
            (folder / "sdk_home").mkdir()
            (folder / "sdk_home" / "config.json").write_text("{}", "utf-8")
            for number in range(cases):
                attempt = folder / "cases" / f"question{number}" / "attempt"
                (attempt / "qdrant").mkdir(parents=True)
                (attempt / "qdrant" / "storage.sqlite").write_bytes(b"vector fixture")
                with closing(sqlite3.connect(attempt / "history.db")) as db:
                    with db:
                        db.execute("CREATE TABLE history (memory TEXT)")
                        db.execute("INSERT INTO history VALUES ('test memory')")
                for suffix in ("-wal", "-shm", "-journal"):
                    (attempt / f"history.db{suffix}").write_bytes(b"sidecar fixture")
                (attempt / "writes.jsonl").write_text('{"memory":"Boston"}\n', "utf-8")
                (attempt / "reader_input.json").write_text('{"retrieved":[]}', "utf-8")
        return folder

    def has_memory(self, folder):
        return bool(list((folder / "cases").glob("*/*/history.db")))

    def test_removes_oldest_two_experiments_and_preserves_all_results(self):
        folders = [self.make_run(i, cases=2) for i in range(1, 13)]
        report = cleanup_memories(root=self.root)
        self.assertEqual(len(report["pruned_runs"]), 2)
        self.assertEqual(len(report["retained_runs"]), 10)
        for i, folder in enumerate(folders):
            self.assertEqual(self.has_memory(folder), i >= 2)
            for name in ("predictions.jsonl", "metrics_deepseek.json", "completed/case.json",
                         "cases/question0/attempt/writes.jsonl", "cases/question0/attempt/reader_input.json"):
                self.assertTrue((folder / name).exists(), name)
            if i < 2:
                self.assertFalse((folder / "sdk_home").exists())
                self.assertFalse(list((folder / "cases").glob("*/*/qdrant")))
                self.assertFalse(list((folder / "cases").glob("*/*/history.db*")))
                self.assertTrue(json.loads((folder / "memory_retention.json").read_text())["memory_pruned"])
        second = cleanup_memories(root=self.root)
        self.assertEqual(second["memory_runs_found"], 10)
        self.assertEqual(second["pruned_runs"], [])

    def test_dry_run_only_previews_deletions(self):
        folders = [self.make_run(i) for i in range(1, 12)]
        original = {str(path): path.read_bytes() for folder in folders for path in folder.rglob("*") if path.is_file()}
        report = cleanup_memories(root=self.root, dry_run=True)
        self.assertEqual(report["pruned_runs"], [str(folders[0])])
        for path, content in original.items():
            self.assertEqual(Path(path).read_bytes(), content)

    def test_locked_old_experiment_is_protected_and_uses_a_retention_slot(self):
        folders = [self.make_run(i) for i in range(1, 12)]
        (self.root / ".locks").mkdir()
        with portalocker.Lock(str(run_lock_path(folders[0], self.root)), timeout=0):
            report = cleanup_memories(root=self.root)
        self.assertTrue(self.has_memory(folders[0]))
        self.assertFalse(self.has_memory(folders[1]))
        self.assertEqual(report["active_runs"], [str(folders[0])])
        self.assertEqual(len(report["retained_runs"]), 10)

    def test_new_active_run_reserves_slot_before_first_sdk_write(self):
        folders = [self.make_run(i) for i in range(1, 11)]
        active = self.make_run(11, mode="no-memory")
        (active / "run_manifest.json").write_text(json.dumps({"identity": {"settings": {"mode": "mem0"}}}), "utf-8")
        touch_run(active, self.root)
        report = cleanup_memories(root=self.root, active_run=active)
        self.assertFalse(self.has_memory(folders[0]))
        self.assertIn(str(active), report["retained_runs"])

    def test_all_busy_runs_defer_deletion_until_next_cleanup(self):
        from contextlib import ExitStack
        folders = [self.make_run(i) for i in range(1, 12)]
        (self.root / ".locks").mkdir()
        with ExitStack() as locks:
            for folder in folders:
                locks.enter_context(portalocker.Lock(str(run_lock_path(folder, self.root)), timeout=0))
            report = cleanup_memories(root=self.root)
        self.assertEqual(report["pruned_runs"], [])
        self.assertEqual(len(report["active_runs"]), 11)
        self.assertTrue(report["warnings"])
        report = cleanup_memories(root=self.root)
        self.assertEqual(report["pruned_runs"], [str(folders[0])])

    def test_oracle_and_unmanaged_directories_do_not_consume_slots(self):
        folders = [self.make_run(i) for i in range(1, 11)]
        self.make_run(11, mode="oracle")
        unmanaged = self.root / "unmanaged" / "cases" / "q" / "a"
        unmanaged.mkdir(parents=True)
        (unmanaged / "history.db").write_bytes(b"unmanaged")
        report = cleanup_memories(root=self.root)
        self.assertEqual(report["pruned_runs"], [])
        self.assertEqual(report["memory_runs_found"], 10)
        self.assertTrue(all(self.has_memory(folder) for folder in folders))
        self.assertTrue((unmanaged / "history.db").exists())

    def test_nested_output_and_resume_recency_are_supported(self):
        oldest = self.make_run(1, nested=True)
        for i in range(2, 12):
            self.make_run(i)
        touch_run(oldest, self.root)
        report = cleanup_memories(root=self.root)
        self.assertTrue(self.has_memory(oldest))
        self.assertEqual(report["pruned_runs"], [str(self.root / "run02")])

    def test_production_and_parent_paths_are_rejected(self):
        for path in (PROJECT / "data", PROJECT / "sessions", self.root, self.root / ".." / "production"):
            with self.assertRaises(ValueError):
                checked_path(path, self.root)

    def test_linked_database_tree_is_refused_before_any_deletion(self):
        folders = [self.make_run(i) for i in range(1, 12)]
        vector_file = folders[0] / "cases" / "question0" / "attempt" / "qdrant" / "storage.sqlite"
        # Model a Windows junction/reparse-point observation without requiring admin symlink privileges.
        from benchmarking.retention import _is_link
        with patch("benchmarking.retention._is_link", side_effect=lambda path: path == vector_file or _is_link(path)):
            report = cleanup_memories(root=self.root)
        self.assertEqual(report["pruned_runs"], [])
        self.assertTrue(report["warnings"])
        self.assertTrue(self.has_memory(folders[0]))
        self.assertTrue(vector_file.exists())

    def test_retention_migration_accepts_only_exact_approved_code_change(self):
        mapping = json.loads((PROJECT / "benchmarking" / "resume_compatibility.json").read_text("utf-8"))
        self.assertEqual(code_version()["adapter_sha256"], mapping["current_adapter_sha256"])
        previous = {"dataset": "s", "settings": {"model": "deepseek-chat"},
                    "version": {"adapter_sha256": mapping["previous_adapter_sha256"], "sdk": "unchanged"}}
        current = copy.deepcopy(previous)
        current["version"]["adapter_sha256"] = mapping["current_adapter_sha256"]
        self.assertTrue(resume_identity_matches(previous, current))
        for bad in ("data", "model", "sdk", "code"):
            changed = copy.deepcopy(current)
            if bad == "data":
                changed["dataset"] = "m"
            elif bad == "model":
                changed["settings"]["model"] = "another-model"
            elif bad == "sdk":
                changed["version"]["sdk"] = "changed"
            else:
                changed["version"]["adapter_sha256"]["runner.py"] = "unapproved change"
            self.assertFalse(resume_identity_matches(previous, changed))


if __name__ == "__main__":
    unittest.main()
