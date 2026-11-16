"""Keep at most ten experiments' SDK databases, while preserving evaluation results."""
import hashlib
import json
import os
import shutil
import stat
import time
from contextlib import ExitStack
from pathlib import Path

import portalocker

from .data import atomic_json
from .download import PROJECT

RUNS_ROOT = PROJECT / "benchmarks" / "runs"
KEEP_MEMORY_RUNS = 10
SQLITE_FILES = ("history.db", "history.db-wal", "history.db-shm", "history.db-journal")


def _is_link(path):
    info = path.lstat()
    return path.is_symlink() or bool(getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def checked_path(path, root=RUNS_ROOT):
    """Validate both lexical containment and resolved containment; refuse junctions."""
    root = Path(os.path.abspath(root))
    path = Path(os.path.abspath(path))
    try:
        parts = path.relative_to(root).parts
    except ValueError as exc:
        raise ValueError(f"Test memory path must be inside {root}: {path}") from exc
    if not parts:
        raise ValueError("The experiments root itself cannot be a run or deletion target")
    current = root
    for part in (None, *parts):
        if part is not None:
            current = current / part
        if current.exists() or current.is_symlink():
            if _is_link(current):
                raise ValueError(f"Refusing linked test memory path: {current}")
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()) or resolved == root.resolve():
        raise ValueError(f"Test memory path escapes experiments root: {path}")
    return path


def run_lock_path(run_dir, root=RUNS_ROOT):
    name = hashlib.sha256(str(Path(run_dir).resolve()).casefold().encode()).hexdigest() + ".lock"
    return Path(root) / ".locks" / name


def touch_run(run_dir, root=RUNS_ROOT):
    run_dir = checked_path(run_dir, root)
    _write_marker(run_dir, {
        "keep_last_runs": KEEP_MEMORY_RUNS, "last_used_ns": time.time_ns(), "memory_pruned": False,
    }, root)


def _write_marker(run_dir, value, root):
    marker = checked_path(run_dir / "memory_retention.json", root)
    checked_path(marker.with_suffix(".json.tmp"), root)
    atomic_json(marker, value)


def _child_dirs(parent, root):
    checked_path(parent, root)
    if not parent.exists():
        return []
    children = []
    for child in parent.iterdir():
        checked_path(child, root)
        if child.is_dir():
            children.append(child)
    return sorted(children)


def _memory_targets(run_dir, root):
    """Enumerate only the SDK paths owned by this benchmark, never whole run folders."""
    targets = []
    for case in _child_dirs(run_dir / "cases", root):
        for attempt in _child_dirs(case, root):
            for name in ("qdrant", *SQLITE_FILES):
                path = checked_path(attempt / name, root)
                if path.exists():
                    if name != "qdrant" and not path.is_file():
                        raise ValueError(f"Expected SQLite file: {path}")
                    targets.append(path)
    home = checked_path(run_dir / "sdk_home", root)
    if home.exists():
        targets.append(home)
    return targets


def _check_tree(target, root):
    checked_path(target, root)
    if target.is_dir():
        for current, directories, files in os.walk(target, followlinks=False):
            for name in directories + files:
                checked_path(Path(current) / name, root)


def _discover(root):
    for current, directories, files in os.walk(root, followlinks=False):
        # Never traverse SDK storage; also keep offline test fixtures out of real retention.
        directories[:] = [name for name in directories if name not in (".locks", "cases", "sdk_home")
                          and not (Path(current) == RUNS_ROOT and name == "_offline_test_tmp")
                          and not _is_link(Path(current) / name)]
        if "run_manifest.json" in files:
            yield Path(current)
            directories[:] = []


def cleanup_memories(*, active_run=None, dry_run=False, root=RUNS_ROOT):
    """Active runs count toward the limit but are never deleted. No LLM/API calls."""
    root = Path(os.path.abspath(root))
    root.mkdir(parents=True, exist_ok=True)
    if _is_link(root):
        raise ValueError(f"Refusing linked experiments root: {root}")
    lock_dir = root / ".locks"
    checked_path(lock_dir, root)
    lock_dir.mkdir(exist_ok=True)
    cleanup_lock = checked_path(lock_dir / "memory_retention.lock", root)
    with portalocker.Lock(str(cleanup_lock), mode="a", timeout=5):
        report = {"keep_last_runs": KEEP_MEMORY_RUNS, "dry_run": dry_run,
                  "memory_runs_found": 0, "retained_runs": [], "pruned_runs": [],
                  "active_runs": [], "warnings": [], "results_preserved": True}
        candidates = []
        paths = set(_discover(root))
        active_run = checked_path(active_run, root) if active_run is not None else None
        if active_run is not None:
            paths.add(active_run)
        for run_dir in sorted(paths):
            try:
                checked_path(run_dir, root)
                manifest = checked_path(run_dir / "run_manifest.json", root)
                identity = json.loads(manifest.read_text("utf-8"))["identity"]
                if identity["settings"]["mode"] != "mem0":
                    continue
                targets = _memory_targets(run_dir, root)
                if not targets and run_dir != active_run:
                    continue
                touched = manifest.stat().st_mtime_ns
                marker = checked_path(run_dir / "memory_retention.json", root)
                if marker.exists():
                    touched = int(json.loads(marker.read_text("utf-8"))["last_used_ns"])
                candidates.append((touched, run_dir, targets))
            except (OSError, ValueError, KeyError, TypeError) as exc:
                report["warnings"].append(f"Skipped {run_dir}: {exc}")
        report["memory_runs_found"] = len(candidates)
        candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
        with ExitStack() as locks:
            active, inactive = [], []
            for entry in candidates:
                path = entry[1]
                if path == active_run:
                    active.append(entry)
                    continue
                lock_path = checked_path(run_lock_path(path, root), root)
                lock = portalocker.Lock(str(lock_path), mode="a", timeout=0)
                try:
                    locks.enter_context(lock)
                    inactive.append(entry)
                except portalocker.exceptions.LockException:
                    active.append(entry)
            slots = max(0, KEEP_MEMORY_RUNS - len(active))
            keep, prune = inactive[:slots], inactive[slots:]
            report["active_runs"] = [str(item[1]) for item in active]
            report["retained_runs"] = [str(item[1]) for item in active + keep]
            for touched, run_dir, targets in prune:
                try:
                    # Validate every target, including descendants, before any recursive delete.
                    for target in targets:
                        _check_tree(target, root)
                    if not dry_run:
                        for target in targets:
                            checked_path(target, root)
                            if target.is_dir():
                                shutil.rmtree(target)
                            else:
                                target.unlink()
                        _write_marker(run_dir, {
                            "keep_last_runs": KEEP_MEMORY_RUNS, "last_used_ns": touched,
                            "memory_pruned": True, "pruned_at_ns": time.time_ns(),
                        }, root)
                    report["pruned_runs"].append(str(run_dir))
                except (OSError, ValueError) as exc:
                    report["retained_runs"].append(str(run_dir))
                    report["warnings"].append(f"Could not prune {run_dir}: {exc}")
            if len(report["retained_runs"]) > KEEP_MEMORY_RUNS:
                report["warnings"].append("Busy or unsafe databases prevented the limit; cleanup will retry next run.")
        return report
