import copy
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from benchmarking.cli import main
from benchmarking.data import history_rounds, iter_questions, select_rows, validate_case
from benchmarking.runner import create_memory, historical_clock, memory_config, rebuild_outputs, run_case, run
from benchmarking.scoring import official_evaluator, summarize
from benchmarking.download import PROJECT


def test_directory():
    parent = PROJECT / "benchmarks" / "runs" / "_offline_test_tmp"
    parent.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=parent)


def example():
    return {"question_id": "case1", "question_type": "knowledge-update", "question": "Where do I live?",
            "question_date": "2023/01/05 (Thu) 08:00", "answer": "SECRET_GOLD_Boston", "answer_session_ids": ["s2"],
            "haystack_session_ids": ["s2", "s1"], "haystack_dates": ["2023/01/03 (Tue) 10:00", "2023/01/01 (Sun) 12:00"],
            "haystack_sessions": [[{"role": "user", "content": "I moved to Boston yesterday", "has_answer": True},
                                    {"role": "assistant", "content": "Good luck!", "has_answer": True}],
                                   [{"role": "user", "content": "I live in Seattle"},
                                    {"role": "assistant", "content": "Nice city!"}]]}


class BenchmarkTests(unittest.TestCase):
    def test_history_sorts_sessions_preserves_roles_and_removes_gold_annotations(self):
        case = example()
        original = copy.deepcopy(case)
        records = list(history_rounds(case))
        self.assertEqual([x["session_id"] for x in records], ["s1", "s2"])
        self.assertEqual(records[1]["messages"][0], {"role": "user", "content": "I moved to Boston yesterday"})
        self.assertNotIn("has_answer", json.dumps(records))
        self.assertNotIn("SECRET_GOLD", json.dumps(records))
        self.assertEqual(case, original)

    def test_orphan_and_trailing_messages_are_not_lost(self):
        case = example()
        case["haystack_sessions"][0] = [{"role": "assistant", "content": "early"}, {"role": "user", "content": "last"}]
        records = list(history_rounds(case))
        self.assertEqual([m["content"] for r in records for m in r["messages"]], ["I live in Seattle", "Nice city!", "early", "last"])

    def test_official_duplicate_session_ids_and_future_dates_are_preserved(self):
        case = example()
        case["haystack_session_ids"] = ["repeated", "repeated"]
        case["question_date"] = "2022/12/01 (Thu) 08:00"
        validate_case(case)
        self.assertEqual(len(list(history_rounds(case))), 2)

    def test_historical_clock_changes_only_dates_and_restores_helper(self):
        import mem0.memory.main as sdk
        before = sdk.generate_additive_extraction_prompt
        with historical_clock("2023/01/03 (Tue) 10:00"):
            prompt = sdk.generate_additive_extraction_prompt(new_messages="Yesterday I moved")
            self.assertIn("## Observation Date\n2023-01-03", prompt)
            self.assertIn("## Current Date\n2023-01-03", prompt)
            self.assertNotIn("## Custom Instructions", prompt)
        self.assertIs(sdk.generate_additive_extraction_prompt, before)

    def test_mem0_pipeline_no_labels_no_question_during_indexing_no_qa_writeback(self):
        memory = MagicMock()
        memory.benchmark_llm_usage = []
        memory.add.return_value = {"results": [{"id": "m1", "memory": "Moved to Boston"}]}
        memory.search.return_value = {"results": [{"id": "m1", "memory": "Moved to Boston", "score": 0.8,
                                                   "metadata": {"observed_at": "2023/01/03", "session": "s2"}}]}
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Boston"), finish_reason="stop")], usage=None)
        settings = {"mode": "mem0", "top_k": 8, "threshold": 0.1, "model": "deepseek-chat", "max_answer_tokens": 2000}
        with test_directory() as folder:
            (Path(folder) / "completed").mkdir()
            result, diag = run_case(example(), settings, Path(folder), client, lambda *_: memory)
            self.assertEqual(result, {"question_id": "case1", "hypothesis": "Boston"})
            self.assertEqual(memory.add.call_count, 2)
            for call in memory.add.call_args_list:
                self.assertTrue(call.kwargs["infer"])
                self.assertNotIn("timestamp", call.kwargs)
                serialized = json.dumps(call.args) + json.dumps(call.kwargs)
                for forbidden in ("has_answer", "answer_session_ids", "SECRET_GOLD", "Where do I live?"):
                    self.assertNotIn(forbidden, serialized)
            memory.search.assert_called_once_with("Where do I live?", filters={"user_id": "longmemeval_case"}, top_k=8, threshold=0.1)
            messages = client.chat.completions.create.call_args.kwargs["messages"]
            self.assertIn("2023/01/05", json.dumps(messages))
            self.assertNotIn("SECRET_GOLD", json.dumps(messages))
            predictions, _ = rebuild_outputs(folder, ["case1"])
            self.assertEqual(predictions, [result])
            self.assertEqual(diag["write_calls"], 2)
            memory.vector_store.client.close.assert_called_once()

    def test_failure_and_incomplete_output_do_not_create_completed_checkpoints(self):
        for response in (RuntimeError("Provider unavailable"), SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="cut off"), finish_reason="length")], usage=None)):
            client = MagicMock()
            if isinstance(response, Exception):
                client.chat.completions.create.side_effect = response
            else:
                client.chat.completions.create.return_value = response
            with test_directory() as folder:
                (Path(folder) / "completed").mkdir()
                with self.assertRaises(RuntimeError):
                    run_case(example(), {"mode": "no-memory", "model": "deepseek-chat", "max_answer_tokens": 2000}, Path(folder), client)
                self.assertFalse(list((Path(folder) / "completed").iterdir()))

    def test_invalid_extraction_is_a_technical_failure_not_an_empty_memory(self):
        memory = MagicMock()
        settings = {"model": "deepseek-chat", "embedding_model": "nomic-embed-text", "ollama_url": "http://127.0.0.1:11434"}
        with test_directory() as folder, patch("mem0.Memory.from_config", return_value=memory):
            provider_mock = memory.llm.generate_response
            provider_mock.return_value = '{"memory": []}'
            monitored = create_memory(folder, settings)
            self.assertEqual(monitored.llm.generate_response(messages=[]), '{"memory": []}')
            provider_mock.return_value = '{"wrong_field": []}'
            with self.assertRaisesRegex(RuntimeError, "Invalid SDK extraction schema"):
                monitored.llm.generate_response(messages=[])

    def test_question_storage_cannot_use_real_memory_paths(self):
        with test_directory() as folder:
            settings = {"model": "deepseek-chat", "embedding_model": "nomic-embed-text", "ollama_url": "http://127.0.0.1:11434"}
            config = memory_config(folder, settings)
            self.assertEqual(Path(config["vector_store"]["config"]["path"]).parent, Path(folder))
            self.assertEqual(Path(config["history_db_path"]).parent, Path(folder))
            self.assertNotIn("custom_instructions", config)

    def test_partial_score_counts_missing_cases_and_abstention_separately(self):
        selected = [{"question_id": "a", "question_type": "multi-session"},
                    {"question_id": "b_abs", "question_type": "multi-session"},
                    {"question_id": "c", "question_type": "knowledge-update"}]
        report = summarize(selected, {"a": True, "b_abs": False}, "gpt-4o")
        self.assertEqual(report["overall_accuracy_on_scored"], 0.5)
        self.assertEqual(report["correct_over_selected"], 1 / 3)
        self.assertEqual(report["coverage"], 2 / 3)
        self.assertEqual(report["abstention_accuracy"], 0)
        self.assertEqual(report["missing_ids"], ["c"])
        self.assertTrue(report["official_paper_judge"])
        self.assertFalse(summarize(selected, {"a": True}, "deepseek")["official_paper_judge"])

    def test_official_prompt_supports_temporal_tolerance_and_abstention(self):
        evaluator = official_evaluator()
        self.assertIn("off-by-one", evaluator.get_anscheck_prompt("temporal-reasoning", "q", "a", "r"))
        self.assertIn("unanswerable", evaluator.get_anscheck_prompt("multi-session", "q", "a", "r", abstention=True))

    def test_selection_is_repeatable_and_budget_rejects_before_api_calls(self):
        index = {"questions": [{"question_id": str(i), "question_type": str(i % 3), "abstention": False,
                               "sessions": 2, "write_calls": 100} for i in range(20)]}
        self.assertEqual(select_rows(index, 7, 42), select_rows(index, 7, 42))
        self.assertEqual(len({x["question_id"] for x in select_rows(index, 7, 42)}), 7)
        with patch("benchmarking.cli.load_index", return_value=index), patch("benchmarking.runner.run") as run, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["run", "--limit", "2", "--max-write-calls", "1"]), 1)
            run.assert_not_called()

    def test_resume_reuses_checkpoints_and_rejects_changed_configuration(self):
        index = {"variant": "s", "revision": "fixture_revision", "sha256": "fixture_sha"}
        selected = [{"question_id": "case1", "question_type": "knowledge-update", "write_calls": 2}]
        settings = {"mode": "no-memory", "model": "deepseek-chat", "max_answer_tokens": 2000}
        client = MagicMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Boston"), finish_reason="stop")], usage=None)
        with test_directory() as folder, patch.dict("os.environ", {"DEEPSEEK_API_KEY": "offline-test-value"}), \
                patch("openai.OpenAI", return_value=client), patch("benchmarking.runner.iter_questions", return_value=iter([example()])), \
                contextlib.redirect_stdout(io.StringIO()):
            output = Path(folder) / "run"
            first = run(index, selected, settings, output)
            self.assertEqual(first["completed"], 1)
            second = run(index, selected, settings, output, resume=True)
            self.assertEqual(second["completed"], 1)
            self.assertEqual(client.chat.completions.create.call_count, 1)
            with self.assertRaisesRegex(ValueError, "Resume config/data/code differ"):
                run(index, selected, {**settings, "model": "changed"}, output, resume=True)

    def test_streaming_load_reports_missing_requested_ids(self):
        with test_directory() as folder:
            path = Path(folder) / "fixture.json"
            path.write_text(json.dumps([example()]), encoding="utf-8")
            self.assertEqual(len(list(iter_questions(path, ["case1"]))), 1)
            with self.assertRaisesRegex(ValueError, "Missing question IDs"):
                list(iter_questions(path, ["unknown"]))


if __name__ == "__main__":
    unittest.main()
