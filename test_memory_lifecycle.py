import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from mem0 import Memory
from memory_lifecycle import process_turn


class MemoryLifecycleTests(unittest.TestCase):
    def sdk_memory(self):
        """Exercise the real SDK add pipeline with offline provider doubles."""
        memory = Memory.__new__(Memory)
        memory.config = SimpleNamespace(llm=SimpleNamespace(config={}))
        memory.api_version = "v1.1"
        memory.custom_instructions = None
        memory.db = MagicMock()
        memory.db.get_last_messages.return_value = []
        memory.vector_store = MagicMock()
        memory.vector_store.search.return_value = []
        memory.embedding_model = MagicMock()
        memory.embedding_model.embed.return_value = [0.1, 0.2]
        memory.embedding_model.embed_batch.return_value = [[0.1, 0.2]]
        memory.llm = MagicMock()
        memory.llm.generate_response.return_value = '{"memory": []}'
        return memory

    def run_turn(self, memory, user_text="我喜欢猫", assistant_text="可以看看这两本养猫指南"):
        with patch("mem0.memory.main.display_first_run_notice"), patch(
            "mem0.memory.main.detect_scale_threshold_from_add_result", return_value=None
        ), patch(
            "mem0.memory.main.extract_entities_batch", return_value=[[]]
        ), patch(
            "mem0.memory.main.capture_event"
        ):
            return process_turn(
                memory,
                user_id="student",
                user_text=user_text,
                assistant_text=assistant_text,
                session_id="session-1",
            )

    def test_sdk_inference_receives_both_roles_and_preserves_long_messages(self):
        memory = self.sdk_memory()
        user_text = "这是需要保留上下文的经历。" * 50
        assistant_text = "推荐书单：《养猫指南》和《猫咪行为学》。"

        result = self.run_turn(memory, user_text, assistant_text)

        self.assertEqual(result, {"results": []})
        memory.llm.generate_response.assert_called_once()
        extraction_prompt = memory.llm.generate_response.call_args.kwargs["messages"]
        self.assertIn(user_text, extraction_prompt[1]["content"])
        self.assertIn(assistant_text, extraction_prompt[1]["content"])
        self.assertIn("BOTH user and assistant", extraction_prompt[0]["content"])
        self.assertNotIn("只保留姓名", extraction_prompt[1]["content"])
        memory.db.save_messages.assert_called_once()
        self.assertEqual(memory.db.save_messages.call_args.args[0], [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ])
        self.assertEqual(memory.vector_store.search.call_args.kwargs["filters"], {"user_id": "student"})

    def test_sdk_extracted_memory_is_persisted_with_user_and_session_metadata(self):
        memory = self.sdk_memory()
        memory.llm.generate_response.return_value = '{"memory": [{"text": "用户喜欢猫"}]}'

        result = self.run_turn(memory)

        self.assertEqual(result["results"][0]["memory"], "用户喜欢猫")
        self.assertEqual(result["results"][0]["event"], "ADD")
        memory.vector_store.insert.assert_called_once()
        payload = memory.vector_store.insert.call_args.kwargs["payloads"][0]
        self.assertEqual(payload["user_id"], "student")
        self.assertEqual(payload["session"], "session-1")
        self.assertEqual(payload["source"], "chat")
        self.assertEqual(payload["data"], "用户喜欢猫")
        memory.db.batch_add_history.assert_called_once()

    def test_forget_request_is_handled_by_sdk_without_app_deletion(self):
        memory = self.sdk_memory()
        request = "请忘记我之前说的住址"

        self.run_turn(memory, request, "收到你的请求")

        memory.llm.generate_response.assert_called_once()
        self.assertIn(request, memory.llm.generate_response.call_args.kwargs["messages"][1]["content"])
        memory.vector_store.delete.assert_not_called()

    def test_sdk_failure_reaches_the_app_error_handler(self):
        memory = self.sdk_memory()
        memory.llm.generate_response.side_effect = RuntimeError("offline provider failure")

        with self.assertRaisesRegex(Exception, "offline provider failure"):
            self.run_turn(memory)

        memory.vector_store.insert.assert_not_called()
        memory.db.save_messages.assert_not_called()


if __name__ == "__main__":
    unittest.main()
