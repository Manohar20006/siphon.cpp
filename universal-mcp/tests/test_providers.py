import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from universal_mcp.models import TaskRequest, WorkerConfig
from universal_mcp.providers import (
    OpenAICompatibleWorker,
    _task_prompt,
    normalize_worker_output,
)


class ProviderTests(unittest.TestCase):
    def test_normalizes_worker_json(self):
        result = normalize_worker_output(
            "task-1",
            "kaggle-qwen27b",
            json.dumps(
                {
                    "status": "completed",
                    "summary": "implemented the change",
                    "patch": "diff --git a/a.txt b/a.txt",
                    "tests_run": ["pytest -q"],
                    "verification_report": "passed",
                    "blockers": [],
                    "usage": {"input_tokens": 100, "output_tokens": 25, "total_tokens": 125},
                    "duration_seconds": 1.5,
                }
            ),
        )
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.tests_run, ["pytest -q"])
        self.assertIn("diff --git", result.patch)
        self.assertEqual(result.usage["total_tokens"], 125)
        self.assertEqual(result.duration_seconds, 1.5)

    def test_plain_text_requires_review(self):
        result = normalize_worker_output("task-1", "worker", "I changed the file.")
        self.assertEqual(result.status, "needs_review")
        self.assertEqual(result.error, "invalid_worker_result")

    def test_openai_compatible_urls(self):
        worker = OpenAICompatibleWorker()
        config = WorkerConfig(
            worker_id="qwen",
            source="kaggle",
            model="qwen-27b",
            base_url="https://example.test/v1",
        )
        self.assertEqual(
            worker._url(config, "chat/completions"),
            "https://example.test/v1/chat/completions",
        )
        self.assertEqual(
            worker._url(config, "models"),
            "https://example.test/v1/models",
        )

    def test_task_prompt_contains_verification_contract(self):
        task = TaskRequest(
            task_id="task-1",
            title="Example",
            objective="Change one file",
            implementation_plan="Edit the file and preserve its API.",
            verification_instructions="Run the targeted unit test.",
        )
        prompt = _task_prompt(task)
        self.assertIn("IMPLEMENTATION PLAN", prompt)
        self.assertIn("VERIFICATION INSTRUCTIONS", prompt)
        self.assertIn("Run the targeted unit test.", prompt)
        self.assertIn("Worker tool access: False", prompt)

    def test_openai_compatible_endpoint_end_to_end(self):
        received = {}

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                return

            def do_GET(self):
                received["health_path"] = self.path
                received["health_auth"] = self.headers.get("Authorization")
                body = json.dumps({"data": [{"id": "qwen-27b"}]}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                length = int(self.headers["Content-Length"])
                received["chat_path"] = self.path
                received["chat_auth"] = self.headers.get("Authorization")
                received["payload"] = json.loads(self.rfile.read(length))
                worker_result = {
                    "status": "completed",
                    "summary": "mock worker received the plan",
                    "patch": "",
                    "tests_run": [],
                    "verification_report": "awaiting real command evidence",
                    "blockers": [],
                }
                body = json.dumps(
                    {
                        "choices": [
                            {"message": {"content": json.dumps(worker_result)}}
                        ]
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        config = WorkerConfig(
            worker_id="qwen",
            source="kaggle",
            model="qwen-27b",
            base_url=f"http://127.0.0.1:{server.server_port}",
            api_key_env="TEST_QWEN_KEY",
        )
        task = TaskRequest(
            task_id="task-http",
            title="Transport test",
            objective="Receive this package",
            implementation_plan="Return a structured result.",
            verification_instructions="Wait for real evidence.",
            verification_commands=[["python3", "-c", "print('ok')"]],
        )
        worker = OpenAICompatibleWorker()
        with patch.dict(os.environ, {"TEST_QWEN_KEY": "secret-token"}):
            self.assertTrue(worker.health(config).healthy)
            result = worker.execute(config, task)

        self.assertEqual(result.status, "completed")
        self.assertEqual(received["health_path"], "/v1/models")
        self.assertEqual(received["chat_path"], "/v1/chat/completions")
        self.assertEqual(received["health_auth"], "Bearer secret-token")
        self.assertEqual(received["chat_auth"], "Bearer secret-token")
        self.assertEqual(received["payload"]["model"], "qwen-27b")
        prompt = received["payload"]["messages"][1]["content"]
        self.assertIn("ASTRA-SPECIFIED VERIFICATION COMMANDS", prompt)


if __name__ == "__main__":
    unittest.main()
