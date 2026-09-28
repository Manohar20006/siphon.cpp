import unittest
from pathlib import Path

from universal_mcp.config import AppConfig
from universal_mcp.orchestrator import Orchestrator
from universal_mcp.server import _handle_request


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.orchestrator = Orchestrator(
            AppConfig(project_root=Path("."), workers=())
        )

    def test_initialize(self):
        response = _handle_request(
            self.orchestrator,
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2024-11-05"},
            },
        )
        self.assertEqual(response["result"]["protocolVersion"], "2024-11-05")
        self.assertIn("tools", response["result"]["capabilities"])

    def test_tools_list(self):
        response = _handle_request(
            self.orchestrator,
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        )
        names = {tool["name"] for tool in response["result"]["tools"]}
        self.assertIn("dispatch_worker_task", names)
        self.assertIn("delegate_workspace_operation", names)
        self.assertIn("continue_worker_task", names)
        self.assertIn("approve_task", names)
        self.assertIn("finalize_project", names)
        self.assertIn("apply_worker_patch", names)
        self.assertIn("run_worker_verification", names)
        self.assertIn("get_task_events", names)
        self.assertIn("send_worker_message", names)
        self.assertIn("pause_worker_task", names)
        self.assertIn("resume_worker_task", names)
        self.assertIn("cancel_worker_task", names)
        self.assertIn("get_worker_job", names)
        self.assertIn("answer_worker_question", names)
        self.assertIn("dispatch_worker_review", names)
        self.assertIn("resolve_worker_review", names)

        dispatch = next(
            tool for tool in response["result"]["tools"]
            if tool["name"] == "dispatch_worker_task"
        )
        self.assertIn(
            "verification_commands", dispatch["inputSchema"]["required"]
        )
        self.assertIn("conversation_id", dispatch["inputSchema"]["properties"])
        self.assertIn("async", dispatch["inputSchema"]["properties"])


if __name__ == "__main__":
    unittest.main()
