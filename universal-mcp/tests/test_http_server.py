import json
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from universal_mcp.config import AppConfig
from universal_mcp.http_server import create_http_server
from universal_mcp.orchestrator import Orchestrator


class HTTPServerTests(unittest.TestCase):
    def setUp(self):
        orchestrator = Orchestrator(AppConfig(project_root=Path("."), workers=()))
        self.server = create_http_server(
            "127.0.0.1",
            0,
            orchestrator,
            "test-token",
            frozenset({"https://antigravity.example"}),
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def _post(self, payload, *, token="test-token", origin=None):
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if origin:
            headers["Origin"] = origin
        request = urllib.request.Request(
            f"{self.base_url}/mcp",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        return urllib.request.urlopen(request, timeout=2)

    def test_authenticated_initialize_and_tools_list(self):
        with self._post(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"protocolVersion": "2025-11-25"},
            },
            origin="https://antigravity.example",
        ) as response:
            result = json.loads(response.read())
        self.assertEqual(result["result"]["protocolVersion"], "2025-11-25")

        with self._post(
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        ) as response:
            result = json.loads(response.read())
        names = {tool["name"] for tool in result["result"]["tools"]}
        self.assertIn("dispatch_worker_task", names)

    def test_rejects_bad_token_and_origin(self):
        with self.assertRaises(urllib.error.HTTPError) as token_error:
            self._post({"jsonrpc": "2.0", "id": 1, "method": "ping"}, token="wrong")
        self.assertEqual(token_error.exception.code, 401)

        with self.assertRaises(urllib.error.HTTPError) as origin_error:
            self._post(
                {"jsonrpc": "2.0", "id": 2, "method": "ping"},
                origin="https://evil.example",
            )
        self.assertEqual(origin_error.exception.code, 403)

    def test_notification_returns_accepted(self):
        with self._post(
            {"jsonrpc": "2.0", "method": "notifications/initialized"}
        ) as response:
            self.assertEqual(response.status, 202)
            self.assertEqual(response.read(), b"")

    def test_health_and_get_behavior(self):
        with urllib.request.urlopen(f"{self.base_url}/healthz", timeout=2) as response:
            self.assertEqual(response.status, 200)
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(f"{self.base_url}/mcp", timeout=2)
        self.assertEqual(error.exception.code, 405)


if __name__ == "__main__":
    unittest.main()
