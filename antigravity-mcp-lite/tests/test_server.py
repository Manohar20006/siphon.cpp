import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import io
from unittest.mock import patch

MODULE_PATH = Path(__file__).parents[1] / "server.py"
SPEC = importlib.util.spec_from_file_location("antigravity_mcp_lite_server", MODULE_PATH)
module = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(module)
Config, Server = module.Config, module.Server


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.server = Server(Config({"project_root": str(self.root), "models": {"gemini": "gemini-3.8-flash", "claude": "claude-sonnet-4-6"}}))

    def tearDown(self):
        self.server.close()
        self.temp.cleanup()

    def test_tools_and_defaults(self):
        self.assertEqual(len(self.server.tools()), 9)
        self.assertIn("create_antigravity_session", {x["name"] for x in self.server.tools()})
        self.assertEqual(self.server.config.max_prompt_bytes, 120000)

    @patch("shutil.which", return_value="/bin/agy")
    @patch("subprocess.run")
    def test_exact_gemini_argv_and_envelope(self, run, _which):
        run.return_value = subprocess.CompletedProcess([], 0, json.dumps({"response": "ok", "conversation_id": "c1", "usage": {"total_tokens": 3}}), "")
        result = self.server.ask({"model": "gemini", "prompt": "hello"})
        argv = run.call_args.args[0]
        self.assertEqual(argv[:-1], ["/bin/agy", "--model", "gemini-3.8-flash", "--effort", "low", "--mode", "plan", "--disable-slash-commands", "--output-format", "json"])
        self.assertEqual(argv[-1], "--print=hello")
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(result["structuredContent"]["response"], "ok")
        self.assertEqual(result["structuredContent"]["conversation_id"], "c1")

    @patch("shutil.which", return_value="/bin/agy")
    @patch("subprocess.run")
    def test_conversation_reuse_and_edit_mode(self, run, _which):
        run.return_value = subprocess.CompletedProcess([], 0, "done", "")
        self.server.ask({
            "model": "gemini", "prompt": "continue", "conversation_id": "c1",
            "mode": "accept-edits", "effort": "medium",
        })
        argv = run.call_args.args[0]
        self.assertIn("--conversation", argv)
        self.assertEqual(argv[argv.index("--conversation") + 1], "c1")
        self.assertEqual(argv[argv.index("--mode") + 1], "accept-edits")
        self.assertEqual(argv[argv.index("--effort") + 1], "medium")

    @patch("shutil.which", return_value="/bin/agy")
    @patch("subprocess.run")
    def test_claude_has_no_effort_flag(self, run, _which):
        run.return_value = subprocess.CompletedProcess([], 0, "answer", "")
        self.server.ask({"model": "claude", "prompt": "hello"})
        self.assertNotIn("--effort", run.call_args.args[0])

    def test_rejections(self):
        self.assertTrue(self.server.ask({"model": "other", "prompt": "x"})["isError"])
        self.assertTrue(self.server.ask({"model": "gemini", "prompt": "x", "working_directory": "../escape"})["isError"])
        self.server.config.max_prompt_bytes = 2
        self.assertEqual(self.server.ask({"model": "gemini", "prompt": "long"})["structuredContent"]["error"], "prompt_too_large")
        self.assertTrue(self.server.ask({"model": "gemini", "prompt": "x", "timeout": 0})["isError"])

    def test_protocol_and_local_only_models(self):
        response = self.server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        self.assertEqual(len(response["result"]["tools"]), 9)

    def test_rejects_invalid_model_entries_and_has_packaging_entrypoint(self):
        config = Config({"models": {"local": "gemini-x"}})
        self.assertEqual(config.models["local"], "gemini-x")
        self.assertIn("antigravity-mcp-lite", (MODULE_PATH.parent / "pyproject.toml").read_text())
        self.assertIn("main", (MODULE_PATH.parent / "pyproject.toml").read_text())
        with self.assertRaises(ValueError):
            Config({"models": {"invalid": None}})

    def test_session_two_turn_stream_contract_and_cleanup(self):
        events = [
            {"event": "init", "conversation_id": "conv"},
            {"event": "result", "result": {"status": "SUCCESS", "response": "one", "conversation_id": "conv"}},
            {"event": "result", "result": {"status": "SUCCESS", "response": "two", "conversation_id": "conv"}},
        ]

        class Input:
            def __init__(self): self.writes = []
            def write(self, value): self.writes.append(value)
            def flush(self): pass
            def close(self): pass

        class Process:
            pid = 42
            returncode = None
            def __init__(self): self.stdin = Input()
            def poll(self): return self.returncode
            def terminate(self): self.returncode = 0
            def wait(self, timeout=None): return 0

        class Session:
            def __init__(self, session_id, process, _limit):
                self.session_id, self.process = session_id, process
                self.lock = __import__("threading").Lock()
                self.conversation_id = "conv"
                self.last_used_at = module.time.time()
                self.events = iter(events)
            def wait_for(self, name, _timeout):
                event = next(self.events)
                assert event["event"] == name
                return event
            def close(self): self.process.terminate()

        from unittest.mock import patch as local_patch
        with local_patch("shutil.which", return_value="/bin/agy"), local_patch("subprocess.Popen", return_value=Process()), local_patch.object(module, "PersistentSession", Session):
            created = self.server.create_session({"model": "gemini"})["structuredContent"]
        session_id = created["session_id"]
        first = self.server.send_session_message({"session_id": session_id, "prompt": "first"})
        second = self.server.send_session_message({"session_id": session_id, "prompt": "second"})
        self.assertEqual(first["structuredContent"]["response"], "one")
        self.assertEqual(second["structuredContent"]["response"], "two")
        written = self.server.sessions[session_id].process.stdin.writes
        self.assertEqual(json.loads(written[0]), {"event": "user", "message": {"role": "user", "content": [{"type": "text", "text": "first"}]}})
        self.server.close_session({"session_id": session_id})
        self.assertNotIn(session_id, self.server.sessions)

    def test_timed_out_session_turn_is_removed_and_child_closed(self):
        class Process:
            pid = 43
            returncode = None
            def __init__(self): self.stdin = io.StringIO()
            def poll(self): return self.returncode
            def terminate(self): self.returncode = 0
            def wait(self, timeout=None): return 0

        class Session:
            def __init__(self, session_id, process, _limit):
                self.session_id, self.process = session_id, process
                self.lock = __import__("threading").Lock()
                self.conversation_id = "c"
                self.last_used_at = module.time.time()
            def wait_for(self, name, _timeout):
                if name == "init": return {"event": "init", "conversation_id": "c"}
                raise TimeoutError(name)
            def close(self): self.process.terminate()

        process = Process()
        from unittest.mock import patch as local_patch
        with local_patch("shutil.which", return_value="/bin/agy"), local_patch("subprocess.Popen", return_value=process), local_patch.object(module, "PersistentSession", Session):
            created = self.server.create_session({"model": "gemini"})["structuredContent"]
        session_id = created["session_id"]
        failed = self.server.send_session_message({"session_id": session_id, "prompt": "wait", "timeout": 1})
        self.assertTrue(failed["isError"])
        self.assertNotIn(session_id, self.server.sessions)
        self.assertEqual(process.returncode, 0)


if __name__ == "__main__":
    unittest.main()
