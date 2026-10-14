"""Local API contract tests: scoped keys, decision prompt/prefix logic, gateway auth and proxying (no GPU)."""

import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "siphon-studio"))

from siphon.api import connect, decide, gateway  # noqa: E402
from siphon.api.keys import KEY_PREFIX, KeyStore  # noqa: E402
from siphon.core import runner  # noqa: E402
from siphon.core.profiles import Profile  # noqa: E402


class FakeLlamaServer(BaseHTTPRequestHandler):
    """Character tokenizer + toy chat template; /decide scores options by prompt keywords."""
    protocol_version = "HTTP/1.0"
    decide_calls: list = []

    def log_message(self, *args):
        pass

    def _send(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        if self.path == "/tokenize":
            toks = [ord(c) for c in body["content"]]
            self._send({"tokens": ([1] if body.get("add_special") else []) + toks})
        elif self.path == "/apply-template":
            sys_msg, user = body["messages"][0]["content"], body["messages"][1]["content"]
            self._send({"prompt": f"<s>{sys_msg}</s><u>{user}</u><a>"})
        elif self.path == "/decide":
            type(self).decide_calls.append(body)
            prompt = body["prompt"]
            if "TOO_LONG" in prompt:
                self._send({"error": {"code": 400, "message": "the decision prompt has 900 tokens, more than "
                                      "--decision-ctx 512", "type": "exceed_context_size_error"}}, 400)
                return
            logprobs = [-0.1 if "score" in prompt and i == 0 else -3.0 for i in range(len(body["token_ids"]))]
            self._send({"logprobs": logprobs, "decide_ms": 12.5, "prompt_tokens": len(prompt) + 1,
                        "prefix_tokens": body["prefix_tokens"], "prefix_hit": True})
        elif self.path == "/v1/chat/completions":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for piece in ("Hel", "lo"):
                self.wfile.write(f'data: {{"choices":[{{"delta":{{"content":"{piece}"}}}}]}}\n\n'.encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
        elif self.path in {"/v1/messages", "/v1/messages/count_tokens", "/v1/responses"}:
            self._send({"route": self.path, "model": body.get("model")})
        else:
            self._send({"error": "not found"}, 404)


def serve(handler):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class KeyStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        patch = mock.patch.dict(os.environ, {"SIPHON_STUDIO_DIR": self.temp.name})
        patch.start()
        self.addCleanup(patch.stop)

    def test_one_permanent_private_key_per_scope(self):
        store = KeyStore()
        decode_key, decision_key = store.get("decode"), store.get("decision")
        self.assertTrue(decode_key.key.startswith(KEY_PREFIX["decode"]))
        self.assertTrue(decision_key.key.startswith(KEY_PREFIX["decision"]))
        self.assertEqual(store.get("decode").key, decode_key.key)          # same key every time
        self.assertEqual(KeyStore().get("decode").key, decode_key.key)     # and across processes
        self.assertEqual(stat.S_IMODE(store.path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(store.path.parent.stat().st_mode), 0o700)
        self.assertEqual((store.path.parent / ".gitignore").read_text(), "*\n")
        self.assertNotIn("key", decode_key.public())
        self.assertEqual(store.verify(decision_key.key).scope, "decision")
        self.assertIsNone(store.verify(decision_key.key + "x"))
        self.assertIsNone(store.verify(""))
        with self.assertRaises(ValueError):
            store.get("admin")

    def test_regenerate_replaces_the_key(self):
        first = KeyStore()
        old = first.get("decision").key
        second = KeyStore()
        self.assertEqual(second.verify(old).scope, "decision")
        new = first.regenerate("decision").key
        self.assertNotEqual(new, old)
        self.assertIsNone(second.verify(old))
        self.assertEqual(second.verify(new).scope, "decision")
        self.assertEqual(second.get("decision").key, new)


class DecisionEngineTests(unittest.TestCase):
    def setUp(self):
        FakeLlamaServer.decide_calls = []
        self.backend = serve(FakeLlamaServer)
        self.addCleanup(self.backend.server_close)
        self.addCleanup(self.backend.shutdown)
        self.url = f"http://127.0.0.1:{self.backend.server_address[1]}"

    def body(self, text="What is the score?"):
        return {"input": text, "question": "Which tool?", "context": "Today is Monday.",
                "options": [{"name": "sports_live", "description": "live scores"}, "web_search", "answer_directly"]}

    def test_request_validation(self):
        bad = [{}, {"input": "x", "question": "q", "options": ["a"]},
               {"input": "x", "question": "q", "options": ["a", "a"]},
               {"input": "x", "question": "q", "options": ["a", {"description": "no name"}]},
               {"input": "x", "question": "q", "options": ["a", "b\nc"]},
               {"input": "x", "question": "q", "options": ["a", "b"], "threshold": 2},
               {"input": "x", "question": "q", "options": ["a"] * 27}]
        for body in bad:
            with self.assertRaises(decide.DecisionError) as ctx:
                decide.parse_request(body)
            self.assertEqual(ctx.exception.status, 400)

    def test_prompt_prefix_and_probabilities(self):
        engine = decide.DecisionEngine(self.url, alias="m")
        result = engine.decide(self.body())
        call = FakeLlamaServer.decide_calls[-1]
        self.assertEqual(call["token_ids"], [ord("A"), ord("B"), ord("C")])
        self.assertIn("Options:\nA. sports_live: live scores\nB. web_search\nC. answer_directly\n\n"
                      "State:\nToday is Monday.\nWhat is the score?\n\nQuestion: Which tool?\n"
                      "Answer with the letter only.", call["prompt"])
        # shared prefix = everything before the input (plus BOS), minus one token, aligned down to the 24-token chunk
        head = call["prompt"].split("What is the score?")[0]
        self.assertEqual(call["prefix_tokens"], (len(head) + 1 - 1) // 24 * 24)
        self.assertEqual(call["prefix_tokens"] % decide.PROMPT_CHUNK, 0)
        z = math.exp(-0.1) + 2 * math.exp(-3.0)
        self.assertAlmostEqual(result["probabilities"]["sports_live"], math.exp(-0.1) / z)
        self.assertEqual(result["choice"], "sports_live")
        self.assertTrue(result["confident"])
        # same options -> same prefix, layout computed once
        engine.decide(self.body("Another request about the weather"))
        self.assertEqual(FakeLlamaServer.decide_calls[-1]["prefix_tokens"], call["prefix_tokens"])
        self.assertEqual(len(engine._layouts), 1)

    def test_too_long_input_is_the_callers_400(self):
        engine = decide.DecisionEngine(self.url)
        with self.assertRaises(decide.DecisionError) as ctx:
            engine.decide(self.body("TOO_LONG"))
        self.assertEqual(ctx.exception.status, 400)
        self.assertIn("more than --decision-ctx 512", str(ctx.exception))

    def test_unreachable_backend_is_503(self):
        engine = decide.DecisionEngine("http://127.0.0.1:9")
        with self.assertRaises(decide.DecisionError) as ctx:
            engine.decide(self.body())
        self.assertEqual(ctx.exception.status, 503)


class StaticResolver:
    def __init__(self, backend):
        self.backend = backend
        self.real = gateway.BackendResolver()

    def current(self):
        return self.backend

    def engine(self, backend):
        return self.real.engine(backend)


class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = KeyStore(Path(self.temp.name) / "keys.json")
        self.decode_key = self.store.get("decode").key
        self.decision_key = self.store.get("decision").key
        self.fake = serve(FakeLlamaServer)
        self.addCleanup(self.fake.server_close)
        self.addCleanup(self.fake.shutdown)
        self.backend = gateway.Backend("p", f"http://127.0.0.1:{self.fake.server_address[1]}", "m", True, True)
        self.gw = gateway.Gateway(0, keys=self.store, resolver=StaticResolver(self.backend))
        threading.Thread(target=self.gw.serve_forever, daemon=True).start()
        self.addCleanup(self.gw.server_close)
        self.addCleanup(self.gw.shutdown)
        self.url = f"http://127.0.0.1:{self.gw.server_address[1]}"

    def call(self, path, body=None, key=None):
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(self.url + path, None if body is None else json.dumps(body).encode(), headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_binds_loopback_only(self):
        self.assertEqual(self.gw.server_address[0], "127.0.0.1")

    def test_scopes(self):
        decide_body = {"input": "score?", "question": "q", "options": ["a", "b"]}
        chat = {"messages": [{"role": "user", "content": "hi"}], "stream": True}
        self.assertEqual(self.call("/health")[0], 200)
        self.assertEqual(self.call("/v1/decide", decide_body)[0], 401)
        self.assertEqual(self.call("/v1/decide", decide_body, "sk-siphon-dec-nope")[0], 401)
        self.assertEqual(self.call("/v1/decide", decide_body, self.decode_key)[0], 403)
        self.assertEqual(self.call("/v1/chat/completions", chat, self.decision_key)[0], 403)
        self.assertEqual(self.call("/v1/decide", {"input": "x"}, self.decision_key)[0], 400)
        status, data = self.call("/v1/decide", decide_body, self.decision_key)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["choice"], "a")
        status, data = self.call("/v1/chat/completions", chat, self.decode_key)
        self.assertEqual(status, 200)
        self.assertIn(b'"content":"Hel"', data)
        self.assertTrue(data.rstrip().endswith(b"data: [DONE]"))
        self.assertEqual(self.call("/v1/models", key=self.decision_key)[0], 200)

    def test_anthropic_and_responses_routes_use_the_decode_key(self):
        for path in ("/v1/messages", "/v1/messages/count_tokens", "/v1/responses"):
            self.assertEqual(self.call(path, {"model": "m"}, self.decision_key)[0], 403)
            status, data = self.call(path + "?beta=true", {"model": "m"}, self.decode_key)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(data)["route"], path)

    def test_activity_counts_requests_that_passed_the_key_check(self):
        def settled(total):     # the gateway records a request after its response is sent
            deadline = time.time() + 5
            while True:
                snap = self.gw.activity.snapshot()
                if sum(p["requests"] for p in snap.values()) >= total or time.time() > deadline:
                    return snap
                time.sleep(0.01)
        decide_body = {"input": "score?", "question": "q", "options": ["a", "b"]}
        self.assertEqual(self.call("/v1/decide", decide_body)[0], 401)
        self.assertEqual(self.call("/v1/decide", decide_body, self.decode_key)[0], 403)
        self.assertEqual(self.call("/v1/nope", {}, self.decode_key)[0], 404)
        self.assertEqual(self.call("/v1/decide", decide_body, self.decision_key)[0], 200)
        self.assertEqual(self.call("/v1/decide", {"input": "x"}, self.decision_key)[0], 400)
        self.assertEqual(self.call("/v1/responses", {"model": "m"}, self.decode_key)[0], 200)
        snap = settled(3)
        decode, decision = snap["decode"], snap["decision"]
        self.assertEqual((decision["requests"], decision["errors"]), (2, 1))
        self.assertEqual((decision["last_status"], decision["last_route"]), (400, "/v1/decide"))
        self.assertEqual((decode["requests"], decode["errors"]), (1, 0))
        self.assertEqual((decode["last_status"], decode["last_route"]), (200, "/v1/responses"))
        self.assertGreater(decode["last_ms"], 0)
        self.assertEqual(sum(decode["per_minute"]) + sum(decision["per_minute"]), 3)

    def test_engine_not_running_is_503(self):
        self.backend.decision = False
        body = {"input": "score?", "question": "q", "options": ["a", "b"]}
        status, data = self.call("/v1/decide", body, self.decision_key)
        self.assertEqual(status, 503)
        self.assertIn(b"decode mode", data)

    def test_backend_mode_from_profile(self):
        def mode(env):
            return gateway.backend_for(Profile(name="x", env=env)).mode
        self.assertEqual(mode({}), "decode")
        self.assertEqual(mode({"LLAMA_ARG_DECISION_CTX": "512"}), "decode+decision")
        self.assertEqual(mode({"LLAMA_ARG_DECISION_ONLY": "1", "LLAMA_ARG_DECISION_CTX": "512"}), "decision")


class ActivityTests(unittest.TestCase):
    def test_counters_and_the_30_minute_window(self):
        activity, t = gateway.Activity(), 1_800_000_000.0        # t starts a minute
        activity.record("decision", "/v1/decide", 200, 700.0, now=t - 31 * 60)
        activity.record("decode", "/v1/chat/completions", 200, 120.0, now=t - 60)
        activity.record("decode", "/v1/chat/completions", 200, 90.0, now=t)
        activity.record("decode", "/v1/messages", 503, 4.0, now=t + 1)
        activity.record("admin", "/v1/x", 200, 1.0, now=t)         # unknown scopes are ignored
        snap = activity.snapshot(now=t + 2)
        self.assertEqual(set(snap), {"decode", "decision"})
        decode, decision = snap["decode"], snap["decision"]
        self.assertEqual((decode["requests"], decode["errors"]), (3, 1))
        self.assertEqual((decode["last_at"], decode["last_ms"], decode["last_status"], decode["last_route"]),
                         (t + 1, 4.0, 503, "/v1/messages"))
        self.assertEqual(len(decode["per_minute"]), gateway.Activity.MINUTES)
        self.assertEqual(decode["per_minute"][-2:], [1, 2])          # oldest first, this minute last
        self.assertNotIn("minutes", decode)
        self.assertEqual(decision["requests"], 1)                     # the session total keeps it...
        self.assertEqual(sum(decision["per_minute"]), 0)              # ...the 30-minute chart does not


class ConnectAppsTests(unittest.TestCase):
    """Connectors write only into a temp HOME; fake opencode/claude/codex commands make them 'installed'."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.home, fake_bin = root / "home", root / "bin"
        fake_bin.mkdir()
        for name in ("opencode", "claude", "codex"):
            (fake_bin / name).write_text("#!/bin/sh\n")
            (fake_bin / name).chmod(0o755)
        env = {"HOME": str(self.home), "XDG_CONFIG_HOME": str(self.home / ".config"),
               "SIPHON_STUDIO_DIR": str(root / "studio"), "PATH": f"{fake_bin}:/usr/bin:/bin"}
        patch = mock.patch.dict(os.environ, env)
        patch.start()
        self.addCleanup(patch.stop)
        self.target = connect.Target("http://127.0.0.1:8088", "sk-siphon-gen-test", "qwen-test", 32768)
        self.oc = self.home / ".config/opencode/opencode.json"

    def test_opencode_adds_and_removes_only_its_provider(self):
        self.oc.parent.mkdir(parents=True)
        user = {"$schema": "https://opencode.ai/config.json", "model": "other/x",
                "provider": {"other": {"npm": "@ai-sdk/openai-compatible", "options": {"baseURL": "http://h/v1"}}}}
        self.oc.write_text(json.dumps(user))
        connect.connect("opencode", self.target)
        data = json.loads(self.oc.read_text())
        self.assertEqual(data["model"], "other/x")                       # user's choice is left alone
        self.assertIn("other", data["provider"])
        siphon = data["provider"]["siphon"]
        self.assertEqual(siphon["options"], {"baseURL": "http://127.0.0.1:8088/v1", "apiKey": "sk-siphon-gen-test"})
        self.assertEqual(siphon["models"]["qwen-test"]["limit"]["context"], 32768)
        self.assertEqual(stat.S_IMODE(self.oc.stat().st_mode), 0o600)
        backups = list(self.oc.parent.glob("opencode.json.bak-siphon-*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(json.loads(backups[0].read_text()), user)
        connect.connect("opencode", self.target)                          # reconnect: no second backup
        self.assertEqual(len(list(self.oc.parent.glob("opencode.json.bak-siphon-*"))), 1)
        connect.disconnect("opencode")
        self.assertEqual(json.loads(self.oc.read_text()), user)

    def test_opencode_file_created_by_siphon_is_removed_on_disconnect(self):
        jsonc = self.home / ".config/opencode/opencode.jsonc"
        jsonc.parent.mkdir(parents=True)
        jsonc.write_text('{\n  // the main config, never edited\n  "model": "x"\n}\n')
        before = jsonc.read_text()
        connect.connect("opencode", self.target)
        self.assertTrue(self.oc.exists())
        self.assertEqual(jsonc.read_text(), before)
        connect.disconnect("opencode")
        self.assertFalse(self.oc.exists())
        self.assertEqual(jsonc.read_text(), before)

    def test_opencode_refuses_to_rewrite_jsonc_content(self):
        self.oc.parent.mkdir(parents=True)
        self.oc.write_text('{ // comment\n "model": "x" }')
        with self.assertRaises(connect.ConnectError):
            connect.connect("opencode", self.target)
        self.assertEqual(self.oc.read_text(), '{ // comment\n "model": "x" }')

    def test_wrappers_and_refresh_after_key_change(self):
        connect.connect("claude-code", self.target)
        connect.connect("codex", self.target)
        claude = self.home / ".local/bin/claude-siphon"
        self.assertEqual(stat.S_IMODE(claude.stat().st_mode), 0o700)
        text = claude.read_text()
        self.assertIn("export ANTHROPIC_BASE_URL=http://127.0.0.1:8088\n", text)
        self.assertIn("export ANTHROPIC_AUTH_TOKEN=sk-siphon-gen-test\n", text)
        self.assertIn("unset ANTHROPIC_API_KEY", text)
        self.assertTrue(text.rstrip().endswith('exec claude "$@"'))
        codex = (self.home / ".local/bin/codex-siphon").read_text()
        self.assertIn('base_url="http://127.0.0.1:8088/v1"', codex)
        self.assertEqual({a["id"]: a["connected"] for a in connect.status()},
                         {"opencode": False, "claude-code": True, "codex": True})
        self.assertEqual(connect.refresh("sk-siphon-gen-new"), ["claude-code", "codex"])
        self.assertIn("ANTHROPIC_AUTH_TOKEN=sk-siphon-gen-new", claude.read_text())
        self.assertNotIn("sk-siphon-gen-test", claude.read_text())
        connect.disconnect("claude-code")
        self.assertFalse(claude.exists())

    def test_never_overwrites_a_foreign_command(self):
        bin_dir = self.home / ".local/bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "claude-siphon").write_text("#!/bin/sh\necho mine\n")
        with self.assertRaises(connect.ConnectError):
            connect.connect("claude-code", self.target)
        connect.disconnect("claude-code")
        self.assertEqual((bin_dir / "claude-siphon").read_text(), "#!/bin/sh\necho mine\n")

    def test_not_installed(self):
        os.environ["PATH"] = "/nonexistent"
        self.assertFalse(any(a["installed"] for a in connect.status()))
        with self.assertRaises(connect.ConnectError):
            connect.connect("opencode", self.target)
        self.assertFalse(self.oc.exists())


class StudioIntegrationTests(unittest.TestCase):
    def test_process_profile_reads_owner(self):
        with mock.patch.dict(os.environ, {"SIPHON_PROFILE": "owner-x"}):
            import subprocess
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
            try:
                import time
                time.sleep(0.2)
                self.assertEqual(runner._process_profile(child.pid), "owner-x")
            finally:
                child.kill()
                child.wait()

    def test_decision_only_vram_estimate_ignores_decode_ctx(self):
        env = {"CTX": "65536", "MOE_GPU_CACHE_MIB": "3456", "SIPHON_EXECUTION_MODE": "gpu"}
        with tempfile.TemporaryDirectory() as root, mock.patch.dict(os.environ, {"SIPHON_REPO_ROOT": root}):
            def vram_issues(extra):
                return [i for i in Profile(name="x", env={**env, **extra}).validate() if "VRAM" in i]
            self.assertTrue(vram_issues({}))
            self.assertFalse(vram_issues({"LLAMA_ARG_DECISION_ONLY": "1", "LLAMA_ARG_DECISION_CTX": "512"}))


if __name__ == "__main__":
    unittest.main()
