#!/usr/bin/env python3
"""Small stdio MCP that uses the locally authenticated Antigravity CLI."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Any
import uuid

PROTOCOL_VERSION = "2024-11-05"
DEFAULT_MODELS = {"gemini": "gemini-3.8-flash", "claude": "claude-sonnet-4-6"}


class Config:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        data = data or {}
        self.agy_path = str(data.get("agy_path", "agy"))
        self.project_root = Path(data.get("project_root", Path.cwd())).resolve()
        self.timeout = int(data.get("timeout", 300))
        self.effort = str(data.get("effort", "low"))
        self.mode = str(data.get("mode", "plan"))
        self.max_prompt_bytes = int(data.get("max_prompt_bytes", 120_000))
        self.max_output_bytes = int(data.get("max_output_bytes", 256_000))
        self.session_idle_seconds = int(data.get("session_idle_seconds", 1800))
        self.models = dict(data.get("models", DEFAULT_MODELS))
        self.max_sessions = int(data.get("max_sessions", 8))
        self.max_response_bytes = int(data.get("max_response_bytes", self.max_output_bytes))
        if not 1 <= self.timeout <= 600:
            raise ValueError("timeout must be between 1 and 600")
        if self.effort not in {"low", "medium", "high"}:
            raise ValueError("effort must be low, medium, or high")
        if self.mode not in {"plan", "accept-edits"}:
            raise ValueError("mode must be plan or accept-edits")
        if not self.models or any(not isinstance(k, str) or not k for k in self.models):
            raise ValueError("models must be a non-empty alias map")
        if not 1 <= self.max_prompt_bytes <= 1_000_000:
            raise ValueError("max_prompt_bytes must be between 1 and 1000000")
        if not 1 <= self.max_output_bytes <= 4_000_000 or not 1 <= self.max_response_bytes <= 4_000_000:
            raise ValueError("output limits must be between 1 and 4000000 bytes")
        if not 1 <= self.max_sessions <= 32 or not 1 <= self.session_idle_seconds <= 86400:
            raise ValueError("session limits must be max_sessions 1..32 and idle seconds 1..86400")
        for alias, model in self.models.items():
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError("model aliases must be non-empty strings")
            if isinstance(model, str):
                if not model:
                    raise ValueError(f"model alias {alias!r} has an empty model")
            else:
                raise ValueError(f"model alias {alias!r} must map to an agy model name")

    @classmethod
    def load(cls) -> "Config":
        source = os.environ.get("ANTIGRAVITY_MCP_LITE_CONFIG")
        if not source:
            return cls()
        path = Path(source).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            return cls(json.load(handle))


class PersistentSession:
    def __init__(self, session_id: str, process: subprocess.Popen[str], max_event_bytes: int) -> None:
        self.session_id, self.process = session_id, process
        self.max_event_bytes = max_event_bytes
        self.events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=8)
        self.stderr_tail = ""
        self.lock = threading.Lock()
        self.conversation_id: str | None = None
        self.created_at = self.last_used_at = time.time()
        self.readers = [
            threading.Thread(target=self._read_stdout, daemon=True),
            threading.Thread(target=self._read_stderr, daemon=True),
        ]
        for reader in self.readers:
            reader.start()

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        while True:
            try:
                line = self.process.stdout.readline(self.max_event_bytes + 1)
            except (OSError, ValueError):
                return
            if not line:
                break
            if len(line.encode("utf-8", "replace")) > self.max_event_bytes or not line.endswith("\n"):
                if not line.endswith("\n"):
                    while line and not line.endswith("\n"):
                        try:
                            line = self.process.stdout.readline(self.max_event_bytes + 1)
                        except (OSError, ValueError):
                            line = ""
                event = {"event": "protocol_error", "detail": "agy event exceeded size limit or was incomplete"}
            else:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    event = {"event": "protocol_error", "detail": "invalid agy event"}
            if not isinstance(event, dict):
                continue
            if event.get("event") == "init":
                self.conversation_id = event.get("conversation_id")
            if event.get("event") not in {"init", "result", "protocol_error"}:
                continue
            try:
                self.events.put_nowait(event)
            except queue.Full:
                try:
                    self.events.get_nowait()
                    self.events.put_nowait({"event": "protocol_error", "detail": "agy event queue overflow"})
                except queue.Empty:
                    pass

    def _read_stderr(self) -> None:
        assert self.process.stderr is not None
        while True:
            try:
                line = self.process.stderr.readline(1024)
            except (OSError, ValueError):
                return
            if not line:
                return
            self.stderr_tail = (self.stderr_tail + line)[-4096:]

    def wait_for(self, name: str, timeout: int) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(name)
            if self.process.poll() is not None and self.events.empty():
                raise RuntimeError(f"agy exited with code {self.process.returncode}")
            try:
                event = self.events.get(timeout=min(remaining, 1))
            except queue.Empty:
                continue
            if event.get("event") == "protocol_error":
                raise RuntimeError("agy returned an invalid stream event")
            if event.get("event") == name:
                return event

    def close(self) -> None:
        try:
            if self.process.stdin:
                self.process.stdin.close()
        except OSError:
            pass
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    pass
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            if stream:
                try:
                    stream.close()
                except OSError:
                    pass
        for reader in self.readers:
            reader.join(timeout=1)

def _result(value: dict[str, Any], error: bool = False) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(value, ensure_ascii=False)}],
        "structuredContent": value,
        "isError": error,
    }


class Server:
    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config.load()
        self.sessions: dict[str, PersistentSession] = {}
        self.session_lock = threading.RLock()
        self.pending_sessions = 0
        self.reaper_stop = threading.Event()
        self.reaper = threading.Thread(target=self._reap_idle_sessions, daemon=True)
        self.reaper.start()

    def _reap_idle_sessions(self) -> None:
        interval = min(60, max(1, self.config.session_idle_seconds / 2))
        while not self.reaper_stop.wait(interval):
            self._cleanup_idle_sessions()

    def tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "list_models",
                "description": "List configured model aliases for the local Antigravity login.",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "ask_model",
                "description": "Ask a model through the local Antigravity login.",
                "inputSchema": {"type": "object", "required": ["model", "prompt"], "properties": {
                    "model": {"type": "string"}, "prompt": {"type": "string"},
                    "system_instruction": {"type": "string"}, "timeout": {"type": "integer", "minimum": 1, "maximum": 600}
                }, "additionalProperties": False},
            },
            {
                "name": "list_antigravity_models",
                "description": "List locally configured Antigravity model aliases.",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "antigravity_health",
                "description": "Check whether the local agy executable is available.",
                "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
            },
            {
                "name": "ask_antigravity",
                "description": "Prompt a model through the existing local Antigravity login.",
                "inputSchema": {
                    "type": "object",
                    "required": ["model", "prompt"],
                    "properties": {
                        "model": {"type": "string"},
                        "prompt": {"type": "string"},
                        "system_instruction": {"type": "string"},
                        "working_directory": {"type": "string"},
                        "timeout": {"type": "integer", "minimum": 1, "maximum": 600},
                        "conversation_id": {"type": "string"},
                        "effort": {"type": "string", "enum": ["low", "medium", "high"]},
                        "mode": {"type": "string", "enum": ["plan", "accept-edits"]},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "create_antigravity_session",
                "description": "Start a persistent Antigravity process and pay initialization cost once.",
                "inputSchema": {"type": "object", "required": ["model"], "properties": {
                    "model": {"type": "string"}, "working_directory": {"type": "string"},
                    "effort": {"type": "string", "enum": ["low", "medium", "high"]},
                }, "additionalProperties": False},
            },
            {
                "name": "send_session_message",
                "description": "Send a turn through an existing persistent Antigravity process.",
                "inputSchema": {"type": "object", "required": ["session_id", "prompt"], "properties": {
                    "session_id": {"type": "string"}, "prompt": {"type": "string"},
                    "timeout": {"type": "integer", "minimum": 1, "maximum": 600},
                }, "additionalProperties": False},
            },
            {
                "name": "antigravity_session_status",
                "description": "Inspect a persistent session without a model request.",
                "inputSchema": {"type": "object", "required": ["session_id"],
                    "properties": {"session_id": {"type": "string"}}, "additionalProperties": False},
            },
            {
                "name": "close_antigravity_session",
                "description": "Terminate and remove a persistent Antigravity session.",
                "inputSchema": {"type": "object", "required": ["session_id"],
                    "properties": {"session_id": {"type": "string"}}, "additionalProperties": False},
            },
        ]

    def _working_directory(self, value: Any) -> Path:
        root = self.config.project_root
        candidate = (root / str(value)).resolve() if value else root
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("working_directory escapes project_root") from exc
        if not candidate.is_dir():
            raise ValueError("working_directory is not a directory")
        return candidate

    def _cleanup_idle_sessions(self) -> None:
        cutoff = time.time() - self.config.session_idle_seconds
        with self.session_lock:
            candidates = list(self.sessions.items())
        for session_id, session in candidates:
            # A turn refreshes last_used_at while holding this lock. Do not let
            # the background reaper terminate an active agy request.
            with session.lock:
                if session.last_used_at < cutoff or session.process.poll() is not None:
                    with self.session_lock:
                        if self.sessions.get(session_id) is not session:
                            continue
                        self.sessions.pop(session_id, None)
                    session.close()

    def create_session(self, args: dict[str, Any]) -> dict[str, Any]:
        self._cleanup_idle_sessions()
        alias = args.get("model")
        if alias not in self.config.models or not isinstance(self.config.models.get(alias), str):
            return _result({"error": "model_not_allowed", "allowed": sorted(self.config.models)}, True)
        try:
            cwd = self._working_directory(args.get("working_directory"))
        except ValueError as exc:
            return _result({"error": str(exc)}, True)
        executable = shutil.which(self.config.agy_path)
        if not executable:
            return _result({"error": "agy executable not found"}, True)
        effort = str(args.get("effort", self.config.effort))
        if effort not in {"low", "medium", "high"}:
            return _result({"error": "effort must be low, medium, or high"}, True)
        model = self.config.models[alias]
        argv = [executable, "--model", model]
        if "gemini" in model.lower():
            argv += ["--effort", effort]
        argv += ["--mode", "accept-edits", "--disable-slash-commands",
                 "--input-format", "stream-json", "--output-format", "stream-json", "--print="]
        with self.session_lock:
            if len(self.sessions) + self.pending_sessions >= self.config.max_sessions:
                return _result({"error": "session_limit_reached", "limit": self.config.max_sessions}, True)
            self.pending_sessions += 1
        session: PersistentSession | None = None
        try:
            process = subprocess.Popen(
                argv, cwd=cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1,
            )
            session_id = f"agy-{uuid.uuid4().hex[:12]}"
            session = PersistentSession(session_id, process, max(self.config.max_output_bytes, self.config.max_response_bytes) + 65536)
            init = session.wait_for("init", min(self.config.timeout, 30))
        except (OSError, TimeoutError, RuntimeError):
            if session:
                session.close()
            with self.session_lock:
                self.pending_sessions -= 1
            return _result({"error": "session_start_failed", "detail": "agy failed to start or returned an invalid stream"}, True)
        with self.session_lock:
            self.pending_sessions -= 1
            self.sessions[session_id] = session
        return _result({"session_id": session_id, "conversation_id": init.get("conversation_id"),
                        "model": alias, "pid": process.pid, "ready": True})

    def send_session_message(self, args: dict[str, Any]) -> dict[str, Any]:
        self._cleanup_idle_sessions()
        with self.session_lock:
            session = self.sessions.get(str(args.get("session_id", "")))
        if not session:
            return _result({"error": "unknown_or_expired_session"}, True)
        prompt = args.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return _result({"error": "prompt must be non-empty text"}, True)
        if len(prompt.encode()) > self.config.max_prompt_bytes:
            return _result({"error": "prompt_too_large"}, True)
        timeout = int(args.get("timeout", self.config.timeout))
        if not 1 <= timeout <= 600:
            return _result({"error": "timeout must be between 1 and 600"}, True)
        message = {"event": "user", "message": {"role": "user", "content": [
            {"type": "text", "text": prompt}
        ]}}
        started = time.monotonic()
        with session.lock:
            try:
                if session.process.poll() is not None or session.process.stdin is None:
                    raise RuntimeError("agy session is not running")
                session.last_used_at = time.time()
                session.process.stdin.write(json.dumps(message) + "\n")
                session.process.stdin.flush()
                event = session.wait_for("result", timeout)
            except (BrokenPipeError, OSError, TimeoutError, RuntimeError):
                with self.session_lock:
                    self.sessions.pop(session.session_id, None)
                session.close()
                return _result({"error": "session_turn_failed", "detail": "agy process failed or timed out"}, True)
            finally:
                session.last_used_at = time.time()
        result = event.get("result", {})
        response = result.get("response", "")
        if not isinstance(response, str):
            response = json.dumps(response, ensure_ascii=False)
        if len(response.encode("utf-8")) > self.config.max_response_bytes:
            response = response.encode("utf-8")[:self.config.max_response_bytes].decode("utf-8", "ignore") + "\n...[truncated]"
        return _result({
            "session_id": session.session_id,
            "conversation_id": result.get("conversation_id", session.conversation_id),
            "status": result.get("status"), "response": response,
            "usage": result.get("usage", {}), "num_turns": result.get("num_turns"),
            "request_duration_seconds": round(time.monotonic() - started, 3),
            "session_duration_seconds": result.get("duration_seconds"),
        }, error=result.get("status") != "SUCCESS")

    def session_status(self, args: dict[str, Any]) -> dict[str, Any]:
        self._cleanup_idle_sessions()
        with self.session_lock:
            session = self.sessions.get(str(args.get("session_id", "")))
        if not session:
            return _result({"error": "unknown_or_expired_session"}, True)
        return _result({"session_id": session.session_id, "conversation_id": session.conversation_id,
                        "running": session.process.poll() is None, "pid": session.process.pid,
                        "idle_seconds": round(time.time() - session.last_used_at, 3)})

    def close_session(self, args: dict[str, Any]) -> dict[str, Any]:
        session_id = str(args.get("session_id", ""))
        with self.session_lock:
            session = self.sessions.get(session_id)
        if not session:
            return _result({"error": "unknown_or_expired_session"}, True)
        with session.lock:
            with self.session_lock:
                if self.sessions.get(session_id) is not session:
                    return _result({"error": "unknown_or_expired_session"}, True)
                self.sessions.pop(session_id, None)
            session.close()
        return _result({"session_id": session_id, "closed": True})

    def close(self) -> None:
        self.reaper_stop.set()
        if threading.current_thread() is not self.reaper:
            self.reaper.join(timeout=1)
        with self.session_lock:
            sessions, self.sessions = list(self.sessions.values()), {}
        for session in sessions:
            with session.lock:
                session.close()

    def health(self) -> dict[str, Any]:
        executable = shutil.which(self.config.agy_path)
        return _result({
            "healthy": bool(executable),
            "agy_path": executable,
            "project_root": str(self.config.project_root),
        }, error=not bool(executable))

    def ask(self, args: dict[str, Any]) -> dict[str, Any]:
        alias = args.get("model")
        prompt = args.get("prompt")
        if alias not in self.config.models:
            return _result({"error": "model_not_allowed", "allowed": sorted(self.config.models)}, True)
        if not isinstance(prompt, str) or not prompt.strip():
            return _result({"error": "prompt must be non-empty text"}, True)
        system = args.get("system_instruction")
        if system is not None and not isinstance(system, str):
            return _result({"error": "system_instruction must be text"}, True)
        combined = f"System instruction:\n{system}\n\nUser request:\n{prompt}" if system else prompt
        size = len(combined.encode("utf-8"))
        if size > self.config.max_prompt_bytes:
            return _result({"error": "prompt_too_large", "bytes": size, "limit": self.config.max_prompt_bytes}, True)

        cwd = self.config.project_root
        if args.get("working_directory"):
            candidate = (cwd / str(args["working_directory"])).resolve()
            try:
                candidate.relative_to(cwd)
            except ValueError:
                return _result({"error": "working_directory escapes project_root"}, True)
            if not candidate.is_dir():
                return _result({"error": "working_directory is not a directory"}, True)
            cwd = candidate

        timeout = int(args.get("timeout", self.config.timeout))
        if not 1 <= timeout <= 600:
            return _result({"error": "timeout must be between 1 and 600"}, True)
        executable = shutil.which(self.config.agy_path)
        if not executable:
            return _result({"error": "agy executable not found"}, True)

        model = self.config.models[alias]
        argv = [executable, "--model", model]
        if "gemini" in model.lower():
            effort = str(args.get("effort", self.config.effort))
            if effort not in {"low", "medium", "high"}:
                return _result({"error": "effort must be low, medium, or high"}, True)
            argv += ["--effort", effort]
        mode = str(args.get("mode", self.config.mode))
        if mode not in {"plan", "accept-edits"}:
            return _result({"error": "mode must be plan or accept-edits"}, True)
        argv += ["--mode", mode, "--disable-slash-commands", "--output-format", "json"]
        conversation_id = args.get("conversation_id")
        if conversation_id:
            if not isinstance(conversation_id, str) or not conversation_id.strip():
                return _result({"error": "conversation_id must be non-empty text"}, True)
            argv += ["--conversation", conversation_id.strip()]
        argv += [f"--print={combined}"]
        try:
            completed = subprocess.run(
                argv, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=timeout, check=False,
            )
        except subprocess.TimeoutExpired:
            return _result({"error": "timeout", "timeout_seconds": timeout}, True)
        except OSError:
            return _result({"error": "execution_failed", "detail": "could not start agy"}, True)

        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        if len(stdout.encode()) > self.config.max_output_bytes:
            stdout = stdout.encode()[:self.config.max_output_bytes].decode(errors="replace") + "\n...[truncated]"
        if len(stderr.encode()) > self.config.max_output_bytes:
            stderr = stderr.encode()[:self.config.max_output_bytes].decode(errors="replace") + "\n...[truncated]"
        if completed.returncode != 0:
            return _result({"error": "agy_failed", "exit_code": completed.returncode, "stderr": stderr, "stdout": stdout}, True)

        response, conversation_id, usage = stdout.strip(), None, {}
        try:
            envelope = json.loads(stdout)
            if isinstance(envelope, dict):
                response = str(envelope.get("response", response))
                conversation_id = envelope.get("conversation_id")
                usage = envelope.get("usage", {}) if isinstance(envelope.get("usage"), dict) else {}
        except json.JSONDecodeError:
            pass
        return _result({"model": alias, "response": response, "conversation_id": conversation_id, "usage": usage})

    def handle(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method, request_id = request.get("method"), request.get("id")
        if method in {"notifications/initialized", "notifications/cancelled"}:
            return None
        if method == "initialize":
            return {"jsonrpc": "2.0", "id": request_id, "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "antigravity-mcp-lite", "version": "0.2.0"},
            }}
        if method == "ping":
            return {"jsonrpc": "2.0", "id": request_id, "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": self.tools()}}
        if method == "tools/call":
            params, args = request.get("params") or {}, (request.get("params") or {}).get("arguments") or {}
            name = params.get("name")
            if name == "list_antigravity_models":
                result = _result({"models": self.config.models})
            elif name == "list_models":
                result = _result({"models": self.config.models})
            elif name == "antigravity_health":
                result = self.health()
            elif name in {"ask_antigravity", "ask_model"}:
                result = self.ask(args)
            elif name == "create_antigravity_session":
                result = self.create_session(args)
            elif name == "send_session_message":
                result = self.send_session_message(args)
            elif name == "antigravity_session_status":
                result = self.session_status(args)
            elif name == "close_antigravity_session":
                result = self.close_session(args)
            else:
                result = _result({"error": "unknown_tool", "tool": name}, True)
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": "Method not found"}}

    def run(self) -> None:
        try:
            for line in sys.stdin:
                try:
                    response = self.handle(json.loads(line))
                except (json.JSONDecodeError, TypeError, ValueError):
                    response = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Invalid request"}}
                if response is not None:
                    print(json.dumps(response), flush=True)
        finally:
            self.close()


def main() -> None:
    Server().run()


if __name__ == "__main__":
    main()
