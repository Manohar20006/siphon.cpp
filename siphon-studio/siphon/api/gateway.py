"""Siphon API gateway: one local endpoint, two key scopes.

    decode   key  -> /v1/chat/completions, /v1/completions, /v1/responses, /v1/models   (OpenAI-compatible)
                     /v1/messages, /v1/messages/count_tokens                        (Anthropic-compatible)
                     proxied to llama-server with streaming passthrough
    decision key  -> /v1/decide, /v1/decide/warmup                        (siphon.api.decide)

The gateway binds 127.0.0.1 only. It serves whichever Studio profile is running and detects what that
server can do from its profile env:
    LLAMA_ARG_DECISION_ONLY=1      -> decision engine only
    LLAMA_ARG_DECISION_CTX > 0     -> both (one process: generation + decision context)
    otherwise                      -> decode only
Requests for an engine that is not running get 503 with the reason.
"""

from __future__ import annotations

import http.client
import json
import os
import threading
import time
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .decide import DecisionEngine, DecisionError, backend_key_from_env, parse_request
from .keys import SCOPES, KeyStore

HOST = "127.0.0.1"
DEFAULT_PORT = 8088
MAX_BODY = 64 * 1024 * 1024
_HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers",
               "transfer-encoding", "upgrade", "content-length"}

DECODE_ROUTES = {("POST", "/v1/chat/completions"), ("POST", "/v1/completions"), ("POST", "/v1/responses"),
                 ("POST", "/v1/messages"), ("POST", "/v1/messages/count_tokens")}   # OpenAI + Anthropic shapes
DECISION_ROUTES = {("POST", "/v1/decide"), ("POST", "/v1/decide/warmup")}


def default_port() -> int:
    try:
        return int(os.environ.get("SIPHON_API_PORT", DEFAULT_PORT))
    except ValueError:
        return DEFAULT_PORT


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "on", "yes", "enabled"}


@dataclass
class Backend:
    profile: str
    base_url: str
    alias: str
    decode: bool
    decision: bool

    @property
    def mode(self) -> str:
        if self.decode and self.decision:
            return "decode+decision"
        return "decision" if self.decision else "decode"


def backend_for(profile) -> Backend:
    env = profile.env
    decision_only = _truthy(env.get("LLAMA_ARG_DECISION_ONLY", ""))
    try:
        decision_ctx = int(env.get("LLAMA_ARG_DECISION_CTX", "0") or 0)
    except ValueError:
        decision_ctx = 0
    return Backend(profile=profile.name, base_url=profile.base_url, alias=env.get("ALIAS", profile.name),
                   decode=not decision_only, decision=decision_only or decision_ctx > 0)


class BackendResolver:
    """Finds the running profile. Status checks spawn processes (~250 ms), so they never run on the request path
    once the first lookup is done: a background thread refreshes the answer every `interval` seconds."""

    def __init__(self, interval: float = 5.0, background: bool = False):
        self.interval = interval
        self._lock = threading.Lock()
        self._resolved = False
        self._backend: Backend | None = None
        self._engines: dict[tuple[str, str, str], DecisionEngine] = {}
        if background:
            threading.Thread(target=self._loop, name="siphon-api-backend", daemon=True).start()

    @staticmethod
    def _lookup() -> Backend | None:
        from ..core import runner
        running = runner.running_profiles()
        return backend_for(running[0]) if running else None

    def _loop(self) -> None:
        while True:
            try:
                backend = self._lookup()
                with self._lock:
                    self._backend, self._resolved = backend, True
            except Exception:   # keep the last answer; the next round retries
                pass
            time.sleep(self.interval)

    def current(self) -> Backend | None:
        with self._lock:
            if self._resolved:
                return self._backend
        backend = self._lookup()
        with self._lock:
            self._backend, self._resolved = backend, True
            return backend

    def engine(self, backend: Backend) -> DecisionEngine:
        key = (backend.profile, backend.base_url, backend.alias)
        with self._lock:
            engine = self._engines.get(key)
            if engine is None:
                self._engines.clear()
                engine = self._engines[key] = DecisionEngine(backend.base_url, backend_key_from_env(), backend.alias)
            return engine


class Activity:
    """Per-path request counters for the Studio API tab. In memory only: they start at zero with the gateway.

    Only requests that passed the key check are counted, so unauthenticated probes never show up as traffic.
    """

    MINUTES = 30

    def __init__(self):
        self._lock = threading.Lock()
        self._paths = {scope: {"requests": 0, "errors": 0, "last_at": 0.0, "last_ms": 0.0, "last_status": 0,
                               "last_route": "", "minutes": {}} for scope in SCOPES}

    def record(self, scope: str, route: str, status: int, ms: float, now: float | None = None) -> None:
        if scope not in self._paths:
            return
        now = time.time() if now is None else now
        minute = int(now // 60)
        with self._lock:
            path = self._paths[scope]
            path["requests"] += 1
            path["errors"] += status >= 400
            path.update(last_at=now, last_ms=ms, last_status=status, last_route=route)
            minutes = path["minutes"]
            minutes[minute] = minutes.get(minute, 0) + 1
            for old in [m for m in minutes if m <= minute - self.MINUTES]:
                del minutes[old]

    def snapshot(self, now: float | None = None) -> dict:
        """{scope: {requests, errors, last_at, last_ms, last_status, last_route, per_minute (oldest first)}}"""
        minute = int((time.time() if now is None else now) // 60)
        with self._lock:
            return {scope: {**{k: v for k, v in path.items() if k != "minutes"},
                            "per_minute": [path["minutes"].get(minute - i, 0) for i in range(self.MINUTES - 1, -1, -1)]}
                    for scope, path in self._paths.items()}


class Gateway(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, port: int, keys: KeyStore | None = None, resolver: BackendResolver | None = None):
        self.keys = keys or KeyStore()
        self.resolver = resolver or BackendResolver(background=True)
        self.activity = Activity()
        super().__init__((HOST, port), Handler)


class Handler(BaseHTTPRequestHandler):
    server: Gateway
    server_version = "siphon-api"
    protocol_version = "HTTP/1.0"   # responses end at connection close -> simple streaming passthrough

    # -- plumbing -------------------------------------------------------------
    def log_message(self, fmt, *args):   # never log headers (keys) or bodies; method + path + status only
        pass

    def send_response(self, code, message=None):
        self._status = code
        super().send_response(code, message)

    def _json(self, status: int, obj: dict) -> None:
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: int, message: str, kind: str = "invalid_request_error") -> None:
        self._json(status, {"error": {"message": message, "type": kind, "code": status}})

    def _body(self) -> bytes | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if length < 0 or length > MAX_BODY:
            self._error(413, "request body too large or invalid Content-Length")
            return None
        return self.rfile.read(length) if length else b""

    def _token(self) -> str:
        auth = self.headers.get("Authorization", "")
        if auth[:7].lower() == "bearer ":
            return auth[7:].strip()
        return self.headers.get("X-API-Key", "").strip()

    def _authorize(self, scopes: set[str]) -> bool:
        record = self.server.keys.verify(self._token())
        if record is None:
            self._error(401, "missing or invalid API key (send 'Authorization: Bearer <key>')",
                        "authentication_error")
            return False
        if record.scope not in scopes:
            self._error(403, f"this is a {record.scope} key; {self.path} needs a {' or '.join(sorted(scopes))} key",
                        "permission_error")
            return False
        self._scope = record.scope
        return True

    def _backend(self, need: str) -> Backend | None:
        backend = self.server.resolver.current()
        if backend is None:
            self._error(503, "no Siphon model is running; start a profile in Siphon Studio", "unavailable")
            return None
        if not getattr(backend, need):
            self._error(503, f"profile '{backend.profile}' runs in {backend.mode} mode; "
                             f"start a profile with the {need} engine", "unavailable")
            return None
        return backend

    # -- routes ---------------------------------------------------------------
    def do_GET(self):
        path = urllib.parse.urlsplit(self.path).path.rstrip("/") or "/"
        if path == "/health":
            backend = self.server.resolver.current()
            self._json(200, {"status": "ok", "backend": None if backend is None else
                             {"profile": backend.profile, "mode": backend.mode, "model": backend.alias}})
            return
        if path == "/v1/models":
            if not self._authorize({"decode", "decision"}):
                return
            backend = self.server.resolver.current()
            data = [] if backend is None else [{"id": backend.alias, "object": "model", "owned_by": "siphon",
                                                "siphon_mode": backend.mode}]
            self._json(200, {"object": "list", "data": data})
            return
        self._error(404, f"unknown route GET {path}")

    def do_POST(self):
        path = urllib.parse.urlsplit(self.path).path.rstrip("/")
        self._scope, self._status, t0 = "", 0, time.perf_counter()
        try:
            if ("POST", path) in DECISION_ROUTES:
                self._decide(path)
            elif ("POST", path) in DECODE_ROUTES:
                self._proxy(path)
            else:
                self._error(404, f"unknown route POST {path}")
        finally:
            if self._scope:     # authorized requests only; a stream counts when it ends
                self.server.activity.record(self._scope, path, self._status, (time.perf_counter() - t0) * 1000)

    def _decide(self, path: str) -> None:
        if not self._authorize({"decision"}):
            return
        raw = self._body()
        if raw is None:
            return
        try:
            body = json.loads(raw or b"{}")
            if not path.endswith("/warmup"):
                parse_request(body)
        except json.JSONDecodeError:
            self._error(400, "body is not valid JSON")
            return
        except DecisionError as exc:
            self._error(exc.status, str(exc))
            return
        backend = self._backend("decision")
        if backend is None:
            return
        engine = self.server.resolver.engine(backend)
        try:
            result = engine.warmup(body) if path.endswith("/warmup") else engine.decide(body)
        except DecisionError as exc:
            self._error(exc.status, str(exc), "decision_error" if exc.status >= 500 else "invalid_request_error")
            return
        self._json(200, result)

    def _proxy(self, path: str) -> None:
        if not self._authorize({"decode"}):
            return
        raw = self._body()
        if raw is None:
            return
        backend = self._backend("decode")
        if backend is None:
            return
        target = urllib.parse.urlsplit(backend.base_url)
        conn = http.client.HTTPConnection(target.hostname, target.port, timeout=3600)
        headers = {"Content-Type": self.headers.get("Content-Type", "application/json"),
                   "Accept": self.headers.get("Accept", "*/*")}
        backend_key = backend_key_from_env()
        if backend_key:
            headers["Authorization"] = f"Bearer {backend_key}"
        try:
            conn.request("POST", path, body=raw, headers=headers)
            resp = conn.getresponse()
        except OSError as exc:
            conn.close()
            self._error(503, f"model server unreachable: {exc}", "unavailable")
            return
        try:
            self.send_response(resp.status)
            for name, value in resp.getheaders():
                if name.lower() not in _HOP_BY_HOP:
                    self.send_header(name, value)
            self.end_headers()
            while True:
                chunk = resp.read1(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass    # client went away; closing the backend connection cancels the generation
        finally:
            conn.close()


def serve(port: int | None = None) -> Gateway:
    """Create a gateway bound to 127.0.0.1 (caller runs serve_forever)."""
    return Gateway(port or default_port())


def start_background(port: int | None = None) -> Gateway:
    gateway = serve(port)
    threading.Thread(target=gateway.serve_forever, name="siphon-api", daemon=True).start()
    return gateway
