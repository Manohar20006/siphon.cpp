"""Authenticated Streamable HTTP transport for the universal MCP bridge."""

from __future__ import annotations

import argparse
import hmac
import json
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .config import load_config
from .orchestrator import Orchestrator
from .server import SERVER_NAME, SUPPORTED_PROTOCOL_VERSIONS, _handle_request


DEFAULT_MAX_REQUEST_BYTES = 2 * 1024 * 1024


class MCPHTTPServer(ThreadingHTTPServer):
    """HTTP server carrying shared MCP state for one authenticated deployment."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        orchestrator: Orchestrator,
        bearer_token: str,
        allowed_origins: frozenset[str],
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
    ):
        super().__init__(address, MCPHTTPRequestHandler)
        self.orchestrator = orchestrator
        self.bearer_token = bearer_token
        self.allowed_origins = allowed_origins
        self.max_request_bytes = max_request_bytes
        self.orchestrator_lock = threading.RLock()


class MCPHTTPRequestHandler(BaseHTTPRequestHandler):
    """Minimal sessionless implementation of MCP Streamable HTTP."""

    server: MCPHTTPServer
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        # Avoid logging Authorization headers or request bodies. The standard
        # request line and status are sufficient for local diagnostics.
        super().log_message(format, *args)

    def _json_response(
        self, status: int, value: dict[str, Any], extra_headers: dict[str, str] | None = None
    ) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra_headers:
            for name, header_value in extra_headers.items():
                self.send_header(name, header_value)
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str, code: int = -32600) -> None:
        self._json_response(
            status,
            {"jsonrpc": "2.0", "id": None, "error": {"code": code, "message": message}},
        )

    def _authorized(self) -> bool:
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {self.server.bearer_token}"
        return hmac.compare_digest(supplied, expected)

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        return origin in self.server.allowed_origins

    def _mcp_path(self) -> bool:
        return urlsplit(self.path).path.rstrip("/") == "/mcp"

    def do_GET(self) -> None:
        path = urlsplit(self.path).path.rstrip("/") or "/"
        if path == "/healthz":
            self._json_response(HTTPStatus.OK, {"status": "ok", "server": SERVER_NAME})
            return
        if self._mcp_path():
            self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
            self.send_header("Allow", "POST")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._error(HTTPStatus.NOT_FOUND, "not found", -32601)

    def do_POST(self) -> None:
        if not self._mcp_path():
            self._error(HTTPStatus.NOT_FOUND, "not found", -32601)
            return
        if not self._origin_allowed():
            self._error(HTTPStatus.FORBIDDEN, "Origin is not allowed")
            return
        if not self._authorized():
            self.send_response(HTTPStatus.UNAUTHORIZED)
            self.send_header("WWW-Authenticate", "Bearer")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        protocol_version = self.headers.get("MCP-Protocol-Version")
        if protocol_version and protocol_version not in SUPPORTED_PROTOCOL_VERSIONS:
            self._error(
                HTTPStatus.BAD_REQUEST,
                "unsupported MCP protocol version; supported versions: "
                + ", ".join(sorted(SUPPORTED_PROTOCOL_VERSIONS)),
            )
            return
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
        if content_type != "application/json":
            self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json")
            return
        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._error(HTTPStatus.LENGTH_REQUIRED, "valid Content-Length is required")
            return
        if content_length < 1 or content_length > self.server.max_request_bytes:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request body is too large")
            return
        try:
            request = json.loads(self.rfile.read(content_length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(HTTPStatus.BAD_REQUEST, "invalid JSON", -32700)
            return
        if not isinstance(request, dict):
            self._error(HTTPStatus.BAD_REQUEST, "MCP message must be a JSON object")
            return

        with self.server.orchestrator_lock:
            response = _handle_request(self.server.orchestrator, request)
        if response is None:
            self.send_response(HTTPStatus.ACCEPTED)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self._json_response(HTTPStatus.OK, response)


def create_http_server(
    host: str,
    port: int,
    orchestrator: Orchestrator,
    bearer_token: str,
    allowed_origins: frozenset[str] = frozenset(),
    max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
) -> MCPHTTPServer:
    if not bearer_token:
        raise ValueError("a non-empty bearer token is required")
    if max_request_bytes < 1024 or max_request_bytes > 16 * 1024 * 1024:
        raise ValueError("max request bytes must be between 1 KiB and 16 MiB")
    return MCPHTTPServer(
        (host, port), orchestrator, bearer_token, allowed_origins, max_request_bytes
    )


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Universal Worker Bridge MCP HTTP server")
    parser.add_argument("--host", default=os.environ.get("UNIVERSAL_MCP_HTTP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("UNIVERSAL_MCP_HTTP_PORT", "8765")))
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    token = os.environ.get("UNIVERSAL_MCP_HTTP_TOKEN", "")
    if not token:
        raise SystemExit("UNIVERSAL_MCP_HTTP_TOKEN is required")
    origins = frozenset(
        item.strip()
        for item in os.environ.get("UNIVERSAL_MCP_ALLOWED_ORIGINS", "").split(",")
        if item.strip()
    )
    max_bytes = int(
        os.environ.get("UNIVERSAL_MCP_MAX_REQUEST_BYTES", str(DEFAULT_MAX_REQUEST_BYTES))
    )
    server = create_http_server(
        args.host,
        args.port,
        Orchestrator(load_config()),
        token,
        origins,
        max_bytes,
    )
    print(f"{SERVER_NAME} listening on http://{args.host}:{args.port}/mcp", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
