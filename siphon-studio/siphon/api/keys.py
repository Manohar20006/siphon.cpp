"""API keys for the local Siphon API: one permanent key per path.

  decode   - text generation (OpenAI-compatible /v1/chat/completions, /v1/completions, /v1/models)
  decision - /v1/decide (choice probabilities on the decision engine)

Each key is created on first use and stays the same until the user regenerates it, so apps can be configured
once. The gateway only listens on 127.0.0.1, and the user must be able to copy the key again at any time, so the
key is kept in state/api/keys.json (mode 0600, directory 0700, git-ignored) rather than as a hash only.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .. import paths

SCOPES = ("decode", "decision")
KEY_PREFIX = {"decode": "sk-siphon-gen-", "decision": "sk-siphon-dec-"}
_LAST_USED_WRITE_INTERVAL_S = 60.0


@dataclass
class ScopeKey:
    scope: str
    key: str
    created: float
    last_used: float = 0.0

    @property
    def hint(self) -> str:
        return f"{KEY_PREFIX[self.scope]}…{self.key[-4:]}"

    def public(self, *, with_key: bool = False) -> dict:
        data = {"scope": self.scope, "hint": self.hint, "created": self.created, "last_used": self.last_used}
        if with_key:
            data["key"] = self.key
        return data


def store_path() -> Path:
    directory = paths.state_dir() / "api"
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    ignore = directory / ".gitignore"
    if not ignore.exists():     # studio state/ is not ignored by the repo; never let keys be committed
        ignore.write_text("*\n")
    return directory / "keys.json"


def _new_key(scope: str) -> str:
    return KEY_PREFIX[scope] + secrets.token_urlsafe(32)


class KeyStore:
    """File-backed store of the two keys; safe to share between gateway threads and processes."""

    def __init__(self, path: Path | None = None):
        self.path = path or store_path()
        self._lock = threading.Lock()
        self._keys: dict[str, ScopeKey] = {}
        self._mtime = -1.0
        self._dirty_since = 0.0

    # -- persistence ---------------------------------------------------------
    def _load(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            self._keys, self._mtime = {}, -1.0
            return
        if mtime == self._mtime:
            return
        data = json.loads(self.path.read_text() or "{}")
        keys = data.get("keys", {}) if data.get("version") == 2 else {}   # v1 held hashes only: start fresh
        self._keys = {scope: ScopeKey(**rec) for scope, rec in keys.items() if scope in SCOPES}
        self._mtime = mtime

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        payload = json.dumps({"version": 2, "keys": {s: asdict(k) for s, k in self._keys.items()}}, indent=1)
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
        os.replace(tmp, self.path)
        self._mtime = self.path.stat().st_mtime
        self._dirty_since = 0.0

    # -- operations ----------------------------------------------------------
    def get(self, scope: str) -> ScopeKey:
        """The key for a scope, created on first use."""
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {', '.join(SCOPES)}")
        with self._lock:
            self._load()
            if scope not in self._keys:
                self._keys[scope] = ScopeKey(scope, _new_key(scope), time.time())
                self._save()
            return self._keys[scope]

    def all(self) -> dict[str, ScopeKey]:
        return {scope: self.get(scope) for scope in SCOPES}

    def regenerate(self, scope: str) -> ScopeKey:
        """Replace a scope's key; the old one stops working immediately."""
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {', '.join(SCOPES)}")
        with self._lock:
            self._load()
            self._keys[scope] = ScopeKey(scope, _new_key(scope), time.time())
            self._save()
            return self._keys[scope]

    def verify(self, token: str) -> ScopeKey | None:
        """Return the key record this token matches, or None."""
        if not token:
            return None
        with self._lock:
            self._load()
            match = None
            for record in self._keys.values():
                if hmac.compare_digest(record.key.encode(), token.encode()):
                    match = record
            if match is None:
                return None
            now = time.time()
            match.last_used = now
            if self._dirty_since == 0.0:
                self._dirty_since = now
            if now - self._dirty_since >= _LAST_USED_WRITE_INTERVAL_S:
                self._save()
            return match

    def flush(self) -> None:
        with self._lock:
            if self._dirty_since:
                self._save()
