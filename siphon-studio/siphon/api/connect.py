"""One-click connections from coding apps to the local decode API.

Each connector writes only what Siphon owns and can take back out again:

  OpenCode     a "siphon" provider in ~/.config/opencode/opencode.json. OpenCode merges that file with
               opencode.jsonc, so a user who keeps their settings in opencode.jsonc is never edited. An existing
               opencode.json is backed up once before the first change, and must be plain JSON.
  Claude Code  a separate `claude-siphon` command next to `claude`. Plain `claude` keeps its normal account.
  Codex CLI    a separate `codex-siphon` command; ~/.codex/config.toml is not touched.

The key is written into these files, so they are created owner-only (0600 / 0700), like state/api/keys.json.
After the decode key is regenerated, refresh() rewrites every connected app with the new key.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

PROVIDER_ID = "siphon"
MARKER = "# siphon-studio connection"


class ConnectError(Exception):
    pass


@dataclass
class Target:
    endpoint: str          # gateway root, e.g. http://127.0.0.1:8088
    key: str               # decode key
    model: str             # model alias served on the decode path
    context: int = 65536


def _home() -> Path:
    return Path(os.environ.get("HOME") or Path.home())


def _config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or _home() / ".config")


def bin_dir() -> Path:
    """Where the wrapper commands go: next to the app's own command when it lives in the user's home."""
    return _home() / ".local" / "bin"


def _write_private(path: Path, text: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".siphon-tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w") as handle:
        handle.write(text)
    os.chmod(tmp, mode)
    os.replace(tmp, path)


def _tilde(path: Path) -> str:
    try:
        return "~/" + str(path.relative_to(_home()))
    except ValueError:
        return str(path)


# -- OpenCode -------------------------------------------------------------------------------------------------
def _opencode_file() -> Path:
    return _config_home() / "opencode" / "opencode.json"


def _opencode_read(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text() or "{}")
    except ValueError as exc:
        raise ConnectError(f"{_tilde(path)} is not plain JSON (comments?), so Siphon will not rewrite it. "
                           "Add the provider by hand with the key and URL above.") from exc
    if not isinstance(data, dict):
        raise ConnectError(f"{_tilde(path)} does not hold a JSON object")
    return data


def _opencode_connected() -> bool:
    try:
        provider = (_opencode_read(_opencode_file()).get("provider") or {}).get(PROVIDER_ID)
    except ConnectError:
        return False
    return isinstance(provider, dict)


def _opencode_connect(target: Target, *, backup: bool) -> str:
    path = _opencode_file()
    data = _opencode_read(path)
    if backup and path.exists() and not _opencode_connected():
        shutil.copy2(path, path.with_name(f"{path.name}.bak-siphon-{int(time.time())}"))
    data.setdefault("$schema", "https://opencode.ai/config.json")
    providers = data.setdefault("provider", {})
    if not isinstance(providers, dict):
        raise ConnectError(f'"provider" in {_tilde(path)} is not an object')
    providers[PROVIDER_ID] = {
        "npm": "@ai-sdk/openai-compatible",
        "name": "Siphon (local)",
        "options": {"baseURL": target.endpoint.rstrip("/") + "/v1", "apiKey": target.key},
        "models": {target.model: {"name": f"{target.model} (Siphon)",
                                  "limit": {"context": target.context, "output": 16384}}},
    }
    _write_private(path, json.dumps(data, indent=2) + "\n")
    return f"In OpenCode, run /models and pick {PROVIDER_ID}/{target.model}."


def _opencode_disconnect() -> None:
    path = _opencode_file()
    data = _opencode_read(path)
    providers = data.get("provider")
    if not isinstance(providers, dict) or PROVIDER_ID not in providers:
        return
    del providers[PROVIDER_ID]
    if not providers:
        del data["provider"]
    if set(data) <= {"$schema"}:      # nothing left but what Siphon added
        path.unlink()
    else:
        _write_private(path, json.dumps(data, indent=2) + "\n")


# -- wrapper commands (Claude Code, Codex) ----------------------------------------------------------------------
def _wrapper_path(name: str) -> Path:
    return bin_dir() / name


def _wrapper_connected(name: str) -> bool:
    path = _wrapper_path(name)
    try:
        return MARKER in path.read_text(errors="replace")[:400]
    except OSError:
        return False


def _write_wrapper(name: str, app: str, env: dict[str, str], unset: list[str], command: list[str]) -> None:
    path = _wrapper_path(name)
    if path.exists() and not _wrapper_connected(name):
        raise ConnectError(f"{_tilde(path)} already exists and was not made by Siphon; not overwriting it")
    lines = ["#!/bin/sh", f"{MARKER}: {app} -> local Siphon decode API.",
             "# Remove it with Disconnect in Siphon Studio (API tab). Your normal command is unchanged."]
    lines += [f"unset {name_}" for name_ in unset]
    lines += [f"export {k}={shlex.quote(v)}" for k, v in env.items()]
    lines.append("exec " + " ".join(shlex.quote(part) for part in command) + ' "$@"')
    _write_private(path, "\n".join(lines) + "\n", mode=0o700)


def _remove_wrapper(name: str) -> None:
    if _wrapper_connected(name):
        _wrapper_path(name).unlink()


def _claude_connect(target: Target, **_) -> str:
    env = {
        "ANTHROPIC_BASE_URL": target.endpoint.rstrip("/"),
        "ANTHROPIC_AUTH_TOKEN": target.key,
        "ANTHROPIC_MODEL": target.model,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": target.model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": target.model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": target.model,
        "CLAUDE_CODE_SUBAGENT_MODEL": target.model,
        "API_TIMEOUT_MS": "3000000",                       # long prompts prefill slowly on a laptop GPU
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }
    _write_wrapper("claude-siphon", "Claude Code", env, ["ANTHROPIC_API_KEY"], ["claude"])
    return "Run claude-siphon in a terminal. Plain claude still uses your normal account."


def _codex_connect(target: Target, **_) -> str:
    provider = (f'{{name="Siphon (local)", base_url="{target.endpoint.rstrip("/")}/v1", '
                f'env_key="SIPHON_API_KEY", wire_api="responses"}}')
    command = ["codex", "-c", f"model_providers.{PROVIDER_ID}={provider}", "-c", f"model_provider={PROVIDER_ID}",
               "-c", f'model="{target.model}"', "-c", f"model_context_window={target.context}"]
    _write_wrapper("codex-siphon", "Codex CLI", {"SIPHON_API_KEY": target.key}, [], command)
    return "Run codex-siphon in a terminal. Plain codex is unchanged."


# -- registry ---------------------------------------------------------------------------------------------------
APPS: dict[str, dict] = {
    "opencode": {"name": "OpenCode", "command": "opencode",
                 "connect": _opencode_connect, "disconnect": _opencode_disconnect, "connected": _opencode_connected,
                 "where": lambda: _tilde(_opencode_file())},
    "claude-code": {"name": "Claude Code", "command": "claude",
                    "connect": _claude_connect, "disconnect": lambda: _remove_wrapper("claude-siphon"),
                    "connected": lambda: _wrapper_connected("claude-siphon"),
                    "where": lambda: _tilde(_wrapper_path("claude-siphon"))},
    "codex": {"name": "Codex CLI", "command": "codex",
              "connect": _codex_connect, "disconnect": lambda: _remove_wrapper("codex-siphon"),
              "connected": lambda: _wrapper_connected("codex-siphon"),
              "where": lambda: _tilde(_wrapper_path("codex-siphon"))},
}

USAGE = {
    "opencode": "choose Siphon (local) in /models",
    "claude-code": "run claude-siphon in a terminal",
    "codex": "run codex-siphon in a terminal",
}


ABOUT = {
    "opencode": "Adds a Siphon provider to opencode.json",
    "claude-code": "Adds a claude-siphon command · claude stays as it is",
    "codex": "Adds a codex-siphon command · codex stays as it is",
}


def _app(app_id: str) -> dict:
    if app_id not in APPS:
        raise ConnectError(f"unknown app {app_id!r}")
    return APPS[app_id]


def status() -> list[dict]:
    result = []
    for app_id, app in APPS.items():
        result.append({"id": app_id, "name": app["name"], "installed": shutil.which(app["command"]) is not None,
                       "connected": bool(app["connected"]()), "where": app["where"](), "usage": USAGE[app_id],
                       "about": ABOUT[app_id]})
    return result


def disconnect(app_id: str) -> None:
    _app(app_id)["disconnect"]()


def _target_file() -> Path:
    from .keys import store_path
    return store_path().with_name("connections.json")


def _save_target(target: Target) -> None:
    """Remember where apps were pointed (not the key), so a key change can be applied without the desktop app."""
    _write_private(_target_file(), json.dumps({"endpoint": target.endpoint, "model": target.model,
                                               "context": target.context}) + "\n")


def connect(app_id: str, target: Target) -> str:
    app = _app(app_id)
    if shutil.which(app["command"]) is None:
        raise ConnectError(f"{app['name']} is not installed ({app['command']} is not on PATH)")
    usage = app["connect"](target, backup=True)
    _save_target(target)
    return usage


def refresh(key: str, target: Target | None = None) -> list[str]:
    """Rewrite every connected app with a new decode key (and target, if given). Returns the app ids updated."""
    if target is None:
        try:
            saved = json.loads(_target_file().read_text())
            target = Target(endpoint=saved["endpoint"], key=key, model=saved["model"],
                            context=int(saved.get("context", 65536)))
        except (OSError, ValueError, KeyError, TypeError):
            return []
    target.key = key
    updated = []
    for app_id, app in APPS.items():
        if app["connected"]():
            app["connect"](target, backup=False)
            updated.append(app_id)
    if updated:
        _save_target(target)
    return updated
