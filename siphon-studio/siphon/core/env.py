"""Small .env parser/writer tailored to the siphon.cpp config files.

Handles the shapes used by configs/*.env:

    KEY=value
    export KEY=value
    KEY='{"preserve_thinking": true}'      # single-quoted
    KEY="some value"                       # double-quoted
    # comments and blank lines
"""

from __future__ import annotations

from pathlib import Path


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2:
        if value[0] == value[-1] and value[0] in ("'", '"'):
            return value[1:-1]
    return value


def parse_env_text(text: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        env[key] = _unquote(value)
    return env


def read_env(path: str | Path) -> dict[str, str]:
    return parse_env_text(Path(path).read_text(encoding="utf-8"))


def format_env(env: dict[str, str]) -> str:
    """Canonical formatting the run scripts can `source` (set -a exports)."""
    out: list[str] = []
    for key, value in env.items():
        if value == "":
            out.append(f"{key}=")
            continue
        # quote only when needed; prefer single quotes so double quotes
        # inside values (e.g. chat-template-kwargs) survive bash sourcing
        if any(ch in value for ch in " \t'\""):
            if "'" not in value:
                out.append(f"{key}='{value}'")
            else:
                escaped = value.replace('"', '\\"')
                out.append(f'{key}="{escaped}"')
        else:
            out.append(f"{key}={value}")
    return "\n".join(out) + ("\n" if out else "")


def write_env(path: str | Path, env: dict[str, str]) -> None:
    Path(path).write_text(format_env(env), encoding="utf-8")


def resolve_value(value: str, repo_root: Path) -> str:
    """Resolve a model/experts path against the repo root like the run script."""
    value = value.strip()
    if not value:
        return value
    if value.startswith(("http://", "https://")):
        return value
    if value.startswith("/"):
        return value
    return str((repo_root / value).resolve())