"""Filesystem layout for siphon-studio and the siphon.cpp repo it manages."""

from __future__ import annotations

import os
from pathlib import Path


def studio_dir() -> Path:
    """Top-level siphon-studio directory (parent of the siphon package)."""
    env = os.environ.get("SIPHON_STUDIO_DIR")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parent.parent


def repo_root() -> Path:
    """Discover the siphon.cpp repository root (checks env, then walks up)."""
    env = os.environ.get("SIPHON_REPO_ROOT")
    if env:
        return Path(env).resolve()

    start = studio_dir().resolve()
    for cand in (start, *start.parents):
        if (cand / "build-gds-cuda" / "bin" / "llama-server").exists():
            return cand
        if (cand / ".git").is_dir() and (cand / "scripts" / "run-qwen36-moe.sh").exists():
            return cand
    return start


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


# studio-managed state -----------------------------------------------------
def profiles_dir() -> Path:      # editable app profiles (JSON)
    return _ensure(studio_dir() / "profiles")

def state_dir() -> Path:         # runtime artifacts
    return _ensure(studio_dir() / "state")

def logs_dir() -> Path:          # per-profile server log files
    return _ensure(state_dir() / "logs")

def run_dir() -> Path:           # pid files (direct driver)
    return _ensure(state_dir() / "run")

def env_dir() -> Path:           # materialized .env files given to run scripts
    return _ensure(state_dir() / "env")

def cmd_dir() -> Path:           # generated bash launcher scripts (tmux/direct)
    return _ensure(state_dir() / "cmd")


# siphon.cpp repo paths -----------------------------------------------------
def configs_dir() -> Path:
    return repo_root() / "configs"

def scripts_dir() -> Path:
    return repo_root() / "scripts"

def models_dir() -> Path:
    return repo_root() / "models"

def suites_dir() -> Path:
    return repo_root() / "expert-suites"


def default_run_script() -> str:
    """Relative path (to repo root) of the launcher used by default profiles."""
    env = os.environ.get("SIPHON_RUN_SCRIPT")
    if env:
        return env
    return "scripts/run-qwen36-moe.sh"


def server_bin() -> Path:
    """CUDA llama-server binary the run scripts invoke."""
    env = os.environ.get("SIPHON_SERVER_BIN")
    if env:
        return Path(env).resolve()
    return repo_root() / "build-gds-cuda" / "bin" / "llama-server"


def cpu_server_bin() -> Path:
    """Find the CPU llama-server, honoring an explicit binary override."""
    override = os.environ.get("SIPHON_CPU_SERVER_BIN")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise FileNotFoundError(
                f"SIPHON_CPU_SERVER_BIN does not name an executable file: {path}"
            )
        return path

    root = repo_root()
    candidates: list[Path] = []
    cpu_root = os.environ.get("SIPHON_CPU_ROOT")
    if cpu_root:
        candidates.append(Path(cpu_root).expanduser() / "build-cpu" / "bin" / "llama-server")
    candidates.extend((
        root / "build-cpu" / "bin" / "llama-server",
        root / "siphon-cpu" / "build-cpu" / "bin" / "llama-server",
        root.parent / "inference" / "siphon-cpu" / "build-cpu" / "bin" / "llama-server",
    ))
    for path in candidates:
        resolved = path.resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    searched = ", ".join(str(path.resolve()) for path in candidates)
    raise FileNotFoundError(
        "CPU llama-server binary not found; set SIPHON_CPU_SERVER_BIN or build it "
        f"with the CPU configuration (searched: {searched})"
    )


def web_ui_url(host: str, port: int) -> str:
    return f"http://{host}:{port}"
