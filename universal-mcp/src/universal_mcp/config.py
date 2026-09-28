"""Configuration loading for the universal worker bridge."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import WorkerConfig


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda match: os.environ.get(match.group(1), ""), value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    workers: tuple[WorkerConfig, ...]
    config_path: Path | None = None
    storage_path: Path | None = None
    allowed_command_prefixes: tuple[tuple[str, ...], ...] = ()

    def worker(self, worker_id: str) -> WorkerConfig:
        for worker in self.workers:
            if worker.worker_id == worker_id:
                return worker
        raise KeyError(f"unknown worker: {worker_id}")


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """Load worker configuration without ever reading literal API keys."""

    explicit_path = config_path or os.environ.get("UNIVERSAL_MCP_CONFIG")
    path = (
        Path(explicit_path).expanduser()
        if explicit_path
        else Path("universal-mcp/workers.json")
    )

    if not path.exists():
        if explicit_path:
            raise FileNotFoundError(f"MCP config does not exist: {path}")
        raw: dict[str, Any] = {}
        resolved_path: Path | None = None
    else:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError("MCP config root must be a JSON object")
        raw = _expand_env(loaded)
        resolved_path = path.resolve()

    project_root_value = os.environ.get(
        "UNIVERSAL_MCP_PROJECT_ROOT", raw.get("project_root", ".")
    )
    project_root = Path(str(project_root_value)).expanduser().resolve()
    if not project_root.is_dir():
        raise ValueError(f"project_root is not a directory: {project_root}")

    raw_workers = raw.get("workers", [])
    if not isinstance(raw_workers, list):
        raise ValueError("workers must be a JSON array")
    workers = tuple(WorkerConfig.from_dict(worker) for worker in raw_workers)

    ids = [worker.worker_id for worker in workers]
    if len(ids) != len(set(ids)):
        raise ValueError("worker ids must be unique")

    raw_prefixes = raw.get("allowed_command_prefixes", [])
    if not isinstance(raw_prefixes, list):
        raise ValueError("allowed_command_prefixes must be a JSON array")
    allowed_command_prefixes: list[tuple[str, ...]] = []
    for index, prefix in enumerate(raw_prefixes):
        if not isinstance(prefix, list) or not prefix or any(
            not isinstance(part, str) or not part or "\0" in part for part in prefix
        ):
            raise ValueError(
                f"allowed_command_prefixes[{index}] must be a non-empty string array"
            )
        allowed_command_prefixes.append(tuple(prefix))

    storage_value = os.environ.get(
        "UNIVERSAL_MCP_STORAGE_PATH", raw.get("storage_path")
    )
    return AppConfig(
        project_root=project_root,
        workers=workers,
        config_path=resolved_path,
        storage_path=Path(str(storage_value)).expanduser().resolve()
        if storage_value else None,
        allowed_command_prefixes=tuple(allowed_command_prefixes),
    )
