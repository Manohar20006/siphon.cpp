"""Persist Studio-only catalogue exclusions without changing core discovery."""

from __future__ import annotations

import json
import uuid

from .. import paths


_EXCLUSION_FILE = "desktop-model-exclusions.json"


def load_excluded_model_keys() -> set[str]:
    """Load stable model keys hidden from the Studio catalogue."""
    state_file = paths.state_dir() / _EXCLUSION_FILE
    try:
        payload = json.loads(state_file.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return set()
    if (not isinstance(payload, dict) or payload.get("version") != 1 or
            not isinstance(payload.get("excluded_models"), list) or
            any(not isinstance(key, str) or not key for key in payload["excluded_models"])):
        raise ValueError(f"Invalid Studio catalogue exclusions in {state_file}")
    return set(payload["excluded_models"])


def save_excluded_model_keys(model_keys: set[str]) -> None:
    """Atomically save Studio catalogue exclusions under its state directory."""
    state_dir = paths.state_dir()
    state_file = state_dir / _EXCLUSION_FILE
    temporary = state_file.with_name(f".{state_file.name}.{uuid.uuid4().hex}.tmp")
    payload = {
        "version": 1,
        "excluded_models": sorted({key for key in model_keys if isinstance(key, str) and key}),
    }
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(state_file)
    finally:
        temporary.unlink(missing_ok=True)
