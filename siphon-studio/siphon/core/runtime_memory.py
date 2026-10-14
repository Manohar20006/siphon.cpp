"""Persist successful engine allocation observations, never guessed workspaces."""
import json
from pathlib import Path
import threading
import uuid

_lock = threading.RLock()


def read_memory_record(path: Path, offset: int) -> dict | None:
    """Only the final real contexts emit this record, not fit/no-alloc probes."""
    record = None
    try:
        with path.open("rb") as stream:
            stream.seek(max(0, offset))
            for raw in stream:
                if not raw.startswith(b"SIPHON_GPU_MEMORY "):
                    continue
                try:
                    value = json.loads(raw[len(b"SIPHON_GPU_MEMORY "):])
                    if (isinstance(value, dict) and value.get("version") == 1 and
                            value.get("device_count") == 1 and isinstance(value.get("gpu_name"), str) and
                            all(type(value.get(key)) is int and 0 <= value[key] < 2**60
                                for key in ("model_bytes", "context_bytes", "compute_bytes"))):
                        record = value
                except (ValueError, TypeError):
                    continue
    except OSError:
        pass
    return record


def _read(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def load_observation(path: Path, name: str, identity: dict) -> dict | None:
    entry = _read(path).get(name)
    if isinstance(entry, dict) and entry.get("identity") == identity and isinstance(entry.get("record"), dict):
        return entry["record"]
    return None


def save_observation(path: Path, name: str, identity: dict, record: dict) -> None:
    """An optional atomic cache. Disk failures must not stop a healthy engine."""
    pending = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        with _lock:
            data = _read(path)
            data[name] = {"identity": identity, "record": record}
            path.parent.mkdir(parents=True, exist_ok=True)
            pending.write_text(json.dumps(data) + "\n", encoding="utf-8")
            pending.replace(path)
    except OSError:
        pass
    finally:
        try:
            pending.unlink(missing_ok=True)
        except OSError:
            pass
