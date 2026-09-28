"""Monitor: poll a running siphon llama-server for health, metrics and state.

Verified metric names (built from the actual server on this host):
  llamacpp:predict_tokens_seconds      -> generation tokens/s (decode)
  llamacpp:prompt_tokens_seconds       -> prompt tokens/s (prefill)
  llamacpp:n_decode_total              -> number of llama_decode() calls (MTP!)
  llamacpp:n_tokens_predicted          -> generated tokens
  llamacpp:n_prompt_tokens_processed   -> processed prompt tokens
  llamacpp:requests_processing         -> currently processing requests
"""

from __future__ import annotations

import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HEALTH_PATH = "/health"
METRICS_PATH = "/metrics"
DSTORAGE_PATH = "/siphon/telemetry"


@dataclass
class HealthState:
    ok: bool = False
    status: str = "down"          # "ok" | "loading" | "down"
    error: str = ""
    latency_ms: float = 0.0

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class MetricsSnapshot:
    decode_tps: float = 0.0
    prefill_tps: float = 0.0
    n_decode: int = 0
    n_predicted: int = 0
    n_prompt: int = 0
    n_processing: int = 0
    ts: float = field(default_factory=time.time)
    raw: dict[str, float] = field(default_factory=dict)


def _http_json(url: str, timeout: float = 3.0) -> dict[str, Any] | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json_loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


def json_loads(text: str) -> dict[str, Any] | None:
    import json
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def fetch_health(base_url: str, timeout: float = 3.0) -> HealthState:
    url = f"{base_url}{HEALTH_PATH}"
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        data = json_loads(body)
        if isinstance(data, dict) and data.get("status") == "ok":
            return HealthState(ok=True, status="ok", latency_ms=elapsed_ms)
        return HealthState(ok=False, status="loading", latency_ms=elapsed_ms)
    except urllib.error.HTTPError as exc:
        # llama-server returns 503 while the model is still loading
        if exc.code == 503:
            return HealthState(ok=False, status="loading", error="model loading")
        return HealthState(ok=False, status="down", error=f"HTTP {exc.code}")
    except (urllib.error.URLError, OSError) as exc:
        return HealthState(ok=False, status="down", error=str(exc))


def fetch_metrics(base_url: str, timeout: float = 3.0) -> MetricsSnapshot:
    """Fetch and parse the Prometheus /metrics endpoint."""
    snap = MetricsSnapshot()
    try:
        with urllib.request.urlopen(f"{base_url}{METRICS_PATH}", timeout=timeout) as resp:
            text = resp.read().decode("utf-8", "replace")
    except (urllib.error.URLError, OSError):
        return snap

    values: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("llamacpp:"):
            name, _, val = line.partition(" ")
            values[name.removeprefix("llamacpp:")] = float(val or 0)

    snap.raw = values
    snap.decode_tps = values.get("predicted_tokens_seconds", 0.0)
    snap.prefill_tps = values.get("prompt_tokens_seconds", 0.0)
    snap.n_decode = int(values.get("n_decode_total", 0))
    snap.n_predicted = int(values.get("n_tokens_predicted", 0))
    snap.n_prompt = int(values.get("n_prompt_tokens_processed", 0))
    snap.n_processing = int(values.get("requests_processing", 0))
    snap.ts = time.time()
    return snap


def fetch_dstorage_metrics(base_url: str, timeout: float = 1.0) -> dict[str, float]:
    """Fetch Siphon's lock-protected DirectStorage telemetry snapshot."""
    try:
        with urllib.request.urlopen(f"{base_url}{DSTORAGE_PATH}", timeout=timeout) as resp:
            data = json_loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    values: dict[str, float] = {}
    for key, value in data.items():
        try:
            values[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return values


# -- log-file based stats (the server logs the same fields) ---------------


def parse_log_stats(log_file: str | Path) -> dict[str, float]:
    """Parse the last 'print_timing' + draft acceptance from a server log.

    Returns keys: decode_tps, prefill_tps, draft_acceptance, prompt_tokens,
    generated_tokens.
    """
    stats: dict[str, float] = {}
    try:
        text = Path(log_file).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return stats

    decode_lines = re.findall(
        r"eval time =\s+[\d.]+ ms /\s+(\d+) tokens \(\s*[\d.]+ ms per token,\s+([\d.]+) tokens per second\)",
        text,
    )
    if decode_lines:
        _, tps = decode_lines[-1]
        stats["decode_tps"] = float(tps)

    prefill_lines = re.findall(
        r"prompt eval time =\s+[\d.]+ ms /\s+(\d+) tokens \(\s*[\d.]+ ms per token,\s+([\d.]+) tokens per second\)",
        text,
    )
    if prefill_lines:
        _, tps = prefill_lines[-1]
        stats["prefill_tps"] = float(tps)

    accept = re.findall(r"draft acceptance =\s+([\d.]+)", text)
    if accept:
        stats["draft_acceptance"] = float(accept[-1])

    return stats
