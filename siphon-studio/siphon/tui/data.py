"""Presentation-layer data for the siphon TUI: live server internals + system.

Everything in here is defensive and never raises. It wraps the llama-server
HTTP endpoints (`/props`, `/slots`, `/models`) that the core monitor does not
parse, plus `psutil` for host-level stats. No siphon.core logic is duplicated —
we only add the JSON/UI shaped data the dashboard needs.
"""

from __future__ import annotations

import time
import urllib.request
import urllib.error
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path  # noqa: F401  (kept for type hints in docstrings)

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


def _http_json(url: str, timeout: float = 2.0) -> dict | list | None:
    """Fetch a JSON endpoint, returning None on any failure. Never raises."""
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            import json

            return json.loads(resp.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, OSError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# History ring — feeds the live charts without unbounded growth.
# --------------------------------------------------------------------------- #


class Ring:
    """A fixed-size deque that can be rendered as a (label, value) list."""

    def __init__(self, size: int = 90, label: str = "") -> None:
        self.size = size
        self.label = label
        self._q: deque[float] = deque(maxlen=size)

    def push(self, value: float) -> None:
        self._q.append(float(value))

    def last(self) -> float:
        return self._q[-1] if self._q else 0.0

    def peak(self) -> float:
        return max(self._q) if self._q else 0.0

    def mean(self) -> float:
        return sum(self._q) / len(self._q) if self._q else 0.0

    def plain(self) -> list[float]:
        return list(self._q)


# --------------------------------------------------------------------------- #
# /props — model load-time properties (context, parallel slots, ...)
# --------------------------------------------------------------------------- #


@dataclass
class Props:
    model: str = ""
    ctx: int = 0
    n_parallel: int = 1
    n_gpu_layers: int = 0
    n_kv: int = 0
    chat_template: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.model)


def fetch_props(base_url: str, timeout: float = 2.0) -> Props:
    data = _http_json(f"{base_url}/props", timeout)
    if not isinstance(data, dict):
        return Props()
    def _num(key: str, default: int = 0) -> int:
        v = data.get(key)
        try:
            return int(v)
        except (TypeError, ValueError):
            return default
    return Props(
        model=str(data.get("model_path") or data.get("model") or ""),
        ctx=_num("default_generation_settings", 0) or _num("ctx", 0),
        n_parallel=_num("n_parallel", 1),
        n_gpu_layers=_num("n_gpu_layers", 0),
        n_kv=_num("n_kv", 0),
        chat_template=str(data.get("chat_template") or ""),
    )


# --------------------------------------------------------------------------- #
# /slots — per-slot live state (state, n_ctx, n_past, tokens, model)
# --------------------------------------------------------------------------- #


@dataclass
class Slot:
    idx: int
    state: str = "idle"
    n_ctx: int = 0
    n_past: int = 0
    tokens: int = 0
    model: str = ""
    n_decode: int = 0

    @property
    def pct(self) -> float:
        return (self.n_past / self.n_ctx * 100.0) if self.n_ctx else 0.0

    @property
    def busy(self) -> bool:
        return self.state not in ("idle", "computed")


@dataclass
class Slots:
    slots: list[Slot] = field(default_factory=list)
    n_parallel: int = 1
    busy: int = 0
    total_past: int = 0
    ts: float = field(default_factory=time.time)

    def __bool__(self) -> bool:
        return self.n_parallel > 0 and len(self.slots) > 0


def fetch_slots(base_url: str, base_props: Props | None = None, timeout: float = 2.0) -> Slots:
    data = _http_json(f"{base_url}/slots", timeout)
    if not isinstance(data, dict):
        data = {}
    slot_list = data.get("slots")
    if not isinstance(slot_list, list):
        return Slots()
    n_par = base_props.n_parallel if (base_props and base_props.ok) else max(len(slot_list), 1)
    out: list[Slot] = []
    busy = 0
    total_past = 0
    for i, s in enumerate(slot_list):
        if not isinstance(s, dict):
            continue
        state = str(s.get("state") or "idle")
        n_ctx = _to_int(s.get("n_ctx"), 0)
        n_past = _to_int(s.get("n_past"), 0)
        tokens = _to_int(s.get("tokens_processed") or s.get("tokens"), 0)
        sc = Slot(
            idx=i,
            state=state,
            n_ctx=n_ctx,
            n_past=n_past,
            tokens=tokens,
            model=str(s.get("model") or "").rsplit("/", 1)[-1],
            n_decode=_to_int(s.get("n_decode_total"), 0),
        )
        if sc.busy:
            busy += 1
        total_past += n_past
        out.append(sc)
    return Slots(slots=out, n_parallel=n_par, busy=busy, total_past=total_past)


def _to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# /v1/models — OpenAI-style model list (metadata + that the endpoint is alive)
# --------------------------------------------------------------------------- #


def fetch_model_report(base_url: str, timeout: float = 2.0) -> str:
    """Return a short human string describing the served model, or empty."""
    data = _http_json(f"{base_url}/v1/models", timeout)
    if not isinstance(data, dict):
        return ""
    items = data.get("data")
    if not isinstance(items, list) or not items:
        return ""
    return ", ".join(str(m.get("id") or "unknown") for m in items[:3])


# --------------------------------------------------------------------------- #
# System stats (host) — guarded import; returns zeros when psutil is missing.
# --------------------------------------------------------------------------- #


@dataclass
class SysStats:
    cpu: float = 0.0
    cpu_temp: float = 0.0
    cpu_freq_mhz: float = 0.0
    mem_pct: float = 0.0
    mem_used_gb: float = 0.0
    mem_total_gb: float = 0.0
    mem_free_gb: float = 0.0
    gpu_mib_free: float = 0.0
    gpu_mib_used: float = 0.0
    gpu_mib_total: float = 0.0
    gpu_temp: float = 0.0
    gpu_util: float = 0.0
    gpu_power_w: float = 0.0
    gpu_name: str = ""
    nvme_temp: float = 0.0
    load: float = 0.0

    @property
    def gpu_pct_free(self) -> float:
        if not self.gpu_mib_total:
            return 0.0
        return self.gpu_mib_free / self.gpu_mib_total * 100.0

    @property
    def gpu_pct_used(self) -> float:
        if not self.gpu_mib_total:
            return 0.0
        return self.gpu_mib_used / self.gpu_mib_total * 100.0


def fetch_sys_stats() -> SysStats:
    stats = SysStats()
    if psutil:
        try:
            stats.cpu = float(psutil.cpu_percent(interval=None))
            vm = psutil.virtual_memory()
            stats.mem_pct = float(vm.percent)
            stats.mem_used_gb = vm.used / (1024 ** 3)
            stats.mem_total_gb = vm.total / (1024 ** 3)
            stats.mem_free_gb = vm.available / (1024 ** 3)
            stats.load = float(psutil.getloadavg()[0])
            freq = psutil.cpu_freq()
            if freq:
                stats.cpu_freq_mhz = float(freq.current)
            tbuf = psutil.sensors_temperatures()
            stats.cpu_temp = _pick_temp(tbuf.get("coretemp"), "Package id 0")
            stats.nvme_temp = _pick_temp(tbuf.get("nvme"), "Composite")
        except Exception:
            pass
    g = _gpu_stats()
    stats.gpu_mib_free = g["free"]
    stats.gpu_mib_used = g["used"]
    stats.gpu_mib_total = g["total"]
    stats.gpu_temp = g["temp"]
    stats.gpu_util = g["util"]
    stats.gpu_power_w = g["power"]
    stats.gpu_name = g["name"]
    return stats


def _pick_temp(entries, label: str = "") -> float:
    """Return the first temperature whose label matches, else the first entry."""
    if not entries:
        return 0.0
    for s in entries:
        if label and s.label and label.lower() in str(s.label).lower():
            return float(s.current)
    return float(entries[0].current)


_gpu_cache: dict = {}
_GPU_CACHE_SECONDS = 2.0

def _gpu_stats() -> dict:
    """Query nvidia-smi for VRAM (+ temp/util/power), cached briefly.

    nvidia-smi can take a couple of seconds, so results are cached to avoid
    stalling the poll loop. Every field degrades to zero on any failure.
    """
    import time as _time

    global _gpu_cache
    now = _time.time()
    if _gpu_cache.get("_ts", 0) and (now - _gpu_cache["_ts"]) < _GPU_CACHE_SECONDS:
        return _gpu_cache
    out = {"name": "", "free": 0.0, "used": 0.0, "total": 0.0,
           "temp": 0.0, "util": 0.0, "power": 0.0}
    try:
        import subprocess

        res = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,memory.free,"
             "temperature.gpu,utilization.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if res.returncode == 0:
            lines = [line for line in res.stdout.splitlines() if line.strip()]
            if lines:
                parts = [part.strip() for part in lines[0].split(",")]
                if len(parts) >= 7:
                    out["name"] = parts[0]

                    # A laptop GPU can expose valid memory values while its
                    # temperature/utilisation sensors are waking up and are
                    # temporarily rendered as ``[N/A]``. Parse each field
                    # independently so partial telemetry remains useful.
                    def number(index: int) -> float:
                        try:
                            return float(parts[index])
                        except (IndexError, TypeError, ValueError):
                            return 0.0

                    out["used"] = number(1)
                    out["total"] = number(2)
                    out["free"] = number(3)
                    out["temp"] = number(4)
                    out["util"] = number(5)
                    out["power"] = number(6)
    except Exception:
        pass
    out["_ts"] = now
    _gpu_cache = out
    return out
