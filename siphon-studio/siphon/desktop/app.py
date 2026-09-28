"""Lightweight GTK/WebKit desktop host for the Siphon Studio surface.

The visual surface intentionally remains the approved HTML prototype.  GTK4
provides the native window and WebKitGTK renders the UI without bundling a
second Chromium runtime.  Runtime actions are delegated to the existing
``siphon.core`` process supervisor instead of reimplementing model launching
in the desktop layer.
"""

from __future__ import annotations

import json
import hashlib
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import gi

# This surface is a repo-local UI and does not browse arbitrary web content.
# Some Ubuntu desktop sessions deny WebKitGTK's Bubblewrap UID-map setup,
# which otherwise crashes the renderer before the window becomes usable.  Keep
# the app launchable by default on those systems; set SIPHON_WEBKIT_SANDBOX=1
# to force the normal WebKit renderer sandbox on a host that supports it.
if os.environ.get("SIPHON_WEBKIT_SANDBOX", "0").lower() not in {
    "1", "true", "yes", "on"
}:
    os.environ["WEBKIT_DISABLE_SANDBOX_THIS_IS_DANGEROUS"] = "1"
else:
    os.environ.pop("WEBKIT_DISABLE_SANDBOX_THIS_IS_DANGEROUS", None)

gi.require_version("Gdk", "4.0")
gi.require_version("Gtk", "4.0")
gi.require_version("WebKit", "6.0")
from gi.repository import Gdk, Gio, GLib, Gtk, WebKit

from .. import paths
from ..core import runner
from ..core.monitor import fetch_dstorage_metrics, fetch_metrics
from ..core.profiles import Profile, discover
from ..console.dstore import parse_log_tail
from ..tui.data import fetch_slots, fetch_sys_stats


APP_ID = "com.siphon.studio"
WINDOW_TITLE = "Siphon Studio"

# The prototype uses friendly catalogue keys.  The core uses the actual
# launcher profile names from configs/*.env.
PROFILE_BY_MODEL_KEY = {
    "qwen35": "qwen3.6-moe-tq3",
    "kat": "kat-coder-mtp",
    "coderNext": "qwen3-coder-next",
}
MODEL_DISPLAY_NAMES = {
    "qwen35": "Qwen3.6 MoE 35B",
    "kat": "KAT-Coder V2.5",
    "coderNext": "Qwen3-Coder-Next",
}


def _model_key_for_profile(profile: Profile) -> str:
    """Return a stable UI id without exposing a filesystem path to JavaScript."""
    for key, profile_name in PROFILE_BY_MODEL_KEY.items():
        if profile.name == profile_name:
            return key
    digest = hashlib.sha256(profile.resolve("MODEL").encode("utf-8")).hexdigest()[:12]
    return f"local-{digest}"


def _profile_map(profiles: dict[str, Profile]) -> dict[str, str]:
    mapping = {
        key: name for key, name in PROFILE_BY_MODEL_KEY.items() if name in profiles
    }
    known_paths = {
        str(Path(profiles[name].resolve("MODEL")).resolve())
        for name in mapping.values() if profiles[name].resolve("MODEL")
    }
    for profile in profiles.values():
        model_path = profile.resolve("MODEL")
        if not model_path:
            continue
        canonical_path = str(Path(model_path).resolve())
        if canonical_path in known_paths:
            continue
        mapping[_model_key_for_profile(profile)] = profile.name
        known_paths.add(canonical_path)
    return mapping

RAM_HEADROOM_MIB = 2 * 1024
VRAM_HEADROOM_MIB = 512


def _resource_state(system: Any) -> dict[str, Any]:
    """Return live allocation ceilings using available, not total, memory."""
    ram_available_mib = max(0, int(float(system.mem_free_gb or 0) * 1024))
    vram_available_mib = max(0, int(float(system.gpu_mib_free or 0)))
    cpu_threads = max(1, int(os.cpu_count() or 1))
    return {
        "ram_available_gb": float(system.mem_free_gb or 0),
        "vram_available_gb": vram_available_mib / 1024.0,
        "ram_safe_mib": max(0, ram_available_mib - RAM_HEADROOM_MIB),
        "vram_safe_mib": max(0, vram_available_mib - VRAM_HEADROOM_MIB),
        "ram_headroom_mib": RAM_HEADROOM_MIB,
        "vram_headroom_mib": VRAM_HEADROOM_MIB,
        "cpu_threads": cpu_threads,
        "gds_threads_max": min(32, cpu_threads),
    }


def _validate_resource_settings(values: dict[str, Any], system: Any) -> tuple[bool, str]:
    """Reject launch profiles that exceed currently safe host capacities."""
    state = _resource_state(system)
    try:
        pinned_mib = max(0, int(values.get("pinned_cache", values.get("l2_cache", 0)) or 0))
        prompt_mib = max(0, int(values.get("cache_ram", 0) or 0))
        vram_mib = max(0, int(values.get("vram_cache", values.get("l1_cache", 0)) or 0))
    except (TypeError, ValueError):
        return False, "Memory allocations must be valid numbers."
    requested_ram = pinned_mib + prompt_mib
    if system.mem_total_gb and requested_ram > state["ram_safe_mib"]:
        return False, (
            f"RAM allocation is {requested_ram / 1024:.1f} GiB, but only "
            f"{state['ram_safe_mib'] / 1024:.1f} GiB is safely available. "
            "Siphon always preserves at least 2 GiB of free RAM."
        )
    if system.gpu_mib_total and vram_mib > state["vram_safe_mib"]:
        return False, (
            f"VRAM cache is {vram_mib / 1024:.1f} GiB, but only "
            f"{state['vram_safe_mib'] / 1024:.1f} GiB is safely available "
            "after the runtime reserve."
        )
    try:
        gds_threads = int(values.get("gds_threads", 1) or 1)
    except (TypeError, ValueError):
        return False, "GDS worker count must be a valid number."
    if gds_threads > state["gds_threads_max"]:
        return False, (
            f"GDS workers are set to {gds_threads}, but this host safely supports "
            f"at most {state['gds_threads_max']} concurrent workers."
        )
    return True, ""


def _recommended_settings(config: dict[str, Any], system: Any) -> dict[str, int]:
    """Produce conservative, model-aware cache budgets for this machine."""
    state = _resource_state(system)
    ram_safe = int(state["ram_safe_mib"])
    vram_safe = int(state["vram_safe_mib"])
    prompt_target = min(int(config.get("cache_ram", 0) or 0), ram_safe // 3)
    prompt_target = max(0, prompt_target // 512 * 512)
    pinned_target = min(int(config.get("l2_cache", 0) or 0), ram_safe - prompt_target)
    pinned_target = max(0, pinned_target // 1024 * 1024)
    gpu_layers_target, vram_target = _recommended_gpu_budget(vram_safe)
    requested_context = int(config.get("context", 0) or 0)
    return {
        "gpu_layers": gpu_layers_target,
        "vram_cache": vram_target,
        "pinned_cache": pinned_target,
        "cache_ram": prompt_target,
        "context": min(requested_context, _recommended_context(vram_safe)),
        "expert_k": int(config.get("expert_k", 0) or 0),
        "gds_threads": min(
            max(1, int(config.get("gds_threads", 1) or 1)),
            int(state["gds_threads_max"]),
        ),
    }


def _recommended_gpu_budget(vram_safe_mib: int) -> tuple[int, int]:
    """Reserve a conservative DirectStorage expert pool.

    DirectStorage expert tensors are GPU addresses and every MoE layer that
    consumes them must therefore remain GPU-routed.  Memory is controlled with
    context/KV size and the bounded expert pool, not partial CPU layer offload.
    """
    usable_mib = max(0, int(vram_safe_mib) - 512)
    gpu_layers = 999
    expert_pool_mib = min(1024, usable_mib)
    expert_pool_mib = expert_pool_mib // 128 * 128
    return gpu_layers, expert_pool_mib


def _recommended_context(vram_safe_mib: int) -> int:
    """Choose a conservative KV budget tier for DirectStorage MoE models."""
    if vram_safe_mib <= 6 * 1024:
        return 4096
    if vram_safe_mib <= 10 * 1024:
        return 8192
    if vram_safe_mib <= 16 * 1024:
        return 16384
    return 32768


def _prototype_path() -> Path:
    """Return the approved UI file, with a useful override for packaging."""
    configured = paths.repo_root() / "siphon-desktop-dark-opendesign.html"
    if configured.is_file():
        return configured

    fallback = paths.studio_dir() / "siphon-desktop-dark-opendesign.html"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        "Siphon Studio UI not found. Expected "
        f"{configured}"
    )


def _cpu_label() -> str:
    """Return the host CPU model without making it a required dependency."""
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "Host CPU"


def _file_size(path: Path) -> int:
    """Return a stable log cursor; an absent log starts at byte zero."""
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _startup_log_error(path: Path, start_offset: int) -> str | None:
    """Return a fatal runtime error emitted after this launch began."""
    fatal_markers = (
        "failed to allocate CUDA VRAM pool",
        "failed to stream experts",
        "failed to compute graph",
        "CUDA error",
        "illegal memory access",
        "Segmentation fault",
    )
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(max(0, start_offset))
            text = handle.read()
    except OSError:
        return None
    return next((marker for marker in fatal_markers if marker in text), None)


def _probe_inference(profile: Profile) -> str | None:
    """Exercise one real prefill/decode before declaring a model ready."""
    payload = json.dumps({
        "prompt": "Siphon capability probe",
        "n_predict": 1,
        "temperature": 0,
        "cache_prompt": False,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"{profile.base_url}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120.0) as response:
            result = json.loads(response.read().decode("utf-8", "replace"))
        if not isinstance(result, dict) or "content" not in result:
            return "inference probe returned an invalid response"
    except Exception as exc:
        return f"inference probe failed: {exc}"
    return None


def _counter_delta(
    current: dict[str, float], baseline: dict[str, float], key: str
) -> float:
    """Return a safe per-request delta for a Prometheus lifetime counter."""
    try:
        return max(0.0, float(current.get(key, 0.0)) - float(baseline.get(key, 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _dstorage_request_metrics(
    current: dict[str, float], baseline: dict[str, float]
) -> dict[str, Any]:
    """Translate live Siphon /metrics counters into one-request dashboard data.

    Older llama-server binaries do not expose these names.  In that case this
    deliberately returns no fields, so the UI says unavailable instead of
    presenting made-up movement data.
    """
    if not baseline or float(current.get("dstorage_active", 0.0) or 0.0) <= 0:
        return {}

    prefill_calls = _counter_delta(current, baseline, "dstorage_prefill_calls")
    decode_calls = _counter_delta(current, baseline, "dstorage_decode_calls")
    decode_hits = _counter_delta(current, baseline, "dstorage_decode_hits")
    decode_misses = _counter_delta(current, baseline, "dstorage_decode_misses")
    cold_experts = _counter_delta(current, baseline, "dstorage_decode_cold_experts")
    pinned_experts = _counter_delta(current, baseline, "dstorage_decode_pinned_experts")
    streamed_bytes = _counter_delta(current, baseline, "dstorage_streamed_file_bytes")
    pinned_bytes = _counter_delta(current, baseline, "dstorage_pinned_transfer_bytes")
    pinned_us = _counter_delta(current, baseline, "dstorage_pinned_transfer_us")
    stream_us = _counter_delta(current, baseline, "dstorage_stream_total_us")
    draft_total = _counter_delta(current, baseline, "server_draft_total")
    draft_accepted = _counter_delta(current, baseline, "server_draft_accepted")
    has_request_data = any((prefill_calls, decode_calls, decode_hits, decode_misses))
    if not has_request_data:
        return {}

    l1_accesses = decode_hits + decode_misses
    l2_accesses = pinned_experts + cold_experts
    gib = float(1024 ** 3)
    result: dict[str, Any] = {
        # Request movement, not cache capacity. Occupied L1/L2 slot counts are
        # normally constant once caches are warm and therefore looked
        # hardcoded even while entirely different experts moved through them.
        "l1_resident": int(decode_hits),
        "l2_resident": int(pinned_experts),
        "cold_misses": int(cold_experts),
        "l1_hit_pct": (decode_hits / l1_accesses * 100.0) if l1_accesses else None,
        "l2_hit_pct": (pinned_experts / l2_accesses * 100.0) if l2_accesses else None,
        "streamed_gb": streamed_bytes / gib if streamed_bytes else 0.0,
        "decode_calls": int(decode_calls),
        "prefill_calls": int(prefill_calls),
        "draft_accept_pct": (
            draft_accepted / draft_total * 100.0 if draft_total else None
        ),
    }
    if stream_us > 0 and streamed_bytes > 0:
        result["gds_gbs"] = streamed_bytes / gib / (stream_us / 1_000_000.0)
    if pinned_us > 0 and pinned_bytes > 0:
        result["ram_vram_gbs"] = pinned_bytes / gib / (pinned_us / 1_000_000.0)
    return result


def _live_engine_metrics(
    current: dict[str, float], previous: dict[str, float]
) -> dict[str, float]:
    """Turn cumulative engine counters into request-scoped live rates.

    ``previous`` is the snapshot taken immediately before the request.  Using
    deltas from that fixed baseline prevents an earlier request from leaking
    into this one.  Prefer Siphon's lock-protected live snapshot; unlike the
    Prometheus task result, it includes counters from an active slot.
    """
    live_keys = (
        "server_prompt_tokens_total",
        "server_prompt_time_us_total",
        "server_predicted_tokens_total",
        "server_generation_time_us_total",
    )
    prometheus_keys = (
        "prompt_tokens_total",
        "prompt_seconds_total",
        "tokens_predicted_total",
        "tokens_predicted_seconds_total",
    )
    if all(key in current and key in previous for key in live_keys):
        keys = live_keys
        # The historical JSON field names say ``_us_``, but server_metrics
        # stores these totals in milliseconds (slot timings divide the native
        # microsecond clock by 1e3 before accumulation).
        time_scale = 1_000.0
    elif all(key in current and key in previous for key in prometheus_keys):
        keys = prometheus_keys
        time_scale = 1.0
    else:
        return {}
    prompt_tokens = max(0.0, float(current[keys[0]]) - float(previous[keys[0]]))
    prompt_time = max(0.0, float(current[keys[1]]) - float(previous[keys[1]]))
    predicted_tokens = max(0.0, float(current[keys[2]]) - float(previous[keys[2]]))
    generation_time = max(0.0, float(current[keys[3]]) - float(previous[keys[3]]))
    result: dict[str, float] = {}
    if prompt_tokens > 0 and prompt_time > 0:
        result["prefill_tps"] = prompt_tokens / (prompt_time / time_scale)
        result["n_prompt"] = prompt_tokens
        result["prefill_sample"] = prompt_tokens
    if predicted_tokens > 0 and generation_time > 0:
        result["decode_tps"] = predicted_tokens / (generation_time / time_scale)
        result["generated_tokens"] = predicted_tokens
        result["decode_sample"] = predicted_tokens
    return result


def _live_transfer_rates(
    current: dict[str, float], previous: dict[str, float]
) -> dict[str, float]:
    """Return interval transfer rates and advance the sampling cursor."""
    byte_key = "dstorage_pinned_transfer_bytes"
    time_key = "dstorage_pinned_transfer_us"
    if byte_key not in current or time_key not in current:
        return {}
    current_bytes = float(current.get(byte_key, 0.0) or 0.0)
    current_us = float(current.get(time_key, 0.0) or 0.0)
    if byte_key not in previous or time_key not in previous:
        previous[byte_key] = current_bytes
        previous[time_key] = current_us
        return {}
    byte_delta = max(0.0, current_bytes - float(previous[byte_key]))
    time_delta = max(0.0, current_us - float(previous[time_key]))
    previous[byte_key] = current_bytes
    previous[time_key] = current_us
    if time_delta <= 0:
        return {"ram_vram_gbs": 0.0}
    return {
        "ram_vram_gbs": byte_delta / float(1024 ** 3) / (time_delta / 1_000_000.0)
    }


def _live_cache_rates(
    current: dict[str, float], previous: dict[str, float]
) -> dict[str, float]:
    """Calculate hit rates for only the work completed since the last poll."""
    keys = (
        "dstorage_decode_hits",
        "dstorage_decode_misses",
        "dstorage_decode_pinned_experts",
        "dstorage_decode_cold_experts",
    )
    if not all(key in current for key in keys):
        return {}
    sample = {key: float(current.get(key, 0.0) or 0.0) for key in keys}
    if not all(key in previous for key in keys):
        previous.update(sample)
        return {}
    delta = {
        key: max(0.0, sample[key] - float(previous.get(key, sample[key])))
        for key in keys
    }
    previous.update(sample)
    hits = delta["dstorage_decode_hits"]
    misses = delta["dstorage_decode_misses"]
    pinned = delta["dstorage_decode_pinned_experts"]
    cold = delta["dstorage_decode_cold_experts"]
    result: dict[str, float] = {}
    if hits + misses > 0:
        result["l1_hit_pct"] = hits / (hits + misses) * 100.0
    if pinned + cold > 0:
        result["l2_hit_pct"] = pinned / (pinned + cold) * 100.0
    return result


def _model_field(reader: Any, key: str, default: Any = None) -> Any:
    """Read a GGUF metadata field without making metadata a hard dependency."""
    field = getattr(reader, "fields", {}).get(key)
    if field is None:
        return default
    try:
        return field.contents()
    except Exception:
        return default


def _profile_config(profile: Profile) -> dict[str, Any]:
    """Return launch settings without touching the model weights.

    This intentionally stays separate from GGUF inspection so the catalogue
    can show the selected profile's real defaults immediately. Reading a
    large GGUF header/tensor table is done asynchronously afterward.
    """
    expert_override = profile.env.get("LLAMA_MOE_EXPERT_USED_OVERRIDE", "")
    expert_mode = profile.env.get("MOE_EXPERT_COUNT_MODE", "default")
    if expert_mode == "native" and not expert_override:
        expert_k = 0
    else:
        try:
            expert_k = int(expert_override or (
                "6" if profile.script_rel.endswith("run-qwen3-coder-next.sh") else "4"
            ))
        except ValueError:
            expert_k = 0
    return {
        "context": int(profile.env.get("CTX", "0") or 0),
        "batch": int(profile.env.get("BATCH", "0") or 0),
        "microbatch": int(profile.env.get("UBATCH", "0") or 0),
        "gpu_layers": int(profile.env.get("GPU_LAYERS", "999") or 999),
        "cache_type": profile.env.get("CACHE_TYPE", "tq3_0"),
        "flash_attention": profile.env.get("FLASH_ATTENTION", "on") not in {"", "0", "off", "false", "False"},
        "rope_scaling": profile.env.get("ROPE_SCALING", "auto").lower(),
        "l1_cache": int(profile.env.get("MOE_GPU_CACHE_MIB", "0") or 0),
        "l2_cache": int(profile.env.get("MOE_PINNED_CACHE_MIB", "0") or 0),
        "cache_ram": int(profile.env.get("CACHE_RAM_MIB", "0") or 0),
        "gds_threads": int(profile.env.get("GDS_READ_THREADS", "0") or 0),
        "spec_type": profile.env.get("SPEC_TYPE", "none"),
        "draft_model": profile.env.get("MODEL_DRAFT", ""),
        "draft_tokens": int(profile.env.get("SPEC_DRAFT_N_MAX", "0") or 0),
        "draft_probability": float(profile.env.get("SPEC_DRAFT_P_MIN", "0") or 0),
        "temperature": float(profile.env.get("SIPHON_TEMPERATURE", "0.70") or 0.70),
        "top_k": int(profile.env.get("SIPHON_TOP_K", "40") or 40),
        "top_p": float(profile.env.get("SIPHON_TOP_P", "0.95") or 0.95),
        "expert_k": expert_k,
        "reasoning_budget": int(profile.env.get("REASONING_BUDGET", "0") or 0),
        "promotion_hits": int(
            profile.env.get("LLAMA_DSTORAGE_EVICT_TO_RAM_MIN_DECODE_HITS", "2") or 2
        ),
        "evict_to_ram": profile.env.get("LLAMA_DSTORAGE_EVICT_TO_RAM", "0") not in {"", "0", "false", "False"},
        "prompt_cache": profile.env.get(
            "CACHE_PROMPT", "1" if profile.script_rel.endswith("run-qwen36-moe.sh") else "0"
        ) not in {"", "0", "false", "False"},
    }


def _metadata_cache_path() -> Path:
    return paths.state_dir() / "model-metadata.json"


def _read_metadata_cache() -> dict[str, Any]:
    try:
        value = json.loads(_metadata_cache_path().read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_metadata_cache(payload: dict[str, Any]) -> None:
    model_path = payload.get("path")
    if not model_path:
        return
    try:
        path = Path(str(model_path)).resolve()
        stat = path.stat()
        cache = _read_metadata_cache()
        cache[str(path)] = {
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "data": {
                key: payload[key]
                for key in (
                    "name", "architecture", "layers", "native_context",
                    "expert_count", "expert_used_count", "artifact_gib",
                    "dense_gib", "expert_gib",
                )
                if key in payload
            },
        }
        cache_path = _metadata_cache_path()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
    except (OSError, TypeError, ValueError):
        # Metadata caching is an optimization. A read-only state directory or
        # an unusual filesystem must never prevent model inspection.
        return


def _cached_model_info(profile: Profile, model_key: str) -> dict[str, Any] | None:
    model_path = Path(profile.resolve("MODEL"))
    if not model_path.is_file():
        return None
    try:
        path = model_path.resolve()
        stat = path.stat()
        entry = _read_metadata_cache().get(str(path), {})
        if (entry.get("size") != stat.st_size or
                entry.get("mtime_ns") != stat.st_mtime_ns or
                not isinstance(entry.get("data"), dict)):
            return None
        return {
            "type": "model_info",
            "model_key": model_key,
            "available": True,
            "profile": profile.name,
            "path": str(path),
            "config": _profile_config(profile),
            **entry["data"],
        }
    except (OSError, TypeError, ValueError):
        return None


def _model_info(profile: Profile, model_key: str) -> dict[str, Any]:
    """Build catalogue data from the selected profile and GGUF header.

    GGUFReader memory-maps the model and reads its header/tensor descriptors;
    it does not load the model weights into RAM. If a model is unavailable or
    the optional reader cannot be imported, the UI receives an explicit
    unavailable state instead of fabricated metadata.
    """
    model_path = Path(profile.resolve("MODEL"))
    info: dict[str, Any] = {
        "type": "model_info",
        "model_key": model_key,
        "available": model_path.is_file(),
        "profile": profile.name,
        "path": str(model_path),
        "config": _profile_config(profile),
    }
    if not model_path.is_file():
        return info

    try:
        from gguf.gguf_reader import GGUFReader

        reader = GGUFReader(model_path)
        architecture = str(_model_field(reader, "general.architecture", ""))
        prefix = architecture
        layers = _model_field(reader, f"{prefix}.block_count", 0)
        context = _model_field(reader, f"{prefix}.context_length", 0)
        expert_count = _model_field(reader, f"{prefix}.expert_count", 0)
        expert_used = _model_field(reader, f"{prefix}.expert_used_count", 0)
        total_tensor_bytes = sum(int(t.n_bytes) for t in reader.tensors)
        expert_tensor_bytes = sum(
            int(t.n_bytes) for t in reader.tensors if "exps" in t.name
        )
        artifact_gib = model_path.stat().st_size / (1024 ** 3)
        info.update({
            "name": MODEL_DISPLAY_NAMES.get(
                model_key,
                str(_model_field(reader, "general.name", model_path.stem)),
            ),
            "architecture": architecture,
            "layers": int(layers or 0),
            "native_context": int(context or 0),
            "expert_count": int(expert_count or 0),
            "expert_used_count": int(expert_used or 0),
            "artifact_gib": round(artifact_gib, 2),
            "dense_gib": round(max(total_tensor_bytes - expert_tensor_bytes, 0) / (1024 ** 3), 2),
            "expert_gib": round(expert_tensor_bytes / (1024 ** 3), 2),
        })
    except Exception as exc:
        info["metadata_error"] = str(exc)
    return info


class DesktopBridge:
    """Translate small JSON messages from the HTML surface into core actions."""

    def __init__(self, window: Gtk.Window, webview: WebKit.WebView) -> None:
        self.window = window
        self.webview = webview
        self.profiles: dict[str, Profile] = discover()
        self.profile_by_model_key = _profile_map(self.profiles)
        self.model_key = "qwen35"
        self._poll_busy = False
        self._poll_paused = False
        self._telemetry_epoch = 0
        self._closed = False
        self._chat_stop_events: dict[str, threading.Event] = {}
        self._chat_lock = threading.Lock()
        self._sampling: dict[str, dict[str, float]] = {}
        self._request_metrics: dict[str, dict[str, Any]] = {}
        # DirectStorage exposes lifetime counters through /metrics.  Each chat
        # request records a baseline so the dashboard can show its own real
        # movement, rather than the totals accumulated by earlier requests.
        self._dstorage_baselines: dict[str, dict[str, float]] = {}
        self._active_generations: dict[str, bool] = {}
        self._live_timing_samples: dict[str, dict[str, float]] = {}
        self._live_stream_metrics: dict[str, dict[str, float]] = {}
        self._dstorage_rate_samples: dict[str, dict[str, float]] = {}
        # Server logs are append-only.  Start every desktop session at the
        # current end so old timing summaries never appear as a fresh request.
        self._log_offsets = {
            profile.name: _file_size(profile.log_file)
            for profile in self.profiles.values()
        }

    def register(self, manager: WebKit.UserContentManager) -> None:
        manager.register_script_message_handler("siphon")
        manager.connect("script-message-received::siphon", self._message_received)

    def close(self) -> None:
        self._closed = True

    def _message_received(
        self,
        _manager: WebKit.UserContentManager,
        javascript_result: Any,
    ) -> None:
        try:
            value = javascript_result
            if hasattr(value, "get_js_value"):
                value = value.get_js_value()
            raw = value.to_string()
            message = json.loads(raw)
        except (AttributeError, TypeError, ValueError) as exc:
            self._emit_event({"type": "bridge_error", "message": str(exc)})
            return

        if not isinstance(message, dict):
            return
        action = str(message.get("action", ""))
        if action in {"page_ready", "select_model"}:
            next_model_key = str(message.get("model", self.model_key))
            if next_model_key != self.model_key or action == "page_ready":
                self._telemetry_epoch += 1
            self.model_key = next_model_key
            self._schedule_model_info(self.model_key)
            if action == "page_ready":
                self._schedule_catalogue()
                GLib.idle_add(self._start_polling)
            return
        if action == "inspect_model":
            self._schedule_model_info(str(message.get("model", "")))
            return
        if action == "add_model":
            GLib.idle_add(self._choose_model_file)
            return
        if action in {"start_model", "stop_model"}:
            self.model_key = str(message.get("model", self.model_key))
            # Drop any poll that started before this lifecycle transition.
            self._telemetry_epoch += 1
            self._run_model_action(action == "start_model", self.model_key)
            return
        if action == "apply_settings":
            threading.Thread(
                target=self._apply_settings,
                args=(
                    str(message.get("model", self.model_key)),
                    message.get("settings", {}),
                ),
                name="siphon-settings-save",
                daemon=True,
            ).start()
            return
        if action == "chat_send":
            self._start_chat(
                str(message.get("model", self.model_key)),
                str(message.get("prompt", "")),
                message.get("history", []),
                str(message.get("generation_id", uuid.uuid4().hex)),
            )
            return
        if action == "chat_stop":
            self._stop_chat(str(message.get("generation_id", "")))
            return
        if action == "refresh_telemetry":
            GLib.idle_add(self._poll)
            return
        if action == "pause_telemetry":
            self._poll_paused = bool(message.get("paused", False))
            return
        if action == "window_action":
            self._window_action(str(message.get("name", "")))

    def _window_action(self, name: str) -> None:
        if name == "minimize":
            self.window.minimize()
        elif name == "maximize":
            if self.window.is_maximized():
                self.window.unmaximize()
            else:
                self.window.maximize()
        elif name == "close":
            self.window.close()

    def _profile_for(self, model_key: str) -> Profile | None:
        profile_name = self.profile_by_model_key.get(model_key)
        if not profile_name:
            return None
        return self.profiles.get(profile_name)

    def _schedule_catalogue(self) -> None:
        entries = [
            {"model_key": key, "profile": name}
            for key, name in self.profile_by_model_key.items()
            if name in self.profiles and self.profiles[name].resolve("MODEL")
        ]
        self._emit_async({"type": "catalogue", "models": entries})

    def _choose_model_file(self) -> bool:
        # Gtk.FileChooserNative on some Ubuntu GTK 4 builds corrupts its
        # internal file-system model when opened from a WebKit-hosted app and
        # ends in repeated GTK_IS_FILE_SYSTEM_MODEL assertions followed by a
        # segfault. Gtk.FileDialog is the supported GTK 4 asynchronous API and
        # keeps the chooser owned by the application for the whole operation.
        dialog = Gtk.FileDialog.new()
        dialog.set_title("Add a GGUF model")

        def responded(file_dialog: Gtk.FileDialog, result: Any, _data: Any) -> None:
            try:
                selected = file_dialog.open_finish(result)
                selected_path = selected.get_path() if selected else None
                if selected_path:
                    threading.Thread(
                        target=self._import_model,
                        args=(Path(selected_path),),
                        name="siphon-model-import",
                        daemon=True,
                    ).start()
            except GLib.Error as exc:
                # Cancellation is normal; other chooser errors should be
                # reported in the UI rather than terminating the desktop host.
                if "cancel" not in str(exc).lower():
                    self._emit_async({"type": "model_imported", "ok": False,
                                      "message": f"Could not open model chooser: {exc}"})

        dialog.open(self.window, None, responded, None)
        return GLib.SOURCE_REMOVE

    def _import_model(self, model_path: Path) -> None:
        if model_path.suffix.lower() != ".gguf" or not model_path.is_file():
            self._emit_async({"type": "model_imported", "ok": False,
                              "message": "Select a readable .gguf model file."})
            return
        try:
            canonical_path = model_path.resolve()
            for existing_key, profile_name in self.profile_by_model_key.items():
                existing = self.profiles.get(profile_name)
                if existing and Path(existing.resolve("MODEL")).resolve() == canonical_path:
                    self._emit_async({"type": "model_imported", "ok": True,
                                      "model_key": existing_key,
                                      "message": "That model is already in the catalogue."})
                    return
            from gguf.gguf_reader import GGUFReader
            reader = GGUFReader(model_path)
            architecture = str(_model_field(reader, "general.architecture", ""))
            if not architecture:
                raise ValueError("The GGUF does not declare general.architecture")
            native_context = int(_model_field(reader, f"{architecture}.context_length", 32768) or 32768)
            expert_count = int(_model_field(reader, f"{architecture}.expert_count", 0) or 0)
            expert_used = int(_model_field(reader, f"{architecture}.expert_used_count", 0) or 0)
            display_name = str(_model_field(reader, "general.name", model_path.stem))
            if expert_count <= 0 or expert_used <= 0:
                raise ValueError(
                    f"{display_name} does not expose valid routed-MoE expert_count "
                    "and expert_used_count metadata"
                )
            system = fetch_sys_stats()
            resources = _resource_state(system)
            profile_name = f"imported-{hashlib.sha256(str(canonical_path).encode()).hexdigest()[:12]}"
            used_ports = {p.port for p in self.profiles.values()}
            port = next((candidate for candidate in range(8090, 8190) if candidate not in used_ports), 8190)
            ram_safe, vram_safe = int(resources["ram_safe_mib"]), int(resources["vram_safe_mib"])
            prompt_mib = min(2048, max(0, ram_safe // 4))
            pinned_mib = min(8192, max(0, ram_safe - prompt_mib))
            gpu_layers, vram_mib = _recommended_gpu_budget(vram_safe)
            profile = Profile.create_app(profile_name, {
                "MODEL": str(canonical_path), "EXPERTS": "", "ALIAS": profile_name,
                "HOST": "127.0.0.1", "PORT": str(port),
                "CTX": str(min(native_context, _recommended_context(vram_safe))),
                "BATCH": "512", "UBATCH": "128", "GPU_LAYERS": str(gpu_layers),
                "RUNTIME_MODE": "qwen-dstorage",
                "DSTORAGE_CAPABILITY": "auto-probe",
                "MOE_EXPERT_COUNT_MODE": "native",
                # Gemma4's mixed expert quantisation and current CUDA kernels
                # require an unquantised KV cache.  Flash attention remains an
                # explicit opt-in until the imported runtime has validated it.
                "CACHE_TYPE": "f16", "FLASH_ATTENTION": "off", "ROPE_SCALING": "auto",
                "MOE_GPU_CACHE_MIB": str(vram_mib), "MOE_PINNED_CACHE_MIB": str(pinned_mib),
                "CACHE_RAM_MIB": str(prompt_mib), "GDS_READ_THREADS": str(resources["gds_threads_max"]),
                # Speculation is opt-in for imported models. A valid MoE GGUF
                # need not contain a native MTP/NextN head, and forcing the
                # bundled Qwen setting would make such a model unlaunchable.
                "SPEC_TYPE": "none", "SPEC_DRAFT_N_MAX": "0", "SPEC_DRAFT_P_MIN": "0",
                "CHECKPOINT_MIN_STEP": "0", "CACHE_PROMPT": "1", "CACHE_REUSE": "0",
                "METRICS": "1", "N_PARALLEL": "1", "REASONING": "off",
                "REASONING_FORMAT": "deepseek", "REASONING_BUDGET": "0",
                "SIPHON_TEMPERATURE": "0.70", "SIPHON_TOP_K": "40", "SIPHON_TOP_P": "0.95",
            }).save()
            self.profiles[profile.name] = profile
            model_key = _model_key_for_profile(profile)
            self.profile_by_model_key[model_key] = profile.name
            self._log_offsets[profile.name] = _file_size(profile.log_file)
            self._schedule_catalogue()
            self._emit_async({"type": "model_imported", "ok": True, "model_key": model_key,
                              "message": f"Added {display_name}"})
            self._schedule_model_info(model_key)
        except Exception as exc:
            self._emit_async({"type": "model_imported", "ok": False,
                              "message": f"Could not add model: {exc}"})

    def _schedule_model_info(self, model_key: str) -> None:
        profile = self._profile_for(model_key)
        if profile is None:
            self._emit_async({"type": "model_info", "model_key": model_key, "available": False})
            return

        # Apply the profile defaults immediately. GGUF tensor inspection can
        # take noticeable time for multi-gigabyte models, but settings such as
        # CTX=86016 must not wait for that metadata pass.
        model_path = Path(profile.resolve("MODEL"))
        self._emit_async({
            "type": "model_info",
            "phase": "profile",
            "model_key": model_key,
            "available": model_path.is_file(),
            "profile": profile.name,
            "path": str(model_path),
            "config": _profile_config(profile),
        })

        def work() -> None:
            try:
                # The profile can be edited while this metadata scan is in
                # flight. Resolve it again before building the final payload
                # so an older scan cannot re-emit stale launch settings after
                # the user has saved new values.
                current_profile = self._profile_for(model_key) or profile
                payload = _cached_model_info(current_profile, model_key)
                if payload is None:
                    payload = _model_info(current_profile, model_key)
                    if payload.get("available") and not payload.get("metadata_error"):
                        _write_metadata_cache(payload)
                model_paths = {
                    Path(item.resolve("MODEL"))
                    for item in self.profiles.values()
                    if item.resolve("MODEL")
                }
                sizes = [
                    path.stat().st_size for path in model_paths if path.is_file()
                ]
                payload["library_storage_gib"] = round(
                    sum(sizes) / (1024 ** 3), 2
                )
                payload["library_count"] = len(sizes)
                # Re-read after the expensive scan too: settings may have
                # been saved while GGUF metadata was being inspected.
                latest_profile = self._profile_for(model_key) or current_profile
                payload["config"] = _profile_config(latest_profile)
                system = fetch_sys_stats()
                payload["resources"] = _resource_state(system)
                payload["recommended"] = _recommended_settings(
                    payload["config"], system
                )
            except Exception as exc:  # metadata must never block the desktop shell
                payload = {"type": "model_info", "model_key": model_key,
                           "available": False, "metadata_error": str(exc)}
            GLib.idle_add(self._finish_model_info, payload)

        threading.Thread(target=work, name="siphon-model-metadata", daemon=True).start()

    def _finish_model_info(self, payload: dict[str, Any]) -> bool:
        self._emit_event(payload)
        return GLib.SOURCE_REMOVE

    def _apply_settings(self, model_key: str, values: Any) -> None:
        profile = self._profile_for(model_key)
        if profile is None or not isinstance(values, dict):
            self._emit_async({
                "type": "settings_applied",
                "ok": False,
                "message": f"No launcher profile is configured for {model_key}.",
            })
            return

        resources_ok, resource_message = _validate_resource_settings(
            values, fetch_sys_stats()
        )
        if not resources_ok:
            self._emit_async({
                "type": "settings_applied",
                "ok": False,
                "message": resource_message,
            })
            return

        def int_setting(name: str, env_key: str, minimum: int = 0) -> None:
            raw = values.get(name)
            try:
                value = max(minimum, int(raw))
            except (TypeError, ValueError):
                return
            profile.env[env_key] = str(value)

        int_setting("context", "CTX", 1024)
        int_setting("batch", "BATCH", 1)
        int_setting("microbatch", "UBATCH", 1)
        int_setting("gpu_layers", "GPU_LAYERS", 0)
        int_setting("vram_cache", "MOE_GPU_CACHE_MIB", 0)
        int_setting("pinned_cache", "MOE_PINNED_CACHE_MIB", 0)
        int_setting("cache_ram", "CACHE_RAM_MIB", 0)
        int_setting("gds_threads", "GDS_READ_THREADS", 1)
        int_setting("draft_tokens", "SPEC_DRAFT_N_MAX", 0)
        int_setting("checkpoints", "LLAMA_ARG_CTX_CHECKPOINTS", 0)
        int_setting("promotion_hits", "LLAMA_DSTORAGE_EVICT_TO_RAM_MIN_DECODE_HITS", 1)
        int_setting("expert_k", "LLAMA_MOE_EXPERT_USED_OVERRIDE", 1)
        int_setting("reasoning_budget", "REASONING_BUDGET", -1)

        cache_type = str(values.get("cache_type", "")).strip().lower()
        if cache_type in {"f32", "f16", "bf16", "q8_0", "q6_k", "q5_k", "q4_k", "q3_k", "tq1_0", "tq2_0", "tq3_0"}:
            profile.env["CACHE_TYPE"] = cache_type
        rope_scaling = str(values.get("rope_scaling", "auto")).strip().lower()
        if rope_scaling in {"auto", "none", "linear", "yarn"}:
            if rope_scaling == "auto":
                profile.env.pop("ROPE_SCALING", None)
            else:
                profile.env["ROPE_SCALING"] = rope_scaling
        if "flash_attention" in values:
            profile.env["FLASH_ATTENTION"] = "on" if bool(values["flash_attention"]) else "off"

        spec_type = str(values.get("spec_type", "")).strip()
        if spec_type in {"draft-mtp", "draft-simple", "draft-dflash", "none", "off"}:
            profile.env["SPEC_TYPE"] = spec_type
        draft_model = str(values.get("draft_model", "")).strip()
        if draft_model in {"Native MTP / NextN", "Disabled", ""}:
            profile.env.pop("MODEL_DRAFT", None)
            if draft_model == "Disabled":
                profile.env["SPEC_TYPE"] = "none"
        elif draft_model == "Qwen3 0.6B Q8":
            candidate = profile.env.get("MODEL_DRAFT", "")
            if not candidate:
                candidate = self.profiles.get("qwen3-coder-next-draft", profile).env.get("MODEL_DRAFT", "")
            candidate_path = Path(candidate)
            if not candidate_path.is_absolute():
                candidate_path = paths.repo_root() / candidate_path
            if not candidate or not candidate_path.is_file():
                self._emit_async({
                    "type": "settings_applied",
                    "ok": False,
                    "message": "The selected draft model is not available on this host.",
                })
                return
            profile.env["MODEL_DRAFT"] = candidate
        for name, env_key in (("draft_probability", "SPEC_DRAFT_P_MIN"),
                              ("temperature", "SIPHON_TEMPERATURE"),
                              ("top_p", "SIPHON_TOP_P")):
            raw = values.get(name)
            try:
                profile.env[env_key] = str(float(raw))
            except (TypeError, ValueError):
                pass
        try:
            profile.env["SIPHON_TOP_K"] = str(max(0, int(values.get("top_k"))))
        except (TypeError, ValueError):
            pass

        if "evict_to_ram" in values:
            profile.env["LLAMA_DSTORAGE_EVICT_TO_RAM"] = (
                "1" if bool(values["evict_to_ram"]) else "0"
            )
        if "prompt_cache" in values:
            profile.env["CACHE_PROMPT"] = (
                "1" if bool(values["prompt_cache"]) else "0"
            )

        try:
            if profile.is_bundled:
                profile = Profile.create_app(
                    profile.name, profile.env, profile.script_rel
                )
            # Persist both the editable profile and the materialized launch
            # environment. Previously only the latter was updated when an
            # app profile already existed, so a restart restored old values.
            profile.save()
            self.profiles[profile.name] = profile
            env_path = profile.materialize_env()
            running = runner.status(profile).running
            self._sampling[model_key] = {
                key: float(values[key])
                for key in ("temperature", "top_p", "top_k")
                if values.get(key) is not None
            }
            self._emit_async({
                "type": "settings_applied",
                "ok": True,
                "model_key": model_key,
                "restart_required": running,
                "path": str(env_path),
                "message": (
                    "Settings saved. Restart the model to load them."
                    if running else "Settings saved to the launch profile."
                ),
            })
            self._schedule_model_info(model_key)
        except (OSError, ValueError) as exc:
            self._emit_async({
                "type": "settings_applied",
                "ok": False,
                "message": f"Could not save settings: {exc}",
            })

    def _emit_async(self, payload: dict[str, Any]) -> None:
        GLib.idle_add(self._deliver_async, payload)

    def _deliver_async(self, payload: dict[str, Any]) -> bool:
        self._emit_event(payload)
        return GLib.SOURCE_REMOVE

    def _stop_chat(self, generation_id: str) -> None:
        with self._chat_lock:
            stop_event = self._chat_stop_events.get(generation_id)
        if stop_event is not None:
            stop_event.set()

    def _start_chat(
        self,
        model_key: str,
        prompt: str,
        history: Any,
        generation_id: str,
    ) -> None:
        profile = self._profile_for(model_key)
        if profile is None:
            self._emit_async({"type": "chat_error", "generation_id": generation_id,
                              "message": f"No launcher profile is configured for {model_key}."})
            return
        prompt = prompt.strip()
        if not prompt:
            return
        stop_event = threading.Event()
        with self._chat_lock:
            self._chat_stop_events[generation_id] = stop_event

        clean_history: list[dict[str, str]] = []
        if isinstance(history, list):
            for item in history:
                if not isinstance(item, dict):
                    continue
                role = str(item.get("role", ""))
                content = str(item.get("content", "")).strip()
                if role in {"user", "assistant"} and content:
                    clean_history.append({"role": role, "content": content})
        clean_history.append({"role": "user", "content": prompt})

        def work() -> None:
            try:
                self._stream_chat(
                    profile, model_key, generation_id, stop_event, clean_history
                )
            finally:
                with self._chat_lock:
                    self._chat_stop_events.pop(generation_id, None)

        threading.Thread(target=work, name="siphon-chat", daemon=True).start()

    def _stream_chat(
        self,
        profile: Profile,
        model_key: str,
        generation_id: str,
        stop_event: threading.Event,
        messages: list[dict[str, str]],
    ) -> None:
        status = runner.status(profile, include_metrics=False)
        if not status.running:
            self._active_generations[profile.name] = False
            self._emit_async({
                "type": "chat_error",
                "generation_id": generation_id,
                "message": f"{profile.name} is not running. Start the model first.",
            })
            return

        self._dstorage_baselines[profile.name] = fetch_dstorage_metrics(profile.base_url)
        # Fixed Prometheus baseline for this request.  All throughput shown
        # while it runs is derived from counters advancing beyond this point.
        self._live_timing_samples[profile.name] = {
            key: value
            for key, value in self._dstorage_baselines[profile.name].items()
            if key.startswith("server_")
        }
        self._dstorage_rate_samples[profile.name] = dict(
            self._dstorage_baselines[profile.name]
        )
        self._request_metrics.pop(profile.name, None)
        self._live_stream_metrics[profile.name] = {}
        self._active_generations[profile.name] = True

        sampling = self._sampling.get(model_key, {})
        if not sampling:
            for key, env_key in (("temperature", "SIPHON_TEMPERATURE"),
                                 ("top_p", "SIPHON_TOP_P"),
                                 ("top_k", "SIPHON_TOP_K")):
                raw = profile.env.get(env_key)
                if raw is None:
                    continue
                try:
                    sampling[key] = float(raw)
                except ValueError:
                    continue
        payload_dict: dict[str, Any] = {
            "model": profile.env.get("ALIAS", profile.name),
            "messages": messages,
            "stream": True,
            "max_tokens": 2048,
        }
        if "temperature" in sampling:
            payload_dict["temperature"] = sampling["temperature"]
        if "top_p" in sampling:
            payload_dict["top_p"] = sampling["top_p"]
        if "top_k" in sampling:
            payload_dict["top_k"] = int(sampling["top_k"])
        budget = profile.env.get("REASONING_BUDGET")
        if budget:
            try:
                budget_value = int(budget)
                if budget_value >= 0:
                    payload_dict["reasoning_budget_tokens"] = budget_value
                    payload_dict["thinking_budget_tokens"] = budget_value
                    if budget_value == 0:
                        payload_dict["chat_template_kwargs"] = {
                            "preserve_thinking": False
                        }
            except ValueError:
                pass

        body = json.dumps(payload_dict).encode("utf-8")
        request = urllib.request.Request(
            f"{profile.base_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        final_decode_tps: float | None = None
        final_prefill_tps: float | None = None
        final_draft_accept_pct: float | None = None
        final_prompt_tokens: int | None = None
        final_generated_tokens: int | None = None
        first_stream_token: float | None = None
        streamed_tokens = 0
        full_text = ""
        thought_text = ""
        in_think_tag = False
        try:
            response = urllib.request.urlopen(request, timeout=300.0)
            with response:
                for raw_line in response:
                    if stop_event.is_set():
                        break
                    line = raw_line.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        timings = chunk.get("timings")
                        if isinstance(timings, dict):
                            try:
                                predicted_per_second = float(
                                    timings.get("predicted_per_second", 0) or 0
                                )
                                if predicted_per_second > 0:
                                    final_decode_tps = predicted_per_second
                            except (TypeError, ValueError):
                                pass
                            try:
                                prompt_per_second = float(
                                    timings.get("prompt_per_second", 0) or 0
                                )
                                if prompt_per_second > 0:
                                    final_prefill_tps = prompt_per_second
                            except (TypeError, ValueError):
                                pass
                            try:
                                prompt_tokens = int(timings.get("prompt_n", 0) or 0)
                                if prompt_tokens > 0:
                                    final_prompt_tokens = prompt_tokens
                            except (TypeError, ValueError):
                                pass
                            try:
                                generated_tokens = int(
                                    timings.get("predicted_n", 0) or 0
                                )
                                if generated_tokens > 0:
                                    final_generated_tokens = generated_tokens
                            except (TypeError, ValueError):
                                pass
                            try:
                                draft_n = float(timings.get("draft_n", 0) or 0)
                                draft_accepted = float(
                                    timings.get("draft_n_accepted", 0) or 0
                                )
                                if draft_n > 0:
                                    final_draft_accept_pct = (
                                        draft_accepted / draft_n * 100.0
                                    )
                            except (TypeError, ValueError):
                                pass
                        reasoning = delta.get("reasoning_content") or ""
                        content = delta.get("content") or ""
                        if reasoning:
                            thought_text += reasoning
                        if content:
                            if "<think>" in content:
                                before, content = content.split("<think>", 1)
                                full_text += before
                                in_think_tag = True
                            if in_think_tag:
                                if "</think>" in content:
                                    thinking, content = content.split("</think>", 1)
                                    thought_text += thinking
                                    in_think_tag = False
                                else:
                                    thought_text += content
                                    content = ""
                            full_text += content
                        if not reasoning and not content:
                            continue
                        now = time.monotonic()
                        streamed_tokens += 1
                        if first_stream_token is None:
                            first_stream_token = now
                            # Prompt accounting is committed by llama-server
                            # immediately before it emits the first token.
                            first_token_dstorage = fetch_dstorage_metrics(
                                profile.base_url
                            )
                            prompt_metrics = _live_engine_metrics(
                                first_token_dstorage,
                                self._live_timing_samples.get(profile.name, {}),
                            )
                            self._live_stream_metrics.setdefault(profile.name, {}).update({
                                key: value for key, value in prompt_metrics.items()
                                if key in ("prefill_tps", "prefill_sample", "n_prompt")
                            })
                            phase_metrics = _dstorage_request_metrics(
                                first_token_dstorage,
                                self._dstorage_baselines.get(profile.name, {}),
                            )
                            if phase_metrics.get("prefill_calls") is not None:
                                self._live_stream_metrics[profile.name]["prefill_calls"] = (
                                    phase_metrics["prefill_calls"]
                                )
                        elapsed = max(now - first_stream_token, 0.001)
                        # llama-server emits one streamed delta per sampled or
                        # accepted token. This is observed output throughput,
                        # not an animation or a configured constant.
                        if streamed_tokens > 1:
                            self._live_stream_metrics.setdefault(profile.name, {}).update({
                                "decode_tps": (streamed_tokens - 1) / elapsed,
                                "decode_sample": float(streamed_tokens),
                                "generated_tokens": float(streamed_tokens),
                            })
                        self._emit_async({
                            "type": "chat_delta",
                            "generation_id": generation_id,
                            "text": full_text,
                            "thought": thought_text,
                        })
                    except (KeyError, IndexError, TypeError, ValueError):
                        continue
        except urllib.error.HTTPError as exc:
            self._active_generations[profile.name] = False
            self._live_stream_metrics.pop(profile.name, None)
            self._dstorage_rate_samples.pop(profile.name, None)
            error = exc.read().decode("utf-8", "replace")
            self._emit_async({"type": "chat_error", "generation_id": generation_id,
                              "message": f"Server error {exc.code}: {error[:500]}"})
            return
        except (OSError, urllib.error.URLError) as exc:
            self._active_generations[profile.name] = False
            self._live_stream_metrics.pop(profile.name, None)
            self._dstorage_rate_samples.pop(profile.name, None)
            self._emit_async({"type": "chat_error", "generation_id": generation_id,
                              "message": f"Chat connection failed: {exc}"})
            return

        final_counter_metrics = _live_engine_metrics(
            fetch_dstorage_metrics(profile.base_url),
            self._live_timing_samples.get(profile.name, {}),
        )
        completed_metrics = {
            "decode_tps": final_decode_tps,
            "prefill_tps": final_prefill_tps,
            "draft_accept_pct": final_draft_accept_pct,
            "n_prompt": final_prompt_tokens,
            "active_experts": _profile_config(profile).get("expert_k"),
        }
        for key in ("decode_tps", "prefill_tps", "n_prompt"):
            if completed_metrics.get(key) is None:
                completed_metrics[key] = final_counter_metrics.get(key)
        if completed_metrics.get("n_prompt") is not None:
            completed_metrics["n_prompt"] = (
                float(completed_metrics["n_prompt"])
                + float(final_generated_tokens or streamed_tokens)
            )
        final_live_stream = dict(self._live_stream_metrics.get(profile.name, {}))
        self._active_generations[profile.name] = False
        self._live_stream_metrics.pop(profile.name, None)
        completed_metrics.update(_dstorage_request_metrics(
            fetch_dstorage_metrics(profile.base_url),
            self._dstorage_baselines.get(profile.name, {}),
        ))
        if final_live_stream.get("prefill_calls") is not None:
            completed_metrics["prefill_calls"] = final_live_stream["prefill_calls"]
        if any(value is not None for value in completed_metrics.values()):
            self._request_metrics[profile.name] = completed_metrics

        self._emit_async({
            "type": "chat_complete",
            "generation_id": generation_id,
            "stopped": stop_event.is_set(),
            "text": full_text,
            "thought": thought_text,
            **completed_metrics,
        })

    def _run_model_action(self, starting: bool, model_key: str) -> None:
        profile = self._profile_for(model_key)
        if profile is None:
            self._emit_event({
                "type": "runtime_complete",
                "starting": starting,
                "ok": False,
                "message": f"No launcher profile is configured for {model_key}.",
            })
            return

        def work() -> None:
            if starting:
                config = _profile_config(profile)
                resources_ok, resource_message = _validate_resource_settings(
                    config, fetch_sys_stats()
                )
                if not resources_ok:
                    GLib.idle_add(
                        self._finish_model_action,
                        starting,
                        False,
                        resource_message,
                    )
                    return
                # Capture before launching: startup output remains visible to
                # the log parser, while metrics from an earlier server run do
                # not leak into this new model session.
                self._log_offsets[profile.name] = _file_size(profile.log_file)
                self._request_metrics.pop(profile.name, None)
                self._active_generations[profile.name] = False
                self._dstorage_baselines.pop(profile.name, None)
                self._live_timing_samples.pop(profile.name, None)
                self._live_stream_metrics.pop(profile.name, None)
                self._dstorage_rate_samples.pop(profile.name, None)
                ok, message = runner.up(profile)
                if ok:
                    healthy = runner.wait_healthy(profile, timeout=600.0, interval=1.0)
                    probe_error = _probe_inference(profile) if healthy else None
                    startup_error = _startup_log_error(
                        profile.log_file, self._log_offsets[profile.name]
                    )
                    if healthy and probe_error is None and startup_error is None:
                        message = f"{message}; model health is ready"
                    else:
                        ok = False
                        runner.down(profile, force=True)
                        failure = startup_error or probe_error
                        message = (f"{message}; DirectStorage startup failed: {failure}"
                                   if failure else
                                   f"{message}; model did not become healthy within 10 minutes")
            else:
                # A visible Stop action is an explicit user request; allow it
                # to clean up a process even if /health has already degraded.
                self._request_metrics.pop(profile.name, None)
                self._active_generations[profile.name] = False
                self._dstorage_baselines.pop(profile.name, None)
                self._live_timing_samples.pop(profile.name, None)
                self._live_stream_metrics.pop(profile.name, None)
                self._dstorage_rate_samples.pop(profile.name, None)
                ok, message = runner.down(profile, force=True)
            GLib.idle_add(
                self._finish_model_action,
                starting,
                ok,
                message,
            )

        threading.Thread(target=work, name="siphon-model-action", daemon=True).start()

    def _finish_model_action(self, starting: bool, ok: bool, message: str) -> bool:
        self._emit_event({
            "type": "runtime_complete",
            "starting": starting,
            "ok": ok,
            "message": message,
        })
        return GLib.SOURCE_REMOVE

    def _start_polling(self) -> bool:
        self._poll()
        GLib.timeout_add(750, self._poll)
        return GLib.SOURCE_REMOVE

    def _poll(self) -> bool:
        if self._closed:
            return GLib.SOURCE_REMOVE
        if self._poll_paused:
            return GLib.SOURCE_CONTINUE
        if self._poll_busy:
            return GLib.SOURCE_CONTINUE
        profile = self._profile_for(self.model_key)
        if profile is None:
            self._emit_event({"type": "telemetry", "running": False})
            return GLib.SOURCE_CONTINUE

        self._poll_busy = True
        telemetry_epoch = self._telemetry_epoch

        def work() -> None:
            try:
                status = runner.status(profile, include_metrics=False)
                # Prometheus still supplies health/slot metadata. Throughput
                # comes from request-scoped live counters below, never from
                # its last-completed-request throughput gauges.
                metrics = fetch_metrics(profile.base_url) if status.running else status.metrics
                system = fetch_sys_stats()
                slots = fetch_slots(profile.base_url) if status.running else None
                dstorage_raw = (
                    fetch_dstorage_metrics(profile.base_url)
                    if status.running else {}
                )
                live_timing = _live_engine_metrics(
                    dstorage_raw,
                    self._live_timing_samples.setdefault(profile.name, {}),
                )
                live_stream = self._live_stream_metrics.get(profile.name, {})
                dstore = (
                    parse_log_tail(
                        profile.log_file,
                        start_offset=self._log_offsets.get(profile.name, 0),
                    )
                    if status.running else None
                )
                request_active = bool(
                    self._active_generations.get(profile.name, False)
                    or (slots and slots.busy > 0)
                    or metrics.n_processing > 0
                )
                live_transfer = _live_transfer_rates(
                    dstorage_raw,
                    self._dstorage_rate_samples.setdefault(profile.name, {}),
                ) if request_active else {}
                live_cache = _live_cache_rates(
                    dstorage_raw,
                    self._dstorage_rate_samples.setdefault(profile.name, {}),
                ) if request_active else {}
                cached_request = self._request_metrics.get(profile.name, {})
                dstorage_request = _dstorage_request_metrics(
                    dstorage_raw,
                    self._dstorage_baselines.get(profile.name, {}),
                )
                fresh_request_data = bool(dstore and any((
                    dstore.decode_tps is not None,
                    dstore.prefill_tps is not None,
                    dstore.draft_accept is not None,
                    dstore.decode_calls is not None,
                    dstore.prefill_calls is not None,
                )))
                fresh_request_data = fresh_request_data or bool(dstorage_request)
                fresh_request_data = fresh_request_data or bool(live_timing)
                fresh_request_data = fresh_request_data or bool(live_stream)
                has_cached_request = any(
                    value is not None for value in cached_request.values()
                )
                has_request_data = (
                    request_active or fresh_request_data or has_cached_request
                )
                decode_tps = (
                    (live_stream.get("decode_tps") if request_active else None)
                    or live_timing.get("decode_tps")
                    or (dstore.decode_tps if dstore else None)
                    or cached_request.get("decode_tps")
                ) if has_request_data else None
                prefill_tps = (
                    (live_stream.get("prefill_tps") if request_active else None)
                    or live_timing.get("prefill_tps")
                    or (dstore.prefill_tps if dstore else None)
                    or cached_request.get("prefill_tps")
                ) if has_request_data else None
                live_context = None
                if request_active and live_stream.get("n_prompt") is not None:
                    live_context = (
                        float(live_stream["n_prompt"])
                        + float(live_stream.get("generated_tokens", 0.0) or 0.0)
                    )
                payload: dict[str, Any] = {
                    "type": "telemetry",
                    "running": status.running,
                    "health": status.health.status,
                    "request_state": (
                        "active" if request_active
                        else "completed" if (fresh_request_data or has_cached_request)
                        else "idle"
                    ),
                    "decode_tps": decode_tps,
                    "prefill_tps": prefill_tps,
                    "decode_sample": (
                        live_stream.get("decode_sample") if request_active
                        else live_timing.get("decode_sample")
                    ),
                    "prefill_sample": (
                        live_stream.get("prefill_sample") if request_active
                        else live_timing.get("prefill_sample")
                    ),
                    "active_experts": (
                        _profile_config(profile).get("expert_k")
                        if request_active else cached_request.get("active_experts")
                    ),
                    "n_prompt": (
                        live_context
                        or live_timing.get("n_prompt")
                        or (slots.total_past if slots and slots.total_past else None)
                        or metrics.n_prompt
                        or cached_request.get("n_prompt")
                        if has_request_data else None
                    ),
                    "n_processing": (slots.busy if slots else metrics.n_processing),
                    "cpu_util": system.cpu if system.mem_total_gb else None,
                    "cpu_temp": system.cpu_temp if system.cpu_temp > 0 else None,
                    "ram_used_gb": system.mem_used_gb if system.mem_total_gb else None,
                    "ram_total_gb": system.mem_total_gb if system.mem_total_gb else None,
                    "ram_available_gb": system.mem_free_gb if system.mem_total_gb else None,
                    "gpu_util": system.gpu_util if system.gpu_mib_total else None,
                    "gpu_temp": system.gpu_temp if system.gpu_mib_total else None,
                    "vram_used_gb": (system.gpu_mib_used / 1024)
                    if system.gpu_mib_total else None,
                    "vram_total_gb": (system.gpu_mib_total / 1024)
                    if system.gpu_mib_total else None,
                    "vram_available_gb": (system.gpu_mib_free / 1024)
                    if system.gpu_mib_total else None,
                    "ram_safe_mib": _resource_state(system)["ram_safe_mib"],
                    "vram_safe_mib": _resource_state(system)["vram_safe_mib"],
                    "ram_headroom_mib": RAM_HEADROOM_MIB,
                    "vram_headroom_mib": VRAM_HEADROOM_MIB,
                    "cpu_threads": _resource_state(system)["cpu_threads"],
                    "gds_threads_max": _resource_state(system)["gds_threads_max"],
                    "recommended": _recommended_settings(
                        _profile_config(profile), system
                    ),
                    "gpu_name": getattr(system, "gpu_name", "") or "GPU telemetry unavailable",
                    "cpu_name": _cpu_label(),
                }
                if dstore and has_request_data:
                    payload.update({
                        "draft_accept_pct": (
                            dstore.draft_accept * 100.0
                            if dstore.draft_accept is not None
                            else cached_request.get("draft_accept_pct")
                        ),
                        "l1_hit_pct": dstore.l1_hit_pct,
                        "l2_hit_pct": dstore.l2_hit_pct,
                        "gds_gbs": (dstore.nvme_mibs / 1024)
                        if dstore.nvme_mibs is not None else None,
                        "ram_vram_gbs": (dstore.ram_vram_mibs / 1024)
                        if dstore.ram_vram_mibs is not None else None,
                        "streamed_gb": (dstore.streamed_mib / 1024)
                        if dstore.streamed_mib is not None else None,
                        "decode_calls": dstore.decode_calls,
                        "prefill_calls": dstore.prefill_calls,
                    })
                elif has_cached_request:
                    payload["draft_accept_pct"] = cached_request.get(
                        "draft_accept_pct"
                    )
                # Prefer the live engine counters when this binary supports
                # them.  The log parser remains a fallback for older builds.
                if dstorage_request:
                    payload.update(dstorage_request)
                if live_timing:
                    payload.update(live_timing)
                if live_transfer:
                    payload.update(live_transfer)
                if live_cache:
                    payload.update(live_cache)
                if live_context is not None:
                    payload["n_prompt"] = live_context
                elif not request_active and cached_request.get("n_prompt") is not None:
                    payload["n_prompt"] = cached_request["n_prompt"]
                if request_active and live_stream.get("prefill_calls") is not None:
                    payload["prefill_calls"] = live_stream["prefill_calls"]
                elif not request_active and cached_request.get("prefill_calls") is not None:
                    payload["prefill_calls"] = cached_request["prefill_calls"]
            except Exception as exc:  # telemetry should never take down the UI
                payload = {"type": "telemetry", "running": False, "error": str(exc)}
            GLib.idle_add(self._finish_poll, payload, telemetry_epoch)

        threading.Thread(target=work, name="siphon-telemetry", daemon=True).start()
        return GLib.SOURCE_CONTINUE

    def _finish_poll(self, payload: dict[str, Any], telemetry_epoch: int) -> bool:
        self._poll_busy = False
        if telemetry_epoch == self._telemetry_epoch:
            self._emit_event(payload)
        return GLib.SOURCE_REMOVE

    def _emit_event(self, payload: dict[str, Any]) -> None:
        if self._closed:
            return
        serialized = json.dumps(payload, separators=(",", ":"))
        script = f"window.siphonDesktopEvent?.({serialized});"
        try:
            self.webview.evaluate_javascript(script, -1, None, None, None, None, None)
        except Exception:
            # The page may be navigating or closing; this is not fatal.
            pass


class SiphonApplication(Gtk.Application):
    """Native application host for the Siphon Studio desktop surface."""

    def __init__(self) -> None:
        super().__init__(application_id=APP_ID)
        self.window: Gtk.ApplicationWindow | None = None
        self.bridge: DesktopBridge | None = None

    def do_activate(self) -> None:
        if self.window is not None:
            self.window.present()
            return

        self.window = Gtk.ApplicationWindow(application=self)
        self.window.set_title(WINDOW_TITLE)
        self.window.set_default_size(1500, 950)
        self.window.set_resizable(True)
        # The approved surface contains its own compact titlebar and controls.
        self.window.set_decorated(False)

        webview = WebKit.WebView.new()
        manager = webview.get_user_content_manager()
        settings = webview.get_settings()
        settings.set_enable_javascript(True)
        # WebKitGTK's accelerated scroll layer can rasterize small catalogue
        # text into a low-resolution texture on fractional-scale displays.
        # This is a local, mostly static control surface, so software painting
        # keeps labels sharp while scrolling without adding a heavy runtime.
        settings.set_hardware_acceleration_policy(
            WebKit.HardwareAccelerationPolicy.NEVER
        )
        settings.set_enable_2d_canvas_acceleration(False)
        webview.set_zoom_level(1.0)
        webview.set_settings(settings)
        self.bridge = DesktopBridge(self.window, webview)
        self.bridge.register(manager)
        self.window.set_child(webview)

        self.window.connect("close-request", self._on_close_request)
        try:
            webview.load_uri(_prototype_path().as_uri())
        except FileNotFoundError as exc:
            self._show_error(webview, str(exc))
        self.window.maximize()
        self.window.present()

    def _show_error(self, webview: WebKit.WebView, message: str) -> None:
        html = (
            "<body style='background:#111;color:#eee;font:16px sans-serif;padding:32px'>"
            "<h1>Siphon Studio could not load</h1><p>"
            + GLib.markup_escape_text(message)
            + "</p></body>"
        )
        webview.load_html(html, "file:///siphon-studio-error.html")

    def _on_close_request(self, _window: Gtk.Window) -> bool:
        if self.bridge is not None:
            self.bridge.close()
        return False


def run() -> int:
    """Launch the native desktop application."""
    if not Gtk.init_check() or Gdk.Display.get_default() is None:
        print(
            "Siphon Studio needs an active graphical display (GTK4 could not initialize).",
            flush=True,
        )
        return 3
    app = SiphonApplication()
    return app.run(None)
