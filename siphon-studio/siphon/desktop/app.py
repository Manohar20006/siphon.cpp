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
from ..core import runner, execution
from ..core.monitor import fetch_dstorage_metrics, fetch_health, fetch_metrics
from ..core.profiles import Profile, discover
from ..core.memory_budget import gpu_reservation, memory_signature
from ..core.runtime_memory import read_memory_record, load_observation, save_observation
from ..core.model_files import (
    inspect_model_files, model_fingerprint, model_part_paths, scan_model_folder,
)
from .catalogue import load_excluded_model_keys, save_excluded_model_keys
from ..api import connect as api_connect
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
VRAM_HEADROOM_MIB = 128


def _cpu_ram_reservation(layout: dict[str, Any] | None, config: dict[str, Any]) -> dict[str, Any] | None:
    """Reserve CPU KV/state and a provisional workspace, not the mmap file size."""
    if not layout:
        return None
    kind = config.get("cache_type", "q8_0")
    per_token = sum(layout["kv_bytes_by_type"][kind])
    mtp = config.get("spec_type") == "draft-mtp"
    draft_per_token = sum(layout["draft_kv_bytes_by_type"]["f16"]) if mtp else 0
    context = max(0, int(config.get("context", 0)))
    rollback = int(config.get("draft_tokens", 0)) if mtp and layout.get("rollback_supported", True) else 0
    recurrent = sum(layout.get("recurrent_bytes_by_layer", [])) / 1048576 * max(1, int(config.get("parallel", 1))) * (1 + rollback)
    return {"kv_bytes_per_token": per_token, "draft_kv_bytes_per_token": draft_per_token,
            "kv_mib": context * (per_token + draft_per_token) / 1048576,
            "recurrent_mib": recurrent, "compute_mib": 512,
            "total_mib": context * (per_token + draft_per_token) / 1048576 + recurrent + 512}


def _resource_state(system: Any, config: dict[str, Any] | None = None,
                    memory_layout: dict[str, Any] | None = None, *, running: bool = False) -> dict[str, Any]:
    """Return live allocation ceilings using available, not total, memory."""
    ram_available_mib = max(0, int(float(system.mem_free_gb or 0) * 1024))
    vram_available_mib = max(0, int(float(system.gpu_mib_free or 0)))
    cpu_threads = max(1, int(os.cpu_count() or 1))
    free_safe = max(0, vram_available_mib - VRAM_HEADROOM_MIB)
    if memory_layout and memory_layout.get("runtime_memory", {}).get("gpu_name") != getattr(system, "gpu_name", ""):
        memory_layout = {key: value for key, value in memory_layout.items() if key != "runtime_memory"}
    budget_error = ""
    cpu_mode = (config or {}).get("execution_mode") == "cpu"
    cpu_reservation = None
    try:
        if cpu_mode:
            cpu_reservation = _cpu_ram_reservation(memory_layout, config or {})
            reservation = None
        else:
            reservation = gpu_reservation(memory_layout, config or {}) if memory_layout else None
    except (ValueError, TypeError, KeyError) as exc:
        reservation = None
        budget_error = str(exc)
    # A loaded engine's weights/KV are already absent from GPU free memory.
    # Locked allocations must never be subtracted a second time or reduced.
    reserved = reservation["total_mib"] if reservation and not running else 0
    return {
        "execution_mode": (config or {}).get("execution_mode", "gpu"),
        "ram_available_gb": float(system.mem_free_gb or 0),
        "vram_available_gb": vram_available_mib / 1024.0,
        "ram_safe_mib": max(0, ram_available_mib - RAM_HEADROOM_MIB),
        "vram_safe_mib": 0 if budget_error else max(0, free_safe - reserved),
        "vram_free_safe_mib": free_safe,
        "vram_reservation": reservation,
        "cpu_ram_reservation": cpu_reservation,
        "memory_layout": memory_layout,
        "vram_budget_known": bool(reservation) if config is not None else True,
        "vram_budget_error": budget_error,
        "ram_headroom_mib": RAM_HEADROOM_MIB,
        "vram_headroom_mib": VRAM_HEADROOM_MIB,
        "cpu_threads": cpu_threads,
        "gds_threads_max": min(32, cpu_threads),
    }


def _validate_resource_settings(
    values: dict[str, Any], system: Any, *, enforce_memory_capacity: bool = True,
    memory_layout: dict[str, Any] | None = None,
    require_model_budget: bool = False,
) -> tuple[bool, str]:
    """Reject launch profiles that exceed currently safe host capacities."""
    cpu_mode = values.get("execution_mode") == "cpu"
    if values.get("execution_mode", "gpu") not in {"cpu", "gpu"}:
        return False, "Execution mode must be CPU or GPU."
    if enforce_memory_capacity and require_model_budget and memory_layout is None:
        return False, "Model memory requirements are unavailable; inspect the model before saving or starting."
    if cpu_mode and values.get("cache_type", "q8_0") not in {"q8_0", "f16"}:
        return False, "CPU mode supports Q8_0 or F16 attention cache precision."
    try:
        state = _resource_state(system, values, memory_layout)
    except (ValueError, TypeError, KeyError) as exc:
        return False, f"Cannot determine model {'CPU RAM' if cpu_mode else 'GPU memory'} budget: {exc}"
    if state["vram_budget_error"]:
        return False, f"Cannot determine model {'CPU RAM' if cpu_mode else 'GPU memory'} budget: {state['vram_budget_error']}"
    if enforce_memory_capacity and system.gpu_mib_total and state["vram_reservation"] and state["vram_reservation"]["total_mib"] > state["vram_free_safe_mib"]:
        return False, "Model weights and buffers exceed available GPU memory after headroom. Reduce context or GPU placement before starting."
    try:
        pinned_mib = max(0, int(values.get("pinned_cache", values.get("l2_cache", 0)) or 0))
        prompt_mib = max(0, int(values.get("cache_ram", 0) or 0))
        vram_mib = max(0, int(values.get("vram_cache", values.get("l1_cache", 0)) or 0))
    except (TypeError, ValueError):
        return False, "Memory allocations must be valid numbers."
    requested_ram = prompt_mib + (state["cpu_ram_reservation"]["total_mib"] if state["cpu_ram_reservation"] else 0) if cpu_mode else pinned_mib + prompt_mib
    if enforce_memory_capacity and system.mem_total_gb and requested_ram > state["ram_safe_mib"]:
        return False, (
            f"RAM allocation is {requested_ram / 1024:.1f} GiB, but only "
            f"{state['ram_safe_mib'] / 1024:.1f} GiB is safely available. "
            "Siphon always preserves at least 2 GiB of free RAM."
        )
    if enforce_memory_capacity and not cpu_mode and system.gpu_mib_total and vram_mib > state["vram_safe_mib"]:
        return False, (
            f"VRAM cache is {vram_mib / 1024:.1f} GiB, but only "
            f"{state['vram_safe_mib'] / 1024:.1f} GiB is safely available "
            "after model weights, KV/state buffers, and runtime/headroom reserves."
        )
    if cpu_mode:
        try:
            threads = int(values.get("cpu_threads", 1))
        except (ValueError, TypeError):
            return False, "CPU threads must be a valid number."
        if not 1 <= threads <= state["cpu_threads"]:
            return False, f"CPU threads must be between 1 and {state['cpu_threads']}."
        return True, ""
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


def _recommended_settings(config: dict[str, Any], system: Any,
                          memory_layout: dict[str, Any] | None = None) -> dict[str, int]:
    """Produce conservative, model-aware cache budgets for this machine."""
    if config.get("execution_mode") == "cpu":
        context = min(int(config.get("context", 4096)), 65536)
        safe = _resource_state(system)["ram_safe_mib"]
        while context > 4096:
            reserve = _cpu_ram_reservation(memory_layout, {**config, "context": context})
            if not reserve or reserve["total_mib"] <= safe / 2:
                break
            context //= 2
        reserve = _cpu_ram_reservation(memory_layout, {**config, "context": context})
        remaining = max(0, safe - (reserve["total_mib"] if reserve else 512))
        return {"context": context, "cache_ram": min(int(config.get("cache_ram", 0)), int(remaining // 512 * 512)),
                "cpu_threads": execution.default_cpu_threads(), "expert_k": int(config.get("expert_k", 0))}
    context = min(int(config.get("context", 0) or 0), _recommended_context(_resource_state(system)["vram_safe_mib"]))
    state = _resource_state(system, {**config, "context": context, "gpu_layers": 999}, memory_layout)
    ram_safe = int(state["ram_safe_mib"])
    vram_safe = int(state["vram_safe_mib"])
    prompt_target = min(int(config.get("cache_ram", 0) or 0), ram_safe // 3)
    prompt_target = max(0, prompt_target // 512 * 512)
    pinned_target = min(int(config.get("l2_cache", 0) or 0), ram_safe - prompt_target)
    pinned_target = max(0, pinned_target // 1024 * 1024)
    gpu_layers_target, vram_target = _recommended_gpu_budget(vram_safe, reserve_mib=0 if memory_layout else 512)
    return {
        "gpu_layers": gpu_layers_target,
        "vram_cache": vram_target,
        "pinned_cache": pinned_target,
        "cache_ram": prompt_target,
        "context": context,
        "expert_k": int(config.get("expert_k", 0) or 0),
        "gds_threads": min(
            max(1, int(config.get("gds_threads", 1) or 1)),
            int(state["gds_threads_max"]),
        ),
    }


def _recommended_gpu_budget(vram_safe_mib: int, *, reserve_mib: int = 512) -> tuple[int, int]:
    """Reserve a conservative DirectStorage expert pool.

    DirectStorage expert tensors are GPU addresses and every MoE layer that
    consumes them must therefore remain GPU-routed.  Memory is controlled with
    context/KV size and the bounded expert pool, not partial CPU layer offload.
    """
    usable_mib = max(0, int(vram_safe_mib) - reserve_mib)
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
        "unknown model architecture",
        "error loading model",
        "failed to load model",
        "exiting due to model loading error",
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
    markers = tuple(marker.lower() for marker in fatal_markers)
    return next((
        line.strip()
        for line in text.splitlines()
        if any(marker in line.lower() for marker in markers)
    ), None)


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


def _profile_config(profile: Profile, *, include_modes: bool = True) -> dict[str, Any]:
    """Return launch settings without touching the model weights.

    This intentionally stays separate from GGUF inspection so the catalogue
    can show the selected profile's real defaults immediately. Reading a
    large GGUF header/tensor table is done asynchronously afterward.
    """
    original = profile
    mode = execution.mode(profile)
    profile = Profile(name=profile.name, env=execution.effective_env(profile), script=profile.script)
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
    config = {
        "execution_mode": mode,
        "gpu_available": execution.gpu_available(),
        "cpu_threads": int(profile.env.get("THREADS", str(execution.default_cpu_threads()))),
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
        "draft_cache_type": "f16",  # Current launchers do not override -ctk/-ctv-draft.
        "parallel": int(profile.env.get("N_PARALLEL", "1") or 1),
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
    if include_modes:
        config["mode_settings"] = {
            name: _profile_config(Profile(name=original.name,
                env={**original.env, "SIPHON_EXECUTION_MODE": name}, script=original.script), include_modes=False)
            for name in ("gpu", "cpu")
        }
    return config


def _metadata_cache_path() -> Path:
    return paths.state_dir() / "model-metadata.json"


_METADATA_CACHE_VERSION = 4
_metadata_cache_lock = threading.RLock()


def _profile_model_parts(profile: Profile) -> list[Path]:
    if profile.model_files:
        return [Path(path) for path in profile.model_files]
    model = profile.resolve("MODEL")
    return model_part_paths(Path(model)) if model else []


def _profile_model_available(profile: Profile) -> bool:
    try:
        parts = _profile_model_parts(profile)
        return bool(parts) and all(path.is_file() and os.access(path, os.R_OK) for path in parts)
    except (OSError, ValueError):
        return False


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
        fingerprint = model_fingerprint([Path(p) for p in payload.get("parts", [str(path)])])
        if payload.get("fingerprint", fingerprint) != fingerprint:
            return
        entry = {
            "version": _METADATA_CACHE_VERSION,
            "fingerprint": fingerprint,
            "data": {
                key: payload[key]
                for key in (
                    "name", "architecture", "layers", "native_context",
                    "expert_count", "expert_used_count", "artifact_gib",
                    "dense_gib", "core_gib", "lookup_gib", "expert_gib",
                    "tensor_gib", "part_count", "parts", "memory_layout",
                )
                if key in payload
            },
        }
        with _metadata_cache_lock:
            cache = _read_metadata_cache()
            cache[str(path)] = entry
            cache_path = _metadata_cache_path()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            pending = cache_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
            try:
                pending.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
                pending.replace(cache_path)
            finally:
                pending.unlink(missing_ok=True)
    except (OSError, TypeError, ValueError):
        # Metadata caching is an optimization. A read-only state directory or
        # an unusual filesystem must never prevent model inspection.
        return


def _cached_model_info(profile: Profile, model_key: str) -> dict[str, Any] | None:
    model_path = Path(profile.resolve("MODEL"))
    if not _profile_model_available(profile):
        return None
    try:
        path = model_path.resolve()
        entry = _read_metadata_cache().get(str(path), {})
        if (entry.get("version") != _METADATA_CACHE_VERSION or
                entry.get("fingerprint") != model_fingerprint(_profile_model_parts(profile)) or
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
        "available": _profile_model_available(profile),
        "profile": profile.name,
        "path": str(model_path),
        "config": _profile_config(profile),
    }
    if not info["available"]:
        return info

    try:
        fingerprint = model_fingerprint(_profile_model_parts(profile))
        info.update(inspect_model_files(model_path))
        if fingerprint != model_fingerprint([Path(p) for p in info["parts"]]):
            raise ValueError("Model parts changed while inspecting; choose the folder again.")
        info["fingerprint"] = fingerprint
        info["name"] = MODEL_DISPLAY_NAMES.get(model_key, info["name"])
    except Exception as exc:
        info["available"] = False
        info["metadata_error"] = str(exc)
    return info


def _memory_observation_identity(profile: Profile) -> dict[str, Any]:
    binary = paths.server_bin()
    runtime = [binary, profile.script]
    runtime.extend(sorted(binary.parent.glob("lib*.so*")))
    environment = {key: value for key, value in os.environ.items()
                   if key.startswith(("GGML_", "LLAMA_", "CUDA_")) or key in {
                        "CTX", "BATCH", "UBATCH", "GPU_LAYERS", "CACHE_TYPE", "FLASH_ATTENTION",
                        "ROPE_SCALING", "MOE_GPU_CACHE_MIB", "MOE_PINNED_CACHE_MIB", "CACHE_RAM_MIB",
                        "SPEC_TYPE", "MODEL_DRAFT", "SPEC_DRAFT_N_MAX", "SPEC_DRAFT_P_MIN", "N_PARALLEL",
                        "CACHE_PROMPT", "N_CTX_CHECKPOINTS", "GDS_READ_THREADS", "MMPROJ", "RUNTIME_MODE"}}
    environment.update(profile.env)
    # Hash effective launcher settings: all profile fields invalidate safely,
    # but their values (which could include credentials) are never persisted.
    return {"model": model_fingerprint(_profile_model_parts(profile)),
            "runtime": model_fingerprint(runtime),
            "env": hashlib.sha256(json.dumps(environment, sort_keys=True).encode()).hexdigest()}


def _record_runtime_memory(profile: Profile, config: dict[str, Any], offset: int) -> None:
    if config.get("execution_mode") == "cpu":
        return  # Never overwrite a GPU workspace observation with CPU bytes.
    record = read_memory_record(profile.log_file, offset)
    if record:
        try:
            save_observation(paths.state_dir() / "runtime-memory.json", profile.name,
                             _memory_observation_identity(profile), {**record, "config": memory_signature(config)})
        except (OSError, ValueError):
            pass


def _profile_memory_layout(profile: Profile, *, inspect: bool = False) -> dict[str, Any] | None:
    info = _cached_model_info(profile, _model_key_for_profile(profile))
    if info is None and inspect and _profile_model_available(profile):
        info = _model_info(profile, _model_key_for_profile(profile))
        if info.get("available"):
            _write_metadata_cache(info)
    layout = info.get("memory_layout") if info else None
    if layout:
        try:
            record = load_observation(paths.state_dir() / "runtime-memory.json", profile.name,
                                      _memory_observation_identity(profile))
            if record:
                layout = {**layout, "runtime_memory": record}
        except (OSError, ValueError):
            pass
    return layout


class DesktopBridge:
    """Translate small JSON messages from the HTML surface into core actions."""

    def __init__(self, window: Gtk.Window, webview: WebKit.WebView) -> None:
        self.window = window
        self.webview = webview
        self.profiles: dict[str, Profile] = discover()
        self._catalogue_lock = threading.RLock()
        self._all_profile_by_model_key = _profile_map(self.profiles)
        self._excluded_model_keys = load_excluded_model_keys()
        self._registration_generations: dict[str, int] = {}
        self.profile_by_model_key = {
            key: name for key, name in self._all_profile_by_model_key.items()
            if key not in self._excluded_model_keys
        }
        self.model_key = "qwen35"
        self._poll_busy = False
        self._poll_paused = False
        self._telemetry_epoch = 0
        self._closed = False
        self._chat_stop_events: dict[str, threading.Event] = {}
        self._chat_lock = threading.Lock()
        self._sampling: dict[str, dict[str, float]] = {}
        self._import_lock = threading.RLock()
        self._import_scan_id = ""
        self._import_candidates: dict[str, dict[str, Any]] = {}
        self._request_metrics: dict[str, dict[str, Any]] = {}
        # DirectStorage exposes lifetime counters through /metrics.  Each chat
        # request records a baseline so the dashboard can show its own real
        # movement, rather than the totals accumulated by earlier requests.
        self._dstorage_baselines: dict[str, dict[str, float]] = {}
        self._active_generations: dict[str, bool] = {}
        self._starting_models: set[str] = set()
        self._stopping_models: set[str] = set()
        self._live_timing_samples: dict[str, dict[str, float]] = {}
        self._live_stream_metrics: dict[str, dict[str, float]] = {}
        self._dstorage_rate_samples: dict[str, dict[str, float]] = {}
        # Server logs are append-only.  Start every desktop session at the
        # current end so old timing summaries never appear as a fresh request.
        self._log_offsets = {
            profile.name: _file_size(profile.log_file)
            for profile in self.profiles.values()
        }
        self._api_init()

    def register(self, manager: WebKit.UserContentManager) -> None:
        manager.register_script_message_handler("siphon")
        manager.connect("script-message-received::siphon", self._message_received)

    def _ensure_shutdown_state(self) -> None:
        if not hasattr(self, "_shutdown_lock"):
            self._shutdown_lock = threading.RLock()
            self._launch_lock = threading.RLock()
            self._shutdown_event = threading.Event()
            self._shutdown_done = threading.Event()
            self._shutdown_thread = None
            self._shutdown_callbacks = []
            self._shutdown_errors = []
            self._model_action_threads = []

    def close(self, on_complete=None) -> None:
        """Cancel new work and asynchronously unload all managed engines."""
        self._ensure_shutdown_state()
        with self._shutdown_lock:
            thread = self._shutdown_thread
            if self._shutdown_done.is_set() and not self._shutdown_errors:
                if on_complete is not None:
                    GLib.idle_add(on_complete, [])
                return
            if on_complete is not None:
                self._shutdown_callbacks.append(on_complete)
            if thread is not None and not self._shutdown_done.is_set():
                return
            self._shutdown_done.clear()
            self._closed = True
            self._shutdown_event.set()
            self._api_stop()
            self._telemetry_epoch += 1
            with self._chat_lock:
                for stop_event in self._chat_stop_events.values():
                    stop_event.set()

            def cleanup() -> None:
                errors = []
                try:
                    # A launch already inside runner.up must finish recording
                    # its ownership before teardown. Later launches see closed.
                    with self._launch_lock:
                        profiles = dict(self.profiles)
                        try:
                            for name, profile in discover().items():
                                profiles.setdefault(name, profile)
                        except Exception as exc:
                            errors.append(f"Profile discovery during shutdown failed: {exc}")
                        errors.extend(runner.shutdown(list(profiles.values())))
                except Exception as exc:
                    errors.append(f"Model shutdown failed: {exc}")
                with self._shutdown_lock:
                    self._shutdown_errors = errors
                    callbacks, self._shutdown_callbacks = self._shutdown_callbacks, []
                    self._shutdown_done.set()
                for error in errors:
                    print(f"Siphon Studio shutdown: {error}", flush=True)
                for callback in callbacks:
                    GLib.idle_add(callback, errors)

            # Do not allow interpreter exit to abandon model termination.
            self._shutdown_thread = threading.Thread(
                target=cleanup, name="siphon-desktop-shutdown", daemon=False,
            )
            self._shutdown_thread.start()

    def wait_closed(self) -> list[str]:
        """Final application shutdown waits for the bounded runner cleanup."""
        self._ensure_shutdown_state()
        if self._shutdown_thread is not None:
            self._shutdown_thread.join()
        return list(self._shutdown_errors)

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
        if self._closed:
            if action == "window_action" and message.get("name") == "close":
                self.window.close()
            return
        if action in {"page_ready", "select_model"}:
            self._ensure_catalogue_state()
            with self._catalogue_lock:
                next_model_key = str(message.get("model", self.model_key))
                if self._profile_for(next_model_key) is None:
                    if action == "select_model":
                        self._schedule_catalogue()
                        return
                    next_model_key = next(iter(self.profile_by_model_key), "")
                if next_model_key != self.model_key or action == "page_ready":
                    self._telemetry_epoch += 1
                self.model_key = next_model_key
                if self.model_key:
                    self._schedule_model_info(self.model_key)
            if action == "page_ready":
                self._schedule_catalogue()
                GLib.idle_add(self._start_polling)
            return
        if action == "inspect_model":
            self._schedule_model_info(str(message.get("model", "")))
            return
        if action == "add_model":
            GLib.idle_add(self._choose_model_folder, str(message.get("scan_id") or uuid.uuid4().hex))
            return
        if action == "cancel_model_import":
            self._cancel_model_import(str(message.get("scan_id", "")))
            return
        if action == "import_models":
            threading.Thread(target=self._import_selected_models,
                             args=(str(message.get("scan_id", "")), message.get("model_ids", [])),
                             name="siphon-folder-import", daemon=True).start()
            return
        if action == "remove_model":
            threading.Thread(
                target=self._remove_model,
                args=(str(message.get("model", "")), str(message.get("removal_id", ""))),
                name="siphon-model-removal",
                daemon=True,
            ).start()
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
        if action.startswith("api_"):
            self._api_message(action, message)
            return
        if action == "window_action":
            self._window_action(str(message.get("name", "")))

    # -- local API: scoped keys + the 127.0.0.1 gateway (siphon.api) ----------------------------
    def _api_init(self) -> None:
        from ..api import gateway as api_gateway
        from ..api.keys import KeyStore
        self._api_port = api_gateway.default_port()
        self._api_keys = KeyStore()
        self._api_server: Any = None
        self._api_error = ""
        self._api_busy = ""
        self._api_busy_mode = ""
        self._api_lock = threading.Lock()
        self._api_trying: set[str] = set()
        try:
            self._api_server = api_gateway.Gateway(self._api_port, keys=self._api_keys)
        except OSError as exc:
            # another Studio or `siphon api serve` may already own the port; that one serves the same keys
            if not self._api_external():
                self._api_error = f"cannot listen on 127.0.0.1:{self._api_port}: {exc.strerror or exc}"
            return
        threading.Thread(target=self._api_server.serve_forever, name="siphon-api", daemon=True).start()

    def _api_external(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self._api_port}/health", timeout=1) as response:
                return response.headers.get("Server", "").startswith("siphon-api")
        except (OSError, ValueError):
            return False

    def _api_stop(self) -> None:
        server, self._api_server = getattr(self, "_api_server", None), None
        if server is not None:
            threading.Thread(target=server.shutdown, name="siphon-api-stop", daemon=True).start()
            server.keys.flush()

    def _api_engines(self) -> dict[str, str | None]:
        """Profiles for the selected model: decode (its catalogue profile), decision-only, and decode+decision."""
        from ..api.gateway import backend_for
        engines: dict[str, str | None] = {"decode": None, "decision": None, "both": None}
        current = self._profile_for(self.model_key)
        if current is None or not current.resolve("MODEL"):
            return engines
        model = str(Path(current.resolve("MODEL")).resolve())
        engines["decode"] = current.name
        for profile in sorted(discover().values(), key=lambda p: p.name):
            if not profile.resolve("MODEL") or str(Path(profile.resolve("MODEL")).resolve()) != model:
                continue
            backend = backend_for(profile)
            mode = "both" if backend.decode and backend.decision else "decision" if backend.decision else ""
            if mode and engines[mode] is None:
                engines[mode] = profile.name
        if engines["decision"] is None:
            engines["both"] = None          # decisions are not set up for this model
        elif engines["both"] is None:
            engines["both"] = f"{current.name}-decide"
        return engines

    # Combined process = the Decode settings + a decision context. The expert cache is capped because the
    # decision compute buffer and the preloaded kernels must fit next to it (2816 MiB OOMs on a 6 GB GPU).
    _COMBINED_GPU_CACHE_MAX_MIB = 2560
    _DECISION_DEFAULTS = {"MOE_GPU_CACHE_MIB": "3456", "MOE_PINNED_CACHE_MIB": "5632", "LLAMA_ARG_DECISION_CTX": "512"}

    def _api_sync_combined(self, engines: dict[str, str | None]) -> Profile | None:
        decode = discover().get(engines.get("decode") or "")
        decision = discover().get(engines.get("decision") or "")
        if decode is None or decision is None or not engines.get("both"):
            return None
        env = {k: v for k, v in decode.env.items() if not k.startswith("CPU_")}
        try:
            gpu_cache = int(env.get("MOE_GPU_CACHE_MIB", "2816"))
        except ValueError:
            gpu_cache = self._COMBINED_GPU_CACHE_MAX_MIB
        env.update({
            "ALIAS": env.get("ALIAS", decode.name),
            "SIPHON_EXECUTION_MODE": "gpu",
            "MOE_GPU_CACHE_MIB": str(min(gpu_cache, self._COMBINED_GPU_CACHE_MAX_MIB)),
            "LLAMA_ARG_DECISION_CTX": decision.env.get("LLAMA_ARG_DECISION_CTX", "512"),
            "LLAMA_ARG_DECISION_UBATCH": decision.env.get("LLAMA_ARG_DECISION_UBATCH", "128"),
            "LLAMA_ARG_DECISION_CHUNK": decision.env.get("LLAMA_ARG_DECISION_CHUNK", "24"),
            "GGML_CUDA_PRELOAD_MMQ": "1",
        })
        if decision.env.get("SIPHON_SERVER_BIN"):
            env["SIPHON_SERVER_BIN"] = decision.env["SIPHON_SERVER_BIN"]
        env.pop("LLAMA_ARG_DECISION_ONLY", None)
        # Whole-layer prefill ubatches need a larger compute buffer than fits next to the decision context, so the
        # combined process keeps the clamped prefill and the ubatch of at most 128 tokens it was sized with.
        if env.pop("MOE_FULL_LAYER_UBATCH", "0") == "1":
            try:
                env["UBATCH"] = str(min(int(env.get("UBATCH", "128")), 128))
            except ValueError:
                env["UBATCH"] = "128"
        profile = Profile.create_app(engines["both"], env, script=decode.script_rel)
        profile.save()
        return profile

    def _api_decision_settings(self, engines: dict[str, str | None]) -> dict[str, Any] | None:
        profile = discover().get(engines.get("decision") or "")
        if profile is None:
            return None
        values = {key: profile.env.get(key, default) for key, default in self._DECISION_DEFAULTS.items()}
        return {"profile": profile.name, "vram_mib": int(values["MOE_GPU_CACHE_MIB"]),
                "ram_mib": int(values["MOE_PINNED_CACHE_MIB"]), "ctx": int(values["LLAMA_ARG_DECISION_CTX"]),
                "defaults": {"vram_mib": 3456, "ram_mib": 5632, "ctx": 512},
                "combined_gpu_cache_max_mib": self._COMBINED_GPU_CACHE_MAX_MIB}

    def _api_save_decision_settings(self, values: Any) -> None:
        engines = self._api_engines()
        profile = discover().get(engines.get("decision") or "")
        if profile is None or not isinstance(values, dict):
            self._emit_async({"type": "api_error", "message": "Decisions are not set up for this model."})
            return
        try:
            vram, ram, ctx = int(values.get("vram_mib")), int(values.get("ram_mib")), int(values.get("ctx"))
        except (TypeError, ValueError):
            self._emit_async({"type": "api_error", "message": "Decision settings must be whole numbers."})
            return
        problems = []
        if not 1024 <= vram <= 3456:
            problems.append("GPU memory must be 1024-3456 MiB (more runs out of GPU memory)")
        if not 1024 <= ram <= 8192:
            problems.append("RAM must be 1024-8192 MiB")
        if ctx not in (256, 512, 1024):
            problems.append("longest request must be 256, 512 or 1024 tokens")
        if problems:
            message = "; ".join(problems)
            self._emit_async({"type": "api_error", "message": message[0].upper() + message[1:] + "."})
            return
        profile.env.update({"MOE_GPU_CACHE_MIB": str(vram), "MOE_PINNED_CACHE_MIB": str(ram),
                            "LLAMA_ARG_DECISION_CTX": str(ctx)})
        profile.save()
        self._emit_async({"type": "api_decision_saved"})
        self._emit_async(self._api_state())

    def _api_state(self) -> dict[str, Any]:
        running = None
        try:
            from ..api.gateway import backend_for
            profiles = runner.running_profiles()
            if profiles:
                backend = backend_for(profiles[0])
                health = fetch_health(profiles[0].base_url, timeout=1.0)
                running = {"profile": backend.profile, "mode": backend.mode, "model": backend.alias,
                           "ready": health.ok}
        except Exception:
            pass
        gateway = "running" if self._api_server is not None else "error" if self._api_error else "external"
        engines = self._api_engines()
        decode_profile = discover().get(engines.get("decode") or "")
        return {
            "type": "api_state",
            "endpoint": f"http://127.0.0.1:{self._api_port}",
            "gateway": gateway,
            "error": self._api_error,
            "running": running,
            "engines": engines,
            "model": decode_profile.env.get("ALIAS", decode_profile.name) if decode_profile else None,
            "decision": self._api_decision_settings(engines),
            "busy": self._api_busy,
            "busy_mode": self._api_busy_mode,
            "keys": {scope: record.public(with_key=True) for scope, record in self._api_keys.all().items()},
            "key_file": self._api_key_file(),
            "apps": api_connect.status(),
            "activity": self._api_server.activity.snapshot() if self._api_server is not None else None,
            "trying": sorted(self._api_trying),
        }

    def _api_key_file(self) -> str:
        path, home = str(self._api_keys.path), str(Path.home())
        return "~" + path[len(home):] if path.startswith(home + os.sep) else path

    def _api_connect_target(self) -> "api_connect.Target":
        """What connected apps should point at: the gateway, the decode key, and the decode model's alias."""
        profile = discover().get(self._api_engines().get("decode") or "")
        model, context = "siphon", 65536
        if profile is not None:
            model = profile.env.get("ALIAS", profile.name)
            try:
                context = int(profile.env.get("CTX", context))
            except ValueError:
                pass
        return api_connect.Target(endpoint=f"http://127.0.0.1:{self._api_port}",
                                  key=self._api_keys.get("decode").key, model=model, context=context)

    def _api_connect_app(self, app_id: str, connect: bool) -> None:
        try:
            if connect:
                api_connect.connect(app_id, self._api_connect_target())
                self._emit_async({"type": "api_notice", "message": f"{api_connect.APPS[app_id]['name']} CONNECTED"})
            else:
                api_connect.disconnect(app_id)
                self._emit_async({"type": "api_notice", "message": f"{api_connect.APPS[app_id]['name']} DISCONNECTED"})
        except (api_connect.ConnectError, OSError) as exc:
            self._emit_async({"type": "api_error", "message": str(exc)})
        self._emit_async(self._api_state())

    def _api_emit_state(self) -> None:
        threading.Thread(target=lambda: self._emit_async(self._api_state()),
                         name="siphon-api-state", daemon=True).start()

    def _api_message(self, action: str, message: dict[str, Any]) -> None:
        if action == "api_state":
            self._api_emit_state()
        elif action == "api_regenerate_key":
            try:
                self._api_keys.regenerate(str(message.get("scope", "")))
            except ValueError as exc:
                self._emit_event({"type": "api_error", "message": str(exc)})
                return
            notice = "NEW KEY CREATED · UPDATE YOUR APPS"
            if message.get("scope") == "decode":
                try:
                    target = self._api_connect_target()
                    if api_connect.refresh(target.key, target):
                        notice = "NEW KEY CREATED · CONNECTED APPS UPDATED · UPDATE ANY OTHER APPS"
                except (api_connect.ConnectError, OSError) as exc:
                    self._emit_event({"type": "api_error", "message": f"Could not update connected apps: {exc}"})
            self._emit_event({"type": "api_notice", "message": notice})
            self._api_emit_state()
        elif action in {"api_copy", "api_copy_key"}:
            text = str(message.get("text", ""))
            if action == "api_copy_key":
                try:
                    text = self._api_keys.get(str(message.get("scope", ""))).key
                except ValueError:
                    return
            display = Gdk.Display.get_default()
            if display is not None:
                display.get_clipboard().set_content(Gdk.ContentProvider.new_for_value(text))
        elif action == "api_try":
            scope = str(message.get("scope", ""))
            with self._api_lock:
                if scope not in {"decode", "decision"} or scope in self._api_trying:
                    return
                self._api_trying.add(scope)
            threading.Thread(target=self._api_try, args=(scope, message.get("request")),
                             name="siphon-api-try", daemon=True).start()
        elif action in {"api_connect_app", "api_disconnect_app"}:
            threading.Thread(target=self._api_connect_app,
                             args=(str(message.get("app", "")), action == "api_connect_app"),
                             name="siphon-api-connect", daemon=True).start()
        elif action == "api_engine":
            self._api_switch_engine(str(message.get("mode", "")))
        elif action == "api_decision_settings":
            threading.Thread(target=self._api_save_decision_settings, args=(message.get("settings"),),
                             name="siphon-api-settings", daemon=True).start()

    def _api_try(self, scope: str, request: Any) -> None:
        """Send one request through the gateway with the path's own key (API tab test console / playground)."""
        request = request if isinstance(request, dict) else {}
        if scope == "decode":
            text = str(request.get("text") or "").strip()[:2000] or "Say hello in five words."
            route = "/v1/chat/completions"
            body: dict[str, Any] = {"messages": [{"role": "user", "content": text}], "max_tokens": 96,
                                    "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
        else:
            route = "/v1/decide"
            body = {k: request[k] for k in ("input", "question", "options", "context", "threshold") if k in request}
        event: dict[str, Any] = {"type": "api_try_result", "scope": scope, "route": route}
        t0 = time.perf_counter()
        try:
            key = self._api_keys.get(scope).key
            http_request = urllib.request.Request(
                f"http://127.0.0.1:{self._api_port}{route}", json.dumps(body).encode(),
                {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(http_request, timeout=900) as response:
                    status, data = response.status, json.loads(response.read() or b"{}")
            except urllib.error.HTTPError as exc:
                status = exc.code
                try:
                    data = json.loads(exc.read() or b"{}")
                except ValueError:
                    data = {}
            event.update(status=status, ok=200 <= status < 300, ms=(time.perf_counter() - t0) * 1000)
            if not event["ok"]:
                error = data.get("error") if isinstance(data, dict) else None
                event["error"] = (error.get("message") if isinstance(error, dict) else str(error or "")) or f"HTTP {status}"
            elif scope == "decode":
                choice = (data.get("choices") or [{}])[0].get("message") or {}
                usage, timings = data.get("usage") or {}, data.get("timings") or {}
                event["result"] = {"reply": str(choice.get("content") or "").strip(),
                                   "completion_tokens": usage.get("completion_tokens"),
                                   "prompt_tokens": usage.get("prompt_tokens"),
                                   "tokens_per_second": timings.get("predicted_per_second"),
                                   "model": data.get("model")}
            else:
                event["result"] = data
        except (urllib.error.URLError, OSError) as exc:
            event.update(status=0, ok=False, ms=(time.perf_counter() - t0) * 1000,
                         error=f"gateway unreachable: {getattr(exc, 'reason', exc)}")
        except Exception as exc:    # malformed reply: report it instead of leaving the button spinning
            event.update(status=0, ok=False, ms=(time.perf_counter() - t0) * 1000, error=f"unexpected reply: {exc}")
        finally:
            with self._api_lock:
                self._api_trying.discard(scope)
        self._emit_async(event)
        self._emit_async(self._api_state())

    def _api_switch_engine(self, mode: str) -> None:
        """Run exactly one engine profile for the selected model (or none for mode 'stop')."""
        self._ensure_shutdown_state()
        engines = self._api_engines()
        if mode != "stop" and not engines.get(mode):
            self._emit_event({"type": "api_error", "message": f"No {mode} profile is configured for this model."})
            return
        with self._api_lock:
            if self._api_busy:
                return
            labels = {"decode": "decode", "decision": "decision", "both": "decode + decision", "stop": ""}
            self._api_busy = f"Starting {labels.get(mode, mode)} engine…" if mode != "stop" else "Stopping engine…"
            self._api_busy_mode = mode
        self._api_emit_state()

        def work() -> None:
            ok, note = True, ""
            try:
                if mode == "both":
                    target = self._api_sync_combined(engines)
                else:
                    target = self.profiles.get(engines.get(mode) or "") or discover().get(engines.get(mode) or "")
                with self._launch_lock:
                    if self._shutdown_event.is_set():
                        return
                    for profile in runner.running_profiles():
                        if target is None or profile.name != target.name:
                            runner.down(profile, force=True)
                    if target is not None:
                        offset = _file_size(target.log_file)
                        ok, note = runner.up(target)
                if ok and target is not None:
                    healthy = runner.wait_healthy(target, timeout=600.0, interval=1.0,
                                                  cancel_event=self._shutdown_event)
                    failure = _startup_log_error(target.log_file, offset)
                    if not healthy or failure:
                        ok, note = False, failure or "engine did not become healthy"
                        runner.down(target, force=True)
            except Exception as exc:
                ok, note = False, f"Engine switch failed: {exc}"
            finally:
                with self._api_lock:
                    self._api_busy = ""
                    self._api_busy_mode = ""
            if self._shutdown_event.is_set():
                return
            if not ok:
                self._emit_async({"type": "api_error", "message": note})
            self._emit_async(self._api_state())

        threading.Thread(target=work, name="siphon-api-engine", daemon=True).start()

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
        self._ensure_catalogue_state()
        with self._catalogue_lock:
            if model_key in self._excluded_model_keys:
                return None
            profile_name = self.profile_by_model_key.get(model_key)
            if not profile_name:
                return None
            return self.profiles.get(profile_name)

    def _ensure_catalogue_state(self) -> None:
        """Supply catalogue state for older tests and bridge instances made via __new__."""
        self._ensure_shutdown_state()
        if not hasattr(self, "_catalogue_lock"):
            self._catalogue_lock = threading.RLock()
        if not hasattr(self, "_excluded_model_keys"):
            self._excluded_model_keys = set()
        if not hasattr(self, "_all_profile_by_model_key"):
            self._all_profile_by_model_key = dict(self.profile_by_model_key)
        if not hasattr(self, "_registration_generations"):
            self._registration_generations = {}
        if not hasattr(self, "_starting_models"):
            self._starting_models = set()
        if not hasattr(self, "_stopping_models"):
            self._stopping_models = set()

    def _visible_profiles(self) -> list[Profile]:
        self._ensure_catalogue_state()
        with self._catalogue_lock:
            return [
                self.profiles[profile_name]
                for model_key, profile_name in self.profile_by_model_key.items()
                if model_key not in self._excluded_model_keys
                and profile_name in self.profiles
            ]

    def _catalogue_totals(self) -> tuple[int, float]:
        profiles = self._visible_profiles()
        available_paths: set[str] = set()
        model_parts: set[Path] = set()
        for profile in profiles:
            try:
                parts = _profile_model_parts(profile)
                model_parts.update(path.resolve() for path in parts if path.is_file())
                if _profile_model_available(profile):
                    available_paths.add(str(Path(profile.resolve("MODEL")).resolve()))
            except (OSError, ValueError):
                continue
        storage_bytes = 0
        for path in model_parts:
            try:
                storage_bytes += path.stat().st_size
            except OSError:
                continue
        return len(available_paths), round(storage_bytes / (1024 ** 3), 2)

    def _remove_model(self, model_key: str, removal_id: str) -> None:
        self._ensure_catalogue_state()
        result = {
            "type": "model_removed",
            "model_key": model_key,
            "removal_id": removal_id,
            "ok": False,
            "message": "",
        }
        with self._catalogue_lock:
            profile = self._profile_for(model_key)
            if profile is None:
                result["message"] = f"No registered model matches {model_key}."
            elif profile.name in self._starting_models:
                result["message"] = "Wait for the model to finish starting before removing it."
            elif profile.name in self._stopping_models:
                result["message"] = "Wait for the model to finish stopping before removing it."
            else:
                try:
                    status = runner.status(profile, include_metrics=False)
                    if getattr(status, "running", False):
                        result["message"] = "Stop the model before removing it."
                    else:
                        exclusions = set(self._excluded_model_keys)
                        exclusions.add(model_key)
                        save_excluded_model_keys(exclusions)
                        self._excluded_model_keys = exclusions
                        self.profile_by_model_key.pop(model_key, None)
                        self._registration_generations[model_key] = (
                            self._registration_generations.get(model_key, 0) + 1
                        )
                        self._telemetry_epoch = getattr(self, "_telemetry_epoch", 0) + 1
                        result.update(ok=True, message="Model removed from Studio.")
                except Exception as exc:
                    result["message"] = f"Could not remove model: {exc}"

            self._emit_async(result)
            if result["ok"]:
                self._schedule_catalogue()

    def _schedule_catalogue(self) -> None:
        self._ensure_catalogue_state()
        with self._catalogue_lock:
            entries = [
                {"model_key": key, "profile": name,
                 "available": _profile_model_available(self.profiles[name])}
                for key, name in list(self.profile_by_model_key.items())
                if key not in self._excluded_model_keys
                and name in self.profiles and self.profiles[name].resolve("MODEL")
            ]
            library_count, library_storage_gib = self._catalogue_totals()
            self._emit_async({
                "type": "catalogue",
                "models": entries,
                "library_count": library_count,
                "library_storage_gib": library_storage_gib,
            })

    def _choose_model_folder(self, scan_id: str) -> bool:
        # Gtk.FileChooserNative on some Ubuntu GTK 4 builds corrupts its
        # internal file-system model when opened from a WebKit-hosted app and
        # ends in repeated GTK_IS_FILE_SYSTEM_MODEL assertions followed by a
        # segfault. Gtk.FileDialog is the supported GTK 4 asynchronous API and
        # keeps the chooser owned by the application for the whole operation.
        dialog = Gtk.FileDialog.new()
        dialog.set_title("Add a model folder")
        with self._import_lock:
            self._import_scan_id = scan_id
            self._import_candidates = {}

        def responded(file_dialog: Gtk.FileDialog, result: Any, _data: Any) -> None:
            try:
                selected = file_dialog.select_folder_finish(result)
                selected_path = selected.get_path() if selected else None
                if selected_path:
                    threading.Thread(
                        target=self._scan_model_folder,
                        args=(Path(selected_path), scan_id),
                        name="siphon-model-import",
                        daemon=True,
                    ).start()
                else:
                    self._finish_folder_scan(scan_id, "", [], "Choose a folder on the local filesystem.")
            except GLib.Error as exc:
                # Cancellation is normal; other chooser errors should be
                # reported in the UI rather than terminating the desktop host.
                if "cancel" in str(exc).lower() or "dismiss" in str(exc).lower():
                    self._cancel_model_import(scan_id)
                else:
                    self._finish_folder_scan(scan_id, "", [], f"Could not open folder chooser: {exc}")

        dialog.select_folder(self.window, None, responded, None)
        return GLib.SOURCE_REMOVE

    def _cancel_model_import(self, scan_id: str) -> None:
        with self._import_lock:
            if scan_id != self._import_scan_id:
                return
            self._import_scan_id = ""
            self._import_candidates = {}
        self._emit_async({"type": "model_folder_scan", "scan_id": scan_id,
                          "phase": "cancelled", "models": []})

    def _scan_model_folder(self, folder: Path, scan_id: str) -> None:
        with self._import_lock:
            if scan_id != self._import_scan_id or self._closed:
                return
        self._emit_async({"type": "model_folder_scan", "scan_id": scan_id,
                          "phase": "scanning", "folder": str(folder), "models": []})
        try:
            candidates = scan_model_folder(folder)
            self._finish_folder_scan(scan_id, str(folder), candidates)
        except Exception as exc:
            self._finish_folder_scan(scan_id, str(folder), [], f"Could not scan folder: {exc}")

    def _finish_folder_scan(self, scan_id: str, folder: str,
                            candidates: list[dict[str, Any]], error: str = "") -> None:
        with self._import_lock:
            if scan_id != self._import_scan_id or self._closed:
                return
            for entry in candidates:
                if entry.get("available"):
                    try:
                        entry["fingerprint"] = model_fingerprint([Path(p) for p in entry["parts"]])
                    except (OSError, ValueError) as exc:
                        entry.update(available=False, error=f"Files changed during scan: {exc}")
            self._import_candidates = {entry["id"]: entry for entry in candidates}
            self._emit_async({"type": "model_folder_scan", "scan_id": scan_id,
                              "phase": "error" if error else "ready", "folder": folder,
                              "models": candidates, "message": error})

    def _import_selected_models(self, scan_id: str, model_ids: Any) -> None:
        self._ensure_catalogue_state()
        with self._catalogue_lock:
            self._import_selected_models_locked(scan_id, model_ids)

    def _import_selected_models_locked(self, scan_id: str, model_ids: Any) -> None:
        results = []
        with self._import_lock:
            if scan_id != self._import_scan_id or not scan_id:
                results.append({"ok": False, "message": "Folder scan expired; choose the folder again."})
            elif not isinstance(model_ids, list) or not model_ids:
                results.append({"ok": False, "message": "Select at least one model."})
            else:
                for model_id in dict.fromkeys(str(value) for value in model_ids):
                    candidate = self._import_candidates.get(model_id)
                    if not candidate or not candidate.get("available"):
                        results.append({"id": model_id, "ok": False,
                                        "message": "This model is not a valid selection from the current scan."})
                        continue
                    try:
                        fingerprint = model_fingerprint([Path(p) for p in candidate["parts"]])
                        if candidate.get("fingerprint", fingerprint) != fingerprint:
                            raise ValueError("Model files changed after the preview; scan the folder again.")
                    except (OSError, ValueError) as exc:
                        results.append({"id": model_id, "ok": False, "message": str(exc)})
                        continue
                    result = self._import_model(Path(candidate["path"]), emit=False)
                    results.append({"id": model_id, **result})
                self._schedule_catalogue()
            self._emit_async({"type": "model_folder_imported", "scan_id": scan_id,
                              "results": results,
                              "message": f"{sum(bool(r['ok']) for r in results)} model(s) added or already registered."})

    def _import_model(self, model_path: Path, *, emit: bool = True) -> dict[str, Any]:
        self._ensure_catalogue_state()
        with self._catalogue_lock:
            return self._import_model_locked(model_path, emit=emit)

    def _import_model_locked(self, model_path: Path, *, emit: bool = True) -> dict[str, Any]:
        def finished(result: dict[str, Any]) -> dict[str, Any]:
            if emit:
                self._emit_async({"type": "model_imported", **result})
                if result.get("ok"):
                    self._schedule_catalogue()
            return result

        if model_path.suffix.lower() != ".gguf" or not model_path.is_file():
            return finished({"ok": False, "message": "Model file is missing or unreadable."})
        try:
            canonical_path = model_path.resolve()
            fingerprint = model_fingerprint(model_part_paths(canonical_path))
            metadata = inspect_model_files(canonical_path)
            if fingerprint != model_fingerprint([Path(p) for p in metadata["parts"]]):
                raise ValueError("Model files changed while importing; scan the folder again.")
            canonical_path = Path(metadata["parts"][0])
            full_profile_map = dict(self._all_profile_by_model_key)
            full_profile_map.update(self.profile_by_model_key)
            for existing_key, profile_name in full_profile_map.items():
                existing = self.profiles.get(profile_name)
                if existing and Path(existing.resolve("MODEL")).resolve() == canonical_path:
                    if existing_key in self._excluded_model_keys:
                        exclusions = set(self._excluded_model_keys)
                        exclusions.discard(existing_key)
                        save_excluded_model_keys(exclusions)
                        self._excluded_model_keys = exclusions
                    self._all_profile_by_model_key[existing_key] = profile_name
                    self.profile_by_model_key[existing_key] = profile_name
                    return finished({"ok": True, "model_key": existing_key,
                                     "message": "That model is registered again; saved settings preserved."})
            native_context = int(metadata["native_context"] or 32768)
            expert_count = metadata["expert_count"]
            expert_used = metadata["expert_used_count"]
            display_name = metadata["name"]
            if expert_count <= 0 or expert_used <= 0:
                raise ValueError(
                    f"{display_name} does not expose valid routed-MoE expert_count "
                    "and expert_used_count metadata"
                )
            system = fetch_sys_stats()
            resources = _resource_state(system)
            profile_name = f"imported-{hashlib.sha256(str(canonical_path).encode()).hexdigest()[:12]}"
            used_ports = {p.port for p in self.profiles.values()}
            port = next((candidate for candidate in range(8090, 65536) if candidate not in used_ports), None)
            if port is None:
                raise ValueError("No unused model-server port is available.")
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
            })
            profile.model_files = metadata["parts"]
            profile.model_folder = str(canonical_path.parent)
            profile.save()
            model_key = _model_key_for_profile(profile)
            if model_key in self._excluded_model_keys:
                exclusions = set(self._excluded_model_keys)
                exclusions.discard(model_key)
                save_excluded_model_keys(exclusions)
                self._excluded_model_keys = exclusions
            _write_metadata_cache({"path": str(canonical_path), **metadata, "fingerprint": fingerprint})
            self.profiles[profile.name] = profile
            self._all_profile_by_model_key[model_key] = profile.name
            self.profile_by_model_key[model_key] = profile.name
            if not hasattr(self, "_log_offsets"):
                self._log_offsets = {}
            self._log_offsets[profile.name] = _file_size(profile.log_file)
            self._schedule_model_info(model_key)
            return finished({"ok": True, "model_key": model_key,
                             "message": f"Added {display_name} ({metadata['part_count']} part(s)); runtime compatibility checked at startup."})
        except Exception as exc:
            return finished({"ok": False, "message": f"Could not add model: {exc}"})

    def _schedule_model_info(self, model_key: str) -> None:
        self._ensure_catalogue_state()
        with self._catalogue_lock:
            if model_key in self._excluded_model_keys:
                return
            profile = self._profile_for(model_key)
            if profile is None:
                self._emit_async({"type": "model_info", "model_key": model_key, "available": False})
                return
            registration_generation = self._registration_generations.get(model_key, 0)

            # Apply profile defaults before doing potentially slow GGUF work.
            model_path = Path(profile.resolve("MODEL"))
            self._emit_async({
                "type": "model_info",
                "phase": "profile",
                "model_key": model_key,
                "available": _profile_model_available(profile),
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
                payload["library_count"], payload["library_storage_gib"] = (
                    self._catalogue_totals()
                )
                # Re-read after the expensive scan too: settings may have
                # been saved while GGUF metadata was being inspected.
                latest_profile = self._profile_for(model_key) or current_profile
                payload["config"] = _profile_config(latest_profile)
                system = fetch_sys_stats()
                payload["memory_layout"] = _profile_memory_layout(latest_profile) or payload.get("memory_layout")
                payload["resources"] = _resource_state(system, payload["config"], payload.get("memory_layout"),
                                                       running=runner.status(latest_profile, include_metrics=False).running)
                payload["memory_layout"] = payload["resources"]["memory_layout"]
                payload["recommended"] = _recommended_settings(
                    payload["config"], system, payload.get("memory_layout")
                )
            except Exception as exc:  # metadata must never block the desktop shell
                payload = {"type": "model_info", "model_key": model_key,
                           "available": False, "metadata_error": str(exc)}
            GLib.idle_add(self._finish_model_info, payload, registration_generation)

        threading.Thread(target=work, name="siphon-model-metadata", daemon=True).start()

    def _finish_model_info(
        self, payload: dict[str, Any], registration_generation: int | None = None
    ) -> bool:
        self._ensure_catalogue_state()
        with self._catalogue_lock:
            if payload.get("model_key") in self._excluded_model_keys:
                return GLib.SOURCE_REMOVE
            # Re-adding clears an exclusion, but does not make work from the
            # previous registration current again.
            if (registration_generation is not None and registration_generation !=
                    self._registration_generations.get(payload.get("model_key"), 0)):
                return GLib.SOURCE_REMOVE
            payload["library_count"], payload["library_storage_gib"] = self._catalogue_totals()
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

        running = runner.status(profile, include_metrics=False).running
        current_config = _profile_config(profile)
        selected_mode = values.get("execution_mode", current_config["execution_mode"])
        if selected_mode not in {"cpu", "gpu"}:
            self._emit_async({"type": "settings_applied", "ok": False, "message": "Choose CPU or GPU execution."})
            return
        if running and selected_mode != current_config["execution_mode"]:
            self._emit_async({"type": "settings_applied", "ok": False, "message": "Stop the model before changing execution mode."})
            return
        candidate = Profile(name=profile.name, env={**profile.env, "SIPHON_EXECUTION_MODE": selected_mode}, script=profile.script)
        selected_config = _profile_config(candidate)
        cpu_mode = selected_mode == "cpu"
        cpu_keys = {"CTX", "BATCH", "UBATCH", "CACHE_TYPE", "CACHE_RAM_MIB", "SPEC_TYPE", "SPEC_DRAFT_N_MAX", "SPEC_DRAFT_P_MIN", "THREADS", "MODEL_DRAFT"}
        def setting_key(key: str) -> str:
            return "CPU_" + key if cpu_mode and key in cpu_keys else key
        current_cache = {
            "vram_cache": selected_config["l1_cache"],
            "pinned_cache": selected_config["l2_cache"],
            "cache_ram": selected_config["cache_ram"],
        }
        try:
            requested_cache = {
                name: max(0, int(values[name] if name in values else current))
                for name, current in current_cache.items()
            }
            if cpu_mode:
                requested_cache.update(vram_cache=0, pinned_cache=0)
        except (TypeError, ValueError):
            self._emit_async({
                "type": "settings_applied",
                "ok": False,
                "message": "Memory allocations must be valid numbers.",
            })
            return
        if running and requested_cache != current_cache:
            self._emit_async({
                "type": "settings_applied",
                "ok": False,
                "message": "Stop the model before changing memory allocations.",
            })
            return

        resources_ok, resource_message = _validate_resource_settings(
            {**selected_config, **values, **requested_cache, "execution_mode": selected_mode},
            fetch_sys_stats(),
            enforce_memory_capacity=not running,
            memory_layout=_profile_memory_layout(profile, inspect=True),
            require_model_budget=_profile_model_available(profile),
        )
        if not resources_ok:
            self._emit_async({
                "type": "settings_applied",
                "ok": False,
                "message": resource_message,
            })
            return

        # Work on a copy so a later validation/save failure cannot alter the
        # in-memory profile or an unrelated execution mode.
        original_profile = profile
        profile = Profile(name=profile.name, env=dict(profile.env), script=profile.script,
                          source=profile.source, path=profile.path,
                          model_files=list(profile.model_files), model_folder=profile.model_folder)

        def int_setting(name: str, env_key: str, minimum: int = 0) -> None:
            raw = values.get(name)
            try:
                value = max(minimum, int(raw))
            except (TypeError, ValueError):
                return
            profile.env[setting_key(env_key)] = str(value)

        int_setting("context", "CTX", 1024)
        int_setting("batch", "BATCH", 1)
        int_setting("microbatch", "UBATCH", 1)
        if not cpu_mode:
            int_setting("gpu_layers", "GPU_LAYERS", 0)
            int_setting("vram_cache", "MOE_GPU_CACHE_MIB", 0)
            int_setting("pinned_cache", "MOE_PINNED_CACHE_MIB", 0)
        int_setting("cache_ram", "CACHE_RAM_MIB", 0)
        int_setting("cpu_threads", "THREADS", 1)
        if not cpu_mode:
            int_setting("gds_threads", "GDS_READ_THREADS", 1)
        int_setting("draft_tokens", "SPEC_DRAFT_N_MAX", 0)
        int_setting("checkpoints", "LLAMA_ARG_CTX_CHECKPOINTS", 0)
        int_setting("promotion_hits", "LLAMA_DSTORAGE_EVICT_TO_RAM_MIN_DECODE_HITS", 1)
        int_setting("expert_k", "LLAMA_MOE_EXPERT_USED_OVERRIDE", 1)
        int_setting("reasoning_budget", "REASONING_BUDGET", -1)

        cache_type = str(values.get("cache_type", "")).strip().lower()
        if cache_type in {"f32", "f16", "bf16", "q8_0", "q6_k", "q5_k", "q4_k", "q3_k", "tq1_0", "tq2_0", "tq3_0"}:
            profile.env[setting_key("CACHE_TYPE")] = cache_type
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
            profile.env[setting_key("SPEC_TYPE")] = spec_type
        draft_model = str(values.get("draft_model", "")).strip()
        if draft_model in {"Native MTP / NextN", "Disabled", ""}:
            profile.env.pop(setting_key("MODEL_DRAFT"), None)
            if draft_model == "Disabled":
                profile.env[setting_key("SPEC_TYPE")] = "none"
        elif draft_model == "Qwen3 0.6B Q8":
            candidate = profile.env.get(setting_key("MODEL_DRAFT"), "")
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
            profile.env[setting_key("MODEL_DRAFT")] = candidate
        for name, env_key in (("draft_probability", "SPEC_DRAFT_P_MIN"),
                              ("temperature", "SIPHON_TEMPERATURE"),
                              ("top_p", "SIPHON_TOP_P")):
            raw = values.get(name)
            try:
                profile.env[setting_key(env_key)] = str(float(raw))
            except (TypeError, ValueError):
                pass
        try:
            profile.env["SIPHON_TOP_K"] = str(max(0, int(values.get("top_k"))))
        except (TypeError, ValueError):
            pass

        if "evict_to_ram" in values and not cpu_mode:
            profile.env["LLAMA_DSTORAGE_EVICT_TO_RAM"] = (
                "1" if bool(values["evict_to_ram"]) else "0"
            )
        if "prompt_cache" in values:
            profile.env["CACHE_PROMPT"] = (
                "1" if bool(values["prompt_cache"]) else "0"
            )

        try:
            if "execution_mode" in values:
                profile.env["SIPHON_EXECUTION_MODE"] = selected_mode
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
            original_profile.env = dict(profile.env)
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
        if self._closed:
            return
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
        self._ensure_catalogue_state()
        if getattr(self, "_closed", False):
            return
        with self._catalogue_lock:
            profile = self._profile_for(model_key)
            if profile is None:
                self._emit_event({
                    "type": "runtime_complete",
                    "starting": starting,
                    "ok": False,
                    "message": f"No launcher profile is configured for {model_key}.",
                })
                return

            if starting:
                # Reserve startup before spawning work so removal cannot race it.
                self._starting_models.add(profile.name)
            else:
                self._stopping_models.add(profile.name)
        launched = False

        def run_action() -> None:
            nonlocal launched
            if self._shutdown_event.is_set():
                return
            if starting:
                config = _profile_config(profile)
                resources_ok, resource_message = _validate_resource_settings(
                    config, fetch_sys_stats(), memory_layout=_profile_memory_layout(profile, inspect=True),
                    require_model_budget=_profile_model_available(profile),
                )
                if not resources_ok:
                    GLib.idle_add(
                        self._finish_model_action,
                        starting,
                        False,
                        resource_message,
                        profile.name,
                    )
                    return
                # Keep startup output available for failure detection. Only
                # output after the readiness probe belongs to user telemetry.
                self._log_offsets[profile.name] = _file_size(profile.log_file)
                self._request_metrics.pop(profile.name, None)
                self._active_generations[profile.name] = False
                self._dstorage_baselines.pop(profile.name, None)
                self._live_timing_samples.pop(profile.name, None)
                self._live_stream_metrics.pop(profile.name, None)
                self._dstorage_rate_samples.pop(profile.name, None)
                with self._launch_lock:
                    if self._shutdown_event.is_set():
                        return
                    ok, message = runner.up(profile)
                if self._shutdown_event.is_set():
                    return
                if ok:
                    launched = True
                    startup_timeout = 600.0
                    healthy = runner.wait_healthy(
                        profile, timeout=startup_timeout, interval=1.0,
                        cancel_event=self._shutdown_event,
                    )
                    if self._shutdown_event.is_set():
                        return
                    probe_error = _probe_inference(profile) if healthy else None
                    if self._shutdown_event.is_set():
                        return
                    startup_error = _startup_log_error(
                        profile.log_file, self._log_offsets[profile.name]
                    )
                    if healthy and probe_error is None and startup_error is None:
                        _record_runtime_memory(profile, config, self._log_offsets[profile.name])
                        message = f"{message}; model health is ready"
                    else:
                        ok = False
                        failure = startup_error or probe_error
                        still_running = (
                            not healthy
                            and failure is None
                            and runner.status(profile, include_metrics=False).running
                        )
                        runner.down(profile, force=True)
                        if failure:
                            message = f"{message}; model startup failed: {failure}"
                        elif still_running:
                            timeout_minutes = int(startup_timeout // 60)
                            message = (
                                f"{message}; model did not become healthy within "
                                f"{timeout_minutes} minutes"
                            )
                        else:
                            message = (
                                f"{message}; model process exited before becoming healthy"
                            )
                    self._log_offsets[profile.name] = _file_size(profile.log_file)
            else:
                # A visible Stop action is an explicit user request; allow it
                # to clean up a process even if /health has already degraded.
                self._request_metrics.pop(profile.name, None)
                self._active_generations[profile.name] = False
                self._dstorage_baselines.pop(profile.name, None)
                self._live_timing_samples.pop(profile.name, None)
                self._live_stream_metrics.pop(profile.name, None)
                self._dstorage_rate_samples.pop(profile.name, None)
                with self._launch_lock:
                    if self._shutdown_event.is_set():
                        return
                    ok, message = runner.down(profile, force=True)
            GLib.idle_add(
                self._finish_model_action,
                starting,
                ok,
                message,
                profile.name,
            )

        def work() -> None:
            try:
                run_action()
            except Exception as exc:
                if self._shutdown_event.is_set():
                    return
                message = f"Model action failed: {exc}"
                if starting and launched:
                    try:
                        runner.down(profile, force=True)
                    except Exception as cleanup_error:
                        message += f"; model cleanup failed: {cleanup_error}"
                    # Even a failed readiness probe is not a user response.
                    self._log_offsets[profile.name] = _file_size(profile.log_file)
                GLib.idle_add(
                    self._finish_model_action, starting, False,
                    message, profile.name,
                )

        thread = threading.Thread(target=work, name="siphon-model-action", daemon=True)
        with self._catalogue_lock:
            self._model_action_threads = [t for t in self._model_action_threads if t.is_alive()]
            self._model_action_threads.append(thread)
        thread.start()

    def _finish_model_action(
        self, starting: bool, ok: bool, message: str, profile_name: str
    ) -> bool:
        self._ensure_catalogue_state()
        with self._catalogue_lock:
            self._starting_models.discard(profile_name)
            self._stopping_models.discard(profile_name)
            # Discard any poll that began during startup/stop and finishes later.
            self._telemetry_epoch += 1
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
        self._ensure_catalogue_state()
        with self._catalogue_lock:
            profile = self._profile_for(self.model_key)
            if profile is None:
                self._emit_event({"type": "telemetry", "running": False})
                return GLib.SOURCE_CONTINUE
            self._poll_busy = True
            telemetry_epoch = self._telemetry_epoch

        def work() -> None:
            try:
                status = runner.status(profile, include_metrics=False)
                request_ready = profile.name not in self._starting_models
                # Prometheus still supplies health/slot metadata. Throughput
                # comes from request-scoped live counters below, never from
                # its last-completed-request throughput gauges.
                metrics = fetch_metrics(profile.base_url) if status.running else status.metrics
                system = fetch_sys_stats()
                config = _profile_config(profile)
                layout = _profile_memory_layout(profile)
                resources = _resource_state(system, config, layout, running=status.running)
                slots = (
                    fetch_slots(profile.base_url)
                    if status.running and request_ready else None
                )
                dstorage_raw = (
                    fetch_dstorage_metrics(profile.base_url)
                    if status.running and request_ready else {}
                )
                live_timing = _live_engine_metrics(
                    dstorage_raw,
                    self._live_timing_samples.setdefault(profile.name, {}),
                )
                live_stream = (
                    self._live_stream_metrics.get(profile.name, {})
                    if request_ready else {}
                )
                dstore = (
                    parse_log_tail(
                        profile.log_file,
                        start_offset=self._log_offsets.get(profile.name, 0),
                    )
                    if status.running and request_ready else None
                )
                request_active = request_ready and bool(
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
                cached_request = (
                    self._request_metrics.get(profile.name, {})
                    if request_ready else {}
                )
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
                    "n_processing": (
                        (slots.busy if slots else metrics.n_processing)
                        if request_ready else 0
                    ),
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
                    **resources,
                    "ram_headroom_mib": RAM_HEADROOM_MIB,
                    "vram_headroom_mib": VRAM_HEADROOM_MIB,
                    "cpu_threads": _resource_state(system)["cpu_threads"],
                    "gds_threads_max": _resource_state(system)["gds_threads_max"],
                    "recommended": _recommended_settings(
                        config, system, layout
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
        self._ensure_catalogue_state()
        with self._catalogue_lock:
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
        self._closing = False

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
        if self.bridge is None:
            return False
        if not self._closing:
            self._closing = True
            _window.set_sensitive(False)
            _window.set_title("Siphon Studio — Stopping models…")
            self.bridge.close(self._finish_close)
        # Keep GTK alive until model/launcher cleanup is verified complete.
        return True

    def _finish_close(self, errors: list[str]) -> bool:
        if self.window is not None:
            if errors:
                self._closing = False
                self.window.set_sensitive(True)
                self.window.set_title("Siphon Studio — Model shutdown failed; close again to retry")
            else:
                self.window.destroy()
        return GLib.SOURCE_REMOVE

    def do_shutdown(self) -> None:
        try:
            if self.bridge is not None:
                self.bridge.close()
                self.bridge.wait_closed()
        finally:
            Gtk.Application.do_shutdown(self)


def run() -> int:
    """Launch the native desktop application."""
    if not Gtk.init_check() or Gdk.Display.get_default() is None:
        print(
            "Siphon Studio needs an active graphical display (GTK4 could not initialize).",
            flush=True,
        )
        return 3
    app = SiphonApplication()
    try:
        return app.run(None)
    finally:
        if app.bridge is not None:
            app.bridge.close()
            app.bridge.wait_closed()
