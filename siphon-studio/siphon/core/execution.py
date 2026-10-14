"""Resolve a Studio profile to its hardware-defaulted CPU or GPU mode."""

from __future__ import annotations

from functools import lru_cache
import math
import os
from pathlib import Path
import subprocess
from typing import Any

from .. import paths


CPU_CACHE_TYPES = frozenset({"q8_0", "f16"})
CPU_SPEC_TYPES = frozenset({
    "none", "off", "draft-simple", "draft-eagle3", "draft-mtp", "draft-dflash",
    "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod", "ngram-cache",
})
_GPU_ENV_PREFIXES = (
    "CUDA_", "NVIDIA_", "GGML_CUDA_", "LLAMA_DSTORAGE_", "LLAMA_GDS_",
    "LLAMA_MOE_PINNED_", "CUFILE_",
)
_GPU_ENV_KEYS = {
    "GDS_READ_THREADS", "MOE_GPU_CACHE_MIB", "MOE_PINNED_CACHE_MIB",
    "RUNTIME_MODE", "DSTORAGE_CAPABILITY", "EXPERTS", "GGML_CPU_BATCHED_EXPERTS",
}
_SAFE_PROFILE_LLAMA_ARGS = {"LLAMA_ARG_CTX_CHECKPOINTS"}


@lru_cache(maxsize=1)
def gpu_available() -> bool:
    """Return whether nvidia-smi can identify a CUDA-capable device."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,   # a laptop GPU waking from runtime suspend can take >2 s
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def mode(profile: Any) -> str:
    """Select the explicit profile mode, or default to available hardware."""
    profile_env = getattr(profile, "env", {})
    value = profile_env.get("SIPHON_EXECUTION_MODE")
    if not value:
        value = os.environ.get("SIPHON_EXECUTION_MODE", "auto")
    selected = str(value).strip().lower() or "auto"
    if selected not in {"auto", "gpu", "cpu"}:
        raise ValueError(
            "SIPHON_EXECUTION_MODE must be one of auto, gpu, or cpu "
            f"(got {value!r})"
        )
    if selected == "auto":
        return "gpu" if gpu_available() else "cpu"
    return selected


def default_cpu_threads() -> int:
    """Return a conservative CPU thread default capped by logical CPUs and eight."""
    return min(8, max(1, int(os.cpu_count() or 1)))


def _number(profile: Any, key: str, default: int) -> int:
    try:
        return int(profile.env.get(key, "") or default)
    except (TypeError, ValueError):
        return default


def effective_env(profile: Any) -> dict[str, str]:
    """Project separately stored CPU settings into a transient launch config."""
    if mode(profile) == "gpu":
        return dict(profile.env)
    env = {str(key): str(value) for key, value in profile.env.items()}
    gpu_context = _number(profile, "CTX", 65536)
    if gpu_context <= 0:
        gpu_context = 65536
    env.update({
        "CTX": str(_number(profile, "CPU_CTX", min(gpu_context, 65536))),
        "BATCH": str(_number(profile, "CPU_BATCH", 512)),
        "UBATCH": str(_number(profile, "CPU_UBATCH", 64)),
        "CACHE_TYPE": str(env.get("CPU_CACHE_TYPE", "q8_0") or "q8_0").lower(),
        "SPEC_TYPE": str(env.get("CPU_SPEC_TYPE", "none") or "none").lower(),
        "SPEC_DRAFT_N_MAX": str(_number(profile, "CPU_SPEC_DRAFT_N_MAX", 3)),
        "SPEC_DRAFT_P_MIN": env.get("CPU_SPEC_DRAFT_P_MIN", "0") or "0",
        "CACHE_RAM_MIB": str(_number(profile, "CPU_CACHE_RAM_MIB", 0)),
        "CPU_THREADS": str(_number(profile, "CPU_THREADS", default_cpu_threads())),
        "THREADS": str(_number(profile, "CPU_THREADS", default_cpu_threads())),
        "GPU_LAYERS": "0",
        "MOE_GPU_CACHE_MIB": "0",
        "MOE_PINNED_CACHE_MIB": "0",
        "MODEL_DRAFT": env.get("CPU_MODEL_DRAFT", "") or "",
        "LLAMA_DSTORAGE_EVICT_TO_RAM": "0",
    })
    return env


def validate_cpu_settings(profile: Any) -> list[str]:
    """Return CPU-only configuration errors before resolving or starting a server."""
    env = profile.env
    issues: list[str] = []
    gpu_context = _number(profile, "CTX", 65536)
    if gpu_context <= 0:
        gpu_context = 65536
    integer_fields = {
        "CPU_CTX": (min(gpu_context, 65536), 1),
        "CPU_BATCH": (512, 1),
        "CPU_UBATCH": (64, 1),
        "CPU_SPEC_DRAFT_N_MAX": (3, 1),
        "CPU_CACHE_RAM_MIB": (0, 0),
        "CPU_THREADS": (default_cpu_threads(), 1),
    }
    parsed: dict[str, int] = {}
    for key, (default, minimum) in integer_fields.items():
        if key not in env or env[key] == "":
            parsed[key] = default
            continue
        raw = env[key]
        if not isinstance(raw, str):
            issues.append(f"{key} must be an integer string")
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            issues.append(f"{key} must be an integer: {raw!r}")
            continue
        if value < minimum:
            issues.append(f"{key} must be at least {minimum}")
        parsed[key] = value

    if parsed.get("CPU_UBATCH", 1) > parsed.get("CPU_BATCH", 0):
        issues.append("CPU_UBATCH must not exceed CPU_BATCH")

    cache_type = str(env.get("CPU_CACHE_TYPE", "q8_0") or "q8_0").lower()
    if cache_type not in CPU_CACHE_TYPES:
        issues.append(f"unsupported CPU_CACHE_TYPE: {cache_type}")
    spec_type = str(env.get("CPU_SPEC_TYPE", "none") or "none").lower()
    if spec_type not in CPU_SPEC_TYPES:
        issues.append(f"unsupported CPU_SPEC_TYPE: {spec_type}")
    raw_probability = env.get("CPU_SPEC_DRAFT_P_MIN", "0")
    if raw_probability == "":
        raw_probability = "0"
    try:
        if not isinstance(raw_probability, str):
            raise ValueError
        probability = float(raw_probability)
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError
    except (TypeError, ValueError):
        issues.append("CPU_SPEC_DRAFT_P_MIN must be a finite number from 0 to 1")

    raw_draft = env.get("CPU_MODEL_DRAFT", "") or ""
    needs_external = spec_type in {"draft-simple", "draft-eagle3", "draft-dflash"}
    if needs_external and not isinstance(raw_draft, str):
        issues.append("CPU_MODEL_DRAFT must be a path string")
    elif needs_external and not raw_draft.strip():
        issues.append(f"CPU_SPEC_TYPE={spec_type} requires CPU_MODEL_DRAFT")
    elif needs_external:
        draft_path = Path(profile.resolve("CPU_MODEL_DRAFT"))
        if not draft_path.is_file():
            issues.append(f"CPU_MODEL_DRAFT not found: {draft_path}")
    return issues


def _model_path(profile: Any) -> Path:
    from .model_files import model_part_paths

    raw_model = profile.resolve("MODEL")
    if not raw_model:
        raise ValueError("MODEL is not set")
    parts = model_part_paths(Path(raw_model).expanduser())
    for part in parts:
        if not part.is_file():
            raise FileNotFoundError(f"GGUF model part not found: {part}")
    return parts[0]


def _launch_environment(effective: dict[str, str]) -> dict[str, str]:
    env = {
        key: value for key, value in os.environ.items()
        if not key.startswith(_GPU_ENV_PREFIXES)
        and key not in _GPU_ENV_KEYS
        and not key.startswith("LLAMA_ARG_")
    }
    for key, value in effective.items():
        if key.startswith(_GPU_ENV_PREFIXES) or key in _GPU_ENV_KEYS:
            continue
        if key.startswith("LLAMA_ARG_") and key not in _SAFE_PROFILE_LLAMA_ARGS:
            continue
        env[key] = value
    env.update({
        "LLAMA_ARG_REPACK": "0",
        "LLAMA_ARG_FIT": "off",
        "LLAMA_MMAP_PREFETCH": "0",
    })
    return env


def preview_environment(profile: Any) -> dict[str, str]:
    """Return the profile-controlled CPU environment after launch sanitization."""
    effective = effective_env(profile)
    launch_env = _launch_environment(effective)
    shown_keys = dict.fromkeys((*effective, "LLAMA_ARG_REPACK", "LLAMA_ARG_FIT",
                                "LLAMA_MMAP_PREFETCH"))
    return {key: launch_env[key] for key in shown_keys if key in launch_env}


def cpu_launch(profile: Any) -> tuple[list[str], dict[str, str]]:
    """Build a direct CPU llama-server command and its isolated environment."""
    if mode(profile) != "cpu":
        raise ValueError("CPU launch requires CPU execution mode")
    issues = validate_cpu_settings(profile)
    if issues:
        raise ValueError("invalid CPU settings: " + "; ".join(issues))
    binary = paths.cpu_server_bin()
    config = effective_env(profile)
    model = _model_path(profile)
    flash = str(config.get("FLASH_ATTENTION", "1")).strip().lower()
    flash_value = "off" if flash in {"0", "off", "false", "no"} else "on"
    command = [
        str(binary), "-m", str(model), "-ngl", "0", "--device", "none",
        "-c", config["CTX"], "-b", config["BATCH"], "-ub", config["UBATCH"],
        "-t", config["CPU_THREADS"], "-ctk", config["CACHE_TYPE"],
        "-ctv", config["CACHE_TYPE"], "-fa", flash_value,
        "--spec-type", config["SPEC_TYPE"],
        "--spec-draft-n-max", config["SPEC_DRAFT_N_MAX"],
        "--spec-draft-p-min", config["SPEC_DRAFT_P_MIN"],
        "--cache-ram", config["CACHE_RAM_MIB"],
        "--checkpoint-min-step", config.get("CHECKPOINT_MIN_STEP", "0") or "0",
        "--reasoning", config.get("REASONING", "off") or "off",
        "--reasoning-format", config.get("REASONING_FORMAT", "deepseek") or "deepseek",
        "--reasoning-budget", config.get("REASONING_BUDGET", "-1") or "-1",
        "--cache-reuse", config.get("CACHE_REUSE", "0") or "0",
        "--alias", config.get("ALIAS", profile.name) or profile.name,
        "--host", config.get("HOST", "127.0.0.1") or "127.0.0.1",
        "--port", config.get("PORT", "8090") or "8090",
        "--parallel", config.get("N_PARALLEL", "1") or "1",
        "--jinja", "--no-warmup",
    ]
    if (config.get("N_PARALLEL", "1") or "1").strip() == "1":
        command.append("--no-cont-batching")
    if (config.get("METRICS", "1") or "1") != "0":
        command.append("--metrics")
    if (config.get("CACHE_PROMPT", "1") or "1") != "0":
        command.append("--cache-prompt")
    if config.get("ROPE_SCALING") not in (None, "", "auto"):
        command.extend(("--rope-scaling", config["ROPE_SCALING"]))
    if config.get("MMPROJ"):
        command.extend(("--mmproj", profile.resolve("MMPROJ")))
    if config.get("CHAT_TEMPLATE"):
        command.extend(("--chat-template", config["CHAT_TEMPLATE"]))
    if config.get("CHAT_TEMPLATE_KWARGS"):
        command.extend(("--chat-template-kwargs", config["CHAT_TEMPLATE_KWARGS"]))
    draft = str(config.get("MODEL_DRAFT", "") or "").strip()
    if draft and config["SPEC_TYPE"] not in {"none", "off", "draft-mtp"}:
        command.extend(("--model-draft", profile.resolve("CPU_MODEL_DRAFT"), "--spec-draft-ngl", "0"))
    return command, _launch_environment(config)
