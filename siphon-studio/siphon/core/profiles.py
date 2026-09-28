"""Profile model: the unit of "how to run a model" for siphon-studio.

A profile is one launch configuration = a set of env keys (the same ones in
configs/*.env) + the launcher script to run. Two sources:

* bundled  - read-only profiles imported 1:1 from siphon.cpp/configs/*.env
* app      - editable JSON profiles stored under siphon-studio/profiles/

Verified on hardware (2026-09-01):
  - launch path is CONFIG=<env file> bash scripts/run-<model>.sh (byte-exact)
  - Qwen3.6 20.3 t/s decode, KAT 15.3 t/s, Qwen3CoderNext 4.2 t/s
  - /metrics names: llamacpp:{predicted_tokens_seconds,prompt_tokens_seconds,n_decode_total,...}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .. import paths
from .env import read_env, format_env


# Well-known config knobs (labels + hints) surfaced in the TUI/help.
KNOB_LABELS: dict[str, tuple[str, str]] = {
    "MODEL": ("Model GGUF", "path to the target model (absolute or repo-relative)"),
    "EXPERTS": ("Expert suite", "path to a moe-static-experts-*.txt suite"),
    "ALIAS": ("Model alias", "name reported by the OpenAI API"),
    "HOST": ("Host", "bind address (default 127.0.0.1)"),
    "PORT": ("Port", "HTTP port for the server + web UI"),
    "CTX": ("Context", "KV cache context size"),
    "BATCH": ("Batch", "max tokens per batch"),
    "UBATCH": ("UBatch", "micro-batch size"),
    "MOE_GPU_CACHE_MIB": ("VRAM expert cache", "L1 hot-expert cache in MiB"),
    "MOE_PINNED_CACHE_MIB": ("Pinned cache", "L2 static-pinned cache in MiB"),
    "CACHE_RAM_MIB": ("Cache RAM", "prompt/radix cache RAM in MiB"),
    "GDS_READ_THREADS": ("GDS threads", "cuFile read worker threads"),
    "SPEC_TYPE": ("Speculator", "draft-mtp | draft-simple | draft-dflash | none | off"),
    "SPEC_DRAFT_N_MAX": ("Draft n_max", "max speculative tokens per step"),
    "SPEC_DRAFT_P_MIN": ("Draft p_min", "min draft probability"),
    "CHECKPOINT_MIN_STEP": ("Checkpoint step", "min spacing between checkpoints"),
    "CHAT_TEMPLATE": ("Chat template", "template name (empty = embedded)"),
    "CHAT_TEMPLATE_KWARGS": ("Template kwargs", 'e.g. {"preserve_thinking": true}'),
    "REASONING": ("Reasoning", "off | on"),
    "REASONING_FORMAT": ("Reasoning format", "deepseek | ..."),
    "REASONING_BUDGET": ("Reasoning budget", "-1 = unlimited"),
    "CACHE_REUSE": ("Cache reuse", "KV-shift reuse (0 = off with MTP)"),
    "METRICS": ("Metrics", "1 = expose /metrics"),
    "N_PARALLEL": ("Parallel slots", "concurrent slots"),
    "MMPROJ": ("Vision mmproj", "optional vision encoder GGUF"),
}

DEFAULT_KEYS: list[str] = [
    "MODEL", "EXPERTS", "ALIAS", "HOST", "PORT", "CTX", "BATCH", "UBATCH",
    "MOE_GPU_CACHE_MIB", "MOE_PINNED_CACHE_MIB", "CACHE_RAM_MIB",
    "GDS_READ_THREADS", "SPEC_DRAFT_N_MAX", "SPEC_DRAFT_P_MIN",
    "CHECKPOINT_MIN_STEP", "CHAT_TEMPLATE", "REASONING", "REASONING_FORMAT",
    "REASONING_BUDGET", "CACHE_REUSE", "METRICS", "CHAT_TEMPLATE_KWARGS",
    "N_PARALLEL", "SPEC_TYPE", "MMPROJ",
]


@dataclass
class Profile:
    name: str
    env: dict[str, str] = field(default_factory=dict)
    script: str = ""                       # repo-relative launcher path
    source: str = "app"                    # "bundled" | "app"
    path: Path | None = None               # backing file

    @property
    def is_bundled(self) -> bool:
        return self.source == "bundled"

    @property
    def script_rel(self) -> str:
        return self.script or paths.default_run_script()

    @property
    def host(self) -> str:
        return self.env.get("HOST", "127.0.0.1")

    @property
    def port(self) -> int:
        try:
            return int(self.env.get("PORT", "8090"))
        except ValueError:
            return 8090

    @property
    def model(self) -> str:
        return self.env.get("MODEL", "")

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def log_file(self) -> Path:
        return paths.logs_dir() / f"{self.name}.log"

    # -- serialization ---------------------------------------------------
    def to_dict(self) -> dict:
        return {"name": self.name, "script": self.script, "env": self.env}

    def save(self) -> "Profile":
        if self.is_bundled:
            raise ValueError("bundled profiles are read-only; clone with 'profile new --from'")
        self.path = paths.profiles_dir() / f"{self.name}.json"
        self.path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return self

    # -- constructors ----------------------------------------------------
    @classmethod
    def from_env_file(cls, path: Path, script: str = "") -> "Profile":
        env = read_env(path)
        name = path.stem
        if not script:
            script = infer_script_for(name)
        return cls(name=name, env=env, script=script,
                   source="bundled", path=path)

    @classmethod
    def from_json(cls, path: Path) -> "Profile":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(name=data["name"], env=data.get("env", {}),
                   script=data.get("script", paths.default_run_script()),
                   source="app", path=path)

    @classmethod
    def create_app(cls, name: str, env: dict[str, str], script: str = "") -> "Profile":
        return cls(name=name, env=dict(env),
                   script=script or paths.default_run_script(), source="app")

    # -- resolution / validation ------------------------------------------
    def resolve(self, key: str) -> str:
        from .env import resolve_value
        return resolve_value(self.env.get(key, ""), paths.repo_root())

    def validate(self) -> list[str]:
        """Return a list of problems (empty = healthy)."""
        issues: list[str] = []
        root = paths.repo_root()

        model = self.resolve("MODEL")
        if not model:
            issues.append("MODEL is not set")
        elif not Path(model).is_file():
            issues.append(f"MODEL not found: {model}")

        experts = self.resolve("EXPERTS")
        if experts and not Path(experts).is_file():
            issues.append(f"EXPERTS suite not found: {experts}")

        script = root / self.script_rel
        if not script.exists():
            issues.append(f"launcher script not found: {self.script_rel}")

        for key in ("PORT", "CTX", "BATCH", "UBATCH", "MOE_GPU_CACHE_MIB",
                    "MOE_PINNED_CACHE_MIB", "CACHE_RAM_MIB", "GDS_READ_THREADS"):
            raw = self.env.get(key, "")
            if raw:
                try:
                    int(raw)
                except ValueError:
                    issues.append(f"{key} is not an integer: {raw!r}")

        try:
            gpu_cache = int(self.env.get("MOE_GPU_CACHE_MIB", "0")) / 1024
            ctx = int(self.env.get("CTX", "0"))
            gpu_total = 6141 / 1024  # RTX 4050 laptop, verified on this host
            estimate = gpu_cache + 2.4 + ctx / 65536 * 0.6
            if estimate > gpu_total:
                issues.append(
                    f"VRAM budget estimate {estimate:.1f} GiB > {gpu_total:.1f} GiB GPU "
                    "(dense ~2.4 + L1 + KV); lower CTX or MOE_GPU_CACHE_MIB")
        except ValueError:
            pass

        return issues

    def materialize_env(self) -> Path:
        """Write the launch .env so the run script can `source` it."""
        target = paths.env_dir() / f"{self.name}.env"
        target.write_text(format_env(self.env), encoding="utf-8")
        return target

    def __str__(self) -> str:
        return f"{self.name} [{self.source}] port={self.port} model={self.model or '?'}"
# ---------------------------------------------------------------------------
# discovery
# ---------------------------------------------------------------------------


def infer_script_for(profile_name: str) -> str:
    """Map a bundled profile name -> its launcher script.

    configs/qwen3-coder-next.env   -> scripts/run-qwen3-coder-next.sh
    configs/kat-coder-mtp.env      -> (no run-kat-coder-mtp.sh) -> default
    configs/qwen3.6-moe-tq3.env    -> scripts/run-qwen36-moe.sh (default)

    Falls back to the default launcher when no matching script exists.
    """
    root = paths.repo_root()
    candidates = [
        f"scripts/run-{profile_name}.sh",
        f"scripts/run-{profile_name.replace('-', '')}.sh",   # qwen3.6-moe-tq3 -> run-qwen36moetq3.sh (n/a)
    ]
    for cand in candidates:
        if (root / cand).exists():
            return cand
    return paths.default_run_script()


def load_bundled_profiles() -> list[Profile]:
    out: list[Profile] = []
    cfg = paths.configs_dir()
    if cfg.is_dir():
        for f in sorted(cfg.glob("*.env")):
            out.append(Profile.from_env_file(f))
    return out


def load_app_profiles() -> list[Profile]:
    out: list[Profile] = []
    pd = paths.profiles_dir()
    if pd.is_dir():
        for f in sorted(pd.glob("*.json")):
            try:
                out.append(Profile.from_json(f))
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"invalid app profile {f}: {exc}") from exc
    return out


def discover() -> dict[str, Profile]:
    """All profiles keyed by name. App profiles override bundled name collisions."""
    profiles: dict[str, Profile] = {}
    for p in load_bundled_profiles():
        profiles[p.name] = p
    for p in load_app_profiles():
        profiles[p.name] = p
    return profiles


def delete_app_profile(name: str) -> bool:
    path = paths.profiles_dir() / f"{name}.json"
    if path.is_file():
        path.unlink()
        return True
    return False


def find_models() -> list[dict]:
    """Discover GGUF model files under the repo models dir (recursive)."""
    out: list[dict] = []
    md = paths.models_dir()
    if md.is_dir():
        for f in sorted(md.rglob("*.gguf")):
            try:
                size_mib = round(f.stat().st_size / (1024 * 1024), 1)
            except OSError:
                size_mib = None  # dangling symlink
            out.append({"name": f.name, "path": str(f), "size_mib": size_mib})
    return out


def preview_command(profile: Profile) -> list[str]:
    """Human-friendly preview of the exact launch command (dry-run)."""
    env_file = profile.materialize_env()
    script = str(paths.repo_root() / profile.script_rel)
    return ["env", f"CONFIG={env_file}", "bash", script]