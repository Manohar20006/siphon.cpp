"""Environment health checks ('doctor').

Checks the things that actually break on this machine, verified on hardware:
  - CUDA driver + nvidia-smi present
  - GPU (ideally the 6 GB target)
  - libcufile present (GDS/cuFile userspace)
  - nvidia_fs kernel module loaded (GDS)
  - llama-server binary built
  - tmux available (preferred driver)
  - textual available (TUI)
  - repo root resolvable
  - profiles load + validate
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .. import paths
from .profiles import discover


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""

    @property
    def status_char(self) -> str:
        return "OK" if self.ok else "FAIL"


def _which(name: str) -> str | None:
    return shutil.which(name)


def run_checks() -> list[Check]:
    checks: list[Check] = []

    # cuda driver / gpu
    nvidia_smi = _which("nvidia-smi")
    if nvidia_smi:
        checks.append(Check("nvidia-smi", True, nvidia_smi))
    else:
        checks.append(Check("nvidia-smi", False, "not found in PATH"))

    gpu_mem_mib: int | None = None
    if nvidia_smi:
        try:
            res = subprocess.run(
                [nvidia_smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            gpu_mem_mib = int(res.stdout.strip().splitlines()[0])
            checks.append(Check("GPU VRAM", True, f"{round(gpu_mem_mib/1024,1)} GiB"))
        except Exception as exc:  # noqa: BLE001
            gpu_mem_mib = None
            checks.append(Check("GPU VRAM", False, str(exc)))

    # gds userspace (cufile)
    cufile = any(
        line.startswith("libcufile.so") or line.startswith("libcufile.so.")
        for line in _ldconfig_output()
    ) or _libcufile_present()
    checks.append(Check("cuFile/GDS lib", cufile,
                        "libcufile.so found" if cufile else "libcufile.so NOT found (GDS will fall back)"))

    # gds kernel module
    nvidia_fs = _module_loaded("nvidia_fs")
    checks.append(Check("nvidia_fs (GDS kernel)", nvidia_fs,
                        "loaded" if nvidia_fs else "NOT loaded - GDS may fail; re-run sudo scripts/enable-gds-platform.sh"))

    # llama-server binary
    server = paths.server_bin()
    checks.append(Check("llama-server", server.exists() and server.is_file(),
                        str(server) if server.exists() else "missing - build with scripts/build-gds-cuda.sh"))

    # tmux
    tmux = _which("tmux")
    checks.append(Check("tmux", bool(tmux), tmux or "not installed (fallback: direct driver)"))

    # textual
    textual = _import_ok("textual")
    checks.append(Check("textual", textual, "ok" if textual else "pip install textual"))

    # repo root + configs
    root = paths.repo_root()
    checks.append(Check("repo root", root.is_dir(), str(root)))

    # profiles
    try:
        profs = discover()
        checks.append(Check("profiles", bool(profs), f"{len(profs)} found"))
    except Exception as exc:  # noqa: BLE001
        checks.append(Check("profiles", False, str(exc)))

    return checks


def _ldconfig_output() -> list[str]:
    try:
        res = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=5)
        return res.stdout.splitlines()
    except (subprocess.SubprocessError, OSError):
        return []


def _libcufile_present() -> bool:
    for cand in ("/usr/local/cuda/targets/x86_64-linux/lib/libcufile.so",
                 "/usr/lib/x86_64-linux-gnu/libcufile.so"):
        if Path(cand).exists():
            return True
    return False


def _module_loaded(name: str) -> bool:
    try:
        res = subprocess.run(["lsmod"], capture_output=True, text=True, timeout=5)
        return any(line.startswith(name + " ") for line in res.stdout.splitlines())
    except (subprocess.SubprocessError, OSError):
        return False


def _import_ok(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False