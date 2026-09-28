"""Expert-suite handling.

Suite files (verified format, see expert-suites/*.txt):

    # Ranked static MoE experts: layer expert miss_count access_count bytes
    # budget_mib=8192 selected=4491 selected_mib=8190.41 trace_avoidable_mib=...
    39 110 7424 8793 2039808
    ...

Each data row is: layer expert miss_count access_count bytes. The header
lines carry the exact budget/selected/selected_mib the generator used.
"""

from __future__ import annotations

import re
from pathlib import Path

from .. import paths


def suite_info(path: Path) -> dict:
    """Parse one suite file into a compact summary dict."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {"path": str(path), "error": str(exc)}

    header: dict[str, float] = {}
    count = 0
    bytes_total = 0
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            for m in re.finditer(r"(\w+)=([\d.]+)", s):
                header[m.group(1)] = float(m.group(2))
            continue
        parts = s.split()
        if len(parts) >= 5:
            try:
                bytes_total += int(parts[4])
                count += 1
            except ValueError:
                pass

    return {
        "path": str(path),
        "name": path.name,
        "count": count,
        "bytes": bytes_total,
        "mib": round(bytes_total / (1024 * 1024), 2),
        "budget_mib": header.get("budget_mib"),
        "selected": header.get("selected"),
        "selected_mib": header.get("selected_mib"),
    }


def list_suites() -> list[dict]:
    sd = paths.suites_dir()
    if not sd.is_dir():
        return []
    infos = [suite_info(f) for f in sorted(sd.glob("*.txt"))]
    return [i for i in infos if "error" not in i]