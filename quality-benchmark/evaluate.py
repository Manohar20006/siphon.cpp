#!/usr/bin/env python3
"""Shared, dependency-free evaluator for both benchmark implementations."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def evaluate(root: Path) -> dict[str, object]:
    files = {name: root / name for name in ("index.html", "styles.css", "script.js")}
    html = files["index.html"].read_text(encoding="utf-8") if files["index.html"].is_file() else ""
    css = files["styles.css"].read_text(encoding="utf-8") if files["styles.css"].is_file() else ""
    js = files["script.js"].read_text(encoding="utf-8") if files["script.js"].is_file() else ""
    combined = html + css + js

    checks = {
        "three_required_files": all(path.is_file() for path in files.values()),
        "semantic_task_form": all(token in html.lower() for token in ("<form", "<input", "<button", "<ul")),
        "trim_and_empty_rejection": ".trim()" in js and ("if (!" in js or "if(!" in js),
        "task_add_toggle_delete": all(token in js for token in ("addEventListener", "completed", "delete")),
        "three_filters": all(f'data-filter="{name}"' in html for name in ("all", "active", "completed")),
        "active_and_total_counts": "active-count" in html and "total-count" in html and ".length" in js,
        "clear_completed_gate": "clear" in js.lower() and "disabled" in js,
        "tasks_and_filter_persisted": js.count("localStorage.setItem") >= 2 and js.count("localStorage.getItem") >= 2,
        "accessible_status_and_focus": "aria-live" in html and ":focus-visible" in css,
        "responsive_and_dependency_free": "@media" in css and not re.search(r'https?://|<script[^>]+src=["\']https?://', combined, re.I),
        "safe_user_text_rendering": ".textContent" in js and not re.search(
            r"innerHTML\s*=\s*(?:`|[A-Za-z_$])", js
        ),
    }
    passed = sum(checks.values())
    return {
        "implementation": str(root),
        "score": passed,
        "maximum": len(checks),
        "percent": round(100 * passed / len(checks), 1),
        "checks": checks,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: evaluate.py IMPLEMENTATION_DIRECTORY")
    result = evaluate(Path(sys.argv[1]))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["score"] == result["maximum"] else 1)
