#!/usr/bin/env python3
"""Run the Universal Worker Bridge over Streamable HTTP."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from universal_mcp.http_server import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
