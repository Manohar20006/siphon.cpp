"""Executable wrapper for the stdio MCP server."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from universal_mcp.server import main


raise SystemExit(main())
