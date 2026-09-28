from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from impact_analyzer import analyze


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze transitive dependency impact")
    parser.add_argument("graph")
    parser.add_argument("changed", nargs="+")
    args = parser.parse_args()
    try:
        graph = json.loads(Path(args.graph).read_text(encoding="utf-8"))
        print(json.dumps(analyze(graph, args.changed), sort_keys=True))
        return 0
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
