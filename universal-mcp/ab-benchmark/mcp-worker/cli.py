"""CLI for Dependency Impact Analyzer."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from impact_analyzer import analyze


def main():
    if len(sys.argv) < 3:
        sys.stderr.write("Usage: cli.py GRAPH.json CHANGED [CHANGED ...]\n")
        sys.exit(2)

    graph_file = sys.argv[1]
    changed = sys.argv[2:]

    try:
        with open(graph_file, "r", encoding="utf-8") as f:
            graph = json.load(f)
    except Exception as e:
        sys.stderr.write(f"Error reading graph file: {e}\n")
        sys.exit(2)

    try:
        result = analyze(graph, changed)
    except Exception as e:
        sys.stderr.write(f"Validation error: {e}\n")
        sys.exit(2)

    print(json.dumps(result, indent=2, sort_keys=True))
    sys.exit(0)


if __name__ == "__main__":
    main()
