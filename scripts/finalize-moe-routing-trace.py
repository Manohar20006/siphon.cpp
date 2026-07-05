#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and combine portable MoE routing-trace shards."
    )
    parser.add_argument("inputs", type=Path, nargs="+", help="Raw routing JSONL shards")
    parser.add_argument("--output", type=Path, required=True, help="Combined JSONL output")
    return parser.parse_args()


def valid_event(event: Any) -> bool:
    if not isinstance(event, dict):
        return False
    required = ("phase", "layer", "expert_bytes", "experts", "misses")
    if any(key not in event for key in required):
        return False
    if event["phase"] not in ("prefill", "decode"):
        return False
    if not isinstance(event["experts"], list) or not event["experts"]:
        return False
    if not isinstance(event["misses"], list):
        return False
    return int(event["layer"]) >= 0 and int(event["expert_bytes"]) > 0


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_lines = 0
    invalid_lines = 0
    written = 0
    request_id = -1
    phase_counts: Counter[str] = Counter()
    layers: set[int] = set()

    with args.output.open("w", encoding="utf-8") as output:
        for input_path in args.inputs:
            request_map: dict[int, int] = {}
            with input_path.open("r", encoding="utf-8") as source:
                for line in source:
                    total_lines += 1
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        invalid_lines += 1
                        continue
                    if not valid_event(event):
                        invalid_lines += 1
                        continue

                    old_request = int(event.get("request", 0))
                    if old_request not in request_map:
                        request_id += 1
                        request_map[old_request] = request_id
                    event["sequence"] = written
                    event["request"] = request_map[old_request]
                    output.write(json.dumps(event, separators=(",", ":")) + "\n")

                    written += 1
                    phase_counts[str(event["phase"])] += 1
                    layers.add(int(event["layer"]))

    print(
        f"inputs={len(args.inputs)} lines={total_lines} written={written} "
        f"invalid={invalid_lines} requests={request_id + 1} "
        f"layers={len(layers)} phases={dict(sorted(phase_counts.items()))} "
        f"output={args.output}"
    )
    if written == 0:
        raise SystemExit("no valid routing events were written")


if __name__ == "__main__":
    main()
