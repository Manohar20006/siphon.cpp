#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select trace-ranked MoE experts for a static pinned RAM cache."
    )
    parser.add_argument("trace", type=Path, help="Routing trace JSONL file")
    parser.add_argument("output", type=Path, help="Output layer/expert selection file")
    parser.add_argument("--budget-mib", type=int, required=True, help="Pinned RAM budget in MiB")
    parser.add_argument(
        "--phase",
        choices=("all", "prefill", "decode"),
        default="all",
        help="Only rank events from this phase. Default: all phases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    miss_counts: Counter[tuple[int, int]] = Counter()
    access_counts: Counter[tuple[int, int]] = Counter()
    expert_bytes: dict[int, int] = {}

    with args.trace.open("r", encoding="utf-8") as trace:
        for line in trace:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if args.phase != "all" and str(event.get("phase")) != args.phase:
                continue
            layer = int(event["layer"])
            expert_bytes[layer] = int(event["expert_bytes"])
            access_counts.update((layer, int(expert)) for expert in set(event["experts"]))
            miss_counts.update((layer, int(expert)) for expert in set(event["misses"]))

    ranked = sorted(
        miss_counts,
        key=lambda key: (
            -miss_counts[key],
            -access_counts[key],
            expert_bytes[key[0]],
            key[0],
            key[1],
        ),
    )

    budget_bytes = args.budget_mib * 1024 * 1024
    selected: list[tuple[int, int]] = []
    selected_bytes = 0
    avoided_reload_bytes = 0
    for key in ranked:
        size = expert_bytes[key[0]]
        if selected_bytes + size > budget_bytes:
            continue
        selected.append(key)
        selected_bytes += size
        avoided_reload_bytes += miss_counts[key] * size

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        output.write("# Ranked static MoE experts: layer expert miss_count access_count bytes\n")
        output.write(
            f"# budget_mib={args.budget_mib} selected={len(selected)} "
            f"phase={args.phase} "
            f"selected_mib={selected_bytes / 1024 / 1024:.2f} "
            f"trace_avoidable_mib={avoided_reload_bytes / 1024 / 1024:.2f}\n"
        )
        for layer, expert in selected:
            output.write(
                f"{layer} {expert} {miss_counts[(layer, expert)]} "
                f"{access_counts[(layer, expert)]} {expert_bytes[layer]}\n"
            )

    print(
        f"selected={len(selected)} phase={args.phase} "
        f"selected_mib={selected_bytes / 1024 / 1024:.2f} "
        f"trace_avoidable_mib={avoided_reload_bytes / 1024 / 1024:.2f}"
    )


if __name__ == "__main__":
    main()
