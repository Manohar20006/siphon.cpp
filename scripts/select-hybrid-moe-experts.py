#!/usr/bin/env python3

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a hybrid general/decode static pinned RAM expert set."
    )
    parser.add_argument("trace", type=Path, help="Routing trace JSONL file")
    parser.add_argument("output", type=Path, help="Output layer/expert selection file")
    parser.add_argument("--general-budget-mib", type=int, required=True)
    parser.add_argument("--decode-budget-mib", type=int, required=True)
    parser.add_argument(
        "--decode-rank",
        choices=("access", "miss"),
        default="access",
        help="Rank the decode partition by decode accesses or decode misses.",
    )
    parser.add_argument(
        "--decode-first",
        action="store_true",
        help="Reserve decode experts before filling the general partition.",
    )
    return parser.parse_args()


def rank_keys(
    counts: Counter[tuple[int, int]],
    tie_counts: Counter[tuple[int, int]],
    expert_bytes: dict[int, int],
) -> list[tuple[int, int]]:
    return sorted(
        counts,
        key=lambda key: (
            -counts[key],
            -tie_counts[key],
            expert_bytes[key[0]],
            key[0],
            key[1],
        ),
    )


def add_ranked(
    ranked: list[tuple[int, int]],
    selected: set[tuple[int, int]],
    expert_bytes: dict[int, int],
    budget_bytes: int,
    used_bytes: int,
) -> tuple[int, list[tuple[int, int]]]:
    added: list[tuple[int, int]] = []
    for key in ranked:
        if key in selected:
            continue
        size = expert_bytes[key[0]]
        if used_bytes + size > budget_bytes:
            continue
        selected.add(key)
        added.append(key)
        used_bytes += size
    return used_bytes, added


def main() -> None:
    args = parse_args()
    all_misses: Counter[tuple[int, int]] = Counter()
    decode_misses: Counter[tuple[int, int]] = Counter()
    all_accesses: Counter[tuple[int, int]] = Counter()
    decode_accesses: Counter[tuple[int, int]] = Counter()
    expert_bytes: dict[int, int] = {}

    with args.trace.open("r", encoding="utf-8", errors="ignore") as trace:
        for line in trace:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            layer = int(event["layer"])
            phase = str(event["phase"])
            expert_bytes[layer] = int(event["expert_bytes"])
            experts = {(layer, int(expert)) for expert in set(event["experts"])}
            misses = {(layer, int(expert)) for expert in set(event["misses"])}
            all_accesses.update(experts)
            all_misses.update(misses)
            if phase == "decode":
                decode_accesses.update(experts)
                decode_misses.update(misses)

    selected: set[tuple[int, int]] = set()
    general_budget = args.general_budget_mib * 1024 * 1024
    total_budget = (args.general_budget_mib + args.decode_budget_mib) * 1024 * 1024

    general_ranked = rank_keys(all_misses, all_accesses, expert_bytes)
    if args.decode_rank == "access":
        decode_ranked = rank_keys(decode_accesses, decode_misses, expert_bytes)
    else:
        decode_ranked = rank_keys(decode_misses, decode_accesses, expert_bytes)

    if args.decode_first:
        decode_budget = args.decode_budget_mib * 1024 * 1024
        used_bytes, decode_added = add_ranked(
            decode_ranked, selected, expert_bytes, decode_budget, 0
        )
        used_bytes, general_added = add_ranked(
            general_ranked, selected, expert_bytes, total_budget, used_bytes
        )
        backfill_added: list[tuple[int, int]] = []
    else:
        used_bytes, general_added = add_ranked(
            general_ranked, selected, expert_bytes, general_budget, 0
        )
        used_bytes, decode_added = add_ranked(
            decode_ranked, selected, expert_bytes, total_budget, used_bytes
        )
        # If decode overlaps heavily with general, backfill remaining bytes with general experts.
        used_bytes, backfill_added = add_ranked(
            general_ranked, selected, expert_bytes, total_budget, used_bytes
        )

    ordered = (
        decode_added + general_added + backfill_added
        if args.decode_first
        else general_added + decode_added + backfill_added
    )
    avoided_all = sum(all_misses[key] * expert_bytes[key[0]] for key in selected)
    avoided_decode = sum(decode_misses[key] * expert_bytes[key[0]] for key in selected)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        output.write("# Hybrid static MoE experts: layer expert all_miss decode_miss all_access decode_access bytes source\n")
        output.write(
            f"# general_budget_mib={args.general_budget_mib} decode_budget_mib={args.decode_budget_mib} "
            f"decode_rank={args.decode_rank} decode_first={int(args.decode_first)} "
            f"selected={len(ordered)} selected_mib={used_bytes / 1024 / 1024:.2f} "
            f"general_added={len(general_added)} decode_added={len(decode_added)} backfill_added={len(backfill_added)} "
            f"trace_avoidable_mib={avoided_all / 1024 / 1024:.2f} "
            f"trace_decode_avoidable_mib={avoided_decode / 1024 / 1024:.2f}\n"
        )
        if args.decode_first:
            sources = ["decode"] * len(decode_added) + ["general"] * len(general_added)
        else:
            sources = (
                ["general"] * len(general_added)
                + ["decode"] * len(decode_added)
                + ["backfill"] * len(backfill_added)
            )
        for source, (layer, expert) in zip(sources, ordered):
            output.write(
                f"{layer} {expert} {all_misses[(layer, expert)]} {decode_misses[(layer, expert)]} "
                f"{all_accesses[(layer, expert)]} {decode_accesses[(layer, expert)]} {expert_bytes[layer]} {source}\n"
            )

    print(
        f"selected={len(ordered)} selected_mib={used_bytes / 1024 / 1024:.2f} "
        f"decode_rank={args.decode_rank} decode_first={int(args.decode_first)} "
        f"general_added={len(general_added)} decode_added={len(decode_added)} "
        f"backfill_added={len(backfill_added)} "
        f"trace_avoidable_mib={avoided_all / 1024 / 1024:.2f} "
        f"trace_decode_avoidable_mib={avoided_decode / 1024 / 1024:.2f}"
    )


if __name__ == "__main__":
    main()
