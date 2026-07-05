#!/usr/bin/env python3

import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Periodic progress logger for MoE trace runs.")
    parser.add_argument("--progress", type=Path, default=Path("moe-routing-suite-1000.progress.jsonl"))
    parser.add_argument("--trace", type=Path, default=Path("moe-routing-suite-1000.jsonl"))
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--runner-unit", default="moe-routing-runner")
    return parser.parse_args()


def load_progress(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def count_trace_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="ignore") as source:
        return sum(1 for _ in source)


def is_active(unit: str) -> bool:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", unit],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def print_status(args: argparse.Namespace) -> bool:
    rows = load_progress(args.progress)
    ok = [row for row in rows if row.get("status") == "ok"]
    errors = len(rows) - len(ok)
    percent = 100.0 * len(rows) / args.total if args.total else 0.0
    elapsed = [float(row["elapsed_seconds"]) for row in ok if "elapsed_seconds" in row]
    prompt_tps = [
        float(row["timings"]["prompt_per_second"])
        for row in ok
        if row.get("timings") and row["timings"].get("prompt_per_second") is not None
    ]
    decode_tps = [
        float(row["timings"]["predicted_per_second"])
        for row in ok
        if row.get("timings") and row["timings"].get("predicted_per_second") is not None
    ]

    avg_elapsed = statistics.mean(elapsed) if elapsed else 0.0
    eta_hours = avg_elapsed * max(0, args.total - len(rows)) / 3600.0
    recent_elapsed = statistics.mean(elapsed[-20:]) if elapsed else 0.0
    recent_eta_hours = recent_elapsed * max(0, args.total - len(rows)) / 3600.0
    trace_lines = count_trace_lines(args.trace)
    runner_active = is_active(args.runner_unit)

    print(
        f"{datetime.now().isoformat(timespec='seconds')} "
        f"progress={len(rows)}/{args.total} ({percent:.1f}%) ok={len(ok)} errors={errors} "
        f"trace_events={trace_lines} avg_prompt_tps={statistics.mean(prompt_tps) if prompt_tps else 0.0:.2f} "
        f"avg_decode_tps={statistics.mean(decode_tps) if decode_tps else 0.0:.2f} "
        f"eta_avg={eta_hours:.2f}h eta_recent20={recent_eta_hours:.2f}h "
        f"runner_active={int(runner_active)}",
        flush=True,
    )

    return len(rows) >= args.total or (not runner_active and len(rows) > 0)


def main() -> None:
    args = parse_args()
    while True:
        done = print_status(args)
        if done:
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
