#!/usr/bin/env python3
"""Summarize Kascade phase-one logs and Nsight kernel CSVs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PROMPT_RE = re.compile(r"\[ Prompt: ([0-9.]+) t/s")
STREAM_RE = re.compile(r"DSTORAGE_MOE_STREAM .*?prefill_total_us=(\d+)")
TIMELINE_RE = re.compile(
    r"DSTORAGE_MOE_TIMELINE .*?ensure_total_us=(\d+).*?"
    r"compute_samples=(\d+) compute_avg_us=([0-9.]+)"
)


def kernel_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def ns(value: str) -> int:
    return int(value.replace(",", ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--kernels", type=Path, required=True)
    args = parser.parse_args()

    log = args.log.read_text(encoding="utf-8", errors="replace")
    prompt_match = PROMPT_RE.search(log)
    stream_match = STREAM_RE.search(log)
    timeline_match = TIMELINE_RE.search(log)
    if prompt_match is None or stream_match is None or timeline_match is None:
        raise SystemExit("required timing fields are missing from the profile log")

    prompt_tps = float(prompt_match.group(1))
    estimated_wall_s = args.target / prompt_tps
    stream_s = int(stream_match.group(1)) / 1e6
    ensure_s = int(timeline_match.group(1)) / 1e6
    compute_s = int(timeline_match.group(2)) * float(timeline_match.group(3)) / 1e6

    rows = kernel_rows(args.kernels)
    total_gpu_s = sum(ns(row["Total Time (ns)"]) for row in rows) / 1e9
    attention_gpu_s = sum(
        ns(row["Total Time (ns)"])
        for row in rows
        if "flash_attn" in row["Name"]
    ) / 1e9
    delta_gpu_s = sum(
        ns(row["Total Time (ns)"])
        for row in rows
        if "gated_delta_net" in row["Name"]
    ) / 1e9
    matmul_gpu_s = sum(
        ns(row["Total Time (ns)"])
        for row in rows
        if row["Name"].startswith("mul_mat")
    ) / 1e9

    print(f"target_tokens={args.target}")
    print(f"prompt_tps={prompt_tps:.3f}")
    print(f"estimated_prompt_wall_s={estimated_wall_s:.3f}")
    print(f"moe_stream_s={stream_s:.3f}")
    print(f"moe_ensure_s={ensure_s:.3f}")
    print(f"sampled_expert_compute_s={compute_s:.3f}")
    print(f"gpu_kernel_s={total_gpu_s:.3f}")
    print(f"flash_attention_gpu_s={attention_gpu_s:.6f}")
    print(f"gated_delta_net_gpu_s={delta_gpu_s:.6f}")
    print(f"matmul_gpu_s={matmul_gpu_s:.3f}")
    print(f"flash_attention_wall_share={100.0 * attention_gpu_s / estimated_wall_s:.4f}%")
    print(f"flash_attention_gpu_share={100.0 * attention_gpu_s / total_gpu_s:.4f}%")


if __name__ == "__main__":
    main()
