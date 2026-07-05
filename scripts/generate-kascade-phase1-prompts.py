#!/usr/bin/env python3
"""Generate deterministic coding prompts near requested raw token counts."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


COUNT_RE = re.compile(r"Total number of tokens:\s+(\d+)")


def token_count(tokenizer: Path, model: Path, text: str) -> int:
    result = subprocess.run(
        [
            str(tokenizer),
            "-m",
            str(model),
            "--stdin",
            "--ids",
            "--show-count",
            "--log-disable",
        ],
        input=text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    )
    match = COUNT_RE.search(result.stdout)
    if match is None:
        raise RuntimeError(f"could not parse tokenizer output: {result.stdout[-500:]}")
    return int(match.group(1))


def make_prompt(seed: str, target: int, tokenizer: Path, model: Path) -> tuple[str, int]:
    seed_tokens = token_count(tokenizer, model, seed)
    repeats = max(1, (target + seed_tokens - 1) // seed_tokens)
    text = (seed.rstrip() + "\n\n") * repeats
    raw = text.encode("utf-8")
    low, high = 1, len(raw)
    best_text, best_count = seed, seed_tokens

    while low <= high:
        mid = (low + high) // 2
        candidate = raw[:mid].decode("utf-8", errors="ignore")
        count = token_count(tokenizer, model, candidate)
        if abs(count - target) < abs(best_count - target):
            best_text, best_count = candidate, count
        if count < target:
            low = mid + 1
        elif count > target:
            high = mid - 1
        else:
            return candidate, count
    return best_text, best_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--targets", type=int, nargs="+", default=[2048, 8192, 16384, 20000])
    args = parser.parse_args()

    seed = args.seed.read_text(encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for target in args.targets:
        prompt, count = make_prompt(seed, target, args.tokenizer, args.model)
        output = args.output_dir / f"coding-{target}.txt"
        output.write_text(prompt, encoding="utf-8")
        print(f"target={target} actual={count} bytes={len(prompt.encode('utf-8'))} output={output}")


if __name__ == "__main__":
    main()
