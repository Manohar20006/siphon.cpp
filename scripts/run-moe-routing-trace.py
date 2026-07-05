#!/usr/bin/env python3

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable JSONL prompt suite through llama-server."
    )
    parser.add_argument("prompts", type=Path, help="Prompt-suite JSONL file")
    parser.add_argument("progress", type=Path, help="Per-prompt result JSONL file")
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8091/completion",
        help="llama-server completion endpoint",
    )
    parser.add_argument("--n-predict", type=int, default=32)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry indices already recorded with status=error",
    )
    return parser.parse_args()


def load_prompts(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        prompts = [json.loads(line) for line in source if line.strip()]
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


def load_completed(path: Path, retry_failures: bool) -> set[int]:
    completed: set[int] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            result = json.loads(line)
            if result.get("status") == "ok" or not retry_failures:
                completed.add(int(result["index"]))
    return completed


def post_completion(
    url: str,
    prompt: str,
    n_predict: int,
    seed: int,
    timeout: int,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "prompt": prompt,
            "n_predict": n_predict,
            "seed": seed,
            "temperature": 0.0,
            "cache_prompt": False,
            "stream": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def main() -> None:
    args = parse_args()
    prompts = load_prompts(args.prompts)
    completed = load_completed(args.progress, args.retry_failures)
    stop = len(prompts) if args.limit is None else min(len(prompts), args.start + args.limit)
    selected = [
        (index, prompts[index])
        for index in range(args.start, stop)
        if index not in completed
    ]

    args.progress.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"suite={args.prompts} total={len(prompts)} selected={len(selected)} "
        f"already_recorded={len(completed)} n_predict={args.n_predict}",
        flush=True,
    )

    run_started = time.monotonic()
    with args.progress.open("a", encoding="utf-8", buffering=1) as progress:
        for position, (index, item) in enumerate(selected, start=1):
            started = time.monotonic()
            result: dict[str, Any] = {
                "index": index,
                "id": item["id"],
                "family": item["family"],
                "source": item["source"],
                "category": item["category"],
                "n_predict": args.n_predict,
            }
            try:
                response = post_completion(
                    args.url,
                    item["prompt"],
                    args.n_predict,
                    args.seed,
                    args.timeout,
                )
                result.update(
                    {
                        "status": "ok",
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "tokens_predicted": response.get("tokens_predicted"),
                        "tokens_evaluated": response.get("tokens_evaluated"),
                        "timings": response.get("timings"),
                    }
                )
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                result.update(
                    {
                        "status": "error",
                        "elapsed_seconds": round(time.monotonic() - started, 3),
                        "error": str(error),
                    }
                )

            progress.write(json.dumps(result, ensure_ascii=False) + "\n")
            elapsed = time.monotonic() - run_started
            rate = position / elapsed if elapsed > 0 else 0.0
            remaining = (len(selected) - position) / rate if rate > 0 else 0.0
            print(
                f"[{position}/{len(selected)}] index={index} id={item['id']} "
                f"status={result['status']} elapsed={result['elapsed_seconds']:.1f}s "
                f"eta={remaining / 3600:.2f}h",
                flush=True,
            )


if __name__ == "__main__":
    main()
