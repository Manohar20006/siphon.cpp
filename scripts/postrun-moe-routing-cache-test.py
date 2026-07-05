#!/usr/bin/env python3

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path


MODEL = "/home/manohar/Desktop/inference/qwen3.6/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select static experts after trace completion and run a cached test.")
    parser.add_argument("--progress", type=Path, default=Path("moe-routing-suite-1000.progress.jsonl"))
    parser.add_argument(
        "--trace",
        type=Path,
        action="append",
        default=None,
        help="Routing trace JSONL part. May be passed multiple times.",
    )
    parser.add_argument(
        "--combined-trace",
        type=Path,
        default=Path("moe-routing-suite-1000-combined.jsonl"),
    )
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--budget-mib", type=int, default=10240)
    parser.add_argument("--selector", type=Path, default=Path("scripts/select-static-moe-experts.py"))
    parser.add_argument("--output-experts", type=Path, default=Path("moe-static-experts-suite-10240.txt"))
    parser.add_argument("--test-output", type=Path, default=Path("suite-static-experts-10240-200.out"))
    parser.add_argument("--server-unit", default="moe-routing-server")
    parser.add_argument("--runner-unit", default="moe-routing-runner")
    return parser.parse_args()


def count_ok(path: Path) -> tuple[int, int]:
    ok = 0
    total = 0
    if not path.exists():
        return ok, total
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            total += 1
            try:
                if json.loads(line).get("status") == "ok":
                    ok += 1
            except json.JSONDecodeError:
                pass
    return ok, total


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def main() -> None:
    args = parse_args()
    if args.trace is None:
        args.trace = [Path("moe-routing-suite-1000.jsonl")]

    print(f"{datetime.now().isoformat(timespec='seconds')} waiting for {args.total} completed prompts", flush=True)
    while True:
        ok, recorded = count_ok(args.progress)
        if ok >= args.total:
            break
        runner_active = subprocess.run(
            ["systemctl", "--user", "is-active", args.runner_unit],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
        if recorded > 0 and not runner_active:
            raise RuntimeError(f"runner stopped before completion: ok={ok}, recorded={recorded}")
        print(f"{datetime.now().isoformat(timespec='seconds')} still running ok={ok}/{args.total}", flush=True)
        time.sleep(300)

    print(f"{datetime.now().isoformat(timespec='seconds')} trace complete; stopping server", flush=True)
    subprocess.run(["systemctl", "--user", "stop", args.server_unit], check=False)

    with args.combined_trace.open("w", encoding="utf-8") as combined:
        for trace_path in args.trace:
            with trace_path.open("r", encoding="utf-8", errors="ignore") as trace:
                for line in trace:
                    combined.write(line)

    run([
        "/usr/bin/python3",
        str(args.selector),
        str(args.combined_trace),
        str(args.output_experts),
        "--budget-mib",
        str(args.budget_mib),
    ])

    prompt = (
        "Write a concise Python function that groups a list of log lines by severity, "
        "counts each group, and returns the result as a dictionary. Then explain the edge cases."
    )
    env = {
        **dict(),
        "LLAMA_GDS_READ_THREADS": "20",
        "LLAMA_DSTORAGE_SUMMARY": "1",
        "LLAMA_MOE_PINNED_STATIC_EXPERTS_FILE": str(args.output_experts.resolve()),
        "LLAMA_MOE_PINNED_CACHE_MIB": str(args.budget_mib),
    }
    with args.test_output.open("w", encoding="utf-8") as output:
        run(
            [
                "env",
                *[f"{key}={value}" for key, value in env.items()],
                "build-gds-cuda/bin/llama-cli",
                "-m",
                MODEL,
                "-ngl",
                "999",
                "-c",
                "4096",
                "-fa",
                "on",
                "-dsm",
                "--dstorage-moe-prefetch",
                "--moe-gpu-cache-mib",
                "2816",
                "--moe-pinned-cache-mib",
                str(args.budget_mib),
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                "2",
                "--spec-draft-p-min",
                "0",
                "-s",
                "4242",
                "-st",
                "-n",
                "200",
                "-p",
                prompt,
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
        )
    print(f"{datetime.now().isoformat(timespec='seconds')} cached test written to {args.test_output}", flush=True)


if __name__ == "__main__":
    main()
