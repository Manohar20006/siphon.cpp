#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
import time
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/home/manohar/Desktop/inference/qwen3.6/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
STATIC_EXPERTS = ROOT / "moe-static-experts-suite-8192.txt"
LLAMA_CLI = ROOT / "build-gds-cuda/bin/llama-cli"


TASKS = [
    {
        "id": "coding_merge_sorted",
        "category": "coding",
        "n": 320,
        "prompt": "Write a Python function merge_sorted_lists(a, b) that merges two sorted lists into one sorted list. Include a short example.",
        "checks": ["def merge_sorted_lists", "while", "return", "example"],
    },
    {
        "id": "coding_debug_off_by_one",
        "category": "coding",
        "n": 260,
        "prompt": "Find and fix the bug in this Python function. Explain in one sentence, then give corrected code:\n\n"
                  "def sum_first_n(nums, n):\n"
                  "    total = 0\n"
                  "    for i in range(n + 1):\n"
                  "        total += nums[i]\n"
                  "    return total\n",
        "checks": ["range(n)", "n + 1", "def sum_first_n"],
    },
    {
        "id": "reasoning_bat_ball",
        "category": "reasoning",
        "n": 220,
        "prompt": "A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. What does the ball cost? Give the final answer in cents and show the equation briefly.",
        "checks": ["5", "cent"],
    },
    {
        "id": "reasoning_machine_rate",
        "category": "reasoning",
        "n": 300,
        "prompt": "Machine A finishes a job in 6 hours. Machine B finishes the same job in 4 hours. If both work together, how many hours do they need? Answer final first, then show the calculation briefly.",
        "checks": ["2.4", "hour"],
    },
    {
        "id": "logic_mislabeled_boxes",
        "category": "puzzle",
        "n": 320,
        "prompt": "There are three boxes labeled Apples, Oranges, and Apples+Oranges. Every label is wrong. You may draw one fruit from one box to identify all boxes. Which box do you draw from and why?",
        "checks": ["apples+oranges", "draw", "wrong"],
    },
    {
        "id": "logic_light_switches",
        "category": "puzzle",
        "n": 320,
        "prompt": "Classic puzzle: You are outside a room with three switches, each controlling one of three bulbs inside. You may enter the room only once. How do you identify which switch controls which bulb?",
        "checks": ["on", "off", "heat"],
    },
    {
        "id": "math_schedule",
        "category": "complex_reasoning",
        "n": 360,
        "prompt": "A job has tasks A, B, C, D. A takes 3 hours. B takes 4 hours and can start only after A. C takes 2 hours and can start only after A. D takes 5 hours and can start only after both B and C. What is the shortest completion time? Answer final first, then explain briefly.",
        "checks": ["12 hours"],
    },
    {
        "id": "tool_call_weather",
        "category": "tool_calling",
        "n": 180,
        "prompt": "You have tools get_weather(city: string) and search_docs(query: string). The user asks: What is the weather in Tokyo? Respond with only a JSON tool call object containing name and arguments.",
        "checks": ["get_weather", "Tokyo"],
    },
    {
        "id": "tool_call_search_docs",
        "category": "tool_calling",
        "n": 180,
        "prompt": "You have tools get_weather(city: string) and search_docs(query: string). The user asks: Find the docs for CUDA streams. Respond with only a JSON tool call object containing name and arguments.",
        "checks": ["search_docs", "CUDA"],
    },
    {
        "id": "tool_call_no_tool",
        "category": "tool_calling",
        "n": 160,
        "prompt": "You have tools get_weather(city: string) and search_docs(query: string). The user asks: What is 17 * 23? If no tool is needed, answer directly with only the number.",
        "checks": ["391"],
    },
]


def run_case(task, mode, out_dir):
    env = os.environ.copy()
    env.update({
        "LLAMA_MOE_PINNED_PREFILL_ONLY": "1",
        "LLAMA_GDS_READ_THREADS": "20",
        "LLAMA_DSTORAGE_SUMMARY": "1",
        "LLAMA_MOE_PINNED_STATIC_EXPERTS_FILE": str(STATIC_EXPERTS),
    })
    m = re.fullmatch(r"topk([0-9]+)", mode)
    if m:
        env["LLAMA_MOE_EXPERT_USED_OVERRIDE"] = m.group(1)
    elif mode == "baseline_k8":
        env.pop("LLAMA_MOE_EXPERT_USED_OVERRIDE", None)
    else:
        raise ValueError(f"unknown mode: {mode}")

    output_path = out_dir / f"{task['id']}__{mode}.out"
    cmd = [
        str(LLAMA_CLI),
        "-m", str(MODEL),
        "-ngl", "999",
        "-c", "4096",
        "-fa", "on",
        "-dsm",
        "--dstorage-moe-prefetch",
        "--moe-gpu-cache-mib", "2816",
        "--moe-pinned-cache-mib", "8192",
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", "3",
        "--spec-draft-p-min", "0",
        "--reasoning", "off",
        "--reasoning-budget", "0",
        "-s", "4242",
        "-st",
        "-n", str(task["n"]),
        "-p", task["prompt"],
    ]

    start = time.time()
    with output_path.open("w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.run(cmd, env=env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, timeout=900)
    elapsed = time.time() - start
    text = output_path.read_text(encoding="utf-8", errors="replace")

    prompt_tps = None
    gen_tps = None
    m = re.search(r"\[ Prompt:\s*([0-9.]+) t/s \| Generation:\s*([0-9.]+) t/s \]", text)
    if m:
        prompt_tps = float(m.group(1))
        gen_tps = float(m.group(2))

    timeline = {}
    matches = re.findall(r"DSTORAGE_MOE_TIMELINE ([^\n]+)", text)
    if matches:
        for key, value in re.findall(r"([a-zA-Z_]+)=([0-9.]+)", matches[-1]):
            timeline[key] = float(value) if "." in value else int(value)

    phase = {}
    matches = re.findall(r"DSTORAGE_MOE_PHASE ([^\n]+)", text)
    if matches:
        for key, value in re.findall(r"([a-zA-Z_]+)=([0-9.]+)", matches[-1]):
            phase[key] = float(value) if "." in value else int(value)

    lower = text.lower()
    checks = {needle: (needle.lower() in lower) for needle in task["checks"]}
    passed = all(checks.values())

    return {
        "task_id": task["id"],
        "category": task["category"],
        "mode": mode,
        "exit_code": proc.returncode,
        "elapsed_sec": round(elapsed, 2),
        "prompt_tps": prompt_tps,
        "generation_tps": gen_tps,
        "selected_experts": timeline.get("selected_experts"),
        "stream_file_mib": timeline.get("stream_file_mib"),
        "hit_rate": timeline.get("hit_rate"),
        "decode_hit_rate": phase.get("decode_hit_rate"),
        "checks": checks,
        "passed": passed,
        "output": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        help="Comma-separated task ids to run. By default the full suite runs.",
    )
    parser.add_argument(
        "--modes",
        default="baseline_k8,topk6",
        help="Comma-separated modes to run. Use baseline_k8 or topkN, for example topk5.",
    )
    args = parser.parse_args()

    tasks = TASKS
    if args.only:
        wanted = {item.strip() for item in args.only.split(",") if item.strip()}
        tasks = [task for task in TASKS if task["id"] in wanted]
        missing = wanted - {task["id"] for task in tasks}
        if missing:
            raise SystemExit(f"unknown task id(s): {', '.join(sorted(missing))}")

    out_dir = ROOT / "topk-quality-sweep-results" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    for mode in modes:
        if mode != "baseline_k8" and not re.fullmatch(r"topk[0-9]+", mode):
            raise SystemExit(f"unknown mode: {mode}")

    results = []
    for task in tasks:
        for mode in modes:
            print(f"RUN {mode} {task['id']}", flush=True)
            result = run_case(task, mode, out_dir)
            results.append(result)
            print(json.dumps({
                "task": result["task_id"],
                "mode": result["mode"],
                "passed": result["passed"],
                "prompt_tps": result["prompt_tps"],
                "generation_tps": result["generation_tps"],
                "selected_experts": result["selected_experts"],
            }), flush=True)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = ["# Top-K Quality Sweep", ""]
    lines.append("| Task | Mode | Pass | Prompt t/s | Gen t/s | Experts | Stream MiB | Hit | Decode Hit |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r['task_id']} | {r['mode']} | {str(r['passed']).lower()} | "
            f"{r['prompt_tps']} | {r['generation_tps']} | {r['selected_experts']} | "
            f"{r['stream_file_mib']} | {r['hit_rate']} | {r['decode_hit_rate']} |")
    lines.append("")
    lines.append("## Check Details")
    for r in results:
        missing = [k for k, ok in r["checks"].items() if not ok]
        lines.append(f"- `{r['task_id']}` `{r['mode']}`: {'pass' if r['passed'] else 'missing ' + ', '.join(missing)}")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"RESULT_DIR={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
