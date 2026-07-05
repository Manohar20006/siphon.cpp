#!/usr/bin/env python3
import argparse
import csv
import gzip
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("/home/manohar/Desktop/inference/qwen3.6/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf")
STATIC_EXPERTS = ROOT / "moe-static-experts-suite-8192.txt"
LLAMA_CLI = ROOT / "build-gds-cuda/bin/llama-cli"
HUMANEVAL = Path("/home/manohar/Desktop/inference/benchmark_results/HumanEval.jsonl.gz")
GPQA = ROOT / ".cache/real-hard-topk/gpqa/dataset/gpqa_diamond.csv"


def load_humaneval() -> dict[str, dict]:
    with gzip.open(HUMANEVAL, "rt", encoding="utf-8") as f:
        return {item["task_id"]: item for item in map(json.loads, f)}


def load_gpqa() -> list[dict]:
    with GPQA.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_gpqa_task(rows: list[dict], idx: int, task_id: str) -> dict:
    row = rows[idx]
    choices = [
        ("correct", row["Correct Answer"].strip()),
        ("wrong1", row["Incorrect Answer 1"].strip()),
        ("wrong2", row["Incorrect Answer 2"].strip()),
        ("wrong3", row["Incorrect Answer 3"].strip()),
    ]
    rnd = random.Random(20260628 + idx)
    rnd.shuffle(choices)
    labels = "ABCD"
    correct_label = next(labels[i] for i, (kind, _) in enumerate(choices) if kind == "correct")
    options = "\n".join(f"{labels[i]}. {answer}" for i, (_, answer) in enumerate(choices))
    prompt = (
        "This is a GPQA Diamond-style graduate-level multiple-choice question.\n"
        "Reason carefully. At the end, output exactly one line in the format FINAL: <letter>.\n\n"
        f"Question:\n{row['Question'].strip()}\n\nOptions:\n{options}\n"
    )
    return {
        "id": task_id,
        "kind": "gpqa",
        "prompt": prompt,
        "expected": correct_label,
        "source": f"gpqa_diamond.csv row {idx}",
        "domain": row.get("High-level domain", ""),
        "subdomain": row.get("Subdomain", ""),
    }


def make_humaneval_task(items: dict[str, dict], task_id: str) -> dict:
    item = items[task_id]
    prompt = (
        "Complete this OpenAI HumanEval Python problem.\n"
        "You may reason briefly, but your final answer must contain only one Python code block with the complete solution.\n"
        "Do not include tests in the final code block.\n\n"
        f"{item['prompt']}"
    )
    return {
        "id": task_id.replace("/", "_"),
        "kind": "humaneval",
        "prompt": prompt,
        "task": item,
        "expected": "official_tests_pass",
        "source": task_id,
    }


def strip_thinking(text: str) -> str:
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    text = re.sub(r"(?is)\[Start thinking\].*?\[End thinking\]", "", text)
    return text.strip()


def extract_answer_region(text: str) -> str:
    text = text.replace("\b", "")

    # Prefer the generated region over the echoed prompt / loading summaries.
    start = 0
    banner = text.rfind("available commands:")
    if banner >= 0:
        start = banner
        commands_end = re.search(r"/glob <pattern>.*?\n\n", text[start:], flags=re.S)
        if commands_end:
            start += commands_end.end()

    generated_markers = [
        "[Start thinking]",
        "<think>",
        "Here's a thinking process",
        "The user wants",
        "Let's analyze",
        "To solve",
        "We need",
        "The problem asks",
        "FINAL:",
        "Final Answer:",
    ]
    marker_positions = [text.find(marker, start) for marker in generated_markers]
    marker_positions = [pos for pos in marker_positions if pos >= 0]
    if marker_positions:
        start = min(marker_positions)

    end_match = re.search(r"DSTORAGE_MOE_SUMMARY|\[ Prompt:", text[start:])
    end = start + end_match.start() if end_match else len(text)
    region = text[start:end]
    region = re.sub(r"(?m)^\d+\.\d+\.\d+\.\d+ E ensure_experts_loaded:.*\n?", "", region)
    return region.strip()


def parse_gpqa(raw: str) -> tuple[str | None, dict]:
    answer = extract_answer_region(raw)
    visible = strip_thinking(answer)
    final_patterns = [
        r"\bFINAL\s*(?:ANSWER)?\s*[:\-]?\s*\**\s*([ABCD])",
        r"\bFinal\s+Answer\s*[:\-]?\s*\**\s*([ABCD])",
        r"\b(?:correct|final)\s+answer\s+is\s*\**\s*([ABCD])",
        r"\banswer\s+is\s*\**\s*([ABCD])",
    ]
    for pattern in final_patterns:
        finals = re.findall(pattern, visible, flags=re.I)
        if finals:
            return finals[-1].upper(), {
                "has_final_line": bool(re.search(r"(?im)^\s*FINAL\s*:", visible)),
                "visible_len": len(visible),
                "has_self_correction": bool(re.search(r"\b(wait|correction|re-check|recheck|mistake|therefore|actually|however)\b", visible, re.I)),
            }
    matches = re.findall(r"\b([ABCD])\b", visible)
    return (matches[-1].upper() if matches else None), {
        "has_final_line": False,
        "visible_len": len(visible),
        "has_self_correction": bool(re.search(r"\b(wait|correction|re-check|recheck|mistake|therefore|actually|however)\b", visible, re.I)),
    }


def extract_code(raw: str) -> str:
    answer = strip_thinking(extract_answer_region(raw))
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", answer, flags=re.S | re.I)
    if blocks:
        return max(blocks, key=len).strip()
    if "```" in answer:
        blocks = re.findall(r"```\s*\n(.*?)```", answer, flags=re.S)
        if blocks:
            return max(blocks, key=len).strip()
    return answer.strip()


def humaneval_candidates(task: dict, raw: str) -> list[str]:
    item = task["task"]
    prompt = item["prompt"].rstrip() + "\n"
    code = extract_code(raw)
    candidates = []
    if f"def {item['entry_point']}" in code:
        candidates.append(code)
    candidates.append(prompt + code)
    candidates.append(code)
    uniq = []
    seen = set()
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            uniq.append(cand)
    return uniq


def run_humaneval_tests(task: dict, raw: str) -> tuple[bool, dict]:
    item = task["task"]
    last_error = ""
    chosen = None
    for candidate in humaneval_candidates(task, raw):
        try:
            compile(candidate, "<candidate>", "exec")
            chosen = candidate
            break
        except SyntaxError as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"
    if chosen is None:
        return False, {"compile_error": last_error, "code_len": len(extract_code(raw))}

    program = chosen + "\n\n" + item["test"] + f"\n\ncheck({item['entry_point']})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(program)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        return proc.returncode == 0, {
            "code_len": len(chosen),
            "stderr_tail": (proc.stderr or proc.stdout)[-1000:],
        }
    except subprocess.TimeoutExpired:
        return False, {"code_len": len(chosen), "stderr_tail": "TimeoutExpired"}
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def parse_metrics(raw: str) -> dict:
    metrics = {}
    m = re.search(r"\[ Prompt:\s*([0-9.]+) t/s \| Generation:\s*([0-9.]+) t/s \]", raw)
    if m:
        metrics["prompt_tps"] = float(m.group(1))
        metrics["generation_tps"] = float(m.group(2))
    timelines = re.findall(r"DSTORAGE_MOE_TIMELINE ([^\n]+)", raw)
    if timelines:
        line = timelines[-1]
        for key in ("selected_experts", "stream_file_mib", "hit_rate"):
            km = re.search(rf"{key}=([0-9.]+)", line)
            if km:
                value = km.group(1)
                metrics[key] = float(value) if "." in value else int(value)
    phases = re.findall(r"DSTORAGE_MOE_PHASE ([^\n]+)", raw)
    if phases:
        dm = re.search(r"decode_hit_rate=([0-9.]+)", phases[-1])
        if dm:
            metrics["decode_hit_rate"] = float(dm.group(1))
    return metrics


def run_task(task: dict, mode: str, out_dir: Path, coding_reasoning: str) -> dict:
    env = os.environ.copy()
    env.update({
        "LLAMA_MOE_PINNED_PREFILL_ONLY": "1",
        "LLAMA_GDS_READ_THREADS": "20",
        "LLAMA_DSTORAGE_SUMMARY": "1",
        "LLAMA_MOE_PINNED_STATIC_EXPERTS_FILE": str(STATIC_EXPERTS),
    })
    if mode != "k8":
        env["LLAMA_MOE_EXPERT_USED_OVERRIDE"] = mode[1:]
    else:
        env.pop("LLAMA_MOE_EXPERT_USED_OVERRIDE", None)

    out = out_dir / f"{task['id']}__{mode}.out"
    if task["kind"] == "humaneval" and coding_reasoning == "off":
        reasoning_args = ["--reasoning", "off", "--reasoning-budget", "0"]
        max_tokens = "650"
    else:
        reasoning_args = [
            "--reasoning", "on",
            "--reasoning-format", "deepseek-legacy",
            "--reasoning-budget", "384",
        ]
        max_tokens = "900"

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
        *reasoning_args,
        "-s", "4242",
        "-st",
        "--no-display-prompt",
        "--simple-io",
        "-n", max_tokens,
        "-p", task["prompt"],
    ]
    start = time.time()
    with out.open("w", encoding="utf-8", errors="replace") as f:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=1500)
    elapsed = time.time() - start
    raw = out.read_text(encoding="utf-8", errors="replace")

    if task["kind"] == "gpqa":
        predicted, pattern = parse_gpqa(raw)
        passed = predicted == task["expected"]
        eval_info = {"predicted": predicted, "pattern": pattern}
    else:
        passed, eval_info = run_humaneval_tests(task, raw)

    return {
        "task_id": task["id"],
        "kind": task["kind"],
        "mode": mode,
        "expected": task["expected"],
        "passed": passed,
        "elapsed_sec": round(elapsed, 2),
        "exit_code": proc.returncode,
        "source": task["source"],
        "eval": eval_info,
        "output": str(out),
        **parse_metrics(raw),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default="k8,k6,k5,k4")
    parser.add_argument("--suite", choices=["small", "broad"], default="small")
    parser.add_argument(
        "--coding-reasoning",
        choices=["on", "off"],
        default="on",
        help="Reasoning mode for HumanEval coding tasks. GPQA always uses reasoning on.",
    )
    args = parser.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    he = load_humaneval()
    gpqa = load_gpqa()
    if args.suite == "broad":
        tasks = [
            make_gpqa_task(gpqa, 2, "gpqa_quantum_expectation"),
            make_gpqa_task(gpqa, 50, "gpqa_grounded_sphere_energy"),
            make_gpqa_task(gpqa, 1, "gpqa_organic_synthesis_count"),
            make_gpqa_task(gpqa, 16, "gpqa_calcium_edta"),
            make_gpqa_task(gpqa, 7, "gpqa_lupine_genetics"),
            make_gpqa_task(gpqa, 10, "gpqa_sars_cov2_molecular"),
            make_humaneval_task(he, "HumanEval/95"),
            make_humaneval_task(he, "HumanEval/81"),
            make_humaneval_task(he, "HumanEval/153"),
            make_humaneval_task(he, "HumanEval/124"),
            make_humaneval_task(he, "HumanEval/141"),
            make_humaneval_task(he, "HumanEval/129"),
        ]
    else:
        tasks = [
            make_gpqa_task(gpqa, 2, "gpqa_quantum_expectation"),
            make_gpqa_task(gpqa, 50, "gpqa_grounded_sphere_energy"),
            make_humaneval_task(he, "HumanEval/129"),
            make_humaneval_task(he, "HumanEval/95"),
        ]

    out_dir = ROOT / "topk-quality-sweep-results" / "real-hard" / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tasks.json").write_text(json.dumps([
        {k: v for k, v in t.items() if k not in {"task"}}
        for t in tasks
    ], indent=2), encoding="utf-8")

    results = []
    for task in tasks:
        for mode in modes:
            print(f"RUN {mode} {task['id']} ({task['kind']})", flush=True)
            result = run_task(task, mode, out_dir, args.coding_reasoning)
            results.append(result)
            print(json.dumps({
                "task": result["task_id"],
                "mode": mode,
                "passed": result["passed"],
                "exit_code": result["exit_code"],
                "prompt_tps": result.get("prompt_tps"),
                "generation_tps": result.get("generation_tps"),
                "stream_file_mib": result.get("stream_file_mib"),
                "eval": result["eval"],
            }), flush=True)

    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# Real Hard Top-K Evaluation",
        "",
        "| Task | Kind | Mode | Pass | Exit | Prompt t/s | Gen t/s | Stream MiB | Selected Experts | Eval |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        eval_short = json.dumps(r["eval"], ensure_ascii=True)
        if len(eval_short) > 160:
            eval_short = eval_short[:157] + "..."
        lines.append(
            f"| {r['task_id']} | {r['kind']} | {r['mode']} | {str(r['passed']).lower()} | {r['exit_code']} | "
            f"{r.get('prompt_tps')} | {r.get('generation_tps')} | {r.get('stream_file_mib')} | "
            f"{r.get('selected_experts')} | `{eval_short}` |"
        )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"RESULT_DIR={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
