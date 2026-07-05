#!/usr/bin/env python3

import argparse
import hashlib
import json
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


API_URL = "https://datasets-server.huggingface.co/rows"
PAGE_SIZE = 100

BBH_CONFIGS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "logical_deduction_five_objects",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "reasoning_about_colored_objects",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "web_of_lies",
    "word_sorting",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic, balanced prompt suite for MoE routing traces."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("moe-trace-prompts-1000.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("moe-trace-prompts-1000.manifest.json"),
        help="Output manifest path",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/moe-trace-prompts"),
        help="Dataset Viewer response cache",
    )
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached Dataset Viewer responses",
    )
    parser.add_argument(
        "--exclude-prompts",
        type=Path,
        help="JSONL prompt suite whose IDs must be excluded",
    )
    return parser.parse_args()


def cache_path(cache_dir: Path, dataset: str, config: str, split: str) -> Path:
    key = "__".join((dataset.replace("/", "--"), config, split))
    return cache_dir / f"{key}.json"


def fetch_rows(
    dataset: str,
    config: str,
    split: str,
    cache_dir: Path,
    refresh: bool,
) -> list[dict[str, Any]]:
    path = cache_path(cache_dir, dataset, config, split)
    partial_path = path.with_suffix(".partial.json")
    if path.exists() and not refresh:
        with path.open("r", encoding="utf-8") as cached:
            return json.load(cached)

    rows: list[dict[str, Any]] = []
    if partial_path.exists() and not refresh:
        with partial_path.open("r", encoding="utf-8") as cached:
            rows = json.load(cached)
    offset = len(rows)
    total = None
    while total is None or offset < total:
        query = urllib.parse.urlencode(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": PAGE_SIZE,
            }
        )
        request = urllib.request.Request(
            f"{API_URL}?{query}",
            headers={"User-Agent": "moe-routing-trace-suite/1.0"},
        )
        for attempt in range(10):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    page = json.load(response)
                break
            except HTTPError as error:
                if attempt == 9 or error.code != 429:
                    raise
                retry_after = int(error.headers.get("Retry-After", "0") or 0)
                delay = max(retry_after, min(60, 5 * (attempt + 1)))
                print(f"rate limited; retrying {dataset}/{config}/{split} in {delay}s")
                time.sleep(delay)
            except Exception:
                if attempt == 9:
                    raise
                time.sleep(min(30, 2**attempt))

        total = int(page["num_rows_total"])
        page_rows = [entry["row"] for entry in page["rows"]]
        rows.extend(page_rows)
        offset += len(page_rows)
        cache_dir.mkdir(parents=True, exist_ok=True)
        with partial_path.open("w", encoding="utf-8") as cached:
            json.dump(rows, cached, ensure_ascii=False)
        if not page_rows:
            break
        print(f"fetched {dataset}/{config}/{split}: {len(rows)}/{total}")
        time.sleep(0.5)

    cache_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as cached:
        json.dump(rows, cached, ensure_ascii=False)
    partial_path.unlink(missing_ok=True)
    return rows


def stable_key(seed: int, source: str, row_id: str) -> bytes:
    value = f"{seed}\0{source}\0{row_id}".encode("utf-8")
    return hashlib.sha256(value).digest()


def sample_rows(
    rows: list[dict[str, Any]],
    count: int,
    seed: int,
    source: str,
    id_fn: Callable[[dict[str, Any], int], str],
) -> list[dict[str, Any]]:
    ranked = sorted(
        enumerate(rows),
        key=lambda item: stable_key(seed, source, id_fn(item[1], item[0])),
    )
    return [row for _, row in ranked[:count]]


def stratified_sample(
    rows: list[dict[str, Any]],
    count: int,
    seed: int,
    source: str,
    category_fn: Callable[[dict[str, Any]], str],
    id_fn: Callable[[dict[str, Any], int], str],
) -> list[dict[str, Any]]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[category_fn(row)].append((index, row))

    for category, values in groups.items():
        values.sort(
            key=lambda item: stable_key(
                seed, f"{source}/{category}", id_fn(item[1], item[0])
            )
        )

    selected: list[dict[str, Any]] = []
    categories = sorted(groups)
    cursor = 0
    while len(selected) < count:
        made_progress = False
        for category in categories:
            values = groups[category]
            if cursor < len(values):
                selected.append(values[cursor][1])
                made_progress = True
                if len(selected) == count:
                    break
        if not made_progress:
            raise ValueError(f"{source} only has {len(selected)} usable rows")
        cursor += 1
    return selected


def record(
    source: str,
    source_id: str,
    family: str,
    category: str,
    prompt: str,
    license_name: str,
) -> dict[str, Any]:
    return {
        "id": f"{source}:{source_id}",
        "family": family,
        "source": source,
        "category": category,
        "prompt": prompt.strip(),
        "license": license_name,
    }


def format_options(options: list[Any]) -> str:
    return "\n".join(
        f"{chr(ord('A') + index)}. {option}" for index, option in enumerate(options)
    )


def row_hash(*values: Any) -> str:
    text = "".join(str(value) for value in values)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def exclude_rows(
    rows: list[dict[str, Any]],
    source: str,
    id_fn: Callable[[dict[str, Any], int], str],
    excluded_ids: set[str],
) -> list[dict[str, Any]]:
    return [
        row
        for index, row in enumerate(rows)
        if f"{source}:{id_fn(row, index)}" not in excluded_ids
    ]


def build_suite(args: argparse.Namespace, excluded_ids: set[str]) -> list[dict[str, Any]]:
    suite: list[dict[str, Any]] = []

    bigcode = fetch_rows(
        "bigcode/bigcodebench", "default", "v0.1.4", args.cache_dir, args.refresh
    )
    bigcode = exclude_rows(
        bigcode,
        "bigcodebench",
        lambda row, i: str(row["task_id"]),
        excluded_ids,
    )
    for row in sample_rows(
        bigcode, 350, args.seed, "bigcodebench", lambda row, i: str(row["task_id"])
    ):
        libraries = row.get("libs") or ["standard-library"]
        category = ",".join(str(value) for value in libraries[:3])
        suite.append(
            record(
                "bigcodebench",
                str(row["task_id"]),
                "coding",
                category,
                str(row["instruct_prompt"]),
                "Apache-2.0",
            )
        )

    mbpp = fetch_rows(
        "Muennighoff/mbpp", "sanitized", "test", args.cache_dir, args.refresh
    )
    mbpp = exclude_rows(
        mbpp,
        "mbpp-sanitized",
        lambda row, i: str(row["task_id"]),
        excluded_ids,
    )
    for row in sample_rows(
        mbpp, 150, args.seed, "mbpp-sanitized", lambda row, i: str(row["task_id"])
    ):
        tests = "\n".join(str(value) for value in row.get("test_list", []))
        prompt = f"{row['prompt']}\n\nYour solution must pass these tests:\n{tests}"
        suite.append(
            record(
                "mbpp-sanitized",
                str(row["task_id"]),
                "coding",
                "python-algorithms",
                prompt,
                "CC-BY-4.0",
            )
        )

    math_rows = fetch_rows(
        "HuggingFaceH4/MATH-500", "default", "test", args.cache_dir, args.refresh
    )
    math_rows = exclude_rows(
        math_rows,
        "math-500",
        lambda row, i: str(row["unique_id"]),
        excluded_ids,
    )
    math_sample = stratified_sample(
        math_rows,
        150,
        args.seed,
        "math-500",
        lambda row: f"{row['subject']}/{row['level']}",
        lambda row, i: str(row["unique_id"]),
    )
    for row in math_sample:
        suite.append(
            record(
                "math-500",
                str(row["unique_id"]),
                "reasoning",
                f"{row['subject']}/{row['level']}",
                f"{row['problem']}\n\nShow your reasoning and give the final answer.",
                "MIT",
            )
        )

    for config in BBH_CONFIGS:
        rows = fetch_rows(
            "maveriq/bigbenchhard", config, "train", args.cache_dir, args.refresh
        )
        rows = exclude_rows(
            rows,
            "bigbenchhard",
            lambda row, i: f"{config}-{row_hash(row['input'])}",
            excluded_ids,
        )
        chosen = sample_rows(
            rows,
            10,
            args.seed,
            f"bigbenchhard/{config}",
            lambda row, i: row_hash(row["input"]),
        )
        for row in chosen:
            source_id = row_hash(row["input"])
            suite.append(
                record(
                    "bigbenchhard",
                    f"{config}-{source_id}",
                    "reasoning",
                    config,
                    f"{row['input']}\n\nReason carefully and give the final answer.",
                    "MIT",
                )
            )

    mmlu = fetch_rows(
        "TIGER-Lab/MMLU-Pro", "default", "test", args.cache_dir, args.refresh
    )
    mmlu = exclude_rows(
        mmlu,
        "mmlu-pro",
        lambda row, i: str(row["question_id"]),
        excluded_ids,
    )
    mmlu_sample = stratified_sample(
        mmlu,
        150,
        args.seed,
        "mmlu-pro",
        lambda row: str(row["category"]),
        lambda row, i: str(row["question_id"]),
    )
    for row in mmlu_sample:
        prompt = (
            f"{row['question']}\n\n{format_options(row['options'])}\n\n"
            "Reason carefully, then give the best answer."
        )
        suite.append(
            record(
                "mmlu-pro",
                str(row["question_id"]),
                "reasoning",
                str(row["category"]),
                prompt,
                "MIT",
            )
        )

    musr_counts = {
        "murder_mysteries": 20,
        "object_placements": 15,
        "team_allocation": 15,
    }
    for split, count in musr_counts.items():
        rows = fetch_rows(
            "TAUR-Lab/MuSR", "default", split, args.cache_dir, args.refresh
        )
        rows = exclude_rows(
            rows,
            "musr",
            lambda row, i: f"{split}-{row_hash(row['narrative'], row['question'])}",
            excluded_ids,
        )
        chosen = sample_rows(
            rows,
            count,
            args.seed,
            f"musr/{split}",
            lambda row, i: row_hash(row["narrative"], row["question"]),
        )
        for row in chosen:
            source_id = row_hash(row["narrative"], row["question"])
            prompt = (
                f"{row['narrative']}\n\nQuestion: {row['question']}\n\n"
                f"{format_options(row['choices'])}\n\n"
                "Reason through the evidence and give the best answer."
            )
            suite.append(
                record(
                    "musr",
                    f"{split}-{source_id}",
                    "reasoning",
                    split,
                    prompt,
                    "CC-BY-4.0",
                )
            )

    return suite


def main() -> None:
    args = parse_args()
    excluded_ids: set[str] = set()
    if args.exclude_prompts:
        with args.exclude_prompts.open("r", encoding="utf-8") as excluded:
            excluded_ids = {str(json.loads(line)["id"]) for line in excluded}
        print(f"excluding {len(excluded_ids)} prompt IDs from {args.exclude_prompts}")

    suite = build_suite(args, excluded_ids)
    suite.sort(key=lambda item: stable_key(args.seed, "final-suite", item["id"]))

    ids = [item["id"] for item in suite]
    if len(suite) != 1000 or len(ids) != len(set(ids)):
        raise ValueError(
            f"expected 1,000 unique prompts, got {len(suite)} rows and "
            f"{len(set(ids))} unique IDs"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for item in suite:
            output.write(json.dumps(item, ensure_ascii=False) + "\n")

    counts: dict[str, dict[str, int]] = {}
    for field in ("family", "source", "category", "license"):
        values: dict[str, int] = defaultdict(int)
        for item in suite:
            values[str(item[field])] += 1
        counts[field] = dict(sorted(values.items()))

    manifest = {
        "seed": args.seed,
        "excluded_prompt_file": str(args.exclude_prompts) if args.exclude_prompts else None,
        "excluded_prompt_count": len(excluded_ids),
        "total_prompts": len(suite),
        "output": str(args.output),
        "counts": counts,
        "sources": {
            "bigcodebench": "https://huggingface.co/datasets/bigcode/bigcodebench",
            "mbpp-sanitized": "https://huggingface.co/datasets/Muennighoff/mbpp",
            "math-500": "https://huggingface.co/datasets/HuggingFaceH4/MATH-500",
            "bigbenchhard": "https://huggingface.co/datasets/maveriq/bigbenchhard",
            "mmlu-pro": "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro",
            "musr": "https://huggingface.co/datasets/TAUR-Lab/MuSR",
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, ensure_ascii=False)
        output.write("\n")

    print(f"wrote {len(suite)} prompts to {args.output}")
    print(f"coding={counts['family']['coding']} reasoning={counts['family']['reasoning']}")


if __name__ == "__main__":
    main()
