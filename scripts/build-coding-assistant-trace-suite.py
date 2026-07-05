#!/usr/bin/env python3

"""Build a coding-assistant prompt suite for MoE routing traces.

The script intentionally uses only the Python standard library. Public datasets
are sampled through the Hugging Face Dataset Viewer API, while local Codex
rollouts are read from the read-only state database. Generated records retain the
``prompt`` field consumed by ``run-moe-routing-trace.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


API_ROOT = "https://datasets-server.huggingface.co"
PAGE_SIZE = 100


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset: str
    count: int
    adapter: str
    license_name: str = "see-source-card"
    preferred_split: str = "train"


PUBLIC_SOURCES = (
    SourceSpec("open-swe-traces", "nvidia/Open-SWE-Traces", 800, "trajectory"),
    SourceSpec(
        "swe-rebench-openhands",
        "nebius/SWE-rebench-openhands-trajectories",
        500,
        "trajectory",
        "Apache-2.0",
    ),
    SourceSpec(
        "opencodeinstruct",
        "nvidia/OpenCodeInstruct",
        400,
        "instruction",
        "CC-BY-4.0",
    ),
    SourceSpec("commitpackft", "bigcode/commitpackft", 200, "commit"),
    SourceSpec(
        "swe-bench-verified",
        "SWE-bench/SWE-bench_Verified",
        50,
        "issue",
        preferred_split="test",
    ),
    SourceSpec("swe-gym", "SWE-Gym/SWE-Gym", 50, "issue"),
)

COMMITPACK_LANGUAGE_COUNTS = {
    "python": 40,
    "javascript": 25,
    "typescript": 25,
    "c++": 20,
    "java": 20,
    "c": 15,
    "go": 15,
    "rust": 15,
    "shell": 10,
    "kotlin": 5,
    "php": 5,
    "ruby": 5,
}


SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\baero_live_[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\b(?:gh[pousr]|hf)_[A-Za-z0-9_-]{20,}\b"),
    re.compile(
        r"(?i)(\b(?:api[_ -]?key|access[_ -]?token|auth[_ -]?token|password|secret)"
        r"\b\s*(?:=|:)\s*)([^\s'\"`]{8,})"
    ),
    re.compile(r"(?i)(\bAuthorization\s*:\s*Bearer\s+)[A-Za-z0-9._~+/-]{12,}"),
)

IDE_REQUEST_MARKER = "## My request for Codex:"
ROLE_LABELS = {
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool",
    "environment": "Tool",
    "observation": "Tool",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic coding-assistant MoE trace suite."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/coding-assistant-trace/combined-suite.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(".cache/coding-assistant-trace/combined-suite.manifest.json"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/coding-assistant-trace"),
    )
    parser.add_argument(
        "--codex-state",
        type=Path,
        default=Path.home() / ".codex/state_5.sqlite",
    )
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--max-prompt-chars", type=int, default=32768)
    parser.add_argument("--skip-codex", action="store_true")
    parser.add_argument("--skip-public", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def stable_digest(seed: int, *parts: Any) -> bytes:
    value = "\0".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(value.encode("utf-8")).digest()


def short_hash(*parts: Any) -> str:
    value = "\0".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def redact(text: str, stats: Counter[str]) -> str:
    text = text.replace("/home/manohar", "$HOME")
    for index, pattern in enumerate(SECRET_PATTERNS):
        def replacement(match: re.Match[str]) -> str:
            stats[f"secret_pattern_{index}"] += 1
            if match.lastindex and match.lastindex >= 2:
                return f"{match.group(1)}[REDACTED]"
            if match.lastindex == 1:
                return f"{match.group(1)}[REDACTED]"
            return "[REDACTED_SECRET]"

        text = pattern.sub(replacement, text)
    return text


def strip_ide_wrapper(text: str) -> str:
    if IDE_REQUEST_MARKER in text:
        return text.split(IDE_REQUEST_MARKER, 1)[1].strip()
    return text.strip()


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        for key in ("text", "content", "value", "message", "output"):
            value = content.get(key)
            if isinstance(value, (str, list, dict)):
                result = content_text(value)
                if result:
                    return result
        return ""
    if isinstance(content, list):
        values = [content_text(item) for item in content]
        return "\n".join(value for value in values if value)
    return ""


def parse_json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def normalize_messages(value: Any) -> list[tuple[str, str]]:
    value = parse_json_value(value)
    if isinstance(value, dict):
        for key in ("messages", "trajectory", "conversation", "conversations"):
            if key in value:
                return normalize_messages(value[key])
        return []
    if not isinstance(value, list):
        return []

    messages: list[tuple[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        role = str(
            item.get("role")
            or item.get("from")
            or item.get("speaker")
            or item.get("type")
            or ""
        ).lower()
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        text = content_text(
            item.get("content", item.get("value", item.get("text", "")))
        ).strip()
        if role in ROLE_LABELS and text:
            messages.append((role, text))
    return messages


def render_messages(
    messages: Iterable[tuple[str, str]],
    max_chars: int,
    stats: Counter[str],
) -> str:
    blocks: list[str] = []
    for role, raw_text in messages:
        if role not in ROLE_LABELS:
            continue
        text = strip_ide_wrapper(raw_text) if role == "user" else raw_text.strip()
        text = redact(text, stats)
        if text:
            blocks.append(f"{ROLE_LABELS[role]}:\n{text}")

    while blocks and len("\n\n".join(blocks)) > max_chars:
        if len(blocks) > 2:
            blocks.pop(0)
            stats["context_blocks_trimmed"] += 1
        else:
            excess = len("\n\n".join(blocks)) - max_chars
            blocks[0] = blocks[0][excess:]
            stats["oversized_blocks_trimmed"] += 1
            break
    return "\n\n".join(blocks).strip()


def infer_category(text: str) -> str:
    lowered = text.lower()
    categories = (
        ("debugging", ("error", "bug", "fix", "failed", "traceback", "crash")),
        ("testing", ("test", "pytest", "benchmark", "verify")),
        ("review", ("review", "audit", "security", "vulnerability")),
        ("refactoring", ("refactor", "cleanup", "simplify", "optimize")),
        ("repository", ("repository", "repo", "codebase", "file", "commit")),
        ("tooling", ("terminal", "command", "build", "install", "configure")),
    )
    for category, needles in categories:
        if any(needle in lowered for needle in needles):
            return category
    return "implementation"


def record(
    source: str,
    source_id: str,
    prompt: str,
    category: str,
    license_name: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": f"{source}:{source_id}",
        "family": "coding-assistant",
        "source": source,
        "category": category,
        "prompt": prompt.strip(),
        "license": license_name,
    }
    if metadata:
        result["metadata"] = metadata
    return result


def extract_codex_records(
    state_path: Path,
    max_chars: int,
    stats: Counter[str],
) -> list[dict[str, Any]]:
    if not state_path.exists():
        raise FileNotFoundError(f"Codex state database not found: {state_path}")

    connection = sqlite3.connect(f"file:{state_path}?mode=ro", uri=True)
    threads = connection.execute(
        "SELECT id, rollout_path FROM threads ORDER BY created_at"
    ).fetchall()
    records: list[dict[str, Any]] = []
    for thread_id, rollout_path in threads:
        path = Path(rollout_path)
        if not path.exists():
            stats["missing_rollouts"] += 1
            continue
        history: list[tuple[str, str]] = []
        turn = 0
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                try:
                    payload = json.loads(line).get("payload", {})
                except json.JSONDecodeError:
                    stats["invalid_rollout_lines"] += 1
                    continue
                if payload.get("type") != "message":
                    continue
                role = payload.get("role")
                phase = payload.get("phase")
                if role == "assistant" and phase != "final_answer":
                    continue
                if role not in ("user", "assistant"):
                    continue
                text = content_text(payload.get("content", [])).strip()
                if not text:
                    continue
                history.append((role, text))
                if role != "user":
                    continue
                prompt = render_messages(history, max_chars, stats)
                if len(prompt) < 4:
                    continue
                records.append(
                    record(
                        "local-codex",
                        f"{short_hash(thread_id)}-{turn:04d}",
                        prompt,
                        infer_category(strip_ide_wrapper(text)),
                        "private-local",
                        {"conversation": short_hash(thread_id), "turn": turn},
                    )
                )
                turn += 1
    connection.close()
    return records


def cached_json(
    url: str,
    path: Path,
    refresh: bool,
    retries: int = 10,
) -> dict[str, Any]:
    if path.exists() and not refresh:
        with path.open("r", encoding="utf-8") as source:
            return json.load(source)
    request = urllib.request.Request(
        url, headers={"User-Agent": "coding-assistant-moe-trace/1.0"}
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                data = json.load(response)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as output:
                json.dump(data, output, ensure_ascii=False)
            return data
        except urllib.error.HTTPError as error:
            if attempt + 1 == retries:
                raise
            if error.code == 429:
                retry_after = int(error.headers.get("Retry-After", "0") or 0)
                delay = min(120, max(retry_after, 10 * (attempt + 1)))
                print(f"rate limited; retrying in {delay}s", flush=True)
                time.sleep(delay)
            else:
                time.sleep(min(30, 2 ** attempt))
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(min(30, 2 ** attempt))
    raise AssertionError("unreachable")


def discover_split(
    spec: SourceSpec, cache_dir: Path, refresh: bool
) -> tuple[str, str]:
    query = urllib.parse.urlencode({"dataset": spec.dataset})
    data = cached_json(
        f"{API_ROOT}/splits?{query}",
        cache_dir / spec.name / "splits.json",
        refresh,
    )
    splits = data.get("splits", [])
    if not splits:
        raise ValueError(f"Dataset Viewer returned no splits for {spec.dataset}")
    preferred = [row for row in splits if row.get("split") == spec.preferred_split]
    selected = (preferred or splits)[0]
    return str(selected["config"]), str(selected["split"])


def fetch_page(
    spec: SourceSpec,
    config: str,
    split: str,
    offset: int,
    cache_dir: Path,
    refresh: bool,
) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "dataset": spec.dataset,
            "config": config,
            "split": split,
            "offset": offset,
            "length": PAGE_SIZE,
        }
    )
    page_dir = cache_dir / spec.name
    if spec.adapter == "commit":
        page_dir = page_dir / f"config-{short_hash(config)}"
    return cached_json(
        f"{API_ROOT}/rows?{query}",
        page_dir / f"rows-{offset:09d}.json",
        refresh,
    )


def first_value(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            text = content_text(value).strip()
            if text:
                return text
    return ""


def trajectory_prompt(
    row: dict[str, Any], max_chars: int, stats: Counter[str]
) -> str:
    for key in ("trajectory", "messages", "conversation", "conversations"):
        messages = normalize_messages(row.get(key))
        if messages:
            prompt = render_messages(messages, max_chars - 80, stats)
            if prompt:
                return f"{prompt}\n\nUser:\nContinue working on this coding task."
    issue = first_value(row, ("problem_statement", "issue", "prompt", "instruction"))
    return redact(issue, stats)


def instruction_prompt(row: dict[str, Any], stats: Counter[str]) -> str:
    instruction = first_value(
        row, ("input", "instruction", "prompt", "question", "problem")
    )
    context = first_value(row, ("context", "code", "starter_code"))
    if context and context not in instruction:
        instruction = f"{instruction}\n\nRelevant code/context:\n{context}"
    return redact(instruction, stats)


def commit_prompt(row: dict[str, Any], stats: Counter[str]) -> str:
    request = first_value(
        row, ("message", "commit_message", "subject", "instruction", "prompt")
    )
    old_code = first_value(
        row, ("old_contents", "old_content", "before", "source", "input")
    )
    if not request:
        return ""
    prompt = f"Implement this repository change:\n\n{request}"
    if old_code:
        prompt += f"\n\nCurrent code:\n```\n{old_code}\n```"
    return redact(prompt, stats)


def issue_prompt(row: dict[str, Any], stats: Counter[str]) -> str:
    issue = first_value(
        row, ("problem_statement", "issue", "prompt", "instruction", "description")
    )
    repo = first_value(row, ("repo", "repository", "repo_name"))
    if not issue:
        return ""
    prefix = f"Repository: {repo}\n\n" if repo else ""
    return redact(f"{prefix}{issue}", stats)


ADAPTERS: dict[str, Callable[[dict[str, Any], int, Counter[str]], str]] = {
    "trajectory": lambda row, limit, stats: trajectory_prompt(row, limit, stats),
    "instruction": lambda row, limit, stats: instruction_prompt(row, stats),
    "commit": lambda row, limit, stats: commit_prompt(row, stats),
    "issue": lambda row, limit, stats: issue_prompt(row, stats),
}


def usable_public_prompt(spec: SourceSpec, prompt: str) -> bool:
    if len(prompt) < 20:
        return False
    if spec.adapter != "instruction":
        return True
    lowered = prompt.lower()
    competitive_markers = (
        "competitive programming",
        "time limit per test",
        "standard input",
        "codeforces",
        "print the answer modulo",
    )
    return not any(marker in lowered for marker in competitive_markers)


def build_public_source(
    spec: SourceSpec,
    args: argparse.Namespace,
    stats: Counter[str],
) -> list[dict[str, Any]]:
    config, split = discover_split(spec, args.cache_dir, args.refresh)
    first = fetch_page(spec, config, split, 0, args.cache_dir, args.refresh)
    total = int(first.get("num_rows_total", 0))
    if total <= 0:
        raise ValueError(f"No rows available for {spec.dataset}/{config}/{split}")

    offsets = list(range(0, total, PAGE_SIZE))
    offsets.sort(key=lambda value: stable_digest(args.seed, spec.name, value))
    adapter = ADAPTERS[spec.adapter]
    candidates: dict[str, dict[str, Any]] = {}
    target_candidates = min(total, max(spec.count * 2, spec.count + 100))
    for offset in offsets:
        page = first if offset == 0 else fetch_page(
            spec, config, split, offset, args.cache_dir, args.refresh
        )
        for entry in page.get("rows", []):
            row = entry.get("row", {})
            prompt = adapter(row, args.max_prompt_chars, stats).strip()
            if not usable_public_prompt(spec, prompt):
                stats[f"{spec.name}_filtered"] += 1
                continue
            if len(prompt) > args.max_prompt_chars:
                prompt = prompt[-args.max_prompt_chars :]
                stats["public_prompts_trimmed"] += 1
            source_id = str(
                row.get("instance_id")
                or row.get("id")
                or row.get("task_id")
                or short_hash(spec.name, prompt)
            )
            unique_id = short_hash(source_id, prompt)
            candidates[unique_id] = record(
                spec.name,
                unique_id,
                prompt,
                infer_category(prompt),
                spec.license_name,
                {"dataset": spec.dataset, "config": config, "split": split},
            )
        print(
            f"{spec.name}: candidates={len(candidates)}/{target_candidates} "
            f"pages={stats[f'{spec.name}_pages'] + 1}",
            flush=True,
        )
        stats[f"{spec.name}_pages"] += 1
        if len(candidates) >= target_candidates:
            break

    if len(candidates) < spec.count:
        raise ValueError(
            f"{spec.name} produced {len(candidates)} usable records; "
            f"expected {spec.count}. Check its current Dataset Viewer schema."
        )
    ranked = sorted(
        candidates.values(),
        key=lambda item: stable_digest(args.seed, spec.name, item["id"]),
    )
    return ranked[: spec.count]


def build_commitpack_source(
    spec: SourceSpec,
    args: argparse.Namespace,
    stats: Counter[str],
) -> list[dict[str, Any]]:
    split_query = urllib.parse.urlencode({"dataset": spec.dataset})
    split_data = cached_json(
        f"{API_ROOT}/splits?{split_query}",
        args.cache_dir / spec.name / "splits.json",
        args.refresh,
    )
    available = {
        (str(row.get("config")), str(row.get("split")))
        for row in split_data.get("splits", [])
    }
    selected: list[dict[str, Any]] = []
    for config, count in COMMITPACK_LANGUAGE_COUNTS.items():
        if (config, "train") not in available:
            raise ValueError(f"CommitPackFT configuration is unavailable: {config}")
        first = fetch_page(spec, config, "train", 0, args.cache_dir, args.refresh)
        total = int(first.get("num_rows_total", 0))
        if total < count:
            raise ValueError(
                f"CommitPackFT/{config} has {total} rows; expected at least {count}"
            )
        offsets = list(range(0, total, PAGE_SIZE))
        offsets.sort(key=lambda value: stable_digest(args.seed, spec.name, config, value))
        candidates: dict[str, dict[str, Any]] = {}
        wanted = min(total, max(count * 2, count + 20))
        for offset in offsets:
            page = first if offset == 0 else fetch_page(
                spec, config, "train", offset, args.cache_dir, args.refresh
            )
            for entry in page.get("rows", []):
                row = entry.get("row", {})
                prompt = commit_prompt(row, stats).strip()
                if not usable_public_prompt(spec, prompt):
                    continue
                if len(prompt) > args.max_prompt_chars:
                    prompt = prompt[-args.max_prompt_chars :]
                    stats["public_prompts_trimmed"] += 1
                unique_id = short_hash(config, row.get("commit"), prompt)
                candidates[unique_id] = record(
                    spec.name,
                    unique_id,
                    prompt,
                    f"maintenance/{config}",
                    spec.license_name,
                    {
                        "dataset": spec.dataset,
                        "config": config,
                        "split": "train",
                    },
                )
            stats[f"{spec.name}_{config}_pages"] += 1
            if len(candidates) >= wanted:
                break
        if len(candidates) < count:
            raise ValueError(
                f"CommitPackFT/{config} produced {len(candidates)} usable records; "
                f"expected {count}"
            )
        ranked = sorted(
            candidates.values(),
            key=lambda item: stable_digest(args.seed, spec.name, config, item["id"]),
        )
        selected.extend(ranked[:count])
        print(
            f"{spec.name}/{config}: selected={count} candidates={len(candidates)}",
            flush=True,
        )
    if len(selected) != spec.count:
        raise AssertionError(
            f"CommitPackFT mix selected {len(selected)} records; expected {spec.count}"
        )
    return selected


def count_values(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in records:
        counts[str(item[field])] += 1
    return dict(sorted(counts.items()))


def main() -> None:
    args = parse_args()
    if args.skip_codex and args.skip_public:
        raise ValueError("--skip-codex and --skip-public cannot be used together")
    if args.max_prompt_chars < 1024:
        raise ValueError("--max-prompt-chars must be at least 1024")

    stats: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    if not args.skip_codex:
        local = extract_codex_records(args.codex_state, args.max_prompt_chars, stats)
        records.extend(local)
        print(f"local-codex: extracted={len(local)}", flush=True)

    if not args.skip_public:
        for spec in PUBLIC_SOURCES:
            if spec.adapter == "commit":
                selected = build_commitpack_source(spec, args, stats)
            else:
                selected = build_public_source(spec, args, stats)
            records.extend(selected)
            print(f"{spec.name}: selected={len(selected)}", flush=True)

    deduplicated: dict[str, dict[str, Any]] = {}
    for item in records:
        prompt_key = hashlib.sha256(item["prompt"].encode("utf-8")).hexdigest()
        if prompt_key in deduplicated:
            stats["duplicate_prompts_removed"] += 1
            continue
        deduplicated[prompt_key] = item
    records = list(deduplicated.values())
    records.sort(key=lambda item: stable_digest(args.seed, "final", item["id"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output:
        for item in records:
            output.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "seed": args.seed,
        "total_prompts": len(records),
        "output": str(args.output),
        "max_prompt_chars": args.max_prompt_chars,
        "counts": {
            "source": count_values(records, "source"),
            "category": count_values(records, "category"),
            "license": count_values(records, "license"),
        },
        "processing_stats": dict(sorted(stats.items())),
        "public_source_targets": {
            spec.name: {"dataset": spec.dataset, "count": spec.count}
            for spec in PUBLIC_SOURCES
        },
        "privacy": {
            "local_system_and_developer_messages_excluded": True,
            "local_assistant_commentary_and_reasoning_excluded": True,
            "known_secret_patterns_redacted": True,
            "home_path_replaced_with": "$HOME",
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8") as output:
        json.dump(manifest, output, ensure_ascii=False, indent=2)
        output.write("\n")

    print(f"wrote {len(records)} prompts to {args.output}")
    print(f"manifest: {args.manifest}")


if __name__ == "__main__":
    main()
