#!/usr/bin/env python3

import argparse
import bisect
import json
import statistics
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Event:
    sequence: int
    request: int
    phase: str
    layer: int
    slots: int
    expert_bytes: int
    stream_file_bytes: int
    transfer_us: int
    experts: tuple[int, ...]

    @property
    def keys(self) -> tuple[tuple[int, int], ...]:
        return tuple((self.layer, expert) for expert in self.experts)


@dataclass
class Entry:
    last_used: int
    decode_hits: int = 0
    prefill_hits: int = 0
    segment: str = "recent"


class LatencyModel:
    def __init__(self, events: list[Event], actual_misses: list[int]):
        samples = defaultdict(list)
        for event, misses in zip(events, actual_misses):
            if misses > 0 and event.transfer_us > 0:
                samples[self.bucket(misses)].append(event.transfer_us)
        all_values = [value for values in samples.values() for value in values]
        fallback = statistics.median(all_values) if all_values else 0.0
        self.medians = {
            bucket: statistics.median(samples[bucket]) if samples[bucket] else fallback
            for bucket in range(4)
        }

    @staticmethod
    def bucket(misses: int) -> int:
        if misses <= 4:
            return 0
        if misses <= 8:
            return 1
        if misses <= 16:
            return 2
        return 3

    def estimate(self, misses: int) -> float:
        return 0.0 if misses == 0 else self.medians[self.bucket(misses)]


class CachePolicy:
    name = "base"

    def __init__(self, capacity: int, n_layers: int, events: list[Event]):
        self.capacity = capacity
        self.n_layers = n_layers
        self.events = events
        self.cache: dict[tuple[int, int], Entry] = {}
        self.tick = 0

    def begin_event(self, index: int, event: Event) -> None:
        del index, event

    def victim_score(self, key: tuple[int, int], index: int, event: Event) -> tuple:
        del index, event
        return (self.cache[key].last_used,)

    def admit(self, key: tuple[int, int], event: Event) -> None:
        self.cache[key] = Entry(
            last_used=self.tick,
            decode_hits=1000 if event.phase == "decode" else 0,
            prefill_hits=1000 if event.phase == "prefill" else 0,
        )

    def hit(self, key: tuple[int, int], event: Event) -> None:
        entry = self.cache[key]
        entry.last_used = self.tick
        if event.phase == "decode":
            entry.decode_hits += 1000
        elif event.phase == "prefill":
            entry.prefill_hits += 1000

    def access(self, index: int, event: Event) -> int:
        self.tick += 1
        self.begin_event(index, event)
        keys = event.keys
        protected = set(keys)
        misses = [key for key in keys if key not in self.cache]
        for key in keys:
            if key in self.cache:
                self.hit(key, event)
        overflow = max(0, len(self.cache) + len(misses) - self.capacity)
        if overflow:
            candidates = [key for key in self.cache if key not in protected]
            victims = sorted(
                candidates,
                key=lambda key: self.victim_score(key, index, event),
            )[:overflow]
            for victim in victims:
                del self.cache[victim]
        for key in misses:
            self.admit(key, event)
        return len(misses)


class LRUPolicy(CachePolicy):
    name = "lru"


class LegacyPolicy(CachePolicy):
    name = "legacy-score"

    def __init__(self, capacity: int, n_layers: int, events: list[Event]):
        super().__init__(capacity, n_layers, events)
        self.request_counts = Counter()
        self.current_request = None

    def begin_event(self, index: int, event: Event) -> None:
        del index
        if event.request != self.current_request:
            self.current_request = event.request
            self.request_counts.clear()
        for key in event.keys:
            self.request_counts[(event.phase, key)] += 1

    def retention(self, key: tuple[int, int], event: Event) -> int:
        entry = self.cache[key]
        layer = key[0]
        decode_count = self.request_counts[("decode", key)]
        prefill_count = self.request_counts[("prefill", key)]
        distance = (layer - event.layer) % self.n_layers
        urgency = self.n_layers - distance
        return (
            decode_count * 1_000_000
            + prefill_count * 10_000
            + entry.decode_hits * 10
            + entry.prefill_hits
            + urgency * 1000
            + (self.n_layers - layer) * 100
        )

    def victim_score(self, key: tuple[int, int], index: int, event: Event) -> tuple:
        del index
        entry = self.cache[key]
        return (self.retention(key, event), entry.last_used)


class HybridARCPolicy(LegacyPolicy):
    name = "hybrid-arc"

    def hit(self, key: tuple[int, int], event: Event) -> None:
        super().hit(key, event)
        if event.phase == "decode":
            self.cache[key].segment = "frequent"

    def admit(self, key: tuple[int, int], event: Event) -> None:
        super().admit(key, event)
        self.cache[key].segment = "frequent" if event.phase == "decode" else "recent"

    def victim_score(self, key: tuple[int, int], index: int, event: Event) -> tuple:
        del index
        entry = self.cache[key]
        arc_bonus = 1_000_000 if entry.segment == "frequent" else 10_000
        return (self.retention(key, event) + arc_bonus, entry.last_used)


class CompleteSetPolicy(LegacyPolicy):
    name = "complete-set"

    def __init__(self, capacity: int, n_layers: int, events: list[Event], window: int = 256):
        super().__init__(capacity, n_layers, events)
        self.window = window
        self.layer_history = defaultdict(deque)
        self.layer_signatures = defaultdict(Counter)
        self.current_completion_bonus = Counter()

    def begin_event(self, index: int, event: Event) -> None:
        super().begin_event(index, event)
        signature = event.keys
        history = self.layer_history[event.layer]
        counts = self.layer_signatures[event.layer]
        history.append(signature)
        counts[signature] += 1
        if len(history) > self.window:
            old = history.popleft()
            counts[old] -= 1
            if counts[old] == 0:
                del counts[old]
        self.current_completion_bonus.clear()
        for candidate, frequency in counts.most_common(4):
            if all(member in self.cache for member in candidate):
                for member in candidate:
                    self.current_completion_bonus[member] += frequency * 250_000

    def completion_bonus(self, key: tuple[int, int]) -> int:
        return self.current_completion_bonus[key]

    def victim_score(self, key: tuple[int, int], index: int, event: Event) -> tuple:
        del index
        entry = self.cache[key]
        return (self.retention(key, event) + self.completion_bonus(key), entry.last_used)


class BeladyPolicy(CachePolicy):
    name = "belady-oracle"

    def __init__(self, capacity: int, n_layers: int, events: list[Event]):
        super().__init__(capacity, n_layers, events)
        self.positions = defaultdict(list)
        for index, event in enumerate(events):
            for key in event.keys:
                self.positions[key].append(index)

    def next_use(self, key: tuple[int, int], index: int) -> int:
        positions = self.positions[key]
        offset = bisect.bisect_right(positions, index)
        return positions[offset] if offset < len(positions) else len(self.events) + 1

    def victim_score(self, key: tuple[int, int], index: int, event: Event) -> tuple:
        del event
        return (-self.next_use(key, index), self.cache[key].last_used)


class CompleteCallOraclePolicy(BeladyPolicy):
    name = "complete-call-oracle"

    def __init__(self, capacity: int, n_layers: int, events: list[Event], horizon: int = 128):
        super().__init__(capacity, n_layers, events)
        self.horizon = horizon
        self.completion_bonus = Counter()

    def begin_event(self, index: int, event: Event) -> None:
        projected = set(self.cache)
        projected.update(event.keys)
        self.completion_bonus.clear()
        end = min(len(self.events), index + 1 + self.horizon)
        for future_index in range(index + 1, end):
            future_keys = self.events[future_index].keys
            if all(key in projected for key in future_keys):
                distance = future_index - index
                bonus = (self.horizon - distance + 1) * 1_000_000
                for key in future_keys:
                    self.completion_bonus[key] += bonus

    def victim_score(self, key: tuple[int, int], index: int, event: Event) -> tuple:
        del event
        return (
            self.completion_bonus[key],
            -self.next_use(key, index),
            self.cache[key].last_used,
        )


def read_trace(path: Path) -> tuple[list[Event], list[int]]:
    events = []
    actual_misses = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            experts = tuple(dict.fromkeys(int(value) for value in row["experts"]))
            event = Event(
                sequence=int(row["sequence"]),
                request=int(row["request"]),
                phase=str(row["phase"]),
                layer=int(row["layer"]),
                slots=int(row["slots"]),
                expert_bytes=int(row["expert_bytes"]),
                stream_file_bytes=int(row["stream_file_bytes"]),
                transfer_us=int(row["transfer_us"]),
                experts=experts,
            )
            if event.sequence != len(events):
                raise ValueError(f"{path}:{line_number}: non-contiguous sequence")
            events.append(event)
            actual_misses.append(len(set(int(value) for value in row["misses"])))
    if not events:
        raise ValueError(f"{path}: no routing events")
    return events, actual_misses


def simulate(policy: CachePolicy, events: list[Event], latency: LatencyModel) -> dict:
    misses = 0
    miss_calls = 0
    bytes_read = 0
    estimated_stall_us = 0.0
    phase_hits = Counter()
    phase_accesses = Counter()
    for index, event in enumerate(events):
        event_misses = policy.access(index, event)
        accesses = len(event.experts)
        misses += event_misses
        miss_calls += event_misses > 0
        bytes_read += event_misses * event.expert_bytes
        estimated_stall_us += latency.estimate(event_misses)
        phase_hits[event.phase] += accesses - event_misses
        phase_accesses[event.phase] += accesses
    return {
        "policy": policy.name,
        "misses": misses,
        "miss_calls": miss_calls,
        "all_hit_calls": len(events) - miss_calls,
        "bytes_read": bytes_read,
        "estimated_stall_us": estimated_stall_us,
        "decode_hit_rate": (
            phase_hits["decode"] / phase_accesses["decode"]
            if phase_accesses["decode"] else 0.0
        ),
        "prefill_hit_rate": (
            phase_hits["prefill"] / phase_accesses["prefill"]
            if phase_accesses["prefill"] else 0.0
        ),
    }


def print_results(results: list[dict]) -> None:
    header = (
        f"{'policy':<16} {'misses':>10} {'miss calls':>11} {'all-hit':>9} "
        f"{'read GiB':>10} {'stall s':>9} {'decode hit':>11} {'prefill hit':>12}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result['policy']:<16} {result['misses']:>10} "
            f"{result['miss_calls']:>11} {result['all_hit_calls']:>9} "
            f"{result['bytes_read'] / (1024 ** 3):>10.2f} "
            f"{result['estimated_stall_us'] / 1e6:>9.2f} "
            f"{result['decode_hit_rate'] * 100:>10.2f}% "
            f"{result['prefill_hit_rate'] * 100:>11.2f}%"
        )


def actual_result(events: list[Event], actual_misses: list[int]) -> dict:
    phase_hits = Counter()
    phase_accesses = Counter()
    for event, misses in zip(events, actual_misses):
        phase_hits[event.phase] += len(event.experts) - misses
        phase_accesses[event.phase] += len(event.experts)
    return {
        "policy": "actual-runtime",
        "misses": sum(actual_misses),
        "miss_calls": sum(value > 0 for value in actual_misses),
        "all_hit_calls": sum(value == 0 for value in actual_misses),
        "bytes_read": sum(event.stream_file_bytes for event in events),
        "estimated_stall_us": sum(event.transfer_us for event in events),
        "decode_hit_rate": phase_hits["decode"] / phase_accesses["decode"],
        "prefill_hit_rate": phase_hits["prefill"] / phase_accesses["prefill"],
    }


def self_test() -> None:
    rows = [
        Event(i, 1, "decode", i % 2, 3, 100, 200, 1000, experts)
        for i, experts in enumerate([
            (0, 1), (0, 1), (2, 3), (0, 1), (2, 3), (0, 1),
        ])
    ]
    misses = [2, 0, 2, 2, 2, 2]
    latency = LatencyModel(rows, misses)
    lru = simulate(LRUPolicy(3, 2, rows), rows, latency)
    belady = simulate(BeladyPolicy(3, 2, rows), rows, latency)
    assert belady["misses"] <= lru["misses"]
    assert belady["miss_calls"] <= lru["miss_calls"]
    assert latency.estimate(0) == 0
    print("moe-cache-sim self-test: ok")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a llama.cpp GDS MoE routing JSONL trace against cache policies.")
    parser.add_argument("trace", nargs="?", type=Path, help="routing JSONL trace")
    parser.add_argument("--slots", type=int, help="override cache slot capacity")
    parser.add_argument(
        "--policies",
        default="all",
        help="comma-separated policies or 'all': lru,legacy-score,hybrid-arc,"
             "complete-set,belady-oracle,complete-call-oracle",
    )
    parser.add_argument("--self-test", action="store_true", help="run deterministic internal tests")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.trace is None:
        parser.error("trace is required unless --self-test is used")

    events, actual_misses = read_trace(args.trace)
    capacity = args.slots or events[0].slots
    n_layers = max(event.layer for event in events) + 1
    latency = LatencyModel(events, actual_misses)
    policy_types = {
        "lru": LRUPolicy,
        "legacy-score": LegacyPolicy,
        "hybrid-arc": HybridARCPolicy,
        "complete-set": CompleteSetPolicy,
        "belady-oracle": BeladyPolicy,
        "complete-call-oracle": CompleteCallOraclePolicy,
    }
    selected = list(policy_types) if args.policies == "all" else args.policies.split(",")
    unknown = [name for name in selected if name not in policy_types]
    if unknown:
        parser.error(f"unknown policies: {', '.join(unknown)}")
    policies = [policy_types[name](capacity, n_layers, events) for name in selected]
    results = [actual_result(events, actual_misses)]
    results.extend(simulate(policy, events, latency) for policy in policies)

    actual_miss_calls = sum(value > 0 for value in actual_misses)
    actual_miss_count = sum(actual_misses)
    actual_stall_us = sum(event.transfer_us for event in events)
    print(
        f"events={len(events)} slots={capacity} layers={n_layers} "
        f"actual_misses={actual_miss_count} actual_miss_calls={actual_miss_calls} "
        f"actual_transfer_s={actual_stall_us / 1e6:.2f}"
    )
    print_results(results)


if __name__ == "__main__":
    main()
