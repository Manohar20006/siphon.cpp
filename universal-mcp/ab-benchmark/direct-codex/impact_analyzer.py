from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def _normalize(graph: Any, changed: Any) -> tuple[dict[str, set[str]], list[str]]:
    if not isinstance(graph, dict):
        raise ValueError("graph must be an object")
    normalized: dict[str, set[str]] = {}
    for node, dependencies in graph.items():
        if not isinstance(node, str) or not node:
            raise ValueError("component names must be non-empty strings")
        if not isinstance(dependencies, (list, tuple, set)):
            raise ValueError(f"dependencies for {node!r} must be an array")
        values: set[str] = set()
        for dependency in dependencies:
            if not isinstance(dependency, str) or not dependency:
                raise ValueError("dependency names must be non-empty strings")
            values.add(dependency)
        normalized[node] = values
    for dependency in {item for values in normalized.values() for item in values}:
        normalized.setdefault(dependency, set())
    if isinstance(changed, (str, bytes)) or not isinstance(changed, Iterable):
        raise ValueError("changed must be an iterable of component names")
    changed_names = []
    for node in changed:
        if not isinstance(node, str) or not node:
            raise ValueError("changed names must be non-empty strings")
        changed_names.append(node)
    return normalized, sorted(set(changed_names))


def _canonical_cycle(cycle: list[str]) -> tuple[str, ...]:
    rotations = [tuple(cycle[index:] + cycle[:index]) for index in range(len(cycle))]
    return min(rotations)


def _cycles(graph: dict[str, set[str]], nodes: set[str]) -> list[list[str]]:
    found: set[tuple[str, ...]] = set()
    for start in sorted(nodes):
        def visit(node: str, path: list[str], active: set[str]) -> None:
            for dependency in sorted(graph[node] & nodes):
                if dependency == start:
                    found.add(_canonical_cycle(path))
                elif dependency not in active and dependency >= start:
                    visit(dependency, path + [dependency], active | {dependency})
        visit(start, [start], {start})
    return [list(cycle) for cycle in sorted(found)]


def analyze(graph: Any, changed: Any) -> dict[str, Any]:
    normalized, changed_names = _normalize(graph, changed)
    unknown = sorted(set(changed_names) - normalized.keys())
    reverse: dict[str, set[str]] = defaultdict(set)
    for node, dependencies in normalized.items():
        for dependency in dependencies:
            reverse[dependency].add(node)
    impacted = set(changed_names) - set(unknown)
    queue = list(impacted)
    while queue:
        dependency = queue.pop()
        for dependent in reverse[dependency]:
            if dependent not in impacted:
                impacted.add(dependent)
                queue.append(dependent)
    cycles = _cycles(normalized, impacted)
    cyclic_nodes = {node for cycle in cycles for node in cycle}
    remaining = impacted - cyclic_nodes
    batches: list[list[str]] = []
    built: set[str] = set(cyclic_nodes)
    while remaining:
        batch = sorted(node for node in remaining if not ((normalized[node] & impacted) - built))
        if not batch:
            break
        batches.append(batch)
        built.update(batch)
        remaining.difference_update(batch)
    return {
        "changed": changed_names,
        "impacted": sorted(impacted),
        "unknown_changed": unknown,
        "cycles": cycles,
        "batches": batches,
    }

