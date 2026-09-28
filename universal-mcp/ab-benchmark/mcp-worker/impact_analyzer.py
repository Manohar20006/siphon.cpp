"""Dependency Impact Analyzer."""

from typing import Any, Dict, List, Set


def _validate_and_normalize(graph: Any, changed: Any):
    if not isinstance(graph, dict):
        raise ValueError("Graph must be a dictionary")

    if isinstance(changed, (str, bytes)):
        raise ValueError("Changed must be a collection of strings, not a string")
    if not hasattr(changed, "__iter__"):
        raise ValueError("Changed must be an iterable")

    normalized_changed: List[str] = []
    for item in changed:
        if not isinstance(item, str) or not item:
            raise ValueError("Changed items must be non-empty strings")
        normalized_changed.append(item)

    normalized_graph: Dict[str, Set[str]] = {}
    for node, deps in graph.items():
        if not isinstance(node, str) or not node:
            raise ValueError("Graph keys must be non-empty strings")
        if not isinstance(deps, (list, tuple, set)) or isinstance(deps, (str, bytes)):
            raise ValueError(f"Dependencies for '{node}' must be a list, tuple, or set")

        if node not in normalized_graph:
            normalized_graph[node] = set()

        for dep in deps:
            if not isinstance(dep, str) or not dep:
                raise ValueError(f"Dependency names for '{node}' must be non-empty strings")
            normalized_graph[node].add(dep)
            if dep not in normalized_graph:
                normalized_graph[dep] = set()

    return normalized_graph, normalized_changed


def _find_cycles(nodes: List[str], adj: Dict[str, List[str]]) -> List[List[str]]:
    cycles = []

    for i, s in enumerate(nodes):
        sub_nodes = set(nodes[i:])
        sub_adj = {u: [v for v in adj.get(u, ()) if v in sub_nodes] for u in sub_nodes}

        blocked = set()
        B: Dict[str, Set[str]] = {u: set() for u in sub_nodes}
        stack: List[str] = []

        def unblock(u: str):
            blocked.discard(u)
            while B[u]:
                w = B[u].pop()
                if w in blocked:
                    unblock(w)

        def circuit(v: str) -> bool:
            f = False
            stack.append(v)
            blocked.add(v)

            for w in sub_adj.get(v, ()):
                if w == s:
                    cycles.append(list(stack))
                    f = True
                elif w not in blocked:
                    if circuit(w):
                        f = True

            if f:
                unblock(v)
            else:
                for w in sub_adj.get(v, ()):
                    B[w].add(v)

            stack.pop()
            return f

        circuit(s)

    unique_cycles = set()
    for c in cycles:
        min_idx = c.index(min(c))
        canonical = tuple(c[min_idx:] + c[:min_idx])
        unique_cycles.add(canonical)

    return [list(c) for c in sorted(unique_cycles)]


def analyze(graph: Any, changed: Any) -> Dict[str, Any]:
    normalized_graph, raw_changed = _validate_and_normalize(graph, changed)

    unique_changed = sorted(set(raw_changed))
    unknown_changed = sorted([c for c in unique_changed if c not in normalized_graph])
    known_changed = [c for c in unique_changed if c in normalized_graph]

    dependents: Dict[str, Set[str]] = {node: set() for node in normalized_graph}
    for node, deps in normalized_graph.items():
        for dep in deps:
            dependents[dep].add(node)

    impacted_set: Set[str] = set(known_changed)
    queue = list(known_changed)
    while queue:
        curr = queue.pop()
        for dep in dependents.get(curr, ()):
            if dep not in impacted_set:
                impacted_set.add(dep)
                queue.append(dep)

    impacted = sorted(impacted_set)

    sub_nodes = sorted(impacted_set)
    sub_adj = {u: sorted([v for v in normalized_graph[u] if v in impacted_set]) for u in sub_nodes}
    cycles = _find_cycles(sub_nodes, sub_adj)

    cycle_nodes = {node for cycle in cycles for node in cycle}
    acyclic_nodes = impacted_set - cycle_nodes

    in_deps: Dict[str, Set[str]] = {
        u: {v for v in normalized_graph[u] if v in acyclic_nodes}
        for u in acyclic_nodes
    }
    remaining = set(acyclic_nodes)
    batches: List[List[str]] = []

    while remaining:
        current_batch = [u for u in remaining if not in_deps[u]]
        current_batch.sort()
        batches.append(current_batch)
        for u in current_batch:
            remaining.remove(u)
            for w in remaining:
                in_deps[w].discard(u)

    return {
        "batches": batches,
        "changed": unique_changed,
        "cycles": cycles,
        "impacted": impacted,
        "unknown_changed": unknown_changed,
    }
