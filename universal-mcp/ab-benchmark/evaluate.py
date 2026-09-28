#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"candidate_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def evaluate(candidate: Path) -> dict:
    checks = []
    def check(name, condition):
        checks.append((name, bool(condition)))

    try:
        module = load_module(candidate / "impact_analyzer.py")
        graph = {
            "api": ["core"], "web": ["api", "ui"], "worker": ["core"],
            "core": [], "ui": [], "docs": ["web"],
        }
        result = module.analyze(graph, ["core"])
        check("transitive impact", result["impacted"] == ["api", "core", "docs", "web", "worker"])
        check("deterministic batches", result["batches"] == [["core"], ["api", "worker"], ["web"], ["docs"]])
        check("unknown changed", module.analyze(graph, ["missing"])["unknown_changed"] == ["missing"])
        cyclic = {"a": ["b"], "b": ["c"], "c": ["a"], "d": ["a"], "e": []}
        cycle_result = module.analyze(cyclic, ["a"])
        check("cycle canonicalization", cycle_result["cycles"] == [["a", "b", "c"]])
        check("cycle exclusion", cycle_result["batches"] == [["d"]])
        self_cycle = module.analyze({"a": ["a"], "b": ["a"]}, ["a"])
        check("self cycle", self_cycle["cycles"] == [["a"]])
        check("dependent outside cycle", self_cycle["batches"] == [["b"]])
        dependency_only = module.analyze({"app": ["lib"]}, ["lib"])
        check("dependency-only node", dependency_only["impacted"] == ["app", "lib"])
        duplicate = module.analyze({"app": ["lib", "lib"], "lib": []}, ["lib", "lib"])
        check("deduplication", duplicate["changed"] == ["lib"] and duplicate["impacted"] == ["app", "lib"])
        for bad_graph, bad_changed in [({1: []}, []), ({"a": "b"}, []), ({"a": [""]}, []), ({}, [1])]:
            try:
                module.analyze(bad_graph, bad_changed)
                valid = False
            except ValueError:
                valid = True
            check(f"validation {bad_graph!r} {bad_changed!r}", valid)

        with tempfile.TemporaryDirectory() as td:
            graph_path = Path(td) / "graph.json"
            graph_path.write_text(json.dumps(graph), encoding="utf-8")
            proc = subprocess.run(
                [sys.executable, str(candidate / "cli.py"), str(graph_path), "core"],
                text=True, capture_output=True, check=False,
            )
            parsed = json.loads(proc.stdout) if proc.returncode == 0 else {}
            check("CLI success", proc.returncode == 0 and parsed.get("impacted") == result["impacted"])
            bad = subprocess.run(
                [sys.executable, str(candidate / "cli.py"), str(Path(td) / "absent.json"), "x"],
                text=True, capture_output=True, check=False,
            )
            check("CLI error contract", bad.returncode == 2 and bool(bad.stderr.strip()))
    except Exception as error:
        checks.append((f"uncaught evaluator error: {error}", False))
    passed = sum(value for _, value in checks)
    return {"candidate": candidate.name, "passed": passed, "total": len(checks), "checks": checks}


if __name__ == "__main__":
    roots = [Path(arg).resolve() for arg in sys.argv[1:]]
    results = [evaluate(root) for root in roots]
    print(json.dumps(results, indent=2))
    raise SystemExit(0 if all(item["passed"] == item["total"] for item in results) else 1)

