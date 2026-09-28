# Dependency Impact Analyzer Benchmark

Create `impact_analyzer.py` and `cli.py` using only the Python standard library.

Input graph format is a JSON object mapping a component to the components it
directly depends on. Every dependency-only component is also a graph node.

`analyze(graph, changed)` must return a JSON-serializable dictionary with:

- `changed`: sorted unique changed components.
- `impacted`: sorted components that are changed or transitively depend on a
  changed component.
- `unknown_changed`: sorted changed names absent from the normalized graph.
- `cycles`: deterministic simple directed cycles within the impacted subgraph.
  Each cycle is represented once, starts with its lexicographically smallest
  member, and does not repeat the first member at the end.
- `batches`: deterministic build batches for impacted acyclic components.
  Dependencies appear in earlier batches; members of each batch are sorted.
  Components belonging to cycles are excluded from batches.

Invalid graph keys, dependency names, or changed values that are not non-empty
strings must raise `ValueError`. A dependency list must be an array-like list,
tuple, or set—not a string. Duplicate dependencies are ignored.

`cli.py GRAPH.json CHANGED [CHANGED ...]` prints the analysis as sorted JSON.
Malformed input or validation errors print a concise message to stderr and exit
with code 2. Successful execution exits with code 0.

