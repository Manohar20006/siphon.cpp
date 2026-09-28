# Universal MCP A/B Benchmark Results

Date: 2026-09-21

## Setup

Both candidates received the same dependency-impact-analyzer specification.
Gemini received only `SPEC.md`; it was explicitly prohibited from inspecting
the evaluator or direct Codex candidate. A separate hidden evaluator scored 15
functional, validation, cycle, ordering, and CLI checks.

## Quality

| Candidate | First attempt | After corrections | Corrections |
|---|---:|---:|---:|
| Direct Codex | 15/15 | 15/15 | 0 |
| Gemini through Universal MCP | 0/15 | 15/15 | 2 |

Gemini's first patch contained a truncated loop body and could not import. The
first correction reached 13/15 but left the CLI main guard truncated. The
second correction reached 15/15. The MCP safely applied each patch and the
final formal verification passed with exit code 0.

Final functional quality is equal on this evaluator, but first-pass quality and
repair efficiency favored direct Codex.

## Token and payload measurements

- Final cumulative Gemini usage reported by Antigravity: 294,547 tokens
  - Input: 242,585
  - Output: 51,962
  - Thinking: 48,627 (reported as a component of output telemetry)
  - Cache reads: 22,831
- Compact final MCP status: 1,225 bytes (~306 tokens at chars/4)
- Full final audit status: 14,889 bytes (~3,722 tokens at chars/4)
- Specification: 1,435 characters
- Direct Codex implementation: 4,312 characters
- Final MCP-worker implementation: 5,521 characters

Codex does not expose exact per-turn token telemetry here, so a precise direct
versus MCP Codex-token percentage cannot be claimed. The lower-bound visible
content for direct Codex (specification plus generated source) is about 1,437
tokens at chars/4, excluding reasoning and test interaction. The MCP route
required multiple polls, two evaluator failure reports, two correction prompts,
and final verification. Even with compact status, this small task does not show
a defensible Codex-token saving and probably used more Codex coordination
tokens than the direct route.

## Conclusion

Universal MCP preserved final quality after evidence-driven correction, but it
was inefficient for this small task. The earlier 79.3% measurement describes
compact status versus full status only; it is not an end-to-end project saving.

MCP delegation is most likely to save Codex tokens on larger projects where:

- repository exploration and implementation context are large;
- worker phases are bounded enough to avoid truncation;
- deterministic syntax/tests run before a patch reaches Codex;
- compact checkpoints replace repeated full-source inspection; and
- worker retries are uncommon.

The next benchmark should use a medium multi-file project and add an automatic
`py_compile`/syntax gate before patch application. The two failures in this run
would then have been caught without spending Codex review tokens.
