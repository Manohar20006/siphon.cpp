# MCP Token-Savings Measurement

Measured on 2026-09-21 using the live authenticated MCP server.

## What can be measured exactly

- Gemini reports its own input, output, cache, and thinking usage through the
  Antigravity adapter.
- MCP HTTP response sizes are measured exactly in bytes.
- The six source files delegated during the worker-side architecture audit
  contain 141,302 characters in total.

Codex does not expose a per-tool-call tokenizer or exact account token counter
to this MCP. Codex token figures below therefore use the conventional
characters/4 approximation and are labeled estimates.

## Compact response measurement

For the live persisted task `persistence-live-proof`:

| Route | HTTP response | Estimated Codex tokens |
|---|---:|---:|
| Compact MCP status (default) | 953 bytes | ~238 |
| Full MCP status with all details | 4,603 bytes | ~1,151 |

Compact status reduced the Codex-visible response by 79.3% for this small task.
The saving grows for tasks containing patches, raw output, context files, long
verification logs, and message history because those fields are opt-in.

## Architecture-audit route

The six audited files total 141,302 characters, approximately 35,326 tokens if
their complete text were injected into Codex. Gemini processed that context and
reported exact worker usage of 56,484 total tokens, including 51,010 input and
5,474 output tokens. Codex needed only the worker report rather than all source
text.

Using the full 5,474-token worker report as the conservative Codex-visible
payload, the estimated Codex-context saving is:

- Direct source context: ~35,326 tokens
- MCP report: 5,474 tokens
- Estimated Codex tokens avoided: ~29,852
- Estimated reduction: 84.5%

With the new compact status/event route, routine polling is only a few hundred
tokens. Codex requests full evidence only for risks, disagreements, final diffs,
or verification failures.

These numbers measure context transferred to Codex, not total system tokens.
Total system usage is higher because Gemini performs the delegated work. The
design intentionally exchanges inexpensive worker tokens for scarce Codex
planning and review tokens.
