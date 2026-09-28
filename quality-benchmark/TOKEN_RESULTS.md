# Measured MCP Token Results

The Antigravity adapter was changed to request JSON output and preserve the
provider's exact usage metadata. A fresh Gemini implementation of the same Task
Board specification produced these measurements:

| Metric | Measured value |
|---|---:|
| Worker input tokens | 22,482 |
| Worker output tokens | 22,181 |
| Worker thinking tokens | 17,182 |
| Worker cache-read tokens | 0 |
| Worker total tokens | 44,663 |
| Worker duration | 55.64 seconds |
| Generated patch size | 12,441 bytes |
| MCP response sent to controller | 4,017 bytes before response de-duplication |

`thinking_tokens` is provider-reported metadata and may be a subset of output
tokens; the authoritative provider total for this request is 44,663.

Codex does not expose per-turn token billing to this local MCP, so an exact
Codex-token saving cannot be calculated. The response size suggests roughly
1,000–1,300 controller-visible tokens before response de-duplication, compared
with several thousand tokens normally required to reason about and emit the
implementation directly.

## Interpretation

The MCP route clearly reduces how much implementation content Astra/Codex must
generate and inspect. It did not reduce total tokens across all models in this
run: Gemini's fixed context plus reasoning made the worker request expensive.
The architecture is therefore useful for conserving scarce Astra capacity, but
should not be described as a total-token optimization without provider-specific
cost and cache measurements.
