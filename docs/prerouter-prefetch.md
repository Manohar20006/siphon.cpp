# Optional Edge0 prerouter prefetch

Siphon can use an Edge0 prerouter checkpoint to predict which Qwen3.6 MoE experts the next token may need. This mode **only warms the DirectStorage expert cache**. The model's own router still selects the experts that execute, so a wrong prediction cannot change the generated text.

The feature is off by default. It currently supports only Qwen3.6-35B-A3B with 40 layers, 2048 hidden channels, 256 experts and a runtime expert top-K of 4. A checkpoint must contain the 33 fp16 heads owned by layers 6–38. Incompatible checkpoints and model configurations are rejected during context creation.

Run it with the existing DirectStorage model setup and add:

```sh
--dstorage-prerouter /absolute/path/prerouter_edge0_35b.safetensors
```

The flag also enables DirectStorage MoE and its prefetch cache. For `scripts/run-qwen36-moe.sh`, export `LLAMA_DSTORAGE_PREROUTER` with the checkpoint path. Keep `LLAMA_MOE_EXPERT_USED_OVERRIDE=4`; other active-expert counts are rejected. Omit the flag or environment variable to use the existing behavior.

This is an experimental CPU implementation. The checkpoint is immutable and shared between compatible base contexts; MTP draft contexts do not load it. It still expands fp16 weights to approximately 264 MiB, but not once per draft context.

Learned weight staging is currently Linux-only. Windows can run shadow prediction, but learned admission is disabled until asynchronous-copy failure retirement is verified there. Native Windows prefetch is unchanged.

There are two modes, selected with `LLAMA_DSTORAGE_PREROUTER_MODE`:

- `shadow` (default): predictions are measured without staging weights. Small base-model verification batches of up to eight tokens are supported; large prompt batches and coupled sequence rows use native behavior. One rotating owner head is evaluated per batch, for each token row. Predictions are keyed by sequence, position and consumer layer, and compared to original exact expert IDs before cache-slot remapping. Recall counts **attempted verification**, including drafts that may later be rejected, not accepted-output-token accuracy.
- `prefetch`: evaluate only one row per graph. In MTP batches, the penultimate row predicts the final row's route at the next layer, so the hint can be consumed in the same verification graph rather than relying on a possibly rejected speculative successor. Single-token decode predicts the next position. After at least eight comparisons and measured expert recall of at least 35%, submit at most four predicted experts through a separate bounded admission path. Set an explicit positive `LLAMA_DSTORAGE_SPECULATIVE_SLOTS` reserve (for example `16`). One learned job may be in flight. Executing, transferring, quarantined and admission-pending slots are protected, and admission rejects work on contention or insufficient capacity. Learned predictions never pretend to have exact-route history. Shadow still evaluates every row; do not compare its prediction counts directly to staging mode's one-row subset.

Prediction and submission occur at the owner-layer router callback, after features are available, not after the whole graph. There is no immediate wait after submission. The existing demand/cache boundary and slot lock retire transfers before execution. Rollback, sequence switches, gaps and state restore invalidate pending learned work; already-launched transfers retire safely before slots are released. Completed cache hints may remain resident until ordinary eviction, but never choose executed experts. Native prefetch remains enabled in both modes. Low measured recall (below 25% after 64 comparisons) or excessive capture/prediction CPU cost (over 10% of measured small-graph time) disables learned work.

For MTP, set `LLAMA_DSTORAGE_SPECULATIVE_RESERVE_ALWAYS=1` to keep the explicit reserve out of demand allocation during prompt and verification batches too. Without this separate opt-in, demand batches can occupy the entire reserve before learned staging begins. This setting requires an explicit positive slot count; it does not create an automatic reserve. Apply it identically to baseline and prerouter comparison runs. Already-resident predictions are rejected without queueing a no-op job.

Multi-row MTP verification normally uses the prefill cache phase. Learned admission now honors the explicit persistent reserve in that phase instead of requiring the single-token phase switch. After calibration, a try-lock capacity check skips feature capture and CPU prediction when no eligible staging slot exists. Admission still revalidates capacity; the precheck reserves nothing.

Empty staging slots are preferred. When they are exhausted, only completed **learned-owned entries that were actually consumed**, with no execution, transfer, quarantine or admission protection, may be recycled in least-recently-used order. The current and preceding layer remain guarded; ordinary/native entries and unused learned entries are never reclaimed by this path. `recycled_slots` counts completed replacements. Unused learned entries or native entries can still saturate the shared reserve. The cache mutex remains held during I/O because the shared loader and host-buffer lifetimes are not independently concurrent; this limitation has not been solved by the recycling patch.

`loaded_experts` counts completed learned loads; `useful_hits` counts subsequent exact cache accesses, including repeated reuse of the same learned-loaded expert. It is not a count of distinct loaded experts or proof of time saved. `resident_unused` is a current-residency count, not a lifetime unused-load total.

Use `LLAMA_DSTORAGE_PREROUTER_PROFILE=1` to print separate base/draft graph wall time and time spent in exact expert-ensure callbacks, including a small-batch subset. `LLAMA_DSTORAGE_SUMMARY=1` adds logical file-read bytes, pinned transfers, copy/lock waits and learned submission/load/use counters. File bytes are not necessarily physical SSD traffic when OS caching or compatibility reads are involved. Shadow summaries report prediction/capture cost, expert recall, nonresident candidates and prediction-to-demand lead time. Nonresident candidates and submissions are **not useful hits or speedups**. Profiling synchronizes graph boundaries and can affect performance, so run the final throughput comparison without the profile switch. Without profiling, the CPU-cost guard uses graph-compute host duration (including existing exact-router waits), not GPU-completion timing, and no extra graph-end fence is inserted.

For a reproducible local A/B comparison:

```sh
python3 scripts/bench-prerouter.py \
  --server /absolute/path/to/llama-server \
  --model /absolute/path/to/qwen.gguf \
  --checkpoint /absolute/path/to/prerouter.safetensors \
  --output /absolute/path/to/new-results-directory \
  --reserve-always
```

The runner executes baseline/shadow/prefetch/baseline with identical MTP, cache reserve, context, prompts, output limit and greedy sampling settings. `--modes` changes the arm order; `--context`, `--batch`, `--ubatch`, `--threads`, `--speculative-slots` and the cache-budget options configure a matched test. It retains server logs, configuration, responses, token IDs, timings and metrics snapshots and checks output/token equivalence against every matching baseline. It fingerprints the executable and the resolved `libllama`; do not rebuild or replace either during a run. `summary.json` reports total tokens divided by total decode time, per-prompt results and nearest-before/after baseline comparisons. A bracketing average does not remove non-linear drift or establish significance.

Resources are sampled once per second: process RSS/anonymous/file memory and swap, available host RAM, host pressure, unified-cgroup memory usage/limit/events/statistics when available, and NVIDIA utilization/temperature/clocks/power/memory. Sampling failures are recorded rather than replaced with fabricated values. The default pinned-cache budget is 2 GiB; this is a bounded test profile, not a replacement for the user's Studio profile. Choose memory limits suitable for the host. A process memory cap can force file-cache reclaim and SSD reads even when global RAM is available; inspect cgroup `max` events and pressure before attributing a speed change to the prerouter. It does not flush OS caches; use repeated/reversed arm orders and distinguish first versus warmed requests. Mode-clustered ordering, memory pressure, temperature and OS caches can confound small speed differences.

**This is not a demonstrated speedup.** CPU prediction and extra SSD reads can outweigh prefetch savings; measure throughput and useful I/O overlap on the target hardware before enabling prefetch broadly. See [the implementation handoff](prerouter-implementation-handoff.md) for retained test evidence and remaining limitations.

The checkpoint is distributed by [Edge0](https://github.com/Edge0-AI/edge0). The head architecture and cross-token feature layout follow Edge0's [prerouter documentation](https://github.com/Edge0-AI/edge0/blob/main/docs/prerouter.md). Edge0 trains its prerouter together with its model adapter; predictions against a plain Qwen GGUF may be less accurate, and this integration does not load that adapter. This mode also does not reproduce Edge0's all-layer staging or replace exact routing. No Edge0 model or checkpoint is included in Siphon.
