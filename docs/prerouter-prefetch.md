# Optional Edge0 prerouter prefetch

Siphon can use an Edge0 prerouter checkpoint to predict which Qwen3.6 MoE experts the next token may need. This mode **only warms the DirectStorage expert cache**. The model's own router still selects the experts that execute, so a wrong prediction cannot change the generated text.

The feature is off by default. It currently supports only Qwen3.6-35B-A3B with 40 layers, 2048 hidden channels, 256 experts and a runtime expert top-K of 4. A checkpoint must contain the 33 fp16 heads owned by layers 6–38. Incompatible checkpoints and model configurations are rejected during context creation.

Run it with the existing DirectStorage model setup and add:

```sh
--dstorage-prerouter /absolute/path/prerouter_edge0_35b.safetensors
```

The flag also enables DirectStorage MoE and its prefetch cache. For `scripts/run-qwen36-moe.sh`, export `LLAMA_DSTORAGE_PREROUTER` with the checkpoint path. Keep `LLAMA_MOE_EXPERT_USED_OVERRIDE=4`; other active-expert counts are rejected. Omit the flag or environment variable to use the existing behavior.

This is an experimental CPU implementation. It copies the normalized MoE input for a single-token decode, runs a bounded rotating subset of trained heads after that token's graph, then offers predicted experts to Siphon's existing cache-admission policy for the next token. The policy may reject predictions when they lack enough observations or cache headroom. It waits for each admitted staged load before the next decode step to avoid contending with exact demand loads, so this initial integration does not overlap SSD I/O with decoding as Edge0 does. Loading the checkpoint expands its fp16 weights to approximately 264 MiB of optional host memory. Batched or interleaved requests do not share prediction state. Measure throughput and cache-hit improvement on the target machine before enabling it broadly: CPU prediction and extra SSD reads can outweigh prefetch savings.

If the expert cache has fewer slots than the 40-layer/4-expert working set plus 16 speculative slots, Siphon logs a warning and falls back to native prefetch. It also falls back after eight consecutive decode steps without an admitted prediction. Exact DirectStorage routing is never replaced. These fallbacks prevent prolonged overhead or cache pressure when the checkpoint is ineffective.

In a Qwen3.6-35B-A3B smoke test on this machine, native prefetch reached about 11.4 tokens/s, while the experimental prerouter path reached about 9.2 tokens/s before the fallback was added. No learned prediction was admitted by the safe cache policy in that run. **This is not a demonstrated speedup**; validating prediction quality with the matching Edge0 model adapter and a representative workload is still required.

The checkpoint is distributed by [Edge0](https://github.com/Edge0-AI/edge0). The head architecture and cross-token feature layout follow Edge0's [prerouter documentation](https://github.com/Edge0-AI/edge0/blob/main/docs/prerouter.md). Edge0 trains its prerouter together with its model adapter; predictions against a plain Qwen GGUF may be less accurate, and this integration does not load that adapter. This mode also does not reproduce Edge0's all-layer staging or replace exact routing. No Edge0 model or checkpoint is included in Siphon.
