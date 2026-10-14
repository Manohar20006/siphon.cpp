# Siphon.cpp Settings Map

This is the source-level map for the future Siphon Studio Model Catalogue. It
separates settings inherited from llama.cpp, Siphon-specific controls, and
environment-only tuning switches. No dashboard or model-catalogue UI should
expose every flag as an equal-level control.

## 1. Model facts: read-only in the catalogue

These values come from the GGUF model metadata and should be displayed as model
information, not treated as arbitrary user settings:

- model path, filename, quantization, architecture, and model alias;
- layer count and context length trained by the model;
- total experts per layer;
- native routed experts per token (the model's original top-K);
- expert groups and shared experts, when present;
- whether the GGUF contains an MTP/NextN head;
- estimated dense-weight footprint, expert footprint, and KV-cache pressure.

The loader reads `n_expert` and `n_expert_used` from model metadata. The
experimental `LLAMA_MOE_EXPERT_USED_OVERRIDE` can only reduce the native
`n_expert_used`; it cannot increase it. See `src/llama-model.cpp`.

## 2. The controls that belong in the first-class catalogue UI

### Runtime and memory

| UI control | Actual setting | Type | Meaning |
| --- | --- | --- | --- |
| Context window | `-c`, `--ctx-size`, wrapper `CTX` | llama.cpp | Maximum prompt/KV context. `0` means use the model value. Larger contexts consume more KV memory. |
| L1 / VRAM expert cache | `--moe-gpu-cache-mib`, `MOE_GPU_CACHE_MIB` | Siphon | Persistent VRAM budget for streamed MoE expert slots. Parser default is 1024 MiB; bundled profiles use values such as 2500–2800 MiB. |
| L2 / pinned RAM expert cache | `--moe-pinned-cache-mib`, `MOE_PINNED_CACHE_MIB` | Siphon | Host-pinned RAM budget for expert entries. `0` disables the pinned expert cache. |
| General prompt-cache RAM | `--cache-ram`, `CACHE_RAM_MIB` | llama.cpp server | RAM budget for prompt/KV checkpoint or radix-cache state. This is **not** the Siphon expert L2 cache. |
| GDS read workers | `LLAMA_GDS_READ_THREADS`, wrapper `GDS_READ_THREADS` | Siphon/GDS | Parallel GDS/cuFile expert-read workers. Source default is 20, clamped to 1–32. |
| Routed experts per token | `LLAMA_MOE_EXPERT_USED_OVERRIDE` | Siphon experimental | Effective MoE top-K. Only values lower than the model’s native K are accepted. This changes quality/performance and expert traffic. |

The catalogue must label the two RAM budgets separately:

- **Pinned expert L2** = `MOE_PINNED_CACHE_MIB`.
- **Prompt/KV cache RAM** = `CACHE_RAM_MIB` / `--cache-ram`.

They compete for system memory but serve different data.

### Prompt reuse

| UI control | Actual setting | Notes |
| --- | --- | --- |
| Prompt cache | `--cache-prompt`, `--no-cache-prompt`, wrapper `CACHE_PROMPT` | Reuses the common prefix/KV state between requests. Upstream server default is enabled, but the Coder-Next wrapper defaults its profile variable to disabled unless set. |
| Checkpoint count | `LLAMA_ARG_CTX_CHECKPOINTS`, `--ctx-checkpoints` | Maximum prompt checkpoints per slot. Common default is 32; Siphon profiles export 12 to limit RAM growth at large contexts. |
| Checkpoint spacing | `CHECKPOINT_MIN_STEP`, `--checkpoint-min-step` | Minimum token spacing between checkpoints. Siphon profiles use 0 so short chat/agent turns can remain cache-friendly. |
| KV-shift reuse | `CACHE_REUSE`, `--cache-reuse` | Optional cache reuse through KV shifting. It requires prompt caching and is currently kept at 0 because the active MTP/hybrid contexts report it unsupported. |

### Speculative decoding

The UI should expose a compact mode-aware section rather than one large flat
form:

| UI control | Actual setting | Applies to |
| --- | --- | --- |
| Speculation mode | `SPEC_TYPE`, `--spec-type` | `none`, `draft-mtp`, `draft-simple`, Eagle/DFlash, or n-gram modes supported by this checkout. |
| Draft model | `MODEL_DRAFT`, `--spec-draft-model` | Required for standalone `draft-simple`; not used by native MTP. |
| Draft length | `SPEC_DRAFT_N_MAX`, `--spec-draft-n-max` | Maximum tokens proposed per speculative step. Profiles commonly use 3. |
| Minimum draft length | `--spec-draft-n-min` | Minimum proposal length. Usually left at 0. |
| Minimum acceptance probability | `SPEC_DRAFT_P_MIN`, `--spec-draft-p-min` | Greedy/speculation gate. Profiles commonly use 0. |
| Split probability | `--spec-draft-p-split` | Advanced speculative scheduling control; upstream default is 0.10. |
| Draft GPU layers | `SPEC_DRAFT_GPU_LAYERS`, `--spec-draft-ngl` | Standalone draft model placement. The launcher forwards it when a draft model is present. |
| Draft KV types | `--spec-draft-type-k`, `--spec-draft-type-v` | Advanced draft-model KV memory/precision controls. |

There are also inherited draft CPU controls—threads, batch threads, CPU
affinity, strict placement, priority, polling, CPU MoE, and draft device. They
belong under an advanced drawer, not in the first screen.

Native MTP should be enabled only for a model whose GGUF actually contains the
MTP/NextN tensors. The repository’s Qwen3.6 and KAT profiles use
`draft-mtp`; Qwen3-Coder-Next has no MTP head and keeps speculation off unless a
separate draft model is intentionally selected.

### Sampling defaults

These are inherited llama.cpp generation defaults, not cache controls:

- temperature: `--temp` / `--temperature` (`0.80` common default);
- sampling top-k: `--top-k` (`40` common default);
- top-p, min-p, typical-p, repetition/frequency/presence penalties, DRY,
  dynamic temperature, Mirostat, seed, and sampler order.

The catalogue can include temperature as a user-requested model/session
setting, but it must not call sampling top-k “top-k experts.” Temperature is a
generation control; it can only affect expert routing indirectly through the
tokens selected.

## 3. Siphon launch flags

These are the actual command-line switches implemented in `common/arg.cpp`:

- `-dsm`, `--dstorage-moe`: enable Siphon’s direct SSD/GDS MoE streaming path.
- `--dstorage-moe-prefetch`: enable predictive/prefetch behavior and imply
  `--dstorage-moe`.
- `--moe-gpu-cache-mib N`: allocate the VRAM expert-slot budget.
- `--moe-pinned-cache-mib N`: allocate the pinned host expert-cache budget.
- `--moe-full-layer-ubatch` (`LLAMA_ARG_MOE_FULL_LAYER_UBATCH`, wrapper
  `MOE_FULL_LAYER_UBATCH=1`): let prefill ubatches cover whole layers instead
  of clamping them to `top-K x tokens <= pool slots`. Every expert of a layer
  is then loaded at most once per ubatch, so a larger `-ub` (512) needs far
  fewer expert reads. It requires every pool group to hold all experts of a
  layer (at least 256 slots per group; `LLAMA_DSTORAGE_LARGE_DOWN_SLOTS=264`)
  and is incompatible with `--dstorage-moe-prefetch` (and therefore with the
  prerouter). Measured on the RTX 4050 profile with `-ub 512` and a 2560 MiB
  VRAM cache: prompt processing 72 -> 153 tok/s at unchanged decode speed;
  the larger compute buffer is why the cache drops from 2816 to 2560 MiB.
- `-ngl`, `--gpu-layers`: layer offload placement inherited from llama.cpp;
  Siphon’s launchers use `-ngl 999` so normal dense/non-expert weights are
  offloaded as far as VRAM allows.
- `-cmoe`, `--cpu-moe` and `-ncmoe`, `--n-cpu-moe`: inherited CPU-MoE paths;
  they are mutually exclusive with `--dstorage-moe`.

The future catalogue should make the DStorage backend a model/runtime mode,
not a free-form toggle that users can combine with CPU-MoE accidentally.

## 4. `evict to RAM`: what it actually does

The relevant implementation is
`DStorageSlotManager::cache_evicted_expert_to_ram()` in
`src/llama-dstorage-slots.cpp`.

When a VRAM slot is evicted:

1. `LLAMA_DSTORAGE_EVICT_TO_RAM=1` and a non-zero pinned-cache budget are
   required.
2. Only a warm expert is promoted; the default threshold is
   `LLAMA_DSTORAGE_EVICT_TO_RAM_MIN_DECODE_HITS=2`.
3. The expert’s VRAM tensor slices are copied device-to-host into the pinned
   RAM cache.
4. If that expert is already resident or pending in pinned RAM, the copy is
   skipped.
5. If the pinned budget is full, dynamic entries are evicted by the pinned LRU
   index to make room.
6. A later VRAM miss can use host-to-device RAM transfer instead of another SSD
   read.

When evict-to-RAM is enabled, the normal SSD re-read pinned-admission path is
skipped to avoid double-caching and stealing SSD bandwidth. This means it is a
dynamic warm-cache policy, not just a boolean cosmetic feature.

The source default for `LLAMA_DSTORAGE_EVICT_TO_RAM` is off, while the current
bundled Siphon profiles enable it. Benchmark notes found a useful tuned
threshold of 1 decode hit in some workloads, but the source default remains 2;
that threshold should therefore be shown as an advanced policy parameter with
a model/profile-specific recommendation.

Do not combine this casually with `LLAMA_MOE_PINNED_PREFILL_ONLY=1`: that mode
releases the pinned cache after prefill and can severely reduce steady-state
decode speed because warm experts must return to NVMe.

## 5. Inherited llama.cpp controls worth supporting later

These are real parser options, but most belong in an Advanced section:

- execution: `--threads`, `--threads-batch`, CPU masks/ranges, strict CPU
  placement, priority, polling, NUMA;
- batching: `--batch-size`, `--ubatch-size`, parallel slots, continuous
  batching;
- attention/KV: Flash Attention, KV offload, KV K/V types, SWA/full cache;
- placement: devices, GPU layers, split mode, tensor split, main GPU, fit,
  fit target, minimum fit context;
- model loading: mmap, mlock, DirectIO, no-host, tensor-buffer overrides;
- chat behavior: chat template, template kwargs, reasoning mode/format/budget;
- serving: host, port, alias, HTTP threads, metrics endpoint, slot endpoint.

## 6. Environment-only tuning surface

These switches exist in the source but are not appropriate first-screen
controls. They are useful for diagnostics, experiments, or a future expert
mode:

### GDS backend internals

`LLAMA_GDS_REGISTER_PER_REQUEST`, `LLAMA_GDS_USE_BATCH`,
`LLAMA_GDS_WORK_CLAIM`, `LLAMA_GDS_ALIGNED_STAGING`,
`LLAMA_GDS_STAGE_UNALIGNED_DEST`, `LLAMA_GDS_THREAD_FILE_HANDLES`,
`LLAMA_GDS_PLAIN_READ`, and `LLAMA_GDS_IO_PROFILE`.

### Graph evaluation

With `-dsm` the scheduler asks the expert-routing callback about every graph
node, and by default the callback asks to see each one, so every node runs
and synchronizes on its own (no CUDA op fusion). `LLAMA_DSTORAGE_GROUPED_EVAL=1`
groups the nodes between routing callbacks for all graphs (the decision
profile sets it together with `GGML_CUDA_DISABLE_FUSION=1`).
`LLAMA_DSTORAGE_GROUPED_EVAL_DECODE=1` groups only graphs of at most 8 tokens
(decode, MTP drafts and verification) and keeps fusion on; contexts with
`moe_pooled_mmq` (decisions) are excluded, so decisions stay bit-exact in a
combined server. The Studio decode profile sets it: decode 19.4 -> 21.0 tok/s
(+8.5%), prompt processing unchanged. Decode text can differ slightly from
ungrouped runs because fused kernels round differently.

### Prefetch and staging

`LLAMA_DSTORAGE_PREDICTIVE_PREFETCH`, `LLAMA_DSTORAGE_PREFETCH_POLICY`,
`LLAMA_DSTORAGE_PREFETCH_MIN_CONF`, `LLAMA_DSTORAGE_PREFETCH_MIN_OBSERVATIONS`,
`LLAMA_DSTORAGE_PREFETCH_MAX_EXPERTS_PER_LAYER`,
`LLAMA_DSTORAGE_PREDICTIVE_PREFETCH_MAX_SUBMISSIONS`,
`LLAMA_DSTORAGE_PREDICTIVE_PREFETCH_MIN_DISTANCE`,
`LLAMA_DSTORAGE_GLOBAL_PREFETCH`, `LLAMA_DSTORAGE_PREFETCH_MAX_DISTANCE`,
`LLAMA_DSTORAGE_FORCE_PREFETCH_MAX_PENDING`,
`LLAMA_DSTORAGE_RELAX_PREFETCH_LAYER_GUARD`,
`LLAMA_DSTORAGE_PREFILL_LAYER_PRELOAD`,
`LLAMA_DSTORAGE_PREFILL_LAYER_PRELOAD_CURRENT`,
`LLAMA_DSTORAGE_PREFILL_LAYER_PRELOAD_DISTANCE`,
`LLAMA_DSTORAGE_PREFILL_LAYER_PRELOAD_MAX_EXPERTS`,
`LLAMA_DSTORAGE_PREFILL_WORKSPACE`, and
`LLAMA_DSTORAGE_PREFILL_WORKSPACE_LOOKAHEAD`.

The current profiles keep prefill workspace disabled because it adds a large
temporary expert workspace on top of the persistent cache. Lookahead is still
configured but has no effect while that workspace path is disabled.

### Cache layout and policy experiments

The default VRAM eviction policy for decode-sized batches (MTP verification and
draft steps) is "next use": it evicts the expert expected to be needed last,
estimated as the layer distance to its next forward pass plus `1/p - 1` passes,
where `p` is the expert's decayed per-pass use rate.
`LLAMA_DSTORAGE_NEXT_USE_HALF_LIFE` (default 3 passes) sets the decay.
Prompt-sized batches keep the older retention score, which measured better
there. `LLAMA_DSTORAGE_CACHE_POLICY=legacy` restores the older score for every
batch; the summary line reports `cache_policy=next_use|legacy`.

`LLAMA_DSTORAGE_CACHE_POLICY=hybrid_arc`,
`LLAMA_DSTORAGE_DECODE_HOT_CACHE`,
`LLAMA_DSTORAGE_DECODE_HOT_CACHE_SLOTS`,
`LLAMA_DSTORAGE_DECODE_HOT_MIN_HITS`,
`LLAMA_DSTORAGE_DECODE_HOT_REBUILD_INTERVAL`,
`LLAMA_DSTORAGE_DECODE_HOT_PREFILL_LIKE`,
`LLAMA_DSTORAGE_SPLIT_DOWN_POOLS`,
`LLAMA_DSTORAGE_LARGE_DOWN_SLOTS`,
`LLAMA_DSTORAGE_COALESCE_EXPERTS`,
`LLAMA_DSTORAGE_MAX_BATCH_STAGING_MIB`,
`LLAMA_DSTORAGE_SORT_BATCH_BY_OFFSET`,
`LLAMA_DSTORAGE_BUNDLE_CONTIGUOUS_READS`,
`LLAMA_DSTORAGE_BUNDLE_MAX_READ_KIB`,
`LLAMA_DSTORAGE_BUNDLE_STRIPE_KIB`,
`LLAMA_DSTORAGE_FUSED_EXPERT_POOLS`, and
`LLAMA_DSTORAGE_SPECULATIVE_SLOTS`.

### Static/dynamic pinned-cache policy

`LLAMA_MOE_PINNED_STATIC_EXPERTS_FILE`,
`LLAMA_MOE_PINNED_STATIC_LAYERS`,
`LLAMA_MOE_PINNED_CACHE_ADMIT_MIN_HITS`,
`LLAMA_MOE_PINNED_CACHE_ADMIT_MAX_PER_CALL`,
`LLAMA_MOE_PINNED_CACHE_ADMIT_MAX_PENDING_MIB`,
`LLAMA_MOE_PINNED_PREFILL_ONLY`, and
`LLAMA_MOE_PINNED_PREFILL_ONLY_MIN_PINNED_MIB`.

## 7. Important interactions and validation rules

- `--cpu-moe` / `--n-cpu-moe` cannot be combined with `--dstorage-moe`.
- `--dstorage-moe-prefetch` implies the DStorage backend.
- L1 VRAM budget shares the GPU with dense weights, KV cache, runtime buffers,
  and any draft model. Increasing it can cause CUDA OOM; reducing context or
  microbatch is another mitigation.
- Pinned expert L2 and general prompt-cache RAM are separate budgets and must
  be validated against available system RAM together.
- Native MTP requires model support. Standalone draft speculation requires a
  draft model and its own placement/memory budget.
- `LLAMA_MOE_EXPERT_USED_OVERRIDE` is quality-sensitive and should show native
  K beside effective K, with a warning when they differ.
- `CACHE_REUSE` should be disabled or visibly marked unsupported when the
  active model/runtime reports that capability as unavailable.
- GDS thread count is a concurrency limit, not a guaranteed transfer speed;
  the dashboard should show measured GDS/RAM-to-VRAM throughput separately.

## 8. Recommended catalogue hierarchy

The first screen should contain:

1. model identity and read-only architecture facts;
2. context window, L1 VRAM cache, pinned L2 RAM cache, prompt-cache RAM;
3. routed experts per token with native-K comparison;
4. speculation mode plus only the mode-relevant draft controls;
5. temperature and the main generation defaults;
6. GDS read threads and evict-to-RAM with its warm-hit threshold.

Everything else should be in expandable Advanced groups, with unsafe or
experimental environment-only switches clearly marked as such.

## Source trail

- `common/arg.cpp`: inherited and Siphon command-line parser options.
- `common/common.h`: defaults and parameter structures.
- `tools/server/README.md`: generated llama.cpp server flag reference.
- `README.md`: supported Siphon profile variables and practical defaults.
- `src/llama-model.cpp`: model metadata and routed-expert override.
- `src/llama-context.cpp`: runtime router callback, prefill/decode phases, and
  prefetch submission.
- `src/llama-dstorage-slots.cpp/.h`: VRAM slots, pinned L2, eviction,
  prefetch, and diagnostics.
- `ggml/src/ggml-cuda/dstorage_loader_gds.cpp`: GDS worker and I/O controls.
- `MOE_CPP_PROJECT_DEEP_DIVE.md`, `MULTI_MODEL_PROGRESS.md`, and
  `SIPHON_STUDIO_PROGRESS.md`: measured behavior and tuning conclusions.
