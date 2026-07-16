# moe.cpp Project Deep Dive

Last updated: 2026-07-15

This file is a detailed handoff for the local `moe.cpp` project in:

```text
/home/manohar/Desktop/inference/moe.cpp
```

It explains what was implemented, how the runtime works end to end, what each
major state means, which experiments worked, which experiments failed, and
where the important code and artifacts live. It is written for a future agent or
developer who needs to understand the project without replaying the full
conversation.

## Short Summary

The project is a fork of `llama.cpp` focused on running Qwen3.6/Qwen3.5 MoE
GGUF models on a Linux laptop with limited VRAM.

The main innovation is a Linux NVIDIA GPUDirect Storage path for exact routed
MoE expert streaming:

```text
router selects experts -> missing experts are streamed from SSD/RAM -> VRAM slot
cache is patched into the graph -> MUL_MAT_ID computes with the selected slots
```

The current practical architecture is:

- Dense and non-expert model weights are loaded through normal llama.cpp GPU
  offload.
- MoE expert tensors are not fully loaded at startup.
- Expert tensor metadata is registered from the GGUF file offsets.
- Runtime router IDs are intercepted by an eval callback.
- Missing expert slices are loaded into a persistent VRAM expert cache using
  NVIDIA GDS/cuFile or optional pinned host RAM.
- The router ID tensor is rewritten from logical expert IDs to VRAM slot IDs.
- The expert tensor metadata is temporarily patched to point at slot pools.
- After the MoE output finishes, tensor metadata and execution pins are restored.

The best stable local speed path is still the original on-demand GDS/VRAM-cache
architecture, not blind full-layer preloading.

## Latest Current Finding Snapshot

This section captures the newest practical state from the latest experiments so
it is visible before the deeper implementation notes.

Current implementation:

```text
original on-demand GDS MoE streaming architecture
VRAM expert slot cache
optional static pinned RAM expert cache
MTP draft speculation
FlashAttention enabled
top-k override available through LLAMA_MOE_EXPERT_USED_OVERRIDE
single-turn CLI/server usage when the IDE is sensitive to interactive output
```

The current checked-out source is not the DFlash/BeeLlama branch and not the
full-layer prefill preload branch. Those were investigated, documented, and
left as references. The active useful path is still demand-loading exactly the
experts selected by the router.

Newest result that changed the interpretation:

- Reducing routed experts from K=8 to K=4 is much more promising than initially
  assumed.
- K=4 does not simply look broken or obviously worse. In the corrected broad
  hard-task comparison it was close to K=8 and sometimes better.
- K=4 roughly halves selected expert pressure, which makes the same VRAM cache
  behave like a larger cache.
- The latest large markdown summarization test with K=4 and reasoning on gave
  about 34.3 prompt tokens/s and about 12.0 generation tokens/s.
- In that run, prefill streamed about 1839 MiB/s and decode streamed about
  1708 MiB/s, with about 39.9% prefill hit rate and about 86.2% decode hit
  rate.
- The remaining bottleneck is not attention. The Kascade profiling showed
  FlashAttention was a tiny part of wall time; expert movement and
  miss-bearing MoE calls dominate long-prompt prefill.
- The next important speed target is improving effective GDS throughput and
  reducing expert misses/traffic, especially in prefill.

Current conclusion:

```text
K=8 remains the conservative quality baseline.
K=6 remains a safer reduced-top-k candidate.
K=4 is now a serious fast mode candidate, not a toy setting.
The project should compare K=4/K=6/K=8 on real coding-agent workloads before
choosing a default.
```

### 2026-07-03/04 update: fused TQ3 attention + opencode serving fixes

Three findings from the latest opencode serving session materially changed the
long-context and agent-use picture. Full detail lives in
`FLASH_ATTENTION_DEQUANT_ISSUE.md` and the two opencode handoff files.

1. Long-context decode bottleneck was the quantized KV path, not expert
   streaming. With `-ctk tq3_0 -ctv tq3_0 -fa on` the old CUDA flash-attention
   path first expanded/dequantized K/V into F16 temporaries. In the main decode
   graph `copy_view_layout` (the KV dequant/copy/layout) was about 57% of graph
   compute while attention math was about 5%. A fused `TQ3_0/TQ3_0` flash-
   attention specialization now decodes K/V inside the kernel and skips the
   graph-level `k_dequant`/`v_dequant` casts. Long README K=4 decode roughly
   doubled: 6.6 -> ~13.5-15.7 t/s; `k_dequant`/`v_dequant` node counts are 0.

2. The fix reaches `llama-server`, not just `llama-cli`. It lives in the shared
   libs (`libggml-cuda.so.0.13.1`, `libllama.so.0.0.1237`, both 2026-07-03
   14:02), so an old server binary still gets it after a restart. Verified via
   `/proc/<pid>/maps` that a fresh server maps both fused libs, and measured
   ~13.6-14 t/s decode at ~10-14k context through `/completion`. In a live
   opencode website build, a file-writing step decoded a ~4500-token file at a
   sustained ~14 t/s (vs ~5.5 t/s on the pre-fused server that same session) --
   the ~2.5x win holds on real agent content. Condition: opencode only benefits
   if pointed at a server (re)started after the 14:02 fused build.

3. opencode follow-up/step reprocessing was a chat-template cache bug, now
   fixed. The template stripped the empty `<think></think>` block from assistant
   turns preceding the last user message, so every new user turn diverged from
   the KV cache at the first assistant token and reprocessed the whole tail
   (~13.6k tok ~= 4 min). Fix: `--chat-template-kwargs '{"preserve_thinking":
   true}'` in `running_server`. A/B: follow-up 9.8s/3 reprocess passes ->
   1.0s/0 passes. Remaining separate cost: opencode title/summarize(compaction)
   aux tasks run on the same single slot, use a different system prompt (diverge
   at token 3), and both reprocess ~12.8k tokens and evict the build cache on
   long sessions. Open lever: route opencode `small_model` to a separate
   endpoint.

### 2026-07-11 to 2026-07-15 update: Memory Optimization & Prompt Cache Tuning

During agentic stress testing up to 32K context, critical memory and cache behaviors were identified and tuned to ensure high speeds and system stability on 16 GB RAM hardware.

1. **Prompt Cache Size Tuning (`CACHE_RAM_MIB`):**
   - **The Problem:** The default `CACHE_RAM_MIB=512` (512 MiB limit) was too small. Under multi-turn reasoning loops with context checkpoints enabled, prompt cache states easily grew past 512 MiB (up to ~800+ MiB). This forced `llama-server` to erase slot checkpoints, causing **forced full prompt re-pre-fills** on the next turn, with latency spikes up to 45 seconds.
   - **The Fix:** Raised `CACHE_RAM_MIB=2048` (2.0 GiB) to provide ample headroom. Checkpoint invalidations are eliminated, preserving fast prefix cache reuse across multi-turn exchanges.

2. **Context Checkpoints RAM Limit (`LLAMA_ARG_CTX_CHECKPOINTS`):**
   - **The Problem:** The server defaulted to keeping 32 context checkpoints. At long context lengths (15k–30k+), each checkpoint consumes ~80-180 MiB of host CPU RAM, causing the server's footprint to exceed 11 GiB and crash the 16 GB system.
   - **The Fix:** Set `export LLAMA_ARG_CTX_CHECKPOINTS=12` to limit active checkpoints. This caps checkpoint RAM usage at ~1.4 GiB, preventing system OOMs without affecting context retrieval.

3. **Pinned Expert Cache (`MOE_PINNED_CACHE_MIB`):**
   - **The Problem:** Attempting to free memory immediately after prefill (`LLAMA_MOE_PINNED_PREFILL_ONLY=1`) with a reduced cache (`1536 MiB`) led to a severe decode speed regression, dropping from **~18-20 tok/s to ~10 tok/s** because hot experts had to be re-fetched from NVMe on every decode step.
   - **The Fix:** Disabled the prefill-only release flag and kept the pinned expert budget warm (`MOE_PINNED_CACHE_MIB=4096`). Once warmed up after the first turn, decode speed recovers to its full **~18-20 tok/s** steady-state target.

## Current Branch And Git State

Current branch observed while writing this file:

```text
codex/moe-topk-sweep-experiment
```

Recent relevant commits:

```text
89bc10cf5 Mark GPQA top-k eval parser caveat
e8476d487 Correct top-k 4 broad eval conclusion
b2afd689e Compare top-k 4 and 8 on broad hard eval
d0851de1a Add real hard top-k evaluation
dabc19be2 Record top-k 4 quality sweep
f9c488abb Record reasoning-on top-k sweep
9ad0f75d1 Add top-k 5 quality sweep report
29f3b4a34 Add top-k 6 quality sweep report
ceec3211e Add experimental MoE top-k override
```

There are many untracked benchmark outputs and trace artifacts in this
workspace. Do not use destructive cleanup casually. Always check:

```bash
cd /home/manohar/Desktop/inference/moe.cpp
git status --short
```

before changing source or docs.

## Workspace Map

Top-level workspace:

```text
/home/manohar/Desktop/inference
```

Important directories and files:

```text
moe.cpp/                 current main project
siphon.cpp/               duplicate/snapshot of the GDS MoE tree
mellum.cpp/              sibling snapshot with similar docs/artifacts
gemma.cpp/               sibling snapshot with similar docs/artifacts
llm_upper/               earlier Windows/Microsoft DirectStorage prototype notes
markdown files/          older reports and prefill experiment notes
dflash_refs/beellama.cpp BeeLlama reference checkout for DFlash research
qwen3.6/                 local Qwen3.6 model files and pasted DFlash drafter
harness/                 coding-agent evaluation harness and PI-agent work
router tracing /         Kaggle trace bundles and downloaded routing results
running_model            current local llama-cli command scratch file
key_notes                tiny note file with current next ideas
start-qwen-*.sh          local server launch helpers outside the git repo
```

Primary model:

```text
/home/manohar/Desktop/inference/qwen3.6/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

Important current static expert files:

```text
moe-static-experts-suite-8192.txt
moe-static-experts-suite-10240.txt
moe-static-experts-coding-600p-8192.txt
moe-static-experts-coding-600p-10240.txt
```

`moe-static-experts-suite-8192.txt` starts with:

```text
# budget_mib=8192 selected=4491 selected_mib=8190.41 ...
```

`moe-static-experts-coding-600p-8192.txt` starts with:

```text
# budget_mib=8192 selected=4495 phase=prefill selected_mib=8191.68 ...
```

The coding-specific files came from larger coding-assistant routing traces.
They should be tested on held-out prompts before treating them as a universal
replacement for the suite files.

## Why This Project Exists

The early project was Windows-oriented and replicated Microsoft DirectStorage
for MoE expert loading. After moving to Linux, the correct native storage path
became NVIDIA GPUDirect Storage.

The target use case is a Qwen3.6 35B A3B MoE GGUF model on a machine where:

- the full model cannot fit in VRAM;
- expert tensors are too large to keep resident;
- decode needs to stay usable for coding-agent workflows;
- long prompt prefill is the main remaining pain point.

The runtime therefore keeps only a working set of experts in VRAM and streams
the rest exactly when the model router requests them.

## Core Source Files

### `src/llama-dstorage-slots.h`

Declares the runtime manager and core data structures:

- `dstorage_moe_phase`
- `ExpertTensorInfo`
- `ExpertSlot`
- `DStorageSlotManager`
- cache layout helpers
- request-local activation statistics
- eviction scoring helpers
- speculative slot helpers
- pinned RAM entry structs
- summary statistics structs

This header is also used by `tests/test-dstorage-cache-layout.cpp`, so many
cache policies have direct unit-testable helper functions.

### `src/llama-dstorage-slots.cpp`

Implements the actual expert cache and streaming runtime:

- environment variable parsing;
- GDS loader initialization;
- VRAM pool allocation;
- static pinned RAM cache loading;
- dynamic pinned RAM admission;
- expert hit/miss classification;
- eviction and slot protection;
- GDS batch request construction;
- host-to-device pinned cache copies;
- runtime route tracing;
- timeline and scheduler dry-run accounting;
- final `DSTORAGE_MOE_*` summary printing.

This is the most important file in the project.

### `ggml/src/ggml-cuda/dstorage_loader.h`

Defines the cross-platform `ds_loader_*` ABI. The name still says
DirectStorage because this project started on Windows. The same API is now used
for:

- Windows Microsoft DirectStorage;
- Linux NVIDIA GPUDirect Storage/cuFile.

Important calls include:

```text
ds_loader_available
ds_loader_create
ds_loader_stream_to_cuda_batch
ds_loader_host_alloc
ds_loader_host_to_cuda_batch
ds_loader_cuda_alloc
ds_loader_cuda_free
ds_loader_cuda_wait_event
```

### `ggml/src/ggml-cuda/dstorage_loader_gds.cpp`

Linux implementation of the loader ABI.

It dynamically loads:

```text
libcuda.so.1 or libcuda.so
libcufile.so.0 or libcufile.so
```

It retains the CUDA primary context, opens the cuFile driver, registers file
handles and CUDA buffers when needed, and submits direct storage reads into CUDA
device pointers. It also provides CUDA pinned host memory and host-to-device
batch copies for the pinned RAM cache.

Important environment variables:

```text
LLAMA_GDS_READ_THREADS
LLAMA_GDS_REGISTER_PER_REQUEST
LLAMA_GDS_USE_BATCH
LLAMA_GDS_IO_PROFILE
```

The commonly tested value is:

```text
LLAMA_GDS_READ_THREADS=20
```

### `src/llama-model-loader.cpp`

Adds two key DirectStorage hooks:

1. `llama_dstorage_parse_expert_tensor()` detects Qwen expert tensors:

```text
blk.<layer>.ffn_gate_up_exps.weight
blk.<layer>.ffn_down_exps.weight
blk.<layer>.ffn_gate_exps.weight
blk.<layer>.ffn_up_exps.weight
```

2. `create_tensor()` creates one-expert placeholder tensors for expert weights
   when DirectStorage/GDS is active.

Instead of allocating/loading the whole expert tensor, it creates:

```text
[dim0, dim1, 1 expert]
```

as a placeholder. Later, `load_all_data()` registers the real GGUF tensor file
offset and size with `DStorageSlotManager::register_expert_tensor()` and skips
normal GPU payload loading for that expert tensor.

### `src/llama-model.cpp`

Owns `std::unique_ptr<DStorageSlotManager>` inside `llama_model::impl`.

During model loading, if `params.dstorage_moe` is true:

- validates the architecture is currently Qwen MoE;
- validates GPU offload is enabled;
- creates the slot manager;
- calls `init(n_layers, n_experts, n_expert_used, gpu_cache_mib, pinned_cache_mib, false)`;
- passes the manager pointer into `llama_model_loader`.

This file also implements the experimental top-k override:

```text
LLAMA_MOE_EXPERT_USED_OVERRIDE=N
```

If `N` is positive and smaller than the model's normal `n_expert_used`, the
runtime uses fewer active experts per token. This is the mechanism behind the
K=4/K=5/K=6/K=8 experiments.

### `src/llama-context.cpp`

This file contains the graph-time callback:

```text
llama_context::dstorage_eval_callback()
```

The callback does the exact expert loading and tensor patching at runtime.

It watches for tensors named:

```text
moe_selected_experts_ds-<layer>
moe_selected_experts_trace-<layer>
ffn_moe_out-<layer>
```

At router callback time it:

1. synchronizes the backend scheduler;
2. reads selected expert IDs from the router tensor;
3. classifies the phase as prefill or decode;
4. calls `ensure_experts_loaded()`;
5. patches expert tensor metadata to point at VRAM slot pools;
6. rewrites the router tensor IDs to slot IDs;
7. optionally submits prefetch work.

At `ffn_moe_out` callback time it:

1. records MoE compute time;
2. restores the original tensor data and strides;
3. releases execution pins for that layer.

### `src/llama-graph.cpp`

In `build_moe_ffn()`, DirectStorage mode duplicates the selected-experts tensor
and names it:

```text
moe_selected_experts_ds
```

Portable trace mode names it:

```text
moe_selected_experts_trace
```

This is what gives the scheduler callback an execution point after routing and
before expert matmul.

### `src/models/qwen35moe.cpp`

Wires the Qwen3.5/Qwen3.6 MoE graph to pass:

```text
cparams.dstorage_moe
cparams.moe_routing_trace
```

into `build_moe_ffn()`.

It also contains the Qwen MTP graph path. The local Qwen3.6 GGUF has an MTP
head, so `--spec-type draft-mtp` works without a separate draft model.

### `common/arg.cpp`

Adds the user-facing flags:

```text
-dsm, --dstorage-moe
--dstorage-moe-prefetch
--moe-gpu-cache-mib N
--moe-pinned-cache-mib N
```

`--dstorage-moe` cannot be combined with `--cpu-moe`.

`--dstorage-moe-prefetch` implies `--dstorage-moe`.

### `common/speculative.cpp`

Contains the upstream-style speculative decoding implementations used here,
especially:

```text
--spec-type draft-mtp
```

This is not BeeLlama DFlash. Current checked-out source has MTP but no active
DFlash implementation.

### Tests

Important local tests:

```text
tests/test-dstorage-cache-layout.cpp
tests/test-dstorage-stream-bench.cpp
```

`test-dstorage-cache-layout.cpp` checks:

- persistent cache layout;
- zero workspace reservation;
- phase cache calculations;
- decode workspace capacity;
- bounded prefill token chunks;
- speculative slot partitioning;
- request-local activation stats;
- retention score behavior;
- hybrid ARC score behavior;
- execution/transfer/admission slot protection.

`test-dstorage-stream-bench.cpp` opens a GGUF, registers one layer's expert
tensors, streams selected experts, and verifies direct persistent-slot
execution by checking that no active staging slot copy is used.

## Build Integration

Relevant CMake wiring:

```text
src/CMakeLists.txt
ggml/src/ggml-cuda/CMakeLists.txt
tests/CMakeLists.txt
```

`src/CMakeLists.txt` includes:

```text
llama-dstorage-slots.cpp
../ggml/src/ggml-cuda/dstorage_loader.cpp on Windows
../ggml/src/ggml-cuda/dstorage_loader_gds.cpp on Linux
```

The main local build used for serious tests is:

```text
build-gds-cuda/bin/llama-cli
build-gds-cuda/bin/llama-server
```

Typical CMake shape:

```bash
cmake -S . -B build-gds-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_FA=ON \
  -DLLAMA_BUILD_TESTS=ON
cmake --build build-gds-cuda -j
```

The exact build options have changed during experiments, but the important
requirements are CUDA, Flash Attention, and the GDS loader source.

## End-To-End Runtime Pipeline

### 1. CLI/server flags are parsed

User flags such as:

```text
-dsm
--dstorage-moe-prefetch
--moe-gpu-cache-mib 2816
--moe-pinned-cache-mib 8192
```

populate `common_params`, then `llama_model_params` and `llama_context_params`.

### 2. Model hparams are loaded

`llama_model_base::load_hparams()` reads:

```text
n_layer_all
n_expert
n_expert_used
n_expert_groups
n_group_used
```

If `LLAMA_MOE_EXPERT_USED_OVERRIDE` is set and valid, `n_expert_used` is reduced
before graph construction. That changes router top-k and all later expert work.

### 3. Layer devices are assigned

Normal llama.cpp offload assigns layers to GPU/CPU devices using `-ngl` and
split settings. For the production local runs:

```text
-ngl 999
```

is used so the runtime tries to offload all possible non-expert work.

### 4. Slot manager is initialized

When `params.dstorage_moe` is true, `llama_model.cpp` creates:

```text
DStorageSlotManager
```

with:

```text
n_layers
n_experts
n_expert_used
moe_gpu_cache_mib
moe_pinned_cache_mib
phase_cache_policy=false
```

The manager does not allocate all VRAM pools immediately. Pool allocation is
deferred until expert tensor sizes are known and the first runtime load needs
the pools.

### 5. Expert tensors become placeholders

During tensor creation, Qwen expert tensors are not allocated at full expert
count. They become one-expert placeholders.

This avoids paying full expert VRAM at model load.

### 6. Expert file offsets are registered

During `load_all_data()`, each expert tensor's real GGUF metadata is registered:

```text
layer index
tensor name
GGUF file path
byte offset
total tensor size
number of experts
row/block alignment size
```

The slot manager uses this to calculate:

```text
file_offset = tensor_file_offset + expert_id * expert_stride
```

for each future expert miss.

### 7. Graph marks router output tensors

The graph builds the normal MoE router. In DirectStorage mode, the selected
expert IDs are duplicated and named so the eval callback can stop at that exact
point:

```text
moe_selected_experts_ds-<layer>
```

The graph itself remains normal after this point.

### 8. Eval callback reads router IDs

When the scheduler reaches the selected-experts tensor, the callback:

- synchronizes the graph scheduler;
- copies expert IDs from backend memory to CPU;
- determines if this call is prefill or decode;
- calls `DStorageSlotManager::ensure_experts_loaded()`.

Current simple phase rule:

```text
n_ids > top_k  -> prefill
n_ids <= top_k -> decode
```

There is additional decode-like accounting for small MTP batches, but the main
phase split above is the core rule.

### 9. Slot manager loads missing experts

`ensure_experts_loaded()`:

1. builds a unique expert list;
2. checks existing resident VRAM slots;
3. checks pinned RAM entries for L2 hits;
4. reserves or evicts one slot per miss;
5. creates host-to-device requests for pinned RAM hits;
6. creates GDS file-to-GPU requests for cold SSD misses;
7. coalesces adjacent expert IDs when file and slot layout allow it;
8. submits chunked GDS batches;
9. marks target slots occupied;
10. returns slot pool base pointers and router ID remaps.

### 10. Callback patches tensor metadata

The callback temporarily changes the expert tensors for that layer:

```text
w->data  = slot_pool_cuda_pointer
w->ne[2] = number_of_execution_slots
w->nb[2] = slot_stride
w->nb[3] = ne[2] * nb[2]
```

It also rewrites the selected expert tensor from:

```text
logical expert ids
```

to:

```text
pool-local slot ids
```

Then `MUL_MAT_ID` runs against the slot pool as if it were a normal expert
tensor.

### 11. MoE output restores the layer

When `ffn_moe_out-<layer>` is reached, `dstorage_restore_layer()` restores the
original tensor metadata and releases execution pins on the slots used by the
layer.

Execution pins are important: without them, an async prefetch or another load
could evict a slot while the graph still needs it.

### 12. Shutdown prints summaries

If enabled:

```text
LLAMA_DSTORAGE_SUMMARY=1
```

the manager prints `DSTORAGE_MOE_*` summaries during destroy. These are the
main source of cache hit rates, streaming byte counts, transfer time, pinned RAM
usage, and dry-run overlap analysis.

## Runtime States And Phases

### Model-load state

Before runtime inference:

- no expert slot pools are populated;
- expert tensor placeholders exist;
- file offsets and expert strides are registered;
- GDS loader is available or initialization fails;
- static pinned expert list is parsed but not necessarily loaded until pools and
  tensor registry are available.

### Prefill phase

Prefill means the model is processing prompt tokens. In this project it is the
hardest phase because many prompt tokens route to many unique experts.

Characteristics:

- many selected expert IDs per callback;
- low exact reuse compared with decode;
- many first-use misses;
- large SSD/RAM traffic;
- VRAM hit rate can be much lower than decode;
- static pinned RAM helps prompt throughput by avoiding part of SSD traffic.

DirectStorage prefill microbatching is bounded by:

```text
max_prefill_workspace_experts() = 192
prefill token chunk = 192 / n_expert_used
```

For top-k 8, this is about 24 tokens per physical DirectStorage prefill chunk.

### Decode phase

Decode means generated-token processing. It has much better expert reuse.

Characteristics:

- usually top-k experts per layer;
- resident expert reuse is high;
- VRAM cache hit rate can reach roughly 90-95% in good long runs;
- MTP can reduce target decode calls;
- storage is no longer the only decode bottleneck.

### Prefetch phase

Prefetch means speculative cache loading that is not yet required by the actual
router.

Prefetch is disabled unless:

```text
--dstorage-moe-prefetch
```

is set. Speculative slots can be reserved with:

```text
LLAMA_DSTORAGE_SPECULATIVE_SLOTS=N
```

Prefetch uses admission controls so low-confidence predictions do not freely
evict useful demand-loaded entries.

### Pinned RAM states

Pinned RAM cache states:

- disabled: `--moe-pinned-cache-mib 0`
- static preload pending: static list parsed but not loaded yet
- static preload done: selected experts occupy pinned host memory
- dynamic admission pending: background thread is filling RAM entries
- prefill-only released: optional RAM-saving mode has freed the RAM cache

The main static list environment variable is:

```text
LLAMA_MOE_PINNED_STATIC_EXPERTS_FILE=/path/to/list.txt
```

Prefill-only release mode:

```text
LLAMA_MOE_PINNED_PREFILL_ONLY=1
```

This frees the pinned RAM cache after enough prefill pinned traffic has been
served.

### Slot states

Each `ExpertSlot` has:

```text
layer_idx
expert_idx
pool_group
pool_index
occupied
admission_pending
transfer_pending
execution_pins
last_used_tick
prefill_hits
decode_hits
prefetch_hits
arc_segment
contiguous_neighbors
```

A slot is protected from eviction when:

```text
execution_pins > 0
transfer_pending
admission_pending
```

Decode-hot protection is optional and tracked separately.

## VRAM Expert Cache

The VRAM cache is a persistent slot table.

One logical slot stores one `(layer_idx, expert_idx)` across all expert tensor
types for that layer. The slot does not hold separate ownership per tensor;
instead each tensor type has a CUDA pool and the slot index selects an offset.

For example, Qwen may have pools for:

```text
ffn_gate_exps.weight
ffn_up_exps.weight
ffn_down_exps.weight
ffn_gate_up_exps.weight when present
```

The current design removed the old full active workspace reservation. The GPU
cache budget now maps to persistent cache capacity, not persistent capacity
minus a large staging workspace.

Important improvements already implemented:

- persistent cache capacity uses the requested budget;
- active workspace reservation is zero at initialization;
- decode workspace is top-k bounded;
- prefill is bounded to at most 192 unique experts per physical chunk;
- computation reads directly from persistent pools;
- selected router IDs are remapped to slot IDs;
- execution pins protect in-flight layer slots;
- transfer and admission states block unsafe eviction.

### Compacted pool allocation

The slot manager can split down-projection pools into groups when different
layers have different expert strides.

This reduces internal fragmentation compared with allocating every slot at the
largest stride for every layer.

Relevant knobs:

```text
LLAMA_DSTORAGE_SPLIT_DOWN_POOLS
LLAMA_DSTORAGE_LARGE_DOWN_SLOTS
```

This was part of the effort to recover extra VRAM without reducing performance.

## Eviction Policy

The baseline eviction policy is request-aware and layer-aware, stronger than
plain LRU.

Main signals:

- current request decode reuse;
- current request prefill reuse;
- historical decode hits;
- historical prefill hits;
- layer urgency;
- shallow-layer priority;
- prefetch pollution penalty;
- LRU tie-breaker;
- slot protection for execution, transfer, and admission.

The policy is implemented in:

```text
DStorageSlotManager::calculate_retention_score()
DStorageSlotManager::is_better_request_eviction_candidate()
```

There is also an experimental hybrid ARC policy:

```text
LLAMA_DSTORAGE_CACHE_POLICY=hybrid_arc
```

Hybrid ARC keeps per-layer recent/frequent state and adds reload-byte and
contiguous-neighbor value. It reduced some misses in tests but did not produce
a reliable end-to-end speed win, so it remains disabled by default.

## Pinned RAM Cache

Pinned RAM is an L2 cache for experts outside VRAM. A pinned RAM hit still needs
a host-to-device copy before `MUL_MAT_ID`, but it avoids SSD/GDS file reads.

Two modes exist:

### Static pinned cache

Static mode loads a preselected list at startup:

```text
LLAMA_MOE_PINNED_STATIC_EXPERTS_FILE=moe-static-experts-suite-8192.txt
--moe-pinned-cache-mib 8192
```

The list file format is:

```text
layer expert miss_count access_count bytes
```

Static pinned cache is the mode that worked best.

### Dynamic pinned admission

Dynamic admission can add experts to pinned RAM during inference if they become
hot enough.

Important knobs:

```text
LLAMA_MOE_PINNED_CACHE_ADMIT_MIN_HITS
LLAMA_MOE_PINNED_CACHE_ADMIT_MAX_PER_CALL
LLAMA_MOE_PINNED_CACHE_ADMIT_MAX_PENDING_MIB
```

Dynamic mode works mechanically but is usually worse for one-shot CLI runs
because admission work happens during inference. It may still matter for long
server sessions.

### RAM cache size findings

From `MOE_RAM_CACHE_SIZE_SWEEP.md`:

```text
0 MiB     prompt 10.2 t/s, gen 8.2 t/s
2048 MiB prompt 12.8 t/s, gen 9.0 t/s
4096 MiB prompt 12.3 t/s, gen 8.6 t/s
6144 MiB prompt 12.5 t/s, gen 8.9 t/s
8192 MiB prompt 15.8 t/s, gen 9.5 t/s
10240 MiB prompt 16.3 t/s, gen 9.4 t/s
```

The 8 GiB static cache is the best practical tradeoff on a 16 GiB RAM system.
The 10 GiB cache reduces SSD traffic more but did not win speed and increases
RAM pressure.

## GDS Backend Details

The Linux GDS backend uses CUDA driver APIs and cuFile through dynamic loading.

Important implementation choices:

- dynamic `dlopen` keeps the binary buildable without GDS installed;
- CUDA primary context is retained instead of creating an unrelated context;
- cuFile file handles are registered for direct reads;
- CUDA device buffers are allocated with `cuMemAlloc`;
- pinned host memory uses `cuMemHostAlloc` when available;
- fallback aligned RAM is allowed if pinned allocation fails;
- batch reads can be split by staging memory limit.

Important GDS performance reality:

The NVMe may benchmark at 6.6 GB/s, but MoE expert streaming is not a single
large sequential file read. It is many small or medium GGUF slices, often with
synchronization and per-request overhead. Effective GDS speed in real model
runs has ranged much lower than raw SSD specs, commonly around 1.7-3.5 GiB/s
depending on run shape, cache hits, request count, and thermal/system state.

## Runtime Metrics

Enable:

```text
LLAMA_DSTORAGE_SUMMARY=1
```

Important summary lines:

### `DSTORAGE_MOE_SUMMARY`

Reports slot occupancy:

```text
slots
occupied
decode_touched
prefill_only
prefetch_touched
speculative_slots
cache_policy
pinned_entries
static_pinned_entries
pinned_used_mib
transfer_ewma_us
layer_compute_ewma_us
```

### `DSTORAGE_MOE_PHASE`

Reports phase hit/miss behavior:

```text
prefill_calls, prefill_hits, prefill_misses
decode_calls, decode_hits, decode_misses
decode_hit_rate
prefetch_calls, prefetch_hits, prefetch_misses
```

### `DSTORAGE_MOE_STREAM`

Reports I/O and transfer volume:

```text
prefill_file_mib
prefill_pinned_mib
prefill_total_us
decode_file_mib
decode_pinned_mib
decode_total_us
requests
chunks
staging_mib
submit_us
wait_us
```

Effective GDS rate can be estimated as:

```text
file_mib / (total_us / 1e6)
```

### `DSTORAGE_MOE_PINNED`

Reports pinned RAM usage and release:

```text
budget_mib
used_mib
entries
queued
completed
skipped_policy
prefill_only_releases
prefill_only_release_mib
prefill_only_release_us
```

### `DSTORAGE_MOE_TIMELINE`

Reports broad latency accounting:

```text
real_calls
prefill_calls
decode_calls
selected_experts
unique_experts
hit_rate
ensure_total_us
stream_total_us
stream_file_mib
compute_avg_us
```

### `DSTORAGE_MOE_SCHED_DRYRUN_*`

Reports how much transfer could theoretically have been hidden behind prior
layer compute:

```text
hidden_ratio
file_fit_ratio
full_hide_calls
partial_hide_calls
visible_after_overlap_us
```

Low hidden ratio means prefetch/scheduling has limited room unless prediction
quality improves or larger useful work chunks are created.

## Current Practical Commands

### Stable llama-cli shape

Use `-st` for local IDE safety.

```bash
cd /home/manohar/Desktop/inference/moe.cpp

LLAMA_GDS_READ_THREADS=20 \
LLAMA_DSTORAGE_SUMMARY=1 \
LLAMA_MOE_PINNED_STATIC_EXPERTS_FILE=/home/manohar/Desktop/inference/moe.cpp/moe-static-experts-suite-8192.txt \
build-gds-cuda/bin/llama-cli \
  -m /home/manohar/Desktop/inference/qwen3.6/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  -ngl 999 \
  -c 4096 \
  -fa on \
  -dsm \
  --dstorage-moe-prefetch \
  --moe-gpu-cache-mib 2816 \
  --moe-pinned-cache-mib 8192 \
  --spec-type draft-mtp \
  --spec-draft-n-max 3 \
  --spec-draft-p-min 0 \
  --reasoning off \
  --reasoning-budget 0 \
  -s 4242 \
  -st \
  -n 500 \
  -p "Write a complete Python implementation of an LRU cache."
```

### RAM-saving prefill-only shape

```bash
LLAMA_MOE_PINNED_PREFILL_ONLY=1
```

This can save host RAM after prefill, but it is not always the fastest decode
mode.

### Top-k 4 experiment shape

```bash
LLAMA_MOE_EXPERT_USED_OVERRIDE=4
```

Use this only when intentionally testing reduced expert count. It is much
faster, but not yet proven zero-loss for all coding-agent workloads.

## MTP Integration

The local Qwen3.6 GGUF has built-in MTP/NextN tensors:

```text
general.architecture = qwen35moe
qwen35moe.nextn_predict_layers = 1
blk.40.nextn.* tensors present
```

MTP can be enabled with:

```text
--spec-type draft-mtp
--spec-draft-n-max 2 or 3
--spec-draft-p-min 0
```

From `MTP_DIRECTSTORAGE_BENCHMARK.md`, early 128-token results:

```text
baseline generation: 3.9 t/s
MTP n=2 generation: 5.3 t/s
MTP n=3 generation: 5.2 t/s
MTP n=4 generation: 5.3 t/s
```

Later local practice found MTP draft 3 can be good, but draft 2 is often more
balanced on the RTX 4050. The best choice depends on prompt, thermal state, and
the top-k override.

MTP changes runtime accounting. Many verification/draft steps appear as
prefill-like calls to the DirectStorage slot manager even though they are part
of generation. This is why MTP runs can show high prefill traffic during decode.

## Top-K Override Experiments

The model normally routes top-k 8 experts per token.

The project added:

```text
LLAMA_MOE_EXPERT_USED_OVERRIDE=N
```

to test fewer active experts.

Main observations so far:

- K=6 looked like the safest reduced-top-k candidate in small reasoning-on
  schedule tests.
- K=5 was faster but showed first-answer instability on at least one reasoning
  task.
- K=4 is a serious speed candidate and should not be dismissed.
- K=4 can be head-to-head with K=8 on several hard tasks and even beat K=8 on
  some science tasks.
- K=4 also had at least one real coding-format failure where K=8 passed.
- A GPQA parser bug affected an earlier report, so GPQA pass/fail counts in
  that run are provisional.

From `MOE_K4_VS_K8_BROAD_EVAL.md`, the corrected broad-run view:

```text
K=4 better: 2 tasks
K=8 better: 1 task
tied: 9 tasks
```

Speed comparison from the same report:

```text
K=4 prompt speed: 2.51x K=8
K=4 generation speed: 1.67x K=8
K=4 streamed file MiB: 0.35x K=8
K=4 selected experts: 0.52x K=8
```

Recent large markdown summary test with K=4 reasoning on:

```text
input file: moe.cpp/README.md
input tokens: about 10355
prompt speed: 34.3 t/s
generation speed: 12.0 t/s
prefill stream: about 1839 MiB/s
decode stream: about 1708 MiB/s
prefill hit rate: about 39.9%
decode hit rate: about 86.2%
```

Conclusion: K=4 makes the cache effectively larger because it halves selected
expert pressure. It is promising, but the final default should be decided with
more coding-agent specific tests, not assumptions.

## Routing Trace Pipeline

Routing traces are how static expert lists are built.

### Trace recording modes

There are two related trace paths:

1. GDS path trace:

```text
LLAMA_DSTORAGE_ROUTING_TRACE=output.jsonl
```

This runs with `--dstorage-moe` and records real misses and stream timing from
the active cache policy.

2. Portable non-GDS trace:

```text
LLAMA_MOE_ROUTING_TRACE=output.jsonl
```

This runs without `--dstorage-moe` and records exact router expert sets for
portable/offline tracing. It was useful for Kaggle T4 tracing where GDS was not
needed.

### Trace event schema

Trace JSONL records include:

```text
version
sequence
request
phase
layer
slots
expert_bytes
stream_file_bytes
transfer_us
total_us
experts
misses
```

### Prompt-suite runners

Important scripts:

```text
scripts/run-moe-routing-trace.py
scripts/monitor-moe-routing-trace.py
scripts/finalize-moe-routing-trace.py
scripts/postrun-moe-routing-cache-test.py
scripts/select-static-moe-experts.py
scripts/select-hybrid-moe-experts.py
scripts/moe-cache-sim.py
```

`run-moe-routing-trace.py` posts JSON completion requests to a running
`llama-server`. It is resumable because it writes a per-prompt progress JSONL
and skips recorded indices.

`finalize-moe-routing-trace.py` validates raw trace shards and rewrites request
IDs so combined shards have clean sequence/request ordering.

`select-static-moe-experts.py` ranks experts by:

```text
miss count descending
access count descending
smaller expert byte size
layer id
expert id
```

Then it fills a MiB budget.

### Dataset work

The earlier balanced 1,000-prompt trace mixed:

```text
BigCodeBench
MBPP
MATH-500
BIG-Bench Hard
MMLU-Pro
MuSR
```

The coding-assistant trace dataset later targeted the real use case more
closely:

```text
800 Open-SWE-Traces
500 SWE-rebench trajectories
400 OpenCodeInstruct
200 CommitPackFT
50 SWE-bench Verified
50 SWE-Gym
local sanitized Codex user turns
```

The combined files can be very large:

```text
coding-assistant-routing-combined-300.jsonl
coding-assistant-routing-combined-600.jsonl
```

These are generated artifacts and not normal source files.

## Kaggle T4 Trace Workflow

Kaggle was used to run portable router tracing on T4 GPUs without GDS.

Important paths used on Kaggle:

```text
/kaggle/input/datasets/manoharrrr/expert-tracing
/kaggle/input/datasets/manoharrrr/qwen3-6/qwen3.6/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

Important doc:

```text
KAGGLE_T4_ROUTING_TRACE.md
```

Practical Kaggle notes:

- use relative `Path.rglob()` patterns, not absolute patterns;
- verify GPU availability with `nvidia-smi`;
- build may take a long time on Kaggle;
- two T4s can be used with split mode, but a single process may initially fill
  only GPU0 if not configured correctly;
- tracing 50 prompts can take about 50 minutes depending on offload settings;
- larger shards were later run and zipped back to the local workspace.

Downloaded trace bundles live under:

```text
/home/manohar/Desktop/inference/router tracing /
```

## Prefill Bottleneck Findings

The main remaining bottleneck is long-prompt prefill.

Why:

- prefill chooses many diverse experts;
- many experts are first-use misses;
- exact full expert sets rarely repeat;
- static RAM cache helps, but does not eliminate unique-expert traffic;
- dense FlashAttention is a very small share of total wall time for this model;
- full-layer loading wasted too much traffic.

From `KASCADE_PHASE1_PROFILE.md`:

```text
FlashAttention share of GPU time at 8192 tokens: 1.58%
FlashAttention share of wall time at 8192 tokens: 0.062%
MoE stream time at 8192 tokens: 459.12 s
MoE ensure time at 8192 tokens: 713.45 s
prefill VRAM-cache hit rate at 8192 tokens: 23.34%
```

Conclusion: Kascade or attention-only acceleration is not the right target for
this Qwen3.6 MoE setup. Expert movement dominates.

## Failed Or Reverted Architecture Experiments

These are important because they explain why the current architecture is still
on-demand selected-expert streaming.

### Blind next-layer full expert preload

Idea:

```text
while layer N computes, preload all experts of layer N+1
```

Observed issue:

- raw GDS could move a full layer reasonably fast in isolation;
- but full-layer preload loaded many experts never used by the router;
- it polluted the normal cache;
- it evicted useful resident experts;
- it added far more traffic than it saved.

From the old analysis:

```text
extra forced prefetch traffic: about 130 GiB
real demand traffic saved: about 3.6 GiB
prompt speed dropped from about 10.2 t/s to about 9.4-9.5 t/s
```

### Dedicated 3-slot full-layer prefill workspace

Idea:

```text
slot 0 current layer
slot 1 next layer
slot 2 layer after next
```

Observed issue:

- it avoided cache pollution but still loaded full layers at tiny callback
  granularity;
- DirectStorage prefill microbatching was still small;
- workspace mapping by `layer_idx % 3` caused layer reloads across chunks;
- synchronous waits dominated.

Observed examples:

```text
baseline prompt: about 7.9 t/s
workspace prompt: about 0.8-1.1 t/s
prefetch traffic: 55-229 GiB in small tests
```

### Fixed prefill VRAM/RAM layer residency

Idea:

```text
keep some full expert layers fixed in VRAM
keep some full expert layers fixed in RAM
stream only layers outside those sets
```

Observed result:

```text
VRAM 0-1 / RAM 2-17 split: about 10.1 prompt t/s, 7.1 gen t/s
```

It worked mechanically but did not beat the recommended original architecture.
Full layers consumed too many cache slots and hurt decode reuse.

### Hybrid CPU experts

There were commits that tried hybrid CPU expert execution for pinned experts,
then reverted:

```text
ad0048543 Enable hybrid CPU experts with GDS
fbcba53f2 Document hybrid CPU expert scaling
7c916cf53 Track CPU split candidates for pinned experts
532cf3096 Revert "Enable hybrid CPU experts with GDS"
```

Conclusion from conversation and tests: CPU expert compute can improve prompt
shape in some cases but hurts decode or adds complexity. The current source is
back to the original GDS architecture.

## DFlash Research State

DFlash is not active in the current checked-out source.

Reference repo:

```text
/home/manohar/Desktop/inference/dflash_refs/beellama.cpp
```

Pasted drafter:

```text
/home/manohar/Desktop/inference/qwen3.6/dflash-draft-3.6-q8_0.gguf
```

The drafter is valid, but current `moe.cpp` cannot load `dflash-draft` without
manual BeeLlama-style architecture support.

Historical DFlash experiment notes were saved in Git history:

```text
3cd17bb05 Document DFlash drafter experiment
01e010a71 Add DFlash target hidden capture stage
```

The practical conclusion from `DFLASH_DRAFTER_EXPERIMENT.md` was:

- DFlash is real and useful in BeeLlama;
- standalone BeeLlama does not have our GDS MoE streaming path;
- blindly merging BeeLlama is too risky;
- DFlash would be a manual staged port;
- DFlash mostly targets decode speed, not large system-prompt prefill.

Stable production path remains:

```text
GDS expert streaming + VRAM cache + static RAM cache + MTP
```

## Context, KV Cache, And Agent Use

The project also explored using the model as a coding assistant through:

```text
opencode
PI agent
mimo code
llama-server OpenAI-compatible endpoint
```

Important lesson:

Large coding-agent system prompts make every request look like a long prefill
unless prompt/KV reuse works.

For fixed system prompts, prefix/KV caching is conceptually the right solution:

```text
process system prompt once -> keep/reuse KV state -> avoid repeating prefill
```

But server logs showed cases where prompt cache reuse did not happen:

```text
forcing full prompt re-processing due to lack of cache data
```

Root cause found (2026-07-03/04 opencode session). The single warning above
covers three distinct situations; only one was a real bug:

```text
1. First turn of a session      -> unavoidable full prefill of the ~8k prompt
2. Autonomous agent-loop steps   -> reuse the live-slot cache correctly (append-only)
3. A NEW user turn after tool use -> was reprocessing the entire tail (the bug)
```

Case 3 was a chat-template bug, not opencode or tool-call re-serialization. The
template stripped the empty `<think>\n\n</think>` block from every assistant turn
before the last user message (`loop.index0 > ns.last_query_index`). During
generation each assistant turn is cached WITH the block; appending a new user
message re-rendered prior turns WITHOUT it, so the prompt diverged from the KV
cache at the first assistant token (proven byte-for-byte with `-v` verbose prompt
dumps). Reasoning is off, so the block is always empty and only served to break
the cache.

Fix, now in `running_server`:

```text
--chat-template-kwargs '{"preserve_thinking": true}'
```

This keeps the block on every turn so replay matches the cache. Verified A/B on
the live server: follow-up 9.8s with 3 reprocess passes -> 1.0s with 0 passes;
scales with session size (real ~13.6k session: ~4 min -> seconds).

Separate remaining cost on long sessions: opencode title and
summarize/compaction aux tasks run on the same single slot with a different
system prompt (they match only ~3 tokens, i.e. diverge at token 3, NOT the
~5778 first-assistant boundary of the fixed bug). Each reprocesses ~12.8k tokens
(~8 min) and evicts the build cache. Open serving lever: point opencode
`small_model` at a separate endpoint/port so those tasks do not run on and evict
the main slot.

Also verified this session: the chat-template fix and the fused TQ3 attention fix
work together in a live opencode build -- fast long-context decode AND no
per-turn reprocessing. See `FLASH_ATTENTION_DEQUANT_ISSUE.md`.

For local `llama-cli` tests, always use:

```text
-st
```

The user observed IDE instability without single-turn mode.

## Current Performance Picture

The exact numbers vary with prompt, top-k, MTP depth, RAM pressure, thermal
state, and whether the static cache is warm/loaded.

The most reliable high-level picture is:

- decode is usable; after the fused TQ3 attention fix, long-context K=4 decode is
  about 13-15 t/s sustained (was about 6.6 t/s), tapering with context depth;
- with quantized KV (`tq3_0`) + flash attention, the KV dequant/copy path was the
  dominant long-context decode cost (~57% of the decode graph), NOT expert
  streaming and NOT attention math -- the fused `TQ3_0/TQ3_0` kernel removed it;
- prefill is still the major bottleneck for large coding-agent prompts;
- static 8 GiB pinned cache improves prompt speed significantly;
- K=4 can push prompt and decode speed higher by selecting half as many
  experts;
- GDS streaming speed in real runs is far below raw NVMe sequential speed
  because the workload is sliced, synchronized, and cache-dependent;
- cache hit rate alone is not enough; miss-bearing calls and transfer wall time
  matter more;
- for agent serving, prompt/KV reuse is now working across turns (see the KV
  cache section) after the `preserve_thinking` chat-template fix; the residual
  long-session cost is aux-task (title/compaction) reprocessing on the single
  slot, which is a serving-config lever, not a kernel problem.

## Current Recommended Modes

### Conservative quality mode

```text
top-k: model default, K=8
VRAM cache: 2816 MiB
pinned RAM: 8192 MiB static
MTP: draft 2 or 3
reasoning: off for normal coding, on for reasoning tests
```

### Balanced experimental mode

```text
top-k: K=6
VRAM cache: 2816 MiB
pinned RAM: 8192 MiB static
MTP: draft 3
```

K=6 looked safer than K=5 in small reasoning tests while being much faster than
K=8.

### Fast experimental mode

```text
LLAMA_MOE_EXPERT_USED_OVERRIDE=4
```

K=4 is not proven zero-loss, but it is much stronger than an unsafe toy setting.
It deserves further coding-agent-specific tests.

## Documentation File Map

Important docs in `moe.cpp`:

```text
MOE_CPP_GDS_HANDOFF_REPORT.md          current high-level GDS handoff
MOE_EXPERT_CACHE_IMPLEMENTATION_STATUS.md staged cache architecture notes
MOE_RAM_CACHE_SIZE_SWEEP.md            pinned RAM cache size sweep
MOE_EXPERT_CACHE_POLICY_RESEARCH.md    eviction/cache policy research
MOE_ROUTING_TRACE_SIMULATOR.md         offline trace simulator explanation
MOE_TRACE_DATASET_PLAN.md              balanced 1000 prompt trace plan
CODING_ASSISTANT_TRACE_DATASET.md      coding-agent trace corpus plan
KASCADE_PHASE1_PROFILE.md              attention-vs-expert prefill profile
RECOVERY_PROCESS.md                    recovery commits and reverted experiments
MTP_DIRECTSTORAGE_BENCHMARK.md         MTP speed and stream comparison
MOE_TOPK4_QUALITY_SWEEP.md             K=4 quality notes
MOE_TOPK5_QUALITY_SWEEP.md             K=5 quality notes
MOE_TOPK6_QUALITY_SWEEP.md             K=6 quality notes
MOE_REASONING_ON_TOPK_SWEEP.md         reasoning-on top-k comparison
MOE_K4_VS_K8_BROAD_EVAL.md             broad hard K=4/K=8 comparison
MOE_REAL_HARD_TOPK_EVAL.md             hard eval setup
MILESTONE1_DIRECTSTORAGE_QWEN.md       initial Qwen DirectStorage milestone
FLASH_ATTENTION_DEQUANT_ISSUE.md       fused TQ3_0 flash-attention fix (KV dequant path); long-context decode ~2x
SESSION_HANDOFF_2026-07-02_opencode-followup-reprocess-fix.md  preserve_thinking chat-template cache fix + opencode tool trim
SESSION_HANDOFF_2026-07-02.md          earlier tq3_0 KV port / context-headroom handoff
```

Backups of the original files edited by the fused TQ3 attention fix are under
`backups/flash-attn-tq3-original/` (fattn.cu, fattn-common.cuh, fattn-vec.cuh,
generate_cu_files.py, llama-graph.cpp).

Important older docs outside `moe.cpp`:

```text
markdown files/directstorage_moe_implementation_report.md
markdown files/PREFILL_SPEED_RESEARCH_NOTES.md
markdown files/FULL_LAYER_PREFILL_FAILURE_ANALYSIS.md
llm_upper/PROJECT_STATE.md
llm_upper/PATH_A_IMPLEMENTATION.md
```

## Recovery Points

From `RECOVERY_PROCESS.md`:

Original protected GDS/MoE architecture:

```text
e706cf5dc Protect GDS MoE streaming work
```

Later known current original-architecture recovery point before Kascade/top-k:

```text
532cf3096 Revert "Enable hybrid CPU experts with GDS"
```

Do not run `git clean -fd` unless untracked experiment files are intentionally
discarded.

## Known Sharp Edges

- `--dstorage-moe` currently supports only Qwen MoE in this branch.
- The DirectStorage/GDS name is historical; Linux uses cuFile underneath.
- `LLAMA_MOE_ROUTING_TRACE` is ignored with `--dstorage-moe`; use
  `LLAMA_DSTORAGE_ROUTING_TRACE` for GDS runs.
- `-st` should be used for local CLI tests to avoid IDE/session instability.
- Dynamic pinned cache is not the same as static pinned cache and can be slower.
- Static trace-selected experts are workload-dependent.
- Prompt speed comparisons are noisy and need matched prompts/seeds/settings.
- GPQA parser results before the fixed harness are provisional.
- Full-layer prefill loading should not be retried without redesigning prefill
  batching and workspace lifetime.
- DFlash references are not active runtime support in the current source.

## What To Do Next

If the next goal is more speed:

1. Continue top-k testing with coding-agent tasks, especially K=4 vs K=6 vs K=8.
2. Use held-out coding-agent prompts, not only trace-training prompts.
3. Measure output quality and formatting, not just pass/fail.
4. Prompt/KV cache reuse for stable system prompts is now working across turns
   (preserve_thinking fix). Remaining agent-serving win: route opencode
   `small_model` (title/summarize/compaction) to a separate endpoint so aux
   tasks stop reprocessing ~12.8k tokens and evicting the main slot on long
   sessions.
5. Investigate why real GDS effective throughput is much lower than raw NVMe
   speed for selected-expert reads.
6. Attention KV-dequant is no longer a bottleneck (fused TQ3 path landed). Only
   revisit attention if profiling changes. Kascade was not useful for the
   earlier bottleneck.
7. Broaden fused TQ3 coverage if needed: it is currently scoped to
   `TQ3_0/TQ3_0` on the vector flash-attention shapes D=64/128/256; mixed
   TQ3/F16 stays on the old fallback.

If the next goal is reliability:

1. Keep K=8 or K=6 for coding-agent production.
2. Keep 8 GiB static pinned RAM cache unless RAM pressure requires
   prefill-only release.
3. Prefer short, matched `-st` tests before long server runs.
4. Keep benchmark outputs separate from committed source.
5. Update this file after each architecture-changing experiment.

## Mental Model

The project is not "loading the model from SSD every token."

It is closer to a virtual expert tensor system:

```text
GGUF expert tensors stay on disk
one-expert placeholders satisfy model loading
runtime router reveals exact needed experts
GDS/RAM fills a limited VRAM expert cache
graph metadata is patched just in time
MUL_MAT_ID runs unchanged
metadata is restored after the layer
```

This is why the architecture is powerful: it preserves exact router behavior and
does not require changing the MoE math kernels.

This is also why prefill is hard: if the router keeps asking for many unique
experts, no cache can avoid first-use misses after the router decision is known.
The only complete fixes for long prefill are to reduce routed expert work,
predict useful experts early with high confidence, reuse prompt/KV state, or
change the model/execution shape.
