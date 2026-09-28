# Multi-User Inference Findings

Findings from converting `siphon.cpp` from a single-slot server to a multi-slot
(multi-user) server on the tested 6 GB VRAM laptop. Includes the per-token
memory math, the config changes made, and every benchmark result measured.

## Test Hardware And Model

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB total, ~5772 MiB usable |
| System RAM | 16 GiB |
| Model | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (22.6 GB, Qwen35Moe arch) |
| Hybrid layers | 41 total: 10 full-attention, 31 SSM/recurrent |
| Active params per token | A3B (~3B), `LLAMA_MOE_EXPERT_USED_OVERRIDE=4` of 256 experts |

## Model Layout (why the KV cache is small)

`full_attention_interval = 4` → only layers `i = 3,7,11,...,39` (10 layers) are
full attention. The other 31 layers are SSM/recurrent and cache a small,
**context-independent** recurrent state instead of a KV cache.

Key GGUF metadata used for the math:

| Field | Value |
| --- | ---: |
| block_count | 41 |
| full_attention_interval | 4 |
| attention.head_count | 16 |
| attention.head_count_kv | 2 |
| attention.key_length / value_length | 256 / 256 |
| embedding_length | 2048 |
| expert_count / expert_used_count | 256 / 8 |
| ssm.state_size / inner_size | 128 / 4096 |
| nextn_predict_layers (MTP) | 1 |

## Per-Token Memory Cost

### Attention KV cache (context-length, TQ3-compressed)

- `tq3_0` block = 14 bytes per 32 elements = **0.4375 B/elem** (3.5-bit)
- KV dims per full-attention layer = `head_count_kv(2) × (key 256 + value 256)` = 1024 elems
- Full-attention layers = 10
- **Per token = 10 × 1024 × 0.4375 = ~4.38 KiB/token**

| Context per slot | Attention KV (TQ3) |
| ---: | ---: |
| 8K | ~36 MiB |
| 16K | ~72 MiB |
| 32K | ~144 MiB |
| 64K | ~287 MiB |

### Recurrent (RS) state cache (context-independent, per slot)

`llama_memory_recurrent` allocates per SSM layer (31 layers), FP32:

```
per layer = (n_embd_r + n_embd_s) × n_seq_max × (1 + n_rs_seq) × 4 bytes
```

Measured: **~251 MiB per slot**, ~502 MiB at 2 slots. This is the **dominant
per-slot cost** — larger than the entire attention KV cache. It does not
amortize across users (each slot keeps its own SSM state).

### Shared vs per-slot budget

| Resource | Type | Cost |
| --- | --- | --- |
| Dense weights | shared (all slots) | ~2.4 GB |
| GPU expert cache | shared | `MOE_GPU_CACHE_MIB` (2816 or 2048 MiB) |
| Pinned expert cache (CPU) | shared | `MOE_PINNED_CACHE_MIB` (4096 MiB) |
| Attention KV | **per slot** | ~4.38 KiB/token |
| Recurrent SSM state | **per slot** | ~251 MiB fixed |
| MTP draft context | per server | ~69 MiB |

### GPU memory budget measured

| Config | GPU used | Free |
| --- | ---: | ---: |
| 1 slot, 64K ctx, expert cache 2816 | 5738 MiB | 35 MiB |
| 2 slots × 16K, expert cache 2816 | 5617 MiB | 157 MiB |
| 2 slots × 16K, expert cache 2048 | 4907 MiB | 867 MiB |

## Config Changes Made

`scripts/run-qwen36-moe.sh`:

- Added `N_PARALLEL` (default 1) → passed as `--parallel`
- Added `SPEC_TYPE` (default `draft-mtp`) → passed as `--spec-type`
- Continuous batching auto-enabled when `N_PARALLEL > 1`, kept
  `--no-cont-batching` for the tested single-slot path (`CONT_BATCH` can force
  either way)

`configs/qwen3.6-moe-tq3.env`:

- Documented `N_PARALLEL` and the `CTX / N_PARALLEL` = per-slot context rule

Multi-slot launches used (validated):

```bash
N_PARALLEL=2 CTX=32768 MOE_GPU_CACHE_MIB=2816 ./scripts/run-qwen36-moe.sh
N_PARALLEL=3 CTX=49152 MOE_GPU_CACHE_MIB=2048 ./scripts/run-qwen36-moe.sh
```

## Benchmark Results

Single-user decode baseline on this server: **~15–17 tok/s** (16.7 measured).

### Two users, different prompts (concurrent)

| Expert cache | Per user | Aggregate |
| ---: | ---: | ---: |
| 2048 MiB | ~6.8 tok/s | ~13.6 tok/s |
| 2816 MiB (baseline) | ~7.6–7.8 tok/s | ~15.4 tok/s |
| **2816 MiB (with Dynamic Draft Depth)** | **~8.8 tok/s** | **~17.6 tok/s** |

`n_busy_slots_per_decode = 1.75` confirms continuous batching interleaves both
slots in a single `decode()`.

*Dynamic Draft Depth Optimization*: Scaling `n_draft_max` dynamically based on the number of active generating slots (`n_draft_max = std::max(1, n_draft_max / n_gen_active)`) reduced total speculative batch inflation during concurrent generation, boosting per-user speed from **7.6-7.8 tok/s → 8.8 tok/s** (a **+14.5% improvement** for different prompts).

### Two users, same prompt (concurrent, expert cache 2816)

| Test | Result |
| --- | ---: |
| Single user baseline | 16.7 tok/s |
| 2 users, same prompt | **9.0 tok/s each**, aggregate 18.0 |

Same-prompt raised aggregate from ~15.4 → 18.0 (expert weights shared/amortized)
but per-user still halved — the per-token GPU compute (MoE GEMMs + attention +
31 SSM state updates) doubles with a second user and is not amortizable.

### Three users (concurrent, expert cache 2048)

GPU budget at 3 slots × 16K: 5291 MiB used / 483 free.

| Test | Per user | Aggregate |
| --- | ---: | ---: |
| Single user baseline | 15.2 tok/s | ~15 |
| 3 users, different prompts | 6.2 / 6.3 / 6.5 tok/s | ~19 |
| 3 users, same prompt | 6.0 / 6.0 / 6.5 tok/s | ~18.5 |

`n_busy_slots_per_decode = 2.0`.

Key result: **aggregate plateaus at ~19 tok/s** — the GPU's compute-saturation
point. Aggregate rose 15 → 19 as users went 1 → 2 → 3, then stopped rising, and
same-vs-different prompts no longer changed the result. At 3 users the card is
fully compute-bound; further users only divide the same pipe.

### Three users with full expert cache (concurrent, expert cache 2816)

3 slots fit with the full 2816 MiB cache by reducing per-slot context to 4K
(CTX=12288). GPU budget: 5729 MiB used / 45 free. MTP context shrank with
context (28 MiB at 4K vs 69 MiB at 16K).

| Test | Per user | Aggregate |
| --- | ---: | ---: |
| Single user baseline | 16.0–17.8 tok/s | ~17 |
| 3 users, different prompts | 6.6 / 6.7 / 6.8 tok/s | ~20 |

Context ceiling at 3 slots + 2816 cache: 6K per slot loads (5767/5772 used) but
**OOMs during actual decode** (compute workspace fails with 7 MiB free). 4K per
slot is the stable maximum.

**Cache hit rate matters more than context size.** The full 2816 cache with 4K
per-slot context beat the 2048 cache with 16K per-slot context: ~6.7 vs ~6.3
tok/s per user, and aggregate ~20 vs 19. Aggregate also keeps climbing with
concurrency (16 → 21), confirming weight-read amortization over the batch.

### Four users (concurrent)

4 slots fit with expert cache 2304 MiB (CTX=32768 → 8K per slot). GPU budget:
5651 MiB used / 123 free. Cache cannot stay at 2816 at 4 slots — the per-slot
recurrent SSM state (~251 MiB) is context-independent and leaves no VRAM room.

| Test | Per user | Aggregate |
| --- | ---: | ---: |
| 4 users, different prompts | 3.8 / 3.8 / 3.8 / 3.9 tok/s | ~15.3 |

`n_busy_slots_per_decode = 3.77` — all 4 slots batch into the same decode.

Notable: **aggregate throughput peaks around 3 users and drops at 4** (20 → 15.3).
Likely the MTP speculative-draft overhead that doesn't scale with interleaved
slots; every additional slot adds draft compute whose acceptance rate falls as
more sequences share each `decode()`.

## Key Conclusions

1. **The GPU's decode throughput is a single shared pipe.** With one RTX 4050,
   total aggregate decode saturates at ~19 tok/s under multi-slot load; users
   divide it. There is no config that gives every user the single-user rate at
   once on one card. Per-user speed falls as users are added: 15 → 7.7 → 6.3
   tok/s at 1 → 2 → 3 active users.

2. **The recurrent SSM state (~251 MiB/slot) is the real per-slot memory cost**,
   not the TQ3 attention KV (~4.4 KiB/token). It also forces the practical slot
   ceiling (~2 slots with the full 2816 MiB expert cache) on this GPU.

3. **Expert streaming is NOT the main multi-user slowdown.** Same-prompt
   (identical experts, all cached) only lifted aggregate ~15.4 → 18.0. The
   dominant cost is per-token GPU compute, which scales ~linearly with active
   users.

4. **`SPEC_TYPE=none` segfaults** (core dump on concurrent decode) — the MTP
   speculative context is load-bearing in this fork; MTP must stay enabled.

5. **How real providers serve millions of users at high speed** (same physics,
   more hardware): thousands of GPUs with model sharding / expert parallelism,
   continuous batching at batch ~1000+ (weight reads amortized across the whole
   batch), PagedAttention-style KV management, prefill/decode disaggregation,
   and enough replicas that each user's request hits a lightly-loaded GPU.
