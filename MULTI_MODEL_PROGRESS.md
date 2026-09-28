# siphon.cpp — Multi-Model Support: Qwen3-Coder-Next Enablement

Status: **A/B RESOLVED — OUR PORT IS NOT THE CAUSE** — Qwen3-Coder-Next loads and
generates via DStorage in BOTH our project and the reference project. The
reference project (`/home/manohar/Desktop/inference/qwen.cpp-git`, commit
11d57c1) reproduces the EXACT SAME deterministic wrong answers at both top-k 4
and top-k 10. Therefore the deterministic multi-digit/factual failure is NOT a
regression in our port — it is inherent to the shared DStorage path or the
model/quant itself. NOTE: several earlier conclusions in this file (cache
pressure, pool split) were DISPROVEN by later tests — see "What we ruled out".

## Target / Goal

`siphon.cpp` is a CUDA/GDS-focused `llama.cpp` fork that runs large MoE models on
a ~6 GB VRAM laptop by **streaming MoE expert weights from SSD on demand**
(DirectStorage / NVIDIA GDS + cuFile), caching hot experts, and using TQ3
compressed FlashAttention.

Repo is hard-tuned for ONE model: **Qwen3.6-35B-A3B** (`qwen35moe` arch, 41
layers, 256 experts, 8 used, has an MTP head). This session is enabling a second
model — **Qwen3-Coder-Next** (`qwen3next` arch, 48 layers, **512 experts**, 10
used, MLA + SSM hybrid, NO MTP head).

Key differences vs the current model:
- arch `qwen3next` (not `qwen35moe`)
- 512 experts (vs 256), 10 used per token (vs 8)
- **No MTP head** → `--spec-type draft-mtp` must NOT be used
- Model file: `/home/manohar/Downloads/Qwen3-Coder-Next-Q4_K_M.gguf` (48.5 GB)
- Hybrid: every 4th layer is full attention, rest are recurrent (SSM/linear attn)

## IMPORTANT — The Reference Project (the "it works" baseline)

**`/home/manohar/Desktop/inference/qwen.cpp-git`** contains a COMMIT that adds
exactly the qwen3next + DStorage support we recreated by hand, and the USER
reported running Qwen3-Coder-Next comfortably there:

```
commit 11d57c1a4bca1dd43c87c704aa966376e8f9e5bb
feat: add support for Qwen3 Coder Next (qwen3next architecture) to GPUDirect Storage MoE streaming
  src/llama-context.cpp    | 2 +-
  src/llama-model.cpp      | 4 ++--
  src/models/qwen3next.cpp | 5 ++++-
```

That commit's exact diff (from `git show 11d57c1`):
1. `src/llama-context.cpp` ~229: trace gate allows `QWEN35MOE || QWEN3NEXT`
   (ours also adds QWEN3MOE — benign superset).
2. `src/llama-model.cpp` ~1308: `--dstorage-moe` gate allows `QWEN35MOE || QWEN3NEXT`
   (ours also adds QWEN3MOE — benign superset).
3. `src/llama-model.cpp` ~2113 (`create_memory`): adds `LLM_ARCH_QWEN3NEXT` to the
   hybrid `filter_attn`/`filter_recr` branch that decides which layers get
   attention-KV memory vs recurrent (SSM) state memory.
4. `src/models/qwen3next.cpp` ~548 `build_layer_ffn`: `build_moe_ffn(...)` now ends
   `nullptr, model.layers[il].ffn_gate_up_exps, nullptr, nullptr, nullptr,
   cparams.dstorage_moe, cparams.moe_routing_trace` — IDENTICAL to ours.

**We were MISSING edit #3** (the `create_memory` is_recr/filter_attn branch did
not include QWEN3NEXT). We applied it on this session's last build:
`src/llama-model.cpp` line ~2116 now reads
`arch == LLM_ARCH_QWEN35 || arch == LLM_ARCH_QWEN35MOE || arch == LLM_ARCH_QWEN3NEXT`.
REBUILT and re-tested — **still produces wrong answers** on the failing prompts,
so this was NOT the (whole) root cause.

The reference project also has unrelated uncommitted changes (`src/llama-ext.h`,
`tools/server/server-context.cpp`) whose relevance to generation correctness is
UNKNOWN — we diffed them but they appear to be GPU-memory/extraction API additions,
not obviously the cause. The reference project text is otherwise at the same
commit as ours. `src/llama-dstorage-slots.cpp` is byte-identical in both projects.

### A/B test result (2026-08-05) — THE decisive finding

We successfully ran the reference project's server (port 8093) on this machine and
reproduced the failure. The earlier "reference keeps dying" was a red herring: the
first foreground run hit CUDA OOM because OUR server (PID 45607, port 8090) was
still holding the GPU. After killing ours, the reference loaded fine.

- **top-k 4 (override 4) on reference**: `3 + 5 =` → nonsense; `What is 27 times
  43?` → rambles about LCM of 12 and 27. Same severe garbage we saw in our project.
  Confirms top-k 4 is bad regardless of which project runs it.
- **top-k 10 (override 0) on reference**: IDENTICAL deterministic failures to ours:
  - `What is 27 times 43?` → "show me how to calculate 27 times 3, 27 times 27, 27 times 2..."
  - `What is 13 times 17?` → computes `13 × 13`
  - `What is 9 times 6?` → computes `9 × 9`
  - `What is the capital of Japan?` → ignores, rambles about "Land of the Rising Sun"
  - `3 + 5 =` → misreads, computes `3 + 3 = 6`
  - `Write a Python function factorial(n)` → CORRECT code

CONCLUSION: our 3-file (now 4-file) port is byte-equivalent in behavior to the
reference. The residual bug is shared across both projects → it lives in the common
code path (llama-dstorage-slots.cpp / dstorage eval callback / qwen3next hybrid
memory) or is intrinsic to Q4_K_M quantization of this model. Code-gen correct →
expert weights + routing are fine; the failure is a prompt/context/state-level issue.

## THE CORE SYMPTOM WE ARE CHASING

With DStorage active, at top-k 4 AND top-k 10(override 0), deterministic prompt
misreading on certain prompts (at temperature 0, identical output across runs):

- `3 + 5 =`  → OK ("8")
- `2 + 2 =`  → OK ("4")
- `factorial(n)` code → CORRECT (full correct Python)
- `reverse_string` / `is_palindrome` code → CORRECT
- `capital of France` → "Paris" (OK)
- haiku → OK
- `What is 27 times 43?` → WRONG (repeats "27 times 3, 27 times 27, ...")
- `What is the capital of Japan?` → WRONG (answers about US capital / "Land of the Rising Sun")
- `What is 13 times 17?` → WRONG
- `What is 9 times 6?` → WRONG

Key insight: **code generation works → DStorage expert WEIGHTS + routing are
correct** (correct code needs correct weights). The failure looks like the model
half-reads / misreads the PROMPT on some inputs — i.e. a prompt/cache/context
issue, NOT expert-corruption. But it is deterministic, so it is NOT random cache
pressure either.

## What we ruled out (disproven hypotheses) — READ BEFORE RE-THEORIZING

- ❌ **Pool split (`LLAMA_DSTORAGE_SPLIT_DOWN_POOLS`)** — DISPROVEN. Tested
  `SPLIT_DOWN_POOLS=0` (single unified pool, all downs at 0.82 MiB stride): still
  garbage on the same prompts. The pool-starving `no available slot` errors DO
  occur with split+topk4, but disabling the split did NOT fix the output.
- ❌ **Cache sizing / slot pressure** — the `no available slot` errors cause
  corruption when present, but the deterministic wrong answers persist even with
  **0** such errors (override 0 config). Not the (only) cause.
- ❌ **Scales (`*_exps_s`)** — GGUF has NO per-expert scale tensors; the `nullptr`s
  in build_moe_ffn are correct.
- ❌ **Prompt-cache / checkpoints** — tested with `--cache-prompt` OFF
  (`CACHE_PROMPT=0` env): still wrong on the same prompts.
- ❌ **Missing `create_memory` filter** — added QWEN3NEXT to the hybrid memory
  filter (from ref commit); rebuilt; STILL wrong.
- ❌ **Cache-prompt KV corruption** — see above; not it.

## What is CONFIRMED

- ✅ **Top-k override matters.** `LLAMA_MOE_EXPERT_USED_OVERRIDE=4` (inherited
  from the Qwen3.6 run script) forces 10→4 experts/token. This model is trained
  for top-k 10. At override 4 the model produces SEVERE garbage even on trivial
  arithmetic (`3+5` → nonsense chain). At override 0 (native 10) trivial tasks
  become correct. **Root cause of the severe/simple-task garbage = the override
  knob.** Both run scripts default it to 4; the default is WRONG for qwen3next.
  → We changed `scripts/run-qwen3-coder-next.sh` default to `0`.
- ✅ **Two distinct failure levels:** (a) top-k 4 → severe garbage (fixed by
  override 0); (b) even at top-k 10, multi-digit/factual prompts still
  deterministically wrong. These are DIFFERENT problems; (b) is still open.
- ✅ DStorage expert streaming produces correct code → weights/routing correct.

## Build / Run

Build (slow, CUDA):
```bash
./scripts/build-gds-cuda.sh
```
Run:
```bash
LLAMA_MOE_EXPERT_USED_OVERRIDE=0 ./scripts/run-qwen3-coder-next.sh
```
Web: `http://127.0.0.1:8090` · API: `http://127.0.0.1:8090/v1`
Model alias: `qwen3-coder-next`

Run-script `CACHE_PROMPT` env added (default 0) to toggle `--cache-prompt` for the
prompt-cache experiment. `LLAMA_DSTORAGE_SPLIT_DOWN_POOLS` env (default 0) added
in this project's run script for the pool-split experiment.

## Progress Checklist

- [x] Inspect Downloads, found model GGUF
- [x] Confirm `qwen3next` arch already implemented
- [x] Relax `--dstorage-moe` arch gate (llama-model.cpp)
- [x] Relax routing-trace arch gate (llama-context.cpp)
- [x] Pass dstorage/trace flags into `build_moe_ffn` for qwen3next
- [x] Verify variable expert sizes handled (SPLIT_DOWN_POOLS, per-layer stride)
- [x] Create config + run script (no MTP)
- [x] Run build — SUCCESS (fixed build_moe_ffn signature: 3 nullptr scale args)
- [x] First launch / smoke test — got code correct, ~7.9-8.8 tok/s
- [x] DISPROVE pool-split hypothesis (SPLIT_DOWN_POOLS=0 still wrong)
- [x] CONFIRM top-k override is root of severe garbage; it was 10→4, changed default to 0
- [x] Apply missing `create_memory` filter (QWEN3NEXT) from reference commit — still wrong
- [x] Rule out scales, prompt-cache, cache-pressure as sole causes
- [x] A/B test against reference project — CONFIRMED: reference reproduces IDENTICAL
      wrong answers at both top-k 4 and top-k 10. OUR PORT IS NOT THE CAUSE. Bug is
      in the shared DStorage path or the model/quant itself.
- [ ] Identify the shared-path / quant root cause of the top-k-10 prompt misreading
- [ ] Generate a fresh static expert suite for 512-expert model
- [ ] Tune cache sizes / verify on 6 GB VRAM
- [ ] Update README to list both models

## Current session actions / state

- RAM-as-L2 (evict→RAM) implemented and measured (2026-08-08) — see caveats for full
  table. Best decode yet: test1 6.1 / test2 6.6 tok/s (baseline 5.2/5.1), ensure_loaded
  −33%. Build: `cmake --build build-gds-cuda -j 16 --target llama`. A/B servers stopped.
- Earlier A/B (2026-08-06): reference project server (port 8093) reproduced our failures
  exactly; no server left running. Our own server (port 8090) is STOPPED.
  Relaunch with the evict→RAM env via `/tmp/opencode/launch-ram.sh`.

## Immediate next steps (for the next agent)

1. **SPEED (now = RAM L2, confirmed working): combine with the correct static suite.**
   `LLAMA_DSTORAGE_EVICT_TO_RAM=1` alone hit 6.1/6.6 tok/s (baseline 5.2/5.1). Next:
   generate a correct Qwen3-Next static expert suite and layer it on top (evict→RAM
   handles warm re-misses; the static suite should add prefetch + first-touch residency).
   Route-trace + select-static experts via
   `scripts/run-moe-routing-trace.py` + `scripts/select-static-moe-experts.py`.
   Also try bumping `MOE_GPU_CACHE_MIB` now that RAM absorbs re-misses cheaply.
2. **Do NOT chase our own port — the A/B settled that.** The reference behaves identically, so refocus on the shared code path or the quant:
   - Inspect how `src/llama-dstorage-slots.cpp` / the dstorage eval callback handle the SSM/recurrent layer states in qwen3next. A stale overlapped expert or wrong state-size in the hybrid (every 4th layer = full attention, rest = recurrent) slice is the most plausible shared-path culprit. Code-gen is fine → expert weights + routing are fine; the failure is at the prompt/context/state level.
   - Compare behavior without DStorage is impossible on this hardware (48.5 GB model vs 15 GB RAM / 6 GB VRAM) — isolate instead by `--cache-ram 0`, empty static suite, `--fit off`.
3. **Test quantization precision** — Q4_K_M arithmetic drift across a 10-expert mixture could explain "confidently wrong" numbers while code stays correct. If a higher-precision checkpoint (Q6_K/Q8_0) is obtainable, run the same prompts on it; correct answers → the fix is a re-quant, not a code change.
4. **Sanity-check the shared dstorage path against upstream vanilla llama.cpp** for the qwen3next recurrent-state handling.

## Known Concerns / Caveats

- **A/B verdict: NOT our port.** Reference project reproduces every failing answer nearly byte-for-byte. Any fix must address the shared code path or the model/quant itself.
- **top-k 6 is the chosen speed/quality default** (measured on reference, 2026-08-05):
  top-k 6 gives IDENTICAL output quality to top-k 10 but ~1.5-2x faster decode
  (~6-7 vs ~4.2 tok/s). k4 is unusable (garbage). k8 == k10 quality, no faster.
  The multi-digit/factual misreads persist at k6/8/10 alike → NOT top-k-related.
  `run-qwen3-coder-next.sh` default changed: `LLAMA_MOE_EXPERT_USED_OVERRIDE` → 6.
- `no available slot` errors appear with split-pools + top-k 4; produce at least transient corruption, but are NOT the deterministic cause.
- Only 15 GB RAM vs 48.5 GB model: the reference (non-DStorage-relevant) full-load path may not fit; DStorage is what makes the model runnable here.
- **DSTORAGE COST PROFILE (2026-08-06, 1155 decode tokens, 221.7s = 192 ms/tok, 5.2 tok/s)**:
  `ensure_loaded` = 151.3s of decode wall (68%) → SSD→GPU expert streaming
  (`ds_loader_stream_to_cuda_batch`) = 120.1s, avg **2.5 ms/miss**; `reserve_or_evict_slots`
  = 19.8s; GPU compute is cheap (`layer_compute_ewma_us=169.5` vs `transfer_ewma_us=3666.3`,
  ~21x gap). Decode hit rate only **52%**; prefetch **0.0%**; static suite pins 4491 experts
  (8133/8192 MiB) but covers barely half of decode's touch set. **VERDICT: decode is I/O-bound
  on expert loading, not compute-bound — the fix is raising hit rate / hiding SSD transfer,
  not bigger compute.**
- **A/B 2026-08-06: DECODE_HOT_CACHE=1 NOT USEFUL.** 256 protected slots raised decode hit
  rate only 52.0→52.9% (+0.9pp, the hot experts were already staying resident: 56097
  protected hits / 4 misses), but the reserved slots shrank the flexible pool →
  `ensure_loaded:total` 151→194s and decode dropped 5.2/5.1→4.2/3.7 tok/s. Keep OFF.
- **A/B 2026-08-06: PREFILL_WORKSPACE=1 CANNOT FIT.** It allocates 2 full expert-pool
  workspaces (gate pool alone 288 MiB, plus up/down) ON TOP of the 2816 MiB expert cache +
  KV → cudaMalloc fails (`failed to allocate prefill workspace ... ffn_gate_exps.weight`).
  Not testable on this 6 GB GPU; would need a smaller `--moe-gpu-cache-mib` (which costs
  decode hit rate). Untested lever that REMAINS: a correct Qwen3-Next static expert suite
  (prefetch hit 0% with the Qwen3.6-tuned one).
- **RAM AS L2 CACHE (evict→RAM) IMPLEMENTED 2026-08-08 — NOW BEATS BASELINE.**
  New `cache_evicted_expert_to_ram(slot_idx)` in `src/llama-dstorage-slots.cpp`: on VRAM
  eviction of a warm expert (`decode_hits >= 2`), it copies the slot VRAM→pinned RAM via
  `ds_loader_cuda_to_host_batch` (D2H) into `pinned_cache_`, instead of discarding it.
  Gated by `LLAMA_DSTORAGE_EVICT_TO_RAM` (default off) + `LLAMA_DSTORAGE_EVICT_TO_RAM_MIN_DECODE_HITS`.
  The SSD-re-read pinned admission is skipped when evict→RAM is active (avoids double-caching
  + SSD bandwidth steal). Two perf bugs fixed en route: (a) D2H request offsets were all 0
  (nullptr-cast bug) → fixed with `dst_offsets`; (b) eviction was an O(n≈5600) LRU scan on
  every insert → added `pinned_lru_index_` (std::set keyed by tick, O(log n)) and a pinned
  buffer freelist to recycle the evicted victim's host buffer instead of re-`cuMemHostAlloc`
  (~475us/op).
  RESULTS (k6, empty static suite, pinned 10240 MiB, 300+800 workload, vs baseline):
  | metric | baseline | evict→RAM (final) |
  |---|---|---|
  | test1 decode | 5.2 tok/s | **6.1** |
  | test2 decode | 5.1 tok/s | **6.6** |
  | SSD reads | 222.8 GiB | 89.1 GiB (−60%) |
  | served from RAM | 53.6 GiB | 321 GiB |
  | transfer_ewma | 3666 us | 1736 us |
  | ensure_loaded:total | 151.3s | **101.5s** (−33%) |
  | reserve_or_evict | 19.8s | 29.4s |
  | ds_loader_stream | 120.1s | 51.4s |
  Evict→RAM copies: 35k, 64 GiB cached, 5.7s copy time. `skipped_already=115k` (repeated
  evictions of already-RAM experts — harmless, just skip-checks). VERDICT: this is now the
  best-measured decode speed on this hardware. Next: combine with a CORRECT Qwen3-Next
  static suite (still 0% prefetch) and/or larger `MOE_GPU_CACHE_MIB`.
- **RAM-L2 TUNING (2026-08-08): MIN_DECODE_HITS=1 + 10752 MiB budget — final config.**
  Static expert suite dropped entirely (0% prefetch hit anyway); pure dynamic RAM-L2 is
  the design. A/B (300+800, 4 runs each, k6): MIN_DECODE_HITS=1 test2 avg **~7.9**
  (7.2/9.3/8.0/6.9) vs MIN_DECODE_HITS=2 **~7.4** (6.6/8.2). test1 noisy ~5.8 both.
  SSD reads dropped to 36-51 GiB/run (baseline 222.8); `ensure_loaded:total` 72.5-93.6s
  (baseline 151.3s). High run-to-run variance (±1.3 tok/s) → treat single runs skeptically.
  VERDICT: keep `LLAMA_DSTORAGE_EVICT_TO_RAM=1`, `EVICT_MIN_DECODE_HITS=1`,
  `PINNED_MIB=10752` (near the 11 GB free-RAM ceiling). RAM→VRAM runs at ~11 GB/s
  (~72% PCIe x8) vs SSD ~1.1 GB/s → the whole win. Remaining levers: correct static
  suite for first-touch residency, or raising `MOE_GPU_CACHE_MIB`.
- **Qwen3.6-35B-A3B EVICT→RAM A/B (2026-08-08): BEST = static suite + evict→RAM.**
  This is the ORIGINAL tuned model (qwen35moe, 256 exp, 8→4 used, MTP spec). Its
  static suite `moe-static-experts-suite-8192.txt` is CORRECTLY tuned for it (unlike
  Coder-Next). Results (300+800 workload):
  | config | test1 | test2 | prefill SSD |
  |---|---|---|---|
  | tuned baseline (suite, no evict) | 18.9 | 20.4 | 79 GB |
  | evict→RAM ONLY (empty suite) | 19.4 | 16.4 | 142 GB |
  | **suite + evict→RAM** | **21.8/21.9** | **21.8/21.9** | **38 GB** |
  Why evict-only hurt: with MTP spec the decode expert-load phase is tiny
  (114-174 decode calls vs 1100 tokens); prefill dominates, and dropping the static
  suite made prefill re-read everything from SSD (142 GB) → 16.4 tok/s. The static
  suite's prefill prefetch is the primary win for THIS model; evict→RAM adds ~+7%
  on top (20.4→21.8) by absorbing decode re-misses.
  VERDICT: for Qwen3.6 keep BOTH the static suite AND evict→RAM. Config:
  `EXPERTS=moe-static-experts-suite-8192.txt`, `LLAMA_DSTORAGE_EVICT_TO_RAM=1`,
  `EVICT_MIN_DECODE_HITS=1`, `PINNED_MIB=10752`.
- **Qwen3.6 WITHOUT MTP DRAFTER (2026-08-08): confirms MTP is the decode accelerator.**
  Removed `--spec-type draft-mtp` (same setup otherwise: static suite + evict→RAM,
  k4, 10752 MiB pinned). Result: **14.7 / 16.6 tok/s** vs 21.8/21.9 WITH MTP
  (−32%). The picture flips to Coder-Next-like: decode calls exploded 174→2945,
  evict→RAM RAM-hit **94%** (50,687 RAM loads vs 47 with MTP), decode RAM served
  86.7 GB vs 5.8 GB SSD. But per-token passes can't be batched (no drafter), so
  even ~11 GB/s H2D can't fully hide load latency → net slower.
  VERDICT: MTP spec head is worth ~32% on Qwen3.6 — keep it ON. The static suite +
  evict→RAM combo is the best config for BOTH models, differing only in whether
  the static suite is used (Qwen3.6: yes — its suite is correctly tuned; Coder-Next:
  no — its suite was mismatched).
- **KAT-Coder-V2.5 (Kwaipilot) RUNS (2026-08-08).** `Kwaipilot_KAT-Coder-V2.5-Dev-Q4_K_M.gguf`
  (21.4 GB). Arch = `qwen35moe` — nearly identical to Qwen3.6 (256 exp, 8 used, 2048
  emb, 40 layers vs 41) BUT **NO MTP head** (0 MTP tensors) → Coder-Next-style
  decode-bound, `--spec-type draft-mtp` must NOT be used. Best config = dynamic
  evict→RAM, empty static suite (Qwen3.6-tuned suite would be mismatched):
  `EVICT=1`, `EVICT_MIN_DECODE_HITS=1`, `PINNED_MIB=10752`, `GPU_MIB=3584`.
  RESULTS (k4): test1 **13.2-16.2** / test2 **16.4-17.6 tok/s** (full 800),
  decode RAM-served 129.9 GB vs 17.9 GB SSD (~88%), 0 slot errors. n_slots=1836.
  Non-expert weights only 1.16 GB, so KV at CTX=65536 is the VRAM ceiling.
  Faster than Coder-Next (7.9) because it's half the size (21 GB vs 48 GB) with
  256 vs 512 experts. Launcher: `/tmp/opencode/launch-kat.sh` (GPU_MIB default
  now 3584).
  **Mixed-quant pool fix (2026-08-09).** KAT is 50/50 mixed-quant: 20 layers
  Q6_K ffn_down (0.82 MiB/expert) + 20 layers Q4_K (0.56 MiB/expert). Two bugs
  found: (1) `ffn_down` pool was sized `n_slots x max_stride`, wasting ~0.26 MiB
  per Q4_K slot (~239 MiB at 3072); (2) with SPLIT_DOWN_POOLS=1 (default), the
  `large_slots` heuristic gave the large-stride group only `(n_slots+7)/8` slots
  (~12.5%), but KAT has 50% big layers -> group-1 starvation -> `no available
  slot` errors. Fixed in `ensure_pools_allocated` (llama-dstorage-slots.cpp,
  ~line 1768): slot counts are now allocated **proportionally to per-group layer
  counts** (KAT: 918/1836 = 50%), with a floor of `max_prefill_workspace +
  speculative_reserve` (the `(n_slots+7)/8` heuristic) so models with few big
  layers keep the old behavior. Qwen3.6 (3 big/38 small) computes floor=256 =
  old heuristic exactly -> unaffected. For KAT this both fixes the starvation
  AND shrinks the pool ~204 MiB at equal GPU_MIB (3584 now fits: pool 3345 vs
  3584 MiB all-max), which is why 3584 is the new default.   VERIFIED: 3584
  full-800 both tests, 0 slot errors, ~88% RAM-served decode.
  **Output quality audit (2026-08-09).** Until now we measured only tok/s, never
  inspected generated text. Audit found KAT output is coherent and correct --
  BUT two launch config errors were producing bad answers:
  1. `LLAMA_MOE_EXPERT_USED_OVERRIDE=4` forced top-k 8->4, degrading quality:
     `print(17*24)` -> 210 (wrong) at k=4, but **408 (correct)** at native k=8.
  2. Raw (non-chat-templated) prompts confused the model on math; with proper
     `<|im_start|>` template `17*24` -> 408, correct. Coding prompts were fine
     in all cases (reverse_string, palindrome, merge, BankAccount all correct;
     only minor model bugs like `list1[i]<=list2[i]` typo).
  IMPORTANT: this means the earlier KAT tok/s numbers (13.6-18.9) were measured
  on a quality-degraded (top-k=4) model. At native top-k=8 the SAME RAM-L2
  config runs 11.3-12.0 tok/s, 0 errors, correct output. `launch-kat.sh` now
  defaults OVERRIDE=8. Top-k=8 also shows the <think> block even with
  --reasoning off (harmless; model-tuned behavior).
  **Context headroom (2026-08-09).** The proportional-split fix left ~227 MiB
  free at CTX=65536 (5547 used / 6141 total). KV is tq3_0-quantized + only 10/40
  layers are full-attention, so it costs only ~6.5 KiB/token (~208 MiB per 32K
  tokens). Bumped default CTX 65536 -> **86016** (+20480 tokens, ~130 MiB KV):
  loads with 97 MiB free, 61-63 MiB free after workload, 0 errors, tok/s
  unchanged (13.6-18.9 / 15.4-18.9). 98304 also fits at load (19 MiB free) but
  leaves only 3 MiB after workload -> too fragile, rejected. Launch: set
  `CTX` env (launch-kat.sh now defaults CTX=86016).
  SPLIT_DOWN_POOLS=0 workaround in launch-kat.sh is REMOVED (split pools work
  correctly). Qwen3.6 RE-TESTED post-change (2026-08-09): suite + evict->RAM at
  GPU_MIB=2816 gives test1 19.3-22.2 / test2 23.0 tok/s (prev best 21.8/21.9),
  0 slot errors, n_slots=1428 -> confirmed unaffected by the proportional split.
- **VRAM expert-cache tuning (tested 2026-08-05, k6, 300-token decode)**: raising
  `MOE_GPU_CACHE_MIB` gives only small gains — 2816→4096 MiB improved warm decode
  ~5.7→~6.8 tok/s (~19%) but hit-rate is bounded by the static pinned suite being
  Qwen3.6-tuned (prefetch hit 0.0%; decode hit ~48-62%). 4096 MiB at full CTX=65536
  fits but leaves only ~350-370 MiB free VRAM (KV + 48-layer compute), which is
  fragile; 3584 MiB at CTX=65536 is the stable compromise (~6.0-6.4 tok/s).
  VERDICT: cache-size increases are NOT the high-value lever — a correct
  Qwen3-Next expert suite / prefetch is where the real headroom is. Config file
  left at 2816 MiB (or raise to 3584 if stable).
- **ngram-mod speculative decoding TESTED (2026-08-05, k6, port 8093): NOT USEFUL**
  for this model. Default thresholds (n_match 24 / n_min 48 / n_max 64): never
  drafted (#gen drafts = 0 in 720 calls). Aggressive (n_match 8 / n_min 16 /
  n_max 48): drafted but hurt — decode dropped 5.5→4.5 tok/s, acceptance only
  181/522 (35%), and the draft-verify decode doubled concurrent expert demand →
  `no available slot` exhaustion → hung/cancelled requests. VERDICT: ngram-mod
  spec is a net loss here; keep `--spec-type` unset for Coder-Next. The real
  speed lever (per A/B + cache findings) is a correct static expert suite, not
  speculation.

## Environment (current)

- Repo: `/home/manohar/Desktop/siphon.cpp`
- Reference repo: `/home/manohar/Desktop/inference/qwen.cpp-git` (commit 11d57c1)
- GPU: NVIDIA GeForce RTX 4050 Laptop, ~6141 MiB VRAM; RAM ~15 GB
- Model file: `/home/manohar/Downloads/Qwen3-Coder-Next-Q4_K_M.gguf`

## Reverting if needed

Code changes now (superset of the earlier 3):
- `src/llama-model.cpp`: arch gate (~1311) allows QWEN3NEXT/MOE; hybrid
  `create_memory` filter (~2116) now includes QWEN3NEXT (the new fix from ref commit).
- `src/llama-context.cpp`: trace gate (~232) allows QWEN3NEXT/MOE.
- `src/models/qwen3next.cpp`: `build_moe_ffn` passes dstorage/trace flags + 3 scale
  nullptrs (matches reference exactly).

New files:
- `configs/qwen3-coder-next.env`
- `scripts/run-qwen3-coder-next.sh` (added `CACHE_PROMPT`, `LLAMA_DSTORAGE_SPLIT_DOWN_POOLS`
  toggles; `OVERRIDE` default → **6**)

RAM-L2 (evict→RAM) additions (2026-08-08) — disable with `LLAMA_DSTORAGE_EVICT_TO_RAM=0`
(any value other than 1 disables; default OFF):
- `src/llama-dstorage-slots.cpp`: `cache_evicted_expert_to_ram()` (~1300), env gates
  (~410), eviction call site (~4166), admission gate off when evict→RAM active (~4878),
  `pin_index_update()` helper, `pinned_lru_index_` + `pinned_buf_freelist_` maintenance
  at all pinned-cache insert/erase/clear sites.
- `src/llama-dstorage-slots.h`: `pinned_lru_index_` (std::set LRU), `pinned_buf_freelist_`,
  `pin_index_update` decl.
- A/B harness: `/tmp/opencode/launch-ram.sh` (env `EVICT=1 PINNED_MIB=10240 GPU_MIB=2816`),
  `/tmp/opencode/run-ab-workload.sh`. Logs: `ab-evict5.log` (final, best), `ab-evict4.log`
  (LRU index only), `ab-evict3.log` (original, O(n) scan).

Kill any server with: `pkill -f 'bin/llama-server'`