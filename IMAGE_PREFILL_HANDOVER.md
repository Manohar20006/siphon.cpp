# Image Prefill Project — Handover

This is a handover for the next agent/session. It summarizes everything done on the **image-prefill** idea: using the vision encoder to speed up large-token prefills.

---

## 1. Objective

*   Normal text prefill: each token is processed sequentially with MoE expert streaming. Even after optimizations (cuFile, VRAM/RAM cache, top-K=4), prefill is ~30 tokens/sec. Small prompts (100-200 tokens) are fine. Large prompts (10k-50k tokens) take minutes (e.g., 9.5k text tokens = 263s).
*   **Idea:** For large prompts, render the bulk text into a dense PNG (white background, black monospace tokens) and feed it as an **image** through the vision encoder. Image token cost is fixed by pixels, not characters — one page is ~1-2k image tokens no matter how much text is crammed in. So a large text prompt becomes far fewer image tokens → faster prefill.
*   Gist is preserved, exact strings are lossy at high density. The practical design is **hybrid**: bulk as image + exact strings (hashes, IDs) as small text factsheet.

---

## 2. What Was Done

### 2.1 Vision Encoder Added to qwen3.6

*   Model: `Qwen3.6-35B-A3B` (GGUF `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`) on `http://127.0.0.1:8080`
*   mmproj: `/home/manohar/Downloads/mmproj-Q8_0-fix.gguf` (original saved as `mmproj-Q8_0.orig.gguf`, 575 MB Q8_0)
*   **Fix 1 — Q8_0 pos_embd crash:** Original mmproj stored `v.position_embd.weight` as Q8_0 (type 8). Warmup image 1472×1472 requires interpolating 48×48→92×92 position embeddings, but `ggml_interpolate` asserts F32 → abort. Fixed by dequantizing that one tensor to F32 (10.6 MB) via `gguf` Python rebuild. See `/tmp/opencode/fix_mmproj.py` and verification.
*   **Fix 2 — Checkpoint/radix crash with vision:** `server_tokens::get_tokens()` asserts `!has_mtmd`. `create_checkpoint()` at `tools/server/server-context.cpp:2096` inserted vision prompts into radix cache via `get_tokens()` → abort at EOG. Fixed by guarding: `if (radix_cache && !slot.prompt.tokens.has_mtmd)` (mirrors restore path at 2908). Rebuilt `build-gds-cuda` (`llama-server`).
*   Service: `siphon-qwen3.6.service` (`PORT=8080`, `MMPROJ=/home/manohar/Downloads/mmproj-Q8_0-fix.gguf`). Conflicts with `siphon-kat-coder.service` on GPU — only one can run (6 GB VRAM). Use `systemctl --user stop siphon-kat-coder` before starting qwen3.6 with vision. Web UI: `http://127.0.0.1:8080/`

### 2.2 Benchmark Harness Built

*   Location: `tools/vision-bench/` (persistent, not /tmp)
    *   `vision_bench.py` — main harness: renders text → PNG pages at 1400×720 (pxpipe-style cap), queries `qwen3.6` with `temperature=0`, scores exact match + Levenshtein distance. Supports `--sizes`, `--configs "font=18,line=1.15,gray"`, `--reps`, `--cold`, `--fresh`, `--variants`, multi-page. Temperature pinned to 0 for determinism.
    *   `vision_bench_types.py` — extended harness for failure-pattern across content types (json, markdown, code, xml, csv, messy_log) using real workspace files + synthetic, with `snippet()` helper, multi-page support.
    *   `corpus/` — generated docs (`doc_8.json`, `doc_16_v0.json`, etc.)
    *   `results/bench-*.json` — saved runs
*   Rendering: PIL, `DejaVuSansMono.ttf` (and Bold), `width=1400, max_h=720`, `cols` computed from `W` width, `line_h = font_size * line_mult`. Modes: `gray` (antialiased, default) vs `bw` (thresholded hard black/white). `invert` (black bg/white text) tested and rejected. Highlight tested (yellow `#fff59d` behind probed lines).

### 2.3 Tests Run

*   **Feasibility:** Rendered JSON configs to PNG at font 18/11/8, asked model to extract exact values (checksums, ports) + count/gist.
*   **Density sweep:** font 18,14,11,8 × line 1.15/1.4 × gray/bw, with Levenshtein scoring.
*   **Color test:** white bg/black vs black bg/white (inverted) vs gray vs bw, normal vs bold vs highlight/blob.
*   **Large-token comparison:** `n=8,16,32,64` synthetic server docs (656 to 5302 chars), text vs image_f14/f18, `max_tokens=600` then `max_tokens=1` for pure prefill, cold vs cached.
*   **Failure-pattern suite:** 6 content types (json, md `siphon-deep-dive.md`, code `llama-sampler.cpp` then synthetic 3-func snippet, xml, csv, messy logs), single-page then multi-page, fixed code ground truth.
*   **Hybrid test:** Pure image vs hybrid (image + factsheet text with exact checksums as text) on `doc_16_v0` at font 11.
*   **Samples saved:** `~/Desktop/vision-samples/` (also `/tmp/opencode/*.png` — but /tmp is wiped on reboot). Real code samples: `real_code_f14.png`, `real_code_f18.png` (first 60 lines of `src/llama-sampler.cpp`).

---

## 3. Key Findings

### 3.1 Speed

*   Large block (16,931 chars, 48 servers): **text 263.9s (9,563 toks) vs image_font8 15.8s (1,064 toks) — ~17× faster prefill, ~9× fewer tokens** (with `max_tokens=1` pure prefill, 720px page cap).
*   Large-token sweep (cold prefill, `max_tokens=600`):
    *   n=8 (656 chars): text 24.5s/438t vs image_f14 18.7s/303t
    *   n=64 (5302 chars, 2 pages): text 112s/3026t vs image_f14 56.6s/1585t (~2× faster, ~48% fewer tokens)
*   Font size ↓ → tokens ↓ → faster, but accuracy ↓ (see 3.2).

### 3.2 Accuracy (exact-match on checksums/IDs)

*   **Readable density (font 18):** 100% on single-doc probe with `temperature=0` (7/7), ~71% avg across 3 docs (5/7) — errors are 1-2 char substitutions (`0`→`a`, `0395`→`b395`). `lev` ~0-7.
*   **Font 14 bw, line 1.4:** ~71-86% (lev 2-4) — best `lev_avg` sweet spot.
*   **Font 11:** ~43-57% (lev 15-79 gray, 20 bw). Font 8: 14% and invents sequential hex `a1b2c3...` (silent confabulation, lev 170+).
*   **Color:** `bw` (hard threshold) beats `gray` (antialiased) at same cost: font 11 lev 15 (bw) vs 79 (gray). Inverted (black bg) is catastrophic (returned deploy ID instead of checksum). Bold/highlight at font 11 did not help (all ~1-2 char errors); at font 14 bold was worse (lev 5 vs 2).
*   **Gist vs exact:** `summary`/`count` gist was almost always OK, even at n=64 with 2 pages. Exact 32-char hex checksums are the hard case. Later positions in long docs (s32/s63 at n=64) hallucinate completely (lev 27-30) — lost-in-middle on images.
*   **Content-type pattern (multi-page, font 14/18):**
    *   Perfect: `xml`, `messy_log` (2/2 at all modes), `json` (3/3 after multi-page fix), `code` synthetic (2/2 after fixing ground truth)
    *   Off by 1: `markdown` (h2 3 vs 4 — one heading on page break missed), `csv` (row off by 1)

### 3.3 The Trap (User’s Insight)

> If you make font bigger for accuracy, image tokens ≈ text tokens → no speed win.

*   Font 18 at n=16: 2948 image toks vs ~3000 text toks — wash on small docs.
*   Win only appears at **large** docs: 50k chars as text ~12.5k toks vs as image font 18 (~3 pages) ~3k toks → still 4× win even with big font.
*   Therefore **pure image cannot be both max-fast and 100% exact at once**. The fix is **hybrid**.

### 3.4 Hybrid

*   Pure image font 11 on `doc_16_v0` (16 servers): 1/7 (all 3 checksums wrong)
*   Hybrid (same image + factsheet text with the 3 probed checksums as text): **4/7** — all 3 checksums now OK (+ count). The factsheet lets the model copy exact values as text; image handles gist/bulk. Fact added ~200 toks.

---

## 4. Current State

*   Server `qwen3.6` with vision is **healthy** on 8080 (KAT stopped). To restart: `systemctl --user stop siphon-kat-coder; systemctl --user start siphon-qwen3.6`
*   Harnesses work and are deterministic (`temperature=0`). Results in `tools/vision-bench/results/`. Samples in `~/Desktop/vision-samples/` (persistent) and `/tmp/opencode/*.png` (ephemeral).
*   No threshold router built yet — the next step the user requested but deferred until accuracy is understood.

---

## 5. Next Steps (As Discussed)

1.  Finish mapping failure pattern across more content types/sizes with multi-page (done for 6 types, code fixed).
2.  Decide on hybrid rendering pipeline: bulk → image at readable density (font 14-18 white bg black normal), exact strings → factsheet text.
3.  Build **threshold router**: if prompt tokens ≤200 → text only (exact, fast enough); if >200 → hybrid image+factsheet (where image wins). Count via tokenizer or chars/4.
4.  Benchmark the threshold router (hybrid vs pure text) on large token sets for time saved vs context loss (gist preserved, exact via factsheet).

---

## 6. Relevant Files & Commands

*   `tools/vision-bench/vision_bench.py` — run: `python3 vision_bench.py --sizes 16 --variants 3 --cold --configs "font=18,line=1.15,gray" "font=14,line=1.4,bw"`
*   `tools/vision-bench/vision_bench_types.py` — run: `python3 -u vision_bench_types.py`
*   `tools/vision-bench/corpus/` — `doc_16_v0.json` etc.; `results/bench-*.json`
*   `~/Desktop/vision-samples/` — real_code_f*.png, types_*.png, hltest_*.png, large_*.png
*   `siphon-deep-dive.md` + `~/Desktop/siphon-deep-dive.{md,pdf}` — project doc
*   `~/Downloads/mmproj-Q8_0-fix.gguf` (active), `mmproj-Q8_0.orig.gguf` (backup)
*   `src/`, `gguf-py/`, `scripts/run-qwen36-moe.sh` (added `MMPROJ` env), `tools/server/server-context.cpp:2096` (radix guard fix)
*   `~/Desktop/pxpipe/` — cloned pxpipe repo (reference for rendering ideas, profitability gate, 5×8 Spleen atlas)
