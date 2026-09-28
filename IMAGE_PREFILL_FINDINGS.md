# Image-Prefill Findings (measured, 2026-08-25)

> All numbers below were measured directly against the running `qwen3.6` vision
> server this session. They replace the prior `IMAGE_PREFILL_HANDOVER.md`, whose
> claims were treated as unverified.

## Setup
- Model: `Qwen3.6-35B-A3B` GGUF on `http://127.0.0.1:8080`
- Vision encoder: `/home/manohar/Downloads/mmproj-Q8_0-fix.gguf`
- Test corpus: `siphon-deep-dive.md` (21,103 chars of natural-language prose)
- `compact` rendering = replace `\n`→`\n` and `\t`→`\t`, render as one continuous
  flow (no blank lines / indentation) to maximize chars per page.
- Prefill time measured with `max_tokens=1` (so reported time ≈ pure prefill).
- Accuracy = char-error-rate (CER) vs ground truth on a 1,500-char excerpt,
  using verbatim transcription with the literal `\n` marker instruction.
- Text baseline for 20k chars ≈ **5,064–5,068 tokens**; cold prefill ≈ **119s**,
  cached (radix) prefill ≈ **10s**.

## 1. Token cost is density-dependent (image token cost is per pixel, fixed)
| Input | Image tokens | vs text | Image prefill time | vs text |
|---|---|---|---|---|
| Dense long-line (16k) | 1,931 (19%) | ✅ | 27s | ~6× faster (cold 157s) |
| Sparse code (8k) | 4,314 (190%) | ❌ | 128s | ~2× slower (cold 64s) |
| Prose, normal render (20k) | 7,880 (155%) | ❌ | 383s | slower |

Image tokens are ~fixed per page (~880 for 1400×720) regardless of how much text
is on the page, so chars-per-page is everything. Short-line content (code, prose)
leaves pages mostly empty → image loses.

## 2. Compact encoding recovers the token win for prose
Compact rendering collapsed the 20k doc from **5 pages → 1 page** (21,519 chars).

| Config | Pages | Tokens | % of text | Prefill time | CER |
|---|---|---|---|---|---|
| compact f12 (1400×720) | 3 | 2,234 | **44%** | ~58s | 20% (flaky: 20–308%) |
| compact f14 (1400×720) | 4 | 3,115 | 61% | 88s | **581%** (hallucinated) |
| compact f16 (1400×720) | 4 | 3,817 | 75% | **53s** | **3%** |
| compact f18 (1400×720) | — | ~5,000 | ~100% | — | 3% |

**Accuracy has a hard resolution threshold ~15–16px.** Below it the encoder often
fully hallucinates (f12 ranged 20%→308% CER across runs; f14 hit 581%). At
**font 16 it is reliable (3% CER)**.

## 3. Larger pages do NOT help
At font 16, varying page height:

| page_h | Pages | Tokens | % text | Time | CER |
|---|---|---|---|---|---|
| 720 | 4 | 3,817 | 75% | **53s** | 3% |
| 1400 | 2 | 3,814 | 75% | 111s | 3% |
| 2000 | 2 | 3,770 | 74% | 107s | 3% |

Tokens are flat (~75%) because total token cost is set by font + total text, not
page splitting. Smaller pages are *faster* (less encoder time per image). CER
stays 3% throughout.

## 4. Accuracy cause: resolution limit, not a broken encoder
Same Qwen encoder on the **same doc**:
- Normal prose, font 12 (real line breaks): **2% CER, 80% line-exact** → encoder reads fine.
- Compact dense flow, font 12: 20%+ CER (unreliable).
- Compact dense flow, font 18: **3% CER** → encoder reads fine.

Conclusion: the accuracy loss is a **vision-OCR resolution/density limit**, not a
defective encoder. Any VLM would likely face the same at low pixels-per-char.

## 5. Cross-model check was not possible
`/home/manohar/Desktop/inference/gemma4/gemma-4-12b-it-qat-q4_0.gguf` has **no
vision tensors** (no clip/vit/encoder) and there is no mmproj on the machine
except Qwen's. Gemma 4 12B here is **text-only** — it cannot read images, so a
second-model vision comparison could not be run. The only vision encoder
available is Qwen3.6's mmproj.

## 6. Where this leaves the original goal
Goal: reduce prefill time for large inputs (500 → thousands of tokens).

- **Cold prefill:** compact font-16 image prefill ≈ **53s vs 119s text = ~2.2×
  faster**, at 75% of tokens, 3% CER (reliable). Real, safe win.
- **Token saving beyond ~25%** requires font ≤12, where accuracy becomes
  unreliable (occasional full hallucination) — unsafe for agentic ingestion.
- **Repeated context (agent loop):** cached text prefill ≈ **10s**, which beats
  every image number. Prefix/KV caching (`--cache-prompt`/`--cache-reuse`, already
  in `scripts/run-qwen36-moe.sh`) is the bigger, safer win for repeated prefixes.
- Image prefill = vision-encoder time + LLM-prefill(vision_tokens). The encoder
  is a roughly fixed cost per request; fewer/smaller pages reduce it.

## 7. Recommended config
**Reliable image-prefill path:** `compact=True`, `font_size=16`, `page_w=1400`,
`page_h=720`, `mode="gray"`. → 3% CER, 75% tokens, ~2.2× faster cold prefill.

**To capture the bigger token win safely:** hybrid — render bulk at compact
font 12 (44% tokens, ~2× faster) for gist/comprehension, and send the exact
strings the agent acts on (IDs, checksums, code, error codes) as real text
tokens. The model trusts the image for understanding and the text for exactness.

## 8. Artifacts produced this session
- `tools/vision-bench/prefill.py` — renderer (block-aware pagination, tab
  handling, `compact` mode, levenshtein).
- `tools/vision-bench/bench_prefill.py` — text vs image pure-prefill comparison.
- `tools/vision-bench/sweep.py` — (broken on a server reset; superseded).
- `tools/vision-bench/nl_test.py`, `compact_test.py`, `compact_tune2.py`,
  `compact_pages.py`, `compact_iso.py`, `compromise.py`, `large_pages.py` —
  the experiments behind the tables above.
- Rendered samples: `/home/manohar/Desktop/vision-render/`
