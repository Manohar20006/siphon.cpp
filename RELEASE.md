# v0.1 — Initial public release

Release notes

- Initial public release of siphon.cpp.
- Key features:
  - SSD expert streaming using NVIDIA GDS/cuFile to enable MoE models on low‑VRAM machines.
  - Support for GGUF Qwen3.6 35B‑A3B model files.
  - Compressed TQ3 FlashAttention for long‑context decoding.
  - Tuned defaults for single‑GPU laptops (tested on RTX 4050, 6 GB VRAM).

How to publish this as a GitHub Release

1. Go to the repository Releases page: https://github.com/Manohar20006/siphon.cpp/releases
2. Click "Draft a new release".
3. Tag version: `v0.1` and title: `v0.1 — initial public release`
4. Paste these notes into the release description and publish the release.

Why publishing a Release helps

- Releases are frequently crawled and indexed by search engines and make the project more discoverable.

After you publish

- Run Google Search Console URL Inspection for the repo URL and the Pages URL (if you enable Pages) and click "Request indexing" to speed up discovery.

