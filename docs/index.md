---
title: "siphon.cpp — Qwen3.6 MoE on a 6GB VRAM Laptop"
---

# siphon.cpp — Qwen3.6 MoE on a 6GB VRAM Laptop

siphon.cpp (siphon cpp, siphon-cpp)

siphon.cpp brings Qwen3.6 35B‑A3B Mixture‑of‑Experts inference to low‑VRAM laptops by streaming MoE experts from SSD with NVIDIA GDS/cuFile and keeping a small VRAM cache. It supports GGUF models and uses compressed TQ3 FlashAttention for long‑context decoding.

Quick links

- Repo: https://github.com/Manohar20006/siphon.cpp
- Quick start: clone, place GGUF into `models/`, `./scripts/build-gds-cuda.sh`, `./scripts/run-qwen36-moe.sh`

Why this helps search engines

- This plain HTML/Markdown page includes the searchable keywords people use: "siphon cpp", "siphon-cpp", "qwen3.6", "MoE", "GGUF", "CUDA", "low VRAM", "local LLM" — which improves exact-match discovery.

Get started

1. Clone the repo:

```bash
git clone https://github.com/Manohar20006/siphon.cpp
cd siphon.cpp
```

2. Place the GGUF model file in `models/` (see README for the exact filename).
3. Build and run:

```bash
./scripts/build-gds-cuda.sh
./scripts/run-qwen36-moe.sh
```

Feedback & testing

If you test this, please open an issue with your hardware details (GPU, VRAM, OS) and the steps you followed so I can improve defaults and tuning.

