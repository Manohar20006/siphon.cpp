# qwen.cpp

`qwen.cpp` is a CUDA-focused `llama.cpp` fork for running Qwen3.6 MoE locally on
constrained VRAM systems.

The default setup is tuned for a single-GPU laptop/server workflow:

- DirectStorage-style expert streaming for MoE weights
- static pinned-RAM expert suites selected from routing traces
- CUDA FlashAttention with fused TQ3 KV handling
- OpenAI-compatible `llama-server`
- built-in web chat UI

## Tested Hardware

This repo was developed and tested on:

| Component | Tested setup |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| VRAM reported by `nvidia-smi` | 6141 MiB |
| CPU | 13th Gen Intel Core i7-13700HX |
| CPU threads | 24 |
| System RAM | 16 GiB |
| NVIDIA driver | 595.71.05 |
| CUDA toolkit | 13.3 |
| OS | Linux |

The default launch keeps the GPU cache modest and uses pinned system RAM for a
static expert suite. Larger GPUs can raise the cache sizes; smaller systems may
need lower context or cache settings.

## Quick Start

Clone the repo, place the GGUF model in `models/`, build, then start the server:

```bash
git clone <your-repo-url> qwen.cpp
cd qwen.cpp

mkdir -p models
# Put this file here:
# models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf

./scripts/build-gds-cuda.sh
./scripts/run-qwen36-moe.sh
```

Open the built-in web chat:

```text
http://127.0.0.1:8090
```

OpenAI-compatible API base URL:

```text
http://127.0.0.1:8090/v1
```

Default model alias:

```text
qwen3.6-moe-tq3
```

Health check:

```bash
curl http://127.0.0.1:8090/v1/models
```

## Required Files

The GGUF model is not included in this repo. Put it here:

```text
models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

The default expert suite is included:

```text
expert-suites/moe-static-experts-suite-8192.txt
```

For prefill-only experiments:

```text
expert-suites/moe-static-prefill-experts-suite-8192.txt
```

Other included static expert-suite sizes:

```text
expert-suites/moe-static-experts-suite-2048.txt
expert-suites/moe-static-experts-suite-4096.txt
expert-suites/moe-static-experts-suite-6144.txt
expert-suites/moe-static-experts-suite-7168.txt
expert-suites/moe-static-experts-suite-8192.txt
expert-suites/moe-static-experts-suite-10240.txt
```

## Build

CUDA/GDS build:

```bash
./scripts/build-gds-cuda.sh
```

The script creates:

```text
build-gds-cuda/bin/llama-server
```

## Run

Default 64k context server:

```bash
./scripts/run-qwen36-moe.sh
```

Override model path, expert suite, port, context, or cache sizes:

```bash
MODEL=/path/to/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
EXPERTS=expert-suites/moe-static-prefill-experts-suite-8192.txt \
PORT=8090 \
CTX=65536 \
MOE_GPU_CACHE_MIB=2816 \
MOE_PINNED_CACHE_MIB=8192 \
./scripts/run-qwen36-moe.sh
```

The default server keeps prompt caching and `/metrics` enabled. It uses the
model's embedded Qwen chat template with reasoning disabled and parsed out.

## Configuration And Tuning

Most users should change settings in:

```text
configs/qwen3.6-moe-tq3.env
```

Every setting in that file can also be overridden for one launch:

```bash
MOE_GPU_CACHE_MIB=4096 CTX=32768 ./scripts/run-qwen36-moe.sh
```

The most important variables are:

| Variable | Default | Where used | What it changes |
| --- | ---: | --- | --- |
| `MODEL` | `models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` | wrapper | Path to the GGUF model. Use an absolute path or a path relative to the repo. |
| `EXPERTS` | `expert-suites/moe-static-experts-suite-8192.txt` | wrapper/env | Static pinned-RAM expert suite. Change this to use another generated suite. |
| `HOST` / `PORT` | `127.0.0.1` / `8090` | server | Web UI and API bind address. |
| `CTX` | `65536` | `-c` | Context window. Higher values use more KV-cache memory and can increase prompt processing cost. |
| `BATCH` | `512` | `-b` | Logical prompt batch size. Larger can improve prefill throughput if memory allows. |
| `UBATCH` | `128` | `-ub` | CUDA microbatch size. Increase carefully on larger GPUs; reduce if CUDA OOM occurs. |
| `MOE_GPU_CACHE_MIB` | `2816` | `--moe-gpu-cache-mib` | VRAM reserved for persistent MoE expert slots. This is the first knob to raise on larger GPUs. |
| `MOE_PINNED_CACHE_MIB` | `8192` | `--moe-pinned-cache-mib` | Host pinned RAM budget for static expert cache. Raise only if you have enough system RAM. |
| `GDS_READ_THREADS` | `20` | `LLAMA_GDS_READ_THREADS` | Number of parallel GDS/cuFile read workers for expert streaming. |
| `SPEC_DRAFT_N_MAX` | `3` | `--spec-draft-n-max` | MTP speculative draft length. Try `2` or `3` when benchmarking. |
| `REASONING` | `off` | `--reasoning` | Enables or disables reasoning mode. |
| `REASONING_BUDGET` | `0` | `--reasoning-budget` | Reasoning token budget when reasoning is enabled. |
| `CACHE_REUSE` | `0` | `--cache-reuse` | KV-shift cache reuse. Kept off by default because the current MTP context reports it unsupported. |
| `METRICS` | `1` | `--metrics` | Enables `/metrics` for runtime monitoring. |

Less common but still useful settings:

| Variable | Default | What it changes |
| --- | ---: | --- |
| `ALIAS` | `qwen3.6-moe-tq3` | Model name exposed by `/v1/models` and used by OpenAI-compatible clients. |
| `CHAT_TEMPLATE` | empty | Leave empty to use the embedded Qwen template. Set `chatml` only as an experiment. |
| `CHAT_TEMPLATE_KWARGS` | empty | Extra chat-template options passed through to `llama-server`. |
| `SPEC_DRAFT_P_MIN` | `0` | Minimum probability threshold for MTP draft acceptance behavior. |

Additional expert and quality knobs are environment variables:

| Variable | Default | What it changes |
| --- | ---: | --- |
| `LLAMA_MOE_EXPERT_USED_OVERRIDE` | `4` | Number of routed experts used per token. `4` is fast; `6` or `8` is safer for quality checks. |
| `LLAMA_DSTORAGE_SUMMARY` | `1` | Prints DirectStorage/MoE cache summaries on shutdown. |
| `GGML_CUDA_DISABLE_GRAPHS` | `1` | Disables CUDA graphs in the wrapper default. Useful while validating this custom path. |
| `LLAMA_MOE_PINNED_STATIC_EXPERTS_FILE` | set from `EXPERTS` | Runtime path to the static expert suite loaded into pinned RAM. |

Hardware tuning examples:

```bash
# More VRAM available: give more space to resident experts.
MOE_GPU_CACHE_MIB=4096 ./scripts/run-qwen36-moe.sh

# More system RAM available: try a larger static pinned expert suite.
EXPERTS=expert-suites/moe-static-experts-suite-10240.txt \
MOE_PINNED_CACHE_MIB=10240 \
./scripts/run-qwen36-moe.sh

# Less VRAM available: reduce context and expert VRAM cache.
CTX=32768 MOE_GPU_CACHE_MIB=2304 ./scripts/run-qwen36-moe.sh

# Higher quality check: use more routed experts per token.
LLAMA_MOE_EXPERT_USED_OVERRIDE=6 ./scripts/run-qwen36-moe.sh
```

Rules of thumb:

- Increase `MOE_GPU_CACHE_MIB` first on larger GPUs. It directly improves the
  chance that routed experts are already resident in VRAM.
- Keep enough VRAM for dense weights, KV cache, compute buffers, and the CUDA
  context. If startup fails with CUDA OOM, reduce `MOE_GPU_CACHE_MIB`, `CTX`,
  `BATCH`, or `UBATCH`.
- Increase `MOE_PINNED_CACHE_MIB` only when the machine has enough free system
  RAM. On a 16 GiB machine, the 8192 MiB suite is the practical default.
- Larger `CTX` gives a longer context window but costs KV-cache memory and can
  make prompt processing slower.
- Treat `LLAMA_MOE_EXPERT_USED_OVERRIDE` as a speed/quality knob. K=4 is the
  fast default; K=6 or K=8 is better for conservative validation.

## API Example

```bash
curl http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-moe-tq3",
    "messages": [{"role": "user", "content": "Write a tiny Python LRU cache."}],
    "max_tokens": 256
  }'
```

## Runtime Notes

- The built-in web UI is available at `http://127.0.0.1:8090`.
- The OpenAI-compatible endpoint is `http://127.0.0.1:8090/v1`.
- For best same-session browser latency, enable **Settings -> Developer ->
  Pre-fill KV cache after response** in the web UI.
- Keep `max_tokens` finite for interactive chat so one long response does not
  block the next request on a single-slot server.
- `CACHE_REUSE` defaults to `0` because the current MTP speculative context
  reports KV-shift cache reuse as unsupported.

## Trace And Expert Suite Generation

To rebuild expert suites from scratch:

```bash
python3 scripts/build-moe-trace-prompt-suite.py

# Start a trace-recording server, then:
python3 scripts/run-moe-routing-trace.py \
  moe-trace-prompts-1000.jsonl \
  moe-routing-suite-1000.progress.jsonl \
  --url http://127.0.0.1:8091/completion \
  --n-predict 32

python3 scripts/select-static-moe-experts.py \
  moe-routing-suite-1000-combined.jsonl \
  expert-suites/moe-static-experts-suite-8192.txt \
  --budget-mib 8192
```

## Repository Notes

- GGUF models, build directories, traces, profiling captures, logs, and local
  artifacts are intentionally ignored.
- `expert-suites/` contains small text files that are safe to commit.
- `models/` is only a placeholder directory for local model files.
