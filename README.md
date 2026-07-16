# siphon.cpp

![Platform](https://img.shields.io/badge/platform-Linux-blue)
![CUDA](https://img.shields.io/badge/CUDA-required-green)
![VRAM](https://img.shields.io/badge/tested-6GB%20VRAM-orange)
![API](https://img.shields.io/badge/API-OpenAI--compatible-purple)

Run **Qwen3.6 35B-A3B MoE locally on a 6 GB VRAM laptop**.

`siphon.cpp` is a CUDA-focused `llama.cpp` fork built for one practical goal:
make a serious MoE model usable on hardware that normally looks too small for
it.

Instead of loading every expert into VRAM or filling system RAM until the
machine becomes unusable, `siphon.cpp` streams MoE experts from SSD on demand with
NVIDIA GDS/cuFile, keeps hot experts cached, and uses compressed TQ3
FlashAttention for long-context decode.

On the tested RTX 4050 Laptop GPU with 6 GB VRAM, this setup has reached:

| Workload | Observed local speed |
| --- | ---: |
| Prompt / prefill | 50+ tokens/s |
| Decode / generation | around 20 tokens/s in favorable runs |

- Runs through the normal `llama-server` web UI.
- Connects to OpenCode, Hermes, and other OpenAI-compatible coding tools.
- Streams experts from SSD instead of forcing full RAM residency.

The important part is not just the speed. The machine stays usable. You can
keep a browser, editor, terminal, and other normal desktop work open because the
project is designed around **SSD expert streaming** and bounded RAM pressure,
not brute-force RAM residency.

The project is especially focused on local coding workflows: use it from the
built-in web UI, or connect OpenCode, Hermes, and other OpenAI-compatible coding
agents to the local server.

## Platform Status

This is currently a **Linux-first CUDA project**. The tested path is Linux +
NVIDIA GPU + CUDA + fast SSD. Windows support is not the primary target yet;
work on a Windows-friendly path is planned/in progress.

> [!TIP]
> If you are currently on Windows and want to test this today, the simplest path
> is to install Linux as a dual-boot setup and run `siphon.cpp` there. That keeps
> your normal Windows install intact while giving this project the Linux
> CUDA/GDS environment it was built around.

## Requirements

System requirements:

- Linux.
- NVIDIA GPU with CUDA support.
- Recent NVIDIA driver.
- CUDA toolkit installed and visible to CMake.
- Fast SSD for model and expert streaming.
- CMake, C++ compiler, Git, Python 3, and pip.

On Ubuntu-like systems, install the common build dependencies first:

```bash
sudo apt update
sudo apt install -y build-essential cmake git python3 python3-pip python3-venv pkg-config
```

Install the NVIDIA driver and CUDA toolkit using NVIDIA's instructions for your
Linux distribution. After installation, this should work:

```bash
nvidia-smi
nvcc --version
```

Python dependencies are mainly needed for helper scripts such as conversion,
tracing, benchmarking, and expert-suite generation:

```bash
python3 -m pip install -r requirements.txt
```

If your distro blocks system-wide pip installs, use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Quick Start

Clone the repo:

```bash
git clone https://github.com/Manohar20006/qwen.cpp.git siphon.cpp
cd siphon.cpp
```

Place the GGUF model in `models/`:

```bash
mkdir -p models
# Put this file here:
# models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```

Build the CUDA/GDS server:

```bash
./scripts/build-gds-cuda.sh
```

This is a one-time build step. You do not need to run it every time you start
the model unless you update the code, clean the build directory, or change build
settings.

The first build can take several minutes. Wait until it finishes and prints the
`build-gds-cuda/bin/llama-server` path before starting the model.

Start the server:

```bash
./scripts/run-qwen36-moe.sh
```

Use these local URLs after the server starts:

| Use | URL |
| --- | --- |
| Web chat | `http://127.0.0.1:8090` |
| OpenAI-compatible API | `http://127.0.0.1:8090/v1` |
| Health check | `http://127.0.0.1:8090/v1/models` |

Default model alias:

```text
qwen3.6-moe-tq3
```

> [!TIP]
> If setup fails, you can paste the README, your terminal output, and your
> hardware details into any coding assistant and ask it to help you debug the
> repo setup. Most issues are missing CUDA/CMake dependencies, a missing model
> file, or port `8090` already being used.

## Why This Exists

Most local MoE setups fail on small GPUs for one of two reasons:

- They need far more VRAM than a laptop GPU has.
- They spill so much into system RAM that the whole machine becomes painful to
  use.

`siphon.cpp` takes a different path:

The default setup is tuned for a single-GPU laptop/server workflow:

- Stream MoE expert weights from SSD instead of permanently loading everything.
- Keep a bounded VRAM expert cache for hot routed experts.
- Use a static pinned-RAM expert suite selected from routing traces.
- Avoid long-context KV dequant overhead with fused TQ3 CUDA FlashAttention.
- Serve through the normal OpenAI-compatible `llama-server`.
- Chat in the browser at `http://127.0.0.1:8090`.
- Connect local coding agents through `http://127.0.0.1:8090/v1`.

## Tested Hardware

This repo was developed and tested with:

| Item | Tested setup |
| --- | --- |
| Model source | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` |
| GGUF file used by default | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` |

Hardware used for the numbers above:

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

## Required Files

The GGUF model is not included in this repo. Download the tested model from
`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, then put the default GGUF file here:

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

## Known Limits

> [!IMPORTANT]
> This is currently Linux/CUDA-first. Windows support is not the main path yet.

- The model weights are not included. You must provide the GGUF file yourself.
- Performance depends heavily on SSD speed, laptop power mode, thermals, CUDA
  version, and how much VRAM is available after the desktop environment.
- The default settings are tuned for the tested 6 GB VRAM laptop. Larger GPUs
  should raise `MOE_GPU_CACHE_MIB`; smaller or busier GPUs may need lower
  `CTX`, `BATCH`, `UBATCH`, or GPU cache size.
- The included expert suites are static routing-trace selections. They are a
  practical default for this model, not a universal optimal expert set for every
  workload.
- If port `8090` is already in use, set another port:

```bash
PORT=8091 ./scripts/run-qwen36-moe.sh
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
| `CHECKPOINT_MIN_STEP` | `0` | `--checkpoint-min-step` | Minimum spacing between prompt checkpoints. `0` keeps OpenAI chat/agent turns cache-friendly by checkpointing short turn boundaries. |
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
- `CHECKPOINT_MIN_STEP=0` keeps same-session OpenAI chat and coding-agent
  prompts from falling back to only the system-prompt checkpoint on short turns.

## Practical Laptop Tips

For best performance:

- Keep the laptop plugged in.
- Use the vendor/OS performance mode instead of silent or battery-saver mode.
- Keep the model and expert files on a fast SSD.
- Let the GPU stay cool; thermal throttling can reduce both prompt and decode
  speed.
- Leave some system RAM free for the OS and browser. The default settings are
  meant to keep the machine usable, not consume every available GiB.

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
