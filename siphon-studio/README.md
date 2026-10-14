# siphon-studio

A local **launcher, dashboard and headless CLI** for
[siphon.cpp](../README.md) — the fork that streams MoE experts from SSD into a
6 GB VRAM laptop via NVIDIA GDS/cuFile.

Siphon-studio wraps the existing (and battle-tested) `scripts/run-*.sh`
launchers and the `llama-server` that siphon.cpp builds. It does **not**
reimplement inference; it is a control plane:

- **Headless CLI** for AI agents / scripts (machine-readable `--json` output).
- **Terminal TUI** (Textual) for humans — start/stop, logs, metrics, profiles.
- **tmux-backed process supervision** so a loaded model keeps running even when
  your terminal or the TUI closes.
- A shared `core/` library reused by the native LM Studio-style desktop app.

## Layout

```
siphon-studio/
├─ siphon/
│  ├─ core/      # profiles, runner (tmux/direct/systemd), monitor, suites
│  ├─ cli/       # `siphon` headless command (--json)
│  ├─ tui/       # Textual terminal app
│  └─ desktop/   # GTK4/WebKitGTK native desktop host
├─ profiles/     # editable app profiles (JSON); configs/*.env stay untouched
├─ state/        # runtime: materialized envs, log files, cmd scripts, pids
└─ bin/siphon    # dev entry point (no pip install needed)
```

## Quick start (no install)

```bash
cd siphon-studio
python3 bin/siphon doctor          # environment checks
python3 bin/siphon list            # bundled profiles + discovered models
python3 bin/siphon up qwen3.6-moe-tq3 --dry-run   # print exact command
python3 bin/siphon up qwen3.6-moe-tq3              # launch in tmux
python3 bin/siphon wait qwen3.6-moe-tq3 --timeout 300
python3 bin/siphon status --json
python3 bin/siphon metrics
python3 bin/siphon logs qwen3.6-moe-tq3 --tail 40
python3 bin/siphon down qwen3.6-moe-tq3
python3 bin/siphon                  # no args -> launch the TUI
python3 bin/siphon desktop           # launch Siphon Studio desktop UI
./bin/siphon-studio                   # direct desktop launcher
```

The desktop app uses GTK4 with the system WebKitGTK runtime.  The approved
Siphon Studio surface remains the UI source, while model start/stop and live
telemetry are delegated to the shared `siphon.core` runner and monitor.  This
keeps the desktop shell lightweight and avoids bundling a second browser
runtime.  Because the surface is local-only, the host disables WebKitGTK's
Bubblewrap renderer sandbox by default on systems where the UID-map setup is
blocked; set `SIPHON_WEBKIT_SANDBOX=1` to force the sandbox when supported.

For a desktop-menu entry on this checkout, copy
`resources/com.siphon.studio.desktop` to `~/.local/share/applications/`.

## For AI agents

For the complete copy-paste-safe headless runbook, see
[`../HEADLESS_AGENT_GUIDE.md`](../HEADLESS_AGENT_GUIDE.md).

The headless surface is JSON-clean and idempotent:

```bash
siphon up <profile> && siphon wait --healthy <profile> \
  && curl http://127.0.0.1:8090/v1/chat/completions -d '...'
```

`status --json` / `metrics --json` / `logs --tail N --json` give agents full
machine-readable visibility.

## Local API for your apps (keys)

Studio serves one endpoint for your own apps and agents, `http://127.0.0.1:8088`
(this computer only; `SIPHON_API_PORT` changes the port). Every key opens exactly
one path:

| key | prefix | routes |
|---|---|---|
| decode | `sk-siphon-gen-` | `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/models` (OpenAI-compatible); `/v1/messages`, `/v1/messages/count_tokens` (Anthropic-compatible); all streaming |
| decision | `sk-siphon-dec-` | `/v1/decide`, `/v1/decide/warmup` |

In the desktop app open the **API** tab (the key icon in the rail). From the
top: **Local gateway** shows the port, a live map of where requests go and the
request counts for this session. **Decode API** and **Decision API** each have
their permanent key (Show / Copy), the URLs, requests per minute over the last
30 minutes, example code and a test box that sends one real request with that
key (the Decision playground shows how likely each option was). If the model
behind a key is not running, its section says so and has a Start button; you
can also start it in **Models → Decode** or **Models → Decision**. The counts
live in memory and start at zero with the app. Headless:

```bash
siphon api keys show                       # both keys (created on first use)
siphon api keys regenerate --scope decision   # new key; the old one stops working
siphon api serve        # the gateway without the desktop app
siphon api info         # endpoint, running engine, request shape
```

### Connect coding apps

The Decode section lists OpenCode, Claude Code and Codex CLI with a **Connect**
button (headless: `siphon api apps`, `siphon api connect <app>`,
`siphon api disconnect <app>`). Connect only adds things Siphon can take out
again:

| app | what Connect does | then |
|---|---|---|
| OpenCode | adds a `siphon` provider to `~/.config/opencode/opencode.json` (OpenCode merges it with `opencode.jsonc`, which is never edited; an existing `opencode.json` is backed up once and must be plain JSON) | pick **Siphon (local)** in `/models` |
| Claude Code | adds a `claude-siphon` command in `~/.local/bin` (Anthropic API env vars, then `exec claude`) | run `claude-siphon`; plain `claude` keeps your account |
| Codex CLI | adds a `codex-siphon` command (`-c` overrides; `~/.codex/config.toml` is not touched) | run `codex-siphon` |

Disconnect removes the provider or the command. Regenerating the decode key
rewrites every connected app with the new key. For any other app, copy the key
and the base URL (`http://127.0.0.1:8088/v1`; Anthropic-style apps take
`http://127.0.0.1:8088`). Coding agents send long prompts (measured first
requests: Claude Code 19.3k tokens, OpenCode 17.8k), so the first reply of a
new session takes about 8-11 minutes at ~36 tokens/s prefill on a 6 GB laptop
GPU. Later turns in the same session reuse the cached prompt (a 5.7k-token
history answered in 0.9 s); a new session starts the prompt over.

The keys stay the same until you regenerate them. They are kept in
`state/api/keys.json` (mode 0600, git-ignored) so they can be copied again;
the gateway only listens on this computer. The desktop app runs the gateway
while it is open.

What runs:

| running | profile | notes |
|---|---|---|
| Decode | the model's normal profile | Models tab → **Decode** settings |
| Decision | `qwen3.6-decisions` | ~0.65 s per decision (tool-choice benchmark: 52/52 < 1 s); Models tab → **Decision** settings |
| both | `qwen3.6-decode-decide`, rebuilt from the Decode settings on every start | one process; chat speed unchanged, decisions ~1.5 s; expert GPU cache capped at 2560 MiB |

The Decision settings are GPU memory for experts (up to 3456 MiB; more runs
out of GPU memory), RAM for experts, and the longest request (256/512/1024
tokens; longer requests get a 400 instead of reaching the engine).

A decision request names the options; Siphon builds the prompt and returns a
probability per option:

```bash
curl http://127.0.0.1:8088/v1/decide -H "Authorization: Bearer $SIPHON_DECISION_KEY" \
  -H "Content-Type: application/json" -d '{
    "input": "What is the latest score of the India vs Pakistan match?",
    "question": "Which tool should the assistant call first?",
    "options": [{"name": "sports_live", "description": "live scores and fixtures"},
                {"name": "web_search", "description": "search the internet"},
                {"name": "answer_directly", "description": "no tool needed"}],
    "context": "optional text shared by every call",
    "threshold": 0.9}'
# {"choice": "sports_live", "confidence": 0.999, "confident": true,
#  "probabilities": {...}, "usage": {"cache_hit": true, ...}, "timing": {"engine_ms": 670, ...}}
```

Speed comes from reusing the shared part of the prompt (options, question,
context). The first call with a new option set builds it (~3 s; call
`/v1/decide/warmup` with the same body minus `input` at startup); later calls
with the same set take ~0.65 s. The engine keeps one shared part at a time, so
apps that alternate between different option sets pay the ~3 s each switch.
Treat `"confident": false` as "ask again or fall back"; in the benchmark every
error except one had confidence below 0.9.

The model servers behind the gateway (port 8090) are still unauthenticated on
localhost; the keys gate the gateway, not the raw server.

## Add a local model folder

In the desktop **Model catalogue**, click **Add model folder** (the + button)
and select the folder containing your GGUF files. Studio registers the files
in place: it does not upload, copy, merge, rename, or load their weights during
import. Only files directly inside the selected folder are scanned.

The preview groups complete split-GGUF sets into one model. The number of parts
comes from the file naming and GGUF split metadata; there is no three-part
limit. Keep the original names, such as `model-00001-of-00005.gguf`, and all
parts in the same folder. Combined artifact and tensor sizes include
every part. Renamed, missing, corrupt, or conflicting parts show an error and
cannot be imported as a complete model.

If the folder contains multiple models or quantizations, select the ones you
want and click **Add selected models**. Nothing is selected automatically in
a mixed folder, even if only one entry is valid. Each selection receives its
own launch profile; adding an existing model preserves its saved settings.
Auxiliary GGUFs (such as vision projectors) and dense-only models are not
standalone routed-MoE import targets.

Imported profiles remember their source folder and ordered part paths. Metadata
is reused on reopen while every part's path, size, and modification time remain
unchanged. Removing a part marks the model unavailable; the full set is checked
again before launch. File validation does not guarantee inference support for
every architecture: the runtime capability check still runs at startup.

The catalogue reports **stored tensor bytes**, not active-per-token parameters
or required RAM/VRAM. Routed expert weights, auxiliary per-layer token embedding
lookup tables, and other non-expert weights are shown separately. Other
non-expert weights include ordinary embeddings, attention, shared experts, and
any remaining non-routed tensors; they are not an always-active memory estimate.
The three categories sum to **All tensor weights**, which excludes GGUF headers
and padding. Unknown measurements show a dash; a measured absent category shows
zero. Classification uses tensor roles, not model names or a fixed part count.

Metadata cached before this breakdown is refreshed on the next open. Existing
imports do not need to be added again, and their launch settings are preserved.

Folder-import regression checks, from the repository root:

```bash
/usr/bin/python3 tests/test-studio-model-files.py
/usr/bin/python3 tests/test-studio-folder-import.py
node --test tests/test-studio-settings.js
```

## Remove a model from Studio

Select an entry in **Model catalogue** and click the trash button next to **+**.
Confirm **Remove from Studio** to remove only that catalogue registration.
GGUF files and saved launch settings are kept. Missing-file entries can be
removed too; running, loading, or stopping models must be stopped first.

Removal persists across app restarts, including bundled entries. Add the same
model folder again to restore its registration and saved settings. Removing
the last model leaves an empty catalogue with **Add model folder** available.
This is a desktop-only catalogue preference; headless CLI profile discovery
is unchanged. It is stored in `state/desktop-model-exclusions.json`.

Removal regression checks, from the repository root:

```bash
python3 -m pytest -q tests/test-studio-model-removal.py
node --test tests/test-studio-settings.js
```

## CPU or GPU execution

In **Model catalogue → Execution device**, leave **Use GPU** checked for GPU
inference, or uncheck it for CPU-only inference. Stop the model before switching,
then **Apply settings → Start model**. The initial default is GPU when an NVIDIA
GPU is detected; machines without a supported GPU default to CPU. A saved
selection is remembered per profile.

CPU uses the optimized `siphon-cpu` runtime with zero GPU layers. Context,
batch/microbatch, KV precision, prompt-state cache, speculation, and thread
settings are stored separately from GPU settings. Switching modes does not
replace the saved GPU L1/L2 allocations. CPU defaults to Q8_0 KV, up to eight
threads, and speculation off. F16 KV and native MTP are optional; TQ3 CPU KV
is not enabled. MTP is not guaranteed to improve CPU throughput.

CPU weights are file-backed and use the OS page cache, **not** the GPU
DirectStorage/L1/L2 promotion path. GPU expert-cache and GDS controls are disabled
in CPU mode. Active CPU KV, recurrent state, saved prompt states, and a provisional
512 MiB workspace estimate are checked against available RAM, preserving at least
2 GiB headroom. This is a launch-time estimate, not a guarantee against later
system-wide memory pressure; allocations are never silently rewritten while running.

Studio searches for `build-cpu/bin/llama-server` in this repository, a nested
`siphon-cpu` checkout, or the sibling `inference/siphon-cpu` checkout. Override
discovery with `SIPHON_CPU_SERVER_BIN=/absolute/path/to/llama-server` or
`SIPHON_CPU_ROOT=/absolute/path/to/siphon-cpu`. A missing CPU executable produces
an actionable error; Studio never silently falls back to GPU. Model architecture
support still depends on the selected runtime and is checked at startup.

Headless CPU example (a separate editable profile preserves the original):

```bash
siphon profile new qwen36-cpu --from qwen3.6-moe-tq3
```

Edit the new profile's `env` object in `profiles/qwen36-cpu.json` (under your
`SIPHON_STUDIO_DIR`, if overridden), retaining its other fields:

```json
{
  "SIPHON_EXECUTION_MODE": "cpu",
  "CPU_THREADS": "8",
  "CPU_CTX": "65536",
  "CPU_CACHE_TYPE": "q8_0",
  "CPU_SPEC_TYPE": "none"
}
```

Then preview and launch it:

```bash
siphon up qwen36-cpu --dry-run
siphon up qwen36-cpu
```

CPU supports the `direct` and `tmux` drivers. The `systemd` driver rejects CPU
launches because its existing units are GPU-specific. Only one managed model
runs at a time, regardless of execution mode.

## Closing the desktop app

Closing Siphon Studio's native desktop window unloads all discovered managed
models, including CPU models and models still loading. Chat and pending launches
are cancelled first. The window waits for model and launcher-process cleanup
before exiting; stubborn processes are terminated with bounded signal escalation.
If cleanup fails, the window remains open so closing again can retry, and the
failure is logged. Model files and saved launch settings are kept.

Cleanup targets managed process ownership, tmux sessions, or systemd units—not
every process named `llama-server` or every process using the same GGUF. An
independently launched inference server is not terminated by desktop close.

The headless CLI and TUI retain their detachable behavior: closing a terminal
alone does not unload a model. Use `siphon down <profile>` explicitly, or close
the native Studio window. An uncatchable kill (`SIGKILL`) or power loss cannot
run the desktop shutdown handler.

## Process supervision

`runner.py` uses three backends, selected automatically (or via
`SIPHON_DRIVER=tmux|direct|systemd`):

| Driver | When |
|---|---|
| `tmux` | default if installed — survives disconnects, scriptable, attachable |
| `direct` | fallback — plain child process with pid file |
| `systemd` | manage existing `--user` units (e.g. `siphon-qwen3.6.service`) |

Install tmux: `sudo apt install tmux`. Install the TUI: `pip install textual`.

Only one managed model can run at a time; `up` refuses to start a second server
until the current one is stopped.

## Making an editable profile

```bash
siphon profile new qwen36-fast --from qwen3.6-moe-tq3
siphon list
```

Edit `profiles/qwen36-fast.json` to change its `env` fields, or select the
profile in the desktop catalogue and use **Apply settings**.

## Desktop settings checks

Apply Settings saves the launch profile; a running model must be restarted to
use launch-time changes. Stop the model before changing its VRAM expert cache,
L2 RAM cache, or prompt-cache allocation. These controls remain fixed during
startup, execution, and shutdown: falling free memory must not shrink the
displayed or saved allocations already committed to the model.

A zero L2 or prompt-cache allocation is a valid setting, not a request to restore
the default. While stopped, resource monitoring preserves the selected values
and warns if they exceed current availability. Apply and Start reject unsafe
allocations, including those that would leave less than 2 GiB of RAM headroom.

The expert-VRAM ceiling also reserves model-specific GPU tensor bytes, KV cache
at the selected context length and precision, separate native-MTP KV memory,
and fixed recurrent state. CPU-backed input embeddings and auxiliary lookup
tables are not counted as GPU-resident weights. There is no flat 512 MiB compute
deduction. The engine emits `SIPHON_GPU_MEMORY` after creating the real contexts:
GPU model, context, compute, and allocated expert-pool bytes (native MTP shares
weights, counted once). After successful health and inference probes, Studio
caches those measurements in `state/runtime-memory.json`. They are reused only
for matching model files, engine/launcher builds, GPU and memory-affecting
settings. Cache compaction is accounted for only for that measured pool setting.
Before measurement, the descriptor-based ceiling is explicitly **provisional**:
workspace is unknown, and startup verifies actual allocation instead of charging
a fabricated allowance. VRAM retains 128 MiB of safety headroom; the separate
2 GiB RAM headroom rule is unchanged. Unknown memory layouts are reported rather
than guessed. This is not a guarantee against CUDA allocation failure or another
application consuming VRAM between validation and launch. The KV
precision control shows per-token payload size, with exact byte counts in its
tooltip. Context/precision edits recalculate the ceiling without silently
rewriting existing allocations; a running model's memory is not reserved twice.

For Qwen3.6-35B-A3B, the main TQ3 KV payload is **4,480 bytes/token (4.375 KiB)**:
10 attention layers × two K/V rows × 512 values × 14 bytes per 32-value block.
The other layers use fixed recurrent state, not token-growing KV rows. With
native MTP, the current launcher uses a separate F16 draft cache adding
**2,048 bytes/token**, for **6,528 bytes/token (6.375 KiB)** combined. These are
KV payload sizes, not total process memory; fixed state and workspace are extra.
Other architectures and KV precisions have different sizes.

Saved profile settings remain visible even if the GGUF file is missing. Such
catalogue entries are marked unavailable, not ready; reconnect or re-import the
model artifact before launching. Startup's one-token readiness probe is excluded
from dashboard request throughput and graphs. They stay empty until an actual
request runs, while hardware telemetry remains available.

Run the focused UI-state regression checks from the repository root:

```bash
node --test tests/test-studio-settings.js
/usr/bin/python3 tests/test-studio-settings.py
```
