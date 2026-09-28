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

## Process supervision

`runner.py` uses three backends, selected automatically (or via
`SIPHON_DRIVER=tmux|direct|systemd`):

| Driver | When |
|---|---|
| `tmux` | default if installed — survives disconnects, scriptable, attachable |
| `direct` | fallback — plain child process with pid file |
| `systemd` | manage existing `--user` units (e.g. `siphon-qwen3.6.service`) |

Install tmux: `sudo apt install tmux`. Install the TUI: `pip install textual`.

Only one model can run on the 6 GB GPU at a time; `up` refuses to start a
second server unless `--force` is given.

## Making an editable profile

```bash
siphon profile new qwen36-fast --from qwen3.6-moe-tq3 \
  --set MOE_GPU_CACHE_MIB=2304 --set CTX=32768
siphon profile ls
```
