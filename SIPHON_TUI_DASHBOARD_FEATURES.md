# siphon TUI — New Dashboard Design & Feature Spec

> Living design doc for rebuilding the siphon-studio TUI around a dashboard-first
> landing screen. Companion: `SIPHON_STUDIO_PROGRESS.md` (build progress) and
> `siphon-deep-dive.md` (how siphon.cpp works).

---

## 1. Goal

Rebuild the TUI so that when it opens, the **Dashboard** is the landing page.
It surfaces the machine's live state (GPU/CPU/RAM + temps) immediately, lists
every available model, and lets the user click any model to see its config
(every setting adjustable), plus the model's real-time metrics once it is
running — and Start/Stop it — all from a single screen.

The current TUI is a 6-tab control center (Dashboard/Metrics/Slots/Logs/
Config/Doctor). The new one is dashboard-first and replaces it (per the
"forget the current TUI" directive).

---

## 2. Final dashboard requirement (from user)

The dashboard has three zones:

1. **Host telemetry** (always visible):
   - VRAM **used** and **free**
   - GPU utilization
   - RAM **used** and **free**
   - CPU utilization
   - GPU **temp**, CPU **temp**, **disk/NVMe temp**

2. **Available models** (list/cards). Clicking a model opens its **detail
   panel** showing:
   - its **settings / config**
   - **every setting is adjustable** (editable, not read-only)
   - status + other important details
   - **Start** and **Stop** button

3. **Running metrics** (shown only while the model is running & the server is up):
   1. decode speed
   2. prefill speed
   3. VRAM expert cache hit        (L1 cache hit)
   4. RAM expert cache hit         (L2 pinned-cache hit)
   5. draft acceptance (if present)
   6. NVMe / GDS transfer speed
   7. RAM → VRAM transfer speed
   8. total streamed data size
   9. no. of decode calls
   10. no. of prefill calls

---

## 3. Feature set

### 3.1 Host telemetry — front and center (persistent)

- Compact **system bar pinned at the top** visible on *every* screen:
  - GPU util %, GPU temp, free VRAM
  - CPU %, CPU temp
  - RAM %, free RAM
  - (optionally NVMe temp)
- **System health panel** on the Dashboard with 4 color-coded cards:
  - **GPU**: util %, VRAM used/total/free GiB, temp (°C), power draw
  - **CPU**: util %, temp (°C), clock GHz
  - **RAM**: %, used/total GiB, free GiB
  - **DISK/NVMe**: temp (°C), load
- **Aggregate VRAM utilization bar** (used vs `gpu_mib_total`) so the fullness
  of the 6 GiB GPU is visible at a glance — critical because only one model
  runs at a time.

All values come from `SysStats` (`siphon/tui/data.py`, `fetch_sys_stats()`),
which already provides: `cpu`, `cpu_temp`, `cpu_freq_mhz`, `mem_pct`,
`mem_used_gb`, `mem_total_gb`, `mem_free_gb`, `gpu_mib_free/used/total`,
`gpu_temp/util/power`, `nvme_temp`, `load`. Every field fails soft to zero.

### 3.2 Model grid + detail (core interaction)

- **Model card grid** (not a flat list). Each card shows:
  - model name
  - GGUF file size (from `profiles.find_models()`)
  - status chip: **running** (green) / **loading** (amber) / **idle** (dim)
- **Live activity** on each card for running models:
  - decode t/s (`predicted_tokens_seconds`)
  - context/slots usage (`/slots`)
  - so the landing screen doubles as a "what's happening now" view.
- **Click / select a model → detail pane** (see §3.4 for the editable config):
  - resolved config rendered with human labels from `KNOB_LABELS`
  - launcher script path
  - status state, latency ms, port, base URL
  - expert suite name
- **Start / Stop / Refresh** buttons (per spec).

### 3.3 Model control & operations

- **Start** runs the launch asynchronously then shows a live **loading**
  progress driven by health state transitions (`loading` → `ok`).
- **Stop** confirms before destructive stop.
- **Refresh** forces an immediate re-poll of all states.
- **Status/health toast** on every action result (success color vs error).
- **Last-error line** shown when a model fails to start, pre-filled from
  `Profile.validate()` issues list.
- **GPU-contention flash**: when starting a model while another is running on
  the 6 GiB GPU, clearly show *which* profile is using the GPU, since
  `runner.up()` refuses to start a second model.

### 3.4 Config display & editing (WRITE-ENABLED — per user)

The config shown for a selected model is **editable, not read-only**.

- **Field editing** — each setting is an editable control (numeric `Input`,
  text field, or selector where a finite set applies, e.g. `SPEC_TYPE`),
  labelled with its human name from `KNOB_LABELS` and its hint.
- **Apply / Save** persists changes:
  - **App profiles**: `env` dict updated + `Profile.save()` + re-`materialize_env()`
    (paths already exist in `core/env.py` + `profiles.Profile.save()`).
  - **Bundled profiles**: read-only at the file level — offer a **Clone to
    editable app profile** action first (`Profile.from_json`/create + save),
    then edit the clone. A bundled profile is never written in place.
- **Validation on change** — re-run `Profile.validate()` live as values change
  and inline-flag problems (e.g. VRAM-budget overflow) before Apply.
- **Changed vs applied state** — mark dirty fields and offer Apply / Revert.
- **Apply-then-restart guard** — a config change only takes effect on a fresh
  Start; if the model is running, warn that it must be restarted to take effect
  (optionally auto `restart`).

### 3.5 Per-model inspection

- **Inline log tail** for the selected model (follow toggle on/off, clear),
  so the dashboard shows recent server output without flipping to a Logs tab.
- **Action history / activity feed**:
  - e.g. "started qwen3.6 at 14:02", "stopped kat-coder", "start failed: OOM".
  - short ring buffer, newest on top.

---

## 4. Running-metrics panel (the 10 metrics, only while running)

Displayed only when the selected model is running and its server is healthy.
Each metric maps to a real data source (see source column); the DSTORAGE-based
ones come from the end-of-run log block and/or the live log tail.

| # | Metric | Threshold/colour | Source |
|---|---|---|---|
| 1 | **Decode speed** (tok/s) | green >15, amber 5–15, red <5 | `/metrics` `predicted_tokens_seconds` |
| 2 | **Prefill speed** (tok/s) | green >25, amber 10–25, red <10 | `/metrics` `prompt_tokens_seconds` |
| 3 | **VRAM expert cache hit** (L1) | green >80%, amber 50–80%, red <50 | DSTORAGE `prefill/decode_hit_rate` (L1) |
| 4 | **RAM expert cache hit** (L2) | green >80%, amber 50–80%, red <50 | DSTORAGE pinned-hit vs cold-expert ratio |
| 5 | **Draft acceptance** (%, if MTP present) | green >80%, amber 50–80%, red <50 | `print_timing` "draft acceptance" |
| 6 | **NVMe/GDS transfer speed** (MiB/s) | amber → green >1 GiB/s | DSTORAGE_IO_SHAPE `effective_mib_s` |
| 7 | **RAM → VRAM transfer speed** | from ENSURE_PROFILE / STREAM pinned timing | DSTORAGE `pinned_host_to_cuda_batch` us/byte |
| 8 | **Total streamed data** (MiB/GiB) | running total | DSTORAGE_STREAM `{prefill,decode}_file_mib` sum |
| 9 | **No. decode calls** | counter | `/metrics` `n_decode_total` |
| 10 | **No. prefill calls** | counter | DSTORAGE/`print_timing` prompt counts |

Notes / honesty placement:
- #1/#2/#9 are live from `/metrics`. #9 maps directly to `n_decode_total`
  (which unlike the DSTORAGE counters streams throughout the run).
- #3–#8 depend on DSTORAGE_MOE data which the server **flushes only on a
  graceful exit** (SIGINT). To show them live, the TUI parses the **running
  server's log tail** for accumulating counters (updating "since start"
  values), and also renders a full **post-stop summary** from the end-of-run
  DSTORAGE block. Until the counters first appear they read "…" / 0 with a
  hint that they accumulate since the server started.
- Paired gauge + value: each metric shows a colour traffic-light and its number
  so a slow decode is explained by its cache-hit rate on the same panel.

---

## 5. Feature mapping to existing core (already implemented)

Everything below exists and can be reused directly — no need to re-implement.

| Need | Where it lives |
|---|---|
| Plot/status states per profile | `siphon.tui.app.poll_states()` → `runner.status()` |
| Start/Stop/Restart/Refresh | `siphon.core.runner.up()/down()/restart()` |
| GPU contention guard | `runner.up()` (`running_profiles`) |
| Health polling (loading→ok) | `siphon.core.monitor.fetch_health()` |
| Per-model metrics | `monitor.fetch_metrics()` |
| Human config labels | `profiles.KNOB_LABELS` |
| Resolved config | `Profile.resolve()` |
| App/bundled profile model | `profiles.Profile` |
| Profile validation issues | `Profile.validate()` |
| Model files + sizes | `profiles.find_models()` |
| Host telemetry | `tui.data.fetch_sys_stats()` → `SysStats` |
| Slots/context usage | `tui.data.fetch_slots()` |
| Env read/write (edit path) | `core.env.read_env()/write_env()` |
| Suite name for a profile | `Profile.resolve("EXPERTS")` + `suites.suite_info()` |

---

## 6. Out of scope for v1 (deliberately deferred)

- **Sparklines / live charts** on the dashboard — visual noise on the landing
  screen; advanced charting can live on a separate Metrics tab later.
- **Auto-management of expert suites** — show suite info but don't edit suites
  from the dashboard.
- **Multi-user / remote-server management** — single-host control only.

---

## 7. Suggested layout (dashboard first pass)

```text
┌──────────────────────────────────────────────────────────────┐
│ HEADER  (siphon-studio · …clock…)                            │
├─────────────────────────── system bar (all screens) ────────┤
│  SYS  GPU 47% 53°C 4.9GiB free   CPU 8% 97°C   RAM 23% free │
├──────────────────────────────────────────────────────────────┤
│  TAB  [ Dashboard ]                                          │
├──────────────────────────────────────────────────────────────┤
│          SYSTEM HEALTH PANEL                                 │
│   ┌─GPU──┐  ┌─CPU──┐  ┌─RAM──┐  ┌─DISK──┐  ← 4 cards        │
│   │ 47%  │  │  8%  │  │ 23%  │  │ 37°C  │                    │
│   │3.9/6 │  │ 97°C │  │3.5/15│  │ load  │                    │
│   └──────┘  └──────┘  └──────┘  └───────┘                    │
│   VRAM bar ▓▓▓▓▓▓░░░░░░  used vs total                       │
├──────────────────────────────────────────────────────────────┤
│  AVAILABLE MODELS                         DETAIL PANE        │
│  ┌ qwen3.6 ●RUNNING 20.3t/s ┐  ┌ config (editable fields) ┐  │
│  ┌ kat-coder ●idle          ┐  │ model [____]  ctx [____] │  │
│  ┌ qwen3next ●idle          ┐  │ L1    [____]  L2  [____] │  │
│  ┌ dflash    ●idle          ┐  │ spec  [____]  ...        │  │
│  └──────────────────────────┘  │ [Apply] [Revert] [Clone] │  │
│                                │ status · health · port    │  │
│                                │ [Start] [Stop] [Refresh]  │  │
│                                └───────────────────────────┘  │
├──────────────────────────────────────────────────────────────┤
│  RUNNING METRICS (visible only while the model is running)   │
│   1 decode __ t/s      2 prefill __ t/s                      │
│   3 L1 hit __%         4 L2 hit __%     5 draft acc __%       │
│   6 NVMe/GDS __ MiB/s  7 RAM→VRAM __ MiB/s                   │
│   8 streamed __ MiB    9 decode calls __  10 prefill calls __ │
├──────────────────────────────────────────────────────────────┤
│  [log tail]                        status / last error...    │
└──────────────────────────────────────────────────────────────┘
```

---

## 8. Discovered features (novel, mined from runtime internals)

These are features NOT implied by the user spec and NOT in the earlier draft.
They surface real, currently-untapped data the project already produces.

### 8.1 Engine-health telemetry (from `DSTORAGE_MOE_*` log block)

On graceful shutdown (SIGINT) the server dumps a rich block in
`src/llama-dstorage-slots.cpp` (`llama_dstorage_summary_enabled`, ~line 5604).
It exposes the MoE cache/I-O engine's real behaviour, which no UI shows today.
Its L1/L2 hit rates, IO-shape speeds and stream totals are exactly what feed
running-metrics #3, #4, #6, #7, #8 in §4 — see that table for the live-vs-exit
placement decision.

- **L2 pinned-cache hit rate** — `DSTORAGE_MOE_PHASE ... decode_hit_rate`,
  `prefill_hit_rate`, `hit_rate`. Colour gauge: >80% green, 50–80% amber,
  <50% red. On the hardware run: decode_hit_rate 0.5486 (amber).
- **L1 vs L2 split (bottleneck flag)** — `DSTORAGE_MOE_STREAM` distinguishes
  `decode_cold_experts` (served from NVMe) vs `decode_pinned_experts` (L2) vs
  `prefetch_*`. If NVMe-fetched experts dominate decode, speed suffers. Live
  example: decode_cold_experts=60 vs decode_pinned_experts=10.
- **Decode I/O efficiency** — `DSTORAGE_MOE_IO_SHAPE phase=decode`:
  `effective_mib_s` (achieved NVMe throughput, ~1900–5700 MiB/s),
  `avg_request_kib`, `contiguous_ratio` (fragmentation). Shows whether GDS is
  transferring cleanly or thrashing.
- **Cache-policy admission** — `DSTORAGE_MOE_PINNED ... evict_to_ram_
  {skipped_cold, skipped_budget, skipped_failed}` and `DSTORAGE_MOE_EVICT`.
  `skipped_failed != 0` is a red flag worth surfacing.
- **Dispatcher latency split** — `DSTORAGE_ENSURE_PROFILE` per step
  (`preload_static_pinned`, `pinned_host_to_cuda_batch`,
  `classify_cache_hits_misses`) shows where the ensure/pin path spends time.

Note: these print only on a clean exit (the progress doc's "zero-counters at
STARTUP are misleading" warning applies). So the honest placement is a
**"since server start" / post-run summary panel**, plus a live partial view
parsed from the running log tail where the counters are flushed.

### 8.2 Speculation health (live, mid-run)

`slot print_timing:` lines stream live for each request and already carry:
`draft acceptance = 0.66667 (24/36)`, `prompt eval time = ... tokens per second`,
`eval time = ... tokens per second`. `monitor.parse_log_stats()` already parses
these but the TUI never renders them.

- **Draft-acceptance gauge** per running model (66% vs 84% for MTP is
  diagnosably different; a sudden drop = MTP degrading).
- **Acceptance traffic light overlayed on the decode t/s**, so a speed change
  is explained by speculation health on the same screen.

### 8.3 VRAM budget planner (prevent, not just stop-time guard)

`Profile.validate()` already computes the fit heuristic
`MOE_GPU_CACHE_MIB + ~2.4 dense + CTX-derived-KV ≤ 6 GiB`. Novel dashboard
behaviour: flag profiles that **won't fit** the GPU with an "⚠ won't fit"
chip on the model card *before* you click Start, turning a stop-time refusal
into a prevent-time warning. Side-by-side "budget vs available VRAM" bar.

### 8.4 Semantic dashboard ("why does this number matter")

- **Tooltips / inline hints** sourced from `KNOB_LABELS` and doctor check
  detail, so each cell explains its significance (e.g. "high decode
  cold_experts ⇒ slow decode — raise MOE_GPU_CACHE_MIB / L1").
- **One-glance system verdict**: combine current GPU util + VRAM free +
  NVMe temp + hit rates into a single `healthy / throttling / idle` appliance
  style verdict, not just a table of raw numbers.

---

## 9. Data-source reality check for new features

| Metric | Live mid-run? | Source |
|---|---|---|
| decode/prefill t/s, requests | Yes | `/metrics` (monitor.py), `print_timing` logs |
| draft acceptance | Yes | `slot print_timing:` log line (streams) |
| slots / context usage | Yes | `/slots` (data.py `fetch_slots`) |
| model config / paths | Yes | `Profile.resolve()`, `KNOB_LABELS` |
| suite budget / avoidable-mib | Yes (static) | `suites.suite_info()` |
| DSTORAGE_MOE hit rates / IO shape | Exit only | end-of-run log block |
| host telemetry (GPU/CPU/RAM/temp) | Yes | `data.fetch_sys_stats()` |

---

## 10. Open decisions

---

## 11. Build log — console v2 scaffold (2026-09-03)

Translated from the `siphon-console.html` OpenDesign mockup (v2 premium
ops-console) into a runnable Textual app. New package `siphon-studio/siphon/
console/` lives **alongside** the legacy `siphon/tui/` — nothing existing
was touched. Entry: `siphon-studio/bin/siphon-console`.

- `console/dstore.py` — log-tail parser for the 10 metrics: live
  `print_timing` (decode/prefill/draft) + guarded DSTORAGE_MOE parsing
  (hit rates, IO shape, stream totals, call counts; stays `None` until the
  startup-zero-counters guard passes). Verified against the real
  `qwen3.6-moe-tq3.log`: decode 21.85, draft 66.7%, L1 54.9%, L2 14.3%,
  NVMe 5743 MiB/s, RAM→VRAM 11158 MiB/s, streamed 1715 MiB, 36/655 calls.
- `console/console.tcss` — v2 tokens (ink `#0B0F19`, surface `#11172A`,
  indigo `#6366F1`, status dots), Textual 8-safe CSS.
- `console/app.py` — sidebar (brand/nav/env) + topbar + 5 KPI cards with
  `Sparkline` histories + roster with VRAM-fit badges + editable config
  form (`Input`s, L1/L2 −/+ steppers — terminal has no slider widget,
  `RadioSet` speculator, Apply/Revert with dirty tracking, auto-clone for
  bundled profiles) + 10 metric tiles (standby zeros when idle, per §4) +
  activity timeline + mono tail strip. Keys: `s`/`x`/`ctrl+r`/`a`/`u`/`q`.
- Pilot-verified headless: all widgets render, selection follows roster,
  stepper 2800→2864 marks dirty, revert clears, tiles populate. No server
  started during tests. SVG-screenshot review caught and fixed 3 real
  layout bugs: collapsed title block (bare `Vertical`→`width:1fr`+`min-width`),
  greedy env-pill (`width:auto`), and stretching panel-head rows + roster
  `ListView` (all bare `Horizontal`/`Vertical` containers now pin
  `height:auto`; roster uses fixed height 12). Lesson: Textual containers
  default to `1fr` — every bare container in a scroll view needs pinning.
- Not yet seen live: RUNNING-state visuals (green chips, live tiles) need a
  real server up — press `s` in the console to verify end to end.

---

## 12. Build log — term v4 native TUI (2026-09-04)

User verdict: v2 console "looks good for a desktop app, not a terminal TUI".
So the approved terminal-native mockup (`siphon-tui-v4.html`, after 8 design
iterations incl. a `tui-design` skill install into opencode) was implemented
natively as `siphon-studio/siphon/term/` + `bin/siphon-term` (global command
`siphon-term`). Legacy `siphon/tui/` and v2 `siphon/console/` untouched.

- `term/app.py` — header/clock, 5-block sysbar (util+bar+cap+spark per
  device, HOT tag ≥90C), model `DataTable` (NAME/STATE/RATE/VRAM-FIT),
  definition-list detail, all **31 knobs** (7 essentials + 6 `Collapsible`
  sections), `Input`/`Select` editors with −/− dirty tracking + per-section
  `(N*)` badges, Apply (integer validation, auto-clone for bundled),
  Revert, Reset-to-saved, Start/Stop(modal confirm)/Refresh, 10 tiles
  (bars for shares, `Sparkline`s for rates/counters), 5-line log tail,
  key-hint footer. Reuses `core/*`, `tui/data`, `console/dstore` — the
  dstore parser was verified earlier against the real server log.
- `term/term.tcss` — v4 tokens (ink `#0A0E14`, indigo `#7C83F5`), Textual
  8-safe. Same lesson re-applied: every bare container pins `height:auto`.
- Textual API findings: `Collapsible` toggles on Enter, NOT on mouse click;
  state messages are `Collapsed`/`Expanded` (no `Toggled` event fires);
  `Collapsible.title` is settable and drives the ▼/▶ + badge labels.
- Pilot-verified headless: 4 table rows, expand/symbol/badge, dirty→revert,
  reset-to-saved, validation rejection, all 10 tiles, live sysbar
  (GPU 50C, CPU 98C + HOT flag observed from real host telemetry).
  No server started during tests — same `s`-to-verify caveat as §11.

1. **Scope**: replace the old TUI entirely, or build the new dashboard app
   alongside a selectable entry point.
2. **First build**: dashboard landing + persistent system bar + model grid +
   config detail + Start/Stop/Refresh (core), or the full feature set including
   log tail / config editor / clone-to-app.
3. **Config editing**: view-only first pass, or allow inline editing of app
   profiles immediately (bundled profiles stay read-only).
4. **DSTORAGE telemetry scope**: post-run "since start" summary only, or also
   the live partial counters parsed from the running log tail (accepting the
   startup-zero-counters caveat).