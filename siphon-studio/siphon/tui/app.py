"""SiphonTUI — Single-Screen Control Center.

Fully functional terminal dashboard replicating siphon-tui-polished.html:
  1. Header Bar: Branding + live ticking clock & calendar date.
  2. System Strip: 3-row, 5-block hardware monitoring (GPU, VRAM, CPU, RAM, NVMe)
     with live telemetry, multi-segment bars, and dynamic sparklines.
  3. Models Table (Left Column): Full model roster with live states, decode rates,
     VRAM-fit bars, and periwinkle active row highlight.
  4. Redesigned Detail Panel (Right Column):
     - Distinct Status Box Card: state, latency, endpoint, suite.
     - Continuous Scrollable Config: Category headers + 2-column packed numeric
       fields + full-width paths. 31 settings accessible via scroll/mouse wheel.
     - Pinned Actions Footer: [Start], [x stop] (2-stage confirm), [apply],
       [u revert], [0 reset], and live dirty status indicator.
  5. Metrics Panel: 2x5 grid (10 tiles) with animated sparklines, rate deltas, and bars.
  6. Log Panel: Last 4 siphond logs with level badges and scroll icon.
  7. Footer: Highlighted primary action badge and hotkey reference.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from rich.text import Text

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Input, Static

from siphon.core import runner
from siphon.core.profiles import Profile, discover
from siphon.tui.data import fetch_sys_stats, SysStats


SPARK_CHARS = "▁▂▃▄▅▆▇█"

STYLE2_LOGO_TEXT = (
    " ▄▀▀▀█ █ █▀▀▀█ █   █ █▀▀▀█ █▀▀▀▄ \n"
    " ▀▀▀▄▄ █ █▀▀▀▀ █▀▀▀█ █   █ █   █ \n"
    " █▄▄▄▀ █ █     █   █ █▄▄▄█ █   █ "
)

SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/new", "Start a new chat session and return to welcome page"),
    ("/clear", "Reset conversation history and clear context"),
    ("/think", "Set thinking budget: off (0), low (1k), mid (4k), max"),
    ("/start", "Launch active model engine in background"),
    ("/model", "Inspect model profile parameters and context limit"),
    ("/status", "View engine orchestrator status & hardware"),
    ("/help", "Show studio commands and keyboard shortcuts"),
]

THINKING_OPTIONS: list[tuple[str, str, str]] = [
    ("off", "0 tokens", "Direct response without internal thinking"),
    ("1k", "1,024 tokens", "Lightweight reasoning for quick tasks"),
    ("4k", "4,096 tokens", "Deep reasoning for complex logic & code"),
    ("max", "Unlimited", "Full cognitive capacity (model limit)"),
]


def filter_commands(val: str, commands: list[tuple[str, str]]) -> list[tuple[str, str]]:
    if not val.startswith("/"):
        return []
    query = val[1:].strip().lower()
    if not query:
        return list(commands)
    prefix_matches = [c for c in commands if c[0].lstrip("/").lower().startswith(query)]
    if prefix_matches:
        return prefix_matches
    name_substr = [c for c in commands if query in c[0].lstrip("/").lower()]
    if name_substr:
        return name_substr
    return [c for c in commands if query in c[1].lower()]





# ─── config layout definition ──────────────────────────────────────────────────
CFG_SECTIONS: list[tuple[str, list[tuple[str, str, str, str, bool]]]] = [
    ("CORE", [
        ("path", "MODEL", "full", "path", False),
        ("ctx", "CTX", "half", "ctx", False),
        ("batch", "BATCH", "half", "batch", False),
        ("port", "PORT", "half", "port", False),
        ("speculator", "SPEC_TYPE", "half", "spec", False),
        ("l1_cache", "MOE_GPU_CACHE_MIB", "half", "l1_cache", False),
        ("l2_pinned", "MOE_PINNED_CACHE_MIB", "half", "l2_pinned", False),
    ]),
    ("IDENTITY", [
        ("alias", "ALIAS", "full", "alias", False),
        ("host", "HOST", "half", "host", False),
        ("metrics", "METRICS", "half", "metrics", False),
    ]),
    ("CONTEXT", [
        ("n_parallel", "N_PARALLEL", "half", "slots", True),
        ("ckpt_step", "CKPT_STEP", "half", "ckpt_step", False),
        ("ctx_ckpt", "CTX_CKPT", "full", "ctx_ckpt", False),
    ]),
    ("CACHE & EXPERTS", [
        ("experts", "EXPERTS", "full", "expert suite", False),
        ("cache_ram", "CACHE_RAM", "half", "prompt cache", False),
        ("evict_ram", "EVICT_RAM", "half", "evict ram", False),
        ("cache_prompt", "CACHE_PROMPT", "half", "prompt on", False),
        ("cache_reuse", "CACHE_REUSE", "half", "cache reuse", False),
    ]),
    ("SPECULATION & DRAFT", [
        ("draft_model", "DRAFT_MODEL", "full", "draft model", False),
        ("draft_nmax", "DRAFT_NMAX", "half", "draft n_max", False),
        ("draft_pmin", "DRAFT_PMIN", "half", "draft p_min", False),
        ("draft_layers", "DRAFT_LAYERS", "full", "draft layers", False),
    ]),
    ("REASONING", [
        ("reasoning", "REASONING", "half", "reasoning", False),
        ("reason_budget", "REASON_BUDGET", "half", "budget", False),
        ("reason_fmt", "REASON_FMT", "half", "reason fmt", False),
        ("chat_tmpl", "CHAT_TMPL", "half", "template", False),
        ("chat_kwargs", "CHAT_KWARGS", "full", "kwargs", False),
    ]),
    ("I/O & THREADS", [
        ("gds_threads", "GDS_THREADS", "half", "gds threads", False),
        ("pref_ws", "PREF_WS", "half", "pref ws", False),
        ("pref_lookahead", "PREF_LOOKAHEAD", "half", "lookahead", False),
        ("mmproj", "MMPROJ", "half", "mmproj", True),
    ]),
]

ALL_CFG_FIELDS: list[tuple[str, str]] = [
    (item[0], item[1])
    for _sec, fields in CFG_SECTIONS
    for item in fields
]


# Ordered load-window stages: (log marker, short label). The deepest marker
# present in the current session's log tail is the load's current stage.
_LOAD_STAGES = [
    ("llama_server: loading model", "starting"),
    ("fitting params to device", "fit to VRAM"),
    ("load_hparams", "hparams"),
    ("register_expert_tensor", "register experts"),
    ("warming up", "warmup"),
    ("MTP draft context", "MTP draft"),
    ("initializing slots", "init slots"),
    ("chat template", "chat template"),
]


def _log_ts_to_s(ts: str) -> float | None:
    """Parse engine log timestamps like '0.34.206.250' (MM.SS.mmm.uuu) to seconds."""
    parts = ts.split(".")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    while len(nums) < 4:
        nums.insert(0, 0)
    mins, secs, ms, us = nums[-4], nums[-3], nums[-2], nums[-1]
    return mins * 60.0 + secs + ms / 1000.0 + us / 1e6


def _parse_engine_telemetry(log_file: Path | None) -> dict[str, float]:
    """Parse real DSTORAGE and MoE performance stats from the server log file."""
    if not log_file or not log_file.exists():
        return {}
    try:
        size = log_file.stat().st_size
        with open(log_file, "rb") as f:
            if size > 262144:
                f.seek(size - 262144)
            text = f.read().decode("utf-8", errors="replace")
    except OSError:
        return {}

    # Isolate the current server execution session if available
    raw_text = text
    start_markers = [
        "srv         start:",
        "srv  llama_server: loading model",
        "llama_server: loading model",
        "load_model: initializing slots",
        "server is listening on",
        "llama_server: model loaded",
    ]
    last_idx = -1
    for marker in start_markers:
        pos = text.rfind(marker)
        if pos > last_idx:
            last_idx = pos
    if last_idx != -1:
        # back up to the start of the marker line so its timestamp survives
        last_idx = text.rfind("\n", 0, last_idx) + 1
        text = text[last_idx:]

    stats: dict[str, float] = {}

    # Model-load window: cut the raw tail at the most recent "loading model"
    # line (the true session start — isolation may have cut past it). If the
    # server hasn't reached "listening" in that window, it is still loading.
    lm_pos = raw_text.rfind("llama_server: loading model")
    if lm_pos != -1:
        load_text = raw_text[raw_text.rfind("\n", 0, lm_pos) + 1:]
        if "server is listening on" not in load_text and "llama_server: model loaded" not in load_text:
            m_start = re.match(r"\s*([\d.]+)\s+[IW]\s+srv\s+llama_server: loading model", load_text)
            if m_start:
                start_ts = _log_ts_to_s(m_start.group(1))
                last_ts = start_ts
                for lm in re.finditer(r"^\s*([\d.]+)\s+[IWED]\s+", load_text, re.M):
                    v = _log_ts_to_s(lm.group(1))
                    if v is not None and (last_ts is None or v > last_ts):
                        last_ts = v
                if start_ts is not None and last_ts is not None:
                    stats["load_elapsed_s"] = max(0.0, last_ts - start_ts)
                stage_idx = 0
                for idx, (marker, _) in enumerate(_LOAD_STAGES):
                    if marker in load_text:
                        stage_idx = idx
                stats["load_stage_idx"] = float(stage_idx)

    dec = re.findall(
        r"eval time =\s+[\d.]+ ms /\s+(\d+) tokens \(\s*[\d.]+ ms per token,\s+([\d.]+) tokens per second\)",
        text,
    )
    if dec:
        stats["decode_tps"] = float(dec[-1][1])

    pref = re.findall(
        r"prompt eval time =\s+[\d.]+ ms /\s+(\d+) tokens \(\s*[\d.]+ ms per token,\s+([\d.]+) tokens per second\)",
        text,
    )
    if pref:
        stats["prefill_tps"] = float(pref[-1][1])

    accept = re.findall(r"draft acceptance =\s+([\d.]+)", text)
    if accept:
        stats["draft_acc"] = float(accept[-1]) * 100.0

    hit = re.findall(r"decode_hit_rate=([\d.]+)", text)
    if hit:
        stats["l2_hit"] = float(hit[-1]) * 100.0

    gds = re.findall(r"DSTORAGE_GDS_BACKEND.*bytes_mib=([\d.]+)", text)
    if gds:
        stats["streamed_mib"] = float(gds[-1])

    nvme = re.findall(r"DSTORAGE_MOE_IO_SHAPE phase=decode.*effective_mib_s=([\d.]+)", text)
    if nvme:
        stats["nvme_speed"] = float(nvme[-1])

    ram = re.findall(r"DSTORAGE_MOE_IO_SHAPE phase=prefill.*effective_mib_s=([\d.]+)", text)
    if ram:
        stats["ram2vram_speed"] = float(ram[-1])

    timeline = re.findall(
        r"DSTORAGE_MOE_TIMELINE.*prefill_calls=(\d+).*decode_calls=(\d+).*hit_rate=([\d.]+)",
        text,
    )
    if timeline:
        stats["pre_calls"] = float(timeline[-1][0])
        stats["dec_calls"] = float(timeline[-1][1])
        stats["l1_hit"] = float(timeline[-1][2]) * 100.0
    else:
        phase = re.findall(r"decode_calls=(\d+).*prefill_calls=(\d+)", text)
        if phase:
            stats["dec_calls"] = float(phase[-1][0])
            stats["pre_calls"] = float(phase[-1][1])

    return stats


def _clean_server_log_line(raw: str) -> dict | None:
    """Filter and format a raw engine line into a clean TUI log entry."""
    s = raw.strip()
    if not s:
        return None
    skip_prefixes = ("DSTORAGE_ENSURE_PROFILE", "DSTORAGE_MOE_POOL", "DSTORAGE_MOE_SCHED")
    if any(s.startswith(sk) for sk in skip_prefixes):
        return None

    # Handle llama.cpp timestamped output
    m = re.match(r"^([\d\.]+)\s+([IWED])\s+(.*)", s)
    if m:
        t_str, lv_char, rest = m.groups()
        lv = {"I": "INFO", "W": "WARN", "E": "ERR", "D": "INFO"}.get(lv_char, "INFO")
        if any(w in rest for w in ("listening on", "all slots are idle", "HTTP 200", "registered checkpoint")):
            lv = "OK"
        elif any(w in rest for w in ("error:", "failed", "crash", "terminat")):
            lv = "ERR"
        elif any(w in rest for w in ("warning:", "override:", "near cap", "experimental")):
            lv = "WARN"

        parts = t_str.split(".")
        if len(parts) >= 3:
            ts = f"{parts[-3][-2:] if len(parts[-3]) >= 2 else parts[-3]}:{parts[-2]}:{parts[-1][:2]}"
        else:
            ts = t_str[:8]
        return {"ts": ts, "lv": lv, "msg": rest}

    lv = "INFO"
    if any(w in s for w in ("listening on", "HTTP 200", "ready")):
        lv = "OK"
    elif any(w in s for w in ("error", "ERR", "fail")):
        lv = "ERR"
    elif any(w in s for w in ("warn", "WARN")):
        lv = "WARN"

    now = time.localtime()
    ts = f"{now.tm_hour:02d}:{now.tm_min:02d}:{now.tm_sec:02d}"
    return {"ts": ts, "lv": lv, "msg": s}


@dataclass
class ProfileState:
    profile: Profile
    running: bool
    health: str
    decode_tps: float
    prefill_tps: float
    n_processing: int
    driver: str
    latency_ms: float = 0.0
    telemetry: dict[str, float] = field(default_factory=dict)


def poll_states(profiles: dict[str, Profile]) -> dict[str, ProfileState]:
    out: dict[str, ProfileState] = {}
    for name, prof in profiles.items():
        st = runner.status(prof)
        m = st.metrics
        tele = _parse_engine_telemetry(prof.log_file) if prof.log_file.exists() else {}
        out[name] = ProfileState(
            profile=prof,
            running=st.running,
            health=st.health.status,
            decode_tps=m.decode_tps or tele.get("decode_tps", 0.0),
            prefill_tps=m.prefill_tps or tele.get("prefill_tps", 0.0),
            n_processing=m.n_processing,
            driver=st.driver,
            latency_ms=st.health.latency_ms,
            telemetry=tele,
        )
    return out


def _bar_rich(pct: float, width: int = 8, on_color: str = "#3FB950", off_color: str = "#2A3550") -> Text:
    filled = max(0, min(width, round(pct / 100 * width)))
    t = Text()
    t.append("█" * filled, style=on_color)
    t.append("░" * (width - filled), style=off_color)
    return t


def _spark_text(vals: list[float], n: int = 8, color: str = "#7C83F5") -> Text:
    if not vals:
        return Text("▁" * n, style=color)
    mn, mx = min(vals), max(vals)
    s = ""
    for v in vals[-n:]:
        if mx > mn:
            idx = round((v - mn) / (mx - mn) * 7)
        elif mx == 0:
            idx = 0
        else:
            idx = 3
        idx = max(0, min(7, idx))
        s += SPARK_CHARS[idx]
    return Text(s.ljust(n, "▁"), style=color)


def _compute_widths(avail_w: int) -> list[int]:
    net_w = avail_w - 12
    base = [15, 21, 21, 24, 21]
    tot_base = sum(base)
    extra = net_w - tot_base
    if extra <= 0:
        return [max(14, b + extra // 5) for b in base]
    widths = [b + extra // 5 + (1 if i < (extra % 5) else 0) for i, b in enumerate(base)]
    return widths


def _pad_cell(left: Text, right: Text, width: int) -> Text:
    r_len = len(right.plain)
    avail = width - r_len - (1 if r_len > 0 and len(left.plain) > 0 else 0)
    l_str = left.plain
    if len(l_str) > avail and avail > 3:
        cut = max(0, avail - 1)
        l_text = Text(l_str[:cut] + "…", style=left.style)
    else:
        l_text = left
    pad = max(0, width - len(l_text.plain) - r_len)
    res = Text()
    res.append_text(l_text)
    res.append(" " * pad)
    res.append_text(right)
    return res


def _make_section_divider(title: str) -> Text:
    t = Text()
    t.append("── ", style="#232D3F")
    t.append(title, style="bold #5E6A7E")
    t.append(" " + "─" * 40, style="#232D3F")
    return t


def _pct_of(v: str) -> float:
    try:
        n = float("".join(c for c in v if c in "0123456789."))
    except Exception:
        return 0.0
    if "%" in v:
        return min(100.0, n)
    if "t/s" in v:
        return min(100.0, n / 30 * 100)
    if "MiB/s" in v:
        return min(100.0, n / 6000 * 100)
    if "MiB" in v:
        return min(100.0, n / 2000 * 100)
    return min(100.0, n / 700 * 100)


class SiphonTUI(App):
    CSS_PATH = "app.tcss"
    TITLE = "siphon — local LLM ops"
    AUTO_FOCUS = None

    BINDINGS = [
        Binding("1", "switch_dashboard", "Dashboard", show=False),
        Binding("2", "switch_chat", "Chat", show=False),
        Binding("ctrl+1", "switch_dashboard", "Dashboard", show=False),
        Binding("ctrl+2", "switch_chat", "Chat", show=False),
        Binding("f1", "switch_dashboard", "Dashboard", show=False),
        Binding("f2", "switch_chat", "Chat", show=False),
        Binding("escape", "handle_escape", "Escape", show=False),
        Binding("up", "move_up", "↑", show=False),
        Binding("down", "move_down", "↓", show=False),
        Binding("enter", "select_model", "Select", show=False),
        Binding("s", "start_model", "Start", show=False),
        Binding("x", "stop_model", "Stop", show=False),
        Binding("r", "refresh_all", "Refresh", show=False),
        Binding("a", "apply_cfg", "Apply", show=False),
        Binding("u", "revert_cfg", "Revert", show=False),
        Binding("0", "reset_cfg", "Reset", show=False),
        Binding("pageup", "scroll_log_up", "Log ▲", show=False),
        Binding("pagedown", "scroll_log_down", "Log ▼", show=False),
        Binding("tab", "cycle_focus", "Focus", priority=True, show=False),
        Binding("ctrl+t", "toggle_think_popup", "Think", show=False),
        Binding("q", "quit", "Quit", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.profiles: dict[str, Profile] = {}
        self.profile_names: list[str] = []
        self.states: dict[str, ProfileState] = {}
        self.sel: int = 0
        self._last_rendered_sel: int = -1
        self._initial_cfg: dict[str, str] = {}
        self._cfg_dirty: bool = False
        self._pending_stop: bool = False
        self._stop_arm_time: float = 0.0

        self.focus_idx: int = 0
        self._focusable_panels = ["panel-table", "panel-detail", "panel-metrics", "panel-log"]

        self._active_tab: str = "dashboard"
        self._chat_messages: list[dict] = []
        self._chat_is_generating: bool = False
        self._chat_abort: bool = False
        self._chat_ctx_tokens: int = 0
        self._chat_ctx_limit: int = 86016
        self._chat_decode_tps: float = 0.0
        self._chat_prefill_tps: float = 0.0
        self._chat_reasoning_mode: str = "max"
        self._cmd_popup_matches: list[tuple[str, str]] = []
        self._cmd_popup_idx: int = 0
        self._think_popup_open: bool = False
        self._think_popup_idx: int = 0
        self._thought_header_lines: dict[int, int] = {}

        self._sys = SysStats()
        self._sparks: dict[str, list[float]] = {
            "gpu": [20, 25, 30, 47, 45, 50, 48, 47],
            "vram": [40, 50, 55, 60, 62, 65, 64, 65],
            "cpu": [5, 8, 12, 10, 7, 9, 8, 8],
            "ram": [20, 21, 22, 22, 23, 23, 23, 23],
            "nvme": [10, 15, 12, 18, 14, 16, 15, 14],
        }

        # Metric history sparks - start clean at 0.0 (real-time telemetry will populate)
        self._metric_sparks = {
            "decode": [0.0] * 8,
            "prefill": [0.0] * 8,
            "streamed": [0.0] * 8,
            "dec_calls": [0.0] * 8,
            "pre_calls": [0.0] * 8,
            "load": [0.0] * 8,
        }

        # Live log tracking
        self._action_logs: list[dict] = [
            {"ts": "09:41:02", "lv": "OK", "msg": "siphond control center ready · engine standby"},
        ]
        self._log_scroll_offset: int = 0


    def compose(self) -> ComposeResult:
        # 1. Header Box
        with Vertical(id="header-box"):
            yield Static("", id="header-content")

        # 2. Navigation Tabs Bar
        with Vertical(id="tabs-bar"):
            yield Static("", id="tabs-content")

        # 3. System Strip Box
        with Vertical(id="sysstrip-box"):
            yield Static("", id="sysstrip-content")

        # Dashboard View
        with Vertical(id="view-dashboard"):
            # Main Split (Models Table + Detail Panel)
            with Horizontal(id="main-split"):
                # Left: Models Table
                with Vertical(id="left-col"):
                    with Vertical(id="panel-table", classes="panel focused"):
                        yield Static("", id="table-header-row")
                        yield Static("", id="model-rows")
                        yield Static(
                            Text("↑↓ move · enter select · s start · x stop", style="#5E6A7E"),
                            id="table-hint",
                        )

                # Right: Detail Panel
                with Vertical(id="right-col"):
                    with Vertical(id="panel-detail", classes="panel"):
                        # Pinned Top: Distinct enclosed Status Box Card
                        with Vertical(id="status-box"):
                            yield Static("", id="detail-dl")

                        # Middle: Continuous scrollable categorized config
                        with Vertical(id="cfg-scroll"):
                            for sec_name, fields in CFG_SECTIONS:
                                yield Static(_make_section_divider(sec_name), classes="cfg-sec-divider")
                                i = 0
                                while i < len(fields):
                                    f1 = fields[i]
                                    if f1[2] == "full":
                                        with Horizontal(classes="cfg-row-full"):
                                            yield Static(Text(f"{f1[3]:<12}", style="#5E6A7E"), classes="cfg-label-full")
                                            yield Static(Text("[", style="#5E6A7E"), classes="bracket")
                                            yield Input(id=f"cfg-{f1[0]}")
                                            yield Static(Text("]", style="#5E6A7E"), classes="bracket")
                                            if f1[4]:
                                                yield Static(Text("unused", style="#5E6A7E"), classes="unused-tag")
                                        i += 1
                                    else:
                                        if i + 1 < len(fields) and fields[i + 1][2] == "half":
                                            f2 = fields[i + 1]
                                            with Horizontal(classes="cfg-row-split"):
                                                with Horizontal(classes="cfg-half"):
                                                    yield Static(Text(f"{f1[3]:<10}", style="#5E6A7E"), classes="cfg-label-half")
                                                    yield Static(Text("[", style="#5E6A7E"), classes="bracket")
                                                    yield Input(id=f"cfg-{f1[0]}")
                                                    yield Static(Text("]", style="#5E6A7E"), classes="bracket")
                                                    if f1[4]:
                                                        yield Static(Text("unused", style="#5E6A7E"), classes="unused-tag")
                                                yield Static(" ", classes="cfg-split-gap")
                                                with Horizontal(classes="cfg-half"):
                                                    yield Static(Text(f"{f2[3]:<10}", style="#5E6A7E"), classes="cfg-label-half")
                                                    yield Static(Text("[", style="#5E6A7E"), classes="bracket")
                                                    yield Input(id=f"cfg-{f2[0]}")
                                                    yield Static(Text("]", style="#5E6A7E"), classes="bracket")
                                                    if f2[4]:
                                                        yield Static(Text("unused", style="#5E6A7E"), classes="unused-tag")
                                            i += 2
                                        else:
                                            with Horizontal(classes="cfg-row-full"):
                                                yield Static(Text(f"{f1[3]:<12}", style="#5E6A7E"), classes="cfg-label-full")
                                                yield Static(Text("[", style="#5E6A7E"), classes="bracket")
                                                yield Input(id=f"cfg-{f1[0]}")
                                                yield Static(Text("]", style="#5E6A7E"), classes="bracket")
                                                if f1[4]:
                                                    yield Static(Text("unused", style="#5E6A7E"), classes="unused-tag")
                                            i += 1

                        # Pinned Bottom: Actions & Status
                        with Vertical(classes="detail-foot"):
                            with Horizontal(id="action-row"):
                                yield Static("[Start]", id="btn-start", classes="btn btn-start", markup=False)
                                yield Static("[x stop]", id="btn-stop", classes="btn btn-stop", markup=False)
                                yield Static("[apply]", id="btn-apply", classes="btn btn-apply", markup=False)
                                yield Static("[u revert]", id="btn-revert", classes="btn btn-revert", markup=False)
                                yield Static("[0 reset]", id="btn-reset", classes="btn btn-reset", markup=False)
                            yield Static(Text("· clean — no pending edits", style="#5E6A7E"), id="cfg-status")

            # Metrics Panel
            with Vertical(id="panel-metrics", classes="panel"):
                yield Static("", id="metrics-grid")

            # Log Panel
            with Vertical(id="panel-log", classes="panel"):
                yield Static("", id="log-lines")

        # Chat View (OpenCode Terminal Interface)
        with Vertical(id="view-chat"):
            # State A: Welcome View
            with Vertical(id="chat-welcome-view"):
                with Vertical(id="chat-welcome-center-box"):
                    yield Static(STYLE2_LOGO_TEXT, id="chat-welcome-logo")
                    yield Static("", id="chat-welcome-cmd-popup", classes="chat-cmd-popup")
                    yield Static("", id="chat-welcome-think-popup", classes="chat-think-popup")
                    with Vertical(id="chat-welcome-card"):
                        yield Input(placeholder='Ask anything... "fix broken tests"', id="chat-welcome-input")
                        with Horizontal(id="chat-welcome-meta-row"):
                            yield Static("Build", id="chat-meta-badge")
                            yield Static("·", classes="chat-meta-dot")
                            yield Static("kat-coder-mtp (Q4_K_M)", id="chat-welcome-model-name")
                            yield Static("·", classes="chat-meta-dot")
                            yield Static("🧠 think: max ▾", id="chat-welcome-think-badge", classes="chat-think-badge")
                    with Horizontal(id="chat-welcome-shortcuts-row"):
                        yield Static("tab agents   ctrl+t think   ctrl+p commands", id="chat-welcome-shortcuts")
                    with Horizontal(id="chat-welcome-tip-row"):
                        yield Static("● Tip: Configure per-model parameters or use \"/help\" for commands", id="chat-welcome-tip")
                with Horizontal(id="chat-welcome-footer-row"):
                    yield Static(os.getcwd(), id="chat-welcome-foot-path")
                    yield Static("v0.2.0", id="chat-welcome-foot-ver")

            # State B: Active Stream View
            with Vertical(id="chat-stream-view"):
                with VerticalScroll(id="chat-messages-scroll"):
                    yield Static("", id="chat-messages-container")
                yield Static("", id="chat-docked-cmd-popup", classes="chat-cmd-popup")
                yield Static("", id="chat-docked-think-popup", classes="chat-think-popup")
                with Vertical(id="chat-docked-card"):
                    yield Input(placeholder='Ask anything... (Enter to send, /new for new chat)', id="chat-docked-input")
                    with Horizontal(id="chat-docked-meta-row"):
                        yield Static("Build", classes="chat-meta-badge")
                        yield Static("·", classes="chat-meta-dot")
                        yield Static("kat-coder-mtp (Q4_K_M)", id="chat-docked-model-name", classes="chat-meta-model")
                        yield Static("·", classes="chat-meta-dot")
                        yield Static("🧠 think: max ▾", id="chat-docked-think-badge", classes="chat-think-badge")
                        yield Static("tab agents", classes="chat-shortcut")
                        yield Static("ctrl+t think", classes="chat-shortcut")
                        yield Static("ctrl+p commands", classes="chat-shortcut")
                        yield Static("[esc stop]", id="chat-btn-stop", classes="chat-shortcut-stop")
                with Horizontal(id="chat-status-bar"):
                    yield Static(os.getcwd(), id="chat-status-path")
                    yield Static("ctx: 0 (0.0%)   ⚡ 0.0 t/s   🚀 0.0 t/s prefill", id="chat-status-metrics")


    def on_mount(self) -> None:
        self._load_profiles()
        self._render_table_header()
        self._render_tabs()
        self._tick()
        self.set_interval(1.0, self._tick)
        self.set_focus(None)

    def _load_profiles(self) -> None:
        try:
            self.profiles = discover()
        except Exception:
            self.profiles = {}
        order = ["qwen3.6-moe-tq3", "kat-coder-mtp", "qwen3-coder-next", "qwen3.6-dflash"]
        detected = list(self.profiles.keys())
        names = [o for o in order if o in detected]
        names += [d for d in detected if d not in names]
        if not names:
            names = ["qwen3.6-moe-tq3", "kat-coder-mtp", "qwen3-coder-next", "qwen3.6-dflash"]
        self.profile_names = names

        for i, name in enumerate(self.profile_names):
            if name in self.profiles:
                try:
                    st = runner.status(self.profiles[name])
                    if st.running:
                        self.sel = i
                        break
                except Exception:
                    pass

    def _tick(self) -> None:
        # Check stop button armed timeout (3s)
        if self._pending_stop and (time.time() - self._stop_arm_time > 3.0):
            self._pending_stop = False
            self._disarm_stop_btn()

        try:
            self._sys = fetch_sys_stats()
            self._sparks["gpu"].append(self._sys.gpu_util if self._sys.gpu_util > 0 else 47.0)
            self._sparks["vram"].append(self._sys.gpu_mib_used / 1024 if self._sys.gpu_mib_used > 0 else 3.9)
            self._sparks["cpu"].append(self._sys.cpu if self._sys.cpu > 0 else 8.0)
            self._sparks["ram"].append(self._sys.mem_pct if self._sys.mem_pct > 0 else 23.0)
            self._sparks["nvme"].append(self._sys.load if self._sys.load > 0 else 1.2)
            for k in self._sparks:
                if len(self._sparks[k]) > 8:
                    self._sparks[k] = self._sparks[k][-8:]
        except Exception:
            pass

        try:
            if self.profiles:
                self.states = poll_states(self.profiles)
        except Exception:
            pass

        self._update_header()
        self._update_sysstrip()
        self._update_table()
        self._update_detail()
        self._update_metrics()
        self._update_logs()

    def _update_header(self) -> None:
        t = Text()
        t.append("siphon ", style="bold #C5CBD6")
        t.append("— local LLM ops", style="#5E6A7E")

        now = time.localtime()
        time_str = f"{now.tm_hour:02d}:{now.tm_min:02d}:{now.tm_sec:02d}"
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        date_str = f"{months[now.tm_mon - 1]} {now.tm_mday:02d}"

        right = Text()
        right.append("local ", style="#5E6A7E")
        right.append(time_str, style="bold #C5CBD6")
        right.append(f" {date_str}", style="#5E6A7E")

        w = self.size.width - 6 if self.size.width > 20 else 114
        pad = max(2, w - len(t.plain) - len(right.plain))
        t.append(" " * pad)
        t.append_text(right)

        try:
            self.query_one("#header-content", Static).update(t)
        except Exception:
            pass

    def _update_sysstrip(self) -> None:
        s = self._sys
        gpu_util = s.gpu_util if s.gpu_util > 0 else 47.0
        gpu_temp = s.gpu_temp if s.gpu_temp > 0 else 53.0
        gpu_w = s.gpu_power_w if s.gpu_power_w > 0 else 59.0

        vram_used = (s.gpu_mib_used / 1024) if s.gpu_mib_used > 0 else 3.9
        vram_total = (s.gpu_mib_total / 1024) if s.gpu_mib_total > 0 else 6.0
        vram_free = max(0.0, vram_total - vram_used)
        vram_pct = (vram_used / vram_total * 100) if vram_total > 0 else 65.0

        cpu_util = s.cpu if s.cpu > 0 else 8.0
        cpu_temp = s.cpu_temp if s.cpu_temp > 0 else 97.0
        cpu_freq_ghz = (s.cpu_freq_mhz / 1000) if s.cpu_freq_mhz > 0 else 2.1

        ram_used = s.mem_used_gb if s.mem_used_gb > 0 else 3.5
        ram_total = s.mem_total_gb if s.mem_total_gb > 0 else 15.4
        ram_free = s.mem_free_gb if s.mem_free_gb > 0 else 11.9
        ram_pct = s.mem_pct if s.mem_pct > 0 else 23.0

        nvme_temp = s.nvme_temp if s.nvme_temp > 0 else 37.0
        load_val = s.load if s.load > 0 else 1.2

        gpu_bar_color = "#D29922" if gpu_util >= 85 else "#3FB950"
        vram_bar_color = "#D29922" if vram_pct >= 85 else "#3FB950"
        cpu_bar_color = "#D29922" if cpu_util >= 85 else "#3FB950"
        ram_bar_color = "#D29922" if ram_pct >= 85 else "#3FB950"

        avail_w = self.size.width - 6 if self.size.width > 20 else 114
        w_list = _compute_widths(avail_w)
        w_gpu, w_vram, w_cpu, w_ram, w_nvme = w_list

        spark_len = 6 if avail_w < 115 else 8

        b1_r1 = _pad_cell(Text("GPU", style="#5E6A7E"), Text(f"{gpu_temp:.0f}C", style="#C5CBD6"), w_gpu)
        b1_r2 = _pad_cell(Text(f"{gpu_util:.0f}%", style="#C5CBD6"), _bar_rich(gpu_util, 8, gpu_bar_color, "#2A3550"), w_gpu)
        b1_r3 = _pad_cell(Text(f"{gpu_w:.0f}W", style="#C5CBD6"), _spark_text(self._sparks["gpu"], spark_len), w_gpu)

        b2_r1 = _pad_cell(Text("VRAM", style="#5E6A7E"), Text(f"{vram_pct:.0f}%", style="#C5CBD6"), w_vram)
        b2_r2 = _pad_cell(_bar_rich(vram_pct, 8, vram_bar_color, "#2A3550"), _spark_text(self._sparks["vram"], spark_len), w_vram)
        vram_lbl = Text(f"{vram_used:.1f}/{vram_total:.1f}GiB", style="#C5CBD6")
        vram_lbl.append(f" ({vram_free:.1f} free)", style="#5E6A7E")
        b2_r3 = _pad_cell(vram_lbl, Text(""), w_vram)

        cpu_r1_r = Text()
        if cpu_temp >= 90:
            cpu_r1_r.append(f"{cpu_temp:.0f}C ", style="bold #F85149")
            cpu_r1_r.append("HOT", style="bold #0A0E14 on #F85149")
        else:
            cpu_r1_r.append(f"{cpu_temp:.0f}C", style="#C5CBD6")
        b3_r1 = _pad_cell(Text("CPU", style="#5E6A7E"), cpu_r1_r, w_cpu)
        b3_r2 = _pad_cell(Text(f"{cpu_util:.0f}%", style="#C5CBD6"), _bar_rich(cpu_util, 8, cpu_bar_color, "#2A3550"), w_cpu)
        b3_r3 = _pad_cell(Text(f"—/— · {cpu_freq_ghz:.1f}GHz", style="#C5CBD6"), _spark_text(self._sparks["cpu"], spark_len), w_cpu)

        b4_r1 = _pad_cell(Text("RAM", style="#5E6A7E"), Text("—", style="#5E6A7E"), w_ram)
        b4_r2 = _pad_cell(Text(f"{ram_pct:.0f}%", style="#C5CBD6"), _bar_rich(ram_pct, 8, ram_bar_color, "#2A3550"), w_ram)
        free_str = f" ({ram_free:.1f} free)"
        avail_for_ram_text = w_ram - spark_len - 1
        used_tot_str = f"{ram_used:.1f}/{ram_total:.1f}GiB"
        if len(used_tot_str) + len(free_str) > avail_for_ram_text:
            cut = max(4, avail_for_ram_text - len(free_str) - 1)
            ram_lbl = Text(used_tot_str[:cut] + "…", style="#C5CBD6")
            ram_lbl.append(free_str, style="#5E6A7E")
        else:
            ram_lbl = Text(used_tot_str, style="#C5CBD6")
            ram_lbl.append(free_str, style="#5E6A7E")
        b4_r3 = _pad_cell(ram_lbl, _spark_text(self._sparks["ram"], spark_len), w_ram)

        b5_r1 = _pad_cell(Text("NVMe", style="#5E6A7E"), Text(f"{nvme_temp:.0f}C", style="#C5CBD6"), w_nvme)
        b5_r2 = _pad_cell(Text("—", style="#5E6A7E"), _bar_rich(20, 8, "#3FB950", "#2A3550"), w_nvme)
        b5_r3 = _pad_cell(Text(f"—/— · load {load_val:.1f}", style="#C5CBD6"), _spark_text(self._sparks["nvme"], spark_len), w_nvme)

        sep = Text(" │ ", style="#2A3550")
        out = Text()
        out.append_text(b1_r1).append_text(sep).append_text(b2_r1).append_text(sep).append_text(b3_r1).append_text(sep).append_text(b4_r1).append_text(sep).append_text(b5_r1).append("\n")
        out.append_text(b1_r2).append_text(sep).append_text(b2_r2).append_text(sep).append_text(b3_r2).append_text(sep).append_text(b4_r2).append_text(sep).append_text(b5_r2).append("\n")
        out.append_text(b1_r3).append_text(sep).append_text(b2_r3).append_text(sep).append_text(b3_r3).append_text(sep).append_text(b4_r3).append_text(sep).append_text(b5_r3)

        try:
            self.query_one("#sysstrip-content", Static).update(out)
        except Exception:
            pass

    # ── model table ───────────────────────────────────────────────────────
    def _render_table_header(self) -> None:
        t = Text()
        t.append(f"{'NAME':<20} ", style="#5E6A7E")
        t.append(f"{'STATE':<10}", style="#5E6A7E")
        t.append(f"{'RATE':>9} ", style="#5E6A7E")
        t.append(f"{'VRAM-FIT':>18}", style="#5E6A7E")
        try:
            self.query_one("#table-header-row", Static).update(t)
        except Exception:
            pass

    def _estimate_fit(self, name: str) -> float:
        fits = {
            "qwen3.6-moe-tq3": 57.0,
            "kat-coder-mtp": 68.0,
            "qwen3-coder-next": 47.0,
            "qwen3-coder-next-draft": 32.0,
            "qwen3.6-dflash": 32.0,
        }
        if name in fits:
            return fits[name]
        prof = self.profiles.get(name)
        if not prof:
            return 50.0
        try:
            l1 = float(prof.resolve("MOE_GPU_CACHE_MIB") or "0")
            if l1 > 0:
                return min(100.0, (l1 + 1000) / 6141.0 * 100.0)
        except Exception:
            pass
        return 50.0

    def _update_table(self) -> None:
        try:
            self.query_one("#panel-table", Vertical).border_title = f" models · {len(self.profile_names)} "
        except Exception:
            pass

        rows = Text()
        for i, name in enumerate(self.profile_names):
            st = self.states.get(name)
            is_sel = (i == self.sel)
            disp_name = name if len(name) <= 20 else (name[:19] + "…")

            state = "IDLE"
            rate = "standby"
            if st and st.running:
                if st.health == "loading":
                    state = "LOADING"
                    elapsed = st.telemetry.get("load_elapsed_s", 0.0)
                    rate = f"load {elapsed:.0f}s"
                else:
                    state = "RUNNING"
                    rate = f"{st.decode_tps:.1f}t/s" if st.decode_tps > 0 else "idle"

            fit_pct = self._estimate_fit(name)
            filled = round(fit_pct / 100 * 10)
            empty = 10 - filled
            fit_label = f"Fits {fit_pct:.0f}%"

            row = Text()
            if is_sel:
                bg = "on #7C83F5"
                row.append(f"{disp_name:<20} ", style=f"bold #0A0E14 {bg}")
                row.append(f"{state:<10}", style=f"bold #0A0E14 {bg}")
                row.append(f"{rate:>9} ", style=f"#0A0E14 {bg}")
                row.append("█" * filled, style=f"#0A0E14 {bg}")
                row.append("░" * empty, style=f"#4D53B5 {bg}")
                row.append(f" {fit_label:>7}", style=f"#0A0E14 {bg}")
            else:
                if state == "LOADING":
                    state_style, rate_style = "bold #D29922", "#D29922"
                elif state == "RUNNING":
                    state_style, rate_style = "bold #3FB950", "#C5CBD6"
                else:
                    state_style, rate_style = "#5E6A7E", "#5E6A7E"
                row.append(f"{disp_name:<20} ", style="#C5CBD6")
                row.append(f"{state:<10}", style=state_style)
                row.append(f"{rate:>9} ", style=rate_style)
                row.append("█" * filled, style="#5E6A7E")
                row.append("░" * empty, style="#232D3F")
                row.append(f" {fit_label:>7}", style="#5E6A7E")

            if i < len(self.profile_names) - 1:
                row.append("\n")
            rows.append_text(row)

        try:
            self.query_one("#model-rows", Static).update(rows)
        except Exception:
            pass

    # ── detail panel ──────────────────────────────────────────────────────
    def _update_detail(self) -> None:
        if not self.profile_names or self.sel >= len(self.profile_names):
            return
        name = self.profile_names[self.sel]
        prof = self.profiles.get(name)
        st = self.states.get(name)

        try:
            self.query_one("#panel-detail", Vertical).border_title = f" detail · {name} "
        except Exception:
            pass

        # Status definition list inside distinct card
        is_running = st and st.running
        is_loading = bool(st and st.running and st.health == "loading")
        if is_loading:
            state_str = "LOADING"
            state_color = "bold #D29922"
        elif is_running:
            state_str = "RUNNING"
            state_color = "bold #3FB950"
        else:
            state_str = "IDLE"
            state_color = "#5E6A7E"

        lat_val = f"{st.latency_ms:.0f}ms" if (st and is_running and st.latency_ms > 0) else "—"
        port = prof.env.get("PORT", "8090") if prof else "8090"
        ep_val = f":{port}/v1" if is_running else "—"

        suite_val = "—"
        if prof and "EXPERTS" in prof.env:
            exp = prof.env.get("EXPERTS", "")
            if exp:
                suite_val = exp.split("/")[-1].replace(".txt", "")

        t = Text()
        t.append(f"{'state':<8}", style="#5E6A7E")
        t.append(" . . . . . . . . . . . . . . ", style="#26314A")
        t.append(f"{state_str:>7}\n", style=state_color)

        t.append(f"{'latency':<8}", style="#5E6A7E")
        t.append(" . . . . . . . . . . . . . . . . ", style="#26314A")
        t.append(f"{lat_val:>5}\n", style="#C5CBD6")

        t.append(f"{'endpoint':<9}", style="#5E6A7E")
        t.append(" . . . . . . . . . . . . . ", style="#26314A")
        t.append(f"{ep_val:>8}\n", style="#C5CBD6")

        suite_disp = suite_val if len(suite_val) <= 29 else suite_val[:28] + "…"
        t.append(f"{'suite':<6}", style="#5E6A7E")
        t.append(" . . . . ", style="#26314A")
        t.append(f"{suite_disp}\n", style="#C5CBD6")

        try:
            self.query_one("#detail-dl", Static).update(t)
        except Exception:
            pass

        if self.sel != self._last_rendered_sel and not self._cfg_dirty:
            self._populate_cfg(prof)

    def _populate_cfg(self, prof: Profile | None) -> None:
        self._last_rendered_sel = self.sel
        self._initial_cfg.clear()

        for key, env_var in ALL_CFG_FIELDS:
            val = ""
            if prof:
                val = str(prof.env.get(env_var, ""))
            elif key == "path":
                val = "models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
            elif key == "ctx":
                val = "65536"
            elif key == "batch":
                val = "512"
            elif key == "port":
                val = "8090"
            elif key == "speculator":
                val = "mtp"
            elif key == "l1_cache":
                val = "2800MiB"
            elif key == "l2_pinned":
                val = "8192MiB"
            elif key == "alias":
                val = self.profile_names[self.sel] if self.sel < len(self.profile_names) else ""
            elif key == "host":
                val = "127.0.0.1"
            elif key == "metrics":
                val = "1"
            elif key == "evict_ram":
                val = "1"
            elif key == "reasoning":
                val = "off"
            elif key == "reason_fmt":
                val = "deepseek"
            elif key == "reason_budget":
                val = "-1"
            elif key == "chat_kwargs":
                val = '{"preserve_thinking": true}'
            elif key == "gds_threads":
                val = "20"
            elif key == "pref_lookahead":
                val = "2"

            self._initial_cfg[key] = val
            try:
                inp = self.query_one(f"#cfg-{key}", Input)
                inp.value = val
            except Exception:
                pass

        self._cfg_dirty = False
        try:
            self.query_one("#cfg-status", Static).update(
                Text("· clean — no pending edits", style="#5E6A7E")
            )
        except Exception:
            pass

    def on_input_changed(self, event: Input.Changed) -> None:
        wid = getattr(event.input, "id", "")
        if wid in ("chat-welcome-input", "chat-docked-input"):
            self._update_cmd_popup_for_input(wid, event.value)
            return
        if not wid or not wid.startswith("cfg-"):
            return
        dirty = any(
            self.query_one(f"#cfg-{k}", Input).value != self._initial_cfg.get(k, "")
            for k, _ in ALL_CFG_FIELDS
            if self.query(f"#cfg-{k}")
        )
        self._cfg_dirty = dirty
        if dirty:
            text = "· edited — press a to apply, u to revert"
            color = "#D29922"
        else:
            text = "· clean — no pending edits"
            color = "#5E6A7E"
        try:
            self.query_one("#cfg-status", Static).update(Text(text, style=color))
        except Exception:
            pass

    # ── metrics panel ─────────────────────────────────────────────────────
    def _update_metrics(self) -> None:
        is_loading = bool(
            self.states
            and self.sel < len(self.profile_names)
            and self.states.get(self.profile_names[self.sel])
            and self.states[self.profile_names[self.sel]].running
            and self.states[self.profile_names[self.sel]].health == "loading"
        )
        try:
            self.query_one("#panel-metrics", Vertical).border_title = (
                " metrics · loading " if is_loading else " metrics · live "
            )
        except Exception:
            pass

        if not self.profile_names or self.sel >= len(self.profile_names):
            return
        name = self.profile_names[self.sel]
        prof = self.profiles.get(name)
        st = self.states.get(name)
        is_live = bool(st and st.running)

        # Parse live engine telemetry from model log file
        tele = st.telemetry if (st and st.telemetry) else (_parse_engine_telemetry(prof.log_file) if prof else {})

        if is_loading:
            # Real load progress from the engine log: elapsed seconds and the
            # deepest loader stage marker seen so far.
            elapsed = tele.get("load_elapsed_s", 0.0)
            stage_idx = int(tele.get("load_stage_idx", 0.0))
            stage_lbl = _LOAD_STAGES[stage_idx][1] if 0 <= stage_idx < len(_LOAD_STAGES) else "loading"
            stage_pct = (stage_idx + 1) / len(_LOAD_STAGES) * 100.0

            self._metric_sparks["load"].append(elapsed)
            if len(self._metric_sparks["load"]) > 8:
                self._metric_sparks["load"] = self._metric_sparks["load"][-8:]

            d = {
                "load": f"{elapsed:.0f}s",
                "stage": stage_lbl,
                "l1_hit": "—",
                "l2_hit": "—",
                "draft_acc": "—",
                "nvme": "0MiB/s",
                "ram2vram": "0MiB/s",
                "streamed": "0MiB",
                "dec_calls": "0",
                "pre_calls": "0",
            }
            sub_dec = "loading model"
            sub_pre = "in progress"
            sub_str = "— loading"
            sub_dcall = "— loading"
            sub_pcall = "— loading"
        elif is_live:
            dec_num = st.decode_tps if (st and st.decode_tps > 0) else tele.get("decode_tps", 0.0)
            pref_num = st.prefill_tps if (st and st.prefill_tps > 0) else tele.get("prefill_tps", 0.0)
            
            l1_val = tele.get("l1_hit")
            l2_val = tele.get("l2_hit")
            draft_val = tele.get("draft_acc")
            
            nvme_num = tele.get("nvme_speed", 0.0)
            ram_num = tele.get("ram2vram_speed", 0.0)
            str_num = tele.get("streamed_mib", 0.0)
            
            dcalls_num = int(tele.get("dec_calls", st.n_processing or 0))
            pcalls_num = int(tele.get("pre_calls", 0))

            # Push live rolling sparks
            self._metric_sparks["decode"].append(dec_num)
            self._metric_sparks["prefill"].append(pref_num)
            self._metric_sparks["streamed"].append(str_num)
            self._metric_sparks["dec_calls"].append(float(dcalls_num))
            self._metric_sparks["pre_calls"].append(float(pcalls_num))
            for k in self._metric_sparks:
                if len(self._metric_sparks[k]) > 8:
                    self._metric_sparks[k] = self._metric_sparks[k][-8:]

            # Compute delta
            prev_dec = self._metric_sparks["decode"][-2] if len(self._metric_sparks["decode"]) >= 2 else dec_num
            delta_dec = dec_num - prev_dec
            if dec_num > 0:
                delta_str = f"▲ +{delta_dec:.1f} vs last req" if delta_dec >= 0 else f"▼ {delta_dec:.1f} vs last req"
            else:
                delta_str = "idle · 0 req" if dcalls_num == 0 else "idle"

            # Streamed sub
            prev_str = self._metric_sparks["streamed"][-2] if len(self._metric_sparks["streamed"]) >= 2 else str_num
            delta_str_val = str_num - prev_str
            if str_num > 0:
                sub_str = f"+{delta_str_val:.0f} MiB last req" if delta_str_val > 0 else f"{str_num:.0f} MiB streamed"
            else:
                sub_str = "0 MiB idle"

            d = {
                "decode": f"{dec_num:.1f}t/s",
                "prefill": f"{pref_num:.1f}t/s",
                "l1_hit": f"{l1_val:.0f}%" if l1_val is not None else "—",
                "l2_hit": f"{l2_val:.0f}%" if l2_val is not None else "—",
                "draft_acc": f"{draft_val:.0f}%" if draft_val is not None else "—",
                "nvme": f"{nvme_num:.0f}MiB/s",
                "ram2vram": f"{ram_num:.0f}MiB/s",
                "streamed": f"{str_num:.0f}MiB",
                "dec_calls": f"{dcalls_num}",
                "pre_calls": f"{pcalls_num}",
            }
            sub_dec = delta_str
            sub_pre = "live engine" if pref_num > 0 else "idle"
            sub_dcall = f"{dcalls_num} calls" if dcalls_num > 0 else "0 calls"
            sub_pcall = f"{pcalls_num} calls" if pcalls_num > 0 else "0 calls"
        else:
            d = {
                "decode": "0.0t/s",
                "prefill": "0.0t/s",
                "l1_hit": "—",
                "l2_hit": "—",
                "draft_acc": "—",
                "nvme": "0MiB/s",
                "ram2vram": "0MiB/s",
                "streamed": "0MiB",
                "dec_calls": "0",
                "pre_calls": "0",
            }
            sub_dec = "— standby"
            sub_pre = "— standby"
            sub_str = "— standby"
            sub_dcall = "— standby"
            sub_pcall = "— standby"

        gap = "   "

        def m_spark(label: str, val: str, spark_key: str, sub: str) -> Text:
            t = Text()
            style_v = "#C5CBD6" if is_live else "#5E6A7E"
            style_sp = "#7C83F5" if is_live else "#2C3650"
            style_sub = "#5E6A7E" if is_live else "#2C3650"
            t.append(f"{label:<10}", style="#5E6A7E")
            t.append(f"{val:>10} ", style=style_v)
            t.append_text(_spark_text(self._metric_sparks[spark_key], 8, style_sp))
            t.append(f" {sub:<20}", style=style_sub)
            return t

        def m_bar(label: str, val: str) -> Text:
            t = Text()
            pct = _pct_of(val) if is_live else 0
            style_v = "#C5CBD6" if is_live else "#5E6A7E"
            style_bar = "#7C83F5" if is_live else "#2C3650"
            t.append(f"{label:<10}", style="#5E6A7E")
            t.append(f"{val:>10} ", style=style_v)
            t.append_text(_bar_rich(pct, 12, style_bar, "#232D3F"))
            t.append(" " * 17)
            return t

        # Build 5 rows of 2 tiles
        if is_loading:
            def m_stage(label: str, val: str, sub: str) -> Text:
                t = Text()
                t.append(f"{label:<10}", style="#5E6A7E")
                t.append(f"{val:>10} ", style="#D29922")
                t.append_text(_bar_rich(stage_pct, 12, "#D29922", "#232D3F"))
                t.append(f" {sub:<20}", style="#5E6A7E")
                return t

            r1 = (
                Text()
                .append_text(m_spark("load", d["load"], "load", sub_dec))
                .append(gap)
                .append_text(m_stage("stage", d["stage"], f"step {stage_idx + 1}/{len(_LOAD_STAGES)}"))
            )
            r2 = Text().append_text(m_bar("l1_hit", d["l1_hit"])).append(gap).append_text(m_bar("l2_hit", d["l2_hit"]))
            r3 = Text().append_text(m_bar("draft_acc", d["draft_acc"])).append(gap).append_text(m_bar("nvme", d["nvme"]))
            r4 = Text().append_text(m_bar("ram2vram", d["ram2vram"])).append(gap).append_text(m_spark("streamed", d["streamed"], "streamed", sub_str))
            r5 = Text().append_text(m_spark("dec_calls", d["dec_calls"], "dec_calls", sub_dcall)).append(gap).append_text(m_spark("pre_calls", d["pre_calls"], "pre_calls", sub_pcall))
        else:
            r1 = Text().append_text(m_spark("decode", d["decode"], "decode", sub_dec)).append(gap).append_text(m_spark("prefill", d["prefill"], "prefill", sub_pre))
            r2 = Text().append_text(m_bar("l1_hit", d["l1_hit"])).append(gap).append_text(m_bar("l2_hit", d["l2_hit"]))
            r3 = Text().append_text(m_bar("draft_acc", d["draft_acc"])).append(gap).append_text(m_bar("nvme", d["nvme"]))
            r4 = Text().append_text(m_bar("ram2vram", d["ram2vram"])).append(gap).append_text(m_spark("streamed", d["streamed"], "streamed", sub_str))
            r5 = Text().append_text(m_spark("dec_calls", d["dec_calls"], "dec_calls", sub_dcall)).append(gap).append_text(m_spark("pre_calls", d["pre_calls"], "pre_calls", sub_pcall))

        out = Text()
        out.append_text(r1).append("\n")
        out.append_text(r2).append("\n")
        out.append_text(r3).append("\n")
        out.append_text(r4).append("\n")
        out.append_text(r5)

        try:
            self.query_one("#metrics-grid", Static).update(out)
        except Exception:
            pass

    # ── log panel ─────────────────────────────────────────────────────────
    def _update_logs(self) -> None:
        if not self.profile_names or self.sel >= len(self.profile_names):
            return
        name = self.profile_names[self.sel]
        prof = self.profiles.get(name)

        # Read actual server log lines from disk
        server_logs: list[dict] = []
        if prof and prof.log_file.exists():
            try:
                size = prof.log_file.stat().st_size
                with open(prof.log_file, "rb") as f:
                    if size > 65536:
                        f.seek(size - 65536)
                    content = f.read().decode("utf-8", errors="replace")

                # If server is running, slice from the start of the current session
                st = self.states.get(name)
                if st and st.running:
                    start_markers = [
                        "srv         start:",
                        "srv  llama_server: loading model",
                        "llama_server: loading model",
                        "load_model: initializing slots",
                        "HTTP server is listening",
                        "main: model loaded",
                    ]
                    last_idx = -1
                    for marker in start_markers:
                        pos = content.rfind(marker)
                        if pos > last_idx:
                            last_idx = pos
                    if last_idx != -1:
                        content = content[last_idx:]

                lines = content.splitlines()
                for line in lines[-50:]:
                    cleaned = _clean_server_log_line(line)
                    if cleaned:
                        server_logs.append(cleaned)
            except Exception:
                pass

        # Combine with recent interactive user events
        combined = list(server_logs)
        for act in self._action_logs[-15:]:
            combined.append(act)

        if not combined:
            now = time.localtime()
            ts = f"{now.tm_hour:02d}:{now.tm_min:02d}:{now.tm_sec:02d}"
            combined = [{"ts": ts, "lv": "INFO", "msg": f"{name} standby — press s to start"}]

        # Apply scroll offset
        total = len(combined)
        max_offset = max(0, total - 4)
        offset = min(self._log_scroll_offset, max_offset)
        if offset > 0:
            display_lines = combined[-(4 + offset) : -offset]
            sub_text = f" ▲▼ -{offset} "
        else:
            display_lines = combined[-4:]
            sub_text = " ▲▼ "

        try:
            p = self.query_one("#panel-log", Vertical)
            p.border_title = f" log · {name} "
            p.border_subtitle = sub_text
        except Exception:
            pass

        t = Text()
        for item in display_lines:
            ts = item["ts"]
            lv = item["lv"]
            msg = item["msg"]

            t.append(f"{ts:<8} ", style="#5E6A7E")
            if lv == "OK":
                t.append("OK   ", style="bold #3FB950")
            elif lv == "WARN":
                t.append("WARN ", style="bold #D29922")
            elif lv == "ERR":
                t.append("ERR  ", style="bold #F85149")
            else:
                t.append("INFO ", style="#5E6A7E")

            t.append("siphond · ", style="#5E6A7E")
            max_msg_len = 84
            m_disp = msg if len(msg) <= max_msg_len else (msg[:max_msg_len - 1] + "…")
            t.append(f"{m_disp}\n", style="#C5CBD6")

        try:
            self.query_one("#log-lines", Static).update(t)
        except Exception:
            pass

    def _push_log(self, lv: str, msg: str) -> None:
        now = time.localtime()
        ts = f"{now.tm_hour:02d}:{now.tm_min:02d}:{now.tm_sec:02d}"
        self._action_logs.append({"ts": ts, "lv": lv, "msg": msg})
        if len(self._action_logs) > 50:
            self._action_logs = self._action_logs[-50:]
        self._log_scroll_offset = 0

    def action_scroll_log_up(self) -> None:
        self._log_scroll_offset += 2
        self._update_logs()

    def action_scroll_log_down(self) -> None:
        self._log_scroll_offset = max(0, self._log_scroll_offset - 2)
        self._update_logs()

    # ── navigation tabs & chat tab management ─────────────────────────────
    def _render_tabs(self) -> None:
        t = Text()
        if self._active_tab == "dashboard":
            t.append(" 1 Dashboard ", style="bold #0A0E14 on #7C83F5")
            t.append("  ")
            t.append(" 2 Chat ", style="#7A889B on #131923")
        else:
            t.append(" 1 Dashboard ", style="#7A889B on #131923")
            t.append("  ")
            t.append(" 2 Chat ● ", style="bold #0A0E14 on #7C83F5")
        try:
            self.query_one("#tabs-content", Static).update(t)
        except Exception:
            pass

    def action_switch_dashboard(self) -> None:
        if isinstance(self.focused, Input):
            wid = getattr(self.focused, "id", "")
            if not wid.startswith("chat-"):
                return
        self._active_tab = "dashboard"
        self._update_tab_visibility()

    def action_switch_chat(self) -> None:
        if isinstance(self.focused, Input):
            wid = getattr(self.focused, "id", "")
            if wid.startswith("cfg-"):
                return
        self._active_tab = "chat"
        self._update_tab_visibility()

    def action_handle_escape(self) -> None:
        if self._chat_is_generating:
            self.action_stop_chat()
            return
        if isinstance(self.focused, Input):
            self.set_focus(None)

    def _update_tab_visibility(self) -> None:
        self._render_tabs()
        self._update_chat_model_names()
        try:
            v_dash = self.query_one("#view-dashboard")
            v_chat = self.query_one("#view-chat")
            if self._active_tab == "dashboard":
                v_dash.styles.display = "block"
                v_chat.styles.display = "none"
                self.set_focus(None)
            else:
                v_dash.styles.display = "none"
                v_chat.styles.display = "block"
                self._update_chat_view_state()
        except Exception:
            pass

    def _update_chat_view_state(self) -> None:
        try:
            welcome = self.query_one("#chat-welcome-view")
            stream = self.query_one("#chat-stream-view")
            if not self._chat_messages:
                welcome.styles.display = "block"
                stream.styles.display = "none"
                self.set_focus(self.query_one("#chat-welcome-input", Input))
            else:
                welcome.styles.display = "none"
                stream.styles.display = "block"
                self.set_focus(self.query_one("#chat-docked-input", Input))
        except Exception:
            pass

    def _update_chat_model_names(self) -> None:
        model_name = self.profile_names[self.sel] if self.profile_names else "siphon-model"
        tag = model_name
        try:
            self.query_one("#chat-welcome-model-name", Static).update(tag)
            self.query_one("#chat-docked-model-name", Static).update(tag)
        except Exception:
            pass
        self._update_reasoning_badges()

    def action_cycle_reasoning(self, explicit_mode: str | None = None) -> None:
        modes = ["max", "off", "1k", "4k"]
        if explicit_mode:
            m = explicit_mode.lower().strip()
            alias = {
                "none": "off", "0": "off", "false": "off", "disabled": "off", "no": "off",
                "low": "1k", "1024": "1k", "1": "1k",
                "mid": "4k", "medium": "4k", "4096": "4k", "4": "4k",
                "unlimited": "max", "-1": "max", "high": "max", "on": "max", "yes": "max",
            }
            m = alias.get(m, m)
            if m in modes:
                self._chat_reasoning_mode = m
        else:
            try:
                curr_idx = modes.index(self._chat_reasoning_mode)
                self._chat_reasoning_mode = modes[(curr_idx + 1) % len(modes)]
            except ValueError:
                self._chat_reasoning_mode = "max"
        self._update_reasoning_badges()
        if self._think_popup_open:
            w_pop, d_pop = self._get_active_think_popups()
            target = w_pop if (len(self._chat_messages) == 0) else d_pop
            if target:
                target.update(self._render_think_popup_markup())

    def _update_reasoning_badges(self) -> None:
        mode = self._chat_reasoning_mode
        color_map = {
            "max": "#A78BFA",
            "4k": "#F59E0B",
            "1k": "#38BDF8",
            "off": "#64748B",
        }
        col = color_map.get(mode, "#A78BFA")
        t = Text()
        t.append("🧠 think: ", style="dim #94A3B8")
        t.append(mode, style=f"bold {col}")
        t.append(" ▾", style="dim #64748B")
        try:
            self.query_one("#chat-welcome-think-badge", Static).update(t)
            self.query_one("#chat-docked-think-badge", Static).update(t)
        except Exception:
            pass

    def _render_think_popup_markup(self) -> str:
        lines = []
        lines.append("[dim #64748B]REASONING BUDGET (↑↓ navigate, Enter select, Esc close)[/]")
        for i, (mode, budget, desc) in enumerate(THINKING_OPTIONS):
            is_active = (mode == self._chat_reasoning_mode)
            active_mark = "[bold #10B981] ✓[/]" if is_active else "  "
            if i == self._think_popup_idx:
                lines.append(f"[bold #A78BFA on #1E2538] ❯ {mode:<5} [/][bold #F8FAFC on #1E2538]{budget:<14} [/][#94A3B8 on #1E2538]{desc}[/]{active_mark}")
            else:
                lines.append(f"[#94A3B8]   {mode:<5} [/][#64748B]{budget:<14} [/][dim #64748B]{desc}[/]{active_mark}")
        return "\n".join(lines)

    def _get_active_think_popups(self) -> tuple[Static | None, Static | None]:
        try:
            w_pop = self.query_one("#chat-welcome-think-popup", Static)
            d_pop = self.query_one("#chat-docked-think-popup", Static)
            return w_pop, d_pop
        except Exception:
            return None, None

    def _show_think_popup(self) -> None:
        self._hide_cmd_popups()
        self._think_popup_open = True
        modes = [m[0] for m in THINKING_OPTIONS]
        if self._chat_reasoning_mode in modes:
            self._think_popup_idx = modes.index(self._chat_reasoning_mode)
        else:
            self._think_popup_idx = 0
        w_pop, d_pop = self._get_active_think_popups()
        markup = self._render_think_popup_markup()
        is_welcome = (len(self._chat_messages) == 0)
        if is_welcome and w_pop:
            w_pop.update(markup)
            w_pop.styles.display = "block"
            if d_pop:
                d_pop.styles.display = "none"
        elif d_pop:
            d_pop.update(markup)
            d_pop.styles.display = "block"
            if w_pop:
                w_pop.styles.display = "none"

    def _hide_think_popup(self) -> None:
        self._think_popup_open = False
        w_pop, d_pop = self._get_active_think_popups()
        if w_pop:
            w_pop.styles.display = "none"
        if d_pop:
            d_pop.styles.display = "none"

    def _toggle_think_popup(self) -> None:
        if self._think_popup_open:
            self._hide_think_popup()
        else:
            self._show_think_popup()

    def action_toggle_think_popup(self) -> None:
        self._toggle_think_popup()

    def _render_cmd_popup_markup(self) -> str:
        lines = []
        lines.append("[dim #64748B]COMMANDS (↑↓ navigate, Tab complete, Enter run)[/]")
        for i, (cmd, desc) in enumerate(self._cmd_popup_matches):
            if i == self._cmd_popup_idx:
                lines.append(f"[bold #7C83F5 on #1E2538] ❯ {cmd:<8} [/][#F8FAFC on #1E2538]{desc}[/]")
            else:
                lines.append(f"[#94A3B8]   {cmd:<8} [/][#64748B]{desc}[/]")
        return "\n".join(lines)

    def _get_active_cmd_popups(self) -> tuple[Static | None, Static | None]:
        try:
            w_pop = self.query_one("#chat-welcome-cmd-popup", Static)
            d_pop = self.query_one("#chat-docked-cmd-popup", Static)
            return w_pop, d_pop
        except Exception:
            return None, None

    def _hide_cmd_popups(self) -> None:
        self._cmd_popup_matches = []
        self._cmd_popup_idx = 0
        w_pop, d_pop = self._get_active_cmd_popups()
        if w_pop:
            w_pop.styles.display = "none"
        if d_pop:
            d_pop.styles.display = "none"

    def _update_cmd_popup_for_input(self, input_id: str, val: str) -> None:
        if self._think_popup_open:
            self._hide_think_popup()
        w_pop, d_pop = self._get_active_cmd_popups()
        if not w_pop or not d_pop:
            return
        if val.startswith("/"):
            matches = filter_commands(val, SLASH_COMMANDS)
            self._cmd_popup_matches = matches
            if matches:
                self._cmd_popup_idx = 0
                markup = self._render_cmd_popup_markup()
                if input_id == "chat-welcome-input":
                    w_pop.update(markup)
                    w_pop.styles.display = "block"
                    d_pop.styles.display = "none"
                else:
                    d_pop.update(markup)
                    d_pop.styles.display = "block"
                    w_pop.styles.display = "none"
                return
        self._hide_cmd_popups()

    def _handle_send_chat(self, text: str) -> None:
        self._hide_cmd_popups()
        self._hide_think_popup()
        raw = text.strip()
        if not raw:
            return

        # Slash commands
        if raw.startswith("/"):
            cmd = raw.lower().split()[0]
            if cmd in ("/new", "/clear"):
                self._chat_messages = []
                self._chat_ctx_tokens = 0
                self._chat_decode_tps = 0.0
                self._chat_prefill_tps = 0.0
                self._render_chat_messages()
                self._update_chat_view_state()
                self._update_chat_status_metrics()
                return
            elif cmd in ("/think", "/reason"):
                parts = raw.split()
                if len(parts) > 1:
                    target_mode = parts[1].strip().lower()
                    self.action_cycle_reasoning(explicit_mode=target_mode)
                    mode = self._chat_reasoning_mode
                    self._chat_messages.append({"role": "user", "content": raw, "ts": time.time()})
                    self._chat_messages.append({
                        "role": "assistant",
                        "thought": "Reasoning Controller",
                        "content": (
                            f"🧠 Model thinking budget set to **`{mode}`**.\n\n"
                            f"- `off`: 0 thinking tokens (instant direct response)\n"
                            f"- `1k`: 1,024 thinking token budget\n"
                            f"- `4k`: 4,096 thinking token budget\n"
                            f"- `max`: unlimited thinking tokens\n\n"
                            f"*(Tip: click `🧠 think: {mode} ▾` or press `Ctrl+T` to expand the reasoning menu)*"
                        ),
                        "decode_tps": 50.0,
                        "prefill_tps": 60.0,
                        "latency": 0.01,
                        "complete": True,
                    })
                    self._render_chat_messages()
                    self._update_chat_view_state()
                    return
                else:
                    self._show_think_popup()
                    return
            elif cmd == "/help":
                self._chat_messages.append({"role": "user", "content": "/help", "ts": time.time()})
                help_text = (
                    "**Siphon Studio Commands**:\n"
                    "- `/new` or `/clear`: Start a new chat session and return to welcome page\n"
                    "- `/think [off|1k|4k|max]`: Open reasoning menu or set token budget\n"
                    "- `/start`: Launch the active model server in the background\n"
                    "- `/model`: Inspect currently selected model profile & parameters\n"
                    "- `/status`: View engine orchestrator status & hardware summaries\n"
                    "- `/help`: Show this command summary\n"
                    "- `Ctrl+T`: Open expandable reasoning budget menu\n"
                    "- `Esc`: Stop ongoing assistant generation immediately\n"
                    "- `1` / `2`: Switch between Dashboard and Chat tabs"
                )
                self._chat_messages.append({
                    "role": "assistant",
                    "thought": "Thought: 12ms",
                    "content": help_text,
                    "decode_tps": 42.0,
                    "prefill_tps": 56.0,
                    "latency": 0.02,
                    "complete": True,
                })
                self._render_chat_messages()
                self._update_chat_view_state()
                return
            elif cmd == "/start":
                prof_name = self.profile_names[self.sel] if self.profile_names else "siphon-model"
                self._chat_messages.append({"role": "user", "content": "/start", "ts": time.time()})
                self.action_start_model(force=True)
                self._chat_messages.append({
                    "role": "assistant",
                    "thought": "Engine Orchestrator",
                    "content": f"🚀 Launching `{prof_name}` server in the background...\nIt will listen on port `:8090/v1` once weights are mapped.\nSend any query once the engine is ready!",
                    "complete": True,
                })
                self._render_chat_messages()
                self._update_chat_view_state()
                return
            elif cmd == "/model":
                prof_name = self.profile_names[self.sel] if self.profile_names else "none"
                self._chat_messages.append({"role": "user", "content": "/model", "ts": time.time()})
                self._chat_messages.append({
                    "role": "assistant",
                    "thought": f"Thought: 14ms",
                    "content": f"**Active Profile**: `{prof_name}`\n- Context Limit: `86,016` tokens\n- Execution Mode: Native GPU MoE with pinned CPU fallbacks\n- Status: Ready",
                    "decode_tps": 38.0,
                    "prefill_tps": 50.0,
                    "latency": 0.03,
                    "complete": True,
                })
                self._render_chat_messages()
                self._update_chat_view_state()
                return
            elif cmd == "/status":
                prof_name = self.profile_names[self.sel] if self.profile_names else "none"
                prof = self.profiles.get(prof_name)
                st = self.states.get(prof_name)
                is_running = st.running if st else False
                health = st.health if st else "down"
                port = prof.env.get("PORT", "8090") if prof else "8090"
                state_str = "🟢 Running (Healthy)" if (is_running and health == "ok") else ("🟡 Loading" if is_running else "⚪ Stopped")
                self._chat_messages.append({"role": "user", "content": "/status", "ts": time.time()})
                self._chat_messages.append({
                    "role": "assistant",
                    "thought": "Engine Orchestrator",
                    "content": (
                        f"**Engine Orchestrator Status**:\n"
                        f"- Active Model: `{prof_name}`\n"
                        f"- Engine State: {state_str}\n"
                        f"- Target Endpoint: `http://127.0.0.1:{port}/v1`\n"
                        f"- Runtime Driver: `{st.driver if st else 'direct'}`\n"
                        f"- Siphon Version: `v0.2.0`\n"
                        f"- Architecture: Native C++ MoE Runtime (CPU/GPU)"
                    ),
                    "decode_tps": 45.0,
                    "prefill_tps": 60.0,
                    "latency": 0.01,
                    "complete": True,
                })
                self._render_chat_messages()
                self._update_chat_view_state()
                return
            else:
                self._chat_messages.append({"role": "user", "content": raw, "ts": time.time()})
                self._chat_messages.append({
                    "role": "assistant",
                    "thought": "Unknown Command",
                    "content": f"Unknown command: `{cmd}`. Type `/` to see available commands or `/help` for guidance.",
                    "complete": True,
                })
                self._render_chat_messages()
                self._update_chat_view_state()
                return

        # Regular user message
        self._chat_messages.append({"role": "user", "content": raw, "ts": time.time()})
        self._chat_ctx_tokens += len(raw.split()) + 8
        self._update_chat_view_state()

        # Assistant placeholder
        initial_thought = "" if self._chat_reasoning_mode == "off" else "Thinking..."
        self._chat_messages.append({
            "role": "assistant",
            "thought": initial_thought,
            "thought_content": "",
            "thought_expanded": False,
            "content": "",
            "decode_tps": 0.0,
            "prefill_tps": 0.0,
            "latency": 0.0,
            "complete": False,
        })
        self._render_chat_messages()

        # Run background worker for generation
        self._chat_is_generating = True
        self._chat_abort = False
        self._stream_chat_worker(raw)

    def _handle_stream_token(
        self,
        full_text: str,
        thought_text: str,
        decode_tps: float,
        prefill_tps: float,
        latency: float,
    ) -> None:
        if not self._chat_messages:
            return
        msg = self._chat_messages[-1]
        if msg.get("role") == "assistant":
            msg["content"] = full_text
            msg["thought_content"] = thought_text
            msg["decode_tps"] = decode_tps
            msg["prefill_tps"] = prefill_tps
            msg["latency"] = latency
            self._chat_decode_tps = decode_tps
            self._chat_prefill_tps = prefill_tps
            if self._chat_reasoning_mode != "off":
                th_clean = thought_text.strip()
                if th_clean and not full_text:
                    th_words = len(th_clean.split())
                    msg["thought"] = f"Thinking... ({th_words} tokens)"
                elif th_clean and full_text:
                    th_words = len(th_clean.split())
                    msg["thought"] = f"Thought ({th_words} tokens)"
                elif not th_clean and not full_text:
                    msg["thought"] = "Thinking..."
            self._render_chat_messages(auto_scroll=True)
            self._update_chat_status_metrics()

    def _handle_stream_complete(self, start_time: float) -> None:
        if not self._chat_messages:
            return
        msg = self._chat_messages[-1]
        if msg.get("role") == "assistant":
            msg["complete"] = True
            if self._chat_reasoning_mode == "off":
                msg["thought"] = ""
                msg["thought_content"] = ""
            else:
                dur_ms = int((time.time() - start_time) * 1000)
                th_words = len(msg.get("thought_content", "").split())
                if th_words > 0:
                    msg["thought"] = f"Thought: {dur_ms}ms · {th_words} tokens"
                else:
                    msg["thought"] = f"Thought: {dur_ms}ms"
        self._chat_is_generating = False
        self._render_chat_messages()
        self._update_chat_status_metrics()

    def _handle_stream_error(self, err_msg: str, thought: str = "Inference Error") -> None:
        if not self._chat_messages:
            return
        msg = self._chat_messages[-1]
        if msg.get("role") == "assistant":
            msg["thought"] = thought
            msg["content"] = err_msg
            msg["decode_tps"] = 0.0
            msg["prefill_tps"] = 0.0
            msg["latency"] = 0.0
            msg["complete"] = True
        self._chat_is_generating = False
        self._render_chat_messages()
        self._update_chat_status_metrics()

    def _handle_stream_offline(self, prof_name: str, port: int, start_time: float) -> None:
        if not self._chat_messages:
            return
        offline_msg = (
            f"⚠️ **Engine Offline**: `{prof_name}` is not currently running on port `:{port}`.\n\n"
            f"To run live inference with your local model:\n"
            f"- Type `/start` right here in Chat to launch `{prof_name}` in the background.\n"
            f"- Or switch to Tab 1 (Dashboard) and press `s` to start it.\n"
            f"- Or run `siphon up {prof_name}` from your terminal."
        )
        msg = self._chat_messages[-1]
        if msg.get("role") == "assistant":
            msg["thought"] = f"Engine Offline (:{port})"
            msg["content"] = offline_msg
            msg["decode_tps"] = 0.0
            msg["prefill_tps"] = 0.0
            msg["latency"] = round(time.time() - start_time, 2)
            msg["complete"] = True
        self._chat_is_generating = False
        self._render_chat_messages()
        self._update_chat_status_metrics()

    @work(thread=True, exclusive=True)
    def _stream_chat_worker(self, prompt: str) -> None:
        prof_name = self.profile_names[self.sel] if self.profile_names else "siphon-model"
        port = 8090
        if prof_name in self.profiles:
            try:
                prof = self.profiles[prof_name]
                port = int(prof.env.get("PORT", "8090"))
            except Exception:
                port = 8090

        start_time = time.time()
        url = f"http://127.0.0.1:{port}/v1/chat/completions"

        # Build clean conversation history (exclude the current assistant placeholder)
        msgs = []
        for m in self._chat_messages[:-1]:
            c = m.get("content", "").strip()
            r = m.get("role", "")
            if r in ("user", "assistant") and c:
                msgs.append({"role": r, "content": c})
        if not msgs:
            msgs = [{"role": "user", "content": prompt}]

        budget_map = {
            "off": 0,
            "1k": 1024,
            "low": 1024,
            "4k": 4096,
            "mid": 4096,
            "max": -1,
        }
        budget = budget_map.get(self._chat_reasoning_mode, -1)

        payload_dict: dict[str, Any] = {
            "model": prof_name,
            "messages": msgs,
            "stream": True,
            "max_tokens": 2048,
        }
        if budget >= 0:
            payload_dict["reasoning_budget_tokens"] = budget
            payload_dict["thinking_budget_tokens"] = budget
            if budget == 0:
                payload_dict["chat_template_kwargs"] = {"preserve_thinking": False}

        payload = json.dumps(payload_dict).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})

        try:
            resp = urllib.request.urlopen(req, timeout=180.0)
        except urllib.error.HTTPError as exc:
            err_body = exc.read().decode("utf-8", errors="replace")
            self.call_from_thread(self._handle_stream_error, f"⚠️ Server Error ({exc.code}): {err_body}")
            return
        except Exception:
            self.call_from_thread(self._handle_stream_offline, prof_name, port, start_time)
            return

        token_count = 0
        first_token_time = None
        full_text = ""
        thought_text = ""
        in_think_tag = False

        try:
            while True:
                if self._chat_abort:
                    break
                line = resp.readline()
                if not line:
                    break
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data: "):
                    data_str = line_str[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta_dict = chunk.get("choices", [{}])[0].get("delta", {})
                        r_delta = delta_dict.get("reasoning_content", "")
                        c_delta = delta_dict.get("content", "")

                        got_token = False
                        if r_delta:
                            thought_text += r_delta
                            got_token = True

                        if c_delta:
                            if "<think>" in c_delta:
                                parts = c_delta.split("<think>", 1)
                                full_text += parts[0]
                                in_think_tag = True
                                c_delta = parts[1]

                            if in_think_tag:
                                if "</think>" in c_delta:
                                    t_part, c_part = c_delta.split("</think>", 1)
                                    thought_text += t_part
                                    in_think_tag = False
                                    full_text += c_part
                                else:
                                    thought_text += c_delta
                            else:
                                full_text += c_delta
                            got_token = True

                        if got_token:
                            now = time.time()
                            if first_token_time is None:
                                first_token_time = now
                                prefill_tps = max(5.0, 15.0 / max(0.01, now - start_time))
                            else:
                                prefill_tps = self._chat_prefill_tps
                            token_count += 1
                            dur = max(0.01, now - (first_token_time or now))
                            decode_tps = token_count / dur
                            latency = now - start_time
                            self.call_from_thread(
                                self._handle_stream_token,
                                full_text,
                                thought_text,
                                decode_tps,
                                prefill_tps,
                                latency,
                            )
                    except Exception:
                        pass
        except Exception as e:
            if not full_text and not thought_text:
                self.call_from_thread(self._handle_stream_error, f"⚠️ Stream connection interrupted: {e}")
                return

        if not full_text and not thought_text and not self._chat_abort:
            self.call_from_thread(
                self._handle_stream_error,
                "⚠️ Model finished without generating tokens. Please send your message again or check server logs."
            )
            return

        self.call_from_thread(self._handle_stream_complete, start_time)



    def _render_chat_messages(self, auto_scroll: bool = True) -> None:
        try:
            container = self.query_one("#chat-messages-container", Static)
        except Exception:
            return

        t = Text()
        if not self._chat_messages:
            container.update(t)
            self._thought_header_lines.clear()
            return

        for m in self._chat_messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role == "user":
                t.append("\n")
                t.append("│ ", style="bold #7C83F5")
                t.append(content + "\n", style="bold #F8FAFC on #131A26")
                t.append("\n")
            elif role == "assistant":
                thought = m.get("thought", "")
                thought_content = m.get("thought_content", "")
                is_expanded = m.get("thought_expanded", False)
                if thought:
                    if is_expanded:
                        t.append("▼ ", style="bold #F59E0B")
                        t.append(thought, style="bold #F59E0B")
                        t.append("  (click to collapse)\n", style="dim italic #64748B")
                        box_w = max(40, min(80, (container.size.width or 80) - 6))
                        t.append("┌─ Thinking Process " + "─" * max(5, box_w - 20) + "\n", style="#475569")
                        th_body = thought_content.strip()
                        if not th_body:
                            t.append("│ ", style="#475569")
                            t.append("Thinking in progress...\n", style="italic #94A3B8")
                        else:
                            for tl in th_body.split("\n"):
                                t.append("│ ", style="#475569")
                                t.append(tl + "\n", style="italic #94A3B8")
                        t.append("└" + "─" * max(20, box_w - 2) + "\n\n", style="#475569")
                    else:
                        t.append("▶ ", style="bold #F59E0B")
                        t.append(thought, style="bold #F59E0B")
                        t.append("  (click to expand)\n\n", style="dim italic #64748B")
                
                lines = content.split("\n")
                in_code = False
                line_no = 1
                for line in lines:
                    if line.startswith("```"):
                        in_code = not in_code
                        if in_code:
                            t.append("┌" + "─" * 60 + "\n", style="#1E293B")
                        else:
                            t.append("└" + "─" * 60 + "\n", style="#1E293B")
                        continue
                    if in_code:
                        t.append(f"{line_no:>2} │ ", style="#475569")
                        line_no += 1
                        if any(k in line for k in ["import ", "async ", "def ", "while ", "try:", "except ", "return "]):
                            t.append(line + "\n", style="#C084FC")
                        elif any(k in line for k in ["worker", "queue", "task", "stop_event"]):
                            t.append(line + "\n", style="#38BDF8")
                        else:
                            t.append(line + "\n", style="#E2E8F0")
                    else:
                        if line.startswith("- ") or line.startswith("* "):
                            t.append("  • ", style="#38BDF8")
                            t.append(line[2:] + "\n", style="#CBD5E1")
                        elif line.startswith("#"):
                            t.append(line + "\n", style="bold #F8FAFC")
                        else:
                            t.append(line + "\n", style="#CBD5E1")
                
                dec = m.get("decode_tps", 0.0)
                pref = m.get("prefill_tps", 0.0)
                lat = m.get("latency", 0.0)
                if m.get("complete", False) or dec > 0:
                    prof_name = self.profile_names[self.sel] if self.profile_names else "siphon"
                    t.append("\n")
                    t.append(f"🬀 Build · {prof_name} · ⚡ {dec:.1f} t/s · 🚀 {pref:.1f} t/s · {lat:.1f}s\n\n", style="#64748B")

        container.update(t)

        width = container.size.width or 80
        res = container.render()
        try:
            wrapped = res.wrap(width)
        except Exception:
            wrapped = []

        self._thought_header_lines.clear()
        asst_indices = [
            i for i, msg in enumerate(self._chat_messages)
            if msg.get("role") == "assistant" and (msg.get("thought") or msg.get("thought_content"))
        ]
        asst_ptr = 0
        in_thought_box = False

        for row_idx, line in enumerate(wrapped):
            plain = line.plain
            if asst_ptr < len(asst_indices):
                target_msg_idx = asst_indices[asst_ptr]
                if plain.startswith("▶ ") or plain.startswith("▼ "):
                    self._thought_header_lines[row_idx] = target_msg_idx
                    if plain.startswith("▼ "):
                        in_thought_box = True
                    else:
                        asst_ptr += 1
                elif in_thought_box:
                    self._thought_header_lines[row_idx] = target_msg_idx
                    if plain.startswith("└"):
                        in_thought_box = False
                        asst_ptr += 1

        if auto_scroll:
            try:
                scroll = self.query_one("#chat-messages-scroll", VerticalScroll)
                scroll.scroll_end(animate=False)
            except Exception:
                pass

    def _update_chat_status_metrics(self) -> None:
        try:
            pct = (self._chat_ctx_tokens / max(1, self._chat_ctx_limit)) * 100
            txt = Text()
            txt.append(f"ctx: {self._chat_ctx_tokens:,} ({pct:.1f}%)   ", style="#94A3B8")
            txt.append(f"⚡ {self._chat_decode_tps:.1f} t/s   ", style="bold #34D399")
            txt.append(f"🚀 {self._chat_prefill_tps:.1f} t/s prefill", style="bold #38BDF8")
            self.query_one("#chat-status-metrics", Static).update(txt)
        except Exception:
            pass

    def action_stop_chat(self) -> None:
        if self._chat_is_generating:
            self._chat_abort = True
            self._chat_is_generating = False
            if self._chat_messages and self._chat_messages[-1]["role"] == "assistant":
                self._chat_messages[-1]["complete"] = True
                self._chat_messages[-1]["content"] += " [Stopped]"
                self._render_chat_messages()


    # ── interactions ──────────────────────────────────────────────────────
    def action_move_up(self) -> None:
        if isinstance(self.focused, Input):
            return
        if self.profile_names:
            self.sel = (self.sel - 1) % len(self.profile_names)
            self._update_table()
            self._update_detail()
            self._update_metrics()
            self._update_logs()

    def action_move_down(self) -> None:
        if isinstance(self.focused, Input):
            return
        if self.profile_names:
            self.sel = (self.sel + 1) % len(self.profile_names)
            self._update_table()
            self._update_detail()
            self._update_metrics()
            self._update_logs()

    def action_select_model(self) -> None:
        if self.profile_names and self.sel < len(self.profile_names):
            name = self.profile_names[self.sel]
            self._push_log("INFO", f"select {name}")
            self._update_logs()

    def action_start_model(self, force: bool = False) -> None:
        if not force and isinstance(self.focused, Input):
            return
        if not self.profile_names or self.sel >= len(self.profile_names):
            return
        name = self.profile_names[self.sel]
        prof = self.profiles.get(name)
        if prof:
            try:
                ok, msg = runner.up(prof)
                if ok:
                    port = prof.env.get("PORT", "8090")
                    self._push_log("OK", f"{name} running · listening on :{port}/v1")
                else:
                    self._push_log("WARN", f"{name}: {msg}")
            except Exception as exc:
                self._push_log("ERR", f"{name} start failed: {exc}")
        self._tick()

    def action_stop_model(self, force: bool = False) -> None:
        if not force and isinstance(self.focused, Input):
            return
        if not self.profile_names or self.sel >= len(self.profile_names):
            return
        name = self.profile_names[self.sel]
        prof = self.profiles.get(name)

        if not self._pending_stop:
            self._pending_stop = True
            self._stop_arm_time = time.time()
            self._arm_stop_btn()
            self._push_log("WARN", f"{name} · press x again to confirm stop")
            try:
                self.query_one("#cfg-status", Static).update(
                    Text("· press x again to confirm stop", style="bold #F85149")
                )
            except Exception:
                pass
            return

        self._pending_stop = False
        self._disarm_stop_btn()
        if prof:
            try:
                ok, msg = runner.down(prof, force=True)
                if ok:
                    self._push_log("ERR", f"{name} stopped · vram released")
                else:
                    self._push_log("WARN", f"{name} stop: {msg}")
            except Exception as exc:
                self._push_log("ERR", f"{name} stop failed: {exc}")
        try:
            self.query_one("#cfg-status", Static).update(
                Text(f"· {name} stopped — vram released", style="#F85149")
            )
        except Exception:
            pass
        self._tick()

    def _arm_stop_btn(self) -> None:
        try:
            b = self.query_one("#btn-stop", Static)
            b.add_class("armed")
            b.update(Text("[x confirm?]"))
        except Exception:
            pass

    def _disarm_stop_btn(self) -> None:
        try:
            b = self.query_one("#btn-stop", Static)
            b.remove_class("armed")
            b.update(Text("[x stop]"))
        except Exception:
            pass

    def action_refresh_all(self) -> None:
        if isinstance(self.focused, Input):
            return
        self._push_log("INFO", "refresh · models re-polled")
        self._tick()

    def action_apply_cfg(self, force: bool = False) -> None:
        if not force and isinstance(self.focused, Input):
            return
        if not self.profile_names or self.sel >= len(self.profile_names):
            return
        name = self.profile_names[self.sel]
        prof = self.profiles.get(name)

        for key, env_var in ALL_CFG_FIELDS:
            try:
                inp = self.query_one(f"#cfg-{key}", Input)
                val = inp.value.strip()
                if prof and hasattr(prof, "env") and isinstance(prof.env, dict):
                    prof.env[env_var] = val
                self._initial_cfg[key] = inp.value
            except Exception:
                pass

        self._cfg_dirty = False
        port = prof.env.get("PORT", "8090") if prof else "8090"
        ctx = prof.env.get("CTX", "65536") if prof else "65536"
        batch = prof.env.get("BATCH", "512") if prof else "512"
        self._push_log("INFO", f"config applied · {name} port {port} ctx {ctx} batch {batch}")
        try:
            self.query_one("#cfg-status", Static).update(
                Text(f"· applied — {name} :{port} · ctx {ctx}", style="#3FB950")
            )
        except Exception:
            pass

    def action_revert_cfg(self, force: bool = False) -> None:
        if not force and isinstance(self.focused, Input):
            return
        self._cfg_dirty = False
        for key, _ in ALL_CFG_FIELDS:
            if key in self._initial_cfg:
                try:
                    self.query_one(f"#cfg-{key}", Input).value = self._initial_cfg[key]
                except Exception:
                    pass
        name = self.profile_names[self.sel] if self.sel < len(self.profile_names) else ""
        self._push_log("INFO", f"config reverted · {name}")
        try:
            self.query_one("#cfg-status", Static).update(
                Text("· reverted — no pending edits", style="#5E6A7E")
            )
        except Exception:
            pass

    def action_reset_cfg(self, force: bool = False) -> None:
        if not force and isinstance(self.focused, Input):
            return
        self._last_rendered_sel = -1
        name = self.profile_names[self.sel] if self.sel < len(self.profile_names) else ""
        prof = self.profiles.get(name)
        self._populate_cfg(prof)
        self._push_log("INFO", f"reset to defaults — press Apply to save · {name}")
        try:
            self.query_one("#cfg-status", Static).update(
                Text("· reset to defaults — press a to apply, u to revert", style="#D29922")
            )
        except Exception:
            pass

    def action_cycle_focus(self) -> None:
        if self._cmd_popup_matches and isinstance(self.focused, Input):
            wid = getattr(self.focused, "id", "")
            if wid in ("chat-welcome-input", "chat-docked-input"):
                cmd, _ = self._cmd_popup_matches[self._cmd_popup_idx]
                self.focused.value = cmd + " "
                self.focused.cursor_position = len(self.focused.value)
                self._update_cmd_popup_for_input(wid, self.focused.value)
                return
        self.focus_idx = (self.focus_idx + 1) % len(self._focusable_panels)
        for i, pid in enumerate(self._focusable_panels):
            try:
                p = self.query_one(f"#{pid}")
                if i == self.focus_idx:
                    p.add_class("focused")
                else:
                    p.remove_class("focused")
            except Exception:
                pass

    def on_key(self, event) -> None:
        if isinstance(self.focused, Input):
            wid = getattr(self.focused, "id", "")
            if event.key in ("ctrl+1", "f1"):
                self._hide_cmd_popups()
                self.action_switch_dashboard()
                event.prevent_default()
                return
            elif event.key in ("ctrl+2", "f2"):
                self.action_switch_chat()
                event.prevent_default()
                return
            elif self._cmd_popup_matches and wid in ("chat-welcome-input", "chat-docked-input"):
                if event.key == "up":
                    self._cmd_popup_idx = (self._cmd_popup_idx - 1) % len(self._cmd_popup_matches)
                    w_pop, d_pop = self._get_active_cmd_popups()
                    target_pop = w_pop if wid == "chat-welcome-input" else d_pop
                    if target_pop:
                        target_pop.update(self._render_cmd_popup_markup())
                    event.prevent_default()
                    event.stop()
                    return
                elif event.key == "down":
                    self._cmd_popup_idx = (self._cmd_popup_idx + 1) % len(self._cmd_popup_matches)
                    w_pop, d_pop = self._get_active_cmd_popups()
                    target_pop = w_pop if wid == "chat-welcome-input" else d_pop
                    if target_pop:
                        target_pop.update(self._render_cmd_popup_markup())
                    event.prevent_default()
                    event.stop()
                    return
                elif event.key == "tab":
                    cmd, _ = self._cmd_popup_matches[self._cmd_popup_idx]
                    self.focused.value = cmd + " "
                    self.focused.cursor_position = len(self.focused.value)
                    self._update_cmd_popup_for_input(wid, self.focused.value)
                    event.prevent_default()
                    event.stop()
                    return
                elif event.key == "escape":
                    self._hide_cmd_popups()
                    event.prevent_default()
                    event.stop()
                    return
                elif event.key == "enter":
                    cmd, _ = self._cmd_popup_matches[self._cmd_popup_idx]
                    self.focused.value = ""
                    self._hide_cmd_popups()
                    self._handle_send_chat(cmd)
                    event.prevent_default()
                    event.stop()
                    return
            elif self._think_popup_open and wid in ("chat-welcome-input", "chat-docked-input"):
                if event.key == "up":
                    self._think_popup_idx = (self._think_popup_idx - 1) % len(THINKING_OPTIONS)
                    w_pop, d_pop = self._get_active_think_popups()
                    target_pop = w_pop if wid == "chat-welcome-input" else d_pop
                    if target_pop:
                        target_pop.update(self._render_think_popup_markup())
                    event.prevent_default()
                    event.stop()
                    return
                elif event.key == "down":
                    self._think_popup_idx = (self._think_popup_idx + 1) % len(THINKING_OPTIONS)
                    w_pop, d_pop = self._get_active_think_popups()
                    target_pop = w_pop if wid == "chat-welcome-input" else d_pop
                    if target_pop:
                        target_pop.update(self._render_think_popup_markup())
                    event.prevent_default()
                    event.stop()
                    return
                elif event.key == "enter":
                    chosen_mode = THINKING_OPTIONS[self._think_popup_idx][0]
                    self.action_cycle_reasoning(explicit_mode=chosen_mode)
                    self._hide_think_popup()
                    event.prevent_default()
                    event.stop()
                    return
                elif event.key == "escape":
                    self._hide_think_popup()
                    event.prevent_default()
                    event.stop()
                    return
                elif event.key in ("ctrl+t", "ctrl+r"):
                    self._think_popup_idx = (self._think_popup_idx + 1) % len(THINKING_OPTIONS)
                    w_pop, d_pop = self._get_active_think_popups()
                    target_pop = w_pop if wid == "chat-welcome-input" else d_pop
                    if target_pop:
                        target_pop.update(self._render_think_popup_markup())
                    event.prevent_default()
                    event.stop()
                    return
            elif event.key == "escape":
                self._hide_cmd_popups()
                self._hide_think_popup()
                if self._chat_is_generating:
                    self.action_stop_chat()
                self.set_focus(None)
                event.prevent_default()
            elif event.key == "enter":
                if wid in ("chat-welcome-input", "chat-docked-input"):
                    val = self.focused.value.strip()
                    if val:
                        self.focused.value = ""
                        self._hide_cmd_popups()
                        self._hide_think_popup()
                        self._handle_send_chat(val)
                    event.prevent_default()
                elif wid.startswith("cfg-"):
                    self.action_apply_cfg(force=True)
                    self.set_focus(None)
                    event.prevent_default()
            elif event.key in ("ctrl+t", "ctrl+r") and wid in ("chat-welcome-input", "chat-docked-input"):
                self._show_think_popup()
                event.prevent_default()
            elif event.key == "1" and not getattr(self.focused, "value", "").strip() and wid in ("chat-welcome-input", "chat-docked-input"):
                self._hide_cmd_popups()
                self._hide_think_popup()
                self.action_switch_dashboard()
                event.prevent_default()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        wid = getattr(event.input, "id", "")
        if wid in ("chat-welcome-input", "chat-docked-input"):
            val = event.value.strip()
            if val:
                event.input.value = ""
                self._hide_cmd_popups()
                self._hide_think_popup()
                self._handle_send_chat(val)
            event.prevent_default()
        elif wid.startswith("cfg-"):
            self.action_apply_cfg(force=True)
            self.set_focus(None)
            event.prevent_default()

    def on_click(self, event) -> None:
        wid = getattr(event.widget, "id", "")
        if wid in ("chat-welcome-think-badge", "chat-docked-think-badge"):
            self._toggle_think_popup()
            return
        elif wid in ("chat-welcome-think-popup", "chat-docked-think-popup"):
            line = event.y - 1
            if 0 <= line < len(THINKING_OPTIONS):
                chosen_mode = THINKING_OPTIONS[line][0]
                self.action_cycle_reasoning(explicit_mode=chosen_mode)
                self._hide_think_popup()
            return
        self._hide_cmd_popups()
        self._hide_think_popup()
        if wid in ("tabs-content", "tabs-bar"):
            if event.x > 15:
                self._active_tab = "chat"
            else:
                self._active_tab = "dashboard"
            self._update_tab_visibility()
        elif wid in ("chat-welcome-think-badge", "chat-docked-think-badge"):
            self.action_cycle_reasoning()
        elif wid == "chat-btn-stop":
            self.action_stop_chat()
        elif wid == "chat-messages-container":
            y = event.y
            if y in self._thought_header_lines:
                msg_idx = self._thought_header_lines[y]
                if 0 <= msg_idx < len(self._chat_messages):
                    cur = self._chat_messages[msg_idx].get("thought_expanded", False)
                    self._chat_messages[msg_idx]["thought_expanded"] = not cur
                    self._render_chat_messages(auto_scroll=False)
        elif wid == "model-rows":
            y = event.y
            if 0 <= y < len(self.profile_names):
                self.sel = y
                self._update_table()
                self._update_detail()
                self._update_metrics()
        elif wid == "btn-start":
            self.action_start_model(force=True)
        elif wid == "btn-stop":
            self.action_stop_model(force=True)
        elif wid == "btn-apply":
            self.action_apply_cfg(force=True)
        elif wid == "btn-revert":
            self.action_revert_cfg(force=True)
        elif wid == "btn-reset":
            self.action_reset_cfg(force=True)



def run() -> None:
    SiphonTUI().run()


main = run


if __name__ == "__main__":
    run()
