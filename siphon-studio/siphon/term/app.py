"""Siphon term v4 — terminal-native Textual TUI.

Native implementation of the approved `siphon-tui-v4.html` mockup:
ink canvas, single-line panels, model table, definition-list detail with
collapsible config sections covering all 31 profile knobs, 10-tile metrics
with sparklines for rates/counters and bars for shares, server log tail,
key-hint footer. Data strictly from `siphon.core` + `siphon.tui.data` +
`siphon.console.dstore`.

Keys: s start · x stop · r refresh · a apply · u revert · 0 reset · q quit.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import (
    Button,
    Collapsible,
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    RichLog,
    Select,
    Sparkline,
    Static,
)

from siphon.core import profiles as P
from siphon.core import runner
from siphon.tui.data import Ring, fetch_sys_stats

from ..console.dstore import DStore, parse_log_tail

REFRESH_SECONDS = 2.0

# (env key, human label, kind) — kind: text | int | sel01 | sel_onoff | spec
Field = tuple[str, str, str]

ESSENTIALS: list[Field] = [
    ("MODEL", "path", "text"),
    ("CTX", "ctx", "int"),
    ("BATCH", "batch", "int"),
    ("PORT", "port", "int"),
    ("MOE_GPU_CACHE_MIB", "l1_cache", "int"),
    ("MOE_PINNED_CACHE_MIB", "l2_pinned", "int"),
    ("SPEC_TYPE", "speculator", "spec"),
]

SECTIONS: list[tuple[str, list[Field]]] = [
    ("IDENTITY", [("ALIAS", "alias", "text"), ("HOST", "host", "text"),
                  ("METRICS", "metrics", "sel01")]),
    ("CONTEXT", [("N_PARALLEL", "parallel slots", "int"),
                 ("CHECKPOINT_MIN_STEP", "checkpoint step", "int"),
                 ("LLAMA_ARG_CTX_CHECKPOINTS", "ctx checkpoints", "int")]),
    ("CACHE", [("CACHE_RAM_MIB", "cache RAM MiB", "int"),
               ("EXPERTS", "expert suite", "text"),
               ("LLAMA_DSTORAGE_EVICT_TO_RAM", "evict to ram", "sel01"),
               ("CACHE_PROMPT", "prompt cache", "sel01"),
               ("CACHE_REUSE", "cache reuse", "sel01")]),
    ("SPECULATION", [("SPEC_DRAFT_N_MAX", "draft n_max", "int"),
                     ("SPEC_DRAFT_P_MIN", "draft p_min", "int"),
                     ("SPEC_DRAFT_GPU_LAYERS", "draft gpu layers", "int"),
                     ("MODEL_DRAFT", "draft model", "text")]),
    ("REASONING", [("REASONING", "reasoning", "sel_onoff"),
                   ("REASONING_FORMAT", "reason format", "text"),
                   ("REASONING_BUDGET", "reason budget", "int"),
                   ("CHAT_TEMPLATE", "chat template", "text"),
                   ("CHAT_TEMPLATE_KWARGS", "template kwargs", "text")]),
    ("IO", [("GDS_READ_THREADS", "gds threads", "int"),
            ("LLAMA_DSTORAGE_PREFILL_WORKSPACE", "prefill workspace", "int"),
            ("LLAMA_DSTORAGE_PREFILL_WORKSPACE_LOOKAHEAD", "prefill lookahead", "int"),
            ("MMPROJ", "vision mmproj", "text")]),
]

SPEC_OPTIONS = [
    ("mtp", "draft-mtp"), ("simple", "draft-simple"), ("dflash", "draft-dflash"),
    ("none", "none"), ("off", "off"),
]
SPEC_TO_ENV = dict(SPEC_OPTIONS)
ENV_TO_SPEC = {v: k for k, v in SPEC_OPTIONS}

SYS_BLOCKS = ("gpu", "vram", "cpu", "ram", "nvme")

TILE_DEFS = [
    ("decode", "decode", "spark"), ("prefill", "prefill", "spark"),
    ("l1", "l1_hit", "bar"), ("l2", "l2_hit", "bar"),
    ("draft", "draft_acc", "bar"), ("nvme", "nvme", "bar"),
    ("ram", "ram2vram", "bar"), ("stream", "streamed", "spark"),
    ("dec", "dec_calls", "spark"), ("pre", "pre_calls", "spark"),
]


def _bar(pct: float | None, width: int = 10) -> Text:
    if pct is None:
        return Text("—", style="#3A4358")
    pct = max(0.0, min(1.0, pct))
    full = int(round(pct * width))
    color = "#3FB950" if pct >= 0.8 else ("#D29922" if pct >= 0.5 else "#F85149")
    return Text("█" * full + "░" * (width - full), style=color)


@dataclass
class RowState:
    running: bool = False
    health: str = "down"
    decode: float = 0.0
    prefill: float = 0.0
    n_decode: int = 0
    latency: float = 0.0


class ConfirmStop(ModalScreen[bool]):
    BINDINGS = [("escape", "cancel", "cancel")]

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static(f"Stop {self._name}?", classes="panel-title")
            yield Label("Press x again is not needed — confirm here.",
                        classes="sys-sub")
            with Horizontal(id="btn-row"):
                yield Button("Stop", id="cf-yes", variant="error")
                yield Button("Cancel", id="cf-no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "cf-yes")

    def action_cancel(self) -> None:
        self.dismiss(False)


class SiphonTerm(App):
    TITLE = "siphon — local LLM ops"
    CSS_PATH = "term.tcss"

    BINDINGS = [
        Binding("s", "start", "start"),
        Binding("x", "stop", "stop"),
        Binding("r", "refresh", "refresh"),
        Binding("a", "apply", "apply"),
        Binding("u", "revert", "revert"),
        Binding("0", "reset", "reset"),
        Binding("q", "quit", "quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.profiles: dict[str, P.Profile] = P.discover()
        self.selected: str | None = None
        self.rows: dict[str, RowState] = {}
        self.dstore: dict[str, DStore] = {}
        self._sys = fetch_sys_stats()
        self._hist = {k: Ring(48) for k in
                      ("gpu", "vram", "cpu", "ram", "nvme",
                       "decode", "prefill", "stream", "dec", "pre")}
        self._field_widgets: dict[str, Input | Select] = {}
        self._sections: dict[str, Collapsible] = {}
        self._form_profile: str | None = None
        self._dirty: set[str] = set()
        self._events: deque[str] = deque(maxlen=40)

    # -- theme ------------------------------------------------------------
    @staticmethod
    def make_theme() -> Theme:
        return Theme(
            name="siphon-term",
            primary="#7C83F5",
            secondary="#5E6A7E",
            success="#3FB950",
            warning="#D29922",
            error="#F85149",
            foreground="#C5CBD6",
            background="#0A0E14",
            surface="#0F1520",
            panel="#0F1520",
            dark=True,
        )

    # -- compose -------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="sysbar"):
            for b in SYS_BLOCKS:
                with Vertical(classes="sys-block"):
                    yield Static(b.upper(), classes="sys-label")
                    yield Static("—", id=f"sys-{b}-mid")
                    yield Static("—", id=f"sys-{b}-cap", classes="sys-sub")
                    yield Sparkline([0.0], id=f"sys-{b}-spark", classes="sys-spark")
        with Horizontal(id="main"):
            with Vertical(id="table-panel"):
                yield Static("MODEL", classes="panel-title")
                yield DataTable(id="models-table", cursor_type="row")
                yield Static("↑↓ move · enter select · s start · x stop",
                             id="table-hint")
            with Vertical(id="detail-panel"):
                yield Static("—", id="detail-title", classes="panel-title")
                yield Static("—", id="dl-state")
                yield Static("—", id="dl-extra")
                yield Static("CONFIG — [EDIT]", classes="panel-title")
                with VerticalScroll(id="detail-scroll"):
                    for key, label, kind in ESSENTIALS:
                        yield Label(label, classes="dl-label")
                        yield self._widget_for(key, kind)
                    for title, fields in SECTIONS:
                        with Collapsible(title=f"▶ {title}", collapsed=True,
                                          id=f"sec-{title}") as col:
                            self._sections[title] = col
                            for key, label, kind in fields:
                                yield Label(label, classes="dl-label")
                                yield self._widget_for(key, kind)
                with Vertical(id="detail-foot"):
                    with Horizontal(id="btn-row"):
                        yield Button("[S]tart", id="btn-start", variant="success")
                        yield Button("[x] stop", id="btn-stop", variant="error")
                        yield Button("[a]pply", id="btn-apply", variant="primary")
                        yield Button("[u] revert", id="btn-revert")
                        yield Button("[0] reset", id="btn-reset", variant="warning")
                    yield Static("clean — no pending edits", id="cfg-status")
        with Vertical(id="metrics-panel"):
            yield Static("METRICS", classes="panel-title")
            with Horizontal(id="metrics-row1"):
                for key, label, viz in TILE_DEFS[:5]:
                    with Vertical(classes="tile"):
                        yield Static(label, classes="tile-label")
                        yield Static("—", id=f"tile-{key}", classes="tile-value")
                        yield Static("", id=f"tile-{key}-sub", classes="tile-sub")
                        if viz == "bar":
                            yield Static("", id=f"tile-{key}-viz", classes="tile-bar")
                        else:
                            yield Sparkline([0.0], id=f"tile-{key}-viz",
                                            classes="tile-spark")
            with Horizontal(id="metrics-row2"):
                for key, label, viz in TILE_DEFS[5:]:
                    with Vertical(classes="tile"):
                        yield Static(label, classes="tile-label")
                        yield Static("—", id=f"tile-{key}", classes="tile-value")
                        yield Static("", id=f"tile-{key}-sub", classes="tile-sub")
                        if viz == "bar":
                            yield Static("", id=f"tile-{key}-viz", classes="tile-bar")
                        else:
                            yield Sparkline([0.0], id=f"tile-{key}-viz",
                                            classes="tile-spark")
        with Vertical(id="log-panel"):
            yield Static("LOG", classes="panel-title")
            yield RichLog(id="log", highlight=True, wrap=True)
        yield Static("", id="status-line")
        yield Footer()

    def _widget_for(self, key: str, kind: str) -> Input | Select:
        if kind == "spec":
            w = Select([(l, l) for l, _ in SPEC_OPTIONS], value="mtp", id=f"f-{key}")
        elif kind == "sel01":
            w = Select([("0", "0"), ("1", "1")], value="0", id=f"f-{key}")
        elif kind == "sel_onoff":
            w = Select([("off", "off"), ("on", "on")], value="off", id=f"f-{key}")
        else:
            w = Input(id=f"f-{key}")
        self._field_widgets[key] = w
        return w

    # -- lifecycle ------------------------------------------------------------
    def on_mount(self) -> None:
        self.register_theme(self.make_theme())
        self.theme = "siphon-term"
        tbl = self.query_one("#models-table", DataTable)
        tbl.add_column("NAME", key="name")
        tbl.add_column("STATE", key="state")
        tbl.add_column("RATE", key="rate")
        tbl.add_column("VRAM-FIT", key="fit")
        self._status("ready — pick a model, s to start")
        self.refresh_all()
        self.set_interval(REFRESH_SECONDS, self._tick)

    def _tick(self) -> None:
        try:
            self.refresh_all()
        except Exception as exc:
            self._status(f"poll error: {exc}")

    # -- data --------------------------------------------------------------------
    def _target(self) -> P.Profile | None:
        if self.selected and self.selected in self.profiles:
            return self.profiles[self.selected]
        return None

    def refresh_all(self) -> None:
        self.profiles = P.discover()
        if not self.selected or self.selected not in self.profiles:
            running = [n for n, p in self.profiles.items()
                       if self.rows.get(n, RowState()).running
                       or runner.status(p).running]
            self.selected = (running or list(self.profiles))[0] if self.profiles else None
        for name, prof in self.profiles.items():
            rs = RowState()
            try:
                st = runner.status(prof)
                rs.running = st.running
                rs.health = st.health.status
                rs.decode = st.metrics.decode_tps
                rs.prefill = st.metrics.prefill_tps
                rs.n_decode = st.metrics.n_decode
                rs.latency = st.health.latency_ms
            except Exception:
                pass
            self.rows[name] = rs
            try:
                self.dstore[name] = parse_log_tail(prof.log_file)
            except Exception:
                self.dstore[name] = DStore()
        try:
            self._sys = fetch_sys_stats()
        except Exception:
            pass
        self._push_histories()
        self._update_sysbar()
        self._rebuild_table()
        self._update_detail()
        self._update_tiles()
        self._update_log()

    def _push_histories(self) -> None:
        s = self._sys
        p = self._target()
        is_run = bool(p and self.rows.get(p.name, RowState()).running)
        ds = self.dstore.get(p.name) if p else DStore()
        self._hist["gpu"].push(s.gpu_util)
        self._hist["vram"].push(s.gpu_mib_used / 1024 if s.gpu_mib_total else 0.0)
        self._hist["cpu"].push(s.cpu)
        self._hist["ram"].push(s.mem_used_gb)
        self._hist["nvme"].push(s.nvme_temp)
        dec = self.rows.get(p.name, RowState()).decode if (p and is_run) else 0.0
        pre = self.rows.get(p.name, RowState()).prefill if (p and is_run) else 0.0
        self._hist["decode"].push(dec)
        self._hist["prefill"].push(pre)
        self._hist["stream"].push(ds.streamed_mib or 0.0)
        self._hist["dec"].push(float(self.rows.get(p.name, RowState()).n_decode if is_run else 0))
        self._hist["pre"].push(float(ds.prefill_calls or 0) if is_run else 0.0)

    # -- sysbar ----------------------------------------------------------------------
    def _update_sysbar(self) -> None:
        s = self._sys
        total = s.gpu_mib_total / 1024 if s.gpu_mib_total else 6.0
        used = s.gpu_mib_used / 1024
        free = s.gpu_mib_free / 1024 if s.gpu_mib_total else 0.0
        blocks = {
            "gpu": (f"{s.gpu_util:.0f}% · {s.gpu_temp:.0f}C",
                    f"{s.gpu_power_w:.0f}W",
                    s.gpu_temp >= 85, False),
            "vram": (f"{used / total * 100:.0f}%" if total else "—",
                     f"{used:.1f}/{total:.1f}GiB ({free:.1f} free)",
                     False, used / total >= 0.9 if total else False),
            "cpu": (f"{s.cpu:.0f}% · {s.cpu_temp:.0f}C",
                    f"{s.cpu_freq_mhz / 1000:.1f}GHz",
                    s.cpu_temp >= 90, False),
            "ram": (f"{s.mem_pct:.0f}%",
                    f"{s.mem_used_gb:.1f}/{s.mem_total_gb:.1f}GiB ({s.mem_free_gb:.1f} free)",
                    False, False),
            "nvme": (f"{s.nvme_temp:.0f}C" if s.nvme_temp else "—",
                     f"load {s.load:.1f}", False, False),
        }
        for b, (mid, cap, hot, warn) in blocks.items():
            try:
                mid_w = self.query_one(f"#sys-{b}-mid", Static)
                style = ("bold #F85149" if hot else
                         ("#D29922" if warn else "#C5CBD6"))
                txt = Text(mid, style=style)
                if b == "cpu" and hot:
                    txt.append(" HOT", style="bold #F85149")
                mid_w.update(txt)
                self.query_one(f"#sys-{b}-cap", Static).update(cap)
                sp = self.query_one(f"#sys-{b}-spark", Sparkline)
                sp.data = self._hist[b].plain() or [0.0]
            except Exception:
                pass

    # -- table --------------------------------------------------------------------------
    def _fit_badge(self, prof: P.Profile) -> str:
        try:
            l1 = int(prof.env.get("MOE_GPU_CACHE_MIB", "0") or 0) / 1024
            ctx = int(prof.env.get("CTX", "0") or 0)
        except ValueError:
            return "unverified"
        total = self._sys.gpu_mib_total / 1024 if self._sys.gpu_mib_total else 6.0
        est = l1 + 2.4 + (ctx / 65536 * 0.6 if ctx else 0.0)
        if est > total:
            return "NO-FIT"
        return f"Fits {est / total * 100:.0f}%"

    def _rebuild_table(self) -> None:
        try:
            tbl = self.query_one("#models-table", DataTable)
        except Exception:
            return
        tbl.clear()
        for name in self.profiles:
            rs = self.rows.get(name, RowState())
            if rs.running and rs.health == "ok":
                state = Text("RUNNING", style="bold #3FB950")
                rate = f"{rs.decode:.1f}t/s"
            elif rs.running:
                state = Text("LOADING", style="#D29922")
                rate = "starting…"
            else:
                state = Text("IDLE", style="#5E6A7E")
                rate = "standby"
            tbl.add_row(name, state, rate, self._fit_badge(self.profiles[name]),
                        key=name)

    # -- detail ----------------------------------------------------------------------------
    def _form_value(self, key: str) -> str:
        w = self._field_widgets.get(key)
        if w is None:
            return ""
        if isinstance(w, Select):
            v = w.value
            return str(v) if v is not None else ""
        return w.value.strip()

    def _set_form_value(self, key: str, value: str) -> None:
        w = self._field_widgets.get(key)
        if w is None:
            return
        try:
            if isinstance(w, Select):
                if kind_of(key) == "spec":
                    w.value = ENV_TO_SPEC.get(value, "mtp") if value in ENV_TO_SPEC.values() else value
                else:
                    w.value = value if value in ("0", "1", "on", "off") else (
                        "off" if kind_of(key) == "sel_onoff" else "0")
            else:
                w.value = value
        except Exception:
            pass

    def _fill_form(self, prof: P.Profile) -> None:
        for key, _label, _kind in ESSENTIALS + [f for _, fs in SECTIONS for f in fs]:
            self._set_form_value(key, prof.env.get(key, ""))
        self._form_profile = prof.name
        self._dirty = set()
        self._sync_section_titles(prof)

    def _sync_section_titles(self, _prof: P.Profile) -> None:
        for title, _fs in SECTIONS:
            col = self._sections.get(title)
            if col is not None:
                self._retitle_section(col, title)

    def _mark_dirty(self) -> None:
        p = self._target()
        if not p or self._form_profile != p.name:
            return
        dirty: set[str] = set()
        for key, _label, _kind in ESSENTIALS + [f for _, fs in SECTIONS for f in fs]:
            cur = self._form_value(key)
            if key == "SPEC_TYPE":
                if cur != ENV_TO_SPEC.get(p.env.get(key, ""), "mtp"):
                    dirty.add(key)
            elif cur != p.env.get(key, ""):
                dirty.add(key)
        self._dirty = dirty
        self._sync_section_titles(p)
        try:
            self.query_one("#cfg-status", Static).update(
                f"· {len(dirty)} edited — a to apply, u to revert" if dirty
                else "clean — no pending edits")
        except Exception:
            pass

    def _update_detail(self) -> None:
        p = self._target()
        if p is None:
            return
        if self._form_profile != p.name:
            self._fill_form(p)
        rs = self.rows.get(p.name, RowState())
        h = rs.health if rs.running else "down"
        state_style = ("bold #3FB950" if h == "ok" else
                       ("#D29922" if h == "loading" else "#5E6A7E"))
        state_txt = {"ok": "RUNNING", "loading": "LOADING"}.get(h, "IDLE")
        try:
            self.query_one("#detail-title", Static).update(f"detail · {p.name}")
            self.query_one("#dl-state", Static).update(
                Text(f"state .... {state_txt}", style=state_style))
            suite = (p.resolve("EXPERTS") or "—").split("/")[-1].replace(".txt", "")
            lat = f"{rs.latency:.0f}ms" if rs.running else "—"
            self.query_one("#dl-extra", Static).update(
                f"latency .... {lat} · endpoint :{p.port}/v1 · suite {suite or '—'}")
        except Exception:
            pass

    # -- tiles ----------------------------------------------------------------------------------
    def _tile_bar(self, key: str, pct: float | None) -> None:
        try:
            w = self.query_one(f"#tile-{key}-viz", Static)
            t = _bar(pct)
            if pct is None:
                w.set_classes("tile-bar")
            elif pct >= 0.8:
                w.set_classes("tile-bar ok")
            elif pct >= 0.5:
                w.set_classes("tile-bar warn")
            else:
                w.set_classes("tile-bar bad")
            w.update(t)
        except Exception:
            pass

    def _tile_spark(self, key: str, hist: list[float]) -> None:
        try:
            sp = self.query_one(f"#tile-{key}-viz", Sparkline)
            sp.data = hist or [0.0]
        except Exception:
            pass

    def _tile(self, key: str, value: str, sub: str) -> None:
        try:
            self.query_one(f"#tile-{key}", Static).update(value)
            self.query_one(f"#tile-{key}-sub", Static).update(sub)
        except Exception:
            pass

    def _update_tiles(self) -> None:
        p = self._target()
        is_run = bool(p and self.rows.get(p.name, RowState()).running)
        ds = self.dstore.get(p.name) if p else DStore()
        if ds is None:
            ds = DStore()
        if not is_run:
            for key, _label, viz in TILE_DEFS:
                if key in ("l1", "l2", "draft"):
                    self._tile(key, "—", "standby")
                    self._tile_bar(key, None)
                elif viz == "spark":
                    unit = {"decode": "t/s", "prefill": "t/s", "stream": "MiB",
                            "dec": "", "pre": ""}[key]
                    self._tile(key, "0" + (" " + unit if unit else ""), "standby")
                    self._tile_spark(key, [0.0])
                else:
                    unit = {"nvme": "MiB/s", "ram": "MiB/s"}[key]
                    self._tile(key, "0 " + unit, "standby")
                    self._tile_bar(key, 0.0)
            return
        rs = self.rows.get(p.name, RowState())
        self._tile("decode", f"{rs.decode:.1f} t/s", "live /metrics")
        self._tile_spark("decode", self._hist["decode"].plain())
        self._tile("prefill", f"{rs.prefill:.1f} t/s", "live /metrics")
        self._tile_spark("prefill", self._hist["prefill"].plain())
        self._tile("l1", f"{ds.l1_hit_pct:.0f}%" if ds.l1_hit_pct is not None else "…",
                   "DSTORAGE since start")
        self._tile_bar("l1", ds.l1_hit_pct / 100.0 if ds.l1_hit_pct is not None else None)
        self._tile("l2", f"{ds.l2_hit_pct:.0f}%" if ds.l2_hit_pct is not None else "…",
                   "pinned vs NVMe")
        self._tile_bar("l2", ds.l2_hit_pct / 100.0 if ds.l2_hit_pct is not None else None)
        if ds.draft_accept is not None:
            self._tile("draft", f"{ds.draft_accept * 100:.0f}%",
                       f"{ds.draft_accepted}/{ds.draft_generated}")
            self._tile_bar("draft", ds.draft_accept)
        else:
            self._tile("draft", "—", "no MTP drafts")
            self._tile_bar("draft", None)
        self._tile("nvme", f"{ds.nvme_mibs:.0f} MiB/s" if ds.nvme_mibs else "…",
                   "decode IO shape")
        self._tile_bar("nvme", min(ds.nvme_mibs / 6000.0, 1.0) if ds.nvme_mibs else None)
        self._tile("ram", f"{ds.ram_vram_mibs:.0f} MiB/s" if ds.ram_vram_mibs else "…",
                   "pinned H2D")
        self._tile_bar("ram", min(ds.ram_vram_mibs / 3000.0, 1.0) if ds.ram_vram_mibs else None)
        self._tile("stream", f"{ds.streamed_mib:.0f} MiB" if ds.streamed_mib else "…",
                   "this run")
        self._tile_spark("stream", self._hist["stream"].plain())
        self._tile("dec", f"{rs.n_decode:,}", "live n_decode_total")
        self._tile_spark("dec", self._hist["dec"].plain())
        self._tile("pre", f"{ds.prefill_calls:,}" if ds.prefill_calls is not None else "…",
                   "DSTORAGE calls")
        self._tile_spark("pre", self._hist["pre"].plain())

    # -- log / status -------------------------------------------------------------------------------
    def _update_log(self) -> None:
        try:
            log = self.query_one("#log", RichLog)
        except Exception:
            return
        p = self._target()
        if not p or not p.log_file.exists():
            return
        try:
            lines = p.log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return
        log.clear()
        for line in lines[-5:]:
            log.write(line)

    def _status(self, msg: str) -> None:
        try:
            self.query_one("#status-line", Static).update(msg)
        except Exception:
            pass
        self._events.append(msg)

    # -- events ------------------------------------------------------------------------------------------
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        if event.data_table.id == "models-table" and event.row_key:
            name = str(event.row_key.value)
            if name in self.profiles:
                self.selected = name
                self.refresh_all()

    @on(Input.Changed)
    def _on_input(self, event: Input.Changed) -> None:
        if event.input.id and event.input.id.startswith("f-"):
            self._mark_dirty()

    @on(Select.Changed)
    def _on_select(self, _event: Select.Changed) -> None:
        self._mark_dirty()

    def _retitle_section(self, col: Collapsible, title: str) -> None:
        fields = dict((t, fs) for t, fs in SECTIONS).get(title, [])
        n = sum(1 for k, _, _ in fields if k in self._dirty)
        sym = "▼" if not col.collapsed else "▶"
        try:
            col.title = f"{sym} {title} ({n}*)" if n else f"{sym} {title}"
        except Exception:
            pass

    def on_collapsible_expanded(self, event: Collapsible.Expanded) -> None:
        for title, _fs in SECTIONS:
            if event.collapsible.id == f"sec-{title}":
                self._retitle_section(event.collapsible, title)

    def on_collapsible_collapsed(self, event: Collapsible.Collapsed) -> None:
        for title, _fs in SECTIONS:
            if event.collapsible.id == f"sec-{title}":
                self._retitle_section(event.collapsible, title)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-start":
            self.action_start()
        elif bid == "btn-stop":
            self.action_stop()
        elif bid == "btn-refresh":
            self.action_refresh()
        elif bid == "btn-apply":
            self.action_apply()
        elif bid == "btn-revert":
            self.action_revert()
        elif bid == "btn-reset":
            self.action_reset()

    # -- actions ---------------------------------------------------------------------------------------------
    def action_start(self) -> None:
        p = self._target()
        if not p:
            self._status("select a model first")
            return
        if self._dirty:
            self._status("unsaved changes — a to apply, u to revert first")
            return
        self._status(f"starting {p.name}…")
        ok, msg = runner.up(p)
        self._status(msg.splitlines()[0][:100])
        self.refresh_all()

    def action_stop(self) -> None:
        p = self._target()
        if not p or not self.rows.get(p.name, RowState()).running:
            self._status("nothing running")
            return

        def _done(confirmed: bool | None) -> None:
            if confirmed:
                ok, msg = runner.down(p, force=True)
                self._status(msg.splitlines()[0][:100])
                self.refresh_all()

        self.push_screen(ConfirmStop(p.name), _done)

    def action_refresh(self) -> None:
        self.refresh_all()
        self._status("refreshed")

    def _collect_env(self, prof: P.Profile) -> dict[str, str]:
        env = dict(prof.env)
        for key, _label, kind in ESSENTIALS + [f for _, fs in SECTIONS for f in fs]:
            v = self._form_value(key)
            if kind == "spec":
                env[key] = SPEC_TO_ENV.get(v, "draft-mtp")
            else:
                env[key] = v
        return env

    def _validate_ints(self, env: dict[str, str]) -> list[str]:
        bad: list[str] = []
        for key, _label, kind in ESSENTIALS + [f for _, fs in SECTIONS for f in fs]:
            if kind == "int" and env.get(key, ""):
                try:
                    int(env[key])
                except ValueError:
                    bad.append(f"{key} is not an integer: {env[key]!r}")
        return bad

    def action_apply(self) -> None:
        p = self._target()
        if not p:
            return
        env = self._collect_env(p)
        bad = self._validate_ints(env)
        if bad:
            self._status("invalid: " + bad[0][:90])
            return
        if p.is_bundled:
            clone_name = (p.name + "-custom").replace("_", "-")
            p = P.Profile.create_app(clone_name, env, script=p.script_rel)
            self._status(f"cloned to editable {clone_name} (bundled is read-only)")
        else:
            p.env = env
        try:
            p.save()
            p.materialize_env()
        except ValueError as exc:
            self._status(str(exc))
            return
        issues = p.validate()
        self.profiles[p.name] = p
        self.selected = p.name
        self._form_profile = None
        if issues:
            self._status("saved with warnings: " + issues[0][:90])
        else:
            note = (" — restart to take effect"
                    if self.rows.get(p.name, RowState()).running else "")
            self._status(f"saved {p.name}{note}")
        self.refresh_all()

    def action_revert(self) -> None:
        p = self._target()
        if p:
            self._fill_form(p)
            self._status("reverted to saved values")

    def action_reset(self) -> None:
        """Reset form to the on-disk defaults (fresh discover)."""
        p = self._target()
        if not p:
            return
        try:
            fresh = P.discover().get(p.name)
        except Exception:
            fresh = None
        if fresh is None:
            self._status("no saved defaults found")
            return
        self._fill_form(fresh)
        self._mark_dirty()
        self._status("reset to defaults — a to apply")


def kind_of(key: str) -> str:
    for _k, _l, kind in ESSENTIALS + [f for _, fs in SECTIONS for f in fs]:
        if _k == key:
            return kind
    return "text"


def run() -> None:
    SiphonTerm().run()


if __name__ == "__main__":
    run()
