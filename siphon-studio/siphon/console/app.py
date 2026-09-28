"""Siphon console v2 — dashboard-first Textual TUI.

Visual language translated from `siphon-console.html` (premium ops-console:
ink canvas, indigo accent, KPI strip with sparklines, roster + editable
config form, 10-tile metrics grid, activity timeline). Data strictly from
`siphon.core` (runner/monitor/profiles) + `siphon.tui.data` (host telemetry)
+ `siphon.console.dstore` (DSTORAGE log-tail parsing).

Layout: sidebar (brand/nav/env) + main (topbar/KPIs/roster+config/tiles/activity).
Keys: s start · x stop · ctrl+r refresh · a apply · u revert · q quit.
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
    Digits,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RadioButton,
    RadioSet,
    RichLog,
    Sparkline,
    Static,
)

from siphon.core import profiles as P
from siphon.core import runner
from siphon.tui.data import Ring, fetch_sys_stats

from .dstore import DStore, parse_log_tail

REFRESH_SECONDS = 2.0

SPEC_CHOICES = [
    ("mtp", "draft-mtp"),
    ("simple", "draft-simple"),
    ("dflash", "draft-dflash"),
    ("none", "none"),
    ("off", "off"),
]
SPEC_TO_ENV = dict(SPEC_CHOICES)
ENV_TO_SPEC = {v: k for k, v in SPEC_CHOICES}


def _bar(pct: float, width: int = 10) -> str:
    pct = max(0.0, min(1.0, pct))
    full = int(round(pct * width))
    return "▓" * full + "░" * (width - full)


def _cls(pct: float | None, ok: float, warn: float) -> str:
    """Traffic-light class: ok above `ok`, warn band, bad below `warn`."""
    if pct is None:
        return ""
    if pct >= ok:
        return "ok"
    if pct >= warn:
        return "warn"
    return "bad"


@dataclass
class TileDef:
    key: str
    label: str


TILES: list[TileDef] = [
    TileDef("decode", "Decode"),
    TileDef("prefill", "Prefill"),
    TileDef("l1", "L1 hit"),
    TileDef("l2", "L2 hit"),
    TileDef("draft", "Draft accept"),
    TileDef("nvme", "NVMe / GDS"),
    TileDef("ram", "RAM → VRAM"),
    TileDef("stream", "Streamed"),
    TileDef("dec", "Decode calls"),
    TileDef("pre", "Prefill calls"),
]


class ConfirmStop(ModalScreen[bool]):
    BINDINGS = [("escape", "cancel", "cancel")]

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name

    def compose(self) -> ComposeResult:
        with Vertical(id="help-box"):
            yield Static(f"Stop {self._name}?", id="help-title")
            yield Label("The server process will be shut down.", classes="help-hint")
            with Horizontal():
                yield Button("Stop", id="cf-yes", variant="error")
                yield Button("Cancel", id="cf-no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "cf-yes")

    def action_cancel(self) -> None:
        self.dismiss(False)


class SiphonConsole(App):
    TITLE = "Siphon"
    SUB_TITLE = "ops console"
    CSS_PATH = "console.tcss"

    BINDINGS = [
        Binding("s", "start", "Start"),
        Binding("x", "stop", "Stop"),
        Binding("ctrl+r", "refresh", "Refresh"),
        Binding("a", "apply", "Apply"),
        Binding("u", "revert", "Revert"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.profiles: dict[str, P.Profile] = P.discover()
        self.selected: str | None = None
        self.running: dict[str, bool] = {}
        self.health: dict[str, str] = {}
        self.decode: dict[str, float] = {}
        self.prefill: dict[str, float] = {}
        self.n_decode: dict[str, int] = {}
        self.latency: dict[str, float] = {}
        self.dstore: dict[str, DStore] = {}
        self._hist = {k: Ring(60) for k in ("vram", "gpu", "decode", "l1", "draft")}
        self._sys = fetch_sys_stats()
        self._events: deque[tuple[str, str, str]] = deque(maxlen=30)
        self._dirty = False
        self._form_profile: str | None = None

    # -- theme ----------------------------------------------------------
    @staticmethod
    def make_theme() -> Theme:
        return Theme(
            name="siphon-console",
            primary="#6366F1",
            secondary="#818CF8",
            success="#10B981",
            warning="#F59E0B",
            error="#F43F5E",
            foreground="#E8ECF5",
            background="#0B0F19",
            surface="#11172A",
            panel="#0E1423",
            dark=True,
        )

    # -- compose ---------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="body"):
            with Vertical(id="sidebar"):
                yield Static("◆ Siphon", id="brand-name")
                yield Static("OPS CONSOLE", id="brand-sub")
                yield Label("Workspace", classes="nav-label")
                with ListView(id="nav"):
                    yield ListItem(Label("Overview"))
                    yield ListItem(Label("Models"))
                    yield ListItem(Label("Activity"))
                    yield ListItem(Label("Settings"))
                with Vertical(id="env-card"):
                    yield Static("local · RTX 4050", id="env-card-title")
                    yield Static("—", id="env-card-sub", classes="env-dim")
                    yield Static("", id="env-card-bar", classes="env-dim")
            with VerticalScroll(id="main"):
                with Horizontal(id="topbar"):
                    with Vertical(id="title-block"):
                        yield Static("Siphon / Local", id="crumb")
                        yield Static("Overview", id="page-title")
                    yield Static("", id="env-pill")
                    with Horizontal(id="btn-row"):
                        yield Button("▶ Start", id="btn-start", variant="primary")
                        yield Button("■ Stop", id="btn-stop", variant="error")
                        yield Button("Refresh", id="btn-refresh")
                with Horizontal(id="kpis"):
                    for key, label in (
                        ("vram", "VRAM"),
                        ("gpu", "GPU Util"),
                        ("decode", "Decode"),
                        ("l1", "L1 Cache Hit"),
                        ("draft", "Draft Acceptance"),
                    ):
                        with Vertical(classes="kpi-card"):
                            yield Static(label, classes="kpi-label")
                            yield Digits("0", id=f"kpi-{key}")
                            yield Static("—", id=f"kpi-{key}-delta", classes="kpi-delta")
                            yield Sparkline([0.0], id=f"spark-{key}", classes="spark")
                with Horizontal(id="grid2"):
                    with Vertical(classes="panel", id="roster-panel"):
                        with Horizontal(classes="panel-head"):
                            yield Static("Model roster", classes="panel-title")
                            yield Static("", id="roster-hint", classes="panel-hint")
                        yield ListView(id="roster")
                    with Vertical(classes="panel", id="config-panel"):
                        with Horizontal(classes="panel-head"):
                            yield Static("Configuration", classes="panel-title")
                            yield Static("", id="cfg-sub", classes="panel-hint")
                        yield Static("Model path", classes="field-label")
                        yield Input(id="f-model")
                        with Horizontal():
                            with Vertical():
                                yield Static("Context", classes="field-label")
                                yield Input(id="f-ctx")
                            with Vertical():
                                yield Static("Batch", classes="field-label")
                                yield Input(id="f-batch")
                            with Vertical():
                                yield Static("Port", classes="field-label")
                                yield Input(id="f-port")
                        yield Static("L1 VRAM cache (MiB)", classes="field-label")
                        with Horizontal():
                            yield Button("−", id="l1-minus")
                            yield Input(id="f-l1")
                            yield Button("+", id="l1-plus")
                        yield Static("L2 pinned memory (MiB)", classes="field-label")
                        with Horizontal():
                            yield Button("−", id="l2-minus")
                            yield Input(id="f-l2")
                            yield Button("+", id="l2-plus")
                        yield Static("Speculator", classes="field-label")
                        with RadioSet(id="spec"):
                            for label, _env in SPEC_CHOICES:
                                yield RadioButton(label)
                        with Horizontal(classes="status-grid"):
                            with Vertical(classes="srow"):
                                yield Static("LATENCY P50", classes="srow-k")
                                yield Static("—", id="st-lat", classes="srow-v")
                            with Vertical(classes="srow"):
                                yield Static("ENDPOINT", classes="srow-k")
                                yield Static("—", id="st-url", classes="srow-v")
                        with Horizontal(classes="status-grid"):
                            with Vertical(classes="srow"):
                                yield Static("SUITE", classes="srow-k")
                                yield Static("—", id="st-suite", classes="srow-v")
                            with Vertical(classes="srow"):
                                yield Static("STATE", classes="srow-k")
                                yield Static("—", id="st-state", classes="srow-v")
                        with Horizontal(classes="status-grid"):
                            yield Button("Apply changes", id="btn-apply", variant="primary")
                            yield Button("Revert", id="btn-revert")
                with Horizontal(id="tiles-row1"):
                    for t in TILES[:5]:
                        with Vertical(classes="tile"):
                            yield Static(t.label, classes="tile-label")
                            yield Static("—", id=f"tile-{t.key}", classes="tile-value")
                            yield Static("", id=f"tile-{t.key}-sub", classes="tile-sub")
                            yield Static("", id=f"tile-{t.key}-bar", classes="tile-bar")
                with Horizontal(id="tiles-row2"):
                    for t in TILES[5:]:
                        with Vertical(classes="tile"):
                            yield Static(t.label, classes="tile-label")
                            yield Static("—", id=f"tile-{t.key}", classes="tile-value")
                            yield Static("", id=f"tile-{t.key}-sub", classes="tile-sub")
                            yield Static("", id=f"tile-{t.key}-bar", classes="tile-bar")
                with Vertical(classes="panel", id="activity-panel"):
                    with Horizontal(classes="panel-head"):
                        yield Static("Activity", classes="panel-title")
                        yield Static("live tail", classes="panel-hint")
                    yield RichLog(id="timeline", highlight=True, wrap=True)
                    yield Static("", id="mono")
                yield Static("", id="status-line")
        yield Footer()

    # -- lifecycle --------------------------------------------------------
    def on_mount(self) -> None:
        self.register_theme(self.make_theme())
        self.theme = "siphon-console"
        self._event("Console ready", "pick a model, press s to start")
        self.refresh_all()
        self.set_interval(REFRESH_SECONDS, self._tick)

    def _tick(self) -> None:
        try:
            self.refresh_all()
        except Exception as exc:  # never take down the UI on a bad probe
            self._status(f"poll error: {exc}")

    # -- data --------------------------------------------------------------
    def _target(self) -> P.Profile | None:
        if self.selected and self.selected in self.profiles:
            return self.profiles[self.selected]
        return None

    def refresh_all(self) -> None:
        self.profiles = P.discover()
        if not self.selected or self.selected not in self.profiles:
            running = [n for n, p in self.profiles.items()
                       if self.running.get(n) or runner.status(p).running]
            self.selected = (running or list(self.profiles))[0] if self.profiles else None
        for name, prof in self.profiles.items():
            try:
                st = runner.status(prof)
            except Exception:
                continue
            self.running[name] = st.running
            self.health[name] = st.health.status
            self.decode[name] = st.metrics.decode_tps
            self.prefill[name] = st.metrics.prefill_tps
            self.n_decode[name] = st.metrics.n_decode
            self.latency[name] = st.health.latency_ms
            try:
                self.dstore[name] = parse_log_tail(prof.log_file)
            except Exception:
                self.dstore[name] = DStore()
        try:
            self._sys = fetch_sys_stats()
        except Exception:
            pass
        self._update_sidebar()
        self._update_topbar()
        self._update_kpis()
        self._rebuild_roster()
        self._update_form_status()
        self._update_tiles()
        self._update_mono()

    # -- sidebar / topbar ----------------------------------------------------
    def _update_sidebar(self) -> None:
        s = self._sys
        total = s.gpu_mib_total / 1024 if s.gpu_mib_total else 6.0
        used = s.gpu_mib_used / 1024
        self.query_one("#env-card-sub", Static).update(
            f"{total:.1f} GB VRAM · {used:.1f} GiB in use")
        pct = used / total if total else 0.0
        self.query_one("#env-card-bar", Static).update(Text(_bar(pct, 20), style="#6366F1"))

    def _update_topbar(self) -> None:
        s = self._sys
        free = s.gpu_mib_free / 1024 if s.gpu_mib_total else 0.0
        total = s.gpu_mib_total / 1024 if s.gpu_mib_total else 6.0
        self.query_one("#env-pill", Static).update(
            Text(f"● local · RTX 4050 · {total - free:.1f} / {total:.1f} GiB"))
        p = self._target()
        is_run = bool(p and self.running.get(p.name))
        self.query_one("#btn-start", Button).disabled = is_run or p is None
        self.query_one("#btn-stop", Button).disabled = not is_run

    # -- KPIs ------------------------------------------------------------------
    def _update_kpis(self) -> None:
        s = self._sys
        p = self._target()
        is_run = bool(p and self.running.get(p.name))
        ds = self.dstore.get(p.name) if p else DStore()
        vram = s.gpu_mib_used / 1024
        gpu = s.gpu_util
        dec = self.decode.get(p.name, 0.0) if (p and is_run) else 0.0
        l1 = ds.l1_hit_pct if (ds and ds.l1_hit_pct is not None and is_run) else 0.0
        draft = (ds.draft_accept * 100.0) if (ds and ds.draft_accept is not None and is_run) else 0.0
        self._hist["vram"].push(vram)
        self._hist["gpu"].push(gpu)
        self._hist["decode"].push(dec)
        self._hist["l1"].push(l1)
        self._hist["draft"].push(draft)

        total = s.gpu_mib_total / 1024 if s.gpu_mib_total else 6.0
        vals = {
            "vram": (f"{vram:.1f}", f"/ {total:.1f} GiB used", self._hist["vram"].plain()),
            "gpu": (f"{gpu:.0f}", "% util", self._hist["gpu"].plain()),
            "decode": (f"{dec:.1f}", "tok/s", self._hist["decode"].plain()),
            "l1": (f"{l1:.0f}", "% hit", self._hist["l1"].plain()),
            "draft": (f"{draft:.0f}" if is_run and ds and ds.draft_accept is not None else "—",
                      "% accepted", self._hist["draft"].plain()),
        }
        for key, (num, unit, hist) in vals.items():
            try:
                self.query_one(f"#kpi-{key}", Digits).update(num)
                d = self.query_one(f"#kpi-{key}-delta", Static)
                d.update(unit)
                d.set_classes(f"kpi-delta {'up' if is_run else 'mut'}")
                sp = self.query_one(f"#spark-{key}", Sparkline)
                sp.data = hist or [0.0]
            except Exception:
                pass

    # -- roster ------------------------------------------------------------------
    def _fit_badge(self, prof: P.Profile) -> tuple[str, bool]:
        try:
            l1 = int(prof.env.get("MOE_GPU_CACHE_MIB", "0") or 0) / 1024
            ctx = int(prof.env.get("CTX", "0") or 0)
        except ValueError:
            return "unverified", False
        total = self._sys.gpu_mib_total / 1024 if self._sys.gpu_mib_total else 6.0
        est = l1 + 2.4 + (ctx / 65536 * 0.6 if ctx else 0.0)
        pct = est / total if total else 1.0
        if est > total:
            return "NO-FIT", True
        return f"Fits · {pct * 100:.0f}%", pct > 0.85

    def _rebuild_roster(self) -> None:
        try:
            lv = self.query_one("#roster", ListView)
        except Exception:
            return
        lv.clear()
        for name, prof in self.profiles.items():
            is_run = self.running.get(name, False)
            h = self.health.get(name, "down")
            if is_run and h == "ok":
                chip, rate = Text("RUNNING", style="bold #10B981"), f"{self.decode.get(name, 0.0):.1f} t/s"
            elif is_run:
                chip, rate = Text("LOADING", style="#F59E0B"), "starting…"
            else:
                chip, rate = Text("IDLE", style="#6B7590"), "standby"
            badge, tight = self._fit_badge(prof)
            line = Text()
            line.append(f"{name}  ", style="bold")
            line.append(chip).append("  ")
            line.append(rate, style="#9AA3B8").append("  ")
            line.append(badge, style="#FCD34D" if tight else "#818CF8")
            item = ListItem(Label(line))
            item.profile_name = name  # type: ignore[attr-defined]
            lv.append(item)
        names = list(self.profiles)
        if self.selected in names:
            lv.index = names.index(self.selected)
        n_run = sum(1 for v in self.running.values() if v)
        try:
            self.query_one("#roster-hint", Static).update(f"{n_run} running · {len(names) - n_run} idle")
        except Exception:
            pass

    # -- config form ---------------------------------------------------------------
    def _form_values(self) -> dict[str, str]:
        def val(wid: str) -> str:
            try:
                return self.query_one(wid, Input).value.strip()
            except Exception:
                return ""

        spec = "mtp"
        try:
            rs = self.query_one("#spec", RadioSet)
            idx = rs.pressed_index
            if idx is not None and 0 <= idx < len(SPEC_CHOICES):
                spec = SPEC_CHOICES[idx][0]
        except Exception:
            pass
        return {"MODEL": val("#f-model"), "CTX": val("#f-ctx"), "BATCH": val("#f-batch"),
                "PORT": val("#f-port"), "MOE_GPU_CACHE_MIB": val("#f-l1"),
                "MOE_PINNED_CACHE_MIB": val("#f-l2"), "SPEC": spec}

    def _fill_form(self, prof: P.Profile) -> None:
        def set_in(wid: str, v: str) -> None:
            try:
                self.query_one(wid, Input).value = v
            except Exception:
                pass

        set_in("#f-model", prof.env.get("MODEL", ""))
        set_in("#f-ctx", prof.env.get("CTX", ""))
        set_in("#f-batch", prof.env.get("BATCH", ""))
        set_in("#f-port", prof.env.get("PORT", ""))
        set_in("#f-l1", prof.env.get("MOE_GPU_CACHE_MIB", ""))
        set_in("#f-l2", prof.env.get("MOE_PINNED_CACHE_MIB", ""))
        want = ENV_TO_SPEC.get(prof.env.get("SPEC_TYPE", ""), "mtp")
        try:
            rs = self.query_one("#spec", RadioSet)
            for i, (label, _e) in enumerate(SPEC_CHOICES):
                if label == want:
                    rs.pressed_index = i
                    break
        except Exception:
            pass
        self._form_profile = prof.name
        self._dirty = False

    def _mark_dirty(self) -> None:
        p = self._target()
        if not p or self._form_profile != p.name:
            return
        cur = self._form_values()
        saved = {k: p.env.get(k, "") for k in ("MODEL", "CTX", "BATCH", "PORT",
                                               "MOE_GPU_CACHE_MIB", "MOE_PINNED_CACHE_MIB")}
        dirty = any(cur[k] != saved[k] for k in saved) or cur["SPEC"] != ENV_TO_SPEC.get(
            p.env.get("SPEC_TYPE", ""), "mtp")
        if dirty != self._dirty:
            self._dirty = dirty
            self._status("● unsaved changes — a to apply, u to revert" if dirty
                         else f"{p.name} config in sync")

    def _update_form_status(self) -> None:
        p = self._target()
        if p is None:
            return
        if self._form_profile != p.name:
            self._fill_form(p)
        try:
            self.query_one("#cfg-sub", Static).update(
                p.name + ("  ●" if self._dirty else ""))
            suite = (p.resolve("EXPERTS") or "—").split("/")[-1].replace(".txt", "")
            self.query_one("#st-lat", Static).update(
                f"{self.latency.get(p.name, 0.0):.0f} ms" if self.running.get(p.name) else "—")
            self.query_one("#st-url", Static).update(f":{p.port} / v1")
            self.query_one("#st-suite", Static).update(suite or "—")
            h = self.health.get(p.name, "down")
            state = {"ok": "Running", "loading": "Loading"}.get(h, "Idle")
            self.query_one("#st-state", Static).update(
                Text(state, style="bold #10B981" if h == "ok"
                     else ("#F59E0B" if h == "loading" else "#6B7590")))
        except Exception:
            pass

    def _step(self, wid: str, delta: int, lo: int, hi: int) -> None:
        try:
            inp = self.query_one(wid, Input)
            cur = int(inp.value.strip() or "0")
            inp.value = str(max(lo, min(hi, cur + delta)))
            self._mark_dirty()
        except (ValueError, Exception):
            pass

    # -- metric tiles ----------------------------------------------------------------
    def _tile(self, key: str, value: str, sub: str, pct: float | None) -> None:
        try:
            self.query_one(f"#tile-{key}", Static).update(value)
            self.query_one(f"#tile-{key}-sub", Static).update(sub)
            b = self.query_one(f"#tile-{key}-bar", Static)
            if pct is None:
                b.update("—")
                b.set_classes("tile-bar")
            else:
                b.update(_bar(pct))
                if pct >= 0.8:
                    b.set_classes("tile-bar ok")
                elif pct >= 0.5:
                    b.set_classes("tile-bar warn")
                else:
                    b.set_classes("tile-bar bad")
        except Exception:
            pass

    def _update_tiles(self) -> None:
        p = self._target()
        is_run = bool(p and self.running.get(p.name))
        ds = self.dstore.get(p.name) if p else DStore()
        if ds is None:
            ds = DStore()
        if not is_run:
            for t in TILES:
                standby = "—" if t.key in ("l1", "l2", "draft") else "0"
                unit = {"decode": " t/s", "prefill": " t/s", "l1": "", "l2": "",
                        "draft": "", "nvme": " MiB/s", "ram": " MiB/s",
                        "stream": " MiB", "dec": "", "pre": ""}[t.key]
                self._tile(t.key, standby + unit, "standby", 0.0 if t.key not in ("l1", "l2", "draft") else None)
            return

        dec = self.decode.get(p.name, 0.0)
        pre = self.prefill.get(p.name, 0.0)
        self._tile("decode", f"{dec:.1f} t/s", "live /metrics", min(dec / 25.0, 1.0))
        self._tile("prefill", f"{pre:.1f} t/s", "live /metrics", min(pre / 35.0, 1.0))
        self._tile("l1", f"{ds.l1_hit_pct:.0f}%" if ds.l1_hit_pct is not None else "…",
                   "DSTORAGE since start",
                   (ds.l1_hit_pct / 100.0) if ds.l1_hit_pct is not None else None)
        self._tile("l2", f"{ds.l2_hit_pct:.0f}%" if ds.l2_hit_pct is not None else "…",
                   "pinned vs NVMe",
                   (ds.l2_hit_pct / 100.0) if ds.l2_hit_pct is not None else None)
        if ds.draft_accept is not None:
            self._tile("draft", f"{ds.draft_accept * 100:.0f}%",
                       f"{ds.draft_accepted}/{ds.draft_generated} accepted",
                       ds.draft_accept)
        else:
            self._tile("draft", "—", "no MTP drafts", None)
        self._tile("nvme", f"{ds.nvme_mibs:.0f} MiB/s" if ds.nvme_mibs else "…",
                   "decode IO shape",
                   min(ds.nvme_mibs / 6000.0, 1.0) if ds.nvme_mibs else None)
        self._tile("ram", f"{ds.ram_vram_mibs:.0f} MiB/s" if ds.ram_vram_mibs else "…",
                   "pinned H2D",
                   min(ds.ram_vram_mibs / 3000.0, 1.0) if ds.ram_vram_mibs else None)
        self._tile("stream", f"{ds.streamed_mib:.0f} MiB" if ds.streamed_mib else "…",
                   "this run", None)
        self._tile("dec", f"{self.n_decode.get(p.name, 0):,}", "live n_decode_total", None)
        self._tile("pre", f"{ds.prefill_calls:,}" if ds.prefill_calls is not None else "…",
                   "DSTORAGE calls", None)

    # -- activity / mono / status ---------------------------------------------------------
    def _event(self, title: str, desc: str) -> None:
        import datetime
        self._events.append((datetime.datetime.now().strftime("%H:%M:%S"), title, desc))
        try:
            log = self.query_one("#timeline", RichLog)
            log.write(Text(f"{self._events[-1][0]}  {title} — {desc}", style="#9AA3B8"))
        except Exception:
            pass

    def _update_mono(self) -> None:
        p = self._target()
        try:
            box = self.query_one("#mono", Static)
        except Exception:
            return
        if not p or not self.running.get(p.name):
            box.update("siphon tail — standby · 0.0 t/s")
            return
        ds = self.dstore.get(p.name) or DStore()
        parts = [f"siphon tail — {p.name}"]
        if ds.decode_tps:
            parts.append(f"decode {ds.decode_tps:.1f} t/s")
        if ds.l1_hit_pct is not None:
            parts.append(f"l1 {ds.l1_hit_pct:.0f}%")
        box.update(" · ".join(parts))

    def _status(self, msg: str) -> None:
        try:
            self.query_one("#status-line", Static).update(msg)
        except Exception:
            pass

    # -- events ------------------------------------------------------------------------------
    def on_list_view_selected(self, event: ListView.Selected) -> None:
        lv = event.list_view
        if lv is None:
            return
        if lv.id == "roster":
            name = getattr(event.item, "profile_name", None)
            if name and name in self.profiles:
                self.selected = name
                self.refresh_all()
        elif lv.id == "nav":
            try:
                label = str(event.item.query_one(Label).renderable)
            except Exception:
                return
            target = {"Overview": "#kpis", "Models": "#roster-panel",
                      "Activity": "#activity-panel", "Settings": "#config-panel"}.get(label)
            if target:
                try:
                    self.query_one("#main", VerticalScroll).scroll_to_widget(
                        self.query_one(target))
                except Exception:
                    pass

    @on(Input.Changed)
    def _on_input(self, event: Input.Changed) -> None:
        if event.input.id and event.input.id.startswith(("f-",)):
            self._mark_dirty()

    @on(RadioSet.Changed)
    def _on_spec(self, _event: RadioSet.Changed) -> None:
        self._mark_dirty()

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
        elif bid == "l1-minus":
            self._step("#f-l1", -64, 256, 4096)
        elif bid == "l1-plus":
            self._step("#f-l1", 64, 256, 4096)
        elif bid == "l2-minus":
            self._step("#f-l2", -128, 256, 8192)
        elif bid == "l2-plus":
            self._step("#f-l2", 128, 256, 8192)

    # -- actions -------------------------------------------------------------------------------
    def action_start(self) -> None:
        p = self._target()
        if not p:
            self._status("select a model first")
            return
        if self._dirty:
            self._status("unsaved changes — press a to apply first")
            return
        self._status(f"starting {p.name}…")
        self._event("Start requested", p.name)
        ok, msg = runner.up(p)
        self._event("Started" if ok else "Start refused", msg.splitlines()[0][:90])
        self._status(msg.splitlines()[0][:100])
        self.refresh_all()

    def action_stop(self) -> None:
        p = self._target()
        if not p or not self.running.get(p.name):
            self._status("nothing running")
            return

        def _done(confirmed: bool | None) -> None:
            if confirmed:
                ok, msg = runner.down(p, force=True)
                self._event("Stopped" if ok else "Stop issue", msg.splitlines()[0][:90])
                self._status(msg.splitlines()[0][:100])
                self.refresh_all()

        self.push_screen(ConfirmStop(p.name), _done)

    def action_refresh(self) -> None:
        self.refresh_all()
        self._status("refreshed")

    def action_apply(self) -> None:
        p = self._target()
        if not p:
            return
        vals = self._form_values()
        env = dict(p.env)
        for k in ("MODEL", "CTX", "BATCH", "PORT", "MOE_GPU_CACHE_MIB", "MOE_PINNED_CACHE_MIB"):
            env[k] = vals[k]
        env["SPEC_TYPE"] = SPEC_TO_ENV.get(vals["SPEC"], "draft-mtp")
        if p.is_bundled:
            clone_name = (p.name + "-custom").replace("_", "-")
            p = P.Profile.create_app(clone_name, env, script=p.script_rel)
            self._event("Cloned profile", f"{p.name} (bundled is read-only)")
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
            self._event("Saved with warnings", issues[0][:90])
        else:
            self._status(f"saved {p.name}" + (" — restart to take effect" if self.running.get(p.name) else ""))
            self._event("Config saved", p.name)
        self.refresh_all()

    def action_revert(self) -> None:
        p = self._target()
        if p:
            self._fill_form(p)
            self._status("reverted to saved values")


def run() -> None:
    SiphonConsole().run()


if __name__ == "__main__":
    run()
