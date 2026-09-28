"""SiphonTUI - Textual control center for siphon.cpp.

Built strictly on siphon.core for profile/runner logic; adds a presentation
layer (server /slots & /props, system stats) for richer, live visuals. No
siphon.core logic is duplicated here.

Tabs (keys 1-6):
  Dashboard  searchable profile picker + live detail + start/stop/restart/web
  Metrics    live throughput charts, token counters, context + RAM bars
  Slots      per-slot table from the running server (state / n_past / n_ctx)
  Logs       live tail with follow-toggle
  Config     read-only resolved env for the selected profile (human labels)
  Doctor     environment checks + host summary

Global bindings (also in the footer and the '?' help overlay):
  /           filter profiles       s   start selected profile
  d / x       stop selected         r   restart selected profile
  o           open web UI           ctrl+r  refresh
  1-6         switch tab            T   toggle dark/light theme
  ?           help overlay          q   quit
"""

from __future__ import annotations

import sys
import webbrowser
from dataclasses import dataclass

from rich.text import Text

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import (
    Button,
    DataTable,
    Digits,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    OptionList,
    ProgressBar,
    RichLog,
    Sparkline,
    Static,
    Switch,
    TabbedContent,
    TabPane,
)

from siphon.core import doctor, runner
from siphon.core.profiles import KNOB_LABELS, Profile, discover

from .data import (
    Ring,
    fetch_model_report,
    fetch_props,
    fetch_slots,
    fetch_sys_stats,
)

REFRESH_SECONDS = 2.0

# --------------------------------------------------------------------------- #
# Lightweight per-profile snapshot (status + health + metrics).
# --------------------------------------------------------------------------- #


@dataclass
class ProfileState:
    profile: Profile
    running: bool
    health: str          # "running" | "loading" | "down"
    decode_tps: float
    prefill_tps: float
    n_processing: int
    driver: str
    latency_ms: float = 0.0

    @property
    def status_chip(self) -> Text:
        if self.health == "running":
            return Text("RUNNING", style="bold #3fb950")
        if self.health == "loading":
            return Text("loading", style="#d29922")
        return Text("idle", style="#8b949e")


def poll_states(profiles: dict[str, Profile]) -> dict[str, ProfileState]:
    out: dict[str, ProfileState] = {}
    for name, prof in profiles.items():
        st = runner.status(prof)
        m = st.metrics
        out[name] = ProfileState(
            profile=prof,
            running=st.running,
            health=st.health.status,
            decode_tps=m.decode_tps,
            prefill_tps=m.prefill_tps,
            n_processing=m.n_processing,
            driver=st.driver,
            latency_ms=st.health.latency_ms,
        )
    return out


# --------------------------------------------------------------------------- #
# Help overlay - a modal listing every binding + controls.
# --------------------------------------------------------------------------- #


class HelpScreen(ModalScreen[None]):
    BINDINGS = [("escape", "dismiss", "close"), ("q", "dismiss", "close")]

    def compose(self) -> ComposeResult:
        with Vertical(id="help-box"):
            yield Static("Help - siphon-studio", id="help-title")
            yield Static(_HELP_TEXT, id="help-body")
            yield Label("press q or escape to close", classes="help-hint")

    def action_dismiss(self) -> None:
        self.dismiss(None)


_HELP_TEXT = """\
CONTROL      ACTION
/            filter profiles on Dashboard
s            start the selected profile
d / x        stop the selected profile
r            restart the selected profile
o            open the selected profile's web UI
ctrl+r       force a refresh now
1 ... 6      switch tabs (Dashboard Metrics Slots Logs Config Doctor)
T            toggle dark / light theme
?            show this help
q            quit

From inside a tab:
  select a profile on the Dashboard, then use the button bar or the keys above.
  Start boots the server, Stop tears it down, Restart cycles both.

Data sources: siphon.core (runner + monitor) for status/metrics; the server's
/props, /slots and /v1/models endpoints for the Slots/Config extras; psutil +
nvidia-smi for the host gauges on Metrics. All of them fail quietly if down.
"""


class SiphonTUI(App):
    TITLE = "siphon-studio"
    SUB_TITLE = "siphon.cpp control center"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("s", "start", "Start"),
        Binding("d", "down", "Stop"),
        Binding("x", "down", "Stop"),
        Binding("r", "restart", "Restart"),
        Binding("o", "open_web", "Web"),
        Binding("ctrl+r", "refresh", "Refresh"),
        Binding("slash", "focus_search", "Find"),
        Binding("1", "tab_dashboard", "Dashboard"),
        Binding("2", "tab_metrics", "Metrics"),
        Binding("3", "tab_slots", "Slots"),
        Binding("4", "tab_logs", "Logs"),
        Binding("5", "tab_config", "Config"),
        Binding("6", "tab_doctor", "Doctor"),
        Binding("T", "toggle_theme", "Theme"),
        Binding("?", "help", "Help"),
        Binding("q", "quit", "Quit"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.profiles = discover()
        self.selected: str | None = None
        self.states: dict[str, ProfileState] = {}
        self.last_metrics: dict[str, tuple[list[float], float]] = {}
        self._history_decode = Ring(90)
        self._history_prefill = Ring(90)
        self._sys = fetch_sys_stats()
        self._props: dict[str, object] = {}
        self._slots: dict[str, object] = {}
        self._model: dict[str, str] = {}
        self._filter = ""
        self._follow_logs = True
        self._doctor_run = False

    def on_mount(self) -> None:
        self.register_theme(self.make_dark_theme())
        self.register_theme(self.make_light_theme())
        self.theme = "siphon-dark"
        self._update_sys()
        self.refresh_all()
        self.set_interval(REFRESH_SECONDS, self.poll_tick)
        self._notify(
            "Control center ready - pick a profile, press s to start, ? for help."
        )

    # -- themes -------------------------------------------------------------
    @staticmethod
    def make_dark_theme() -> Theme:
        return Theme(
            name="siphon-dark",
            primary="#79c0ff",
            secondary="#8250df",
            success="#3fb950",
            warning="#d29922",
            error="#f85149",
            foreground="#c9d1d9",
            background="#0d1117",
            surface="#161b22",
            panel="#161b22",
            dark=True,
        )

    @staticmethod
    def make_light_theme() -> Theme:
        return Theme(
            name="siphon-light",
            primary="#0969da",
            secondary="#8250df",
            success="#1a7f37",
            warning="#9a6700",
            error="#cf222e",
            foreground="#1f2328",
            background="#ffffff",
            surface="#f6f8fa",
            panel="#f0f1f3",
            dark=False,
        )

    # -- compose ------------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="sysbar"):
            yield Static("SYS", id="sys-title", classes="sys-tile")
            yield Static("gpu —", id="sys-gpu", classes="sys-tile")
            yield Static("cpu —", id="sys-cpu", classes="sys-tile")
            yield Static("ram —", id="sys-ram", classes="sys-tile")
            yield Static("disk —", id="sys-disk", classes="sys-tile")
        with TabbedContent(initial="dashboard") as tabs:
            with TabPane("Dashboard", id="dashboard", classes="tabpane"):
                yield Input(placeholder="filter profiles…  ( / )", id="filter",
                          classes="hidden")
                with Horizontal(id="overview"):
                    yield Vertical(id="card-gpu", classes="sys-card gpu")
                    yield Vertical(id="card-cpu", classes="sys-card cpu")
                    yield Vertical(id="card-mem", classes="sys-card mem")
                    yield Vertical(id="card-drive", classes="sys-card drive")
                with Horizontal(id="dash-body"):
                    with Vertical(id="picker-col"):
                        yield Label("Profiles", classes="panel-title")
                        yield ListView(id="profiles")
                    with Vertical(id="detail-col"):
                        yield Static("—", id="detail", classes="panel panel-body")
                        yield Label("Context / slots usage", classes="panel-title")
                        yield Static("no running server", id="ctx-detail",
                                     classes="panel panel-body")
                with Horizontal(id="action-bar"):
                    yield Button("Start", id="btn-start", variant="success")
                    yield Button("Stop", id="btn-stop", variant="error")
                    yield Button("Restart", id="btn-restart")
                    yield Button("Web", id="btn-web")
                    yield Button("Refresh", id="btn-refresh")
                yield Static("", id="dash-status", classes="dash-status")
            with TabPane("Metrics", id="metrics", classes="tabpane"):
                with VerticalScroll(id="metrics-scroll"):
                    yield Static("", id="metrics-summary", classes="hero")
                    yield Static("no metrics (start a running profile)",
                                 id="metrics-empty", classes="hidden")
                    with Horizontal(id="metric-cols"):
                        with Vertical(id="col-decode"):
                            yield Label("DECODE (tok/s)", classes="metric-label")
                            yield Digits("0.0", id="g-decode")
                            yield Sparkline([0.0], id="sp-decode", classes="spark")
                        with Vertical(id="col-prefill"):
                            yield Label("PREFILL (tok/s)", classes="metric-label")
                            yield Digits("0.0", id="g-prefill")
                            yield Sparkline([0.0], id="sp-prefill", classes="spark")
                    yield Label("token counters", classes="metric-label")
                    with Horizontal(id="token-cols") as row:
                        yield Static("-", id="tokens-decode", classes="panel panel-body mini")
                        yield Static("-", id="tokens-prefill", classes="panel panel-body mini")
                        yield Static("-", id="g-proc", classes="panel panel-body mini")
                    yield Label("system", classes="metric-label")
                    yield Label("CPU", classes="gauge-label")
                    yield ProgressBar(0, id="bar-cpu", show_bar=True)
                    yield Label("RAM", classes="gauge-label")
                    yield ProgressBar(0, id="bar-mem", show_bar=True)
                    yield Label("GPU VRAM (free)", classes="gauge-label")
                    yield ProgressBar(0, id="bar-gpu", show_bar=True)
                    yield Static("", id="sys-summary", classes="sys-summary")
            with TabPane("Slots", id="slots", classes="tabpane"):
                with VerticalScroll(id="slots-scroll"):
                    yield Static("select a running profile to inspect server slots",
                                 id="slots-empty")
                    yield DataTable(id="slots-table", zebra_stripes=True)
            with TabPane("Logs", id="logs", classes="tabpane"):
                with Horizontal(id="log-toolbar"):
                    yield Label("server log tail", classes="panel-title")
                    yield Switch(value=True, id="sw-follow")
                    yield Label("follow", id="follow-label", classes="toolbar-hint")
                    yield Static("", id="log-placeholder")
                    yield Button("Clear", id="btn-log-clear", variant="error")
                yield RichLog(id="log", highlight=True, wrap=True)
            with TabPane("Config", id="config", classes="tabpane"):
                with VerticalScroll(id="config-scroll"):
                    yield Static("select a profile to inspect its resolved config",
                                 id="config-empty")
                    yield Static("", id="config-out", classes="panel panel-body")
            with TabPane("Doctor", id="doctor", classes="tabpane"):
                with VerticalScroll(id="doctor-scroll"):
                    yield Static("", id="doctor-summary", classes="hero")
                    yield Button("Run checks", id="btn-doctor")
                    yield Static("", id="doctor-out")
        yield Footer()

    # ------------------------------------------------------------------ #
    # refresh pipeline - every widget update is guarded so a bad probe can
    # never take down the app (a stale case surfaces as a status note).
    # ------------------------------------------------------------------ #
    def poll_tick(self) -> None:
        """Interval callback: refresh whole panel every REFRESH_SECONDS."""
        try:
            self.refresh_all()
            self._update_sys()
            self._update_slots()
        except Exception as exc:
            self._err(f"poll error: {exc}")

    def refresh_all(self) -> None:
        try:
            self.states = poll_states(self.profiles)
            self._rebuild_picker()
            self._update_detail()
            self._update_metrics()
            self._auto_select()
            self._update_overview()
        except Exception as exc:
            import traceback
            self._err(f"refresh error: {exc}")
            traceback.print_exc(file=sys.stderr)

    def _target(self):
        if self.selected and self.selected in self.profiles:
            return self.profiles[self.selected]
        return None

    def _err(self, msg: str) -> None:
        self._notify(msg)

    # ------------------------------------------------------------------ #
    # Dashboard - stat strip, profile picker, detail + context bars.
    # ------------------------------------------------------------------ #
    def _card(self, card_id: str, title: str, value: str, sub: str = "") -> None:
        """Idempotently build one dashboard overview card."""
        card = self.query_one(card_id, Vertical)
        prefix = card_id.lstrip("#")
        if not card.children:
            card.mount(Static(title, id=f"{prefix}-title", classes="sys-card-title"))
            card.mount(Static(value, id=f"{prefix}-value", classes="sys-card-value"))
            card.mount(Static(sub, id=f"{prefix}-sub", classes="sys-card-sub"))
        else:
            self.query_one(f"#{prefix}-value", Static).update(value)
            if sub:
                self.query_one(f"#{prefix}-sub", Static).update(sub)

    def _update_overview(self) -> None:
        s = self._sys
        # GPU card
        free_gb = s.gpu_mib_free / 1024
        used_gb = s.gpu_mib_used / 1024
        total_gb = s.gpu_mib_total / 1024
        if s.gpu_mib_total:
            self._card("#card-gpu", "GPU",
                       f"{s.gpu_util:.0f}%",
                       f"{used_gb:.1f}/{total_gb:.1f} GiB · {free_gb:.1f} free · {s.gpu_temp:.0f}°C")
        else:
            self._card("#card-gpu", "GPU", "n/a", "no GPU detected")
        # CPU card
        self._card("#card-cpu", "CPU",
                   f"{s.cpu:.0f}%",
                   f"{s.cpu_temp:.0f}°C · {s.cpu_freq_mhz/1000:.1f} GHz")
        # RAM card
        self._card("#card-mem", "RAM",
                   f"{s.mem_pct:.0f}%",
                   f"{s.mem_used_gb:.1f}/{s.mem_total_gb:.1f} GiB · {s.mem_free_gb:.1f} free")
        # Drive card
        self._card("#card-drive", "DISK",
                   f"{s.nvme_temp:.0f}°C" if s.nvme_temp else "—",
                   f"load {s.load:.1f}")

    def _update_sysbar(self) -> None:
        s = self._sys
        free_gb = s.gpu_mib_free / 1024
        self.query_one("#sys-gpu", Static).update(
            Text("GPU ", style="bold #79c0ff")
            + Text(f"{s.gpu_util:.0f}% {s.gpu_temp:.0f}°C {free_gb:.1f}GiB"
                   if s.gpu_mib_total else "n/a"))
        self.query_one("#sys-cpu", Static).update(
            Text("CPU ", style="bold #3fb950")
            + Text(f"{s.cpu:.0f}% {s.cpu_temp:.0f}°C"))
        self.query_one("#sys-ram", Static).update(
            Text("RAM ", style="bold #d29922")
            + Text(f"{s.mem_pct:.0f}% {s.mem_free_gb:.0f}GiB free"))
        self.query_one("#sys-disk", Static).update(
            Text("NVMe ", style="bold #8b949e")
            + Text(f"{s.nvme_temp:.0f}°C" if s.nvme_temp else "—"))

    def _rebuild_picker(self) -> None:
        lv = self.query_one("#profiles", ListView)
        lv.clear()
        names = [n for n in self.profiles if self._filter in n.lower()]
        for name in names:
            st = self.states.get(name)
            lv.append(self._make_item(name, st))
        if self.selected and self.selected in names:
            lv.index = names.index(self.selected)

    def _make_item(self, name: str, st: ProfileState | None) -> ListItem:
        p = self.profiles.get(name)
        if p is None:
            st = ProfileState(Profile(name=name), False, "down", 0.0, 0.0, 0, "")
        elif st is None:
            st = ProfileState(p, False, "down", 0.0, 0.0, 0, "")
        model = (p.model if p is not None else "").split("/")[-1] or "?"
        line = Text().append(f"{name}", style="bold")
        line.append("   ").append(st.status_chip)
        if st.health == "running":
            line.append(f"   {st.decode_tps:.1f} t/s", style="#79c0ff")
        line.append("   ").append(model, style="#a371f7")
        item = ListItem(Label(line, classes="pl-label"))
        item.profile_name = name  # type: ignore[attr-defined]
        return item

    def _auto_select(self) -> None:
        if not self.profiles:
            self.selected = None
            return
        if self.selected and self.selected in self.states:
            return
        running = [n for n, s in self.states.items() if s.running]
        self.selected = (running or list(self.profiles))[0]

    # ------------------------------------------------------------------ #
    # Detail panel (Dashboard) + context/slot usage (Dashboard).
    # ------------------------------------------------------------------ #
    def _update_detail(self) -> None:
        detail = self.query_one("#detail", Static)
        if not self.selected or self.selected not in self.profiles:
            detail.update("pick a profile")
            return
        st = self.states.get(self.selected)
        p = self.profiles[self.selected]
        if st is None:
            detail.update(str(p))
            return
        try:
            suite = (p.resolve("EXPERTS") or "-").split("/")[-1].replace(".txt", "")
        except Exception:
            suite = "-"
        t = Text()
        state_color = {"ok": "#3fb950", "loading": "#d29922", "down": "#f85149"}.get(
            st.health, "#8b949e")
        t.append(f"{'state':13}", style="#8b949e").append(
            f"{st.health}\n", style=f"bold {state_color}")
        t.append(f"{'model':13}", style="#8b949e").append(
            f"{p.model.split('/')[-1] or '?'}\n", style="#a371f7")
        t.append(f"{'driver':13}", style="#8b949e").append(f"{st.driver or '-'}\n")
        t.append(f"{'latency':13}", style="#8b949e").append(
            f"{st.latency_ms:.1f} ms\n" if st.running else "-\n")
        t.append(f"{'port':13}", style="#8b949e").append(f"{p.port}\n")
        t.append(f"{'url':13}", style="#8b949e").append(f"{p.base_url}\n")
        t.append(f"{'decode':13}", style="#8b949e").append(
            f"{st.decode_tps:.2f} tok/s\n" if st.running else "-\n", style="#79c0ff")
        t.append(f"{'prefill':13}", style="#8b949e").append(
            f"{st.prefill_tps:.2f} tok/s\n" if st.running else "-\n", style="#79c0ff")
        t.append(f"{'requests':13}", style="#8b949e").append(
            f"{st.n_processing}\n" if st.running else "-\n")
        t.append(f"{'suite':13}", style="#8b949e").append(f"{suite}\n", style="#d29922")
        detail.update(t)
        self._update_ctx_detail(p, st)

    def _update_ctx_detail(self, p: Profile, st: ProfileState | None) -> None:
        box = self.query_one("#ctx-detail", Static)
        if st is None or not st.running:
            box.update(Text("no running server", style="dim"))
            return
        slots = self._slots.get(p.name)
        txt = Text()
        if slots:
            busy = getattr(slots, "busy", 0)
            npar = getattr(slots, "n_parallel", 1)
            busy_color = "#3fb950" if busy == 0 else "#d29922"
            txt.append(f"slots {busy}/{npar} busy", style=f"bold {busy_color}")
            txt.append("\n")
            total_ctx = getattr(slots, "n_parallel", 1) * (getattr(slots, "slots", [])[0].n_ctx if getattr(slots, "slots", []) else 0)
            txt.append(f"context used {getattr(slots, 'total_past', 0)} tok", style="#79c0ff")
        else:
            txt.append(Text("slots unknown (server busy?)", style="dim"))
        box.update(txt)

    def _update_metrics_summary(self, st: ProfileState) -> None:
        self.query_one("#metrics-summary", Static).update(
            Text(f"{self.selected}  ", style="bold")
            + Text(f"{st.decode_tps:.1f} t/s decode  ", style="#3fb950")
            + Text(f"{st.prefill_tps:.1f} t/s prefill  ", style="#79c0ff")
            + Text(f"{st.n_processing} requests")
        )

    def _show_metric_widgets(self) -> None:
        self.query_one("#g-decode", Digits).remove_class("hidden")
        self.query_one("#g-prefill", Digits).remove_class("hidden")
        self.query_one("#g-proc", Static).remove_class("hidden")
        self.query_one("#metrics-summary", Static).remove_class("hidden")
        self.query_one("#tokens-decode", Static).remove_class("hidden")
        self.query_one("#tokens-prefill", Static).remove_class("hidden")

    def _update_metrics(self) -> None:
        empty = self.query_one("#metrics-empty", Static)
        st = self.states.get(self.selected) if self.selected else None
        if st is None or not st.running:
            empty.remove_class("hidden")
            self.query_one("#g-decode", Digits).add_class("hidden")
            self.query_one("#g-prefill", Digits).add_class("hidden")
            self.query_one("#sp-decode", Sparkline).add_class("hidden")
            self.query_one("#sp-prefill", Sparkline).add_class("hidden")
            self.query_one("#g-proc", Static).add_class("hidden")
            self.query_one("#metrics-summary", Static).add_class("hidden")
            self.query_one("#tokens-decode", Static).add_class("hidden")
            self.query_one("#tokens-prefill", Static).add_class("hidden")
            return

        empty.add_class("hidden")
        self._show_metric_widgets()
        self._update_metrics_summary(st)

        ser = self.selected
        prev = self.last_metrics.get(ser, ([], 0.0))[0] if ser in self.last_metrics else []
        prev = (prev + [st.decode_tps])[-90:]
        self.last_metrics[ser] = (prev, st.decode_tps)
        self._history_decode.push(st.decode_tps)
        self._history_prefill.push(st.prefill_tps)

        self.query_one("#g-decode", Digits).update(f"{st.decode_tps:.1f}")
        sp = self.query_one("#sp-decode", Sparkline)
        sp.remove_class("hidden")
        sp.data = prev
        sp.summary_function = lambda data, th=st.decode_tps: th

        self.query_one("#g-prefill", Digits).update(f"{st.prefill_tps:.1f}")
        sp2 = self.query_one("#sp-prefill", Sparkline)
        sp2.remove_class("hidden")
        sp2.data = prev
        sp2.summary_function = lambda data, th=st.prefill_tps: th

        self.query_one("#g-proc", Static).update(f"{st.n_processing} reqs")
        self.query_one("#tokens-decode", Static).update(
            Text("decode ", style="#8b949e") + Text(f"{st.decode_tps*st.n_processing:.0f} t/s", style="bold"))
        self.query_one("#tokens-prefill", Static).update(
            Text("prefill ", style="#8b949e") + Text(f"{st.prefill_tps:.0f} t/s", style="bold"))
        self.query_one("#sp-decode", Sparkline).data = prev

    # ------------------------------------------------------------------ #
    # Slots tab - DataTable of the running server's /slots.
    # ------------------------------------------------------------------ #
    def _update_slots(self) -> None:
        table = self.query_one("#slots-table", DataTable)
        empty = self.query_one("#slots-empty", Static)
        p = self._target()
        if p is None:
            table.clear()
            empty.remove_class("hidden")
            return
        st = self.states.get(p.name)
        if st is None or not st.running:
            table.clear()
            empty.remove_class("hidden")
            empty.update(f"{p.name} is not running")
            return

        data = fetch_slots(p.base_url)
        self._slots[p.name] = data
        if not data.slots:
            table.clear()
            empty.remove_class("hidden")
            empty.update(f"{p.name}: /slots returned no rows yet")
            return
        empty.add_class("hidden")
        if not table.columns:
            table.add_column("idx", key="idx")
            table.add_column("state", key="state")
            table.add_column("model", key="model")
            table.add_column("busy", key="busy")
            table.add_column("n_past", key="n_past")
            table.add_column("n_ctx", key="n_ctx")
            table.add_column("usage %", key="pct")
            table.add_column("tokens", key="tokens")
        table.clear()
        for s in data.slots:
            table.add_row(
                *[str(s.idx),
                  Text(s.state, style="bold" if s.busy else "#8b949e"),
                  s.model or "-",
                  "yes" if s.busy else "no",
                  str(s.n_past),
                  str(s.n_ctx),
                  f"{s.pct:.0f}%",
                  str(s.tokens),
                ]
            )

    # ------------------------------------------------------------------ #
    # Logs tab - tail + follow toggle + clear.
    # ------------------------------------------------------------------ #
    def update_log(self) -> None:
        log = self.query_one("#log", RichLog)
        if not self.selected or self.selected not in self.profiles:
            return
        p = self.profiles[self.selected]
        if not p.log_file.exists():
            self._write_log_placeholder(log, f"(no log yet for {self.selected})")
            return
        try:
            lines = p.log_file.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            self._write_log_placeholder(log, f"(could not read log: {exc})")
            return
        if not self._follow_logs:
            return
        log.clear()
        for line in lines[-140:]:
            log.write(line)
        log.scroll_end(animate=False)

    def _write_log_placeholder(self, log: RichLog, text: str) -> None:
        log.clear()
        log.write(text)
        log.scroll_end(animate=False)

    def on_switch_changed(self, event) -> None:
        if event.switch.id == "sw-follow":
            self._follow_logs = bool(event.value)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn = event.button.id
        if btn == "btn-start":
            self.action_start()
        elif btn == "btn-stop":
            self.action_down()
        elif btn == "btn-restart":
            self.action_restart()
        elif btn == "btn-web":
            self.action_open_web()
        elif btn == "btn-refresh":
            self.action_refresh()
        elif btn == "btn-doctor":
            self.action_doctor()
        elif btn == "btn-log-clear":
            self.query_one("#log", RichLog).clear()

    # ------------------------------------------------------------------ #
    # Config tab - render the resolved env with human labels.
    # ------------------------------------------------------------------ #
    _CONFIG_KEYS = None

    def _config_key_order(self) -> list[str]:
        if self._CONFIG_KEYS is None:
            self._CONFIG_KEYS = self._ordered_config_keys()
        return self._CONFIG_KEYS

    def _ordered_config_keys(self) -> list[str]:
        prof = self._target()
        if prof is None:
            return []
        known = list(KNOB_LABELS.keys())
        keys = getattr(prof, "env_order", None) or list(KNOB_LABELS.keys())
        return [k for k in keys if k in KNOB_LABELS]

    def _update_config(self) -> None:
        box = self.query_one("#config-out", Static)
        empty = self.query_one("#config-empty", Static)
        p = self._target()
        if p is None:
            box.update("")
            empty.remove_class("hidden")
            return
        empty.add_class("hidden")
        t = Text()
        t.append(Text(f"resolved config for {p.name}\n", style="bold"))
        t.append(Text(f"launcher: {getattr(p, 'script', '-')}\n", style="dim"))
        t.append(Text("_" * 40, style="dim") + Text("\n"))
        for key in self._config_key_order():
            label, hint = KNOB_LABELS.get(key, (key, ""))
            value = None
            try:
                value = p.resolve(key)
            except Exception:
                value = None
            if value in ("", None):
                continue
            t.append(Text(f"{label:22}", style="#8b949e"))
            t.append(Text(str(value), style="bold"))
            if hint:
                t.append(Text(f"  · {hint}", style="dim"))
            t.append(Text("\n"))
        box.update(t)

    # ------------------------------------------------------------------ #
    # System gauges (Metrics) + host summary, driven by psutil/nvidia-smi.
    # ------------------------------------------------------------------ #
    def _update_sys(self) -> None:
        try:
            stats = fetch_sys_stats()
        except Exception:
            stats = self._sys
        self._sys = stats
        self._update_sysbar()
        self._update_overview()
        try:
            self.query_one("#bar-cpu", ProgressBar).update(progress=round(stats.cpu, 1), total=100)
            self.query_one("#bar-mem", ProgressBar).update(progress=round(stats.mem_pct, 1), total=100)
            self.query_one("#bar-gpu", ProgressBar).update(progress=round(stats.gpu_pct_free, 1), total=100)
            self.query_one("#sys-summary", Static).update(
                Text(f"cpu {stats.cpu:.0f}%   ", style="bold")
                + Text(f"ram {stats.mem_used_gb:.1f}/{stats.mem_total_gb:.1f} GiB   ")
                + Text(f"gpu free {stats.gpu_mib_free/1024:.1f} GiB", style="#79c0ff"))
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Status line + per-tab concrete updates on activation.
    # ------------------------------------------------------------------ #
    def _notify(self, msg: str, kind: str = "info") -> None:
        db = self.query("#dash-status")
        if db:
            db[0].update(msg)
        else:
            self.notify(msg, severity="warning" if kind == "err" else "information")

    def _update_tab_for(self, pane_id: str) -> None:
        if pane_id == "logs":
            self.update_log()
        elif pane_id == "config":
            self._update_config()
        elif pane_id == "slots":
            self._update_slots()
        elif pane_id == "doctor":
            if not self._doctor_run:
                self.action_doctor()
                self._doctor_run = True

    def on_tabbed_content_tab_activated(self, event) -> None:
        pane = event.pane
        pid = getattr(pane, "id", "") if pane else ""
        self._update_tab_for(pid)
        if "metrics" in pid:
            self._update_metrics()

    # ------------------------------------------------------------------ #
    # Actions.
    # ------------------------------------------------------------------ #
    def action_focus_search(self) -> None:
        f = self.query_one("#filter", Input)
        f.remove_class("hidden")
        f.focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        self._filter = event.value.strip().lower()
        self._rebuild_picker()

    def on_list_view_selected(self, event) -> None:
        if event.item:
            self.selected = getattr(event.item, "profile_name", None)
            self.action_refresh()
            self._update_detail()
            self._update_config()

    def action_start(self) -> None:
        p = self._target()
        if not p:
            self._notify("select a profile first", "warn")
            return
        ok, msg = runner.up(p)
        self._notify(msg, "ok" if ok else "err")
        self.refresh_all()

    def action_down(self) -> None:
        p = self._target()
        if not p:
            self._notify("select a profile first", "warn")
            return
        ok, msg = runner.down(p, force=True)
        self._notify(msg, "ok" if ok else "err")
        self.refresh_all()

    def action_restart(self) -> None:
        p = self._target()
        if not p:
            self._notify("select a profile first", "warn")
            return
        self._notify(f"restarting {p.name}…", "info")
        ok, msg = runner.restart(p)
        self._notify(msg, "ok" if ok else "err")
        self.refresh_all()

    def action_refresh(self) -> None:
        self.refresh_all()

    def action_open_web(self) -> None:
        p = self._target()
        if not p:
            self._notify("no profile selected", "warn")
            return
        st = self.states.get(p.name)
        if st and not st.running:
            self._notify(f"{p.name} is not running; start it first", "warn")
            return
        try:
            ok_open = webbrowser.open(p.base_url)
        except Exception as exc:
            self._notify(f"could not open browser: {exc}", "err")
            return
        self._notify("opened " + p.base_url if ok_open else "open request sent")

    def action_next_tab(self) -> None:
        tc = self.query_one(TabbedContent)
        ids = [p.id for p in tc._tab_content if p.id]
        if not ids:
            return
        try:
            cur = ids.index(tc.active)
        except Exception:
            cur = -1
        tc.active = ids[(cur + 1) % len(ids)]
        self._update_tab_for(tc.active)

    def action_prev_tab(self) -> None:
        tc = self.query_one(TabbedContent)
        ids = [p.id for p in tc._tab_content if p.id]
        if not ids:
            return
        try:
            cur = ids.index(tc.active)
        except Exception:
            cur = 1
        tc.active = ids[(cur - 1) % len(ids)]
        self._update_tab_for(tc.active)

    def action_toggle_theme(self) -> None:
        target = "siphon-light" if self.theme == "siphon-dark" else "siphon-dark"
        try:
            self.theme = target
            self.notify(f"theme: {self.theme}", severity="information")
        except Exception as exc:
            self.notify(f"theme error: {exc}", severity="error")

    def _set_tab(self, name: str) -> None:
        try:
            tc = self.query_one(TabbedContent)
            tc.active = name
            self._update_tab_for(name)
        except Exception:
            pass

    def action_tab_dashboard(self) -> None:
        self._set_tab("dashboard")

    def action_tab_metrics(self) -> None:
        self._set_tab("metrics")

    def action_tab_slots(self) -> None:
        self._set_tab("slots")

    def action_tab_logs(self) -> None:
        self._set_tab("logs")

    def action_tab_config(self) -> None:
        self._set_tab("config")

    def action_tab_doctor(self) -> None:
        self._set_tab("doctor")

    def action_help(self) -> None:
        self.push_screen(HelpScreen())

    def action_doctor(self) -> None:
        out = self.query_one("#doctor-out", Static)
        checks = doctor.run_checks()
        t = Text()
        ok = sum(1 for c in checks if c.ok)
        self.query_one("#doctor-summary", Static).update(
            Text(f"{ok}/{len(checks)} checks passed", style="bold") + Text("   run anytime", style="dim"))
        for c in checks:
            mark = "✓" if c.ok else "✗"
            color = "#3fb950" if c.ok else "#f85149"
            t.append(Text(f"{mark} ", style=color + " bold"))
            t.append(Text(f"{c.name:22}", style="bold"))
            t.append(Text(f"{c.detail}\n", style="dim"))
        out.update(t)

    def try_quit(self) -> None:
        self.exit()


def run() -> None:
    SiphonTUI().run()
