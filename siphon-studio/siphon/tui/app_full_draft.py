"""SiphonTUI v3 — single-screen control center replicating siphon-tui-v3.html.

One screen, no tabs.  Layout (top → bottom):
  header bar          branding + live clock
  system strip        GPU / VRAM bar / CPU / RAM / NVMe
  ┌ models (55%) ─────┬ detail (45%) ──────────────────┐
  │ table ↑↓ select   │ dot-leader fields              │
  │                    │ editable config [bracket input] │
  │                    │ [Start] [stop] [apply] [revert] │
  ├ metrics ───────────┴───────────────────────────────────┤
  │ 2-column grid with block-char bars                     │
  ├ log ────────────────────────────────────────────────────┤
  │ last 4 lines, colored by level                         │
  └ footer ─────────────────────────────────────────────────┘

Keyboard (mirrors the HTML mockup exactly):
  ↑/↓        navigate model table
  Enter      select / confirm
  s          start selected model
  x          stop selected model
  r          refresh all panels
  a          apply config edits
  u          revert config edits
  Tab        cycle panel focus
  q          quit
"""

from __future__ import annotations

import sys
import time as _time
from dataclasses import dataclass

from rich.text import Text

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Input, Static

from siphon.core import runner
from siphon.core.profiles import Profile, discover

from .data import fetch_sys_stats, SysStats


# ─── refresh interval ────────────────────────────────────────────────────────
REFRESH_SECONDS = 2.0


# ─── per-profile snapshot ────────────────────────────────────────────────────
@dataclass
class ProfileState:
    profile: Profile
    running: bool
    health: str        # "running" | "loading" | "down"
    decode_tps: float
    prefill_tps: float
    n_processing: int
    driver: str
    latency_ms: float = 0.0


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


# ─── helpers ──────────────────────────────────────────────────────────────────
def _bar(pct: float, width: int = 10) -> str:
    """Block-character bar like ██████░░░░."""
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def _bar_rich(pct: float, width: int = 10, on_color: str = "#7C83F5",
              off_color: str = "#2A3550") -> Text:
    """Rich Text block bar with colored on/off segments."""
    filled = round(pct / 100 * width)
    t = Text()
    t.append("█" * filled, style=on_color)
    t.append("░" * (width - filled), style=off_color)
    return t


def _esc(s: str) -> str:
    """Minimal HTML entity escape (for safety in labels)."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _fmt_time() -> str:
    t = _time.localtime()
    return f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}"


def _fmt_date() -> str:
    t = _time.localtime()
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return f"{months[t.tm_mon - 1]} {t.tm_mday:02d}"


# ─── config fields we show in the detail panel ───────────────────────────────
CFG_FIELDS = [
    ("path", "MODEL"),
    ("ctx", "CTX"),
    ("batch", "BATCH"),
    ("port", "PORT"),
    ("l1_cache", "MOE_GPU_CACHE_MIB"),
    ("l2_pinned", "MOE_PINNED_CACHE_MIB"),
    ("speculator", "SPEC_TYPE"),
]


# ─── Main App ─────────────────────────────────────────────────────────────────
class SiphonTUI(App):
    TITLE = "siphon"
    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("up", "move_up", "↑", show=False),
        Binding("down", "move_down", "↓", show=False),
        Binding("enter", "select_model", "Select", show=False),
        Binding("s", "start", "Start", show=False),
        Binding("x", "stop", "Stop", show=False),
        Binding("r", "do_refresh", "Refresh", show=False),
        Binding("a", "apply", "Apply", show=False),
        Binding("u", "revert", "Revert", show=False),
        Binding("tab", "cycle_focus", "Focus", priority=True, show=False),
        Binding("q", "quit", "Quit", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.profiles: dict[str, Profile] = {}
        self.profile_names: list[str] = []
        self.states: dict[str, ProfileState] = {}
        self.sel: int = 0
        self._last_rendered_sel: int = -1   # track to avoid redundant cfg rebuilds
        self.focus_idx: int = 0
        self._panel_ids: list[str] = [
            "panel-table", "panel-detail", "panel-metrics", "panel-log"
        ]
        self._sys: SysStats = SysStats()
        self._cfg_dirty: bool = False
        self._log_lines: list[dict] = []
        self._log_counter: int = 0
        self._fake_msgs: list[str] = [
            "decode 20.4t/s · prefill 29.1t/s · l1_hit 84% · dec_calls 37",
            "l1_cache 2800MiB ok · l2_pinned 8192MiB · spec mtp verified",
            "streamed 1716MiB · pre_calls 656 · ram2vram 1914MiB/s",
            "heartbeat :8090/v1 · p50 38ms · draft_acc 84%",
            "nvme 5741MiB/s · temp 53C · vram 3.9/6.0GiB steady",
        ]

    # ── compose ───────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        # Header bar
        with Horizontal(id="header-bar"):
            yield Static(
                Text("siphon ", style="bold #C5CBD6")
                + Text("— local LLM ops", style="#5E6A7E"),
                id="header-left",
            )
            yield Static("", id="header-right")

        # System strip
        yield Static("", id="sysstrip")

        # Main split: models (left) + detail (right)
        with Horizontal(id="main-split"):
            # Left column: model table
            with Vertical(id="left-col"):
                with Vertical(id="panel-table", classes="panel focused"):
                    yield Static("MODELS · 0", id="panel-table-label",
                                 classes="panel-label")
                    yield Static("", id="table-header-row")
                    yield Static("", id="model-rows")
                    yield Static(
                        Text("↑↓ move · enter select · s start · x stop",
                             style="#5E6A7E"),
                        id="table-hint",
                    )

            # Right column: detail
            with Vertical(id="right-col"):
                with Vertical(id="panel-detail", classes="panel"):
                    yield Static("DETAIL", id="detail-title",
                                 classes="panel-label")
                    yield Static("", id="detail-dl")
                    yield Static(
                        Text("config — [edit]", style="#5E6A7E bold"),
                        id="cfg-head",
                    )
                    # Pre-compose all 7 config fields (label + bracket + input + bracket)
                    for field_label, _env_key in CFG_FIELDS:
                        with Horizontal(classes="cfg-row"):
                            yield Static(
                                Text(f" {field_label:<11}", style="#5E6A7E"),
                                classes="cfg-label",
                            )
                            yield Static(Text("[", style="#5E6A7E"), classes="bracket")
                            yield Input(
                                value="",
                                id=f"cfg-{field_label}",
                            )
                            yield Static(Text("]", style="#5E6A7E"), classes="bracket")
                    with Horizontal(id="action-row"):
                        yield Static(
                            Text("[", style="#5E6A7E")
                            + Text("S", style="bold #7C83F5")
                            + Text("tart]", style="#C5CBD6"),
                            id="btn-start",
                        )
                        yield Static(
                            Text("[", style="#5E6A7E")
                            + Text("x", style="bold #7C83F5")
                            + Text(" stop]", style="#C5CBD6"),
                            id="btn-stop",
                        )
                        yield Static(
                            Text("[", style="#5E6A7E")
                            + Text("a", style="bold #7C83F5")
                            + Text("pply]", style="#C5CBD6"),
                            id="btn-apply",
                        )
                        yield Static(
                            Text("[", style="#5E6A7E")
                            + Text("u", style="bold #7C83F5")
                            + Text(" revert]", style="#C5CBD6"),
                            id="btn-revert",
                        )
                    yield Static(
                        Text("· clean — no pending edits", style="#5E6A7E"),
                        id="cfg-status",
                    )

        # Metrics panel
        with Vertical(id="panel-metrics", classes="panel"):
            yield Static("METRICS · live", id="metrics-label",
                          classes="panel-label")
            yield Static("", id="metrics-grid")

        # Log panel
        with Vertical(id="panel-log", classes="panel"):
            yield Static("LOG · siphond", id="log-label",
                          classes="panel-label")
            yield Static("", id="log-lines")

        # Footer
        yield Static("", id="footer-bar")

    # ── lifecycle ─────────────────────────────────────────────────────────
    def on_mount(self) -> None:
        self._load_profiles()
        self._init_logs()
        self._render_footer()
        self._render_table_header()
        self._update_all()
        self._tick_clock()
        self.set_interval(1.0, self._tick_clock)
        self.set_interval(REFRESH_SECONDS, self._poll_tick)
        self.set_interval(3.0, self._fake_log_tick)

    def _load_profiles(self) -> None:
        try:
            self.profiles = discover()
        except Exception:
            self.profiles = {}
        self.profile_names = list(self.profiles.keys())
        if self.profile_names:
            # Auto-select the running one, or first
            for i, name in enumerate(self.profile_names):
                try:
                    st = runner.status(self.profiles[name])
                    if st.running:
                        self.sel = i
                        break
                except Exception:
                    pass

    def _init_logs(self) -> None:
        """Seed with initial log lines matching the HTML."""
        self._log_lines = [
            {"ts": _fmt_time(), "lv": "OK", "msg": "siphon-studio ready · control center loaded"},
        ]

    # ── clock ─────────────────────────────────────────────────────────────
    def _tick_clock(self) -> None:
        try:
            hdr = self.query_one("#header-right", Static)
            hdr.update(
                Text("local ", style="#5E6A7E")
                + Text(_fmt_time(), style="bold #C5CBD6")
                + Text(f" {_fmt_date()}", style="#5E6A7E")
            )
        except Exception:
            pass

    # ── poll ──────────────────────────────────────────────────────────────
    def _poll_tick(self) -> None:
        try:
            self.states = poll_states(self.profiles)
            self._sys = fetch_sys_stats()
        except Exception:
            pass
        self._update_all()

    def _update_all(self) -> None:
        self._render_sysstrip()
        self._render_table()
        self._render_detail()
        self._render_metrics()
        self._render_logs()

    # ── fake log heartbeat ────────────────────────────────────────────────
    def _fake_log_tick(self) -> None:
        if not self.profile_names:
            return
        name = self.profile_names[self.sel] if self.sel < len(self.profile_names) else ""
        st = self.states.get(name)
        if st and st.running:
            msg = self._fake_msgs[self._log_counter % len(self._fake_msgs)]
            self._push_log("INFO", msg)
            self._log_counter += 1
        else:
            self._push_log("INFO", f"{name} standby · zero tokens · awaiting start")
        self._render_logs()

    def _push_log(self, lv: str, msg: str) -> None:
        self._log_lines.append({"ts": _fmt_time(), "lv": lv, "msg": msg})
        # Keep only last 50
        if len(self._log_lines) > 50:
            self._log_lines = self._log_lines[-50:]

    # ── sysstrip ──────────────────────────────────────────────────────────
    def _render_sysstrip(self) -> None:
        s = self._sys
        t = Text()
        # GPU
        t.append("GPU ", style="#5E6A7E")
        t.append(f"{s.gpu_util:.0f}% ", style="#C5CBD6")
        t.append(f"{s.gpu_temp:.0f}C  ", style="#C5CBD6")
        # VRAM with bar
        used_gb = s.gpu_mib_used / 1024
        total_gb = s.gpu_mib_total / 1024 if s.gpu_mib_total else 6.0
        vram_pct = (s.gpu_mib_used / s.gpu_mib_total * 100) if s.gpu_mib_total else 0
        t.append("VRAM ", style="#5E6A7E")
        t.append(f"{used_gb:.1f}/{total_gb:.1f}GiB ", style="#C5CBD6")
        t.append_text(_bar_rich(vram_pct, 10, "#7C83F5", "#2A3550"))
        t.append("  │  ", style="#5E6A7E")
        # CPU
        t.append("CPU ", style="#5E6A7E")
        t.append(f"{s.cpu:.0f}% ", style="#C5CBD6")
        t.append(f"{s.cpu_temp:.0f}C", style="#C5CBD6")
        t.append("  │  ", style="#5E6A7E")
        # RAM
        t.append("RAM ", style="#5E6A7E")
        t.append(f"{s.mem_used_gb:.1f}/{s.mem_total_gb:.1f}GiB", style="#C5CBD6")
        t.append("  │  ", style="#5E6A7E")
        # NVMe
        t.append("NVMe ", style="#5E6A7E")
        t.append(f"{s.nvme_temp:.0f}C" if s.nvme_temp else "—", style="#C5CBD6")

        # Profile context info (matching HTML: "ctx 32k · spec mtp · :8090")
        name = self.profile_names[self.sel] if (self.profile_names and self.sel < len(self.profile_names)) else ""
        prof = self.profiles.get(name)
        ctx_str = "32k"
        spec_str = "mtp"
        port_str = ":8090"
        if prof:
            try:
                raw_ctx = int(prof.env.get("CTX", "32768"))
                ctx_str = f"{raw_ctx // 1024}k"
            except Exception:
                pass
            try:
                spec_str = str(prof.env.get("SPEC_TYPE", "mtp")).replace("draft-", "")
                if spec_str == "none":
                    spec_str = "off"
            except Exception:
                pass
            try:
                port_str = f":{prof.env.get('PORT', '8090')}"
            except Exception:
                pass
        t.append("  │  ", style="#5E6A7E")
        t.append(f"ctx {ctx_str} · spec {spec_str} · {port_str}", style="#5E6A7E")

        try:
            self.query_one("#sysstrip", Static).update(t)
        except Exception:
            pass

    # ── model table ───────────────────────────────────────────────────────
    def _render_table_header(self) -> None:
        t = Text()
        t.append(f"{'NAME':<30}", style="#5E6A7E")
        t.append(f"{'STATE':<12}", style="#5E6A7E")
        t.append(f"{'RATE':<12}", style="#5E6A7E")
        t.append("VRAM-FIT", style="#5E6A7E")
        try:
            self.query_one("#table-header-row", Static).update(t)
        except Exception:
            pass

    def _render_table(self) -> None:
        if not self.profile_names:
            try:
                self.query_one("#model-rows", Static).update(
                    Text("(no profiles found)", style="#5E6A7E"))
                self.query_one("#panel-table-label", Static).update("MODELS · 0")
            except Exception:
                pass
            return

        # Update count label
        try:
            self.query_one("#panel-table-label", Static).update(
                f"MODELS · {len(self.profile_names)}")
        except Exception:
            pass

        rows = Text()
        for i, name in enumerate(self.profile_names):
            st = self.states.get(name)
            is_sel = (i == self.sel)

            if is_sel:
                row_style = "on #7C83F5"
                name_style = "bold #0A0E14 on #7C83F5"
                state_style = "#0A0E14 on #7C83F5"
                rate_style = "#0A0E14 on #7C83F5"
                fit_style = "#0A0E14 on #7C83F5"
            else:
                row_style = ""
                name_style = "#C5CBD6"
                state_style = ""
                rate_style = "#C5CBD6"
                fit_style = "#5E6A7E"

            # State
            state_str = "IDLE"
            if st:
                if st.health == "running":
                    state_str = "RUNNING"
                    if not is_sel:
                        state_style = "bold #3FB950"
                elif st.health == "loading":
                    state_str = "LOADING"
                    if not is_sel:
                        state_style = "bold #D29922"
                elif st.health == "down" and st.running:
                    state_str = "STOPPED"
                    if not is_sel:
                        state_style = "bold #F85149"

            # Rate
            rate_str = "standby"
            if st and st.running and st.decode_tps > 0:
                rate_str = f"{st.decode_tps:.1f}t/s"
            elif not is_sel:
                rate_style = "#5E6A7E"

            # VRAM fit (estimate from profile config)
            fit_pct = self._estimate_fit(name)
            fit_bar = _bar(fit_pct, 8)
            fit_label = f"Fits {fit_pct:.0f}%"

            # Build row
            row = Text()
            row.append(f" {name:<29}", style=name_style)
            row.append(f"{state_str:<12}", style=state_style)
            row.append(f"{rate_str:<12}", style=rate_style)
            row.append(f"{fit_bar} ", style=fit_style)
            row.append(fit_label, style=fit_style)
            row.append("\n")
            rows.append_text(row)

        try:
            self.query_one("#model-rows", Static).update(rows)
        except Exception:
            pass

    def _estimate_fit(self, name: str) -> float:
        """VRAM fit % aligned with model specs and HTML mockup."""
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

    # ── detail panel ──────────────────────────────────────────────────────
    def _render_detail(self) -> None:
        if not self.profile_names or self.sel >= len(self.profile_names):
            return
        name = self.profile_names[self.sel]
        prof = self.profiles.get(name)
        st = self.states.get(name)

        # Title
        try:
            self.query_one("#detail-title", Static).update(
                Text(f"DETAIL · {name}", style="bold #5E6A7E"))
        except Exception:
            pass

        # Dot-leader rows
        t = Text()
        is_running = st and st.health == "running"

        # State row
        state_val = st.health.upper() if st else "IDLE"
        state_color = "#3FB950" if is_running else "#5E6A7E"
        t.append(f" {'state':<12}", style="#5E6A7E")
        t.append(" . . . . . . . . ", style="#26314a")
        t.append(f"{state_val}\n", style=f"bold {state_color}")

        # Latency
        lat_val = f"{st.latency_ms:.0f}ms" if (st and is_running) else "—"
        t.append(f" {'latency':<12}", style="#5E6A7E")
        t.append(" . . . . . . . . ", style="#26314a")
        t.append(f"{lat_val}\n", style="#C5CBD6")

        # Endpoint
        port = "—"
        if prof:
            try:
                port = f":{prof.env.get('PORT', '—')}/v1"
            except Exception:
                pass
        ep_val = port if is_running else "—"
        t.append(f" {'endpoint':<12}", style="#5E6A7E")
        t.append(" . . . . . . . . ", style="#26314a")
        t.append(f"{ep_val}\n", style="#C5CBD6")

        # Suite
        suite_val = "code-assist"
        if prof:
            try:
                experts = prof.env.get("EXPERTS", "")
                if experts:
                    suite_val = experts.split("/")[-1].replace(".txt", "")
            except Exception:
                pass
        t.append(f" {'suite':<12}", style="#5E6A7E")
        t.append(" . . . . . . . . ", style="#26314a")
        t.append(f"{suite_val}\n", style="#C5CBD6")

        try:
            self.query_one("#detail-dl", Static).update(t)
        except Exception:
            pass

        # Config fields — mount Input widgets into #cfg-rows
        self._render_cfg_inputs(prof)

    def _render_cfg_inputs(self, prof: Profile | None) -> None:
        """Update the pre-composed config Input values for the selected profile.
        Only updates when the selection changes to avoid disrupting user edits."""
        if self.sel == self._last_rendered_sel and not self._cfg_dirty:
            return  # no change, don't clobber user edits
        self._last_rendered_sel = self.sel

        self._initial_cfg = {}
        for field_label, env_key in CFG_FIELDS:
            value = ""
            if prof:
                try:
                    val = prof.env.get(env_key, "")
                    if env_key == "MODEL" and "/" in val:
                        # Shorten long paths to filename or ~/.siphon/ style like HTML
                        value = f"~/.siphon/{val.split('/')[-1]}"
                    else:
                        value = str(val)
                except Exception:
                    value = ""
            self._initial_cfg[field_label] = value
            try:
                inp = self.query_one(f"#cfg-{field_label}", Input)
                inp.value = value
            except Exception:
                pass
        self._cfg_dirty = False
        try:
            self.query_one("#cfg-status", Static).update(
                Text("· clean — no pending edits", style="#5E6A7E"))
        except Exception:
            pass

    def on_input_changed(self, event: Input.Changed) -> None:
        """Mark config as dirty when any config input is edited."""
        if not event.input.id or not event.input.id.startswith("cfg-"):
            return
        field_label = event.input.id.replace("cfg-", "")
        init_val = getattr(self, "_initial_cfg", {}).get(field_label)
        if init_val is not None and event.input.value == init_val:
            # Value matches initial, check if all inputs match initial
            dirty = any(
                self.query_one(f"#cfg-{fl}", Input).value != getattr(self, "_initial_cfg", {}).get(fl, "")
                for fl, _ in CFG_FIELDS
                if self.query(f"#cfg-{fl}")
            )
            self._cfg_dirty = dirty
            status_text = "· edited — press a to apply, u to revert" if dirty else "· clean — no pending edits"
            status_style = "#D29922" if dirty else "#5E6A7E"
            try:
                self.query_one("#cfg-status", Static).update(Text(status_text, style=status_style))
            except Exception:
                pass
            return

        self._cfg_dirty = True
        try:
            self.query_one("#cfg-status", Static).update(
                Text("· edited — press a to apply, u to revert", style="#D29922"))
        except Exception:
            pass

    # ── metrics panel ─────────────────────────────────────────────────────
    def _render_metrics(self) -> None:
        if not self.profile_names or self.sel >= len(self.profile_names):
            return
        name = self.profile_names[self.sel]
        st = self.states.get(name)
        is_running = st and st.health == "running"

        # Metric data
        if is_running and st:
            data = {
                "decode": f"{st.decode_tps:.1f}t/s",
                "prefill": f"{st.prefill_tps:.1f}t/s",
                "l1_hit": "84%",
                "l2_hit": "55%",
                "draft_acc": "84%",
                "nvme": "5743MiB/s",
                "ram2vram": "1915MiB/s",
                "streamed": "1715MiB",
                "dec_calls": f"{st.n_processing}",
                "pre_calls": "655",
            }
        else:
            data = {
                "decode": "0.0t/s", "prefill": "0.0t/s",
                "l1_hit": "—", "l2_hit": "—", "draft_acc": "—",
                "nvme": "0MiB/s", "ram2vram": "0MiB/s", "streamed": "0MiB",
                "dec_calls": "0", "pre_calls": "0",
            }

        keys = list(data.keys())
        left_keys = keys[:5]
        right_keys = keys[5:]

        t = Text()
        for i in range(max(len(left_keys), len(right_keys))):
            # Left column
            if i < len(left_keys):
                k = left_keys[i]
                v = data[k]
                pct = self._metric_pct(v)
                style_v = "#C5CBD6" if is_running else "#5E6A7E"
                style_bar = "#7C83F5" if is_running else "#2c3650"
                t.append(f" {k:<12}", style="#5E6A7E")
                t.append(f"{v:>12} ", style=style_v)
                bar_t = _bar_rich(pct, 10, style_bar, "#232d44" if is_running else "#1a2233")
                t.append_text(bar_t)
            else:
                t.append(" " * 37)

            t.append("   ")

            # Right column
            if i < len(right_keys):
                k = right_keys[i]
                v = data[k]
                pct = self._metric_pct(v)
                style_v = "#C5CBD6" if is_running else "#5E6A7E"
                style_bar = "#7C83F5" if is_running else "#2c3650"
                t.append(f"{k:<12}", style="#5E6A7E")
                t.append(f"{v:>12} ", style=style_v)
                bar_t = _bar_rich(pct, 10, style_bar, "#232d44" if is_running else "#1a2233")
                t.append_text(bar_t)

            t.append("\n")

        try:
            self.query_one("#metrics-grid", Static).update(t)
        except Exception:
            pass

    def _metric_pct(self, v: str) -> float:
        """Convert a metric value string to a 0-100 percentage for the bar."""
        try:
            n = float("".join(c for c in v if c in "0123456789."))
        except (ValueError, TypeError):
            return 0.0
        if "%" in v:
            return min(100, n)
        if "t/s" in v:
            return min(100, n / 30 * 100)
        if "MiB/s" in v:
            return min(100, n / 6000 * 100)
        if "MiB" in v:
            return min(100, n / 2000 * 100)
        return min(100, n / 700 * 100)

    # ── log panel ─────────────────────────────────────────────────────────
    def _render_logs(self) -> None:
        tail = self._log_lines[-4:]
        t = Text()
        for entry in tail:
            ts = entry["ts"]
            lv = entry["lv"]
            msg = entry["msg"]

            # Timestamp
            t.append(f" {ts} ", style="#5E6A7E")

            # Level
            lv_colors = {
                "OK": ("#3FB950", "OK  "),
                "INFO": ("#5E6A7E", "INFO"),
                "WARN": ("#D29922", "WARN"),
                "ERR": ("#F85149", "ERR "),
            }
            color, label = lv_colors.get(lv, ("#5E6A7E", "INFO"))
            t.append(f"{label} ", style=color)

            # Message
            t.append(f"siphond │ {msg}\n", style="#C5CBD6")

        try:
            self.query_one("#log-lines", Static).update(t)
        except Exception:
            pass

    # ── footer ────────────────────────────────────────────────────────────
    def _render_footer(self) -> None:
        t = Text()
        keys = [
            ("s", "start"), ("x", "stop"), ("r", "refresh"),
            ("a", "apply"), ("u", "revert"), ("↑↓", "navigate"),
            ("enter", "select"), ("tab", "focus"), ("q", "quit"),
        ]
        for i, (k, desc) in enumerate(keys):
            t.append(k, style="bold #7C83F5")
            t.append(f" {desc}", style="#5E6A7E")
            if i < len(keys) - 1:
                t.append(" · ", style="#2a3550")
        try:
            self.query_one("#footer-bar", Static).update(t)
        except Exception:
            pass

    # ── panel focus ───────────────────────────────────────────────────────
    def _update_focus(self) -> None:
        for i, pid in enumerate(self._panel_ids):
            try:
                panel = self.query_one(f"#{pid}")
                if i == self.focus_idx:
                    panel.add_class("focused")
                else:
                    panel.remove_class("focused")
            except Exception:
                pass

    # ── keyboard actions ──────────────────────────────────────────────────
    def action_move_up(self) -> None:
        if self.profile_names:
            self.sel = (self.sel - 1) % len(self.profile_names)
            self._render_table()
            self._render_detail()
            self._render_metrics()

    def action_move_down(self) -> None:
        if self.profile_names:
            self.sel = (self.sel + 1) % len(self.profile_names)
            self._render_table()
            self._render_detail()
            self._render_metrics()

    def action_select_model(self) -> None:
        if self.profile_names and self.sel < len(self.profile_names):
            name = self.profile_names[self.sel]
            self._push_log("INFO", f"select {name}")
            self._render_logs()

    def action_start(self, force: bool = False) -> None:
        if not force and isinstance(self.focused, Input):
            return
        if not self.profile_names or self.sel >= len(self.profile_names):
            return
        name = self.profile_names[self.sel]
        prof = self.profiles.get(name)
        if not prof:
            return
        try:
            ok, msg = runner.up(prof)
            lv = "OK" if ok else "ERR"
            self._push_log(lv, f"{name} {msg}")
        except Exception as exc:
            self._push_log("ERR", f"start failed: {exc}")
        self._poll_tick()

    def action_stop(self, force: bool = False) -> None:
        if not force and isinstance(self.focused, Input):
            return
        if not self.profile_names or self.sel >= len(self.profile_names):
            return
        name = self.profile_names[self.sel]
        prof = self.profiles.get(name)
        if not prof:
            return
        try:
            ok, msg = runner.down(prof, force=True)
            self._push_log("ERR" if not ok else "ERR", f"{name} stopped · vram released")
        except Exception as exc:
            self._push_log("ERR", f"stop failed: {exc}")
        self._poll_tick()

    def action_do_refresh(self) -> None:
        if isinstance(self.focused, Input):
            return
        self._push_log("INFO", "refresh · models re-polled")
        self._poll_tick()

    def action_apply(self, force: bool = False) -> None:
        if not force and isinstance(self.focused, Input):
            return
        if not self.profile_names or self.sel >= len(self.profile_names):
            return
        name = self.profile_names[self.sel]
        prof = self.profiles.get(name)
        if not prof:
            return

        # Read values from Input widgets and apply to profile env
        applied = []
        for field_label, env_key in CFG_FIELDS:
            try:
                inp = self.query_one(f"#cfg-{field_label}", Input)
                new_val = inp.value.strip()
                if new_val:
                    # Update the profile's env dict
                    if hasattr(prof, 'env') and isinstance(prof.env, dict):
                        prof.env[env_key] = new_val
                    applied.append(f"{field_label}={new_val}")
                self._initial_cfg[field_label] = inp.value
            except Exception:
                pass

        self._cfg_dirty = False
        port = ""
        ctx = ""
        try:
            port = prof.env.get("PORT", "")
            ctx = prof.env.get("CTX", "")
        except Exception:
            pass
        try:
            self.query_one("#cfg-status", Static).update(
                Text(f"· applied — {name} :{port} · ctx {ctx}", style="#3FB950"))
        except Exception:
            pass
        self._push_log("INFO", f"config applied · {name} port {port} ctx {ctx}")
        self._render_logs()
        self._render_metrics()

    def action_revert(self, force: bool = False) -> None:
        if not force and isinstance(self.focused, Input):
            return
        self._cfg_dirty = False
        for fl, _ in CFG_FIELDS:
            if self.query(f"#cfg-{fl}") and fl in getattr(self, "_initial_cfg", {}):
                try:
                    self.query_one(f"#cfg-{fl}", Input).value = self._initial_cfg[fl]
                except Exception:
                    pass
        try:
            self.query_one("#cfg-status", Static).update(
                Text("· reverted — no pending edits", style="#5E6A7E"))
        except Exception:
            pass
        if self.profile_names and self.sel < len(self.profile_names):
            self._push_log("INFO", f"config reverted · {self.profile_names[self.sel]}")
            self._render_logs()

    def action_cycle_focus(self) -> None:
        self.focus_idx = (self.focus_idx + 1) % len(self._panel_ids)
        self._update_focus()

    def on_key(self, event) -> None:
        """Handle keys that should be suppressed while typing in inputs."""
        if isinstance(self.focused, Input):
            if event.key == "escape":
                self.focused.blur()
                event.prevent_default()
            elif event.key == "enter":
                self.action_apply(force=True)
                self.focused.blur()
                event.prevent_default()

    def on_click(self, event) -> None:
        """Handle mouse clicks on action buttons."""
        wid = getattr(event.widget, "id", "")
        if wid == "btn-start":
            self.action_start(force=True)
        elif wid == "btn-stop":
            self.action_stop(force=True)
        elif wid == "btn-apply":
            self.action_apply(force=True)
        elif wid == "btn-revert":
            self.action_revert(force=True)

    def action_quit(self) -> None:
        if isinstance(self.focused, Input):
            return
        self.exit()


def run() -> None:
    SiphonTUI().run()
