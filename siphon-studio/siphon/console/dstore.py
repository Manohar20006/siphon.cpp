"""DSTORAGE_MOE log-tail parsing for the console metrics grid.

Two signal families, with different freshness (see SIPHON_TUI_DASHBOARD_FEATURES.md §4):

* `slot print_timing:` lines stream LIVE per request: decode/prefill tok/s
  and `draft acceptance`. Always safe to show when present.
* `DSTORAGE_MOE_*` lines flush (mostly) on graceful exit: hit rates, IO
  shape, stream totals, call counts. The STARTUP zero-counters are
  misleading, so fields stay None until their guards pass (e.g. the PHASE
  line only counts once prefill_calls + decode_calls > 0).

Everything here is defensive and never raises.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DStore:
    # live (print_timing)
    decode_tps: float | None = None
    prefill_tps: float | None = None
    draft_accept: float | None = None      # fraction 0..1
    draft_accepted: int = 0
    draft_generated: int = 0
    # DSTORAGE (exit-flushed; None until guards pass)
    l1_hit_pct: float | None = None        # decode_hit_rate * 100
    prefill_hit_pct: float | None = None
    l2_hit_pct: float | None = None        # pinned/(pinned+cold) decode
    nvme_mibs: float | None = None         # decode effective_mib_s
    ram_vram_mibs: float | None = None     # pinned_mib / pinned_us
    streamed_mib: float | None = None      # prefill_file + decode_file
    decode_calls: int | None = None
    prefill_calls: int | None = None

    @property
    def has_dstorage(self) -> bool:
        return self.decode_calls is not None or self.prefill_calls is not None


_RE_TIMING = re.compile(
    r"eval time =\s*[\d.]+\s*ms /\s*(\d+)\s*tokens\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\s*\)"
)
_RE_PREFILL = re.compile(
    r"prompt eval time =\s*[\d.]+\s*ms /\s*(\d+)\s*tokens\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\s*\)"
)
_RE_DRAFT = re.compile(
    r"draft acceptance =\s*([\d.]+)\s*\(\s*(\d+)\s*accepted\s*/\s*(\d+)\s*generated\s*\)"
)
_RE_PHASE = re.compile(r"DSTORAGE_MOE_PHASE ([^\n]+)")
_RE_STREAM = re.compile(r"DSTORAGE_MOE_STREAM ([^\n]+)")
_RE_IO_DECODE = re.compile(r"DSTORAGE_MOE_IO_SHAPE phase=decode ([^\n]+)")
_KV = re.compile(r"([A-Za-z_][\w]*)=([-\d.]+)")


def _kv(line: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for m in _KV.finditer(line):
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            continue
    return out


def _last(pat: re.Pattern, text: str) -> re.Match | None:
    ms = list(pat.finditer(text))
    return ms[-1] if ms else None


def parse_log_tail(
    log_file: str | Path,
    max_chars: int = 400_000,
    start_offset: int = 0,
) -> DStore:
    """Parse a recent portion of a server log into a DStore. Never raises.

    ``start_offset`` lets a caller limit the parse to the current server
    session.  Server logs are append-only, so without it an idle, newly
    launched model could inherit timing numbers from an earlier request.
    """
    ds = DStore()
    try:
        raw = Path(log_file).read_bytes()
    except OSError:
        return ds
    try:
        offset = min(max(int(start_offset), 0), len(raw))
        text = raw[offset:][-max_chars:].decode("utf-8", "replace")
    except Exception:
        return ds

    m = _last(_RE_TIMING, text)
    if m:
        try:
            ds.decode_tps = float(m.group(2))
        except ValueError:
            pass
    m = _last(_RE_PREFILL, text)
    if m:
        try:
            ds.prefill_tps = float(m.group(2))
        except ValueError:
            pass
    m = _last(_RE_DRAFT, text)
    if m:
        try:
            ds.draft_accept = float(m.group(1))
            ds.draft_accepted = int(m.group(2))
            ds.draft_generated = int(m.group(3))
        except ValueError:
            pass

    # DSTORAGE guards: ignore the misleading startup zero-counters.
    ph = _last(_RE_PHASE, text)
    if ph:
        kv = _kv(ph.group(1))
        pcalls = int(kv.get("prefill_calls", 0))
        dcalls = int(kv.get("decode_calls", 0))
        if pcalls > 0:
            ds.prefill_calls = pcalls
            denom = kv.get("prefill_hits", 0) + kv.get("prefill_misses", 0)
            if denom > 0:
                ds.prefill_hit_pct = kv.get("prefill_hits", 0) / denom * 100.0
        if dcalls > 0:
            ds.decode_calls = dcalls
            if "decode_hit_rate" in kv:
                ds.l1_hit_pct = kv["decode_hit_rate"] * 100.0

    st = _last(_RE_STREAM, text)
    if st:
        kv = _kv(st.group(1))
        cold = kv.get("decode_cold_experts", 0)
        pinned = kv.get("decode_pinned_experts", 0)
        if cold + pinned > 0:
            ds.l2_hit_pct = pinned / (cold + pinned) * 100.0
        fsum = kv.get("prefill_file_mib", 0) + kv.get("decode_file_mib", 0)
        if fsum > 0:
            ds.streamed_mib = fsum
        p_mib = kv.get("decode_pinned_mib", 0)
        p_us = kv.get("decode_pinned_us", 0)
        if p_mib > 0 and p_us > 0:
            ds.ram_vram_mibs = p_mib / (p_us / 1_000_000.0)

    io = _last(_RE_IO_DECODE, text)
    if io:
        kv = _kv(io.group(1))
        if kv.get("effective_mib_s", 0) > 0:
            ds.nvme_mibs = kv["effective_mib_s"]

    return ds
