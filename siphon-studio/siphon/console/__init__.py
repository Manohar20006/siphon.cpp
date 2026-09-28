"""console: v2 dashboard-first TUI for siphon-studio.

Visual language translated from the `siphon-console.html` OpenDesign mockup
(premium ops-console: ink canvas, indigo accent, KPI strip, roster + config
form, 10-tile metrics grid, activity timeline). Built strictly on
`siphon.core` for profile/runner logic plus `siphon.tui.data` for host
telemetry; `dstore` adds the DSTORAGE_MOE log-tail parsing the v2 metrics
need. Lives alongside the legacy `siphon.tui` package.
"""
