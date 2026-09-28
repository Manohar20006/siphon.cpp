"""term: terminal-native TUI for siphon-studio (v4 design).

Implements the approved `siphon-tui-v4.html` mockup natively in Textual:
ink canvas, single-line box panels, model table, definition-list detail
with collapsible config sections, 10-tile metrics with sparklines, server
log tail, key-hint footer. Data strictly from `siphon.core` +
`siphon.tui.data` + `siphon.console.dstore`. Lives alongside the legacy
`siphon.tui` and the v2 `siphon.console` packages.
"""
