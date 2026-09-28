#!/usr/bin/env bash
# Launch the siphon.cpp server with persistent logs that survive reboots.
#
# Usage:
#   ./scripts/run-qwen36-persistent.sh              # default (PORT=8080, ALIAS=qwen3.6-gds)
#   PORT=8090 ALIAS=kat-coder ./scripts/run-qwen36-persistent.sh
#
# Logs are written to $STATE_DIR/<alias>.log with rotation via logrotate-style
# timestamped backups on each start. The previous log is preserved as
# <alias>.log.1 instead of being overwritten.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="${STATE_DIR:-$HOME/.local/state/siphon}"
ALIAS="${ALIAS:-qwen3.6-gds}"

mkdir -p "$STATE_DIR"

LOG="$STATE_DIR/$ALIAS.log"

# rotate: keep one previous run
if [[ -f "$LOG" ]]; then
  mv -f "$LOG" "$LOG.1"
fi

echo "Persistent log: $LOG"
exec env ALIAS="$ALIAS" "$ROOT/scripts/run-qwen36-moe.sh" > "$LOG" 2>&1