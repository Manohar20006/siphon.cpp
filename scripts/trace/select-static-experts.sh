#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRACE="${TRACE:-$ROOT/moe-routing-suite-1000-combined.jsonl}"
BUDGET_MIB="${BUDGET_MIB:-8192}"
PHASE="${PHASE:-all}"

if [[ "$PHASE" == "all" ]]; then
  OUT="${OUT:-$ROOT/expert-suites/moe-static-experts-suite-${BUDGET_MIB}.txt}"
  python3 "$ROOT/scripts/select-static-moe-experts.py" \
    "$TRACE" \
    "$OUT" \
    --budget-mib "$BUDGET_MIB"
else
  OUT="${OUT:-$ROOT/expert-suites/moe-static-${PHASE}-experts-suite-${BUDGET_MIB}.txt}"
  python3 "$ROOT/scripts/select-static-moe-experts.py" \
    "$TRACE" \
    "$OUT" \
    --budget-mib "$BUDGET_MIB" \
    --phase "$PHASE"
fi

echo "Wrote: $OUT"
