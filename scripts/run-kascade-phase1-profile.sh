#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=${MODEL:-/home/manohar/Desktop/inference/qwen3.6/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}
STATIC_EXPERTS=${STATIC_EXPERTS:-$ROOT/moe-static-experts-suite-8192.txt}
OUT_DIR=${OUT_DIR:-$ROOT/kascade-phase1-results}
TARGET=${1:-2048}
MODE=${2:-nsys}
PROMPT=$OUT_DIR/prompts/coding-$TARGET.txt
LOG=$OUT_DIR/profile-$TARGET.log
REPORT=$OUT_DIR/profile-$TARGET

mkdir -p "$OUT_DIR"

if [[ ! -f "$PROMPT" ]]; then
    echo "missing prompt: $PROMPT" >&2
    echo "run scripts/generate-kascade-phase1-prompts.py first" >&2
    exit 1
fi

CMD=(
    "$ROOT/build-gds-cuda/bin/llama-cli"
    -m "$MODEL"
    -ngl 999
    -fit off
    -c 20480
    -b 2048
    -ub 1024
    -fa on
    -dsm
    --dstorage-moe-prefetch
    --moe-gpu-cache-mib 2304
    --moe-pinned-cache-mib 8192
    --spec-type draft-mtp
    --spec-draft-n-max 3
    --spec-draft-p-min 0
    --reasoning off
    --reasoning-budget 0
    -s 4242
    -st
    -n 1
    -f "$PROMPT"
    --no-display-prompt
    --simple-io
    --perf
    --no-warmup
)

export LLAMA_GDS_READ_THREADS=${LLAMA_GDS_READ_THREADS:-20}
export LLAMA_DSTORAGE_SUMMARY=1
export LLAMA_MOE_PINNED_STATIC_EXPERTS_FILE="$STATIC_EXPERTS"

if [[ "$MODE" == "nsys" ]]; then
    rm -f "$REPORT.nsys-rep" "$REPORT.sqlite"
    nsys profile \
        --trace=cuda,nvtx \
        --sample=none \
        --cuda-graph-trace=node \
        --force-overwrite=true \
        --output="$REPORT" \
        "${CMD[@]}" 2>&1 | tee "$LOG"
    nsys stats \
        --report cuda_gpu_kern_sum:base \
        --format csv \
        --output "$REPORT-kernels" \
        --force-overwrite=true \
        "$REPORT.nsys-rep"
else
    "${CMD[@]}" 2>&1 | tee "$LOG"
fi
