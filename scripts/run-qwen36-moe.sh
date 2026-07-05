#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/qwen3.6-moe-tq3.env}"

CONFIG_VARS=(
  MODEL
  EXPERTS
  ALIAS
  HOST
  PORT
  CTX
  BATCH
  UBATCH
  MOE_GPU_CACHE_MIB
  MOE_PINNED_CACHE_MIB
  GDS_READ_THREADS
  SPEC_DRAFT_N_MAX
  SPEC_DRAFT_P_MIN
  CHAT_TEMPLATE
  REASONING
  REASONING_FORMAT
  REASONING_BUDGET
  CACHE_REUSE
  CHAT_TEMPLATE_KWARGS
  METRICS
)

declare -A USER_OVERRIDES=()
for var in "${CONFIG_VARS[@]}"; do
  if [[ -v "$var" ]]; then
    USER_OVERRIDES["$var"]="${!var}"
  fi
done

if [[ -f "$CONFIG" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$CONFIG"
  set +a
fi

for var in "${CONFIG_VARS[@]}"; do
  if [[ ${USER_OVERRIDES[$var]+set} ]]; then
    printf -v "$var" '%s' "${USER_OVERRIDES[$var]}"
  fi
done

MODEL="${MODEL:-models/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf}"
EXPERTS="${EXPERTS:-expert-suites/moe-static-experts-suite-8192.txt}"
ALIAS="${ALIAS:-qwen3.6-moe-tq3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8090}"
CTX="${CTX:-65536}"
BATCH="${BATCH:-512}"
UBATCH="${UBATCH:-128}"
MOE_GPU_CACHE_MIB="${MOE_GPU_CACHE_MIB:-2816}"
MOE_PINNED_CACHE_MIB="${MOE_PINNED_CACHE_MIB:-8192}"
GDS_READ_THREADS="${GDS_READ_THREADS:-20}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-3}"
SPEC_DRAFT_P_MIN="${SPEC_DRAFT_P_MIN:-0}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
REASONING="${REASONING:-off}"
REASONING_FORMAT="${REASONING_FORMAT:-deepseek}"
REASONING_BUDGET="${REASONING_BUDGET:-0}"
CACHE_REUSE="${CACHE_REUSE:-0}"
METRICS="${METRICS:-1}"

if [[ "$MODEL" != /* ]]; then
  MODEL="$ROOT/$MODEL"
fi
if [[ "$EXPERTS" != /* ]]; then
  EXPERTS="$ROOT/$EXPERTS"
fi

SERVER="$ROOT/build-gds-cuda/bin/llama-server"

if [[ ! -x "$SERVER" ]]; then
  echo "Missing $SERVER"
  echo "Run ./scripts/build-gds-cuda.sh first."
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Missing model: $MODEL"
  echo "Place the GGUF in $ROOT/models/ or set MODEL=/path/to/model.gguf"
  exit 1
fi

if [[ ! -f "$EXPERTS" ]]; then
  echo "Missing expert suite: $EXPERTS"
  exit 1
fi

cd "$ROOT"

SERVER_ARGS=(
  -m "$MODEL"
  -ngl 999
  -c "$CTX"
  -b "$BATCH"
  -ub "$UBATCH"
  -fa on
  -ctk tq3_0
  -ctv tq3_0
  -dsm
  --dstorage-moe-prefetch
  --moe-gpu-cache-mib "$MOE_GPU_CACHE_MIB"
  --moe-pinned-cache-mib "$MOE_PINNED_CACHE_MIB"
  --spec-type draft-mtp
  --spec-draft-n-max "$SPEC_DRAFT_N_MAX"
  --spec-draft-p-min "$SPEC_DRAFT_P_MIN"
  --reasoning "$REASONING"
  --reasoning-format "$REASONING_FORMAT"
  --reasoning-budget "$REASONING_BUDGET"
  --cache-prompt
  --cache-reuse "$CACHE_REUSE"
  --alias "$ALIAS"
  --host "$HOST"
  --port "$PORT"
  --parallel 1
  --no-cont-batching
)

if [[ "$METRICS" != "0" ]]; then
  SERVER_ARGS+=(--metrics)
fi

if [[ -n "$CHAT_TEMPLATE" ]]; then
  SERVER_ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi

if [[ -n "${CHAT_TEMPLATE_KWARGS:-}" ]]; then
  SERVER_ARGS+=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS")
fi

GGML_CUDA_DISABLE_GRAPHS="${GGML_CUDA_DISABLE_GRAPHS:-1}" \
LLAMA_GDS_READ_THREADS="$GDS_READ_THREADS" \
LLAMA_DSTORAGE_SUMMARY="${LLAMA_DSTORAGE_SUMMARY:-1}" \
LLAMA_MOE_EXPERT_USED_OVERRIDE="${LLAMA_MOE_EXPERT_USED_OVERRIDE:-4}" \
LLAMA_MOE_PINNED_STATIC_EXPERTS_FILE="$EXPERTS" \
"$SERVER" "${SERVER_ARGS[@]}"
