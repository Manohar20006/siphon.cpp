#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/qwen3-coder-next.env}"

CONFIG_VARS=(
  MODEL
  MODEL_DRAFT
  EXPERTS
  ALIAS
  HOST
  PORT
  CTX
  BATCH
  UBATCH
  GPU_LAYERS
  CACHE_TYPE
  FLASH_ATTENTION
  ROPE_SCALING
  MOE_GPU_CACHE_MIB
  MOE_PINNED_CACHE_MIB
  GDS_READ_THREADS
  SPEC_TYPE
  SPEC_DRAFT_N_MAX
  SPEC_DRAFT_P_MIN
  SPEC_DRAFT_GPU_LAYERS
  CHECKPOINT_MIN_STEP
  CHAT_TEMPLATE
  REASONING
  REASONING_FORMAT
  REASONING_BUDGET
  CACHE_REUSE
  CACHE_PROMPT
  CHAT_TEMPLATE_KWARGS
  METRICS
  CACHE_RAM_MIB
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

MODEL="${MODEL:-/home/manohar/Downloads/Qwen3-Coder-Next-Q4_K_M.gguf}"
MODEL_DRAFT="${MODEL_DRAFT:-}"
EXPERTS="${EXPERTS:-}"
ALIAS="${ALIAS:-qwen3-coder-next}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8090}"
CTX="${CTX:-65536}"
BATCH="${BATCH:-512}"
UBATCH="${UBATCH:-128}"
GPU_LAYERS="${GPU_LAYERS:-999}"
CACHE_TYPE="${CACHE_TYPE:-tq3_0}"
FLASH_ATTENTION="${FLASH_ATTENTION:-on}"
ROPE_SCALING="${ROPE_SCALING:-auto}"
MOE_GPU_CACHE_MIB="${MOE_GPU_CACHE_MIB:-2816}"
MOE_PINNED_CACHE_MIB="${MOE_PINNED_CACHE_MIB:-8192}"
GDS_READ_THREADS="${GDS_READ_THREADS:-20}"
SPEC_TYPE="${SPEC_TYPE:-none}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-3}"
SPEC_DRAFT_P_MIN="${SPEC_DRAFT_P_MIN:-0}"
SPEC_DRAFT_GPU_LAYERS="${SPEC_DRAFT_GPU_LAYERS:-999}"
CHECKPOINT_MIN_STEP="${CHECKPOINT_MIN_STEP:-0}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
REASONING="${REASONING:-off}"
REASONING_FORMAT="${REASONING_FORMAT:-deepseek}"
REASONING_BUDGET="${REASONING_BUDGET:-0}"
CACHE_REUSE="${CACHE_REUSE:-0}"
CACHE_PROMPT="${CACHE_PROMPT:-0}"
METRICS="${METRICS:-1}"
CACHE_RAM_MIB="${CACHE_RAM_MIB:-8192}"

if [[ "$MODEL" != /* ]]; then
  MODEL="$ROOT/$MODEL"
fi
if [[ -n "$EXPERTS" ]]; then
  if [[ "$EXPERTS" != /* ]]; then
    EXPERTS="$ROOT/$EXPERTS"
  fi

  if [[ ! -f "$EXPERTS" ]]; then
    echo "Missing expert suite: $EXPERTS"
    exit 1
  fi
fi

SERVER="$ROOT/build-gds-cuda/bin/llama-server"

if [[ ! -x "$SERVER" ]]; then
  echo "Missing $SERVER"
  echo "Run ./scripts/build-gds-cuda.sh first."
  exit 1
fi

if [[ ! -f "$MODEL" ]]; then
  echo "Missing model: $MODEL"
  exit 1
fi

cd "$ROOT"

SERVER_ARGS=(
  -m "$MODEL"
  -ngl "$GPU_LAYERS"
  -c "$CTX"
  -b "$BATCH"
  -ub "$UBATCH"
  -ctk "$CACHE_TYPE"
  -ctv "$CACHE_TYPE"
  -dsm
  --dstorage-moe-prefetch
  --moe-gpu-cache-mib "$MOE_GPU_CACHE_MIB"
  --moe-pinned-cache-mib "$MOE_PINNED_CACHE_MIB"
  --spec-type "$SPEC_TYPE"
  --spec-draft-n-max "$SPEC_DRAFT_N_MAX"
  --spec-draft-p-min "$SPEC_DRAFT_P_MIN"
  --checkpoint-min-step "$CHECKPOINT_MIN_STEP"
  --cache-ram "$CACHE_RAM_MIB"
  --reasoning "$REASONING"
  --reasoning-format "$REASONING_FORMAT"
  --reasoning-budget "$REASONING_BUDGET"
  --jinja
  --cache-reuse "$CACHE_REUSE"
  --alias "$ALIAS"
  --host "$HOST"
  --port "$PORT"
  --parallel 1
  --no-cont-batching
)

if [[ -n "$MODEL_DRAFT" ]]; then
  SERVER_ARGS+=(--model-draft "$MODEL_DRAFT" --spec-draft-ngl "$SPEC_DRAFT_GPU_LAYERS")
fi

if [[ "$CACHE_PROMPT" != "0" ]]; then
  SERVER_ARGS+=(--cache-prompt)
fi

if [[ "$FLASH_ATTENTION" != "0" && "$FLASH_ATTENTION" != "off" ]]; then
  SERVER_ARGS+=(--flash-attn on)
else
  SERVER_ARGS+=(--flash-attn off)
fi

if [[ "$ROPE_SCALING" != "auto" && -n "$ROPE_SCALING" ]]; then
  SERVER_ARGS+=(--rope-scaling "$ROPE_SCALING")
fi

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
LLAMA_DSTORAGE_EVICT_TO_RAM_MIN_DECODE_HITS="${LLAMA_DSTORAGE_EVICT_TO_RAM_MIN_DECODE_HITS:-2}" \
LLAMA_DSTORAGE_SPLIT_DOWN_POOLS="${LLAMA_DSTORAGE_SPLIT_DOWN_POOLS:-1}" \
LLAMA_DSTORAGE_SUMMARY="${LLAMA_DSTORAGE_SUMMARY:-1}" \
LLAMA_MOE_EXPERT_USED_OVERRIDE="${LLAMA_MOE_EXPERT_USED_OVERRIDE:-6}" \
LLAMA_MOE_PINNED_STATIC_EXPERTS_FILE="${EXPERTS:-}" \
LLAMA_DSTORAGE_EVICT_TO_RAM="${LLAMA_DSTORAGE_EVICT_TO_RAM:-1}" \
"$SERVER" "${SERVER_ARGS[@]}"
