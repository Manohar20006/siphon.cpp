#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/qwen3.6-moe-tq3.env}"

CONFIG_VARS=(
  MODEL
  MMPROJ
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
  MODEL_DRAFT
  SPEC_DRAFT_GPU_LAYERS
  MOE_GPU_CACHE_MIB
  MOE_PINNED_CACHE_MIB
  GDS_READ_THREADS
  CACHE_PROMPT
  SPEC_DRAFT_N_MAX
  SPEC_DRAFT_P_MIN
  CHECKPOINT_MIN_STEP
  CHAT_TEMPLATE
  REASONING
  REASONING_FORMAT
  REASONING_BUDGET
  CACHE_REUSE
  CHAT_TEMPLATE_KWARGS
  METRICS
  CACHE_RAM_MIB
  N_PARALLEL
  SPEC_TYPE
  RUNTIME_MODE
  DSTORAGE_CAPABILITY
  MOE_EXPERT_COUNT_MODE
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
MMPROJ="${MMPROJ:-}"
EXPERTS="${EXPERTS-default}"
ALIAS="${ALIAS:-qwen3.6-moe-tq3}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8090}"
CTX="${CTX:-65536}"
BATCH="${BATCH:-512}"
UBATCH="${UBATCH:-128}"
GPU_LAYERS="${GPU_LAYERS:-999}"
CACHE_TYPE="${CACHE_TYPE:-tq3_0}"
FLASH_ATTENTION="${FLASH_ATTENTION:-on}"
ROPE_SCALING="${ROPE_SCALING:-auto}"
MODEL_DRAFT="${MODEL_DRAFT:-}"
SPEC_DRAFT_GPU_LAYERS="${SPEC_DRAFT_GPU_LAYERS:-999}"
MOE_GPU_CACHE_MIB="${MOE_GPU_CACHE_MIB:-2816}"
MOE_PINNED_CACHE_MIB="${MOE_PINNED_CACHE_MIB:-8192}"
GDS_READ_THREADS="${GDS_READ_THREADS:-20}"
CACHE_PROMPT="${CACHE_PROMPT:-1}"
SPEC_DRAFT_N_MAX="${SPEC_DRAFT_N_MAX:-3}"
SPEC_DRAFT_P_MIN="${SPEC_DRAFT_P_MIN:-0}"
CHECKPOINT_MIN_STEP="${CHECKPOINT_MIN_STEP:-0}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
REASONING="${REASONING:-off}"
REASONING_FORMAT="${REASONING_FORMAT:-deepseek}"
REASONING_BUDGET="${REASONING_BUDGET:-0}"
CACHE_REUSE="${CACHE_REUSE:-0}"
METRICS="${METRICS:-1}"
CACHE_RAM_MIB="${CACHE_RAM_MIB:-8192}"
N_PARALLEL="${N_PARALLEL:-1}"
SPEC_TYPE="${SPEC_TYPE:-draft-mtp}"
RUNTIME_MODE="${RUNTIME_MODE:-qwen-dstorage}"
DSTORAGE_CAPABILITY="${DSTORAGE_CAPABILITY:-configured}"
MOE_EXPERT_COUNT_MODE="${MOE_EXPERT_COUNT_MODE:-default}"

if [[ "$MODEL" != /* ]]; then
  MODEL="$ROOT/$MODEL"
fi
if [[ -n "$MMPROJ" && "$MMPROJ" != /* ]]; then
  MMPROJ="$ROOT/$MMPROJ"
fi
if [[ -n "$EXPERTS" && "$EXPERTS" != /* ]]; then
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

if [[ -n "$EXPERTS" && ! -f "$EXPERTS" ]]; then
  echo "Missing expert suite: $EXPERTS"
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
  --parallel "$N_PARALLEL"
)

# DirectStorage expert residency is enabled only for profiles explicitly marked
# qwen-dstorage.  Imported MoE profiles use this same Siphon path after their
# architecture has passed capability validation; generic loading is an
# explicit fallback, never an automatic bypass of the expert-cache path.
if [[ "$RUNTIME_MODE" == "qwen-dstorage" ]]; then
  if [[ "$DSTORAGE_CAPABILITY" == "auto-probe" ]]; then
    echo "DirectStorage capability probe: validating MoE tensors and first compute (fail-closed)."
  fi
  SERVER_ARGS+=(
    -dsm
    --dstorage-moe-prefetch
    --moe-gpu-cache-mib "$MOE_GPU_CACHE_MIB"
    --moe-pinned-cache-mib "$MOE_PINNED_CACHE_MIB"
  )
fi

# Continuous batching interleaves tokens from all slots in one decode(). It is
# what makes multi-slot serving fair for several concurrent users. The tested
# single-slot path keeps --no-cont-batching to stay byte-for-byte equivalent to
# the known-good setup; multi-slot enables it automatically. Set CONT_BATCH=0
# to force it off for any N_PARALLEL.
if [[ "${CONT_BATCH:-auto}" == "auto" ]]; then
  if [[ "$N_PARALLEL" -gt 1 ]]; then
    SERVER_ARGS+=(--cont-batching)
  else
    SERVER_ARGS+=(--no-cont-batching)
  fi
elif [[ "${CONT_BATCH:-auto}" != "0" ]]; then
  SERVER_ARGS+=(--cont-batching)
else
  SERVER_ARGS+=(--no-cont-batching)
fi

if [[ -n "$MMPROJ" ]]; then
  SERVER_ARGS+=(--mmproj "$MMPROJ")
fi

if [[ "$METRICS" != "0" ]]; then
  SERVER_ARGS+=(--metrics)
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

if [[ -n "$MODEL_DRAFT" ]]; then
  SERVER_ARGS+=(--model-draft "$MODEL_DRAFT" --spec-draft-ngl "$SPEC_DRAFT_GPU_LAYERS")
fi

if [[ -n "$CHAT_TEMPLATE" ]]; then
  SERVER_ARGS+=(--chat-template "$CHAT_TEMPLATE")
fi

if [[ -n "${CHAT_TEMPLATE_KWARGS:-}" ]]; then
  SERVER_ARGS+=(--chat-template-kwargs "$CHAT_TEMPLATE_KWARGS")
fi

SERVER_ENV=(
  "GGML_CUDA_DISABLE_GRAPHS=${GGML_CUDA_DISABLE_GRAPHS:-1}"
  "LLAMA_GDS_READ_THREADS=$GDS_READ_THREADS"
)
if [[ "$RUNTIME_MODE" == "qwen-dstorage" ]]; then
  SERVER_ENV+=(
    "LLAMA_DSTORAGE_EVICT_TO_RAM_MIN_DECODE_HITS=${LLAMA_DSTORAGE_EVICT_TO_RAM_MIN_DECODE_HITS:-2}"
    "LLAMA_DSTORAGE_SUMMARY=${LLAMA_DSTORAGE_SUMMARY:-1}"
    "LLAMA_MOE_PINNED_STATIC_EXPERTS_FILE=$EXPERTS"
  )
  if [[ -n "${LLAMA_MOE_EXPERT_USED_OVERRIDE:-}" ]]; then
    SERVER_ENV+=("LLAMA_MOE_EXPERT_USED_OVERRIDE=$LLAMA_MOE_EXPERT_USED_OVERRIDE")
  elif [[ "$MOE_EXPERT_COUNT_MODE" != "native" ]]; then
    SERVER_ENV+=("LLAMA_MOE_EXPERT_USED_OVERRIDE=4")
  fi
fi
env "${SERVER_ENV[@]}" "$SERVER" "${SERVER_ARGS[@]}"
