#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-gds-cuda}"

if [[ -z "${CUDACXX:-}" && -x /usr/local/cuda/bin/nvcc ]]; then
  export CUDACXX=/usr/local/cuda/bin/nvcc
fi

if [[ -f "$BUILD_DIR/CMakeCache.txt" ]] && grep -q "CMAKE_CUDA_COMPILER:FILEPATH=CMAKE_CUDA_COMPILER-NOTFOUND" "$BUILD_DIR/CMakeCache.txt"; then
  rm -rf "$BUILD_DIR"
fi

cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DGGML_CUDA=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$BUILD_DIR" --config Release -j"$(nproc)"

echo "Built: $BUILD_DIR/bin/llama-server"
