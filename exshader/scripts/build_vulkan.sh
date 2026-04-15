#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source exshader/env.sh

if [[ "${1:-}" == "--clean" ]]; then
  rm -rf "$ET_BUILD_DIR"
fi

cmake -S . -B "$ET_BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$ET_BUILD_DIR/install" \
  -DPYTHON_EXECUTABLE="$REPO_ROOT/.venv/bin/python" \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DEXECUTORCH_BUILD_XNNPACK=OFF \
  -DEXECUTORCH_BUILD_COREML=OFF \
  -DEXECUTORCH_BUILD_OPENVINO=OFF \
  -DEXECUTORCH_BUILD_QNN=OFF \
  -DEXECUTORCH_BUILD_TESTS=OFF \
  -DEXECUTORCH_BUILD_EXTENSION_LLM=ON \
  -DEXECUTORCH_BUILD_EXTENSION_LLM_RUNNER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXSHADER_RUNTIME=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_LLM=ON \
  -DEXECUTORCH_BUILD_KERNELS_LLM_AOT=ON

cmake --build "$ET_BUILD_DIR" -j"$(nproc)" --target install --config Release

cmake -S examples/models/llama -B "$ET_BUILD_DIR/examples/models/llama" \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE="$REPO_ROOT/.venv/bin/python" \
  -DCMAKE_PREFIX_PATH="$ET_BUILD_DIR/install" \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DEXECUTORCH_BUILD_XNNPACK=OFF

cmake --build "$ET_BUILD_DIR/examples/models/llama" -j"$(nproc)" --config Release

echo "[build_vulkan] done"
echo "[build_vulkan] ET_BUILD_DIR=$ET_BUILD_DIR"
