#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source exshader/env.sh

# 1) Build core targets first (pure Vulkan config)
bash exshader/scripts/build_vulkan.sh "${1:-}"

# 2) Refresh install tree used by examples/models/llama find_package(executorch)
cmake --build "$ET_BUILD_DIR" -j"$(nproc)" --target install

# 3) Configure + build standalone llama_main against the refreshed install tree
cmake -S examples/models/llama -B "$ET_BUILD_DIR/examples/models/llama" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DPYTHON_EXECUTABLE="$REPO_ROOT/.venv/bin/python" \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_PTHREADPOOL=ON \
  -DEXECUTORCH_BUILD_CPUINFO=ON \
  -DCMAKE_PREFIX_PATH="$ET_BUILD_DIR/install"

cmake --build "$ET_BUILD_DIR/examples/models/llama" -j"$(nproc)" --target llama_main

echo "[build_llama_main] done"
echo "[build_llama_main] binary=$ET_BUILD_DIR/examples/models/llama/llama_main"
file "$ET_BUILD_DIR/examples/models/llama/llama_main" | sed 's/^/[build_llama_main] /'
