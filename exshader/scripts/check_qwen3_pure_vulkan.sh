#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source exshader/env.sh

PTE_NAME="${1:-qwen3_0_6b_vulkan_pure_candidate.pte}"
PTE_PATH="$ET_PTE_DIR/$PTE_NAME"
TOKENIZER_JSON=$(ls ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json | head -n1)

"$REPO_ROOT/.venv/bin/python" exshader/check_pure_vulkan.py \
  --pte "$PTE_PATH" \
  --flatc "$REPO_ROOT/.venv/bin/flatc" \
  --run-cmd "$ET_BUILD_DIR/examples/models/llama/llama_main --model_path $PTE_PATH --tokenizer_path ${TOKENIZER_JSON:?} --prompt 'Write a short poem about Vulkan.'"
