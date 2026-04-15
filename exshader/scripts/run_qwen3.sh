#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source exshader/env.sh

PTE_NAME="${1:-qwen3_0_6b_vulkan_pure_candidate.pte}"
PROMPT="${2:-Write a short poem about Vulkan.}"
PTE_PATH="$ET_PTE_DIR/$PTE_NAME"

TOKENIZER_JSON=$(ls ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/tokenizer.json | head -n1)

"$ET_BUILD_DIR/examples/models/llama/llama_main" \
  --model_path "$PTE_PATH" \
  --tokenizer_path "${TOKENIZER_JSON:?}" \
  --prompt "$PROMPT"
