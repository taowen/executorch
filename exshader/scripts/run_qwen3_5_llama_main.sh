#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source exshader/env.sh
mkdir -p "$ET_ARTIFACTS_ROOT/logs"

PTE_NAME="${1:-qwen3_5_0_8b_vulkan_fp32_placeholder_fix.pte}"
PROMPT="${2:-Write one short sentence about Vulkan.}"
PTE_PATH="$ET_PTE_DIR/$PTE_NAME"

if [[ ! -f "$PTE_PATH" ]]; then
  echo "PTE not found: $PTE_PATH" >&2
  exit 1
fi

TOKENIZER_JSON="$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/* | head -n1)/tokenizer.json"
if [[ ! -f "$TOKENIZER_JSON" ]]; then
  echo "Qwen3.5 tokenizer not found under ~/.cache/huggingface/hub" >&2
  exit 1
fi

MANGOHUD=0 \
ET_DELEGATE_ABI_TRACE_PATH="$ET_ARTIFACTS_ROOT/logs/runtime_abi_llama_main_qwen3_5_latest.jsonl" \
ET_VULKAN_INIT_TRACE_PATH="$ET_ARTIFACTS_ROOT/logs/vk_llama_main_qwen3_5_latest.jsonl" \
"$ET_BUILD_DIR/examples/models/llama/llama_main" \
  --model_path="$PTE_PATH" \
  --tokenizer_path="$TOKENIZER_JSON" \
  --prompt="$PROMPT" \
  --max_new_tokens="${MAX_NEW_TOKENS:-8}" \
  --temperature="${TEMPERATURE:-0}"
