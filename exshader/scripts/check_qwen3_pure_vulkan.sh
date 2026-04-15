#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source exshader/env.sh
source exshader/scripts/common.sh

PTE_NAME="$(exshader_resolve_qwen3_pte_name "$ET_PTE_DIR" "${1:-}" || true)"
PTE_PATH="$ET_PTE_DIR/$PTE_NAME"
if [[ -z "${PTE_NAME:-}" ]]; then
  echo "No Qwen3 PTE found under $ET_PTE_DIR" >&2
  exit 1
fi
if [[ ! -f "$PTE_PATH" ]]; then
  echo "PTE not found: $PTE_PATH" >&2
  exit 1
fi
TOKENIZER_JSON="$(exshader_find_qwen3_tokenizer)"

"$REPO_ROOT/.venv/bin/python" exshader/check_pure_vulkan.py \
  --pte "$PTE_PATH" \
  --flatc "$REPO_ROOT/.venv/bin/flatc" \
  --run-cmd "$REPO_ROOT/.venv/bin/python -m exshader.recipes.llm_decode --model $PTE_PATH --tokenizer ${TOKENIZER_JSON:?} --prompt 'Write a short poem about Vulkan.' --max-new-tokens 16"
