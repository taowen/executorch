#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source exshader/env.sh

OUT_NAME="${1:-qwen3_0_6b_vulkan_pure_candidate.pte}"
OUT_PATH="$ET_PTE_DIR/$OUT_NAME"

PROFILE_JSON=''
if [[ -n "${ET_VULKAN_PARTITIONER_PROFILE:-}" ]]; then
  PROFILE_JSON="${ET_VULKAN_PARTITIONER_PROFILE}"
else
  compile_opts=()
  if [[ "${ET_USE_SHADER_BUNDLE_PROFILE:-0}" == "1" ]]; then
    compile_opts+=("\"shader_bundle_path\":\"$ET_SHADER_BUNDLE_DIR\"")
  fi
  if [[ "${ET_ENABLE_QUERYPOOL_PROFILE:-0}" == "1" ]]; then
    compile_opts+=("\"enable_querypool\":true")
  fi
  if [[ ${#compile_opts[@]} -gt 0 ]]; then
    compile_opts_joined="$(IFS=,; echo "${compile_opts[*]}")"
    PROFILE_JSON="{\"compile_options\":{${compile_opts_joined}}}"
  fi
fi

if [[ -n "$PROFILE_JSON" ]]; then
  ET_VULKAN_PARTITIONER_PROFILE="$PROFILE_JSON" \
  FLATC_EXECUTABLE="$REPO_ROOT/.venv/bin/flatc" \
  "$REPO_ROOT/.venv/bin/python" -m exshader.models.qwen3_0_6b.export \
    --output "$OUT_PATH"
else
  FLATC_EXECUTABLE="$REPO_ROOT/.venv/bin/flatc" \
  "$REPO_ROOT/.venv/bin/python" -m exshader.models.qwen3_0_6b.export \
    --output "$OUT_PATH"
fi

echo "[export_qwen3] done"
echo "[export_qwen3] OUT_PATH=$OUT_PATH"
