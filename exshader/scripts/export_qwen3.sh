#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source exshader/env.sh

OUT_NAME="${1:-qwen3_0_6b_vulkan_pure_candidate.pte}"
OUT_PATH="$ET_PTE_DIR/$OUT_NAME"

PROFILE_JSON=''
if [[ "${ET_USE_SHADER_BUNDLE_PROFILE:-0}" == "1" ]]; then
  PROFILE_JSON="{\"compile_options\":{\"shader_bundle_path\":\"$ET_SHADER_BUNDLE_DIR\"}}"
fi

if [[ -n "$PROFILE_JSON" ]]; then
  ET_VULKAN_PARTITIONER_PROFILE="$PROFILE_JSON" \
  FLATC_EXECUTABLE="$REPO_ROOT/.venv/bin/flatc" \
  "$REPO_ROOT/.venv/bin/python" -m exshader.export_llm \
    base.model_class=qwen3_0_6b \
    base.params=examples/models/qwen3/config/0_6b_config.json \
    model.enable_dynamic_shape=true \
    model.use_kv_cache=true \
    model.use_sdpa_with_kv_cache=true \
    model.quantize_kv_cache=false \
    backend.vulkan.enabled=true \
    backend.vulkan.force_fp16=true \
    model.dtype_override=fp32 \
    export.output_name="$OUT_PATH"
else
  FLATC_EXECUTABLE="$REPO_ROOT/.venv/bin/flatc" \
  "$REPO_ROOT/.venv/bin/python" -m exshader.export_llm \
    base.model_class=qwen3_0_6b \
    base.params=examples/models/qwen3/config/0_6b_config.json \
    model.enable_dynamic_shape=true \
    model.use_kv_cache=true \
    model.use_sdpa_with_kv_cache=true \
    model.quantize_kv_cache=false \
    backend.vulkan.enabled=true \
    backend.vulkan.force_fp16=true \
    model.dtype_override=fp32 \
    export.output_name="$OUT_PATH"
fi

echo "[export_qwen3] done"
echo "[export_qwen3] OUT_PATH=$OUT_PATH"
