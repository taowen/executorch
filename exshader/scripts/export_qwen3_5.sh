#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source exshader/env.sh

OUT_NAME="${1:-qwen3_5_0_8b_vulkan_fp32.pte}"
OUT_PATH="$ET_PTE_DIR/$OUT_NAME"
ETRECORD_PATH_DEFAULT="$ET_ARTIFACTS_ROOT/etrecord/${OUT_NAME%.pte}.etrecord.bin"
ETRECORD_PATH="${ET_ETRECORD_PATH:-$ETRECORD_PATH_DEFAULT}"
mkdir -p "$(dirname "$ETRECORD_PATH")"

common_args=(
  base.model_class=qwen3_5_0_8b
  base.params=examples/models/qwen3_5/config/0_8b_config.json
  model.enable_dynamic_shape=false
  model.use_kv_cache=true
  model.use_sdpa_with_kv_cache=false
  model.quantize_kv_cache=false
  backend.vulkan.enabled=true
  backend.vulkan.force_fp16=false
  model.dtype_override=fp32
  export.max_seq_length=2048
  export.max_context_length=2048
  export.output_name="$OUT_PATH"
)

if [[ "${ET_GENERATE_ETRECORD:-0}" == "1" ]]; then
  common_args+=("debug.generate_etrecord=true")
fi

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
  ET_ETRECORD_PATH="$ETRECORD_PATH" \
  FLATC_EXECUTABLE="$REPO_ROOT/.venv/bin/flatc" \
  "$REPO_ROOT/.venv/bin/python" -m exshader.export_llm \
    "${common_args[@]}"
else
  ET_ETRECORD_PATH="$ETRECORD_PATH" \
  FLATC_EXECUTABLE="$REPO_ROOT/.venv/bin/flatc" \
  "$REPO_ROOT/.venv/bin/python" -m exshader.export_llm \
    "${common_args[@]}"
fi

echo "[export_qwen3_5] done"
echo "[export_qwen3_5] OUT_PATH=$OUT_PATH"
if [[ "${ET_GENERATE_ETRECORD:-0}" == "1" ]]; then
  echo "[export_qwen3_5] ETRECORD_PATH=$ETRECORD_PATH"
fi
