#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source exshader/env.sh

PTE_NAME="${1:-qwen3_0_6b_vulkan_pure_candidate.pte}"
METHOD_NAME="${2:-forward}"
TOKEN_ID="${TOKEN_ID:-5328}"
INPUT_POS="${INPUT_POS:-0}"
DEBUG_BUFFER_SIZE="${DEBUG_BUFFER_SIZE:-1073741824}"
INSPECTOR_NUMERIC_GAP="${INSPECTOR_NUMERIC_GAP:-1}"
NUMERIC_GAP_METRICS="${NUMERIC_GAP_METRICS:-MSE}"
SAVE_PATCHED_ETRECORD="${SAVE_PATCHED_ETRECORD:-0}"

PTE_PATH="$ET_PTE_DIR/$PTE_NAME"
if [[ ! -f "$PTE_PATH" ]]; then
  echo "PTE not found: $PTE_PATH" >&2
  exit 1
fi

BASE="${PTE_NAME%.pte}"
ETDUMP_PATH="$ET_ARTIFACTS_ROOT/logs/${BASE}.etdp"
ETDUMP_DEBUG_PATH="$ET_ARTIFACTS_ROOT/logs/${BASE}.etdp.debug.bin"
INSPECTOR_CSV="$ET_ARTIFACTS_ROOT/logs/${BASE}.inspector.csv"
INSPECTOR_JSON="$ET_ARTIFACTS_ROOT/logs/${BASE}.inspector.json"
INSPECTOR_SUMMARY_JSON="$ET_ARTIFACTS_ROOT/logs/${BASE}.inspector.summary.json"
ETRECORD_PATH="${ET_ETRECORD_PATH:-$ET_ARTIFACTS_ROOT/etrecord/${BASE}.etrecord.bin}"
NUMERIC_GAP_JSON="$ET_ARTIFACTS_ROOT/logs/${BASE}.numeric_gap.json"
INSPECTOR_PRINT_TABLE="${INSPECTOR_PRINT_TABLE:-0}"

print_table_args=()
if [[ "$INSPECTOR_PRINT_TABLE" == "1" ]]; then
  print_table_args+=(--print-table)
fi

numeric_gap_args=()
if [[ "$INSPECTOR_NUMERIC_GAP" == "1" ]]; then
  numeric_gap_args+=(--numeric-gap-metrics "$NUMERIC_GAP_METRICS")
  numeric_gap_args+=(--numeric-gap-json-out "$NUMERIC_GAP_JSON")
else
  numeric_gap_args+=(--numeric-gap-metrics "")
fi

etrecord_patch_args=()
if [[ "$SAVE_PATCHED_ETRECORD" == "1" ]]; then
  ETRECORD_PATCHED_PATH="$ET_ARTIFACTS_ROOT/etrecord/${BASE}.inspector_patched.etrecord.bin"
  etrecord_patch_args+=(--etrecord-patched-out "$ETRECORD_PATCHED_PATH")
fi

PYTHONPATH="$ET_BUILD_DIR:$REPO_ROOT/src" \
MANGOHUD=0 \
FLATC_EXECUTABLE="${FLATC_EXECUTABLE:-$REPO_ROOT/.venv/bin/flatc}" \
"$REPO_ROOT/.venv/bin/python" -m exshader.diag.collect_inspector_artifacts \
  --pte "$PTE_PATH" \
  --method "$METHOD_NAME" \
  --token-id "$TOKEN_ID" \
  --input-pos "$INPUT_POS" \
  --debug-buffer-size "$DEBUG_BUFFER_SIZE" \
  --etdump-out "$ETDUMP_PATH" \
  --etdump-debug-out "$ETDUMP_DEBUG_PATH" \
  --etrecord "$ETRECORD_PATH" \
  "${etrecord_patch_args[@]}" \
  --inspector-csv-out "$INSPECTOR_CSV" \
  --inspector-json-out "$INSPECTOR_JSON" \
  --summary-json-out "$INSPECTOR_SUMMARY_JSON" \
  "${numeric_gap_args[@]}" \
  "${print_table_args[@]}"

echo "[inspect_with_inspector] done"
echo "[inspect_with_inspector] ETDUMP_PATH=$ETDUMP_PATH"
echo "[inspect_with_inspector] INSPECTOR_CSV=$INSPECTOR_CSV"
echo "[inspect_with_inspector] INSPECTOR_JSON=$INSPECTOR_JSON"
echo "[inspect_with_inspector] INSPECTOR_SUMMARY_JSON=$INSPECTOR_SUMMARY_JSON"
echo "[inspect_with_inspector] NUMERIC_GAP_JSON=$NUMERIC_GAP_JSON"
