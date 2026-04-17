#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source exshader/env.sh

BASE_PTE="${1:-qwen3_5_0_8b_vulkan_allowlist_empty_after_fix.pte}"
CAND_PTE="${2:-qwen3_5_0_8b_vulkan_fp32_diag_base.pte}"
METHOD_NAME="${3:-forward}"

bash exshader/scripts/inspect_with_inspector.sh "$BASE_PTE" "$METHOD_NAME"
bash exshader/scripts/inspect_with_inspector.sh "$CAND_PTE" "$METHOD_NAME"

BASE_KEY="${BASE_PTE%.pte}"
CAND_KEY="${CAND_PTE%.pte}"

BASE_CSV="$ET_ARTIFACTS_ROOT/logs/${BASE_KEY}.inspector.csv"
CAND_CSV="$ET_ARTIFACTS_ROOT/logs/${CAND_KEY}.inspector.csv"
BASE_SUMMARY="$ET_ARTIFACTS_ROOT/logs/${BASE_KEY}.inspector.summary.json"
CAND_SUMMARY="$ET_ARTIFACTS_ROOT/logs/${CAND_KEY}.inspector.summary.json"

REPORT_MD="$ET_ARTIFACTS_ROOT/logs/compare_${BASE_KEY}_vs_${CAND_KEY}.inspector.md"
REPORT_JSON="$ET_ARTIFACTS_ROOT/logs/compare_${BASE_KEY}_vs_${CAND_KEY}.inspector.json"

PYTHONPATH="$ET_BUILD_DIR:$REPO_ROOT/src" \
"$REPO_ROOT/.venv/bin/python" -m exshader.diag.compare_inspector_runs \
  --baseline-csv "$BASE_CSV" \
  --candidate-csv "$CAND_CSV" \
  --baseline-summary-json "$BASE_SUMMARY" \
  --candidate-summary-json "$CAND_SUMMARY" \
  --report-out "$REPORT_MD" \
  --json-out "$REPORT_JSON"

echo "[inspect_compare] done"
echo "[inspect_compare] REPORT_MD=$REPORT_MD"
echo "[inspect_compare] REPORT_JSON=$REPORT_JSON"
