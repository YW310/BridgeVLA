#!/usr/bin/env bash
# Rebuild calibrated observations/labels from RAW data. Never alter old runs.
# Shared data for the calibrated baseline and optional motion-loss ablation.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if [[ $# != 2 ]]; then
  echo "Usage: $0 RAW_OHT_SEG_ROOT NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
RAW_ROOT="$(realpath "$1")"
OUTPUT_ROOT="$(realpath -m "$2")"
[[ ! -e "${OUTPUT_ROOT}" ]] || { echo "Use a new output directory: ${OUTPUT_ROOT}" >&2; exit 1; }
mkdir -p "${OUTPUT_ROOT}"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/finetune"
"${PYTHON_BIN}" -m finetune.OHT.cli audit \
  --root "${RAW_ROOT}" --output "${OUTPUT_ROOT}/audit.json" --seed 0
"${PYTHON_BIN}" -m finetune.OHT.cli build \
  --root "${RAW_ROOT}" --manifest "${OUTPUT_ROOT}/audit.json" \
  --config finetune/OHT/configs/dataset.yaml \
  --gripper-open -0.04 --gripper-close -0.006 \
  --replay-version 4 --chunk-horizon 16 --sample-stride 10 \
  --action-frame-stride 1 --action-dt 0.016666666666666666 \
  --output "${OUTPUT_ROOT}/replay-source-v4"
"${PYTHON_BIN}" -m finetune.OHT.cli validate \
  --replay "${OUTPUT_ROOT}/replay-source-v4" --workers 4
