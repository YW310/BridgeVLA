#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
PIPELINE_PY="${REPO_ROOT}/tools/prepare_oht_seg_data.py"
if [[ ! -f "${PIPELINE_PY}" ]]; then
  echo "[OHT prepare] Missing pipeline script: ${PIPELINE_PY}. Sync the complete BridgeVLA repository." >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" > /dev/null 2>&1; then
  echo "[OHT prepare] Python interpreter not found: ${PYTHON_BIN}. Activate bridgevla or set PYTHON_BIN." >&2
  exit 1
fi
echo "[OHT prepare] repo=${REPO_ROOT} python=${PYTHON_BIN}"
exec "${PYTHON_BIN}" -u "${PIPELINE_PY}" "$@"
