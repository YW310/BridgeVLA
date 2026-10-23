#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
exec "${PYTHON_BIN:-python}" -u "${REPO_ROOT}/tools/prepare_oht_seg_data.py" "$@"
