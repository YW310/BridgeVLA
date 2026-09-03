#!/usr/bin/env bash
set -euo pipefail

# O2 closed-loop evaluation wrapper.
# Caller-provided environment variables override these defaults.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export EXP_CFG_PATH="${EXP_CFG_PATH:-configs/rlbench_o2_semantic_gt.yaml}"
export ORACLE_PROVIDER="${ORACLE_PROVIDER:-rlbench_gt}"

case "${ORACLE_PROVIDER}" in
  rlbench_gt)
    export ORACLE_STRICT="${ORACLE_STRICT:-1}"
    export ORACLE_DEBUG="${ORACLE_DEBUG:-1}"
    ;;
  none)
    export ORACLE_STRICT="${ORACLE_STRICT:-0}"
    export ORACLE_DEBUG="${ORACLE_DEBUG:-0}"
    ;;
  *)
    echo "Unsupported ORACLE_PROVIDER=${ORACLE_PROVIDER}; expected rlbench_gt or none." >&2
    exit 2
    ;;
esac

exec bash "${SCRIPT_DIR}/eval.sh"
