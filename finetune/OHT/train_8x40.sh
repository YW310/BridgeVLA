#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
export TRANSFORMERS_VERBOSITY=error

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    -m finetune.OHT.train \
    "$@"
