#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-error}"

# Use a static single-node rendezvous so MASTER_PORT selects the actual port.
# All training arguments (config, replay, checkpoints, role cache) pass through.
exec torchrun \
    --nnodes=1 \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" \
    -m finetune.OHT.train \
    "$@"
