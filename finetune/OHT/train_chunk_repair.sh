#!/usr/bin/env bash
# Validate a fresh calibrated replay before starting a NEW chunk training run.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
usage() {
  cat <<'EOF'
Usage: train_chunk_repair.sh REPLAY NEW_RUN INIT_CHECKPOINT [--check-only|--dry-run]
  REPLAY must contain a complete calibrated V4 replay, not the raw dataset.
  --check-only validates all splits and the training configuration on CPU.
  --dry-run prints commands only; does not validate files or start training.
Environment:
  PYTHON_BIN=python GPUS_PER_NODE=8 VALIDATE_WORKERS=4 MASTER_PORT=29500
  CONFIG=finetune/OHT/configs/oht_chunk_v4_delta_state_tcp_guided_periodic_8x40.yaml
  Default: periodic joint inputs on calibrated data, keeping original losses/LR.
  Use CONFIG=...tcp_guided_repair_8x40.yaml for the combined objective/optimizer experiment.
  New encoding requires a new run; --init-checkpoint retains the encoder only.
  GPUS_PER_NODE changes process count; it does not make this 8x40GB recipe fit 16GB.
EOF
}
if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if [[ $# -lt 3 || $# -gt 4 ]]; then usage >&2; exit 2; fi
REPLAY="$(realpath -m "$1")"
RUN="$(realpath -m "$2")"
INITIAL="$(realpath -m "$3")"
MODE="${4:-run}"
case "${MODE}" in run|--check-only|--dry-run) ;; *) usage >&2; exit 2 ;; esac
cd "${REPO_ROOT}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
VALIDATE_WORKERS="${VALIDATE_WORKERS:-4}"
MASTER_PORT="${MASTER_PORT:-29500}"
CONFIG="${CONFIG:-finetune/OHT/configs/oht_chunk_v4_delta_state_tcp_guided_periodic_8x40.yaml}"
for value in "${GPUS_PER_NODE}" "${VALIDATE_WORKERS}" "${MASTER_PORT}"; do
  [[ ${value} =~ ^[1-9][0-9]*$ ]] || { echo 'GPU/worker counts and port must be positive integers' >&2; exit 2; }
done
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/finetune${PYTHONPATH:+:${PYTHONPATH}}"
validate=("${PYTHON_BIN}" -m finetune.OHT.cli validate --replay "${REPLAY}" --workers "${VALIDATE_WORKERS}")
check=(env WORLD_SIZE="${GPUS_PER_NODE}" RANK=0 LOCAL_RANK=0
  "${PYTHON_BIN}" -m finetune.OHT.train --config "${CONFIG}" --replay "${REPLAY}" --validate-only)
train=("${PYTHON_BIN}" -m torch.distributed.run --standalone --nnodes=1
  --nproc_per_node="${GPUS_PER_NODE}" --master_port="${MASTER_PORT}" -m finetune.OHT.train
  --config "${CONFIG}" --replay "${REPLAY}" --init-checkpoint "${INITIAL}"
  --output "${RUN}" --data-checksum once)
if [[ ${MODE} == --dry-run ]]; then
  for name in validate check train; do
    declare -n command="${name}"
    printf '%q ' "${command[@]}"; printf '\n'
    unset -n command
  done
  exit 0
fi
[[ ! -e ${RUN} ]] || { echo "Use a new run directory: ${RUN}" >&2; exit 1; }
[[ -f ${INITIAL} ]] || { echo "Checkpoint not found: ${INITIAL}" >&2; exit 1; }
[[ -f ${REPLAY}/complete.json ]] || { echo "Replay is incomplete: ${REPLAY}" >&2; exit 1; }
"${validate[@]}"
"${check[@]}"
if [[ ${MODE} == --check-only ]]; then exit 0; fi
# Checks above are CPU-only. Fail before torchrun/model construction if the
# requested training hardware is absent. No changes to a conda environment.
"${PYTHON_BIN}" -c 'import sys,torch; requested=int(sys.argv[1]); available=torch.cuda.device_count(); assert available >= requested, f"Requested {requested} GPUs, found {available}; use an appropriate training machine"' "${GPUS_PER_NODE}"
exec "${train[@]}"
