#!/usr/bin/env bash
set -euo pipefail

# BridgeVLA OHT model_path: a trained checkpoint, or a run with model_last.pth.
# The existing loader still needs cached/accessible PaliGemma and processor assets.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGEVLA_ROOT="${BRIDGEVLA_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SERVER_ARGS=("$@")
SERVER_DEFAULTS=()

die() { echo "[ERROR] $*" >&2; exit 1; }
[[ -f "${BRIDGEVLA_ROOT}/finetune/OHT/vla_server.py" ]] || \
  die "Missing BridgeVLA server: ${BRIDGEVLA_ROOT}/finetune/OHT/vla_server.py"
command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python interpreter not found: ${PYTHON_BIN}"
cd "${BRIDGEVLA_ROOT}"
BRIDGEVLA_ROOT="$(pwd)"

# Prefer this checkout over stale project PYTHONPATH entries.
export PYTHONPATH="${BRIDGEVLA_ROOT}:${BRIDGEVLA_ROOT}/finetune"
export PYTHONUNBUFFERED=1
if [[ -n "${GPU_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID}"
fi
# Offline mode is optional: the BridgeVLA loader may need uncached base assets.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-${HF_HUB_OFFLINE}}"

# Help must work without a checkpoint/GPU and must not create output directories.
for argument in "${SERVER_ARGS[@]}"; do
  if [[ "${argument}" == "--help" || "${argument}" == "-h" ]]; then
    exec "${PYTHON_BIN}" -m finetune.OHT.vla_server "${SERVER_ARGS[@]}"
  fi
done

# Read known CLI options for checks/banner. Keep original arguments intact so
# Python retains its own validation, mutual exclusion and passthrough options.
# CLI takes precedence over environment defaults; support --name=value as well.
configure_option() {
  local destination="$1" canonical="$2" value="$3"
  shift 3
  local aliases=("${canonical}" "$@")
  local found=0 index argument option
  for ((index=0; index<${#SERVER_ARGS[@]}; index++)); do
    argument="${SERVER_ARGS[index]}"
    for option in "${aliases[@]}"; do
      if [[ "${argument}" == "${option}" ]]; then
        ((index+1<${#SERVER_ARGS[@]})) || die "Missing value for ${option}"
        value="${SERVER_ARGS[index+1]}"
        [[ -n "${value}" && "${value}" != --* ]] || die "Missing value for ${option}"
        found=1
      elif [[ "${argument}" == "${option}="* ]]; then
        value="${argument#*=}"
        [[ -n "${value}" ]] || die "Missing value for ${option}"
        found=1
      fi
    done
  done
  printf -v "${destination}" '%s' "${value}"
  if [[ "${found}" == 0 && -n "${value}" ]]; then
    SERVER_DEFAULTS+=("${canonical}" "${value}")
  fi
}

configure_option OHT_MODEL_PATH --model-path "${OHT_MODEL_PATH:-}" --model_path --checkpoint
configure_option OHT_HOST --host "${OHT_HOST:-127.0.0.1}"
configure_option OHT_PORT --port "${OHT_PORT:-8010}"
configure_option OHT_DEVICE --device "${OHT_DEVICE:-cuda:0}"
configure_option OHT_SERVER_OUTPUT --output-dir "${OHT_SERVER_OUTPUT:-logs/oht_vla_server}" --output_dir
configure_option OHT_PRETRAIN_PATH --pretrain-path "${OHT_PRETRAIN_PATH:-}"

[[ -n "${OHT_MODEL_PATH}" ]] || die "Set OHT_MODEL_PATH or pass --model-path/--checkpoint"
if [[ -d "${OHT_MODEL_PATH}" ]]; then
  OHT_CHECKPOINT="${OHT_MODEL_PATH%/}/model_last.pth"
else
  OHT_CHECKPOINT="${OHT_MODEL_PATH}"
fi
[[ -f "${OHT_CHECKPOINT}" ]] || die "Checkpoint not found: ${OHT_CHECKPOINT}"
if [[ -n "${OHT_PRETRAIN_PATH}" ]]; then
  [[ -f "${OHT_PRETRAIN_PATH}/model.safetensors.index.json" ]] || \
    die "Missing BridgeVLA pretrain index: ${OHT_PRETRAIN_PATH}/model.safetensors.index.json"
fi
[[ "${OHT_PORT}" =~ ^[0-9]+$ && ${#OHT_PORT} -le 5 ]] || die "Invalid port: ${OHT_PORT}"
((10#${OHT_PORT} >= 1 && 10#${OHT_PORT} <= 65535)) || die "Port must be between 1 and 65535"
mkdir -p "${OHT_SERVER_OUTPUT}"

echo "======================================================================"
echo "BridgeVLA OHT FastAPI Inference Server"
echo "======================================================================"
echo "BRIDGEVLA_ROOT       : ${BRIDGEVLA_ROOT}"
echo "SERVER_MODULE        : finetune.OHT.vla_server"
echo "PYTHON_BIN           : ${PYTHON_BIN}"
echo "CUDA_VISIBLE_DEVICES : ${CUDA_VISIBLE_DEVICES-<all visible>}"
echo "MODEL_PATH           : ${OHT_MODEL_PATH}"
echo "CHECKPOINT           : ${OHT_CHECKPOINT}"
echo "PRETRAIN_PATH        : ${OHT_PRETRAIN_PATH:-<not supplied>}"
echo "DEVICE               : ${OHT_DEVICE}"
echo "HOST                 : ${OHT_HOST}"
echo "PORT                 : ${OHT_PORT}"
echo "OUTPUT_DIR           : ${OHT_SERVER_OUTPUT}"
echo "PYTHONPATH           : ${PYTHONPATH}"
echo "HF_HUB_OFFLINE       : ${HF_HUB_OFFLINE}"
echo "TRANSFORMERS_OFFLINE : ${TRANSFORMERS_OFFLINE}"
echo "======================================================================"

exec "${PYTHON_BIN}" -m finetune.OHT.vla_server "${SERVER_DEFAULTS[@]}" "${SERVER_ARGS[@]}"
