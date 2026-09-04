#!/usr/bin/env bash
set -euo pipefail

# cd finetune
# # export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 
# # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
# # export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
# # export DISPLAY=:1.0
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"


# pip uninstall -y opencv-python opencv-contrib-python
# pip install  opencv-python-headless  
# pip uninstall  -y opencv-python-headless      
# pip install  opencv-python-headless   # in my machine   i have to repeat the installation process to avoid the error: "Could not find the Qt platform plugin 'xcb'"   

# xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24 -ac'  

# 
# The command below works well
# xvfb-run -a \
#   -s "-screen 0 1280x1024x24 +extension GLX +render -noreset" \
#   bash eval.sh


# xvfb-run -a -s "-screen 0 1280x1024x24 +extension GLX +render -noreset" 


export TRANSFORMERS_VERBOSITY=error
export PYTHONWARNINGS="ignore"
export DISPLAY="${DISPLAY:-109.105.4.172:0.0}"
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __GL_PROVIDER_VERSION=GLX
export LIBGL_ALWAYS_SOFTWARE=0
export LIBGL_ALWAYS_INDIRECT=0

MODEL_FOLDER="${MODEL_FOLDER:-/common-data-32t/usr/yiwei/hugging_download/data_tmp/LPY/BridgeVLA/checkpoints/RLBench}"
MODEL_NAME="${MODEL_NAME:-model_60.pth}"
EVAL_DATAFOLDER="${EVAL_DATAFOLDER:-/common-data-32t/usr/yiwei/hugging_download/data_tmp/LPY/BridgeVLA_RLBench_EVAL_DATA}"
EVAL_EPISODES="${EVAL_EPISODES:-25}"
EPISODE_LENGTH="${EPISODE_LENGTH:-25}"
DEVICE="${DEVICE:-0}"
ORACLE_PROVIDER="${ORACLE_PROVIDER:-none}"
ORACLE_STRICT="${ORACLE_STRICT:-0}"
ORACLE_DEBUG="${ORACLE_DEBUG:-0}"
ORACLE_NUM_POINTS="${ORACLE_NUM_POINTS:-512}"
ORACLE_ROLE_CONFIG="${ORACLE_ROLE_CONFIG:-${SCRIPT_DIR}/configs/rlbench_o2_semantic_roles.yaml}"
EXP_CFG_PATH="${EXP_CFG_PATH:-}"
REPLAY_GROUND_TRUTH="${REPLAY_GROUND_TRUTH:-0}"
GT_REPLAY_RETRIES="${GT_REPLAY_RETRIES:-3}"
SAVE_VIDEO="${SAVE_VIDEO:-1}"
VISUALIZE="${VISUALIZE:-0}"
VISUALIZE_ROOT_DIR="${VISUALIZE_ROOT_DIR:-exp/RLBench_vis}"

oracle_args=(
  --oracle-provider "${ORACLE_PROVIDER}"
  --oracle-role-config "${ORACLE_ROLE_CONFIG}"
  --oracle-num-points "${ORACLE_NUM_POINTS}"
)
[[ "${ORACLE_STRICT}" == "1" ]] && oracle_args+=(--oracle-strict)
[[ "${ORACLE_DEBUG}" == "1" ]] && oracle_args+=(--oracle-debug)
exp_cfg_args=()
[[ -n "${EXP_CFG_PATH}" ]] && exp_cfg_args+=(--exp_cfg_path "${EXP_CFG_PATH}")
ground_truth_args=()
[[ "${REPLAY_GROUND_TRUTH}" == "1" ]] && ground_truth_args+=(
  --ground-truth
  --ground-truth-retries "${GT_REPLAY_RETRIES}"
)
video_args=()
[[ "${SAVE_VIDEO}" == "1" ]] && video_args+=(--save-video)
visualize_args=(--visualize_root_dir "${VISUALIZE_ROOT_DIR}")
[[ "${VISUALIZE}" == "1" ]] && visualize_args+=(--visualize)

tasks=(
    # "close_jar"
    # "insert_onto_square_peg"
    # "light_bulb_in"
    # "meat_off_grill"
    # "open_drawer"
    "place_cups"
    # "place_shape_in_shape_sorter"
    # "push_buttons"
    # "put_groceries_in_cupboard"
    # "put_item_in_drawer"
    # "put_money_in_safe"
    # "reach_and_drag"
    # "stack_blocks"
    # "stack_cups"
    # "turn_tap"
    # "place_wine_at_rack_location"
    # "slide_block_to_color_target"
    # "sweep_to_dustpan_of_size"
)
if [[ -n "${TASKS:-}" ]]; then
  read -r -a tasks <<< "${TASKS}"
fi

for task in "${tasks[@]}"; do
  echo "=========================================="
  echo "Processing task: $task"
  echo "=========================================="
      
  python3 eval.py \
    --model-folder "${MODEL_FOLDER}" \
    --eval-datafolder "${EVAL_DATAFOLDER}" \
    --tasks "${task}" \
    --eval-episodes "${EVAL_EPISODES}" \
    --episode-length "${EPISODE_LENGTH}" \
    --log-name "${task}/${ORACLE_PROVIDER}" \
    --device "${DEVICE}" \
    --headless \
    --model-name "${MODEL_NAME}" \
    "${oracle_args[@]}" \
    "${exp_cfg_args[@]}" \
    "${ground_truth_args[@]}" \
    "${video_args[@]}" \
    "${visualize_args[@]}"
  # --visualize_root_dir "exp/RLBench_vis" --save-video --visualize
      
  echo "Completed task: $task"
  echo ""
done

echo "=========================================="
echo "All tasks completed!"
echo "=========================================="


eval_root="${MODEL_FOLDER}/eval"
model_dir="${MODEL_NAME%.pth}"
output_csv="${eval_root}/${model_dir}_${ORACLE_PROVIDER}_merged_eval_results.csv"

# 写入统一表头
echo "task,success rate,length,total_transitions" > "${output_csv}"

for task in "${tasks[@]}"; do
    csv_path="${eval_root}/${task}/${ORACLE_PROVIDER}/${model_dir}/eval_results.csv"

    if [[ ! -f "${csv_path}" ]]; then
        echo "[WARN] File not found: ${csv_path}" >&2
        continue
    fi

    # 跳过每个文件的表头，追加数据行
    tail -n +2 "${csv_path}" >> "${output_csv}"
done

echo "Merged CSV saved to: ${output_csv}"
