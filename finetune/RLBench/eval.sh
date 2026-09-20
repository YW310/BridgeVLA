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
START_EPISODE="${START_EPISODE:-0}"
EPISODE_LENGTH="${EPISODE_LENGTH:-25}"
DEVICE="${DEVICE:-0}"
ORACLE_PROVIDER="${ORACLE_PROVIDER:-none}"
ORACLE_STRICT="${ORACLE_STRICT:-0}"
ORACLE_DEBUG="${ORACLE_DEBUG:-0}"
ORACLE_DEBUG_INTERVAL="${ORACLE_DEBUG_INTERVAL:-1}"
HEATMAP_ACTION_ANCHOR="${HEATMAP_ACTION_ANCHOR:-${HEATMAP_TARGET_OBJECT:-0}}"
BRIDGEVLA_ALIGNED_OBJECTS="${BRIDGEVLA_ALIGNED_OBJECTS:-0}"
ORACLE_NUM_POINTS="${ORACLE_NUM_POINTS:-512}"
ORACLE_HANDLE_ALIGNMENT="${ORACLE_HANDLE_ALIGNMENT:-verified}"
ORACLE_HANDLE_MAP_DIR="${ORACLE_HANDLE_MAP_DIR:-}"
ORACLE_ROLE_CONFIG="${ORACLE_ROLE_CONFIG:-${SCRIPT_DIR}/configs/rlbench_o2_semantic_roles.yaml}"
EXP_CFG_PATH="${EXP_CFG_PATH:-}"
REPLAY_GROUND_TRUTH="${REPLAY_GROUND_TRUTH:-0}"
GT_REPLAY_RETRIES="${GT_REPLAY_RETRIES:-3}"
MANIFEST_PHASE_SOURCE="${MANIFEST_PHASE_SOURCE:-sim_replay}"
MANIFEST_RESUME="${MANIFEST_RESUME:-0}"
EVAL_RESUME="${EVAL_RESUME:-${MANIFEST_RESUME}}"
MANIFEST_CONTINUE_ON_ERROR="${MANIFEST_CONTINUE_ON_ERROR:-0}"
if [[ -z "${SAVE_VIDEO+x}" ]]; then
  if [[ "${EVAL_RESUME}" == "1" ]]; then
    SAVE_VIDEO=0
  else
    SAVE_VIDEO=1
  fi
fi
VISUALIZE="${VISUALIZE:-0}"
VISUALIZE_ROOT_DIR="${VISUALIZE_ROOT_DIR:-exp/RLBench_vis}"

oracle_args=(
  --oracle-provider "${ORACLE_PROVIDER}"
  --oracle-role-config "${ORACLE_ROLE_CONFIG}"
  --oracle-num-points "${ORACLE_NUM_POINTS}"
  --oracle-handle-alignment "${ORACLE_HANDLE_ALIGNMENT}"
)
[[ -n "${ORACLE_HANDLE_MAP_DIR}" ]] && oracle_args+=(--oracle-handle-map-dir "${ORACLE_HANDLE_MAP_DIR}")
[[ "${ORACLE_STRICT}" == "1" ]] && oracle_args+=(--oracle-strict)
[[ "${ORACLE_DEBUG}" == "1" ]] && oracle_args+=(
  --oracle-debug
  --oracle-debug-interval "${ORACLE_DEBUG_INTERVAL}"
)
[[ "${HEATMAP_ACTION_ANCHOR}" == "1" ]] && oracle_args+=(--heatmap-action-anchor)
[[ "${BRIDGEVLA_ALIGNED_OBJECTS}" == "1" ]] && oracle_args+=(--bridgevla-aligned-objects)
exp_cfg_args=()
[[ -n "${EXP_CFG_PATH}" ]] && exp_cfg_args+=(--exp_cfg_path "${EXP_CFG_PATH}")
ground_truth_args=()
[[ "${REPLAY_GROUND_TRUTH}" == "1" ]] && ground_truth_args+=(
  --ground-truth
  --ground-truth-retries "${GT_REPLAY_RETRIES}"
  --manifest-phase-source "${MANIFEST_PHASE_SOURCE}"
)
resume_args=()
[[ "${EVAL_RESUME}" == "1" ]] && resume_args+=(--eval-resume)
[[ "${MANIFEST_CONTINUE_ON_ERROR}" == "1" ]] && resume_args+=(--manifest-continue-on-error)
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
if [[ "${EVAL_RESUME}" == "1" && "${#tasks[@]}" -eq 1 && "${tasks[0]}" == "all" ]]; then
  # Resume must isolate task state. A single multi-task process cannot skip all
  # resets for one task and still advance RLBench safely to the next task.
  tasks=(
    close_jar
    reach_and_drag
    insert_onto_square_peg
    meat_off_grill
    open_drawer
    place_cups
    place_wine_at_rack_location
    push_buttons
    put_groceries_in_cupboard
    put_item_in_drawer
    put_money_in_safe
    light_bulb_in
    slide_block_to_color_target
    place_shape_in_shape_sorter
    stack_blocks
    stack_cups
    sweep_to_dustpan_of_size
    turn_tap
  )
  echo "[Evaluation][RESUME] TASKS=all expanded into 18 isolated task processes."
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
    --start-episode "${START_EPISODE}" \
    --episode-length "${EPISODE_LENGTH}" \
    --log-name "${task}/${ORACLE_PROVIDER}" \
    --device "${DEVICE}" \
    --headless \
    --model-name "${MODEL_NAME}" \
    "${oracle_args[@]}" \
    "${exp_cfg_args[@]}" \
    "${ground_truth_args[@]}" \
    "${resume_args[@]}" \
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
result_filename="eval_results.csv"
result_header="task,success rate,length,total_transitions"
if [[ "${MANIFEST_PHASE_SOURCE}" == "demo_events" ]]; then
    output_csv="${eval_root}/${model_dir}_${ORACLE_PROVIDER}_merged_manifest_results.csv"
    result_filename="manifest_results.csv"
    result_header="task,generated coverage,generated episodes,requested episodes,logical transitions"
fi

# 写入统一表头
echo "${result_header}" > "${output_csv}"

for task in "${tasks[@]}"; do
    csv_path="${eval_root}/${task}/${ORACLE_PROVIDER}/${model_dir}/${result_filename}"

    if [[ ! -f "${csv_path}" ]]; then
        echo "[WARN] File not found: ${csv_path}" >&2
        continue
    fi

    # 跳过每个文件的表头，追加数据行
    tail -n +2 "${csv_path}" >> "${output_csv}"
done

echo "Merged CSV saved to: ${output_csv}"
