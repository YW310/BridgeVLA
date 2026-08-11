# cd finetune
# # export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 
# # export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
# # export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
# # export DISPLAY=:1.0
# cd RLBench


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
export DISPLAY="109.105.4.172:0.0"
export __GLX_VENDOR_LIBRARY_NAME=nvidia
export __GL_PROVIDER_VERSION=GLX
export LIBGL_ALWAYS_SOFTWARE=0
export LIBGL_ALWAYS_INDIRECT=0

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

for task in "${tasks[@]}"; do
  echo "=========================================="
  echo "Processing task: $task"
  echo "=========================================="
      
  python3 eval.py --model-folder /common-data-32t/usr/yiwei/hugging_download/data_tmp/LPY/BridgeVLA/checkpoints/RLBench --eval-datafolder   /common-data-32t/usr/yiwei/hugging_download/data_tmp/LPY/BridgeVLA_RLBench_EVAL_DATA --tasks ${task} --eval-episodes 25 --log-name "${task}" --device 0 --headless --model-name "model_60.pth" --visualize_root_dir "exp/RLBench_vis_${task}" --save-video
  # --visualize_root_dir "exp/RLBench_vis" --save-video --visualize
      
  echo "Completed task: $task"
  echo ""
done

echo "=========================================="
echo "All tasks completed!"
echo "=========================================="


eval_root="/home/yiwei/project/BridgeVLA/exp/RLBench/eval"
model_dir="model_80"
output_csv="${eval_root}/${model_dir}_merged_eval_results.csv"

# 写入统一表头
echo "task,success rate,length,total_transitions" > "${output_csv}"

for task in "${tasks[@]}"; do
    csv_path="${eval_root}/${task}/${model_dir}/eval_results.csv"

    if [[ ! -f "${csv_path}" ]]; then
        echo "[WARN] File not found: ${csv_path}" >&2
        continue
    fi

    # 跳过每个文件的表头，追加数据行
    tail -n +2 "${csv_path}" >> "${output_csv}"
done

echo "Merged CSV saved to: ${output_csv}"
