cd /PATH_TO_BRIDGEVLA/finetune
export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY=:0.0
# sudo apt-get install xvfb

pip uninstall -y  opencv-python   
pip uninstall -y opencv-contrib-python
pip install  opencv-python-headless
export DISPLAY=:0.0
DEMO_PATH_ROOT=PATH_TO_COLOSSEUM_EVAL_DATA
export eval_root_dir=opensource_colosseum
export eval_episodes=25
export variation=$1  
export trial=$2  
export epoch=$3  
export model_folder=$4  # PATH_TO_MODEL_FOLDER




cd Colosseum
for root_task in \
    "basketball_in_hoop" \
    "close_box" \
    "empty_dishwasher" \
    "get_ice_from_fridge" \
    "hockey" \
    "meat_on_grill" \
    "move_hanger" \
    "wipe_desk" \
    "open_drawer" \
    "slide_block_to_target" \
    "reach_and_drag" \
    "put_money_in_safe" \
    "place_wine_at_rack_location" \
    "insert_onto_square_peg" \
    "turn_oven_on" \
    "straighten_rope" \
    "setup_chess" \
    "scoop_with_spatula" \
    "close_laptop_lid" \
    "stack_cups"; do
    echo $root_task
    export DEMO_PATH=$DEMO_PATH_ROOT/$root_task
    echo $DEMO_PATH
    export task_list=${root_task}_${variation}
    export eval_dir=$eval_root_dir/epoch_${epoch}_trial_${trial}/$variation/$root_task/  

    export DATA_PATH=$DEMO_PATH/$task_list/
    if [ ! -e "$DATA_PATH" ]; then
        echo "$DATA_PATH does not exist, skipping this iteration."
        continue
    fi    

    echo $eval_dir
    xvfb-run --auto-servernum --server-args='-screen 0 1024x768x24 -ac' python3 eval.py \
        --model-folder $model_folder  \
        --eval-datafolder $DEMO_PATH \
        --tasks $task_list \
        --eval-episodes $eval_episodes \
        --log-name $eval_dir \
        --device 0 \
        --headless \
        --model-name model_$epoch.pth \
        --save-video
done