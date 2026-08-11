cd /home/yiwei/project/BridgeVLA/finetune
export COPPELIASIM_ROOT=$(pwd)/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT
export DISPLAY="109.105.4.172:0.0"
export PYTHONWARNINGS="ignore"

cd /home/yiwei/project/BridgeVLA/finetune/RLBench

port=29500
GPUS_PER_NODE=2
NNODES=1
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$port \
   train.py \
   $@ 
