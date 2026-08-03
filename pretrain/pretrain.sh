cd  pretrain
port=MASTER_PORT
GPUS_PER_NODE=NUMBER_OF_GPU_PER_MACHINE
NNODES=NUMBER_OF_MACHINE
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_port=$port \
    pretrain.py "$@"