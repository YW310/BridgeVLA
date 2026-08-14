#!/bin/bash

# 源目录和目标目录
SRC_DIR="/home/yiwei/project/BridgeVLA/LPY/BridgeVLA_RLBench_TRAIN_DATA/train"
DEST_DIR="/home/yiwei/project/BridgeVLA/LPY/BridgeVLA_RLBench_TINY_DATA/train"

# 创建目标目录（如果不存在）
mkdir -p "$DEST_DIR"

# 遍历所有 task 文件夹
for task_dir in "$SRC_DIR"/*/; do
    # 获取 task 名称
    task_name=$(basename "$task_dir")
    
    # 创建目标 task 文件夹
    mkdir -p "$DEST_DIR/$task_name"
    mkdir -p "$DEST_DIR/$task_name/all_variations"
    mkdir -p "$DEST_DIR/$task_name/all_variations/episodes"
    # 复制前 3 个 episode (episode0, episode1, episode2)
    for i in 0 1 2; do
        episode_dir="all_variations/episodes/episode$i"
        if [ -d "$task_dir$episode_dir" ]; then
            ln -s "$SRC_DIR/$task_name/$episode_dir" "$DEST_DIR/$task_name/all_variations/episodes/"
            echo "ln -s $SRC_DIR/$task_name/$episode_dir" "$DEST_DIR/$task_name/all_variations/episodes/"
        fi
    done
done

echo "Done! All tasks' first 3 episodes copied to $DEST_DIR"


#!/bin/bash

# 源目录和目标目录
SRC_DIR="/home/yiwei/project/BridgeVLA/LPY/BridgeVLA_RLBench_TRAIN_DATA/train"
DEST_DIR="/home/yiwei/project/BridgeVLA/LPY/BridgeVLA_RLBench_TINY_DATA/train"

# 创建目标目录（如果不存在）
mkdir -p "$DEST_DIR"

# 遍历所有 task 文件夹
for task_dir in "$SRC_DIR"/*/; do
    # 获取 task 名称
    task_name=$(basename "$task_dir")
    
    # 创建目标 task 文件夹
    mkdir -p "$DEST_DIR/$task_name"
    mkdir -p "$DEST_DIR/$task_name/all_variations"
    mkdir -p "$DEST_DIR/$task_name/all_variations/episodes"
    # 复制前 3 个 episode (episode0, episode1, episode2)
    for i in 0 1 2; do
        episode_dir="all_variations/episodes/episode$i"
        if [ -d "$task_dir$episode_dir" ]; then
            ln -s "$SRC_DIR/$task_name/$episode_dir" "$DEST_DIR/$task_name/all_variations/episodes/"
            echo "ln -s $SRC_DIR/$task_name/$episode_dir" "$DEST_DIR/$task_name/all_variations/episodes/"
        fi
    done
done

echo "Done! All tasks' first 3 episodes copied to $DEST_DIR"

