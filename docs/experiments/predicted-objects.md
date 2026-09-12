# O2 predicted objects

[文档索引](../README.md) · [Relation anchor](relation-anchor.md)

该实验不向策略输入 Oracle object。外部 object predictor 先选择一个
Target/Reference pair，并为每个角色写入固定大小点集、存在性、有效性和置信度；
BridgeVLA 继续复用 relation adapter 与 relation anchor。

配置：

    configs/rlbench_o2_predicted_objects.yaml

训练示例：

    bash train.sh \
        --exp_cfg_path configs/rlbench_o2_predicted_objects.yaml \
        --train_replay_storage_dir /path/to/predicted_object_buffer \
        --init_checkpoint /path/to/current_o2/model_last.pth \
        --train_object_adapter_only

每个 replay transition 必须提供：

    predicted_target_object_points       float32 [512, 3]
    predicted_target_object_valid        bool
    predicted_target_present             bool
    predicted_target_confidence          float32
    predicted_reference_object_points    float32 [512, 3]
    predicted_reference_object_valid     bool
    predicted_reference_present          bool
    predicted_reference_confidence       float32

`present=False` 表示当前 relation 不需要该角色；`present=True, valid=False`
表示角色存在但预测器没有提供可靠几何。后一种情况不会被当作 NULL Reference，
而会关闭该样本的 object residual。confidence 低于配置阈值同样视为无效。

当前边界是“外部预测器 + BridgeVLA policy”，仓库尚未包含 detector/segmentor
本身。闭环评估时，预测器 wrapper 必须在 `agent.act()` 前向 observation 注入相同
字段；严格模式下缺失字段会立即报错，不会回退到 Oracle。
