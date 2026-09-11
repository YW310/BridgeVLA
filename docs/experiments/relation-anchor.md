# O2 implicit relation anchor

[文档索引](../README.md) · [O2 训练](o2-training.md) · [研究设计](../design/role-relation-prior.md)

该实验在现有 Target/Reference relation adapter 之后增加一个仅作用于
translation 的轻量模块。它不使用 phase 标签、contact 标签或手工
action-anchor。

    PaliGemma feature + T/R prior + T/R 3-D geometry
                             |
                             +-- existing relation adapter -- R/G/C
                             |
                             +-- implicit relation anchor --- translation head

Relation query 由 T/R 区域内的 masked-pooled feature、T/R 中心与尺度、
相对位移以及已有的四维 low-dimensional state 组成。Reference 无效时使用
learned NULL token；Target 无效时 residual 关闭。输出投影为零初始化，因此
新模块刚启用时保持旧模型的 translation 输出。

Anchor 只在 feature 层进行 residual modulation。它不会与最终 translation
logit 相加，也不会改变 rotation/gripper/collision 的 feature 路径。训练直接
沿用现有 translation loss，不需要新增 replay 字段。

## 配置

    bash train.sh \
        --exp_cfg_path configs/rlbench_o2_semantic_gt_relation_anchor.yaml \
        --train_replay_storage_dir /path/to/semantic_gt_buffer \
        --init_checkpoint /path/to/current_o2/model_last.pth \
        --train_oracle_adapter_only

关键开关：

    oracle_prior_adapter_rank: 16
    oracle_relation_gated_adapter: True
    oracle_adapter_translation_only: False
    oracle_relation_anchor_rank: 16

将 oracle_relation_anchor_rank 设为 0 可完全关闭新模块。训练与闭环可视化中
会输出 relation_anchor / o2_relation_anchor，用于检查 anchor 是否从 object
occupancy 收缩到 phase-dependent action region。验收时应同时比较 decoded
argmax、1/2/5 voxel recall、三维 waypoint error 和 closed-loop success；
不能只依据 translation cross-entropy。
