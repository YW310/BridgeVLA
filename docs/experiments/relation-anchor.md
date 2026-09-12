# O2 implicit relation anchor

[文档索引](../README.md) · [O2 训练](o2-training.md) · [研究设计](../design/role-relation-prior.md)

该实验直接增强现有 `OracleRelationGatedFeatureAdapter`，不再串联第二个
adapter。它不使用 phase、contact 或手工 action-anchor 标签。

    PaliGemma feature + T/R prior + T/R 3-D geometry
                             │
                    relation-conditioned hidden
                             ├── 原 relation residual ── shared feature ── R/G/C
                             └── masked pooling + spatial anchor
                                                  └──── translation residual

Relation query 复用原 adapter 的 relation-conditioned hidden，并结合 T/R
区域 masked pooling、中心/尺度、相对位移和当前夹爪的三维观测状态。归一化
timestep 不进入 query，避免把演示进度当作 phase 捷径。Reference 无效时使用
learned NULL token；Target 无效时 translation residual 关闭。

原 adapter 的参数名和 shared feature 路径保持不变；新增 `anchor_*` 参数也位于
`oracle_prior_feature_adapter1/2` 内。Anchor 只在 feature 层调制 translation，
不会 post-hoc 修改 heatmap。新增输出投影零初始化，启用时先严格退化为原
relation adapter。训练沿用现有 translation loss，不新增 replay 字段。

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

`oracle_relation_anchor_rank: 0` 使用原 `OracleRelationGatedFeatureAdapter`；
大于 0 时在同一个 adapter 内启用 anchor 分支。训练与闭环可视化会输出
`relation_anchor` / `o2_relation_anchor`。验收应同时比较 decoded argmax、
1/2/5 voxel recall、三维 waypoint error 和 closed-loop success，不能只看
translation cross-entropy。

当前 `reference_valid=False` 仍无法区分“语义上没有 Reference”和“Reference
存在但被遮挡”。因此 NULL 分支只适用于前一种情况；接入预测 object 时应把
`reference_present` 与 `reference_visible` 分开，遮挡不能伪装成 NULL。
