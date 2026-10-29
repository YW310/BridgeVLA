# OHT

当前流程：**V4 数据 → 关键目标 + 连续 EEF chunk → 离线评估 → IsaacLab**。

OHT 说明统一维护在以下三份文档：

| 要做什么 | 文档 |
| --- | --- |
| 准备 V4/V3、角色标注、检查深度/相机/点云 | [数据](../../docs/oht/data.md) |
| 选择配置、理解模型、训练/续训/离线评估 | [训练与模型](../../docs/oht/training.md) |
| 启动服务/仿真、恢复初态、执行与 GT 回放 | [部署与回放](../../docs/oht/runtime.md) |

已确认有效的轨迹基线为 [oht_chunk_v4_regression_state_8x40.yaml](configs/oht_chunk_v4_regression_state_8x40.yaml)；运行确认、核心修改与排查经验并入[训练文档](../../docs/oht/training.md#已确认基线与交接)。`delta_state` / `flow_state` 仍为对照实验。

默认 K=16、执行前缀 H=4。当前 V4 chunk 使用 `mode: baseline` / `action_mode: eef`；导出 T/R 分割标注不会自动启用物体辅助。训练预算、冻结范围和数据复用条件均以对应文档与解析配置为准。

[完整文档索引](../../docs/README.md) · [环境依赖](../../docs/guides/installation.md#oht-数据与模型环境)
