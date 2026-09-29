# 项目审查与代码索引

[文档索引](../README.md) · [配置流程](../guides/object-conditioning.md#配置选择) · [接口契约](../design/role-relation-prior.md#当前实现与数据契约) · [联合实验](../guides/object-conditioning.md#gt-联合对照)

> 本页定位现有代码。角色监督/继承/token 保留已有 opt-in 实现；语义 queries 与 temporal memory 仍未实现。

## Semantic-GT 数据流

```text
role YAML + stored demo → Oracle provider → phase/handle manifest
  → replay 重写：T/R XYZ + audit → schema/loader
  → 读取时派生 presence/known → 预测模式仅用于辅助监督
```

| 阶段 | 文件 | 关键函数 |
| --- | --- | --- |
| 角色/manifest | [o2_oracle_provider.py](../../finetune/RLBench/utils/o2_oracle_provider.py) | `_build_assignment()`、`build_demo_event_manifest()`、`_sample_entity_points()` |
| handle 证书 | [oracle_handle_alignment.py](../../finetune/RLBench/utils/oracle_handle_alignment.py) | `align_handles()`、`align_semantic_handle_group()` |
| site 几何 | [site_geometry.py](../../finetune/RLBench/utils/site_geometry.py) | `site_geometry_from_object()`、`sample_site_geometry()` |
| 重写/审计 | [rewrite_replay_with_semantic_roles.py](../../tools/rewrite_replay_with_semantic_roles.py) | `_load_manifest()`、`_build_oracle()`、`_fill_slot()`、`_audit_fields()` |
| 校验/可视化 | 同上 | `_validate_semantic_transition()`、`_validate_task_output()`、`_visualize_task_output()` |
| schema/loader | [dataset.py](../../finetune/RLBench/utils/dataset.py) / [get_dataset.py](../../finetune/RLBench/utils/get_dataset.py) | `create_replay()`、`get_dataset()` |
| 旧数据 presence | [YARR uniform_replay_buffer.py](../../finetune/bridgevla/libs/YARR/yarr/replay_buffer/uniform_replay_buffer.py) | `_derive_role_presence()`、`_copy_required_disk_fields()`，只读派生 |
| train/eval 契约 | [semantic_contract.py](../../finetune/RLBench/utils/semantic_contract.py) / [train.py](../../finetune/RLBench/train.py) / [eval.py](../../finetune/RLBench/eval.py) | `_validate_semantic_replay_schema()`、`load_agent()` |

## Policy 数据流

```text
RVTAgent.update/act
  → mvt.MVT.forward：模式解析、prior/crop 与输入隔离
  → mvt_single.MVT.forward：同次 VLM features / text context
  → slots（预测模式）→ 原 relation/anchor adapter
  → action_feature_routes → translation / R/G/C
```

| 文件 | 当前职责与函数 |
| --- | --- |
| [oracle_prior.py](../../finetune/bridgevla/models/oracle_prior.py) | `resolve_object_prior_mode()`；`rasterize_instance_points()`；`InternalObjectSlotPredictor.forward()` / `_extract_points()`；`OracleRelationAnchorFeatureAdapter.forward_with_anchor()` |
| [bridgevla_agent.py](../../finetune/bridgevla/models/bridgevla_agent.py) | `_select_oracle_prior_points()`、`_oracle_network_kwargs()`；teacher-only `_object_slot_auxiliary_losses()` |
| [mvt.py](../../finetune/bridgevla/mvt/mvt.py) | `MVT.forward()`、`_build_oracle_instance_prior()`；管理 coarse/refine 投影与 crop |
| [mvt_single.py](../../finetune/bridgevla/mvt/mvt_single.py) | `MVT.forward()`；VLM/decoder、预测策略的 GT 隔离、共享模式 global 重池化 |
| [object_conditioning.py](../../finetune/bridgevla/models/object_conditioning.py) | `pool_instruction_context()`、`soft_role_geometry()`、`action_feature_routes()` |
| 同上 | `role_supervision_mask()` 统一 stage/role/view 支持；`mixed_role_map_losses()`、`hungarian_role_slot_losses()`、`reference_null_loss()` |
| [cross_scale_roles.py](../../finetune/bridgevla/models/cross_scale_roles.py) | `inherit_coarse_roles()`、`transform_role_geometry()`；本步角色传递与局部提示支持 |
| [train.py](../../finetune/RLBench/train.py) | `load_initial_model_checkpoint()`、`load_training_checkpoint()` |
| [training_utils.py](../../finetune/RLBench/training_utils.py) | `resolve_training_epochs()`、`build_batch_plan()`、`optimizer_steps_per_epoch()`；显式 CLI 优先，实际预算存入训练配置 |
| [compare_paired_success.py](../../tools/compare_paired_success.py) | `compare()`，同 episodes 配对闭环 CI |

## 配置入口

均位于 [finetune/RLBench/configs](../../finetune/RLBench/configs)。

| 配置 | 当前行为 |
| --- | --- |
| `rlbench_o2_semantic_gt.yaml` | Oracle relation adapter |
| `rlbench_o2_semantic_gt_relation_anchor.yaml` | 原 translation-anchor 路由 |
| `rlbench_o2_predicted_objects.yaml` | 外部预测 T/R |
| `rlbench_o2_internal_slots.yaml` | 2 个无序 slots，Hungarian warm-up；NULL loss weight = 0.25，diversity 关闭 |
| `rlbench_o2_semantic_gt_joint.yaml` | opt-in shared action + instruction，先做 GT 对照 |
| `rlbench_o2_internal_slots_joint.yaml` | 6 个无序 slots，soft roles/geometry + joint training；GT 准入后实验 |
| `rlbench_o2_internal_slots_cross_scale.yaml` | 同 joint 预算，显式启用 mixed supervision / inheritance / token preservation |

默认旧配置不切换新路由；数值与闭环收益须在目标环境验收。
`current_state[B,3]` 是当前夹爪状态，旧 relation-state 名称兼容；
预测部署不得读取 simulator ID、GT phase 或未来目标动作，Oracle 实验另作上界。

## 计划代码落点

| 尚未实现 | 设计入口 |
| --- | --- |
| 两个语义 queries、可训练 refine mask readout | [角色预测](../design/role-relation-prior.md#最小角色预测计划未实现)；已实现开关见[操作指南](../guides/object-conditioning.md#角色一致性开关) |
| 跨步 object bank、序列 loader、act/reset 状态 | [Memory](../design/role-relation-prior.md#object-centric-memory后续未实现) |
| visibility / 当前 EE pose / phase graph / object-local views / risk head | 不进入当前主线，按标签与瓶颈另验 |

predictor 内的 attention `memory` 不是历史 bank。`RVTAgent.reset()` 会清理 simulator alignment 的锁与计数，
但当前没有学习式 role memory/object bank。

## 项目审查（2026-09-29）

本轮按运行链路审查 first-party 入口和关键实现：pretrain、RLBench 数据/模型/训练/评估、
Semantic-GT 工具、Colosseum/GemBench launcher 与统计、相关 tests 和 docs。
不是全部文件的逐行审计；vendor 库仅检查关联接口，也未运行 GPU/模拟器。初次审查只改文档；后续按授权实现了三个角色一致性开关，以下运行问题暂不修复。

### 已确认、尚未修复的运行问题

| 优先级 | 问题与影响 | 代码证据 | 当前处理 |
| --- | --- | --- | --- |
| P1 | 旧 launcher 硬编码路径、未引用 `$@`，会拆开 overrides/含空格路径 | [RLBench train.sh](../../finetune/RLBench/train.sh)；Colosseum/GemBench 同类脚本 | RLBench 示例改用 `train_8x40.sh`；其他分支先适配脚本 |
| P1 | eval resume 没有覆盖全部策略依赖，模型模块改变后可能复用旧结果 | [eval.py](../../finetune/RLBench/eval.py) 的 `runtime_files` 未含 `mvt_single.py` / `object_conditioning.py` / `oracle_prior.py` | 代码/数据改变后使用新评估目录，不复用旧 journals |
| P1 | 配对 CI 只要求双方已有 episode 集合相同，双方同样中断仍可能通过 | [compare_paired_success.py](../../tools/compare_paired_success.py) 的 `load_episodes()` / `compare()` 未校验 requested summary | 先核对每 task 完成数与预定 episode 范围，再运行 CI |
| P2 | installer 与其他 benchmark 流程不完全可移植；统计依赖目录/顺序假设 | [pretrain installer](../../pretrain/pretrain_install.sh)、[GemBench client](../../finetune/GemBench/run_client.sh)、[GemBench aggregation](../../finetune/GemBench/cal_results.py)、[Colosseum aggregation](../../finetune/Colosseum/cal_statics.py) | 预先配置依赖/路径，核对任务与 episode 覆盖；不能直接套用 RLBench 统计保障 |

P1 指会破坏实验可运行性或结果可信度，P2 指复现/配置风险；以上没有宣称已修复。
resume 签名对 checkpoint 使用 path/size/mtime，对 raw data 使用目录路径，不是全内容 provenance。

### 模型与数据审查结论

| 已实现 | 当前边界 |
| --- | --- |
| opt-in shared action routes、同次 text pooling、预测 teacher 隔离 | 旧 YAML 不自动启用；训练仍用 GT waypoint/crop，推理用预测 waypoint/crop |
| soft T/R posterior、可微几何、Reference NULL loss | 默认仍按 geometry valid 屏蔽 token；opt-in 保留可靠角色；没有 Target NULL head |
| 默认独立 slots；opt-in 混合 map 监督/本步继承 | 继承时只有 coarse 学习角色，refine 提示不另算辅助 loss；无完整 distractor 负例 |
| kind 只读派生 presence、全量语义校验 | role YAML 是全文件 hash；resume 不迁移旧 replay；mask-only/局部身份证书不认证整场景 XYZ |
| RLBench 稀疏 reward 统计、完成数校验、episode/video 同编号 | 配对 CI 只接收 `100/0` journals；历史表格不能当当前 O2 验收结果 |

当前不具备学习式时空 memory、物理遮挡补全或已验证的失败恢复/真机能力。
详细接口只维护在[主设计](../design/role-relation-prior.md)，操作与临时规避只维护在[O2 指南](../guides/object-conditioning.md)。

### 验证与下一步

本机完成 16 个 first-party Bash 脚本的 `bash -n`，以及未引用/已引用 `$@` 的独立参数复现。
实施后复核 13 份 Markdown：185 个本地链接、77 个锚点引用、46 个 Bash 示例通过静态校验；
语法通过不代表 placeholder、依赖或模型数值正确。
隔离环境（Python 3.12 / PyTorch 2.5.1 CPU）已通过 132 项相关回归，覆盖角色 loss/梯度、跨尺度继承、小模型动作前向、可视化、配置与训练预算；20 个新增/修改的 Python 文件通过 AST/编译检查。
同时修复了 RLBench eval 的 `--use-input-place-with-mean` 未初始化变量，以及 YAML epochs 被 CLI 默认 100 覆盖的问题。完整 PaliGemma/CUDA 单步训练、真实 renderer 与模拟器闭环尚未运行。

目标 Linux/CUDA 环境按以下顺序验收：

1. `python -m pytest -q tests`；重点检查 [动作/teacher 隔离](../../tests/test_object_conditioning_forward.py)、[语义契约](../../tests/test_semantic_contract.py)、[评估报告](../../tests/test_eval_reporting.py)、[配对 CI](../../tests/test_paired_success.py)。现有部分测试是源码检查，不替代数值测试。
2. 修运行问题时补 launcher 参数保真、依赖模块变更使 resume 失效、双方同样缺 episode 时 CI 拒绝的回归测试。
3. 分别做 GT/预测单步训练与单 episode smoke，检查梯度、coarse/refine 坐标、最终 waypoint 下 R/G/C；预测动作前向不读 GT。
4. 固定版本/预算/完整 episodes，3 seeds 配对闭环；GT 准入后验证三个 opt-in 开关，不先追加 memory。
