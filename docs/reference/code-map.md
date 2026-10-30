# 项目审查与代码索引

[文档索引](../README.md) · [配置流程](../guides/object-conditioning.md#配置选择) · [接口契约](../design/role-relation-prior.md#当前实现与数据契约) · [联合实验](../guides/object-conditioning.md#gt-联合对照)

> 2026-10-29 按代码入口核对。本页维护函数索引、配置入口与当前运行限制；操作命令见各指南。对象条件策略、OHT 单关键点与 V4 chunk 分别选择数据和配置，验证结果不能互用。

## Semantic-GT 数据流

```text
role YAML + stored demo → model-free demo_events → phase/handle manifest
  → replay 重写：T/R XYZ + audit → schema/loader
  → 读取时派生 presence/known → 预测模式仅用于辅助监督
```

| 阶段 | 文件 | 关键函数 |
| --- | --- | --- |
| 角色/manifest | [o2_oracle_provider.py](../../finetune/RLBench/utils/o2_oracle_provider.py) | `_build_assignment()`、`build_demo_event_manifest()`、`_sample_entity_points()`；`demo_events` 不加载策略模型，`sim_replay` 为另一路诊断 |
| handle 证书 | [oracle_handle_alignment.py](../../finetune/RLBench/utils/oracle_handle_alignment.py) | `align_handles()`、`align_semantic_handle_group()` |
| site 几何 | [site_geometry.py](../../finetune/RLBench/utils/site_geometry.py) | `site_geometry_from_object()`、`sample_site_geometry()` |
| 重写/审计 | [rewrite_replay_with_semantic_roles.py](../../tools/rewrite_replay_with_semantic_roles.py) | `_load_manifest()`、`_build_oracle()`、`_fill_slot()`、`_audit_fields()` |
| 校验/可视化 | 同上 | `_validate_semantic_transition()`、`_validate_task_output()`、`_visualize_task_output()` |
| schema/loader | [dataset.py](../../finetune/RLBench/utils/dataset.py) / [get_dataset.py](../../finetune/RLBench/utils/get_dataset.py) | `create_replay()`、`get_dataset()` |
| 旧数据 presence | [YARR uniform_replay_buffer.py](../../finetune/bridgevla/libs/YARR/yarr/replay_buffer/uniform_replay_buffer.py) | `_derive_role_presence()`、`_copy_required_disk_fields()`，只读派生 |
| train/eval 契约 | [semantic_contract.py](../../finetune/RLBench/utils/semantic_contract.py) / [train.py](../../finetune/RLBench/train.py) / [eval.py](../../finetune/RLBench/eval.py) | `_validate_semantic_replay_schema()`、`load_agent()`；严格训练审计读取各任务全量验证报告并抽查 transition，不在启动时重新逐条校验整个 buffer |

## Policy 数据流

```text
RVTAgent.update/act
  → mvt.MVT.forward：模式解析、prior/crop 与输入隔离
  → mvt_single.MVT.forward：同次 VLM features / text context
  → 无序 slots / 固定 T/R queries（预测模式）→ 原 relation/anchor adapter
  → action_feature_routes → translation / R/G/C
```

| 文件 | 当前职责与函数 |
| --- | --- |
| [oracle_prior.py](../../finetune/bridgevla/models/oracle_prior.py) | `resolve_object_prior_mode()`；`rasterize_instance_points()`；`InternalObjectSlotPredictor.forward()` / `InternalRoleQueryPredictor.forward()`；`OracleRelationAnchorFeatureAdapter.forward_with_anchor()` |
| [bridgevla_agent.py](../../finetune/bridgevla/models/bridgevla_agent.py) | `_select_oracle_prior_points()`、`_oracle_network_kwargs()`；teacher-only `_object_slot_auxiliary_losses()` |
| [mvt.py](../../finetune/bridgevla/mvt/mvt.py) | `MVT.forward()`、`_build_oracle_instance_prior()`；管理 coarse/refine 投影与 crop |
| [mvt_single.py](../../finetune/bridgevla/mvt/mvt_single.py) | `MVT.forward()`；VLM/decoder、预测策略的 GT 隔离、共享模式 global 重池化 |
| [object_conditioning.py](../../finetune/bridgevla/models/object_conditioning.py) | `pool_instruction_context()`、`soft_role_geometry()`、`action_feature_routes()`；`role_supervision_mask()`、`mixed_role_map_losses()`、`hungarian_role_slot_losses()`、`reference_null_loss()` 管理有效监督 |
| [soft_coarse_queries.py](../../finetune/bridgevla/models/soft_coarse_queries.py) | `SoftCoarseQueryConditioner.forward()`：coarse soft T/R tokens → 独立 refine slots 的零初始化 query 提示；不继承有效性或几何 |
| [cross_scale_roles.py](../../finetune/bridgevla/models/cross_scale_roles.py) | `inherit_coarse_roles()`、`transform_role_geometry()`；本步角色传递与局部提示支持 |
| [refine_role_masks.py](../../finetune/bridgevla/models/refine_role_masks.py) | `CoarseConditionedLocalQuery.forward()` 更新 coarse-bound tokens；局部 mask head 读出；`refine_role_mask_losses()` 屏蔽未知像素 |
| [heatmap_transport.py](../../finetune/bridgevla/models/heatmap_transport.py) | `reproject_role_heatmaps()` 用观测 XYZ 将完整 coarse 角色概率传入实际 refine crop，保留投影有效性 |
| [role_separation.py](../../finetune/bridgevla/models/role_separation.py) | `role_separation_loss()` 在可核实的角色表面构造 T/R 交叉负例；与旧无序 slot cosine loss 分开 |
| [train.py](../../finetune/RLBench/train.py) | `load_initial_model_checkpoint()`、`load_training_checkpoint()` |
| [training_utils.py](../../finetune/RLBench/training_utils.py) | `resolve_training_epochs()`、`build_batch_plan()`、`optimizer_steps_per_epoch()`；显式 CLI 优先，实际预算存入训练配置 |
| [eval_reporting.py](../../finetune/RLBench/utils/eval_reporting.py) | `evaluation_result()`、`build_eval_run_signature()`；完成数校验、任务汇总与续跑签名 |
| [compare_paired_success.py](../../tools/compare_paired_success.py) | `load_episodes()`、`compare()`；三训练 seed 起的配对 bootstrap，预定 episode 完整性仍需先验收 |

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
| `rlbench_o2_internal_slots_cross_scale.yaml` | 同 joint 预算，6 无序 slots，显式启用 mixed supervision / inheritance / token preservation |
| `rlbench_o2_internal_slots_refine.yaml` | 2 固定 T/R queries；coarse 热图经 XYZ 重投影形成 refine 空间 tokens，与 VLM 输出融合并预测零初始化残差；microbatch 24 / effective batch 192，projector 冻结 |
| `rlbench_o2_internal_slots_soft_coarse.yaml` | 旧两级独立无序 slots + coarse token 软条件；旧动作路由、objectness 阈值 0.25，无硬继承或热图残差；microbatch 48 / effective batch 192 |
| `rlbench_o2_role_queries.yaml` | 2 有序 T/R queries；直接角色图 + Reference NULL loss，本步继承/token 保留；与 6-slot 跨尺度路线分开消融 |

默认旧配置不切换新路由；数值与闭环收益须在目标环境验收。
`current_state` 形状为 `[B,3]`（gripper open + 两维 finger state），兼容旧 relation-state 参数名；
预测部署不得读取 simulator ID、GT phase 或未来目标动作，Oracle 实验另作上界。

## OHT 数据、连续轨迹与部署

单关键点 replay 使用 `oht_bridgevla_v2`，连续轨迹使用独立 `oht_bridgevla_v4`；chunk 不能读取旧单关键点标签补出密集路径。运行命令见[训练与评估](../oht/training.md#训练与评估)，当前状态条件配置见[连续轨迹模型](../oht/training.md#连续轨迹模型)。

| 阶段 | 当前入口与职责 |
| --- | --- |
| 审计与单关键点 replay | [audit_oht_dataset.py](../../tools/audit_oht_dataset.py)、[data/replay.py](../../finetune/OHT/data/replay.py)、[data/dataset.py](../../finetune/OHT/data/dataset.py)：划分、观测/标定契约与下一关键点标签 |
| V4 构建与读取 | [chunk_actions.py](../../finetune/OHT/data/chunk_actions.py)：`make_chunk_targets()`；[chunk_replay.py](../../finetune/OHT/data/chunk_replay.py)：`build()`；[chunk_dataset.py](../../finetune/OHT/data/chunk_dataset.py)：`OHTChunkDataset`、`load_chunk_contract()` |
| 模型与监督 | [chunk_model.py](../../finetune/OHT/chunk_model.py)：`ChunkAgent`、独立 key loss / prediction-only chunk 条件、`chunk_loss()`；[spatial_chunk.py](../../finetune/OHT/spatial_chunk.py)：空间 attention、状态条件与 flow / regression |
| 训练统计 | [chunk_normalization.py](../../finetune/OHT/chunk_normalization.py)：仅 train 的动作/本体统计；[chunk_config.py](../../finetune/OHT/chunk_config.py)：结构、数据与参数化约束 |
| 离线与执行 | [chunk_eval.py](../../finetune/OHT/chunk_eval.py)：按未来 offset 评估；[chunk_protocol.py](../../finetune/OHT/runtime/chunk_protocol.py)：K/dt/H 协商；[chunk_executor.py](../../finetune/OHT/runtime/chunk_executor.py)：固定时间消费绝对 EEF 前缀 |
| 初态与批次结果 | [trajectory_initialization.py](../../finetune/OHT/runtime/trajectory_initialization.py)：`InitialState`，记录初态恢复和有限标定检查；[isaaclab_client.py](../../finetune/OHT/isaaclab_client.py)：`_run()`，提前停止及 attempted/planned 统计 |
| 外部对象接口（单关键点路线） | [predicted_wrapper.py](../../finetune/OHT/runtime/predicted_wrapper.py)：`PredictedObjectWrapper` 与观测白名单；插件不等于已提供训练好的 detector |

V4 chunk 当前要求 `mode=baseline`、`action_mode=eef`，不接入 role cache。regression_state 整体配置的用户确认与定量/CUDA/闭环验收边界见 [已确认基线与交接](../oht/training.md#已确认基线与交接)。

## 尚未实现的代码落点

| 尚未实现 | 设计入口 |
| --- | --- |
| 跨步 object bank、序列 loader、act/reset 状态 | [Memory](../design/role-relation-prior.md#object-centric-memory后续未实现) |
| RLBench 对象条件策略的独立 visibility / 当前 EE pose / phase graph / object-local views / risk head | 不进入当前对象条件策略主线，按标签与瓶颈另验；OHT chunk 已有实测 TCP/proprio 输入 |

predictor 内的 attention `memory` 不是历史 bank。`RVTAgent.reset()` 会清理 simulator alignment 的锁与计数，
但当前没有学习式 role memory/object bank。

<a id="项目审查2026-09-29"></a>
<a id="已确认尚未修复的运行问题"></a>

## 当前运行限制

以下按当前源码复核，覆盖相关入口与 vendor 接口；不代表逐行审计全部文件或完成 GPU/模拟器验收。旧审查锚点保留在此处，避免从旧链接进入过时快照。

| 优先级 | 问题与影响 | 代码证据 | 当前处理 |
| --- | --- | --- | --- |
| P1 | 旧 launcher 硬编码路径、未引用 `$@`，会拆开 overrides/含空格路径 | [RLBench train.sh](../../finetune/RLBench/train.sh)；Colosseum/GemBench 同类脚本 | RLBench 示例改用 `train_8x40.sh`；其他分支先适配脚本 |
| P1 | eval resume 没有覆盖全部策略依赖，模型模块改变后可能复用旧结果 | [eval.py](../../finetune/RLBench/eval.py) 的 `runtime_files` 未含 `mvt_single.py` / `object_conditioning.py` / `oracle_prior.py` | 代码/数据改变后使用新评估目录，不复用旧 journals |
| P1 | 配对 CI 只要求双方已有 episode 集合相同，双方同样中断仍可能通过 | [compare_paired_success.py](../../tools/compare_paired_success.py) 的 `load_episodes()` / `compare()` 未校验 requested summary | 先核对每 task 完成数与预定 episode 范围，再运行 CI |
| P1 | OHT 执行错误会停止后续 episodes；`success_rate` 的分母为 attempted，可能小于 planned | [isaaclab_client.py](../../finetune/OHT/isaaclab_client.py) 的 `_run()` | 检查 planned/attempted/unattempted 和停止原因；[固定 case 验收](../oht/runtime.md#时间预算与停止结果) |
| P2 | OHT 记录初态的 contract loader 只支持 V3；data-config 路线不自动比对完整相机/状态标定 | [trajectory_replay.py](../../finetune/OHT/runtime/trajectory_replay.py)、[trajectory_initialization.py](../../finetune/OHT/runtime/trajectory_initialization.py) 的 `InitialState._check_calibration()` | V4 使用匹配的 data-config；自动检查字段及手动核对项见[同初态说明](../oht/runtime.md#相同记录初态运行策略) |
| P2 | installer 与其他 benchmark 流程不完全可移植；统计依赖目录/顺序假设 | [pretrain installer](../../pretrain/pretrain_install.sh)、[GemBench client](../../finetune/GemBench/run_client.sh)、[GemBench aggregation](../../finetune/GemBench/cal_results.py)、[Colosseum aggregation](../../finetune/Colosseum/cal_statics.py) | 预先配置依赖/路径，核对任务与 episode 覆盖；不能直接套用 RLBench 统计保障 |
| P2 | 预算测试仍要求 OHT 与 RLBench full 配方相同，当前 epochs 为 60 / 100 | [test_oht_training_plan.py](../../tests/test_oht_training_plan.py) 的 `test_full_preset_matches_rlbench_recipe_and_mvt()` | 同步测试预期与各自配方；当前 OHT 数值见[预算说明](../oht/training.md#预算与卡数)，不能改配置凑旧断言 |

P1 指会破坏实验可运行性或结果可信度，P2 指复现/配置风险；以上没有宣称已修复。
resume 签名对 checkpoint 使用 path/size/mtime，对 raw data 使用目录路径，不是全内容 provenance。

<a id="模型与数据审查结论"></a>

模型输入、监督隔离与角色有效性以[当前数据契约](../design/role-relation-prior.md#当前实现与数据契约)为准；训练和评估步骤集中在[对象条件策略指南](../guides/object-conditioning.md)。mask-only 身份证书不认证整场景 XYZ，presence 不等于几何可用，预测模式的 teacher 只用于辅助监督。

<a id="验证与下一步"></a>

## 验证入口

| 目的 | 入口 |
| --- | --- |
| 本地文档链接与锚点 | 仓库根目录运行 `python tools/check_docs.py`；只做离线结构检查 |
| RLBench 动作/teacher 隔离及语义契约 | [test_object_conditioning_forward.py](../../tests/test_object_conditioning_forward.py)、[test_semantic_contract.py](../../tests/test_semantic_contract.py) |
| RLBench 汇总与配对比较 | [test_eval_reporting.py](../../tests/test_eval_reporting.py)、[test_paired_success.py](../../tests/test_paired_success.py)；完成集合仍须对照预定 cases |
| OHT 数据、状态与执行回归 | [OHT 本地验证](../oht/training.md#本地验证)；对应改动选取现有测试 |
| 真实模型与控制 | 目标 Linux/CUDA 环境做单步训练、同初态 GT/模型对照和固定 case/seed 闭环；CPU 替身不提供此类运行证据 |

## 历史项目审查（2026-09-29）

早期审查覆盖 first-party launcher、角色一致性与评估口径；后续新增的 OHT、soft coarse、refine masks 已反映在上面的当前索引。旧测试数量不作为当前验收结果重复维护；OHT 已确认配置与历史验证记录见 [已确认基线与交接](../oht/training.md#已确认基线与交接)。
