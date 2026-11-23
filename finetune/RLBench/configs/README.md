# RLBench 实验配置

本目录的 YAML 由 `train.py` / `eval.py` 通过 yacs 合并到 [bridgevla/config.py](../../bridgevla/config.py) 的默认值上，**未知键或类型不符会直接报错**。可用下面的命令离线校验全部配置：

```bash
cd finetune
python - <<'EOF'
import glob, bridgevla.config as c
for f in sorted(glob.glob('RLBench/configs/*.yaml')):
    if f.endswith('semantic_roles.yaml'):
        continue  # 角色契约，不是实验配置
    c.get_cfg_defaults().merge_from_file(f); print('OK', f)
EOF
```

[rlbench_o2_semantic_roles.yaml](rlbench_o2_semantic_roles.yaml) 不是训练配置，它定义各任务的 Target/Reference 语义，由 `oracle_semantic_contract.role_config` 引用，整个文件的 SHA-256 进入数据/checkpoint 契约，修改注释也会改变摘要。

## 通用约定

| 规则 | 说明 |
| --- | --- |
| 有效 batch | `global_batch_size` 必须能被 `bs × GPU 数` 整除，梯度累积 = `global_batch_size / (bs × GPU 数)`；不满足会在启动时报错（`training_utils.build_batch_plan`）。`global_batch_size: 0` 表示每个 micro-batch 一次更新 |
| 步数 | 每 epoch 的 optimizer step = `train_iter // global_batch_size`；`max_optimizer_steps > 0` 时以它为总预算，否则为 `epochs × 每 epoch 步数` |
| 8 卡 | `train_8x40.sh` 默认 `GPUS_PER_NODE=8`。`bs: 24` 的配置可用 1/2/4/8 卡；`internal_slots.yaml` 与 `base`/`trend` 见下表 |
| 冻结 | `freeze_vision_tower`、`freeze_multimodal_projector`、`freeze_gemma_prefix_layers` 为配置项；`lm_head` / `embed_tokens` 始终冻结 |
| 优化器 | 基线 `full_8x40`/`trend_8x40` 与 o2 系列使用 `adamw` 且 `use_scheduler: False`；`rlbench_config*.yaml` 与两个对照基线沿用默认 `adam` + scheduler |
| 结构变化 | 路由/predictor 类型改变后用 `--init_checkpoint`，不要 `--resume` 旧 optimizer；评估使用 run 目录保存的 `exp_cfg.yaml` |

## 配置一览

下表由合并后的实际配置生成（`bs` 为每卡 micro-batch）。

### 基线

| 配置 | bs | global batch | train_iter | epochs | max steps | freeze (vision / Gemma 前 N 层 / projector) | lr | 优化器 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `rlbench_config` | 4 | 0（旧逻辑） | 160000 | 100 | — | 否 / 0 / 否 | 8e-5 | adam + scheduler |
| `rlbench_config_small` | 2 | 0（旧逻辑） | 40000 | 100 | — | 否 / 0 / 否 | 1e-5 | adam + scheduler |
| `rlbench_full_8x40` | 4 | 192 | 160000 | 100 | 83300 | 否 / 6 / 是 | 8e-5 | adamw |
| `rlbench_trend_8x40` | 4 | 192 | 38400 | 100 | 20000 | 否 / 6 / 是 | 8e-5 | adamw |

### 对象条件策略（`rlbench_o2_*`）

均为 `train_iter 38400`、`global_batch_size 192`、`epochs 50`（10000 步）、lr `4e-5`。

| 配置 | Object 来源 | bs | vision 冻结 | projector 冻结 | `gemma_lr` | 路由开关 / predictor |
| --- | --- | --- | --- | --- | --- | --- |
| `gt_instance` | 启发式 T/R replay（`o2_gt_instance`） | 24 | 是 | 是 | — | 原 relation 路由 |
| `semantic_gt` | Semantic-GT | 24 | 是 | 是 | — | 原 relation 路由 |
| `semantic_gt_relation_anchor` | Semantic-GT | 24 | 是 | 是 | — | + relation anchor |
| `semantic_gt_joint` | Semantic-GT | 24 | 是 | 是 | 1e-5 | `shared_action_features`、`use_context` |
| `predicted_objects` | 外部 predictor（`o2_predicted_relation`） | 24 | 是 | 是 | — | 仅预测点集，`oracle_prior_strict` |
| `internal_slots` ⚑ | 内部无序 slots ×2 | **48** | **否** | 是 | — | 原动作路由，adam + scheduler |
| `internal_slots_soft_coarse` | 无序 slots ×2 | 24 | 是 | 是 | 1e-5 | `soft_coarse_queries` |
| `internal_slots_soft_coarse_matched` ⚑ | 无序 slots ×2 | 24 | 是 | 是 | — | `soft_coarse_queries`，adam + scheduler |
| `internal_slots_joint` | 无序 slots ×6 | 24 | 是 | **否** | 1e-5 | `shared_action_features`、`use_context` |
| `internal_slots_cross_scale` | 无序 slots ×6 | 24 | 是 | 是 | 1e-5 | joint + `inherit_coarse_roles` 等 |
| `internal_slots_refine` | 固定 T/R queries ×2（`role_queries`） | 24 | 是 | 是 | 1e-5 | cross-scale + refine mask / local query / 热图残差 |
| `role_queries` | 固定 T/R queries ×2 | 24 | 是 | **否** | 1e-5 | joint + `inherit_coarse_roles` |

以上除标 ⚑ 的两项外，o2 配置的 Gemma 前 18 层均冻结，优化器均为 `adamw`、不使用 scheduler。

⚑ **对照基线，刻意保持原样**：[tests/test_soft_coarse_matched_config.py](../../../tests/test_soft_coarse_matched_config.py) 要求 `internal_slots.yaml` 不变，`soft_coarse_matched` 与之只差 `exp_id` 和 `soft_coarse_queries`；A 侧运行时用 `--freeze_vision_tower --exp_cfg_opts "bs 24 pin_memory False"` 对齐。做 soft-coarse 的单因素对照请用这一对；`internal_slots_soft_coarse` 采用 joint 系列的训练配方，不是严格匹配的对照。

### 与 GPU 数的关系

| 配置 | 可用卡数（`global_batch_size 192`） |
| --- | --- |
| `bs: 24` 的 o2 配置 | 1、2、4、8 |
| `internal_slots`（`bs: 48`） | 1、2、4 |
| `rlbench_full_8x40`、`rlbench_trend_8x40`（`bs: 4`） | 1、2、3、4、6、8、12、16、24、48 |

## Slot-guided SAM3 online inference

### Role relocalize：同帧角色 heatmap 与 SAM3 分割

`rlbench_o2_role_relocalize.yaml` 训练的模型使用
[relocalize online 配置](rlbench_slot_guided_sam3_relocalize_online.yaml)。保留
`--slot-sam3-guidance-stage refine --slot-sam3-stage action`，并指定
`--slot-sam3-diagnostics-dir "$RUN_DIR/sam3"`，会在每步保存首遍策略的 Coarse/Refine
Target、Reference heatmap 和当前相机的 SAM3 分割，不增加网络推理次数。
打开 `sam3/run_*/episode_*/frame_*/index.html`，可同时查看两者；日志也打印该路径。
首次排查建议 `--eval-episodes 1 --episode-length 3`。

热图按各阶段自己的虚拟视图显示；Refine 使用局部 crop，不能直接把两阶段像素坐标对比。
Coarse/Refine 热图的最大叠加权重为 **85%**：角色颜色按绝对概率加权，绿色 action 图按相对空间 softmax 加权。`heatmaps/prompt_report.json`
记录角色状态、Reference NULL 概率、`guidance_selection` 及响应强度；原始数组在
`heatmaps/slot_arrays.npz`。Target/Reference 的 soft overlap 只衡量两张预测图的重叠，
不代表相对 GT 的精度。详细排查方法见[在线诊断](../../../docs/slot_guided_sam3_online.md#same-frame-heatmaps)。

### bbox 太大与分割叠加

原 bbox 在投影点包围框每侧增加 `4px + 10%` 的 padding。框过大时，可在已有命令中加：

```bash
--slot-sam3-perception-config "$PWD/finetune/RLBench/configs/slot_sam3_tight_bbox.json"
```

该配置只将 padding 改为每侧 `1px + 3%`，保留点过滤、分割选择和融合阈值。
它同时支持 internal_slots 与 role_relocalize；不会改变模型权重。如果预测点本身跨越多个
物体，缩 padding 不能纠正角色预测；框变紧也可能截掉完整物体，需要对照分割结果确认。

在线诊断 `<camera>.diagnostic.png` 的 SAM3 mask 叠加权重提高至 **75%**，原 RGB 占 25%。
旁边显示提示框和选中 mask 的宽高；`bbox_diagnostics.json` 记录各相机的数值。
提示框尺寸是坐标跨度，mask 尺寸是占据的像素范围，两者可以相差一个边界像素。
这些显示修改不改变 SAM3 输入、mask 或动作。

<a id="gt-sam3"></a>

### 已训练 Semantic-GT 模型：GT 提示 → SAM3 → BridgeVLA

已有 `rlbench_o2_semantic_gt.yaml` 训练结果可以直接使用：
新 [online 配置](rlbench_gt_guided_sam3_online.yaml) 保留原 Relation adapter、
Coarse/Refine 动作网络和点云池化方式，Slot 关闭、Anchor rank 为 0，不增加参数。
训练配置与训练流程不变，训练时仍不需要 SAM3 对象几何。
`semantic_gt_joint`、Slot 和 role-query checkpoint 使用不同结构，不适用这份配置。

当前实时相机 GT handle mask + live semantic phase → Target/Reference GT 点集 →
原相机 box/point → 冻结 SAM3 → mask + 当前 RGB-D → world 对象表面点 →
原场景归一化及 Refine crop → 原两阶段 Relation adapter / Action Head。
GT 只提供提示和角色身份；动作输入的对象点与投影 prior 均来自 SAM3 分割。
GT mask/点不进入策略 observation，不调用 Slot、不读取未来 observation 或 expert action。
这属于使用特权 GT 的 Oracle 实验，不能当作自主角色识别结果。

在服务器仓库根目录运行，`MODEL_FOLDER` 改为**已经训练完成的 semantic_gt run**，
目录需包含 `model_last.pth`、`exp_cfg.yaml` 和 `mvt_cfg.yaml`。
`SAM31` 是 TASK 工程目录，必须含 `slot_guidance_producers.py` 等接口；SAM3 Python 包另行安装。

```bash
conda activate bridgevla
MODEL_FOLDER=/path/to/trained_semantic_gt_run
python finetune/RLBench/gt_guided_sam3_online.py \
  --model-folder "$MODEL_FOLDER" --model-name model_last.pth \
  --eval-datafolder "$PWD/LPY/BridgeVLA_RLBench_EVAL_DATA" \
  --tasks open_drawer --eval-episodes 1 --episode-length 25 --device 0 \
  --oracle-role-config "$PWD/finetune/RLBench/configs/rlbench_o2_semantic_roles.yaml" \
  --slot-sam3-task-repo /data/project/SAM31 \
  --slot-sam3-checkpoint "$PWD/checkpoints/sam3/sam3.pt" \
  --slot-sam3-prompt-mode point+box \
  --slot-sam3-diagnostics-dir "$PWD/outputs/gt_sam3"
```

默认是 `oracle_online_experiment` / `stage=action`。不要加 `--oracle-provider rlbench_gt`：
这会选择直接 GT 动作条件接口；本入口自动采集仅供 SAM3 提示的 GT side channel。
增加 `--slot-sam3-stage segmentation` 可先观察分割并执行当前无对象条件动作；
`--slot-sam3-prompt-mode box-only` 做提示消融。GT 路径不使用 Slot `prompts` 诊断模式。
无需加载权重的配置检查：

```bash
python finetune/RLBench/gt_guided_sam3_online.py --check-config
python finetune/RLBench/slot_guided_sam3_offline.py \
  --config finetune/RLBench/configs/rlbench_gt_guided_sam3_offline.yaml --check-config
```

启动应显示 `current GT prompts -> SAM3 -> trained GT Relation adapter`、`no Slot prediction`。
诊断图、原图提示、mask 与融合结果位于 `outputs/gt_sam3/run_*/episode_*/frame_*`；
评测目录下 `slot_sam3_steps.jsonl` 记录 `guidance_source=oracle_gt`、
`slot_prediction_used=false`、`status`、失败原因、当前源帧索引、延迟及 SAM3 显存峰值。
`conditioned` 才表示该步实际使用了 SAM3 对象条件；`baseline_fallback` 使用同一模型、
同一当前观测的**无对象条件动作**，不会改用 GT 几何或 Slot。
想让感知失败立即报错可加 `--slot-sam3-fallback fail`。

Reference 的 `NULL` 表示语义不存在；存在但当前 mask 为空是 `NOT_VISIBLE`；
几何/分割不可靠为 `LOW_CONFIDENCE`，不会因空 mask 自动判 NULL。
接触点、成功区域等非可视 `site` 没有可分割实例边界，当前明确报错并按 fallback 处理；
因此不能假定所有任务和阶段都适用。运动 wrist 使用当前观测的外参和模拟器 microstep。

离线单帧使用同一已训练模型和 [offline 配置](rlbench_gt_guided_sam3_offline.yaml)：

```bash
python finetune/RLBench/slot_guided_sam3_offline.py \
  --config finetune/RLBench/configs/rlbench_gt_guided_sam3_offline.yaml \
  --checkpoint "$MODEL_FOLDER/model_last.pth" \
  --task-repo /data/project/SAM31 \
  --frame /path/to/frame.json --state /path/to/state.json \
  --oracle-labels /path/to/oracle_labels.json \
  --sam3-checkpoint "$PWD/checkpoints/sam3/sam3.pt" \
  --device cuda:0 --mode point+box --output-dir outputs/gt_sam3_snapshot
```

快照/状态/逐相机 GT mask NPY 格式见 [offline 输入契约](../../../docs/slot_guided_sam3_offline.md)。
使用新的空输出目录。输出 `action_report.json` 中动作差值不是对 GT 的动作误差或成功率。
本机软件/Mock 测试只能验证接口、坐标变换、数据隔离和回退；真实 SAM3 mask IoU、
对象中心误差及 paired RLBench 成功率仍需在有权重/GPU/模拟器的服务器测量。

[rlbench_slot_guided_sam3_internal_slots_online.yaml](rlbench_slot_guided_sam3_internal_slots_online.yaml) 基于原始 `rlbench_o2_internal_slots.yaml`，保留无序 slots、独立 Refine 和旧 Anchor，直接加载原版 checkpoint。仅启用冻结的 SAM3 推理及几何动作条件；[完整命令](../../../docs/slot_guided_sam3_online.md#original-slots-sam3)包含分割诊断模式。继续训练仍使用原始 YAML。

[rlbench_slot_guided_sam3_online.yaml](rlbench_slot_guided_sam3_online.yaml) enables `online_experiment` in the normal RLBench evaluation loop. Run `slot_guided_sam3_online.py`; its [commands and runtime contract](../../../docs/slot_guided_sam3_online.md) describe the persistent model, current-observation provider and TASK SAM3 reuse. Runtime paths live in `slot_guided_sam3` and support `--slot-sam3-*` overrides. The default config remains disabled.

## Slot-guided SAM3 offline snapshots

[rlbench_slot_guided_sam3_offline.yaml](rlbench_slot_guided_sam3_offline.yaml) enables frozen `offline_experiment` inference using `slot_guided_sam3_offline.py`. Read the [commands and single-frame input contract](../../../docs/slot_guided_sam3_offline.md). The loader checks the checkpoint's saved architecture and routing metadata, then loads existing weights strictly. It never starts an RLBench control rollout or trains a model.

[rlbench_slot_guided_sam3_draft.yaml](rlbench_slot_guided_sam3_draft.yaml) remains a disabled compatibility profile for existing commands; new offline experiments use the profile above. Other configs and global defaults remain unchanged.

## 常用开关

| 键 | 作用 |
| --- | --- |
| `use_oracle_objects` / `use_predicted_objects` | 二选一，互斥；对象模式非 `none` 时需 `--train_replay_storage_dir` |
| `rvt.object_prior_mode` / `rvt.oracle_prior_mode` | 对象来源，须与上面两个开关一致（`train.py` 启动时校验） |
| `object_slots.*` | slots 数量、维度、decoder 层数；`predictor_type` 为 `slots` 或 `role_queries` |
| `object_conditioning.*` | 动作路由与 refine 相关开关，详见[操作指南](../../../docs/guides/object-conditioning.md#配置选择) |
| `object_conditioning.soft_coarse_null_renorm` | 默认 `False`；需 `soft_coarse_queries`。Reference 为 NULL 时把注意力转给 Target 而不是稀释提示，可用 `--exp_cfg_opts "object_conditioning.soft_coarse_null_renorm True"` 做对照 |
| `rvt.img_aug` / `img_aug_2`（MVT 配置） | `img_aug` 在渲染后加图像噪声；`img_aug_2` 在渲染前对逐点 RGB 加噪并截断到 [-1, 1]（仅训练模式，0 关闭）。`img_aug_2` 曾因未写回特征而不生效，修复后才真正增强，旧 run 不可直接比较 |
| `oracle_semantic_audit` / `oracle_semantic_contract.enforce` | 严格 train/eval 契约校验；各配置默认关闭，开启须两者同时为 `True` |

## 修改配置时

1. 新增键先加到 [bridgevla/config.py](../../bridgevla/config.py) 的默认值，再写入 YAML。
2. 影响推理路由的键同时写入 `train.py` 的 checkpoint `object_conditioning` 元数据，并加入 `routing_keys`（续训）和 `eval.py` 的校验列表；缺失的旧键按 `False` 处理。
3. 新增推理依赖源码时同步 `eval.py` 的 `_FINETUNE_RUNTIME_FILES`，否则续跑会复用旧评估结果。
4. 改完运行上面的校验脚本和 `python -m pytest tests/test_soft_coarse_matched_config.py tests/test_eval_runtime_fingerprint.py`。
