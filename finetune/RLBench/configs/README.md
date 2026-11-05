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
