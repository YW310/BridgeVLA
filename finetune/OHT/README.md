# OHT：数据、训练与运行

当前主流程是 **V4 数据 → 关键目标 + 连续 EEF chunk → 离线评估 → IsaacLab**。以下命令在 BridgeVLA 仓库根目录执行；路径按服务器实际挂载位置修改。

| 要做什么 | 入口 |
| --- | --- |
| 已有 `replay-source-v4`，开始训练 | [训练](#3-训练) |
| 只有原始数据，生成 V4 | [准备数据](#2-准备数据) |
| 查看分割角色或稠密 GT 轨迹 | [数据准备与标注](../../docs/guides/oht-data-preparation.md) |
| 检查深度、相机、ROI 或对象点云 | [数据诊断](../../docs/guides/oht-data-diagnostics.md) |
| 查看初始化、续训、日志和预算 | [训练与评估说明](../../docs/guides/oht-training.md) |
| 运行旧单关键点或物体辅助实验 | [单关键点与物体辅助](../../docs/guides/oht-single-keypoint.md) |
| 启动仿真、在线可视化、GT replay | [VLA server 与 IsaacLab](../../docs/guides/oht-vla-server.md) |

## 1. 环境与路径

数据处理环境：

```bash
python -m pip install -r finetune/OHT/requirements-data.txt
```

训练需使用已安装 PyTorch/CUDA、Transformers、PaliGemma 和 point-renderer 的 BridgeVLA 环境，见[安装说明](../../docs/guides/installation.md)。独立环境的附加依赖：

```bash
python -m pip install -r finetune/OHT/requirements-model.txt
python -m pip install --no-build-isolation -e finetune/bridgevla/libs/point-renderer
```

统一设置后续命令使用的路径：

```bash
ROOT=/common-data-32t/data/robot_data/oht_curobo_pd_v423-seg
OUT="$ROOT/bridgevla-seg-v4-run01"
PRETRAIN=/home/yiwei/project/BridgeVLA/checkpoints/pretrain
```

`ROOT` 是原始数据父目录，`OUT` 是生成流水线目录，训练读取 `$OUT/replay-source-v4`。`PRETRAIN` 是 BridgeVLA 热图预训练目录；PaliGemma 基础模型和 processor 资源仍需单独可访问或已缓存。资源已完整缓存时可设置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。

## 2. 准备数据

**已有完整 V4 时直接进入训练。** 只含原始 `disassemble_left-seg/lerobot_dataset` 时执行：

```bash
bash finetune/OHT/prepare_seg_data.sh \
  --root "$ROOT" --output "$OUT" \
  --chunk-horizon 16 --action-frame-stride 1
```

脚本依次生成 audit、V4 replay、校验结果、分割角色标注和预览，无需提前准备 `contract.json`。首次生成时 `OUT` 必须尚不存在；只包含 `disassemble_left` 也可运行。

| 产物 | 用途 |
| --- | --- |
| `$OUT/replay-source-v4/contract.json`、`complete.json` | 完整缓存及数据契约；训练传入它们所在的目录 |
| `$OUT/replay-source-v4/targets/` | 下一关键目标与未来连续实测 EEF/关节/夹爪标签 |
| `$OUT/replay-source-v4/preview/index.html` | sensor 与点云投影上的稠密 GT 轨迹 |
| `$OUT/role-preview/index.html` | target 左轮 ID 1、reference 放置盘 ID 4 的标注预览 |

V4 当前 chunk 配置使用 `mode: baseline`，**不接入 role cache 或分割角色监督**。导出 T/R 标注不等于模型已启用 object-aware。普通非分割数据的构建、标注刷新和参数说明见[数据准备](../../docs/guides/oht-data-preparation.md)。

## 3. 训练

### 配置选择

先跑 `regression_state` 作为确定性对照，再用同一份 V4、初始化来源和 seed 比较另外两组。它们是待验证的实验，不代表已确认哪组精度最好。

| 配置 | 动作头 |
| --- | --- |
| [oht_chunk_v4_regression_state_8x40.yaml](configs/oht_chunk_v4_regression_state_8x40.yaml) | 预测相对当前 TCP 的未来 pose |
| [oht_chunk_v4_delta_state_8x40.yaml](configs/oht_chunk_v4_delta_state_8x40.yaml) | 预测逐步平移/旋转增量，累积后监督完整轨迹 |
| [oht_chunk_v4_flow_state_8x40.yaml](configs/oht_chunk_v4_flow_state_8x40.yaml) | 同样状态条件下的 flow matching |

三组均使用 coarse/refine 视觉特征、本体状态归一化、逐层 AdaLN 状态条件与可学习关键目标 gate；最终都输出世界坐标绝对 EEF。默认 K=16，客户端建议执行前 H=4 项。结构、损失和旧消融配置见[算法设计](../../docs/design/oht-spatial-flow-chunk.md)。

### 启动命令

```bash
CONFIG=finetune/OHT/configs/oht_chunk_v4_regression_state_8x40.yaml
EXP=oht-seg-v4-regression-state-s0

GPUS_PER_NODE=2 bash finetune/OHT/train_8x40.sh \
  --config "$CONFIG" \
  --replay "$OUT/replay-source-v4" \
  --pretrain-path "$PRETRAIN" \
  --log-dir "$OUT/runs" --exp-id "$EXP"
```

其他两组分别替换 `CONFIG` 和 `EXP`，使用新的实验名。当前训练设置：

| 项目 | 当前三组配置 |
| --- | --- |
| 训练方式 | `stage: joint`；关键目标与 chunk 联合训练 |
| 预算 | 60 epochs，`train_iter=40000`，最多 83,300 优化步 |
| 优化器 | AdamW，学习率 `8e-5`，全局有效 batch 16 |
| 冻结 | 视觉塔、multimodal projector、Gemma 前 16 层；后续层与动作头训练 |
| 数据增强 | 关闭全局刚体变换增强 |

每卡 batch 8；两卡累积 1 次，单卡累积 2 次。直接增加到四卡或八卡时需在新配置中减小每卡 batch，使 `每卡 batch × 卡数` 整除 16。启动脚本名 `8x40` 不会自动调整这些设置。

默认 `joint` 不要求旧 OHT checkpoint。需要复用已训练 encoder 时增加 `--init-checkpoint /path/to/model_last.pth`；新结构/归一化会重建 decoder 和 optimizer。相同实验续训用 `--resume`，不可用它切换结构；完整规则见[初始化与续训](../../docs/guides/oht-training.md#初始化与续训)。

当前上限会在约 33.32 个采样 epoch 时停止；完整 60 轮的设置见[预算说明](../../docs/guides/oht-training.md#预算与卡数)。首次启动从 train 拟合动作尺度及关节归一化，向各 rank 广播，写入 `chunk_normalization.json`、`proprio_normalization.json` 和 checkpoint；推理自动复用。现有 V4 无需重建。

### 查看训练

```bash
tensorboard --logdir "$OUT/runs/train" --port 6006
```

训练启动时打印 `Run directory: ...`，该目录含 `model_last.pth`、`metrics.jsonl`、`tensorboard/` 和 `train_visualizations/`。Images 中的 `train_visualization/chunk`：红色预测、绿色实测未来、白色当前 TCP。重点比较 `chunk_position_error_m`、`chunk_prefix_1_position_error_m`、`chunk_prefix_4_position_error_m` 与 `chunk_stationary_error_m`；loss 数值不能直接当作米误差。flow 的轨迹来自独立部署采样，重建误差另记。日志、图像和保存规则见[训练与评估说明](../../docs/guides/oht-training.md)。

## 4. 离线评估

将实际 run 目录填入 `RUN`，使用不再更新的固定 checkpoint：

```bash
RUN=/path/to/actual/run
python -u -m finetune.OHT.eval open-loop \
  --checkpoint "$RUN/model_last.pth" \
  --replay "$OUT/replay-source-v4" --split val \
  --output "$OUT/eval/regression-state-val.json"
```

报告及图片目录使用新路径。先检查少量样本时增加 `--limit 8 --visualize-every 1`，正式评估去掉 `--limit`。V4 报告按未来 offset 统计误差；图片中查看关键目标热图与 chunk 预测/GT。离线误差不等于闭环成功率。

## 5. 部署与仿真

在模型服务器启动同一个 run：

```bash
GPU_ID=1 bash finetune/OHT/vla_server.sh \
  --model-path "$RUN/model_last.pth" \
  --device cuda:0 --host 0.0.0.0 --port 8004
```

IsaacLab 服务器的启动命令、原始 sensor/3D 在线可视化、chunk 前缀执行和 GT replay 统一见[VLA server 与 IsaacLab 指南](../../docs/guides/oht-vla-server.md)。客户端根据 `/model_info.chunk_spec` 选择 chunk 执行；它按时序跟踪前缀，与单关键点的“到位后开闭夹爪”不同。

## 6. 其他路线与说明

- 单关键点 `oht_full_8x40.yaml`、`role_queries` 和 `predicted_external` 使用单关键点 replay，见[单关键点与物体辅助](../../docs/guides/oht-single-keypoint.md)。full 配置冻结前 6 层 Gemma；当前 chunk 配置覆盖为 16 层。
- 旧 `displacement`、`relative_goal`、`spatial`、`flow_scaled/temporal` 是独立消融，配置差异见[算法设计](../../docs/design/oht-spatial-flow-chunk.md#旧配置与消融)。
- 四种单动作表示见[动作模式指南](../../docs/guides/oht-action-modes.md)；当前 chunk 固定 EEF。
- 本地验证命令见[训练与评估说明](../../docs/guides/oht-training.md#本地验证)。CPU/接口回归验证正确性，真实 CUDA 吞吐、轨迹精度和闭环结果需在服务器测量。
