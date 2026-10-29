# OHT 训练、续训与评估说明

当前 V4 三组训练命令和配置选择统一见 [OHT README](../../finetune/OHT/README.md#3-训练)。本页说明预算、checkpoint、日志与评估；旧单关键点和物体辅助命令见[单关键点指南](oht-single-keypoint.md)。所有命令从 BridgeVLA 根目录执行。

## 预算与卡数

当前 `regression_state`、`delta_state`、`flow_state` 均继承以下设置：

| 设置 | 值 |
| --- | --- |
| `epochs` / `train_iter` | 60 / 40000 |
| `global_batch_size` / `batch_size` | 16 / 每卡 8 |
| `max_optimizer_steps` | 83300 |
| `checkpoint_every_epochs` | 10 |
| 优化器 | AdamW，学习率 `8e-5`，weight decay `1e-6`，固定学习率 |
| 冻结 | 视觉塔、projector、Gemma 前 16 层；lm_head/embed_tokens 沿用冻结规则 |

`train_iter` 表示每轮任务均衡、有放回采样的预算，不是数据集长度。每轮 `floor(40000/16)=2500` 个优化步，目标步数为 `min(60×2500,83300)=83300`，约 33.32 个采样 epoch。要完整训练 60 轮，需在新配置中将 `max_optimizer_steps` 设为 `0` 或至少 `150000`；文档整理不改变现有 YAML。

`accumulation_steps: null` 自动计算累积次数：两卡为 1，单卡为 2。四卡可在新的 YAML 中设每卡 batch 4，八卡设 2，保持全局 batch 16；内存是否足够需实测。配置必须满足 `global_batch_size` 能被 `batch_size × 卡数` 整除。严格 resume 不能更换卡数或 batch 设置。

`train_8x40.sh` 默认启动八卡，当前例子显式设置 `GPUS_PER_NODE=2`。脚本只负责 torchrun，不会因文件名自动调节显存或 batch。`--standalone` 自动分配 rendezvous endpoint，不应依赖 `MASTER_PORT` 固定实际 rendezvous 端口。

## 初始化与续训

| 操作 | 参数 | 恢复内容 |
| --- | --- | --- |
| 从基础模型开始 joint 训练 | `--pretrain-path` 可指定热图预训练 | 不要求已训练的 OHT checkpoint；仍需 PaliGemma 基础资源 |
| 新实验复用旧模型 | `--init-checkpoint /path/to/model_last.pth` | 兼容权重；optimizer、采样位置重新开始 |
| 原实验继续训练 | `--resume /path/to/model_last.pth` | 模型、optimizer 和采样位置 |

joint 同时训练关键目标和 chunk；frozen_key 冻结整个关键目标网络，必须有已训练 checkpoint。关键目标 GT 只用于独立监督分支，chunk 条件始终来自预测路由。V4 数据和推理输出均为世界绝对 EEF，内部 delta 参数化不会改变客户端协议。

改变本体归一化、状态条件、动作尺度或 decoder 架构时，用 `--init-checkpoint` 在新 run 中保留 encoder、重新初始化 decoder。仅改变允许的 loss 权重或采样设置，新 run 可保留 decoder；具体边界由 `load_chunk_weights()` 检查。已有单关键点 checkpoint 只能初始化兼容 encoder，不能 resume 成 chunk 模型。

严格 resume 检查解析继承后的训练配置、replay/role cache 契约和卡数。允许调整训练预算、保存周期、`num_workers` 与独立可视化设置；学习率、batch、冻结、动作定义、seed 等必须一致。epoch 预算可调整 `epochs`、`max_optimizer_steps`、`checkpoint_every_epochs`；step 预算可调整 `optimizer_steps`、`checkpoint_interval`，两套预算不能同时设置。GPU RNG 尚未恢复，不承诺逐位重现。

```bash
CONFIG=finetune/OHT/configs/oht_chunk_v4_regression_state_8x40.yaml
OUT=/common-data-32t/data/robot_data/oht_curobo_pd_v423-seg/bridgevla-seg-v4-run01
RUN=/path/to/actual/run
GPUS_PER_NODE=2 bash finetune/OHT/train_8x40.sh \
  --config "$CONFIG" --replay "$OUT/replay-source-v4" \
  --pretrain-path /home/yiwei/project/BridgeVLA/checkpoints/pretrain \
  --resume "$RUN/model_last.pth"
```

## 预检与短测

CPU 数据预检不启动模型或创建训练输出目录；新配置需要拟合 train 的动作/本体归一化，因此仍可能读取训练 NPZ：

```bash
python -m finetune.OHT.train \
  --config "$CONFIG" --replay "$OUT/replay-source-v4" --validate-only
```

GPU 短测可新建 `finetune/OHT/configs/my_chunk_smoke.yaml`：

```yaml
extends: oht_chunk_v4_regression_state_8x40.yaml
epochs: 1
max_optimizer_steps: 2
```

用该 config 和新实验名运行 README 的训练命令。`smoke.yaml` 是旧单关键点两步测试，不能直接用于 V4。

## 输出目录

`--log-dir ROOT` 自动创建 `ROOT/train/<exp_id>/<exp_note>/MM_DD_HH_MM/`，空备注省略。时间采用 Asia/Shanghai，同分钟重复启动追加后缀。rank 0 创建并广播路径，终端打印 `Run directory: ...`；以该绝对路径为准。

```text
run/
  run.json
  config.json
  contract.json
  training_plan.json
  chunk_normalization.json
  proprio_normalization.json
  metrics.jsonl
  tensorboard/
  train_visualizations/
  model_last.pth
```

归一化文件仅在对应功能启用时生成。解析后的数值还保存在 checkpoint config 和 decoder buffer 中，推理无需训练 replay。

`--output DIR` 表示精确目录，与 `--log-dir` 互斥，首次运行要求新目录。`--resume` 写回 checkpoint 所在 run，不创建新时间目录。按 epoch/step 保存的固定快照适合评估；不要在训练持续覆盖 `model_last.pth` 时同时评估该文件。

## 日志与可视化

默认启用 TensorBoard，`--log-backend none` 关闭 events，仍保存 `metrics.jsonl`。rank 0 写日志，通常为本地 microbatch 经梯度累积平均的指标，不应当作跨卡均值；关键目标指标的有效 key 归一化按训练实现单独处理。

```bash
tensorboard --logdir "$OUT/runs/train" --port 6006
```

远程使用可在本机执行 `ssh -L 6006:127.0.0.1:6006 yiwei@serv92`，再访问 `http://localhost:6006`。

```yaml
train_visualization:
  enabled: true
  interval: 500
  save_png: true
  tensorboard: true
  output_dir: train_visualizations
```

记录第 1 步和每 500 步的 rank 0 样本；`mvt1/mvt2` 为实际 coarse/refine 前向，`chunk` 为预测与实测未来。chunk 图红线为预测、绿线为有效未来 GT、白色十字为当前 TCP；世界坐标、相同有效时刻、等比例轴。PNG 与 NPZ 记录对应预测、标签、mask、当前状态和 sample ID，flow 使用独立部署采样而非带噪 GT 重建。

| 指标 | 如何解释 |
| --- | --- |
| `chunk_position_error_m` | 完整预测轨迹的平均欧氏位置误差 |
| `chunk_prefix_1/4_position_error_m` | 当前建议执行前缀的误差；4 对应默认 H |
| `chunk_stationary_error_m` | 始终保持当前 TCP 的对照误差 |
| `chunk_position_loss` | 按 `position_loss_scale` 归一化的 Smooth L1；旧配置回退到 `position_scale` |
| `chunk_reconstruction_position_error_m` | flow 的带噪监督重建误差，不能替代部署采样误差 |
| `chunk_key_condition_gate` | 显式 key 条件 gate；不能据此判断 refine 已与 key 解耦 |

loss 小不代表物理误差小，应同时检查米制误差、静止基线和轨迹。可视化会增加 CPU 传输与写盘开销，可增大 interval；不改变策略输入或训练标签。

只有 JSONL 的旧 run 可补导标量，但不能补出未保存的图像：

```bash
python tools/export_oht_tensorboard.py --run "$RUN"
```

默认输出新目录 `tensorboard-from-jsonl/`，不修改原日志或 checkpoint。先等待训练结束或使用完整 JSONL 快照，截断末行会报错。

## 离线与闭环评估

完整命令见 [README](../../finetune/OHT/README.md#4-离线评估)。先加 `--limit 8 --visualize-every 1` 检查；完整验证去掉 limit，默认每 50 个样本保存图像。`--visualize-every 0` 关闭诊断图，适合测速。采样步的推理耗时包含诊断载荷采集，图片写盘在该计时之外。

报告和图片目录必须为新路径。未指定图像目录时保存到 `<报告名>_visualizations/`。V4 的 `index.html` 展示 chunk，`sample_*/` 保存 `chunk.png/npz` 及可用的 coarse/refine 图片，报告 JSON 保存逐点误差与 key 指标。单关键点评估另有 `prediction.json` 和热图 NPZ；PNG 按每视图颜色拉伸，不能只凭颜色比较置信度。GT 只在推理后绘图，不进入策略输入。

评估阶段日志区分模型构建、权重应用、读取 replay 和逐样本预测；`--progress-interval 5` 调整提示间隔，`--quiet` 关闭。心跳不等于 GPU 正在计算。第一次推理可能建立反投影缓存，应结合 GPU 利用率、显存和后续样本速度判断。

闭环使用[原生 IsaacLab 客户端](oht-vla-server.md#isaaclab-实时-rgb-d-客户端)。默认 `--score none`，不会输出任务成功率；任务评分需真实初始化和成功条件。自定义单动作 `eval closed-loop` 工厂与协议见[服务设计](../design/oht-vla-server.md)和[动作模式指南](oht-action-modes.md)。模型预测 chunk 与 GT replay 的运行正常是两项独立检查。

## 本地验证

按改动范围选择测试；文档编辑无需重新执行完整 GPU 训练。数据快速回归：

```bash
python -m pytest -q \
  tests/test_oht_migration.py \
  tests/test_oht_object_cloud_build.py \
  tests/test_oht_replay_viewer.py
```

动作头、归一化与状态条件回归：

```bash
python -m pytest -q \
  tests/test_oht_spatial_chunk.py \
  tests/test_oht_chunk_temporal.py \
  tests/test_oht_chunk_state.py
```

测试包含 CPU 合成数据、模型/渲染接口替身和 Gloo DDP；它们验证坐标、标签隔离、梯度及保存恢复，不代表完整 CUDA 模型的精度、吞吐或 IsaacLab 闭环成绩。浏览器测试可通过 `OHT_BROWSER_EXECUTABLE` 指定 Chromium/Chrome/Edge，使用独立 profile。
