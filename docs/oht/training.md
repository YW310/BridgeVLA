# OHT 模型、训练与评估

[文档索引](../README.md) · [数据准备与契约](data.md) · [服务与执行](runtime.md)

当前主线为 **V4 → 关键目标 + 连续 EEF chunk → 离线评估 → IsaacLab**。所有命令从 BridgeVLA 仓库根目录执行，替换示例路径。数据构建/标注见 [data.md](data.md)，HTTP、客户端、GT 回放和 waypoint 执行见 [runtime.md](runtime.md)。

基于当前结果的后续研究动机、候选方向与实验门槛统一见[设计讨论](../design/role-relation-prior.md#研究动机与候选方向)；本页维护已实现模型与运行方法。

| 阅读目标 | 入口 |
| --- | --- |
| 已解决的配置、核心修改及排查经验 | [已确认基线与交接](#已确认基线与交接) |
| 状态条件、动作表示、损失与归一化 | [连续轨迹模型](#连续轨迹模型) |
| 启动、预算、续训、日志和评估 | [训练与评估](#训练与评估) |
| 旧单关键点、内部/外部对象辅助 | [单关键点与物体辅助](#单关键点与物体辅助) |
| EEF、相对 EEF、关节与相对关节头 | [单动作表示](#单动作表示) |

## 已确认基线与交接

更新日期：**2026-10-29**。用户确认 OHT 轨迹问题已解决，有效配置为 [oht_chunk_v4_regression_state_8x40.yaml](../../finetune/OHT/configs/oht_chunk_v4_regression_state_8x40.yaml)，后续以此为已确认有效的基线。

这是用户对实际运行的确认，反馈未附成功 run 路径、checkpoint 或完整定量报告，复现时应补存。结论不扩展为所有任务/样本的误差统计或闭环成功率；`delta_state`、`flow_state` 尚未获得相同确认，也没有独立消融可以将效果全部归因于下列某一项。

### 核心修改

| 修改 | 实际作用 |
| --- | --- |
| 确定性回归 | `decoder_type: regression`，直接预测 K 个未来 pose；每点相对同一次观测的 TCP，最终还原 world 绝对 EEF |
| 本体归一化与逐层注入 | train 拟合 q6，实测夹爪采用固定映射；状态进入初始动作 queries 和每层 AdaLN |
| 门控关键目标 | 相对 TCP 的预测 key 经可学习 gate 注入，初值 0.1；保留 coarse/refine 特征及预测 key 裁剪 |

回归头还继承 spatial/temporal 分支的三维 patch XYZ/有效性、未来物理时间、时序卷积、尺度分离、相邻位移和匹配 GT 的二阶差分监督。它不是旧 `displacement` 配置简单增大 loss 权重，也不是 `delta_state` 的逐步增量累积。

配置继承为 `regression_state → flow_state → flow_temporal → flow_scaled → flow → spatial → relative_goal → chunk_v4 → full → baseline`。继承文件名含 flow 不代表实际使用 flow；最终 `decoder_type=regression` 决定动作头。

此次修改沿用现有 V4，不需重建数据。K=16、建议 H=4，输出仍为 world 米/TCP/xyzw/1 开 0 关；joint 冻结视觉塔、projector 和 Gemma 前 16 层，后续层与动作头训练。当前 chunk 是 baseline，不接 role cache；导出 T/R 分割标注不等于启用物体辅助。

### 排查经验

1. 先建立可学好的确定性基线，再用相同数据、初始化来源和预算比较 flow/delta，复杂生成头不保证更准确。
2. 短期动作需直接读取实测状态；归一化 q6/夹爪、逐层条件、视觉和 key 是本次组合改动，尚未逐项归因。
3. loss 与物理误差分开：同时看完整轨迹、前 1/H 步米制误差、静止基线和同时间/同尺度轨迹图。
4. flow 带噪 GT 重建不能替代独立部署采样；回归使用相同部署预测路径。
5. 不把早期猜测写成根因：chunk 已读 coarse/refine，没有证据证明仅删除 refine 可解决；key gate 也不能切断 refine 裁剪依赖。

### 历史验证记录

实现提交为 `c8852a9`，此前文档整理为 `f9ed851`。交接记录报告实现阶段通过 312 项 CPU 回归、两组双进程 Gloo DDP 检查，并对照修改前版本检查旧配置输出兼容。这是历史记录，不表示本次文档合并重跑了这些测试。

2026-10-29 文档核对另有 30 项状态条件 CPU 回归通过；full 和三组 state 的实际预算解析也已核对。上述正确性检查、用户轨迹确认与完整 CUDA 吞吐/显存/定量/闭环验收分别记录，不互相替代。

模型入口为 [spatial_chunk.py](../../finetune/OHT/spatial_chunk.py)、[chunk_model.py](../../finetune/OHT/chunk_model.py) 和 [chunk_normalization.py](../../finetune/OHT/chunk_normalization.py)；状态条件与保存恢复回归见 [test_oht_chunk_state.py](../../tests/test_oht_chunk_state.py)。

## 连续轨迹模型

### 当前三组配置

| 配置 | 内部表示 | 生成方式 |
| --- | --- | --- |
| [regression_state](../../finetune/OHT/configs/oht_chunk_v4_regression_state_8x40.yaml) | 相对当前 TCP 的未来 pose | 确定性回归；已获用户解决确认 |
| [delta_state](../../finetune/OHT/configs/oht_chunk_v4_delta_state_8x40.yaml) | 逐步平移/旋转增量 | 回归后累积，监督完整轨迹 |
| [flow_state](../../finetune/OHT/configs/oht_chunk_v4_flow_state_8x40.yaml) | 相对当前 TCP 的未来 pose | conditional flow matching |

三组共用 coarse/refine 视觉、实测本体状态、相对 key、三维 attention、AdaLN、时序卷积与物理监督。当前只支持 `mode=baseline`、`action_mode=eef`，不接历史帧、role cache、异步执行或碰撞规划。

### 一次预测的数据流

1. 当前多视角 RGB-D/指令经 BridgeVLA，预测关键目标并产生 coarse 全局与 refine 局部特征；refine 围绕预测 key 裁剪。
2. 将视觉 patch 对应的可见表面 XYZ、有效性、view/scale 标识送入 decoder。
3. 当前 TCP 定义局部坐标，归一化 q6/夹爪与门控 key 注入动作 block。
4. 预测 K 个未来动作并还原 world EEF；客户端按实际 dt 消费前 H 项，执行规则见 [runtime.md](runtime.md)。

key 只是方向条件，可以位于 K 步窗口之外，不要求 chunk 末点到达它。第一点监督真实 `t+dt`，不强制等于当前 TCP。V4 标签、padding、缺帧及 no-key 尾段契约见 [data.md](data.md)。

### 本体状态条件与增量回归

`state_conditioning` 将当前 TCP 局部状态投影为 `s`，相对预测 key 投影为 `g`：

```text
c = time_embedding(tau) + s + sigmoid(key_gate_logit) * g
```

`c` 同时加入初始 queries 和各层 AdaLN 的 self-attention、cross-attention、FFN 调制。AdaLN 残差 gate 零初始化时，状态仍能通过初始 queries 影响输出。回归 tau=1；flow tau 是生成时间，与未来物理时间 `k*action_dt` 不同。

保留一个始终有效的状态 memory token；全部视觉 patch 无效时仍可读状态。显式 key 只有 gate 这条路径，不另作未门控 memory/XYZ token；gate 初值 0.1，记录为 `chunk_key_condition_gate`。refine 仍依赖 key 裁剪，gate 接近 0 不等于网络完全不使用 key。

q6 使用 train 1%/99% 分位数的中点/半区间归一化，scale 下限 0.1 rad；实测夹爪固定 center=0.5、scale=0.5。不裁剪分位区间外值，也不改变关节角分支；原始 TCP 状态保留用于坐标恢复。

### 几何与空间 attention

点云渲染在原 RGB 后附加 XYZ/覆盖通道，共享 z-buffer、splat 和抗锯齿，不重复渲染或修改 RGB/correlation/depth。patch 池化预乘 XYZ 与覆盖再相除，无有效表面则屏蔽；world 原点真实表面仍有效，token 顺序为 view→row→column。

coarse 撤销全局 cube 变换；refine 先撤销实际预测 crop，再撤销 cube。世界米坐标转为 `R_0ᵀ(x_world-p_0)`，按 `geometry_scale=0.25 m` 缩放。这统一 decoder 几何定义，不使上游 VLM 成为严格 SE(3) 等变模型。

各 attention head 从相对方向和对数距离生成三维 bias，视觉带 view/scale embedding，queries 带未来物理时间。动作与几何归一化尺度不同，位置比较乘 `action_position_scale/geometry_scale`；delta 先累积平移再比较。

### 动作表示

内部 pose 为 K×9（平移 3、连续旋转 6），另预测夹爪 logit。六维旋转正交化恢复 SO(3)，转单位 xyzw 四元数；推理拒绝退化旋转、非有限或非法动作。

| 表示 | 平移编码 | 旋转编码 |
| --- | --- | --- |
| `relative_pose` | `R_0ᵀ(p_k-p_0)/scale` | `R_0ᵀR_k` 的前两列 |
| `delta_pose` | `R_0ᵀ(p_k-p_(k-1))/scale` | `D_k=R_(k-1)ᵀR_k` 的前两列 |

relative 的各点相对同一当前 TCP；delta 平移轴仍为初始 TCP，恢复为 `p_k=p_0+R_0*scale*sum(d_1...d_k)`，旋转为 `R_k=R_0D_1...D_k`。不能直接累加 Euler/旋转向量。delta 当前只支持 spatial regression；公共输出始终是 `[x,y,z,qx,qy,qz,qw,open01]` 绝对 world TCP。

### 训练目标

joint 优化 `key_loss_weight*key_loss + chunk_loss`，key 默认权重 1。三组的物理监督作用于恢复后的完整世界轨迹：

| 项 | 权重/尺度 | 监督对象 |
| --- | --- | --- |
| 位置 Smooth L1 | 5 / 0.01 m | 各有效未来点 |
| 旋转 MSE | 1 | 世界旋转矩阵；四元数符号等价 |
| 夹爪 BCE | 1 | 对应未来 tick 二值状态 |
| 相邻位移 Smooth L1 | 1 / 0.01 m | 首段从实测 TCP 开始，后续连续有效帧 |
| 二阶差分 Smooth L1 | 0.1 / 0.001 m | 匹配 GT 位移变化，保留真实转弯 |

mask 屏蔽无效标签，位移/二阶差分不跨 padding 或缺失点。二阶差分单位为米/控制步²，不是 m/s²，也不惩罚曲率本身。输出前有 kernel=3 的 depthwise 时序卷积，残差零初始化。

flow 用高斯噪声 epsilon 和 tau 构造 `A_tau=(1-tau)*epsilon+tau*A_gt`，拟合 `v_theta=A_gt-epsilon`；平移/旋转 MSE 按上述权重加权，再乘 `flow_weight=1`。物理辅助监督作用于 `A_hat=A_tau+(1-tau)*v_theta`，向量场速度不是机器人速度。

当前 tau 为 `0.999*(1-U**(1/1.5))`，偏向高噪声端；推理 8 次 Euler 更新后另调用一次 field 计算干净端点夹爪，共 9 次 field。默认 seed 0 在每次请求生成同一初始噪声并扩展到 batch，使同条件结果不随 batch 大小改变，不修改全局 RNG；null 才逐样本随机采样。回归不计算 flow loss/积分。

### joint 与 frozen_key

| 阶段 | 关键目标网络 | chunk 条件与监督 |
| --- | --- | --- |
| `joint`，当前默认 | 兼容参数联合训练，保留独立 key loss | 重新走预测 key/crop 路径，chunk loss 可更新未冻结视觉特征 |
| `frozen_key` | 从已训练 EEF checkpoint 加载并整体冻结 | inference-equivalent 特征提取，只有新 decoder 训练 |

joint 的 key 监督分支只选 `key_valid=true` 子批，沿用 GT crop/rotation conditioning；chunk 分支重新从当前观测预测 key，不共享 GT 选出的局部特征。GT waypoint、未来状态和 key_valid 不进入 chunk 条件；no-key 尾段只算 chunk，key loss 按全卡有效 key 数归一化，允许某 rank 没有 key。

两条路径位于同一外层 DDP forward，最后一次 backward/step。预测条件当前逐样本复用部署 `act`，并非整个 batch 仅两次 VLM，也尚未批量优化 backbone。冻结模块保持 eval、BatchNorm 运行统计固定，支持 singleton key 子批。

预测 key 的离散 pose/gripper 与 crop 位置 detach，chunk 可沿预测视觉特征反传，frozen_key 则全部 detach。关键目标 GT 仅独立监督，不能将 GT-crop 训练特征直接接入 chunk 后声称条件与部署一致。

### 归一化与 checkpoint

rank 0 按 train 样本索引等间隔选最多 4096 个样本，广播统计或错误；不是任务/episode 均衡随机采样，val/test 与无效 padding 不参与。

| 用途 | 设置/产物 |
| --- | --- |
| 动作平移编码、flow 位置噪声 | `action_position_scale: train_quantile`；当前 TCP 到有效未来点半径 99% 分位数，下限 0.01 m |
| attention 的 scene/key/query 坐标 | `geometry_scale: 0.25` |
| 世界位置 loss 容差 | `position_loss_scale: 0.01` |
| 动作统计 | `chunk_normalization.json`、`config.chunk_normalization` |
| q6/夹爪统计 | `proprio_normalization.json`、`config.chunk_proprio_normalization`，同时为 decoder buffer |

delta 也统计完整未来位移，不改成单步 delta 尺度。动作统计读小型 targets NPZ 与哈希绑定 current_tcp；本体统计只解压 q6/开度，但 checksum 仍读完整文件，启动打印进度。数值存入 checkpoint，部署无需 replay 或重新拟合。旧配置未分离尺度时回退 `position_scale=0.25`。

### 旧配置与消融

| 配置（均位于 `finetune/OHT/configs/`） | 主要变化 |
| --- | --- |
| `oht_chunk_v4_8x40.yaml` | 基础 token Transformer，位置权重 5，无 spatial/state 条件 |
| `oht_chunk_v4_displacement_8x40.yaml` | 基础配置加相邻位移监督 |
| `oht_chunk_v4_relative_goal_8x40.yaml` | key 改为相对 TCP，不自动叠加位移 loss |
| `oht_chunk_v4_spatial_8x40.yaml` | 相对 key、三维 attention、位移监督、回归头 |
| `oht_chunk_v4_flow_8x40.yaml` | spatial 加 flow matching |
| `oht_chunk_v4_flow_scaled_8x40.yaml` | 分离动作/几何/loss 尺度，高噪声 tau |
| `oht_chunk_v4_flow_temporal_8x40.yaml` | scaled 加时间 AdaLN、时序卷积、二阶差分 |

flow_state 再加本体归一化、逐层状态与 key gate，regression_state/delta_state 从这里分出。旧 checkpoint 仍按保存配置解释。基础 token decoder 与 spatial/state 坐标不同，不能混称相同结构。

对照固定数据划分、输入、初始化、seed、预算和执行器；K=1/H=1、K=16/H=1 与 H=4 分开测。删显式 key embedding 仍保留 key-crop，不是“全部无 key”；若检验全部 key 条件，应共用全局特征/裁剪规则。GT-key chunk 单列 privileged 上界，不以辅助 loss 或热图观感证明闭环收益。

时间采样参考 [pi0](https://arxiv.org/abs/2410.24164)，相关方法另见 [ChainedDiffuser](https://proceedings.mlr.press/v229/xian23a.html)、[3D Diffuser Actor](https://3d-diffuser-actor.github.io/)。时序卷积、物理差分与尺度分离为本仓库设计，不是原样复现。

## 训练与评估

### 环境与路径

使用已安装 PyTorch/CUDA、Transformers、PaliGemma 与 point-renderer 的 BridgeVLA 环境，见[安装说明](../guides/installation.md)。附加依赖：

```bash
python -m pip install -r finetune/OHT/requirements-model.txt
python -m pip install --no-build-isolation -e finetune/bridgevla/libs/point-renderer
```

本页独立设置变量，已有完整 V4 可直接训练：

```bash
ROOT=/common-data-32t/data/robot_data/oht_curobo_pd_v423-seg
OUT="$ROOT/bridgevla-seg-v4-run01"
PRETRAIN=/home/yiwei/project/BridgeVLA/checkpoints/pretrain
CONFIG=finetune/OHT/configs/oht_chunk_v4_regression_state_8x40.yaml
EXP=oht-seg-v4-regression-state-s0
```

`PRETRAIN` 是热图预训练目录，PaliGemma 模型/processor 仍需可访问或已完整缓存；离线缓存齐备时可设置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。数据未构建时先按 [data.md](data.md)准备，不手工补 contract。

### 启动命令

```bash
GPUS_PER_NODE=2 bash finetune/OHT/train_8x40.sh \
  --config "$CONFIG" --replay "$OUT/replay-source-v4" \
  --pretrain-path "$PRETRAIN" \
  --log-dir "$OUT/runs" --exp-id "$EXP"
```

delta/flow 对照分别改为 `oht_chunk_v4_delta_state_8x40.yaml`、`oht_chunk_v4_flow_state_8x40.yaml`，并用新 EXP；初始化来源和 seed 保持一致。joint 不要求旧 OHT checkpoint，复用 encoder 时另加 `--init-checkpoint /path/to/model_last.pth`。

### 预算与卡数

| 当前三组共用设置 | 值 |
| --- | --- |
| `epochs` / `train_iter` | 60 / 40000 |
| `global_batch_size` / `batch_size` | 16 / 每卡 8 |
| `max_optimizer_steps` | 83300 |
| `checkpoint_every_epochs` | 10 |
| 优化器/学习率/weight decay | AdamW / `8e-5` / `1e-6`，固定学习率 |
| 冻结 | vision tower、projector、Gemma 前 16 层；lm_head/embed_tokens 沿用冻结 |
| 全局刚体增强 | 关闭；恢复前须定义机器人基座/关节等变契约 |

`train_iter` 是任务均衡、有放回的采样预算，不是数据集长度。一轮 `floor(40000/16)=2500` 优化步，实际目标 `min(60×2500,83300)=83300`，约 **33.32 个采样 epoch**；60 是配置目标，不是已完成轮数。跑满 60 轮需在新配置中将上限设为 0 或至少 150000。

`accumulation_steps: null` 自动计算：两卡累积 1、单卡 2；四卡需每卡 batch 4，八卡需 2，满足全局 batch 可被 `每卡 batch×卡数` 整除。显存需实测，严格 resume 不能更换卡数/batch。

脚本默认八卡，示例显式设两卡；名称 `8x40` 不自动调整 batch/显存。torchrun `--standalone` 自动分配 rendezvous endpoint，不能依赖 `MASTER_PORT` 固定它。

### 初始化与续训

| 操作 | 参数 | 恢复内容 |
| --- | --- | --- |
| 基础 joint | `--pretrain-path` 可指定热图预训练 | 不要求 OHT checkpoint，仍需 PaliGemma |
| 新实验初始化 | `--init-checkpoint /path/to/model_last.pth` | 兼容权重，optimizer/采样重新开始 |
| 原实验续训 | `--resume /path/to/model_last.pth` | 模型、optimizer、已完成采样位置 |

结构、状态/本体归一化或动作尺度变化时 init 新 run，保留 encoder、重建 decoder；仅允许的 loss/采样设置变化可保留 decoder，边界由 `load_chunk_weights()` 检查。单关键点 checkpoint 只初始化匹配 data profiles/标定的 encoder，不能 resume 为 chunk。frozen_key 必须加载已训练网络。

```bash
RUN=/path/to/actual/run
GPUS_PER_NODE=2 bash finetune/OHT/train_8x40.sh \
  --config "$CONFIG" --replay "$OUT/replay-source-v4" \
  --pretrain-path "$PRETRAIN" --resume "$RUN/model_last.pth"
```

严格 resume 校验解析后配置、replay/role cache 契约、卡数、学习率、batch、冻结、动作和 seed。允许调预算、保存周期、num_workers 与独立可视化；epoch 的 `epochs/max_optimizer_steps/checkpoint_every_epochs` 与 step 的 `optimizer_steps/checkpoint_interval` 两套预算不同时设置。

采样器按完成 step 跳过抽样位置；Python/NumPy/torch 的完整 RNG 状态未恢复，随机增强/采样不承诺逐位复现。续训写回原 run，不创建新时间目录。

### 预检与短测

完整 replay 校验检查所有非空划分，训练预检仅检查 train，不能代替前者：

```bash
python tools/validate_oht_replay.py --replay "$OUT/replay-source-v4" --workers 4
WORLD_SIZE=2 python -m finetune.OHT.train \
  --config "$CONFIG" --replay "$OUT/replay-source-v4" --validate-only
```

预检不构建模型/DDP 或输出目录，但归一化会读取 train NPZ。未设 WORLD_SIZE 时通常按一张卡计算；八卡仍配每卡 8 会因全局 batch 16 不可整除而失败，CPU 数据检查不证明 GPU 配置正确。

短测新建 `finetune/OHT/configs/my_chunk_smoke.yaml`：

```yaml
extends: oht_chunk_v4_regression_state_8x40.yaml
epochs: 1
max_optimizer_steps: 2
```

用此配置、新 EXP 启动同一训练命令。现有 `smoke.yaml` 属旧单关键点，不能直接读 V4。

### 输出目录

`--log-dir ROOT` 创建 `ROOT/train/<exp_id>/<exp_note>/MM_DD_HH_MM/`，空备注省略，Asia/Shanghai 时间，同分钟追加后缀。rank 0 创建/广播并打印 `Run directory: ...`，以该路径为准。

目录保存 `run.json/config.json/contract.json/training_plan.json`、启用时的两个 normalization JSON、`metrics.jsonl`、`tensorboard/`、`train_visualizations/` 与 checkpoints。`--output DIR` 使用精确新目录，和 `--log-dir` 互斥；resume 回原目录。评估使用不再覆盖的固定快照，避免一边训练覆盖 `model_last.pth` 一边读它。

### 日志与可视化

TensorBoard 默认启用但需安装 `tensorboard`，model requirements 未列入；`--log-backend none` 关闭 events，JSONL 仍保存。

```bash
python -m pip install tensorboard
tensorboard --logdir "$OUT/runs/train" --port 6006
```

rank 0 写日志，通常是本地 microbatch/累积平均，不是跨卡均值；key 指标另按有效 key 归一化。`train_visualization` 控制 enabled/interval/save_png/tensorboard/output_dir，记录第 1 步和间隔步；可视化增加传输/写盘成本，可增大 interval。

`mvt1/mvt2` 对应实际 coarse/refine；chunk 图红色预测、绿色有效未来、白色当前 TCP，采用世界坐标、相同时间和等比例轴。PNG/NPZ 含预测、标签、mask、状态和 sample ID；GT 只监督/绘图，不进入部署。flow 复用观测 memory 做 eval/no-grad 独立部署采样，不重跑 VLM。

| 指标 | 解读 |
| --- | --- |
| `chunk_position_error_m` | 完整部署预测的平均位置误差 |
| `chunk_prefix_1_position_error_m` / `chunk_prefix_4_position_error_m` | 前 1/默认 H 项误差 |
| `chunk_stationary_error_m` | 始终保持当前 TCP 的对照 |
| `chunk_displacement_error_m` / `chunk_second_difference_error_m` | GT 位移/位移变化匹配误差 |
| `chunk_position_loss` | 按 position_loss_scale 归一化的 Smooth L1，不是米 |
| `chunk_reconstruction_position_error_m` | flow 带噪重建，不能代替部署采样 |
| `chunk_flow_loss` / `chunk_flow_weighted_loss` | 向量场原始/加权项；tau 三档 count/position loss 需按有效 count 汇总 |
| `chunk_key_condition_gate` | 显式 key gate，不代表整个 refine 已与 key 解耦 |

`chunk_sampled_position_loss`、`chunk_sampled_displacement_loss` 仅作诊断不反传。JSONL 旧 run 可用 `python tools/export_oht_tensorboard.py --run "$RUN"` 导到新 `tensorboard-from-jsonl/`，不能补出未保存图片；用完整日志或快照，截断末行会报错。

### 离线与闭环评估

```bash
RUN=/path/to/actual/run
python -u -m finetune.OHT.eval open-loop \
  --checkpoint "$RUN/model_last.pth" \
  --replay "$OUT/replay-source-v4" --split val \
  --output "$OUT/eval/regression-state-val.json"
```

先检查少量样本可加 `--limit 8 --visualize-every 1`，正式去掉 limit；默认每 50 样本保存图，0 关闭。报告/图片用新路径，默认图目录为 `<报告名>_visualizations/`。V4 `index.html/sample_*/chunk.png/npz` 展示轨迹及可用 coarse/refine 图，JSON 按未来 offset 统计误差、有效数与 key 指标。

PNG 热图逐视图拉伸，不能用颜色比较置信度。采样计时含诊断载荷采集、图片写盘在外；`--progress-interval 5` 调心跳，`--quiet` 关闭。首次推理可能构建反投影缓存，结合后续速度和 GPU 指标判断，不以心跳证明 GPU 正在计算。

离线误差不等于闭环成功率；闭环、模型/GT 回放、任务评分与自定义工厂见 [runtime.md](runtime.md)。原生客户端默认 `--score none` 不报成功率，实际成绩须有真实初始化、成功条件和完整 cases，失败保留分母。

### 本地验证

按变化范围选测试；本次文档合并没有重跑 GPU 或历史整组测试。动作、归一化和状态回归入口：

```bash
python -m pytest -q \
  tests/test_oht_spatial_chunk.py \
  tests/test_oht_chunk_temporal.py \
  tests/test_oht_chunk_state.py
```

CPU 合成数据、renderer/VLM 替身和 Gloo DDP 检查坐标、mask、梯度、GT 隔离与保存恢复，不代表完整 CUDA 精度/吞吐或 IsaacLab 成绩。数据与浏览器回归入口见 [data.md](data.md)。

## 单关键点与物体辅助

以下为旧单目标路线，不接受 V4 chunk replay。常用目录 `replay-source-v3` 的实际 schema 是 `oht_bridgevla_v2`；不要改 contract 混用。数据/teacher 构建统一见[单关键点数据](data.md#单关键点数据)。

| 路线 | 配置 | 额外来源 |
| --- | --- | --- |
| 单关键点 EEF | [oht_full_8x40.yaml](../../finetune/OHT/configs/oht_full_8x40.yaml) | 无对象输入 |
| 内部 T/R | [role_queries.yaml](../../finetune/OHT/configs/role_queries.yaml) | role teacher 仅训练监督 |
| 外部预测 | [predicted_external.yaml](../../finetune/OHT/configs/predicted_external.yaml) | 预测缓存及真实 predictor |

### 单关键点训练

```bash
OHT_REPLAY=/common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3
PRETRAIN=/home/yiwei/project/BridgeVLA/checkpoints/pretrain
GPUS_PER_NODE=2 bash finetune/OHT/train_8x40.sh \
  --config finetune/OHT/configs/oht_full_8x40.yaml \
  --replay "$OHT_REPLAY" --pretrain-path "$PRETRAIN" \
  --log-dir exp/OHT --exp-id oht-keypoint-eef-s0
```

full 同为 60/40000/batch16/83300 上限，实际约 33.32 采样轮；冻结视觉塔、projector、Gemma 前 **6** 层，启用 scene/EEF 联合变换增强。它与 chunk 前 16 层/禁用全局增强不同。`baseline.yaml` 为早期 step 预算，`smoke.yaml` 为两步 GPU 短测，不是 full 配方。

### 内部角色预测

两个固定 T/R queries 从当前渲染特征/任务上下文预测角色，coarse/refine 继承角色与几何。teacher 只监督 maps、Reference NULL 和可见表面 T/R 分离，推理不读 GT 对象；角色顺序不是持久实例 ID。

`role_loss.diversity/refine_diversity` 各 0.01，后者只在 `object.refine_role_masks: true` 的局部 head 生效，该 head 默认关闭。只在可靠、可见、XYZ 一致且 footprint 不重叠的另一角色表面抑制响应，不把 NULL/unknown/遮挡当背景；允许角色共享对象区域时关闭分离或提供互斥标签。[O2 分离边界](../guides/object-conditioning.md#tr-分离监督与日志)

```bash
ROLE_CACHE=/path/to/role-teacher
python -m finetune.OHT.train \
  --config finetune/OHT/configs/role_queries.yaml \
  --replay "$OHT_REPLAY" --role-cache "$ROLE_CACHE" \
  --output /path/to/new/role-queries-s0 \
  --init-checkpoint /path/to/trained/baseline/model_last.pth
```

查看 `object_slot_diversity_loss/object_refine_diversity_loss` 及 `*_diversity_supervised_pixels`，像素数 0 时 loss 为 0 是正常跳过。有效 replay/teacher 不重建；新 loss 需新训练才影响预测。baseline 继续同样优化预算，或两组同预训练各训同预算，3 seeds 配对；不能把继续训练收益全部算作物体辅助。

### 外部物体预测

仓库有严格缓存/在线 wrapper，未附新的检测模型/感知权重。应用提供无参 `module:factory`，返回 callable：

```python
from finetune.OHT.data.role_cache import role_fields

class Predictor:
    def reset(self):
        pass

    def __call__(self, observation, goal):
        # 用当前观测计算 points/valid/present/confidence，不读 GT 或未来。
        return role_fields("predicted", points, valid, present,
                           confidence=confidence)
```

observation 只含当前 low_dim、每相机 RGB/depth/world XYZ/K/外参；points 为有限 float32 `[2,512,3]`，valid/present 为 bool `[2]`，confidence 为 `[2]`、范围 [0,1]，顺序 T/R。缺几何设 valid=false，遮挡不改成语义 absence；必需 Reference 不可用时当前门控关闭整对辅助。

provenance YAML 至少含真实权重 `model_sha256`、`training_split`；工具附加 factory 并写缓存/checkpoint。这是调用方声明，不独立证明无测试集污染；OHT 感知训练只用 train。

```bash
PREDICTIONS=/path/to/new/predictions-v1
PROVENANCE=/path/to/predictor-provenance.yaml
python tools/predict_oht_objects.py \
  --replay "$OHT_REPLAY" --predictor my_oht_predictor:create \
  --provenance "$PROVENANCE" --output "$PREDICTIONS"
python -m finetune.OHT.train \
  --config finetune/OHT/configs/predicted_external.yaml \
  --replay "$OHT_REPLAY" --role-cache "$PREDICTIONS" \
  --output /path/to/new/predicted-external-s0 \
  --init-checkpoint /path/to/trained/baseline/model_last.pth
```

缓存覆盖对应 train/val 集合。离线按 replay 帧序列、在线按控制采样频率调用，episode 开始 reset；有历史 predictor 须对齐采样策略。在线 factory/provenance 必须与训练缓存一致。

单关键点 open-loop 使用上方 eval 命令但 replay 换为 `$OHT_REPLAY`；外部模式还加 `--predictor my_oht_predictor:create --predictor-provenance "$PROVENANCE"`，由真实 predictor 生成当前角色，不将 teacher 输入策略。部署见 [runtime.md](runtime.md)。

## 单动作表示

`action_mode` 与对象辅助 mode 独立，baseline/role_queries/predicted_external 支持下列四种单目标头；**V4 chunk 仅支持绝对 EEF**，不能扩展解释为四种 chunk。

### 坐标与标签

| 模式 | 动作维度/内容 | 标签 |
| --- | --- | --- |
| `eef` | 8：world TCP xyz/xyzw/grip | 未来关键点实测 TCP |
| `relative_eef` | 7：delta_xyz/rotation_vector/grip | 当前实测 TCP 到同一未来目标 |
| `joint` | 7：q1…q6/grip，rad | 未来关键点实测关节 |
| `relative_joint` | 7：delta_q1…q6/grip，rad | 同一未来关节减当前关节 |

夹爪均为 0 关 1 开，raw action 前六维不参与标签。相对量是到未来 key 的目标位移/转角，不是速度或必然一个控制步的增量。relative_eef 默认 world 平移、当前 TCP body 旋转；body 为 `R_currentᵀ(p_target-p_current)`、`log(R_currentᵀR_target)`，world/base 用相应左乘误差，不减四元数分量。base 必须显式给固定 `world_from_base`。

joint 顺序为记录 `observation.state[:6]`，joint_names 声明但不重排训练数据，执行映射见 [runtime.md](runtime.md)。默认关节直接相减；仅确认连续关节才设置 joint_wrap，不能和 joint_limits 同用。限位配置使训练拒绝越界 GT、解码限制预测，不替代仿真独立限位检查。

prepared 输入中 eef 沿用旧行为，relative_eef 加当前 tcp_pose，joint/relative_joint 加当前 joint_positions；未来 joint_action 只作标签。raw HTTP 各模式仍需实测关节/EE/夹爪供公共预处理与执行，不能据 prepared 白名单删 HTTP 字段。缺当前状态/未来关节的旧缓存须重建，不能填零；数据迁移见 [data.md](data.md)。

### 向量动作头与损失

eef 用原 heatmap、旋转分类和夹爪头。其余模式用物体辅助后的 global/local 特征和当前状态回归六维量+夹爪 logit，优化 Smooth L1、夹爪 BCE、正权重辅助 EEF translation 及已启用角色 loss；不叠加旧旋转/夹爪分类训练项。日志看 `vector_action_loss`、`vector_gripper_loss`、`eef_aux_translation_loss`，关节图中的定位热图是辅助 EEF，不是关节预测。

`mvt.final_dim=2048` 对齐 VLM global/local 特征；旧 64 分类头尺寸不能直接套用。向量模式冻结未用分类头，初始化允许其尺寸不同，EEF 仍严格校验。

### 动作配置与初始化

现有 `relative_eef.yaml/joint.yaml/relative_joint.yaml` 对应三种向量模式，EEF 默认 `baseline.yaml`。如组合 full/角色路线，在 configs 目录新建继承配置，例如：

```yaml
extends: oht_full_8x40.yaml
action_mode: relative_joint
agent:
  transform_augmentation: false
```

相对继承路径按该 YAML 目录解析。向量模式必须关闭全局 transform_augmentation；action 的 hidden_dim 默认 256，translation/rotation/joint scale 默认 1，auxiliary_translation_weight 默认 1 且需大于 0。这些 scale 是训练归一化，输出还原物理单位，不是控制器 gain。adapter_only 时新动作头仍训练。

checkpoint 保存完整动作配置，strict load/resume 拒绝不匹配；显式 init 可迁移骨干并在动作设置改变时重建回归头，必须重训。不能只改 EEF 配置就部署为 joint；预算/卡数/RNG 限制沿用[初始化与续训](#初始化与续训)。完整 CUDA/IsaacLab 四模式仍需现场验收。
