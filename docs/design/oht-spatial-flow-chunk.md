# OHT 连续轨迹模型：状态条件、回归与 flow

本文说明当前 V4 chunk 模型，运行命令统一见 [OHT README](../../finetune/OHT/README.md#3-训练)。最初的轻量 Transformer 和数据/执行契约见[基础设计](key-goal-action-chunk-baseline.md)。

## 当前三组配置

| 配置 | 内部动作表示 | 生成方式 |
| --- | --- | --- |
| [regression_state](../../finetune/OHT/configs/oht_chunk_v4_regression_state_8x40.yaml) | 相对当前 TCP 的未来 pose | 确定性回归，作为首个对照 |
| [delta_state](../../finetune/OHT/configs/oht_chunk_v4_delta_state_8x40.yaml) | 逐步平移/旋转增量 | 确定性回归，累积后监督 |
| [flow_state](../../finetune/OHT/configs/oht_chunk_v4_flow_state_8x40.yaml) | 相对当前 TCP 的未来 pose | conditional flow matching |

三组共用 V4 数据、coarse/refine 视觉特征、实测本体状态、相对关键目标、三维 attention、AdaLN、时序卷积和物理轨迹监督。默认 K=16、建议执行 H=4；最终都还原为 world 米、TCP/xyzw、1 开 0 关的 EEF 动作。当前不加入历史帧、异步执行或碰撞规划；V4 chunk 不接入 role cache。

冻结视觉塔、multimodal projector、Gemma 前 16 层，联合训练其后续层、关键目标头与 chunk decoder。Gemma 不是全部冻结；`oht_full_8x40.yaml` 的独立单关键点默认值为 6，chunk 配置覆盖为 16。

## 一次预测的数据流

1. 当前多视角 RGB-D 与指令经 BridgeVLA，产生关键目标和 coarse 全局、refine 局部特征；refine 仍围绕预测关键目标裁剪。
2. 将每个视觉 patch 对应的当前可见表面 XYZ、有效性、view/scale 标识送入 decoder。
3. 实测 TCP 定义局部坐标系；归一化关节/夹爪状态和门控关键目标直接注入每层动作 block。
4. 一次预测 K 个未来动作，恢复世界 EEF；客户端按实际 dt 消费前 H 项，再采集新观测。

关键目标只提供条件，不强制 chunk 末点到达它；目标可能位于 K 步窗口之外。第一未来点监督真实 `t+dt`，不强制等于当前 TCP。

## 本体状态条件与增量回归

开启 `state_conditioning` 后，当前 TCP 局部坐标下的状态投影为 `s`，相对预测关键目标投影为 `g`。条件为：

```text
c = time_embedding(tau) + s + sigmoid(key_gate_logit) * g
```

`c` 加入初始动作 queries，并送入每层 AdaLN 的 self-attention、cross-attention 和 FFN 调制。即使 AdaLN 残差 gate 零初始化，状态也可通过 queries 直接影响输出。回归时 tau 固定为 1；flow 时 tau 是生成过程时间，与未来物理时间 `k * action_dt` 不同。

保留一个始终有效的状态 memory token，保证全部视觉 patch 无效时仍有可用输入。显式关键目标只通过 gate 注入，不再额外走未门控的 memory/XYZ 路径。gate 初值 0.1，日志为 `chunk_key_condition_gate`；refine 特征仍依赖 key 裁剪，不能据 gate 接近 0 声称整个网络已与 key 解耦。

关节 q6 按训练集 1%/99% 分位数的中点和半区间归一化，scale 下限 0.1 rad；夹爪实测开度固定 center=0.5、scale=0.5。不裁剪区间外数值，不改变关节角分支。网络使用局部 TCP 坐标，原始状态仍用于坐标恢复。

## 几何与空间 attention

原有点云渲染特征后附加当前渲染帧 XYZ 和覆盖通道，共享 RGB 的 z-buffer、splat 和抗锯齿，不重复渲染，也不改 RGB/correlation/depth 通道。对 VLM patch 分别池化预乘 XYZ 与覆盖再相除；无有效表面的 patch 被屏蔽，世界原点处真实表面仍有效。token 顺序保持 view → patch row → patch column。

coarse 撤销全局 cube 变换；refine 先撤销实际预测裁剪，再撤销全局 cube 变换，得到世界米坐标。统一转为当前 TCP 坐标 `R_0ᵀ(x_world-p_0)`，按 `geometry_scale=0.25 m` 缩放。上游 VLM 不是严格 SE(3) 等变，这里只统一 decoder 几何和动作的坐标定义。

每个 attention head 根据相对方向和对数距离生成三维 bias。视觉 tokens 带 view/scale embedding，动作 queries 带未来物理时间。动作平移的编码尺度与几何尺度不同，比较位置前需乘 `action_position_scale/geometry_scale`；delta 参数化先累积平移再比较。

## 动作表示

pose 内部为 K×9：三维平移和六维旋转，另预测夹爪 logit。六维旋转经正交化恢复 SO(3)，协议中转换为单位四元数。

| 表示 | 平移 | 旋转 |
| --- | --- | --- |
| `relative_pose` | `R_0ᵀ(p_k-p_0)/scale` | `R_0ᵀ R_k` 的前两列 |
| `delta_pose` | `R_0ᵀ(p_k-p_{k-1})/scale` | `D_k=R_{k-1}ᵀR_k` 的前两列 |

relative_pose 各未来点相对同一个当前 TCP；delta_pose 使用初始 TCP 朝向的平移轴，恢复为 `p_k=p_0+R_0*scale*sum(d_1...d_k)`。旋转按 `R_k=R_0 D_1...D_k` 连乘，不能直接相加 Euler 角或旋转向量。delta 当前仅支持 spatial regression。

位置、旋转、相邻位移和二阶差分 loss 都作用于恢复后的完整世界轨迹，前面增量误差会影响后续位置监督；不是只拟合很小的单步增量。公共输出、V4 标签和客户端执行方式不随参数化改变。

## 训练目标

joint 为 `key_loss_weight * key_loss + chunk_loss`，关键目标权重默认 1。GT 关键目标仅进入独立监督分支；chunk 重新走预测关键目标及预测裁剪路径。两分支共享权重，不共享 GT 选出的局部特征。离散 key 解码和裁剪位置 detach，chunk loss 可沿视觉特征更新未冻结的 backbone。frozen_key 则冻结整个关键目标网络；完整训练路由见[基础设计](key-goal-action-chunk-baseline.md#42-gt-裁剪与训练部署一致性)。

当前三组共用的物理监督为：

| 项 | 权重 / 尺度 | 监督对象 |
| --- | --- | --- |
| 位置 Smooth L1 | 5 / 0.01 m | 各有效未来点的世界位置 |
| 旋转 MSE | 1 | 恢复后的世界旋转矩阵 |
| 夹爪 BCE | 1 | 对应未来 tick 的二值夹爪 |
| 相邻位移 Smooth L1 | 1 / 0.01 m | 首段从实测 TCP 出发，后续比较连续有效帧 |
| 二阶差分 Smooth L1 | 0.1 / 0.001 m | 匹配 GT 位移变化，保留真实转弯 |

只监督有效标签；相邻位移和二阶差分不跨 padding 或缺失点。二阶差分量是米/控制步²，不是 m/s²，也不惩罚曲率本身。输出前增加 kernel=3 的 depthwise 时序卷积，残差支路零初始化。

flow 额外采样高斯噪声 epsilon 与 tau，构造 `A_tau=(1-tau)*epsilon+tau*A_gt`，拟合 `v_theta=A_gt-epsilon`；平移/旋转 MSE 按各自权重加权，再乘 `flow_weight=1`。物理辅助监督作用于重建 `A_hat=A_tau+(1-tau)*v_theta`。flow 向量场是生成过程速度，不是机器人运动速度。

当前 flow 的 tau 使用 `0.999*(1-U**(1/1.5))`，偏向 tau=0 高噪声端；旧 flow 默认均匀采样。推理从噪声执行 8 次 Euler 积分，干净端点计算夹爪。默认 seed 0 保持同条件样本固定、跨 batch 大小一致，不改变全局 RNG；设 null 才随机采样。回归配置虽继承 flow 配置中的时间采样字段，但不计算 flow loss 或积分。

## 归一化与 checkpoint

rank 0 只从 train 中均匀选取最多 4096 个样本，计算并广播统计或错误。val/test 和无效 padding 不参与。三组共享动作尺度定义：当前 TCP 到有效未来点的位移半径 99% 分位数，下限 0.01 m；不会因改成 delta 参数化就换统计对象。

| 用途 | 配置或产物 |
| --- | --- |
| 动作平移编码与 flow 位置噪声 | `action_position_scale: train_quantile`，启动解析为数值 |
| 场景/key/query 的 attention 坐标 | `geometry_scale: 0.25` |
| 世界位置 loss 容差 | `position_loss_scale: 0.01` |
| 动作统计 | `chunk_normalization.json`、`config.chunk_normalization` |
| 关节/夹爪统计 | `proprio_normalization.json`、`config.chunk_proprio_normalization` |

动作统计读取小型 targets NPZ 和哈希绑定的当前 TCP 索引。本体统计只解压 observation NPZ 的 q6/开度，但 checksum 校验仍读取完整文件；启动会打印进度。数值写入 checkpoint config；本体参数同时注册为 decoder buffer，部署无需 replay。旧配置未指定分离尺度时回退到 `position_scale=0.25`。

改变结构、动作尺度或本体归一化，用 `--init-checkpoint` 新建实验并重新初始化 decoder/optimizer；原配置可严格 resume。仅 loss/采样设置变化的兼容迁移由加载器检查，见[训练说明](../guides/oht-training.md#初始化与续训)。现有 V4 无需重建。

## 指标与可视化

flow 的带噪重建训练损失不能用于展示部署性能。训练复用观测 memory，另做一次 eval/no-grad 的完整策略采样，不接收未来动作或有效长度，也不再次运行 VLM。训练图像、PNG/NPZ 和以下物理指标都来自该采样：

- `chunk_position_error_m`、`chunk_displacement_error_m`、`chunk_second_difference_error_m`。
- `chunk_prefix_1_position_error_m`、`chunk_prefix_<H>_position_error_m`，与静止基线 `chunk_stationary_error_m` 对比。
- `chunk_sampled_position_loss`、`chunk_sampled_displacement_loss` 仅作诊断，不加入反传。

`chunk_reconstruction_position_error_m` 单独记录 flow 重建误差。`chunk_flow_loss/weighted_loss` 为向量场训练项，tau 按 high/mid/low noise 三等分记录 count 和位置项，汇总需按有效 count 加权。loss 尺度与毫米误差不同；固定验证集的采样误差和闭环结果才用于比较方法。

HTTP `/model_info.chunk_policy` 保存解析配置，`chunk_spec` 声明 K/dt/H。在线诊断不含 GT；训练与离线图像中的 GT 只用于监督或绘图，不进入部署输入。运行及可视化入口见 [README](../../finetune/OHT/README.md)。

## 旧配置与消融

以下配置保留各自行为，不能因仓库更新就把旧模型解释为新版状态条件模型。

| 配置 | 相对父配置的主要变化 |
| --- | --- |
| [chunk_v4](../../finetune/OHT/configs/oht_chunk_v4_8x40.yaml) | 基础 token decoder，位置权重 5；不启用 spatial/state 条件 |
| [displacement](../../finetune/OHT/configs/oht_chunk_v4_displacement_8x40.yaml) | 基础配置增加相邻位移监督，不改变动作参数化 |
| [relative_goal](../../finetune/OHT/configs/oht_chunk_v4_relative_goal_8x40.yaml) | 基础配置将 key 条件改为相对 TCP，不自动叠加位移 loss |
| [spatial](../../finetune/OHT/configs/oht_chunk_v4_spatial_8x40.yaml) | 相对 key + 三维 attention + 位移监督，回归头 |
| [flow](../../finetune/OHT/configs/oht_chunk_v4_flow_8x40.yaml) | spatial 上增加 flow matching |
| [flow_scaled](../../finetune/OHT/configs/oht_chunk_v4_flow_scaled_8x40.yaml) | 分离动作/几何/loss 尺度，高噪声 tau 采样 |
| [flow_temporal](../../finetune/OHT/configs/oht_chunk_v4_flow_temporal_8x40.yaml) | scaled 上增加时间 AdaLN、时序卷积及二阶差分 |

当前 flow_state 在 temporal 上增加本体归一化、逐层状态条件与 key gate；regression_state、delta_state 从这里分出。公平对照保持数据划分、输入、初始化来源、seed 与预算一致，分别报告物理轨迹误差和实际运行开销。

CPU 回归覆盖坐标、mask、梯度、标签隔离、DDP 和保存恢复；真实 CUDA 渲染、显存、吞吐和 OHT 闭环仍需服务器验证。这些组合改动不是已经证明有效的性能结论。

相关方法：[ChainedDiffuser](https://proceedings.mlr.press/v229/xian23a.html)、[3D Diffuser Actor](https://3d-diffuser-actor.github.io/)、[pi0](https://arxiv.org/abs/2410.24164)。时间采样参考 pi0；时序卷积、物理差分和尺度分离是本仓库的实验设计，并非原样复现这些论文。
