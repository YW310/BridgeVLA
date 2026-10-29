# OHT：显式三维条件与 flow matching chunk

这版保留 BridgeVLA 关键目标预测，提供两种共用条件输入的连续动作头：三维相对 attention 回归，以及 conditional flow matching。任务、相机、TCP、夹爪、V4 标签、epoch、AdamW 和按层冻结沿用现有配置；未开启 `spatial_conditioning` 的旧模型沿用原 decoder。

## 观测几何与坐标

每次点云渲染在原有特征后附加未按 `norm_corr` 额外缩放的当前渲染帧 XYZ 和值为 1 的覆盖通道。它们与 RGB 使用相同 z-buffer、splat 和抗锯齿，不需要第二次渲染，也不改动原 RGB、correlation/depth 通道。空白覆盖为 0，原点处真实表面仍有效。

对每个 VLM patch，分别池化预乘的 XYZ 与覆盖，再相除，得到可见表面的平均位置；没有有效表面的 patch 被屏蔽。token 顺序保持 view → patch row → patch column。coarse 点撤销全局 cube 变换；refine 点先撤销实际预测裁剪，再撤销该样本的全局 cube 变换。最后统一为世界米坐标，使用 `R_tcpᵀ(x_world−p_tcp)` 转到当前 TCP 坐标系。

新 decoder 将视觉特征、view/scale embedding、当前本体状态和预测关键目标作为 memory。动作 tokens 包含未来物理时间 `k * action_dt`；flow 分支另有生成过程时间 `tau`，两者不混用。自注意力和 cross-attention 的每个 head 都使用三维相对位置偏置：相对方向与对数距离经过 MLP 生成 bias。无效视觉 tokens 不参与 attention，当前状态与关键目标始终可用。

上游 VLM 并非严格 SE(3) 等变；这里统一的是 decoder 收到的几何与动作坐标，不据此声称完整系统具备等变性。

## 动作表示与目标

新 decoder 的连续 pose 为 `K×9`：`R_tcpᵀ(p_future−p_tcp)/position_scale` 三维平移，以及 `R_tcpᵀ R_future` 的前两列共六维旋转。平移始终相对本次观测 TCP，而非累加预测增量。旋转通过正交化恢复 SO(3)，再与当前 TCP 旋转相乘；最终返回的动作仍是世界绝对 EEF xyzw 加二值夹爪。

当前 TCP 是已知起点；第一未来点仍监督 `t+dt` 的真实位置，不强制等于起点。预测 key pose 只作为条件，不强制 K 步终点等于该关键目标。目标可能在 K 步窗口之外。

回归分支使用直接 pose 输出及夹爪 BCE。flow 分支训练时采样 `epsilon~N(0,I)` 和 `tau~U(0,1)`，构造 `A_tau=(1−tau)epsilon+tau*A_gt`，学习向量场 `v_theta≈A_gt−epsilon`。平移与旋转分别按现有 position/rotation 权重加权 MSE，再乘 `flow_weight`；同时对 `A_hat=A_tau+(1−tau)v_theta` 使用原有物理位置、相邻位移、旋转和夹爪损失。这里的 flow 向量场描述去噪过程，不是机器人实际运动速度。

仅有效未来标签参与监督，无效 padding 在混合前清理，不向模型暴露 NaN 或未来有效长度。推理从噪声执行 `flow_steps` 次 Euler 积分，在干净端点计算夹爪，再还原世界 EEF。默认种子 0 使用固定条件样本，跨批大小与逐样本推理一致且不改变全局 RNG；`flow_inference_seed: null` 使用随机采样。

## 日志与可视化

flow 训练的 `chunk_*_loss` 是带噪监督及其重建的训练损失。`chunk_flow_loss` 记录向量场损失，`chunk_flow_weighted_loss` 记录加入总损失的值；`chunk_total_loss` 包含 flow 与物理辅助项，再加关键目标监督得到 `total_loss`。

带噪 GT 重建不能用于展示部署性能。训练会复用已计算的观测 memory，另做一次不读取未来标签的完整策略采样。`chunk_position_error_m`、`chunk_displacement_error_m`、`chunk_prefix_1_position_error_m`、`chunk_prefix_<H>_position_error_m` 以及 TensorBoard/PNG/NPZ 曲线均来自该采样。`chunk_reconstruction_position_error_m` 单独标明重建误差。采样在 decoder eval/no-grad 下执行，随后恢复训练状态；不重复运行 VLM。

现有单次前向 heatmap、在线 `/visualization`、HTTP chunk schema 和客户端执行方式保持一致。`/model_info.chunk_policy` 与离线报告保存解析后的 decoder 配置，便于核对是否启用了 flow、相对目标和位移监督。

## 配置与迁移

- `finetune/OHT/configs/oht_chunk_v4_spatial_8x40.yaml`：三维回归，`displacement_weight=1`。
- `finetune/OHT/configs/oht_chunk_v4_flow_8x40.yaml`：继承上述输入与物理 loss，增加 flow，默认 8 次积分。

已有 V4 数据不需重建；位置由现有 RGB-D 点云在当前 forward 中获取。两种新结构均可使用旧 EEF/chunk checkpoint 的 `--init-checkpoint`：保留兼容 encoder，新建 decoder 与 optimizer，不能直接 resume 改变架构。仅调整 flow loss 权重或采样设置进行新 run 初始化时保留整个 decoder；原配置 resume 仍完整加载模型与 optimizer。

CPU 测试覆盖渲染通道对齐、抗锯齿/空白/原点、裁剪逆变换、旋转与世界坐标恢复、相对 attention、有效 mask、flow 积分、标签隔离、梯度、DDP、HTTP 和离线评估。实际 CUDA 渲染、吞吐、内存和 OHT 闭环精度需在训练/仿真服务器验收。

## 尺度与时序实验

`oht_chunk_v4_flow_scaled_8x40.yaml` 在旧 flow 上单独启用尺度与时间采样实验；`oht_chunk_v4_flow_temporal_8x40.yaml` 再开启结构与二阶差分。旧配置与旧 checkpoint 默认行为不变。

| 设置 | 新实验 | 用途 |
| --- | --- | --- |
| `action_position_scale` | `train_quantile`，训练启动后解析为数值 | 当前 TCP 到各未来点的平移编码/解码；单位高斯位置噪声对应这个米尺度 |
| `geometry_scale` | 0.25 m | 观测表面、关键目标和动作 query 的统一 attention 坐标尺度 |
| `position_loss_scale` | 0.01 m | 物理位置 Smooth L1 的独立容差，默认 beta=1 |
| `flow_time_sampling` | `beta_high_noise` | `tau=.999*(1-U**(1/1.5))`，偏向 tau=0 的高噪声端 |
| `adaln` | temporal 配置为 true | 各 block 的 self/cross/FFN 使用 flow-time 条件的 shift/scale/residual gate，gate 零初始化 |
| `temporal_kernel_size` | temporal 配置为 3 | 输出前的 depthwise 时间卷积，零初始化的残差支路 |
| `second_difference_weight/scale` | temporal 配置为 0.1 / 0.001 m | 匹配 GT 的位移二阶差分，不惩罚真实曲率本身 |

未设置前三个尺度时各自回退到原 `position_scale`。动作 query 必须乘 `action_position_scale/geometry_scale` 后才能与场景位置比较；预测目标的空间条件使用 `geometry_scale`。几何仍在当前 TCP 坐标系，最终动作仍为世界绝对 EEF。

动作归一化使用训练集至多 `normalization_max_samples=4096` 个均匀选取的样本。仅检查并读取小型 targets NPZ；当前 TCP 来自已绑定 checksum 的 sample index。只取模型 horizon 内的有效未来点，统计位移半径的 `normalization_quantile=.99` 分位数，设 `normalization_min_scale=.01` m 下限。标量半径尺度不放大近静止轴，且不依赖坐标系朝向。val/test 与无效 padding 不进入统计。rank 0 计算并向全部 rank 广播数值或错误，避免其他 rank 在构建 DDP 时等待失败进程。

统计写入 run 的 `chunk_normalization.json` 和 checkpoint 的 `config.chunk_normalization`；解析后的 float 写入 `config.chunk_policy.action_position_scale`。checkpoint 推理无需训练数据。相同配置 resume 会重新校验训练统计；更换尺度或结构必须新建 run，通过 `--init-checkpoint` 保留 encoder、重新初始化 decoder/optimizer。仅调整 loss 权重/容差或 flow 时间采样时，新 run 初始化可保留 decoder。已有 V4 标签无需重建。

二阶差分以当前 TCP 为第 0 点，匹配 `(p[k]-p[k-1])-(p[k-1]-p[k-2])` 的 GT；相邻两段位移都有效时才监督。数值尺度是米/控制步²，不是 m/s²。时间卷积不接收 GT 或未来有效长度；不强制第一未来点等于当前 TCP，也不强制末点到达预测 key。

`chunk_second_difference_loss/weighted_loss` 属于重建辅助损失；`chunk_second_difference_error_m` 来自独立部署采样。新增 `chunk_sampled_position_loss` 与 `chunk_sampled_displacement_loss` 仅用于诊断，不加入反传。`chunk_flow_tau_{high,mid,low}_noise_count/position_loss` 将 tau 按三等分记录，位置项为动作归一化空间的速度场 MSE；空分箱 count/loss 均为 0，跨 batch 汇总应按 count 加权。这些训练指标不能替代固定验证集上的真实采样误差和闭环效果。

参考方法：[ChainedDiffuser](https://proceedings.mlr.press/v229/xian23a.html)、[3D Diffuser Actor](https://3d-diffuser-actor.github.io/)、[pi0](https://arxiv.org/abs/2410.24164)。时间采样参考 pi0；时序卷积、物理差分项与尺度分离是本仓库的实验设计，不据此宣称精度提升。这是面向现有 BridgeVLA 的组合实现，不是三篇论文的原样复现。

## 本体状态条件与增量回归

`oht_chunk_v4_flow_state_8x40.yaml` 在 temporal 配置上开启 `state_conditioning`、`proprio_normalization: train_quantile` 和 `key_condition_gate`。`regression_state` 仅替换为确定性回归；`delta_state` 在同一回归头上改为逐步增量参数化。旧配置默认全部关闭，已有模型参数与行为不变。三组沿用 Gemma 前 16 层冻结、视觉编码器和 projection 冻结、联合关键目标训练的设置，不加入历史帧或改变预测裁剪。

归一化仅统计 train 中最多 4096 个均匀选取的实测状态。逐关节采用 1%/99% 分位数的中点和半区间作为 center/scale，scale 下限 `proprio_min_scale=0.1` rad，避免近常量关节噪声被过度放大；夹爪输入已是实测开度 [0,1]，固定 center=0.5、scale=0.5。不对超出区间的值裁剪，不改变关节角分支。只解压 observation NPZ 的 `joint_positions` 和 `gripper_open01`；文件 checksum 校验仍需读取完整文件，启动会打印进度。val/test、关键目标标签和未来动作不参与本体统计。

rank 0 计算并广播统计或错误。`proprio_normalization.json` 和 checkpoint 的 `config.chunk_proprio_normalization` 保存来源与统计；七维 center/scale 写入 `chunk_policy`，同时注册为 decoder buffer。推理直接使用 checkpoint 中的数值，无需 replay。动作归一化仍按当前 TCP 到未来点的位移半径统计，三组共用同一尺度定义，不随增量参数化切换。

decoder 把当前 TCP 局部坐标下的状态投影为 `s`，预测关键目标投影为 `g`。条件 `c = time_embedding(tau) + s + sigmoid(key_gate_logit) * g` 同时加入初始动作 queries，并送入每一层 AdaLN 的 self-attention、cross-attention 和 FFN 调制。即使 AdaLN 残差 gate 零初始化，状态也能通过 queries 直接影响输出。保留一个始终有效的状态 memory token，保证所有视觉 patch 无效时 attention 有可用输入。

显式关键目标不再额外走未门控的 memory/XYZ 路径；gate 初值为 0.1，训练指标 `chunk_key_condition_gate` 记录其 sigmoid 值。它只控制显式 key 条件，refine 视觉特征仍依赖原有预测关键目标裁剪，因此不能将 gate 接近 0 解读为整个网络与关键目标完全解耦。

默认回归仍预测每个未来点相对当前 TCP 的 pose。`prediction_representation: delta_pose` 则预测初始 TCP 朝向坐标轴下的平移增量 `d_k=R_0ᵀ(p_k-p_{k-1})/action_position_scale`，累积为 `p_k=p_0+R_0*scale*sum(d_1...d_k)`。旋转增量为 `D_k=R_{k-1}ᵀR_k` 的六维表示，通过矩阵连乘 `R_k=R_0 D_1...D_k` 恢复，不能将 Euler 角或旋转向量直接相加。该参数化目前仅支持 spatial regression。

位置、旋转、相邻位移与二阶差分监督都作用于恢复后的完整世界轨迹；首点以实测 TCP 为起点，前面增量误差会影响后续轨迹损失。输出仍是世界绝对 EEF xyzw 和二值夹爪，V4 数据、HTTP、客户端执行与可视化无需改变。

使用已有 checkpoint 时通过 `--init-checkpoint` 保留 encoder；改变本体归一化、状态条件或动作参数化会重新初始化 decoder 和 optimizer。新实验的原配置可严格 resume。验证应在同一验证集比较完整采样的前 1/4/16 步物理误差和静止基线；CPU 正确性测试不代表轨迹精度已改善。
