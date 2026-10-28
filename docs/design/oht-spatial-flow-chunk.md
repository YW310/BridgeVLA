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

参考方法：[ChainedDiffuser](https://proceedings.mlr.press/v229/xian23a.html)、[3D Diffuser Actor](https://3d-diffuser-actor.github.io/)、[pi0](https://arxiv.org/abs/2410.24164)。这是面向现有 BridgeVLA 的组合实现，不是三篇论文的原样复现。
