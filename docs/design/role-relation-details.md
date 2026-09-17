# Object-conditioned Latent Phase：详细设计

[精简设计](role-relation-prior.md) · [真实机器人设计](real-world-deployment.md) · [代码索引](../reference/code-map.md)

> 更新时间：2026-09-17。状态：设计阶段，尚未完成跨 query memory 与 latent phase 实现。

## 0. 设计决定

当前 MVP 只增加三项能力：

1. 从共享 BridgeVLA feature 预测 task-relevant object slots 与 T/R/NULL role binding；
2. 为 committed Target/Reference 各维护一个短时 memory，主要解决遮挡和 identity switch；
3. 用 objects、instruction、proprioception 和 short history 隐式预测 relation-phase token，并条件化完整动作。

```text
BridgeVLA scene feature
  -> shared object slots
  -> T/R/NULL soft role binding
  -> latent relation-phase token
  -> shared translation/rotation/gripper/collision decoder
```

当前不需要：

- 显式 APPROACH/CONNECT/TRANSFER classifier；
- relation graph、goal ledger 或 rollback state machine；
- `K_T × K_R` pair action 枚举；
- 每个 object 单独运行 VLM；
- learned execution-risk head。

这些模块只有在基础方案暴露对应 failure mode 后才考虑。

## 1. 最小状态与前向

### 1.1 Object-role state

```text
RoleState = {
  token,
  soft_heatmap,
  present_probability,
  visible_probability,
  confidence,
  center_mean,
  center_covariance,
  age
}
```

约束：

- Target 必须存在，Reference 可以为 NULL；
- NULL 表示语义上不需要 Reference，不表示 Reference 被遮挡；
- geometry 无效、object 不存在和 object 不可见必须分开；
- memory 只保存 T/R 两个角色，不维护完整场景 map。

### 1.2 Latent relation-phase

令当前共享场景特征为 `F_t`，T/R token 为 `q_t^T,q_t^R`，短历史为 `h_{t-1}`：

\[
z_t = f_\theta(F_t, q_t^T, q_t^R, g_t^{TR}, l, p_t, h_{t-1}),
\]

其中：

- `g_t^{TR}`：T/R 相对几何；Reference 为 NULL 时使用 learned NULL embedding；
- `l`：instruction embedding；
- `p_t`：EE pose、gripper 等 proprioception；
- `h_{t-1}`：上一动作和短时 role memory，可先不增加独立 RNN。

`z_t` 同时表示当前 relation 和动作阶段，不需要人为拆成两个 head。每个 control query 都重新计算，
因此 phase 可以随当前 object state 向前或向后变化。

### 1.3 Full-action conditioning

同一最终 feature 预测：

```text
translation heatmap
rotation
gripper
ignore-collision action label
```

若 relation-conditioned feature 改变了 waypoint，R/G/C 也必须从该 feature 重新预测，不能沿用
conditioning 之前的位置输出。这是当前设计区别于 post-hoc translation fusion 的关键。

## 2. Object 与 memory

### 2.1 Slot 预测什么

`InternalObjectSlotPredictor` 产生少量无序 soft slots。role head 根据 instruction 和 scene context
对 slots 做 T/R/NULL mixing，而不是先生成完整 object list 再做昂贵组合搜索。

最小输出：

```text
target/reference prior
target/reference token
target/reference present
target/reference visible
role confidence
reference_is_null
```

默认使用 soft posterior。Top-2 仅用于离线 Recall@1/2 诊断；低置信不能自动改用第二候选，因为
“第二名”不等于“正确且可执行”。

### 2.2 Short role memory

memory 的目的只有两个：

- object 短时被遮挡时保持 identity；
- object 重新出现时减少 T/R switch。

可用共享参数的 gated update：

\[
m_t^r = \operatorname{GRU}([q_t^r, p_t], m_{t-1}^r),\quad r\in\{T,R\}.
\]

visible/confidence 决定更新强度；不可见时保留旧 token 并提高不确定性。memory 在 episode reset 时
清空，在同一 coarse/refine query 中只更新一次。

memory 不保存 task phase。若当前可见 object state 已改变，新的视觉证据应覆盖旧动作倾向。

## 3. Phase 不需要显式状态机

### 3.1 正常长程操作

长程任务中的动作切换来自 object relation 的连续变化。例如放置任务中：

```text
block far from gripper
→ block grasped
→ block near container
→ block inside container
→ gripper released
```

网络不需要输出这些字符串；`z_t` 只需在对应状态产生正确动作。

### 3.2 Stack collapse

若 stack 到一半坍塌：

- 当前 object positions、相对高度和接触关系发生变化；
- 下一 query 重新计算 slots、relative geometry 和 `z_t`；
- policy 应重新产生抓取或重建动作。

因此不需要显式 rollback。若坍塌后场景接近初始状态，latent phase 自然接近初始动作模式。

### 3.3 什么时候当前状态不够

只有下面情况才需要更长历史：

> 两个时刻拥有近似相同的可见 object state 和 proprioception，但合法下一动作因过去事件不同而不同。

先用上一动作、gripper event 或一个小 recurrent token 解决。只有该方案仍失败，才考虑 explicit
progress/ledger；不要预先增加符号状态。

### 3.4 遮挡

遮挡不是 phase：

- `present=True, visible=False`：保持 identity，增加 uncertainty；
- `present=False`：角色语义上不存在，可进入 NULL；
- 几何证据不足：再观测或停止，不猜测 hidden relation。

同一可见点云重新投影不会产生新信息；真实再观测必须改变传感器证据。

## 4. 与当前仓库的关系

当前可复用模块：

| 位置 | 作用 |
| --- | --- |
| `finetune/bridgevla/models/oracle_prior.py` | internal slots、T/R/NULL mixing、relation adapter/anchor |
| `finetune/bridgevla/mvt/mvt.py` | coarse/refine 两阶段编排 |
| `finetune/bridgevla/mvt/mvt_single.py` | adapted feature 与完整动作头 |
| `finetune/bridgevla/models/bridgevla_agent.py` | action loss、act/reset 与运行时状态入口 |
| `finetune/RLBench/replay_buffer.py` | replay schema |
| `finetune/RLBench/dataset.py` | batch 采样与设备转换 |

默认 `rvt2.yaml` 每个 stage 使用 `top/front/right` 三个正交视角，即 `3 × 2`。当前设计复用这些
scene features，不创建 per-object view bank。

建议的数据流：

```text
previous role memory + current RGB-D/instruction/proprio
→ current slots and role tokens
→ current latent relation-phase token
→ complete action
→ final stage updates role memory once
```

slot predictor 不能依赖尚未生成的 current `z_t`，避免同一步循环依赖。

## 5. 训练

### 5.1 数据契约

必须提供或可靠派生：

```text
target/reference points or masks
target/reference present
target/reference visible
geometry valid
full action labels
short temporal correspondence
```

GT object、GT phase 和 future frame 只能生成 label，不能进入 policy-side forward。

### 5.2 损失

\[
L = L_{action}
  + \lambda_{role}L_{role}
  + \lambda_{pv}L_{present/visible}
  + \lambda_{temp}L_{temporal}.
\]

- `L_action`：translation、rotation、gripper 和现有 collision action label；
- `L_role`：T/R heatmap 与 slot-role binding；
- `L_present/visible`：区分 NULL、遮挡与无效几何；
- `L_temporal`：短时 identity/action consistency，仅在对应可靠时计算。

首版不增加 phase-class、relation-class、ledger 或 risk loss。可选 phase label 只能作为 auxiliary
ablation，不能成为部署依赖。

### 5.3 训练顺序

1. 复现原 BridgeVLA 与 Oracle-role diagnostic。
2. 训练 predicted slots、T/R/NULL 和 present/visible。
3. 接入 latent relation-phase，并让同一 feature 预测完整动作。
4. 逐步解冻 action decoder、projector 和上层 backbone。
5. 用短序列训练 role memory。
6. 在 predicted-only sim/real 数据上适配和校准。

## 6. 实验与 Go/No-Go

### 6.1 必做消融

| 对照 | 回答的问题 |
| --- | --- |
| Base BridgeVLA | 结构是否真正提升闭环 |
| Oracle role | object 信息的收益上限 |
| Predicted role | 无 GT 部署是否成立 |
| Without `z_phase` | latent relation-phase 是否被使用 |
| Without memory | memory 是否只改善遮挡场景 |
| Adapter-only vs joint tuning | 提升是否受 adapter 容量限制 |

### 6.2 必做反例

- Reference 为 NULL 与 Reference 被遮挡；
- T/R swap、同类 distractor 和 object identity 重现；
- visible→hidden→visible；
- 不同执行速度、停顿和 stack collapse；
- 相同 instruction、不同 object relation；
- 相似 object state、不同历史，用于检测 observation aliasing。

### 6.3 指标

- closed-loop task success；
- T/R grounding、NULL、visibility 和 identity-switch rate；
- translation/rotation/gripper/collision 分量失败；
- collapse 后恢复成功率与额外动作数；
- latency、显存和拒绝率。

若 object/phase auxiliary loss 下降但 closed-loop 不提升，不继续增加结构。

## 7. 可选扩展

| 触发条件 | 扩展 | 首版状态 |
| --- | --- | --- |
| 投影碰撞或小物体分辨率不足 | T/R role-layered orthographic refine | 待验证，不启用 |
| soft roles 召回不足 | bounded pair hypotheses | 不启用 |
| 明确存在历史混淆 | longer recurrent state / explicit progress | 不启用 |
| endpoint 可达但执行失败 | execution-risk calibration | 不启用 |
| assembly 需要接触位点 | contact/keypoint head | 不启用 |

### 7.1 Object-layered Orthographic Refine

该扩展针对 BridgeVLA/RVT 类正交投影的两个问题：不同物体投到同一像素后的 z-buffer 覆盖，以及
小物体在全局视图中分辨率过低。它不能解决原始 RGB-D 未观测到的真实传感器遮挡。

| 情况 | 是否可改善 |
| --- | --- |
| 点已存在于输入点云，但被其他物体在虚拟投影中覆盖 | 是，分角色独立 rasterize |
| Target 太小，全局投影只占少量像素 | 是，使用 object-centered bounded scale |
| 物体表面未被任何真实相机观测 | 否，需要多相机、temporal memory 或 active perception |

建议只改 refine stage：

```text
global point cloud
→ 原 coarse top/front/right views
→ soft T/R role membership
→ lift membership to 3D points
→ separately render Target and Reference layers
→ fuse global + T-layer + R-layer + relative geometry
→ shared full-action decoder
```

每个 role layer 使用 world-aligned `top/front/right`，中心来自预测 object center，尺度由 object
extent 乘固定 margin 后裁剪到上下界。首版不学习任意相机旋转。Reference 为 NULL 时不渲染
Reference layer，使用 learned NULL embedding。

为避免重新引入 post-hoc fusion，分层视图只作为 decoder 输入：translation、rotation、gripper
和 collision 仍从同一最终 feature 预测。保留 global branch 以提供障碍物与 workspace context；
role confidence 只控制 local layer 的权重，不能把低置信 crop 当作确定事实。

```text
F_action = Fuse(
  F_global,
  confidence_T * F_T,
  confidence_R * F_R,
  relative_geometry
)
```

推荐按成本递增比较：

1. 原 global views；
2. global views + T/R soft mask/depth channels；
3. global views + independently rasterized T/R layers；
4. 额外真实视角或 active perception。

若第 2 项已解决问题，不增加 local rendering；若点云本身缺失，第 3 项不应有效。报告 projection
support recall、heatmap error、额外延迟/显存和 closed-loop success，避免仅凭可视化判断收益。

## 8. 实施优先级与方法边界

```text
P0  present / visible / valid 数据契约
P1  predicted slots 与 T/R/NULL role tokens
P2  relation-phase token 条件化完整动作
P3  joint tuning，而非只训练 adapter
P4  两角色短时 memory
P5  predicted-only sim/real 闭环
P6  仅按已确认瓶颈加入可选扩展
```

主方法的边界是：**object-conditioned latent-phase full-action policy**。它不主张首次使用 object、
relation 或 memory；需要证明的是，在不引入显式 phase taxonomy 和全图规划的情况下，T/R role
conditioning 是否改善完整动作与长程闭环。
