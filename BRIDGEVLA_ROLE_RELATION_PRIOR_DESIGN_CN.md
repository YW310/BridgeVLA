# BridgeVLA 角色与关系先验设计说明

> 更新时间：2026-08-07  
> 状态：Oracle 上限实验之后的候选训练方案  
> 核心判断：单纯提供 object center 与 BridgeVLA 已有 translation/keypoint 学习高度重合；更值得建模的是实体角色、interaction site 和随任务阶段变化的空间关系。

## 1. 研究判断

BridgeVLA 在预训练中已接触 object-center 类信号，微调又以动作 keypoint/translation 为监督。因此，再把目标物体中心叠加到 translation heatmap，通常只能提供粗位置提示，还可能把预测从真正的抓取点、接触点或放置点拉向几何中心。

object 信息仍可用于候选区域选择和同类实例消歧，但不应作为新的中心点监督。更值得注入的信息是：

- `target`：当前被操作或抓取的实体；
- `reference`：放置、插入、堆叠所依赖的参考实体；
- `site`：按钮表面、插孔、容器内部、堆叠顶面等交互位点；
- 末端、target、reference/site 之间的方向、距离和任务阶段。

不建议把 raw object prior 直接交给低层 planner。planner 主要消费目标位姿、夹爪和碰撞标志；只有先验被转换为候选目标位姿、接近方向、可达区域或约束时才有直接意义。

## 2. 推荐的信息表达

### 2.1 三角色热图

在 BridgeVLA 三个正交视图中额外渲染：

```text
H_target       当前 target 实体区域
H_reference    当前 reference 实体区域
H_site         当前可执行 interaction site
```

阶段规则：

- 抓取前重点使用 `target + site`；
- 抓取后重点使用 `reference + site`；
- 按钮、开关、插孔等任务始终优先 `site`。

object mask 使用平坦区域提示，不在物体中心制造额外高峰；site 可用局部 Gaussian 或距离变换软化。这样表达的是“角色与可交互区域”，而不是重复动作 keypoint 标签。

### 2.2 关系特征

首版优先使用紧凑的 relation token：

```text
p_ee, p_target, p_reference, p_site
p_site - p_ee
p_reference - p_target
对应距离、单位方向、可见性/置信度
grasped/phase one-hot
```

三视图热图回答“在哪里”，relation token 回答“当前末端与任务实体是什么关系”。

稠密向量场可作为后续对照，但首版只建议测试 site-relative field：每个有效像素指向 site 的二维单位方向，并提供距离与有效 mask。target/reference/EE 全量向量场通道多且跨视图冗余，不作为主线。

## 3. 推荐接入结构

不要把角色图直接拼入预训练 RGB 通道。新增小型 `RoleRelationAdapter`：

```text
H_target/H_reference/H_site
        -> role encoder -> role tokens --+
                                         +-> zero-init gated residual -> MVT visual tokens
relation features -> MLP -> relation token+
```

实现要求：

- role encoder 的 patch/grid 与 MVT 对齐；
- 残差门控零初始化，初始行为等价于原模型；
- stage-1 与 stage-2 按各自坐标重新渲染，不复用 stage-1 像素图；
- rotation、gripper、collision heads 首版不改；
- 保留现有 Oracle logit fusion，作为无需训练的诊断基线，而非最终部署结构。

## 4. 训练方式

Oracle truth 用于回答信息上限，不应成为部署时唯一输入。建议：

1. 冻结 BridgeVLA，只训练 `RoleRelationAdapter` 与门控；
2. 稳定后解冻 MVT 后层或使用 LoRA，小学习率联合微调；
3. 用当前项目的预测角色/site 替换 truth，做噪声鲁棒训练。

训练批次混合：

- 完整 truth prior；
- 腐蚀、膨胀、偏移、漏检、错实例等 noisy prior；
- `30%~50%` prior dropout，退化为 B0；
- 必要时加入 B0 输出一致性约束，避免网络完全依赖 oracle 通道。

监督仍使用 BridgeVLA 原 translation/keypoint loss。角色图是条件输入，不是新的动作标签；梯度通过 adapter 和 translation head 学习何时使用这些信息。

## 5. 最小实验矩阵

| 实验 | 输入/融合 | 目的 |
| --- | --- | --- |
| B0 | 原 BridgeVLA | 基线 |
| O1-flat | truth target/reference mask + logit fusion | 验证实体选择价值 |
| O3-V | truth visible site + logit fusion | 验证 interaction-site 价值 |
| A-oracle | truth 三热图 + relation token + adapter | 验证可学习融合上限 |
| A-noisy | noisy/dropout prior + adapter | 验证鲁棒性 |
| A-pred | 当前项目预测 prior + adapter | 验证部署价值 |

结果解释：

- O1/O3-V 均不改善 translation error 或 PCK：停止开发 adapter，先检查先验定义与投影；
- O1 明显改善：角色/实例消歧有价值；
- 只有 O3-V 明显改善：集中开发 interaction-site，而非 object center；
- translation 改善但在线 success 不变：瓶颈在 rotation、gripper、collision 或低层执行；
- A-oracle 有效但 A-pred 无效：问题在 prior 预测质量或训练域差，而非融合结构。

沿用 Oracle handoff 的 Go 标准：hard-task success 提升至少 `10pp`，或四任务总体提升至少 `5pp`；简单任务 harm 不超过 `2pp`。

## 6. 当前结论

该方向靠谱，但研究问题应从“再告诉模型 object 在哪”改为：

> 显式的 target/reference/site 角色和末端关系，能否帮助 BridgeVLA 完成同类实例消歧、阶段切换与精确 interaction-site 定位？

优先路径：`Oracle O1/O3-V -> 三角色热图 + relation token -> zero-init adapter -> noisy/predicted prior`。稠密多向量场和直接 planner 注入仅作为后续对照。
