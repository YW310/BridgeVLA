# BridgeVLA-ARE：以主动关系编辑统一 object、relation 与 phase

> 更新时间：2026-09-07（根据数学、监督与执行接口 review 修订）
> 状态：顶会候选研究设计，尚未实现
> 当前代码边界：仓库默认 `rvt2.yaml` 使用 coarse/refine 两阶段，每阶段渲染 `top/front/right` 三个正交视角，即 `3 x 2`；另已实现 GT Target/Reference point prior、两通道 relation prior、zero-init feature adapter、translation fusion，以及同一次 forward 中的 `trans_base` / `trans_raw` 输出。本文提出的 phase-gated object candidates、per-object view bank、pair-specific heatmaps、pair reliability cost、关系编辑头与事件切换均是下一阶段工作。

## 0. 最终判断

你的理解方向是对的，但要再精确一层：

> **relation 是世界状态；phase 不是另一套独立语义，而是当前正在执行哪一个 relation edit、完成到什么程度、何时应切换的执行状态。long-horizon manipulation 不是持续切换 relation，而是在较长区间内维持一个 active edit，正常推进由稀疏完成事件驱动，失败恢复和身份纠错另行记录。**

因此，不建议把 `预测 object`、`预测 relation`、`预测 phase` 做成三个并列贡献。主线应收敛成一个紧凑的结构化控制变量：

> **Active Relation Edit（ARE）：当前要让谁与谁之间的哪一种关系发生什么变化，同时保持哪些已经成立的关系。**

object prediction 负责 ARE 中的动态任务角色绑定；phase 由 edit、进度与已完成证据定义。grounding confidence、动作执行风险及未来 completion hazard 分别表达，并通过有界 beliefs/事件记录支持持续执行。

本文建议的论文暂名：

> **BridgeVLA-ARE: Active Relation Edits as the Control State for Long-Horizon Manipulation**

一句话方法：

> BridgeVLA-ARE 从普通视觉语言特征中预测一个稀疏、角色绑定的关系编辑，并让同一个编辑同时决定关注哪个物体、动作服务于什么局部目标以及何时切换；部署时不需要外部 detector、分割器或符号规划器。

## 1. 核心建模

### 1.1 从关系状态到关系编辑

令当前 object-relation graph 为

\[
G_t=(V_t,R_t).
\]

相邻时刻的大多数关系不变，真正与操作有关的变化通常很稀疏。定义当前主动编辑

\[
e_t =
\big(
b_t^T, b_t^R, \kappa_t, r_t^- \rightarrow r_t^+,
R_t^{keep}
\big),
\]

其中：

- `b_t^T`、`b_t^R`：任务 target 与 reference 的软角色分布；reference 允许 NULL。跨帧关联保留有界 binding ID，与谓词参数顺序分开；
- `\kappa_t`：关系编辑算子；
- `r_t^- \rightarrow r_t^+`：当前关系及期望关系；
- `R_t^{keep}`：执行当前编辑时必须保持的关系。

第一版只需要五类算子：

| 算子 | 典型关系变化 | 必须保持 |
| --- | --- | --- |
| `APPROACH` | `far(EE,T) -> near(EE,T)` | free、collision-free |
| `CONNECT` | `free(EE,T) -> attached(EE,T)` | target identity |
| `TRANSFER` | `far(T,R) -> near(T,R)` | attached(EE,T) |
| `SATISFY` | `near(T,R) -> on/inside/aligned(T,R)` | attached，直到目标关系稳定 |
| `DISCONNECT` | `attached(EE,T) -> detached(EE,T)` | 已建立的 on/inside/aligned |

抽屉、门等 articulated task 不必另建 phase taxonomy，只增加 `ARTICULATE: closed -> open` 或 `open -> closed`，并保持 `attached(EE,handle)`。

### 1.2 phase 是 edit 的执行状态

\[
\mathrm{phase}_t=(e_t,u_t,d_t,\beta_t^{next}),\quad u_t\in[0,1].
\]

`u=0` 表示该 edit 尚未实现，`u=1` 表示已到目标；几何残差归一化时必须统一这个方向。`d_t` 是当前 goal predicate 已成立的证据，`beta_next` 是下一 action interval 内完成的预测 hazard。两者分别监督：预期马上完成不能替代已经完成的证据。

`h_t` 保存当前 edit、近期观测/本体历史、物体 beliefs 和有界 event/invariant ledger。采用 APPROACH/CONNECT 等 operator 词表是参数化选择，并不意味着 phase 完全不含类别；贡献应由角色、目标与结束条件的联合建模检验，不能靠改名区分普通 phase head。

### 1.3 long horizon 的正常推进与失败恢复

一个 edit 可持续多个 query/action intervals；只有 `d_t` 连续稳定成立、keep 约束满足后，才允许按完成事件推进。新增关系进入带实体锚点的 keep ledger；将来不再需要或已被合法后续动作撤销的约束才能移除。

completion 只控制正常推进。抓取失败、滑落、错误绑定或低观测置信度也可以触发重规划/恢复，但必须标记为 failure/recovery transition，不能算作 edit 完成。身份纠错可以发生在同一个 operator 内；保持 phase 不意味着必须永久保持错误 pair。

新的 edit 由当前观测、指令和已完成记录提议。仅保留上一个 edit 未必能区分重复子任务/已完成前缀不同的场景，故 `minimal sufficient state` 只能是待检验假设，不能预设成立。每个任务应定义 ledger 容量和超限行为，并在 observation-aliasing split 下验证。

## 2. object 到底预测什么

### 2.1 任务角色、谓词参数与 reference arity

`target=T` 始终表示当前被操作的场景物体；`reference=R` 表示当前 edit 需要的第二个场景物体。它们与谓词的参数顺序分开：

| 操作 | task pair | 当前谓词 |
| --- | --- | --- |
| 抓 block | `(block, NULL)` | `attached(EE, block)` |
| 将 block 放入 bowl | `(block, bowl)` | `inside(block, bowl)` |
| 松开 block | `(block, NULL)` | `detached(EE, block)`；keep 中仍可引用 bowl |
| 拉开 drawer | `(handle, drawer)` | `open(drawer)`，keep `attached(EE, handle)` |

因此 `attached(EE, block)` 中 block 是谓词第二个参数，却仍是 task target。EE、world 与 gravity 是显式可用的锚点，不进入场景 object 候选排名。不同 episode/edit 可让同一物体承担不同任务角色，但不能仅因谓词参数位置改变而交换标签。

`reference=NULL` 只表示当前 edit 不需要第二个场景物体；它不表示 target 可信，也不表示 reference 暂时看不见。需要的 bowl 被遮挡时仍应由 tracked bowl hypothesis 承担 reference。`keep` 可以包含不属于当前 task pair 的实体，须另存少量带身份的 invariant records。

### 2.2 当前 phase 的有界候选与显式 NULL 配额

由 coarse 三视角特征、指令和上一执行状态产生点级软 membership slots：

\[
o_i=(m_i,z_i,c_i,s_i,\Sigma_i),\qquad
\mathcal C_T=\operatorname{TopK}_{K_T}p_T(i),\quad
\mathcal G_t=\mathcal C_T\times\mathcal C_R.
\]

第一版固定 `K_T=K_R=2`，`K_R` **包含 NULL 配额**：

- reference 必需的 edit：取最多两个非空 reference；硬 mask 只来自明确任务 arity/类型约束，不从低置信 phase argmax 推出；
- NULL 合法的 edit：`C_R={NULL}∪Top(K_R-1)` 个非空候选；允许不足配额，不填充重复物体；
- target 永远是场景实体；instance 去重和合法性过滤后最多四个有序 pair。self-pair 默认过滤；part/whole 例外需显式定义。

object score 只负责高召回提议。最终 ground score 利用联合语言、关系和身份一致性评价 pair，不能将独立 target/reference confidence product 当作已校准的联合概率。训练需支持多个语义等价的正确 pair，而非把示范中未选的合法实例一律当负例。

### 2.3 belief 必须在 top-k 之前进入候选池

候选来源为 `当前观测 slots ∪ 当前 committed binding 的预测 belief`。当前角色短时不可见时，其 belief 在对应 role 的固定配额中保留一个名额；并不是在 top-k 结束后才加 temporal cost。优先匹配 3D motion、appearance 与 identity，再决定是否更新或重绑定。

- 为 committed pair 的非空物体维护最多两个短时几何 beliefs；NULL 不分配几何状态。新 pair 中沿用的物体继续继承原 belief。
- 其余候选是当前帧假设；同一实体跨 role/pair 去重。两个角色关联结果必须检测 ID swap。
- 首帧、完成边界或 recovery 时，先由 coarse observation + instruction + 已完成事件记录提议 edit/operator，再在该 edit 下选对象；无需未来帧或 GT phase。
- phase 置信度低或全部候选不可信时，重新打开同一 phase 的候选提议，并重新评估 edit；不能让上一错误 phase 永久屏蔽正确物体。
- belief 超时只降低身份/位置置信度并触发重观测，不能自动变成 NULL 或 relation completion。

保留少量候选是为了处理歧义；最终动作必须对应一个 committed pair，`M=2` 仅表示评估两组假设，不能直接平均两组机器人动作。

## 3. 为什么这条主线比 `object + relation + phase` 更像论文贡献

### 3.1 近期工作已经覆盖的方向

以下为相关工作定位；本轮重点对照直接基于 BridgeVLA 的 ActiveVLA、TVVE 和 object-local affordance。各项仅用于确定实验边界，不能代替系统 novelty 检索：

| 已拥挤方向 | 代表性工作 | 与本文的边界 |
| --- | --- | --- |
| object-centric token / mask | [Oat-VLA](https://arxiv.org/abs/2509.23655)、[OBEYED-VLA](https://arxiv.org/abs/2512.22519) | ARE 不把 object token 本身当贡献，而把 object 绑定为 relation edit 的角色 |
| object-object relation / keypoint plan | [TAX-Policy](https://openreview.net/pdf?id=4CLiGBQV3U)、[ReKep](https://arxiv.org/abs/2409.01652) | ARE 不输出完整关键点计划；VLA 产生动作 proposal，IK/path checker 可作为独立执行验证，不应再声称系统不调用任何外部几何检查 |
| scene graph 与 graph edit planning | [EPoG](https://arxiv.org/abs/2602.04419)、[POT-VLA](https://arxiv.org/abs/2607.18016) | ARE 只保留当前一个局部 edit 和 keep set，不维护可搜索的完整符号图 |
| progress / memory / intent | [ChainVLA](https://arxiv.org/abs/2608.02326)、[INDI](https://arxiv.org/abs/2608.23478) | ARE 的执行状态必须显式绑定 subject、reference 和关系变化，可做对象/关系反事实干预 |
| privileged residual distillation | [PFD](https://arxiv.org/abs/2604.25859) | 有/无 Oracle 输入的动作差已不是独立创新；只可作训练技巧或 baseline |
| future/world variation | [DeltaVLA](https://arxiv.org/abs/2603.08361) | ARE 不预测 future latent 或世界变化分布，而预测当前应执行的最小关系编辑 |
| occlusion、view imagination 与 spatial memory | [LIBERO-Occ](https://arxiv.org/abs/2606.10862)、[SOMA](https://arxiv.org/abs/2605.22283) | 遮挡处理本身不是 novelty；本文只研究它与 BridgeVLA 投影及 ARE transition 的特定耦合 |
| BridgeVLA 上的虚拟视角评分与 zoom | [ActiveVLA](https://arxiv.org/html/2601.08325v1) | 直接近邻基线：需在匹配计算下检验 per-object 复用和 pair heatmap，而非只声称改善视角/分辨率 |
| task-aware virtual view selection | [TVVE](https://openaccess.thecvf.com/content/CVPR2026/html/Bai_Learning_to_See_and_Act_Task-Aware_Virtual_View_Exploration_for_CVPR_2026_paper.html) | 按任务选虚拟相机本身也不是 novelty；本文采用当前 phase 下 per-object 局部 frame 与 pair-conditioned heatmaps；需与单一 task region 的选视角方法直接比较 |
| object-centric pair affordance | [Pick2Place](https://arxiv.org/abs/2304.04100) | object-local action space 与 pick/place 条件化已有先例；本文需要证明 phase-gated top-k pair view bank 对 VLA 的长链 role rebinding 与 relation transition 有独立价值 |

因此，文中不应声称首次使用 object/relation/phase/privileged distillation。可主张并检验的是下面这个更具体的命题：

> **一个稀疏、角色绑定、带完成机制的 Active Relation Edit，是否能成为 end-to-end VLA 的最小充分控制状态，并比彼此独立的 object grounding、phase label 或全图规划更适合 compositional long-horizon manipulation？**

这是论文的核心假设，不是预设事实。若 matched baseline 实验不支持，就应否定它。

### 3.2 三个真正连贯的贡献点

1. **表示贡献：可干预的 Active Relation Edit conditioning。**
   用同一变量联合表达 `操作谁`、`建立什么关系`、`保持什么关系`，而不是拼接三个互不约束的辅助 head。

2. **执行贡献：relation-completion transition。**
   edit 持续状态由当前完成证据驱动正常推进；身份纠错/失败恢复可在未完成时发生。未来 hazard 独立预测，不作为已完成证据。

3. **验证贡献：causal compositional evaluation。**
   不只报告 role accuracy；还交换 target/reference、目标 predicate 或 edit progress，检查动作是否按 edit 发生可预期变化，并在未见过的 object-relation 组合上测试。

注意：options/SMDP、termination head、soft object heatmap 都不是新的。novelty 必须来自三者围绕 **ARE 及其有界 belief/事件记录** 的结构与证据链。

## 4. 最小可实现架构

```text
RGB-D + instruction + proprioception + previous edit/ledger
  -> original coarse 3-view VLM
       +-> original waypoint-centered refine 3-view VLM -> baseline full action
       +-> edit proposal/update + phase-gated observed slots
             + reserved tracked-object beliefs + NULL arity
             -> bounded candidates + pre-encoding grounding cost
             -> unique object RGB view banks + geometry side inputs
             -> shared per-object encoder
             -> each legal pair: support-aware local heatmap group
             -> <=2 full pose proposals including gated residual + execution checks
             -> commit one pair / collect information / reject
             -> return selected checked proposal (baseline + gated residual)
completion evidence + keep monitor -> update edit/ledger for next query
```

### 4.1 Phase-gated Object Candidate Head

复用 coarse multi-view features，产生少量 soft object slots；将 image-grid features 回查到输入 3D 点，明确 projected visibility、point membership 和未观测部分的 mask。候选分数与几何只来自当前观测或已记录 belief，不访问 simulator GT object pose。

先做 instance/identity matching，合入 committed tracked objects，再按第 2.2 节的固定角色配额提议。高召回提议、联合 grounding 排序、后续 execution gate 是不同职责；全部不可靠时须允许重提议，而不是无条件 argmax。

第一版 hard top-k/crop 使用离散选择，可通过 GT slot matching、候选排序与有效 local action supervision 训练；被裁掉候选的 action loss 无法直接反传到 selector。需报告 proposal recall 与 GT-to-predicted exposure gap，不能仅以 joint fine-tuning 宣称全选择过程可微。是否采用 soft selection/straight-through 是独立后续实验。

### 4.2 Relation Edit Head

首帧/边界/恢复时先从 global features 提议 operator/goal；普通 query 使用上一 committed edit 作为先验。随后对每个 pair 结合角色特征、相对几何、EE、本体状态与 ledger 评估 edit compatibility。NULL 用 null embedding 与显式 anchor，并 mask 不存在的 object-object geometry。

输出 operator/goal distribution、current predicate evidence、progress `u`、已完成证据 `d`、未来 hazard `beta_next` 和 keep predicates。任务角色与谓词参数通过显式 mapping 连接，例如 `(T=block,R=NULL) -> attached(EE,T)`。

第一版关系词表限制在 near/contact/attached/on/inside/aligned/open；标签可靠性不足时使用 unknown/soft label。对只观察质心无法判别的 inside/aligned/contact，不构造假确定性标签。

### 4.3 Sticky edit 与可纠错的角色绑定

在 logits 上加入有界 stay bias，而不是给归一化概率直接加常数：

\[
\tilde\ell_t(e)=\ell_t(e)+b_{stay}\mathbf 1[e=e_{t-1}](1-d_{t-1}),
\qquad q_t=\operatorname{softmax}(\tilde\ell_t).
\]

正常推进需要 goal evidence 连续稳定、keep 仍满足；grounding 纠错、失败恢复则走单独 transition reason，不要求先宣布完成。已建立接触但目标关系尚未实现时，可能仍需要在同 edit 内重新选动作。

候选身份关联、believed geometry、relation completion 分别更新，避免 cost 升高被解释成成功或暂时不可见被解释成角色不存在。短时 hysteresis 的收益通过 premature/delayed switch、wrong-binding persistence 与 recovery success 分开衡量。

### 4.4 保留完整 baseline，并只融合受 gate 控制的增量

设 `Q_B` 是原始 BridgeVLA **coarse + waypoint-centered refine** 的输出，保留其 crop、投影、解码与原动作 head。object view bank 是额外分支；其 pair heatmap 输出需要定义为相对于同 frame 无 ARE 分支的增量：

\[
\Delta Q_g^{world}(x)=
\operatorname{Lift}(Q_{g,adapted};x)
-\operatorname{Lift}(Q_{g,unconditioned};x),
\]
\[
Q_{final}(x)=Q_B(x)+\gamma_t D_g(x)\Delta Q_g^{world}(x),
\quad \gamma_t\in[0,1].
\]

`g` 是已选定的一个 pair；`D_g` 是该分支有效空间 mask。两次 Lift 使用相同坐标、support 和归一化，不能把 object-local base heatmap 再加到全局 baseline 上。最后一层 residual zero-init，保证初始化时增量为零；关闭 `gamma` 时直接返回原始完整动作，以避免插值/重解码破坏 baseline 等价性。该等价性指同一 checkpoint、eval 模式下的消融；若联合更新主干，不能再声称等价于原始冻结 checkpoint。

`trans_base` 只是在**当前 crop**上 adapter 前的输出，不自动等于原始两阶段模型的 counterfactual。需要独立保留原 waypoint-centered refine 的中间结果/动作，才能计算可靠的 rescue/harm。等价性在 policy proposal 输出处验证；最终 execution gate 对 baseline 与 ARE 按相同规则处理。

第一版 pair 分支只改 translation，但必须明确 rotation/gripper 的读取契约：原 head 参数可复用，所采样的位置 feature 会随 waypoint 改变。最终用选定 waypoint 构造完整 `SE(3)+gripper` 动作，再评估 execution risk；不能用旧 waypoint 的 rotation 去验证新 translation。execution check 必须针对实际返回的 residual-corrected proposal，任何后续动作修改均需重检。若选定点超出 baseline refine crop，需要显式改用 coarse/选中 object frame 的 action features 并另计接口与计算，不能静默 clamp。最小版只允许 baseline refine support 内的 correction，较大重定位应触发新的 coarse query，作为明确限制记录。

`M=2` 时各自构造并检查实际完整 proposal，再按第 4.7 节的身份/可执行规则提交其中一个 pair 的动作。禁止平均不同 pair 的坐标或 logits；例如两个单峰 logit 的平均可以把最高点移到两个物体之间。需要分布混合的后续实验，应先在统一 world support 上归一化概率，并保留离散 pair 标签到执行选择，不能通过期望坐标消除多峰。

### 4.5 遮挡必须进入 belief，而不能混入 task relation

这里要区分两个问题：

1. **sensor occlusion**：真实 RGB-D 相机没有观测到目标表面，源点云中根本没有这些点；
2. **projection collision**：源点云中已有多个 3D 表面，但它们在某个正交虚拟 view 中落到相同或相邻像素。

默认 `rvt2.yaml` 实际使用 point renderer：coarse 与 refine 各渲染 `top/front/right` 三个正交视角，所以是 `3 x 2` 张投影图，而不是五个视角。refine 先用 `trans_pc` 围绕 coarse waypoint 平移、缩放，再重复同样三种朝向。六张图仍来自同一份融合输入点云，只能缓解第二类问题，不能恢复第一类缺失几何。

当前 point renderer 已产生 `pc_depths`；`get_pt_loc_on_img` 内部也计算了 camera-space points，但只返回 xy pixel。Oracle prior 的 `rasterize_instance_points` 仍只用投影 xy 做 `scatter_add`，没有把 role point depth 与场景 front depth 比较。这会产生一种典型冲突：

```text
role prior: 后方 target 在该像素上概率很高
rendered feature: 该像素实际显示前景 occluder
```

此外，stage-2 只是围绕 stage-1 waypoint 重新中心化并重渲染，能提高局部分辨率，但不能补回源点云里从未出现的表面；stage-1 一旦被遮挡误导，stage-2 还可能放大该错误。

#### 4.5.1 把 visibility 与 physical relation 分开

`inside/on/attached/open` 是世界中的 task relation；`visible/occluded/unknown` 是依赖相机或虚拟 view 的认识状态。定义

\[
v_{t,k}^{r}\in
\{\text{visible},\text{occluded},\text{unknown}\},
\]

表示 role `r` 在 view `k` 下的可见性。`v` 不进入 ARE operator taxonomy，也不能直接触发 phase transition：

- target 被 gripper 挡住，不等于 `CONNECT` 已完成；
- object 暂时消失，不等于应切换到下一个 relation edit；
- 当前看不见 `inside` 结果，不等于该 relation 不成立。

#### 4.5.2 有界的角色 beliefs 与跨 edit invariant records

为当前 task target/reference 的非空物体维护最多两个短时几何状态：

\[
B_t^r=(\mu_t^r,\Sigma_t^r,F_t^r,t_{seen}^r,\text{motion mode},\text{binding id}).
\]

`mu,Sigma` 表达估计位置及其不确定性，`F` 是最近可见的语义/外观特征。可见点的离散程度与物体位姿估计协方差分开：物体尺寸大不等于定位不可靠。新物体没有历史时标为 uninitialized，不能以零 age/零协方差伪造高可信度。

- 可见时用实际观测更新；协方差缩小应取决于测量质量，不能只因虚拟 view 增多而重复计算同一份点云证据。
- static reference 短时遮挡时保持均值并增加不确定性。
- attached target 用 `T_EE,t T_EE,t-1^{-1}` 传播几何；需要保存 EE-object 刚性关系，不能仅将 EE 平移加到质心而忽略旋转。
- free object 快速增加位置不确定性，失去约束时停止精确定位。
- edit 切换时仅丢弃不再使用的几何状态，沿用物体按 ID 传递。`open(drawer)` 等 keep 关系另存有界 event/invariant ledger，含实体锚点、最后验证时刻和置信度；它不是完整场景图，但不能只靠上一个 pair 记住全部长链进度。

belief 在第 2.3 节的候选池中显式参与提议。单纯把记忆 feature 输入 head，并不能保证全遮挡目标仍在 top-k。

#### 4.5.3 分开当前投影证据、记忆和未知区域

renderer 暴露未归一化的 front depth；对当前点云中确实存在的 role point，且像素/深度有效时：

\[
M_{vis}=\mathbf 1[|z_r-z_{scene}|<\tau_z],\qquad
M_{occ}=\mathbf 1[z_r>z_{scene}+\tau_z].
\]

这里 `occ` 可能只是**已观测点在该虚拟 view 被别的已观测点遮住**，不必来自 memory。真正 sensor-hidden 且仅由 belief 支持的区域，使用按协方差扩散的 remembered distribution；仅有质心 belief 不能生成精确 amodal 表面。

每个非空 object view 的 side input 包含：

- 当前点的 visible/virtual-occluded membership；
- remembered density 及其 age/covariance/provenance；
- raw depth、depth-valid、in-frame mask。

未知深度/视野外不填零值高置信 evidence。渲染原场景时保留射线上可能的 occluder，避免 crop/near-plane clipping 删除遮挡物而制造可见性；stage-2/每个 object frame 均重新计算这些量。原先简单的四张 role maps 可作为消融，但不足以完整区分上述 evidence 来源。

这些通道通过小型 geometry encoder 与 VLM token/adapter 显式融合。现有 `mvt_single.py` 会取 `img[:,:,3:6]` 只送 RGB 到 PaliGemma；仅在输入 tensor 尾部追加通道不会生效。

#### 4.5.4 共同坐标、有效 support 与三维可观测性

在同一 world candidate `x` 上查询各 view 的 2D evidence，使用有效范围与可靠性权重：

\[
s_g(x)=\frac{\sum_{a,k}D_{g,a,k}(x)w_{g,a,k}(x)
\,Q_{g,a,k}(\pi_{a,k}(x))}
{\epsilon+\sum_{a,k}D_{g,a,k}(x)w_{g,a,k}(x)}.
\]

该式是一个 scoring heuristic，不应直接称为已校准的三维 posterior。局部尺度不同导致每像素对应的物理面积不同；需在统一 world grid/候选集合上定义分数和归一化，再比较候选或进行概率融合。

只有一个可靠正交 view 时，一个像素仍对应一条射线。至少需要两个非平行投影约束，或已验证的深度/接触面/位姿 belief 约束来消除缺失维度；不能把整条射线上的最高分任取为精确 3D waypoint。可以通过加权投影 Jacobian 的信息矩阵秩和 posterior covariance 检查可观测性。同源点云的多个虚拟 view 不作为独立传感器重复累积置信度。

scene front depth 约束的是物体表面，不一定等于 EE waypoint 深度；夹爪中心、抓取接触点和 pre-place 自由空间点必须区分。后续一致性损失使用联合可观测区域的重投影/分布一致性，不比较单 view 中不存在的唯一 world peak。

#### 4.5.5 拒绝后究竟执行什么

身份置信度、几何可观测性和 motion feasibility 分别决定行为：

| 情形 | 行为 |
| --- | --- |
| 已绑定 target，attached 可信，搬运路径已验证 | 可在预测不确定性预算内继续 TRANSFER |
| 身份明确，但新接触所需几何不可观测 | 保留绑定，请求 REOBSERVE；不能改抓可见 distractor |
| 候选漏召回/角色歧义 | 重新提议同 phase 候选或执行可用的信息采集动作 |
| keep relation 破坏 | RECOVER 或重新评估 edit；不标记 completion |
| SATISFY 已有稳定证据 | 允许进入 DISCONNECT；预测未来会完成不足以触发释放 |

`REOBSERVE` 是待实现的控制分支，不能当作免费、已存在的安全原语。移动相机或改变遮挡物需要真正的新 RGB-D 采集；相同固定相机下仅重渲染旧点云不会增加观测。退让/抬升动作只有在能改善观测且满足 collision、joint limit、attached/keep 约束时才执行，并记录超时、重试次数和观测增益。

原 BridgeVLA fallback 只恢复基线行为，不保证安全。在真实部署/带执行约束的实验中，baseline action 也须通过同一 execution gate；若没有已验证的观测/恢复动作，则保持当前可保持状态或终止该次尝试，计为失败/拒绝，不声称已恢复。

### 4.6 Phase-gated Object View Bank 与 pair-specific heatmaps

#### 4.6.1 每个物体有独立 frame，共享编码器参数

对每个去重后的候选 object 定义

\[
\mathcal F_i=(c_i,R_i,s_i),\qquad p_i'=R_i^\top(p-c_i)/s_i.
\]

第一版 `R_i=I`，以 object center 和 phase-dependent scale 做局部三正交投影，保留 `top/front/right` 方向。scale 需覆盖物体相关的夹爪偏移及动作空间，并设置上下界；不能用当前 GT action 设置测试时 crop。

这相当于 object-centered zoom：它能缓解有限像素分辨率的混叠，但同一射线上的几何重合在平移/正尺度变换后仍重合，不能改变前后遮挡顺序。真正改变投影方向的少量确定性旋转可单独消融；恢复源点云中缺失表面则需要新传感器证据或显式不确定的 memory。

\[
V_i^{1:3}=\operatorname{Render}_{top/front/right}
(\mathcal F_i^{-1}P_{scene};RGB,m_i,depth,visibility,provenance).
\]

渲染包含场景 context，不删除其他物体；view/crop 的有效范围与被裁掉的 occluder 必须检查。RGB 经过共享 PaliGemma，depth/membership/belief 经过显式 side encoder，融合为 `Z_i`。同一物体被不同 pair 使用时复用本 query 的 `Z_i`。动态 context/depth 每 query 更新；静态物体不意味着整组包含机械臂的 VLM features 可以跨时间直接缓存。

#### 4.6.2 pair heatmap 的预测对象和空间支持

对每个合法有序 pair

\[
z_{ij}=f_{pair}(z_i,z_j,T_{i\leftarrow j},L,h_t)
\]

使用共享 FiLM/low-rank head，得到 `H_g^{T,1:3}` 与非空 reference 的 `H_g^{R,1:3}`，以及每组的 in-frame/action-support mask。它们预测的是**同一个 EE translation waypoint 在各 frame 内的投影**；若要预测两处不同 contact sites，需要另一套 correspondence 与 EE-offset 映射，不能复用当前一致性标签。

令 `Omega_i` 是 frame 的有效 3D 动作范围，则

\[
D_i(x)=\mathbf 1[x\in\Omega_i],\qquad
\mathcal L_{hm}^{g,i}\text{ only applies if }x_{GT}\in\Omega_i.
\]

两物体相距较远时，waypoint 未必同时落入两个局部范围。此时该组输出 out-of-support/不可用，不能把标签 clamp 到边缘，不能产生全零目标去做 softmax CE，也不能作为错误 pair 或低执行置信度的证据。

- `APPROACH/CONNECT`：target view 通常提供动作定位。
- `TRANSFER/SATISFY`：reference view 常作为主要动作 anchor；target view 可仅提供身份、持握状态与几何条件。
- 两组都覆盖同一动作区域时，才在共同 world support 上施加重投影/分布一致性。共用点云带来的相关性需要保留，不假设六张图是六次独立测量。
- `reference=NULL`：使用 null embedding 和显式 EE/world anchor，不创建 reference frame、reference heatmap 或伪 reference 深度。

每个候选 pair 的 heatmap group 保持独立；`M=2` 保留两个动作假设并分别验证。不同 pair 的 logit/坐标平均不作为第一版执行方式。

#### 4.6.3 计算预算包含原始 refine 与真实 VLM forward

设 `U` 为早期合法性检查后需要编码的唯一非空候选物体。**保留原完整 baseline** 的第一版预算为

\[
N_{images}=6+3|U|,\qquad
N_{VLM\ groups}=2+|U|.
\]

`K_T=K_R=2`、最多四个唯一物体时，上限是 18 张图、6 组三视角 forward，而 baseline 为 6 张图、2 组。按 batch 并行可减少墙钟延迟，但不会消除 FLOPs/激活显存；不能预先把 3 倍组数称为低开销。若以 object bank **替换**原 refine，则是 `3+3|U|`，但不再享有完整 baseline 的严格回退；这应是独立变体。

pair heatmap head 的成本还包含空间维：

\[
C_{pair}=O(K_TK_R\,VHW\,r),
\]

其中 `r` 是低秩/调制通道宽度，`V<=6`；原始 3D cube recovery 还包含体素网格查询，需报告 grid size、显存和实际延迟，不能只计 pooled vector 的 `O(K_TK_Rd)`。预筛选 ground cost 放在 object VLM 编码之前；第一版对剩余小集合保留所有合法 pair 的轻量 heatmaps，再只对 `M<=2` 做较重的 pose/path feasibility 检查。

后验的 `M=1/2` 主要分配执行验证预算，不追溯节省已经完成的 view 编码。要进一步裁剪 object bank，需另评估早期筛选导致的 pair recall 损失。报告等训练预算、等实测延迟和等图像组数三个对照，区分表示收益与更多计算的收益。

### 4.7 Pair cost：身份排序与执行风险分别建模

保留 cost 作为核心，但将其输出定义成两个可解释量：

\[
C_G(g)=-\log p_G(g\mid o,L,h),\qquad
C_E(a,g)=-\log p_E(\mathrm{ready}(a)\mid g,o,h).
\]

`p_G` 评价候选是否符合任务的 target/reference 绑定；`p_E` 评价**给定该绑定及具体完整动作**是否在执行约束内可用。正确目标被遮挡时可以是 `p_G` 高、`p_E` 低；不能用较低 execution cost 把它换成任务不匹配的 distractor。completion 则是第三个量：当前 goal predicate 已经成立的证据，不属于这两个 cost。

#### 4.7.1 重编码前的 grounding cost

用 coarse features、语言、arity、相对几何及 tracked identity 构造

\[
E_G^{pre}(g)=\lambda_s C_{sem}
+\lambda_r C_{relation}
+\lambda_a C_{arity}
+\lambda_t C_{identity}.
\]

- `C_sem` 包括 target/reference 语义匹配；独立 confidence product 只作 baseline。
- `C_relation` 判断物体几何与语言关系是否匹配，例如指定容器、尺寸条件；不能因指定目标离 EE 较远而认定其身份错误。
- `C_arity` 确定 NULL 与物体/部件类型是否合法。高置信显式约束可硬过滤，预测不确定时保留软分数。
- `C_identity` 依据观测关联与 motion belief 支持原 binding；上限受控，接触失败/身份反证时允许纠错，不用无限 stay penalty 锁死错误目标。

所有输入必须在部署时可获得；候选真实 GT recall 只作训练标签/评估指标，不能作为测试时输入。grounding 分布还包含 `OTHER_G`，表示正确绑定未被当前候选集合覆盖。它与合法的 `reference=NULL` 分开；已知 tracked target 全遮挡不自动标成 OTHER_G。

第一版固定配额、去除重复实体、按候选集合训练 listwise 或多正例 set likelihood，再在 held-out episodes 上校准 `p_G`。校准/测试必须包含配额不足、NULL 比例变化、candidate drop 和 phase proposal 错误。softmax 的相对 energy 可用于排序，但未经独立标定的原始 energy 不直接作为跨 episode 拒绝阈值。

#### 4.7.2 heatmap 之后的 execution cost

为存活 pair 解码少量完整候选动作 `a=(x,R,gripper)`，预测/检查

\[
p_E(\mathrm{ready}(a)\mid g,o,h)
=f_E(\mathrm{observability},\mathrm{support},
\mathrm{pose/path\ checks},\mathrm{keep},\mathrm{belief}).
\]

输入证据包括有效三维可观测性、统一物理尺度下的不确定性、IK/joint-limit、路径和夹爪碰撞检查、接触有效性及 keep relation 约束。不能把只有 endpoint 的可达性写成整条轨迹安全；也不能将原模型预测的 `ignore_collisions` 标志当作 collision-free 概率。

heatmap entropy 是辅助特征，不是正确性的定义：多个合法抓取方式可导致高熵，错误预测也可很尖锐。不同 local scale/有效 support 的熵不能直接比较；多 view 共用点云，相关一致性不能重复放大信心。早期训练将这些诊断从动作 head stop-gradient 后送入小 risk head，避免网络仅通过尖化 heatmap 降低 cost。

risk head 需要对模型实际提出的动作做监督：离线 IK/路径检查、仿真分支或真实 outcome 可提供有效正负例。只有专家成功示范、role ID 或 `ignore_collisions` 标签时，不能宣称已经学到可执行概率；无法评估的项标为 unknown，并报告覆盖率。

NULL pair 仍需验证 target 及完整动作。其 reference 项用 applicability mask 表示不适用，并与真实 pair 在同一 ready-event 上校准；不能因为少了几项非负 cost 就天然更优。

#### 4.7.3 从 cost 到明确的控制选择

第一版采用有约束的决策次序，避免任意加权和让执行便利性补偿语义错误：

```text
grounding 不可信 / OTHER_G 高
  -> 重新提议同 phase 候选，或进入可用的信息采集分支
grounding 明确
  -> commit 该 pair，选择其内部通过 execution gate 的动作
grounding 明确但无可执行动作
  -> 保留 pair，REOBSERVE / RECOVER / 拒绝该次尝试
两个 pair 身份仍模糊
  -> M=2 分别验证与收集证据；执行便宜不单独构成改绑依据
goal predicate 已稳定成立
  -> 完成当前 edit，提议下一 edit 和角色
```

若任务明确许可多个语义等价的 pair，可在该合法集合内按 execution risk 选择；这种自由度来自任务语义许可，不能由易执行性反推。对必须辨认特定实例的任务，身份歧义仍需补充证据。

阈值基于 held-out 的 grounding error 与 conditional execution failure 风险预算，分别记录错误接受率和 coverage；不使用任意绝对总 energy 阈值。若后续希望使用统一决策 cost，应写成明确损失的期望风险：

\[
J(a,g)=c_{wrong}[1-p_G(g)]
+c_{fail}p_G(g)[1-p_E(\mathrm{ready}(a)\mid g)]
+c_{time}\Delta t,
\]

并与实际可用的 observe/recover 动作代价比较，声明概率近似、损失权重及语义约束；这一扩展不属于第一版。它不是用廉价执行任意补偿错误任务身份的许可。

候选排序、拒绝与完成输出分别为 `ground_decision`、`execution_decision`、`completion_evidence`。runtime reason 至少区分 `missing_candidate / ambiguous_binding / unobservable_geometry / out_of_support / infeasible_action / keep_violation`，防止把所有问题归为同一个 reject。

## 5. 监督如何获得

### 5.1 将正确绑定、动作可执行与关系完成分别标注

利用 simulator instance、joint state、gripper、3D geometry 与 demonstration，离线生成以下标签，并记录置信度与可用性：

1. **角色/arity**：task target、reference/NULL、谓词参数 mapping。合法等价实例形成多正例集合；同类 hard negative 必须确实违背语言或关系。
2. **relation/edit**：contact/attached、on/inside/aligned、open/closed 等 predicate 的稳定事件。启发式阈值需要人工抽检、hysteresis 与 unknown 标签；单凭距离近不能直接判定 attached。
3. **进度与完成**：统一 `u=1` 为完成；分别缓存当前完成 `d_t`、下个 action interval 完成的 hazard target。未来帧可用于离线标签，不能作为策略输入、测试 crop 或恢复状态。
4. **membership 与视角**：当前可见点 membership、virtual visibility、belief provenance。Oracle shape/pose 只作诊断上限；预测部署分支不能利用相机从未观测的 GT 点进行输入补全。
5. **local action**：将示范 EE waypoint 投影到模型实际使用的 frame，保存 in-frame/support mask；out-of-support 不做 pixel loss，不能 clamp 或作为错误 pair 标签。
6. **grounding**：正确 pair 即使全遮挡，只要通过有效 identity belief 保留在候选中仍为正例；正确绑定不在候选集合中才监督 OTHER_G。新增观测不足以确认身份时可采用软标签/abstention。
7. **execution**：针对候选的具体完整动作，用 IK、路径/碰撞/接触检查或额外 rollout 获取 ready/outcome 标签。错误角色但物理可执行的动作，可以同时是 ground-negative、exec-positive；正确角色但动作不可达则相反。无法验证的部分保持 unknown。
8. **恢复**：单独记录再观测动作、前后信息变化、keep violation、失败原因和耗时；无相应数据/控制器时不声称已学得 REOBSERVE。

visibility/memory 训练片段需覆盖 visible -> hidden -> visible，使用实际时间间隔传播 motion model。GT candidates 用于 warm-up；随后渐进替换为 predicted candidates、predicted crop 与推理时相同的记忆状态，避免 Oracle phase/centroid 泄漏掩盖真实误差。

### 5.2 损失与梯度路径

\[
\mathcal L=\mathcal L_{action}
+\lambda_{role}\mathcal L_{role}
+\lambda_G\mathcal L_G
+\lambda_E\mathcal L_E
+\lambda_{edit}\mathcal L_{edit}
+\lambda_u\mathcal L_u+\lambda_d\mathcal L_d
+\lambda_\beta\mathcal L_{\beta}
+\lambda_{keep}\mathcal L_{keep}
+\lambda_{cf}\mathcal L_{cf}.
\]

- `L_action`：原有 translation、rotation、gripper 与 ignore-collision-label loss；后者不等价于碰撞风险监督。local heatmap 仅对语义正确且 GT waypoint 在 support 内的 pair/view 计算。
- `L_role/L_G`：point/slot matching、角色提议、多正例 grounding set likelihood（含 OTHER_G）、NULL arity 与去重。合法多正例可用 `-log sum_{g in G+} p_G(g)`，不强迫选中特定示范实例。
- `L_E`：具体完整动作的 ready/outcome BCE 或其他 proper scoring loss，明确 mask unknown 项；初期对 action-head diagnostics stop-gradient。按实际采样的模型动作分布训练/校准，不能只看成功专家数据。
- `L_edit/L_u/L_d/L_beta`：operator/goal、归一化进度、已完成事件和未来 hazard 分开监督。
- `L_keep`：有真实 transition 标签时监督动作后的 invariant；纯 learned next-state self-consistency 不能作为安全保证，需报告预测误差并做独立检查。
- `L_cf`：仅在换角色/关系确实改变当前合法动作集合时使用 action-likelihood margin。两种后续目标共享同一抓取动作时，不能强迫当前动作不同来证明 causal dependence。

固定拟合后的候选生成和动作策略，在不重叠的 episode calibration set 上分别校准 grounding 与 execution，阈值选择与最终测试集分离。temperature scaling 是可选后处理，不替代 missing-candidate、NULL shift、错误 phase、未知几何等覆盖性检验。

### 5.3 训练与闭环验证顺序

1. 复现完整 B0/O2，验证关闭 residual 返回同一 checkpoint 的完整 baseline action。
2. Oracle roles/edit 的 ARE-only 与 Oracle object-view 的 PairHM 分别建立上限；一个分支失败不自动否定另一个分支。
3. 固定小候选集和已有 teacher edit，验证 local support、frame transforms、pair-specific heatmaps 与相同计算预算的对照。
4. 学习 predicted slots/grounding，使用部署时的 role mapping、NULL 配额和 reserved beliefs；报告候选 recall。
5. 为实际候选动作生成 execution 数据，学习 risk head；若缺数据，先以可核验检查器替代，并明确 coverage。
6. 用短序列逐渐替换 teacher edit/centroid，训练 progress、completion、memory 与 recovery；训练时使用部署一致的状态更新规则。
7. 最后联合 fine-tuning 并在 held-out episodes 校准；报告所有开销、拒绝动作数和 end-to-end success。

第一版固定 `K_T=K_R=2`、`M<=2`、固定正交方向；先确认这些有界机制有效，再考虑任意旋转、动态大候选集和更长 memory。

## 6. 实验必须回答什么

### 6.1 分开表示、计算和执行选择的收益

保持 backbone、训练数据和监督可用性一致，同时报告等参数、等训练预算和等实测推理预算对照。新增 role/phase/risk 标签应提供给 matched auxiliary-head baseline，避免把额外监督当成结构贡献。

| 代号 | 条件 | 要回答的问题 |
| --- | --- | --- |
| B0 | 原完整 BridgeVLA | 同 checkpoint 原动作基线 |
| O2 | B0 + GT task role prior | 已有 Oracle 空间先验上限 |
| Obj | B0 + predicted role prior | 单独 object grounding 是否足够 |
| ObjBank | phase 小候选集 + per-object views + 独立 object heatmaps | 局部尺度与有界搜索收益 |
| PairHM | ObjBank + 每 pair heatmaps，固定 identity selector | pair conditioning 是否有效 |
| GroundCost | PairHM + structured grounding ranking/OTHER_G | 是否减少错误绑定与漏召回后的错误接受 |
| ExecGate | GroundCost + 完整动作检查/校准 execution risk | 是否在相同绑定下改善执行选择和拒绝 |
| Phase/Rel/Obj+Phase+Rel | 独立 phase、relation 或匹配容量的组合 heads | ARE 联合状态的收益 |
| ARE-static/full | 无持续状态 vs 有 completion、belief/ledger/recovery | 状态更新与长链收益 |
| Oracle-role/ARE | GT roles 或 GT roles+edit/boundary | 与可部署预测量明确区分的上限 |
| View-selection baseline | 单一 task region + virtual view scoring/zoom | 与直接视角方法比较 |

ARE-only 与 PairHM 分别验证；最终比较完整组合与相同监督/计算的 independent-head 和 view-selection baselines，不只比较 B0。ActiveVLA 已直接采用 BridgeVLA backbone 并做视角评分/zoom，应作为近邻对照；TVVE 的任务视角选择提供另一条对照路径，不能仅在 related work 中提及。

### 6.2 组合、身份和候选覆盖测试

- held-out object-by-relation、同类 distractors、合法 task-role reversal、未见 edit chain；
- reference arity/NULL 比例变化；需要但不可见的 reference 与真正 NULL 分开统计；
- observed -> tracked -> observed 身份关联，以及 committed 对象短时消失后是否仍在候选池；
- 正确 pair 不在候选集合、候选数量不足、重复 slot、错误 phase proposal；
- 单帧相似但已完成前缀不同的 observation aliasing；keep ledger 容量与遗忘/重新验证策略；
- 正确但难执行、错误但易执行、多个语义合法实例、同一物体有多个合法动作四类交叉情形。

### 6.3 区分三类观测问题和局部 support

| 受控变化 | 检验 |
| --- | --- |
| fixed directions + object center/scale | 仅局部分辨率/像素混叠收益 |
| 同射线前后表面，保持源点云不变 | 平移/缩放不能改变遮挡顺序 |
| 改变虚拟投影方向 | 在已有点云上的投影重合是否减轻 |
| 真正 sensor occlusion / camera dropout | 源证据缺失，不能借 GT amodal 点补输入 |
| gripper 自遮挡与 disappear/reappear | belief、identity matching 与完成判断 |
| waypoint 只在一个 object crop 内 | in-frame mask、loss 和 consistency 是否正确 |
| 仅一个可靠正交方向 | 缺失深度是否被不确定性表达 |
| 有/无真实信息采集动作 | REOBSERVE 额外观测的价值与动作/延迟代价 |

对照包括等预算的更多/随机 views、frame stacking、object zoom only、pair heatmaps only、reserved belief only、geometry side input only、execution checker only、完整组合。每个消融需同时报告识别错误、动作错误和新增计算；不能把删除源点的一种增强代表所有真实遮挡。

### 6.4 指标与校准协议

- 整体任务成功率、completed relations、错误绑定、keep violation、premature/delayed completion 与 recovery success；
- role proposal recall、多正例 joint pair recall、OTHER_G/NULL precision-recall，按 observed/tracked source 分组；
- grounding 的 NLL/Brier/ECE 与错误接受率，execution 的 conditional failure calibration/risk-coverage 分别报告；
- 不同候选数量、NULL 比例、局部尺度、视角覆盖、motion mode 下的校准；
- out-of-support、三维不可观测、候选漏召回、路径失败、身份纠错失败的分项计数；
- 同 checkpoint B0 的 rescue/harm；report 基于独立原路径，不用当前 modified crop 的 trans_base 冒充；
- PaliGemma 图像组、VLM FLOPs、side encoder、3D recovery、IK/path checks 的平均与 p95 延迟、峰值显存；
- 单 episode 总 query/动作数、实际新观测次数、重试/超时与总耗时；拒绝和超时计入总体失败，另报 conditional accepted success。

校准集与测试集按 episode/任务划分，不能通过相邻帧泄漏。risk-coverage 用置信区间判断整体趋势；不因有限样本的单点非单调直接 No-Go。阈值在验证集选定后冻结，不从测试成功率反调。

### 6.5 干预验证与可识别性

固定观测，分别干预 task-role binding、goal、ground posterior、execution gate、当前 completion evidence 和未来 hazard。重点验证：

1. 替换 target/reference 后，在任务语义要求动作改变的状态中，候选动作是否相应改变；
2. 固定 ground identity、降低 execution readiness，是否保留身份并拒绝/观测，而非改抓 distractor；
3. 固定执行证据、替换 ground posterior，是否改变绑定选择；
4. 只提高未来 hazard，但当前 goal 未成立，是否避免提前切换/释放；
5. 当前 goal 成立但 keep 破坏，是否进入 recovery；
6. 关闭 residual 是否恢复原 policy proposal。

对早期共享同一抓取动作的不同目标，不强求单步 action 改变；检查它们真正分叉时的动作或整段 outcome。保留 baseline bypass 的方法只能证明 ARE conditioning 的因果影响，不能据此宣称所有动作信息都经过一个硬 bottleneck。

### 6.6 Go / No-Go 按分支判定

- ARE-only 无闭环收益：收缩 ARE 表示主张，继续独立评估 PairHM；
- PairHM 不优于相同计算/监督的 independent heatmaps 或直接 view selection：不把 pair grouping 列为贡献；
- grounding cost 不改善错误绑定/OTHER_G 风险：退回已验证的 confidence baseline；
- execution gate 只提高 accepted success，但拒绝/再观测开销导致总体成功率无改善：报告取舍，不能称整体更强；
- 预测部署分支只能依赖 Oracle phase/GT 隐藏几何获得收益：尚不具备可部署证据；
- 错误但易执行的 distractor 被选中、单 view 被当作精确三维、out-of-support 被强制监督：先修正确性，再比较指标；
- zero-residual 等价性或计算预算不成立：修订实现/声明，不用当前结果支持 baseline rescue/harm；
- 无可执行 REOBSERVE 控制器：从主方法中移除此收益项，显式统计拒绝/失败。

各项采用多 seed、置信区间与错误分解；不事先假定固定的成功率提升阈值，也不把某一分支的 No-Go 外推为所有 object-centric 方法无价值。

## 7. 与当前仓库的落点

### 7.1 可直接复用

- `finetune/bridgevla/models/oracle_prior.py`：已有 Target/Reference 选择、relation descriptor、两通道 prior 与 relation-gated adapter；
- `finetune/bridgevla/mvt/mvt.py`：已有 stage-1/stage-2 prior 构建与 fusion；
- `finetune/bridgevla/mvt/mvt_single.py`：已有 adapter 前的 feature `x`，并可同时产出 `trans_base`；
- `finetune/bridgevla/libs/point-renderer/point_renderer/rvt_renderer.py`：默认 `rvt2.yaml` 实际使用的三视角 renderer；可复用 `top/front/right` cameras、`pc_depths` 与 3D heatmap recovery，但当前 `get_pt_loc_on_img` 丢弃 camera-space depth，且 recovery 跨 view 无权平均；
- `finetune/bridgevla/mvt/renderer.py`：PyTorch3D fallback，定义五视角但不支持 `rend_three_views=True`；不要用它描述默认 BridgeVLA 的 view 数量；
- `finetune/bridgevla/models/bridgevla_agent.py`：已有 Oracle point selection 与 base/raw/fused translation loss 日志。

### 7.2 最小代码增量与接口契约

按以下依赖顺序实现，并先用 Oracle teacher 和小规模回放验证：

1. **Baseline path**：完整保留 coarse、原 waypoint refine、原动作头；把原 full action 与当前 crop 的 `trans_base` 区分命名。新增分支关闭时不改变其输入或重新解码。
2. **State/proposal**：显式 event/keep ledger、任务角色到谓词参数的 mapping、observed + tracked candidate union、固定 NULL 配额和去重。
3. **Grounding pre-cost**：在额外 VLM forward 前计算 `p_G/OTHER_G`；筛选合法、有足够身份支持的候选，允许重新提议。
4. **Geometry interface**：默认 point renderer 暴露 raw depth、projected (u,v,z)、in-frame/depth-valid/provenance；区分虚拟遮挡与真正未观测几何。
5. **Object bank**：每个唯一候选沿 batch 维展开三视角，记录可逆 frame、support 与真实时间戳；新增 side encoder 将 membership/depth/belief 接入 RGB VLM features。不能只追加被 RGB slice 丢弃的通道。
6. **Pair head**：共享参数产生每个合法 pair 的独立 heatmap group；mask out-of-support supervision，在共同坐标上构造受 gate 控制的 residual proposal。
7. **Action/risk**：生成最多两组完整 pose/gripper proposal；更新与最终 waypoint 一致的 feature sampling，验证最终动作及实际 controller path 后再选择一组。selection 后任何动作修改均需重新检查。
8. **Execution/recovery**：binding ambiguity、geometry unknown、IK/path failure、keep violation 分开处理；若没有有效 REOBSERVE 控制器，显式拒绝，不隐式调用未经验证的抬升。
9. **Profiling/calibration**：完整路径计入 PaliGemma、side encoder、3D recovery、risk checks、再观测和原 refine 成本；冻结策略后做独立 calibration。

新增字段至少包括：

- 任务/状态：`task_target_id`、`task_reference_id/null`、`predicate_bindings`、`edit_goal`、`progress`、`completed_now`、`hazard_next`、`keep_ledger`；
- 候选：`candidate_id`、`source=observed/tracked`、`membership`、`frame/inverse`、`support`、`belief_initialized/age/covariance`、`reserved_slot`；
- 输出：`ground_posterior`、`other_ground_prob`、`pair_heatmaps`、`support_mask`、`candidate_full_actions`、`execution_ready_prob`、`check_known_mask`、`decision_reason`；
- 诊断：proposal recall、身份关联错误、ground/exec calibration、错误接受率、out-of-support rate、zero-residual action equality、端到端延迟/显存和 recovery outcome。

第一版只维护 committed pair 的最多两个几何 beliefs，以及有界 invariant ledger；ledger 容量与超限行为需配置。不缓存所有候选的长期几何，也不为每个 pair 复制一套 VLM/action expert。

### 7.3 实现前后的验收反例

| 构造场景 | 必须成立的结果 |
| --- | --- |
| residual 初始化为零，或关闭 gamma | policy proposal 与同 checkpoint 原两阶段 action 一致；执行 gate 对两分支按同一规则处理 |
| 正确物体被挡、错误 distractor 易抓 | ground identity 保留；execution 可拒绝，不能仅因易抓而改绑 |
| committed target 全遮挡 | belief 候选仍在保留名额；不得被删除后用 temporal cost 假装修复 |
| waypoint 仅在 reference crop 内 | target-view pixel loss 与交叉一致性被 mask；不得 clamp 到 target 边缘 |
| 只有一个正交 view、无额外深度约束 | 沿射线保持不确定性；不输出被伪装成精确定位的任意深度 |
| 两 pair 的峰值分别在两个物体上 | 选定一个 pair 的完整动作；不输出两峰中点 |
| 增加重复 slots 或 NULL 比例改变 | 去重与 applicability 正确；校准指标单独报告，不把 raw energy 阈值当通用标尺 |
| 固定方向、只平移/缩放，前后点同射线 | 遮挡顺序不变；不将 zoom 增益记为新观测 |
| CONNECT(block,NULL) -> TRANSFER(block,bowl) | block 的 task target ID/belief 连续；不因 attached 谓词参数位置改变而变成 reference |
| 返回动作的 rotation/waypoint 被后处理改变 | 重新运行 execution check，原检查结论不得直接复用 |
| 没有可用的信息采集动作 | 明确拒绝/结束尝试并计入失败；不把 baseline fallback 称为安全恢复 |

这些是尚待实现与实测的验收项，本次文档修订不代表已通过机器人闭环测试。

## 8. 两个任务例子

### 8.1 Put the red block in the blue bowl

```text
APPROACH:
  task_pair = (red block, NULL)
  goal      = near(EE, red block)
CONNECT:
  task_pair = (red block, NULL)
  goal      = attached(EE, red block)
TRANSFER:
  task_pair = (red block, blue bowl)
  goal      = near(red block, blue bowl)
  keep      = attached(EE, red block)
SATISFY:
  task_pair = (red block, blue bowl)
  goal      = inside(red block, blue bowl)
  keep      = attached(EE, red block)
DISCONNECT:
  task_pair = (red block, NULL)
  goal      = detached(EE, red block)
  keep      = inside(red block, blue bowl)
```

block 在这条链中一直是 task target；变化的是 reference arity 与 goal predicate。DISCONNECT 不需要 reference action view，但 bowl 的锚点和 inside completion evidence 仍保留在 keep ledger 中。

若 TRANSFER 时 block 被 gripper 挡住，保留 tracked block；若 bowl 身份明确但接触几何不可观测，保持 `(block,bowl)`，进入有约束的 REOBSERVE，而非改选另一个可见 bowl。只有任务语义允许多个等价容器时，才把它们作为多正例候选。

### 8.2 Open drawer and put the block inside

```text
APPROACH:    pair=(handle,NULL),  goal=near(EE,handle)
CONNECT:     pair=(handle,NULL),  goal=attached(EE,handle)
ARTICULATE:  pair=(handle,drawer), goal=open(drawer), keep=attached(EE,handle)
DISCONNECT:  pair=(handle,NULL),  goal=detached(EE,handle), keep=open(drawer)
APPROACH:    pair=(block,NULL),   goal=near(EE,block), keep=open(drawer)
CONNECT:     pair=(block,NULL),   goal=attached(EE,block), keep=open(drawer)
TRANSFER:    pair=(block,drawer), goal=near(block,drawer),
             keep=attached(EE,block),open(drawer)
SATISFY:     pair=(block,drawer), goal=inside(block,drawer), keep=open(drawer)
DISCONNECT:  pair=(block,NULL),   goal=detached(EE,block), keep=inside(block,drawer)
```

handle 与 drawer 的 part/whole link 显式记录，不把它当普通 self-pair；如果感知系统不能区分部件，则需定义 task mapping 并单独报告该限制。drawer 在切换操作 block 后可能退出当前 pair，但 open(drawer) 的证据不能因清空上一 pair 的 memory 而消失。若 drawer 意外关闭，进入 recovery，不能因为上一 open edit 曾完成就继续插入。

## 9. 论文叙事与风险控制

### 9.1 推荐 claim

不要写：

> We predict objects, relations and phases for long-horizon manipulation.

推荐写：

> We test whether the active relation edit (the role-bound change a manipulation action is currently trying to realize) forms a compact control state for end-to-end VLAs. This structured conditioning interface grounds the acting entities, specifies the desired relational change and exposes its completion evidence, with separately calibrated grounding and action-readiness estimates.

三条 claim 必须分别由实验支撑：

- **compactness**：明确 active beliefs/ledger 容量、状态字节和计算；比较 matched-capacity `Obj+Phase+Rel` 与 full-graph alternative，性能更优本身不证明最小充分性；
- **compositionality**：held-out bindings/chains 与 role-swap interventions；
- **long-horizon execution**：更少 premature/delayed switching 和 keep-relation violation，而不只是更低 action loss。

### 9.2 最大 novelty 风险

最大的风险不是方法太简单，而是 ARE 被审稿人理解为 `relation-labeled options`。应对方式不是继续加模块，而是把下面三点做扎实：

1. ARE 是保留 baseline bypass 的可干预 conditioning interface；它不构成所有动作信息必须经过的硬 bottleneck，也不因联合训练自动成为全选择流程可微的系统；
2. object role、relation goal、termination 和 invariants 共享同一个结构，matched independent-head baseline 能检验联合结构；
3. held-out binding 与 forced-edit 实验显示其因果、组合行为，而非只展示可视化。

当前使用 `relation-edit-conditioned VLA` 作为主张。即使因果与组合实验成立，`minimal sufficient control state` 仍需额外的状态充分性、压缩和历史混淆实验支持。

### 9.3 明确降级为附录或未来工作

- **Oracle-delta distillation**：与 PFD 的 privileged residual 思路过近；
- **完整动态图/graph planner**：与 EPoG、POT-VLA、TAX、ReKep 重叠且系统复杂；
- **通用 phase memory**：与 ChainVLA、INDI、ProgressVLA 类工作重叠；
- **future object/world prediction**：计算重且与 DeltaVLA/world-model 路线重叠；
- **无界 full-scene proposals 或每个 pair 独立 action expert**：当前只做 phase-gated bounded top-`K`、共享 object encoder 和轻量 pair head；
- **任意 learned camera rotation / 大规模 view search**：先固定每个候选物体的 `top/front/right` 三视角，只学习候选、尺度、pair heatmap 与 cost；
- **contact correspondence/site head**：对 insertion/assembly 有价值，但先作为 ARE 的细化，不单列贡献。

## 10. 实施优先级

```text
P0  固定角色/谓词 mapping、NULL 配额、support、completion 与两种 risk 标签；
    准备第 7.3 节的小型反例与 zero-residual 验收
P1  分别验证 Oracle-ARE-only 和 Oracle PairHM；
    完整保留 baseline，记录 6+3|U| 图像与 2+|U| VLM 组的真实开销
P2  observed+tracked 候选、pre-encoding grounding cost、OTHER_G 与 NULL calibration
P3  support-aware pair heatmaps，完整动作构造及 execution 检查；
    测试正确但不可执行 / 错误但易执行 / 多个合法动作
P4  已完成证据 d、future hazard、identity-preserving beliefs、
    有界 keep ledger 与可用的 REOBSERVE/RECOVER 控制
P5  predicted phase/candidates/crops 的部署分布训练与独立 calibration
P6  matched-compute / compositional / occlusion / arity-shift /
    candidate-count / risk-coverage / full closed-loop evaluation
P7  依据实际瓶颈再探索确定性旋转、refine replacement、
    dynamic compute 或更长 memory，逐项取消基线等价等已失效声明
```

ARE 表示与 pair-view 机制分别验收；一个无收益，不自动否定另一个。cost 的价值也应分解：grounding 是否减少错误绑定，execution gate 是否降低错误接受，新增拒绝/观测开销是否仍带来整体成功率提升。先证明这些可测结论，再扩大架构与理论主张。

## 11. 修订后的方法边界

当前方案保留用户提出的核心：只处理当前 phase 的少量 target/reference 候选，每个唯一物体拥有自己的三视角，每个合法有序 pair 拥有独立 heatmap group，reference 可为 NULL。

实现上采用完整 BridgeVLA baseline + 受 gate 控制的 object/pair residual；额外输入显式携带几何、support 与证据来源。固定正交方向的 object views 提供局部尺度优势，不产生新的传感器观测。严格 baseline 回退需要计入原 refine 的计算，不能同时声称只有 `3+3|U|`。

cost 分别表达身份可信度与给定动作的执行风险，已完成证据独立控制正常 phase 推进。全遮挡目标通过保留的 belief 候选维持身份；超出局部视野、三维几何不可观测、动作不可达分别处理。候选歧义可以保留多个 hypothesis，但最终执行一个绑定一致的完整动作。

论文主张暂限定为可干预的 relation-edit conditioning、object-view 复用及可校准的执行选择。是否达到最小充分控制状态、是否比 matched independent heads 或直接虚拟视角选择更有效，都需由实验决定；目前仍是未实现的研究设计。
