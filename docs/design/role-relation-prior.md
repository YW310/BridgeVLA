# BridgeVLA-ARE：角色绑定的关系编辑

[文档索引](../README.md) · [详细设计](role-relation-details.md) · [Oracle 交接](../handoff/oracle-prior.md) · [O2 实验](../experiments/o2-training.md)

> 更新：2026-09-10。本文是未实现的研究方案。主文档定义建议 MVP；详细设计保留完整扩展、公式和验收条件。

**建议 MVP：phase-conditioned top-k object pairs + per-object view bank + pair-specific heatmaps + calibrated grounding cost。**

Learned execution risk、主动再观测和通用长期记忆暂不作为首版贡献。

## 1. 主线：用 ARE 组织 object、relation 和 phase

**ARE（Active Relation Edit）表示：当前操作谁、相对谁、要改变什么关系，以及哪些关系必须保持。**

- **Object**：绑定当前 target / reference，不要求先预测完整场景图。
- **Relation**：物体之间或物体与 EE 等锚点之间的状态，例如 attached、inside、open。
- **Phase**：当前 ARE 的执行状态，包含目标、进度、已完成证据与保持约束。

Long-horizon 操作不是每步切换 relation，而是持续执行一个 edit，目标稳定成立且 keep 约束满足后再推进。抓取失败、身份纠错和恢复单独记录，不能冒充完成；“预计即将完成”也不能替代“已经完成”。

例如放 block 入 bowl：抓取 `(block,NULL)` → 搬运／放置 `(block,bowl)` → 释放 `(block,NULL)`。block 的 target 身份连续；释放阶段仍保留 `inside(block,bowl)` 的 keep 证据。

## 2. 当前 phase 的候选、物体视角与 pair heatmap

### 候选只保留小集合

首帧、完成边界或恢复时，由 coarse observation、指令和已完成记录提议 edit；普通 query 沿用并校验上一 edit。phase 不可信时允许重提议，不能永久屏蔽正确物体。

第一版固定 `K_T=K_R=2`，最多四个合法有序 pair：

- 候选池先合并当前观测与 committed object beliefs，再做 top-k；短时遮挡对象在固定角色配额内保留名额。
- `K_R` 包含 NULL 配额：NULL 合法时保留一个 NULL 和至多一个非空 reference；reference 必需时取至多两个非空候选。
- NULL 表示“不需要第二个场景物体”，不表示“看不见”。target 不为 NULL；去重、self-pair 与部件关系需显式处理。
- task target/reference 与谓词参数顺序分开。EE、world 等锚点不占场景候选名额。

### 每个唯一物体有视角，每个合法 pair 有 heatmap

1. 每个唯一非空物体建立局部 frame：中心 + 有界尺度，第一版固定 top/front/right 三个方向。
2. 渲染保留场景 context；同一 query 内复用每个物体的编码，不给每个 pair 复制 VLM。
3. 共享轻量 pair head 以 T/R 特征、相对几何、指令及 edit 为条件，输出该 pair 独立的 heatmap group。
4. T-frame 与非空 R-frame 的 heatmap 都预测**同一个 EE waypoint**，不是分别预测两个接触点。NULL 不生成 reference frame 或 heatmap。

每个 frame 显式记录有效三维范围（support）。waypoint 不在某个 crop 内，就屏蔽该组 pixel loss；只在共同有效区域约束一致性，不把标签压到边缘。不同 pair 保持独立动作假设，最终选一组完整动作，不平均两个物体上的峰值。

### 遮挡与计算边界

默认 BridgeVLA 是 coarse/refine 各三个正交视角，即 `3×2`，仍来自同一份可见点云。object-centered zoom 可改善局部分辨率，但固定方向的平移／缩放不改变同射线遮挡顺序，也不补回真实相机未见的表面。

当前点、虚拟投影遮挡、记忆与未知区域分别编码，并显式融合 depth/membership/belief；只追加 RGB 输入之外的通道不会自动生效。单个正交 view 不能独自确定唯一三维点，记忆也不能伪装成精确表面。

保留完整 baseline 时，输入预算为 `6+3|U|` 张图、`2+|U|` 组三视角 VLM 编码；`|U|≤4` 时最多 18 张图、6 组。pair head、3D recovery 和执行检查另计，是否可控必须实测，不能只凭候选少下结论。

## 3. Pair cost：先选对对象，再判断动作是否可用

| 输出 | 回答的问题 | 决策用途 |
| --- | --- | --- |
| Grounding：`p_G(g)` | T/R 绑定是否符合任务？ | 额外物体编码前排序；含候选漏召回的 `OTHER_G` |
| Execution readiness | 给定绑定下，这个完整动作是否可执行？ | 首版由检查器给出；有 proposal outcome 后再学习 `p_E` |
| Completion：`d_t` | 当前目标是否已经稳定成立？ | 与 keep 证据共同控制正常 phase 推进 |

前两者可写为负对数 cost，但第一版**不做任意加权总分**：易执行不能补偿错误身份；heatmap 尖锐也不等于正确或可执行。

- 身份不明确：重提议或获取信息；`OTHER_G` 与合法 NULL 分开。
- 身份明确且动作通过检查：提交该 pair 的动作。
- 身份明确但几何未知／动作不可达：保留绑定，再观测、恢复或拒绝，不能改抓容易抓的 distractor。
- 最多对 `M≤2` 个完整 proposal 做较重检查；仅当任务允许多个等价绑定时，才可在它们之间按执行风险选择。
- completion 只看已成立证据；遮挡、cost 升高或未来 completion hazard 都不直接触发完成。

首版将 **grounding cost** 作为 learned pair score；support、IK 和碰撞只作为可核验的 validity gate。只有采集到模型 proposal 的成功／失败数据后，才学习并校准 execution probability。两类分数必须分开，未知检查项保留 unknown。

## 4. 最小实现：完整 baseline + 有条件的局部修正

保留原 coarse、waypoint-centered refine 与完整动作路径；新增 object/pair 分支只贡献有效 support 内、受 gate 控制的 translation residual。

- **可验证回退**：zero-init 或 gate=0 时，直接返回同 checkpoint 的原完整 proposal，不重新裁剪／解码。`trans_base` 只是当前 crop 的局部对照，不能代替原两阶段 baseline。
- **完整动作一致**：最终 waypoint 对应的 rotation/gripper 与动作 feature 采样要一致；execution check 检查实际返回动作，后处理改变动作就重检。
- **限制修正范围**：第一版只在 baseline refine support 内修正；大幅重定位触发新 coarse query，不静默 clamp。
- **有界状态**：最多维护 committed pair 的两个短时几何 beliefs；跨 edit 的 keep 证据另存有容量限制、实体锚点和更新时间的 ledger。
- **恢复不是免费原语**：REOBSERVE 必须有真实新观测和可验证控制器；没有时显式拒绝／结束尝试。baseline fallback 本身不保证安全。

复用当前 O2 的 prior、adapter/fusion 与日志接口，但不把现有 Oracle 实现等同于 ARE。

首版只实现：

1. teacher phase/edit 下的 T/R/NULL 候选与 calibrated grounding score；
2. 去重后的 per-object 三视角编码；
3. 每个合法 pair 的独立 translation heatmap 和 baseline residual；
4. event-gated phase transition。

首版不实现 learned execution-risk head、主动 REOBSERVE、完整图规划、任意视角搜索、长期全场景记忆或每 pair 独立 expert。

## 5. 验证顺序：先证明有用，再增加预测与控制

1. **契约与基线**：通过 zero-residual 等价、NULL、support mask 与 pair 不混合等小型反例。
2. **Oracle PairHM**：固定 GT phase 和 pair，只验证 per-object views 与 pair heatmap 是否优于等计算的 object-view／独立-head baseline。
3. **预测 pair**：加入 phase-conditioned top-k、grounding cost 和 `OTHER_G`，报告 pair recall、校准和闭环收益。
4. **关系切换**：加入 completion/keep 事件，验证长链中的过早切换、延迟切换和身份连续性。
5. **可选扩展**：只有瓶颈明确且具备数据时，再加入 learned execution risk、belief/recovery 或主动观测。

最终至少报告：总体闭环成功率、错误绑定、过早／延迟切换、keep violation、grounding 校准、validity-gate coverage，以及端到端延迟、显存和总动作数。拒绝、超时和再观测代价不能从总体结果中剔除；只有启用 learned execution risk 时才报告其校准。

PairHM、grounding cost 和 relation transition 逐级决定是否继续；上一阶段没有收益，不用后续复杂模块掩盖。只有得到匹配预算下的实测证据，才主张结构或效率收益；暂不宣称“最小充分状态”。

完整公式、监督项、接口字段、任务例子与 11 项验收反例见 [详细设计](role-relation-details.md)。
