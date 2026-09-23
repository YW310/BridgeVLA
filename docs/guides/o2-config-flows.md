# O2 配置简易流程图

[文档索引](../README.md) · [Semantic-GT 数据](semantic-gt.md) · [O2 实验](../experiments/o2-training.md) · [联合训练](../experiments/object-conditioned-joint.md)

本文按当前 **finetune/RLBench/configs/rlbench_o2_*.yaml** 整理数据流。实线表示策略
forward 或数据准备的主路径，虚线表示只在训练期使用的监督。冻结范围还取决于启动参数：
带 **--train_object_adapter_only** 时只训练 object slots 与 Oracle adapter；joint 配置按对应
实验文档启动时不使用该开关。

## 一览

| 配置 | 策略使用的 object 来源 | instruction / 完整动作共享 | 推理时是否需要 GT |
| --- | --- | --- | --- |
| **rlbench_o2_gt_instance.yaml** | 启发式 replay T/R 点云 | 否 / 旧 relation 路由 | 是 |
| **rlbench_o2_semantic_gt.yaml** | 严格 Semantic-GT T/R 点云 | 否 / 旧 relation 路由 | 是 |
| **rlbench_o2_semantic_gt_relation_anchor.yaml** | 严格 Semantic-GT T/R 点云 | 否 / anchor 只增强 translation | 是 |
| **rlbench_o2_predicted_objects.yaml** | 外部 predictor 的 T/R 点云 | 否 / relation + anchor | 否，但需要外部 predictor |
| **rlbench_o2_internal_slots.yaml** | 网络内部 2 个 slots | 否 / 旧路由 | 否 |
| **rlbench_o2_semantic_gt_joint.yaml** | 严格 Semantic-GT T/R 点云 | 是 / shared action features | 是 |
| **rlbench_o2_internal_slots_joint.yaml** | 网络内部 6 个 slots | 是 / shared action features | 否 |

## rlbench_o2_gt_instance.yaml

用途：启发式 Oracle buffer 的旧上界/消融。它不执行严格 semantic contract，不能和正式
Semantic-GT 结果混为一组。

~~~mermaid
flowchart LR
    R[启发式 replay T/R 点云] --> P[多视角 T/R prior]
    F[冻结的 BridgeVLA 特征] --> A[relation gated adapter]
    P --> A
    A --> H[translation 与 R/G/C heads]
    E[expert action] --> L[action losses]
    H --> L
~~~

- **oracle_prior_mode: o2_gt_instance**：T/R 点云直接进入策略。
- **oracle_adapter_translation_only: False**：relation residual 同时参与 translation 与
  rotation/gripper/collision。
- **oracle_log_base_loss: True**：额外记录未适配动作 loss，用于计算 action gain。

## rlbench_o2_semantic_gt.yaml

用途：正式严格 Semantic-GT adapter-only 上界。与上一配置的策略结构相近，但数据必须通过
manifest、role YAML 和全量 validation contract。

~~~mermaid
flowchart LR
    Y[semantic roles YAML] --> M[demo-events manifest]
    D[stored demos] --> M
    M --> R[重写并验证 T/R replay]
    R --> P[多视角 GT T/R prior]
    F[BridgeVLA 特征] --> A[relation gated adapter]
    P --> A
    A --> H[translation 与 R/G/C heads]
    H --> L[action losses]
~~~

- GT T/R 是 policy condition，不只是标签，因此闭环评估仍需相同的 Oracle provider。
- **oracle_semantic_audit** 与 contract 用于拒绝错误 phase、role YAML 或不完整 validation。
- 不含显式 relation anchor；主要测量 relation prior + adapter 的上界。

## rlbench_o2_semantic_gt_relation_anchor.yaml

用途：在严格 GT relation adapter 上增加隐式空间 anchor，诊断关系信息能否改善 translation。

~~~mermaid
flowchart LR
    R[严格 GT T/R 点云] --> P[GT T/R priors 与 geometry]
    F[BridgeVLA 特征] --> A[relation adapter]
    P --> A
    P --> Q[anchor query]
    S[当前 gripper state] --> Q
    A --> Q
    Q --> G[relation anchor gate]
    G --> T[translation feature]
    A --> C[R/G/C feature]
    T --> H[action heads]
    C --> H
    H --> L[action losses]
~~~

- anchor 是低分辨率 feature gate，不是分割 mask，也没有单独的 anchor GT。
- anchor residual 只增强 translation；基础 relation residual 仍可进入其他动作分支。
- **oracle_relation_anchor_rank: 16** 打开该路径。

## rlbench_o2_predicted_objects.yaml

用途：把感知误差与控制误差分开测量。策略不读取 oracle 点云，由外部 predictor 提供
Target/Reference 点集。

~~~mermaid
flowchart LR
    X[外部 detector 或 segmentor] --> C[角色选择与 confidence]
    C --> V[T/R valid 与点云]
    V --> P[预测 T/R priors]
    F[BridgeVLA 特征] --> A[relation adapter + anchor]
    P --> A
    A --> H[translation 与 R/G/C heads]
    H --> L[action losses]
~~~

- **object_prior_mode: o2_predicted_relation** 与 **use_predicted_objects: True** 隔离 Oracle 输入。
- **oracle_prior_strict: True**：缺少 predictor 输出时失败，不静默退化成无 object 条件。
- 推理不需要 GT，但必须部署产生相同字段的外部 predictor。

## rlbench_o2_internal_slots.yaml

用途：先冻结原 BridgeVLA，验证内部 slots 能否学习 Target/Reference，并通过旧 action 路由产生
收益。当前配置使用 2 个无序 slots，不为其他物体预留 slot。

~~~mermaid
flowchart LR
    F[VLM 多视角特征] --> S[2 个无序 slots]
    Z[当前 gripper state] --> S
    S --> R[预测 T/R maps tokens geometry 与 NULL]
    R --> A[relation adapter + anchor]
    F --> A
    A --> H[translation 与 R/G/C heads]
    G[GT T/R heatmaps 与 presence] -.训练期 teacher.-> M[一对一 Hungarian matching]
    S -.slot logits.-> M
    M -.mask role objectness NULL loss.-> S
    H --> L[action losses]
~~~

- GT object 字段只生成 teacher heatmap/presence，不进入 policy adapter；推理时可完全移除。
- Target/Reference 与 slots 通过一对一 matching 监督，mask loss 包含 BCE、Dice 与 role loss。
- 当前权重为 NULL **0.25**、旧 cosine diversity **0.0**；日志中的 diversity 值不参与反向传播。
- 默认未启用 instruction context 或 shared action features；anchor residual 主要增强 translation。

## rlbench_o2_semantic_gt_joint.yaml

用途：在可靠 GT 输入上先测试 instruction 与完整动作共享是否带来闭环收益，是预测 slots 之前的
同预算准入实验。

~~~mermaid
flowchart LR
    R[严格 GT T/R 点云] --> P[GT priors 与 geometry]
    V[同次 VLM hidden] --> C[instruction pooling]
    F[视觉语言特征] --> A[relation adapter + anchor]
    P --> A
    C --> A
    A --> S[shared adapted action feature]
    S --> T[translation]
    S --> G[rotation grip collision]
    T --> L[action losses]
    G --> L
~~~

- **shared_action_features: True**：最终适配特征同时供 translation、局部动作和 global pooling。
- **use_context: True**：同次 VLM hidden 的 instruction context 进入 anchor query。
- 配置冻结 vision tower、multimodal projector 与 Gemma 前缀层；其余可训练范围取决于启动模式。
- 仍是 Oracle 上界，因为 GT T/R 是策略输入，不能作为无 GT 部署结果。

## rlbench_o2_internal_slots_joint.yaml

用途：GT joint 通过闭环准入后，联合训练内部感知和完整动作策略。当前配置保留 6 slots，
可表达多余候选，但 Semantic-GT 仍只直接监督当前 Target/Reference。

~~~mermaid
flowchart LR
    V[VLM feature + instruction] --> S[6 个无序 slots]
    Z[当前 gripper state] --> S
    S --> R[soft T/R maps tokens geometry 与 NULL]
    R --> A[relation adapter + anchor]
    V --> A
    A --> F[shared final action feature]
    F --> T[translation]
    F --> G[rotation grip collision]
    D[GT T/R maps 与 presence] -.训练期 teacher.-> M[Hungarian matching]
    S -.slot logits.-> M
    M -.辅助 loss.-> S
    T --> L[action losses]
    G --> L
~~~

- GT maps/presence 只进入辅助 loss；adapter 始终接收预测 slots。
- 6 slots 中未匹配 slots 没有完整 distractor mask GT，不能把它们全部强制成背景。
- 配置打开 instruction、shared action features、NULL loss，并保留小权重 diversity。
- 推理不需要 GT 或外部 predictor，但真实部署仍需验证视觉域偏移、遮挡与时序稳定性。

## rlbench_o2_semantic_roles.yaml

该文件不是训练配置，而是严格数据准备契约：定义各任务/variation 中 Target、Reference、
顺序和 NULL 语义。它的摘要会写入 replay 与 checkpoint contract。

~~~mermaid
flowchart LR
    Y[semantic roles YAML] --> A[任务 variation 与 phase 角色解析]
    D[stored demonstrations] --> M[demo-events manifest]
    A --> M
    M --> H[stored handle 对齐]
    H --> R[重写 T/R 点云 mask presence]
    R --> V[全量 validation 与 SHA-256]
    V --> G[semantic GT configs]
    V -.teacher labels.-> S[internal slot configs]
~~~

- 修改 role YAML 后必须重新生成/验证对应 replay，旧 checkpoint contract 也不再匹配。
- **Reference 不存在**、**Reference 存在但几何无效**是不同状态，不能都当成 NULL。

## 推荐实验顺序

~~~mermaid
flowchart LR
    H[heuristic GT instance smoke] --> G[严格 Semantic-GT adapter]
    G --> A[GT relation anchor]
    A --> J[GT shared + context joint]
    J --> P{闭环配对 CI 为正}
    P -- 否 --> D[诊断 slots anchor 与 action gain]
    P -- 是 --> I[internal slots joint]
    I --> E[无 GT closed-loop evaluation]
~~~

如果目标只是定位某个 YAML 的输入和输出，先看本文的一览表；如果需要命令、冻结范围和配对
验收标准，继续阅读 [O2 实验](../experiments/o2-training.md) 与
[联合训练](../experiments/object-conditioned-joint.md)。实际运行行为以当前 YAML 与训练脚本为准。
