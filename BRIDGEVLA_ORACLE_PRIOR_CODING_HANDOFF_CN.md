# BridgeVLA Oracle Prior Coding 交接文档（精简版）

> 更新时间：2026-08-28
> 目标分支：BridgeVLA 官方仓库 `bridgevla` 分支  
> 目标：用 RLBench 真值 entity/site prior 测量 BridgeVLA 的性能上限

## 1. 实验目标与首版范围

回答一个问题：如果 BridgeVLA 在当前时刻知道正确的任务相关实体或 interaction site，它的 translation prediction 与 closed-loop success 最多能改善多少？

首版只做：

1. 从 RLBench segmentation mask 获取 simulator handle truth；
2. 将 handle 映射为 `target/reference/site`；
3. 为点云附加 role，并投影到 BridgeVLA 相同的三个正交视图；
4. 保留原 translation heatmap，用零初始化的轻量 fusion head 学习 GT prior；
5. 先跑离线 keyframe replay，再跑 paired online evaluation。

首版不运行 Qwen/SAM，不解冻 rotation、gripper、collision heads，不实现完整 world repair；
但联合动作实验允许这些冻结头的 loss 穿过动作头更新新增 Adapter。

### 当前轻量适配实现状态（2026-08-31）

本轮先完成 O2 训练闭环，原则是尽量不改变原 BridgeVLA：

- 新增模式默认关闭；`rvt.oracle_prior_mode=none` 时不创建 adapter/fusion 参数，也不读取
  Oracle replay 字段，旧 replay、旧 checkpoint 和原前向路径保持不变；
- O2 主配置同时使用当前状态下唯一的 Target 与 Reference 完整点云，而不是物体中心；
  两组点云按固定 `[T,R]` 顺序，经过与场景相同的 SE(3) 增强、立方体归一化和
  MVT 三视图投影，以保留二者的空间关系；
- 第一、二阶段均先用 rank-16 feature adapter 将双通道实例 prior 注入 2048 维视觉特征；
  可开关的 relation encoder 共享编码 T/R 点集，并用 pooled feature、中心、尺度和相对位移
  生成 gated FiLM。Stage 2 使用与局部 heatmap 相同的平移/缩放坐标系；随后通过 hidden-64
  多尺度 residual fusion 修正 translation logits。输出层零初始化，初始结果仍等于 baseline。
  主配置设置 `oracle_adapter_translation_only=False` 与 `peract.add_rgc_loss=True`，
  adapted feature 同时进入 translation、rotation、gripper、collision 分支；六项 loss
  联合更新新增 Adapter，translation Fusion 只接收 translation 梯度；
- 推荐从已训练 baseline checkpoint 初始化，冻结原 BridgeVLA（包括 Gemma），只训练
  relation-gated feature adapter 与 fusion，精确为 223,878 个参数（约 22.4 万）；
- 正式 semantic-GT O2 参数集中在
  `finetune/RLBench/configs/rlbench_o2_semantic_gt.yaml`；旧
  `rlbench_o2_gt_instance.yaml` 保留给启发式 buffer 消融；checkpoint 和冻结模式仍作为
  运行时命令行参数；
- 同一个 batch 内增加无梯度 baseline 动作支路，不重复 PaliGemma 前向；日志同时输出
  `total_loss_base`/`total_loss`/`total_loss_gain`、各 R/G/C base/O2 loss、全 batch 的
  `trans_loss_base`/`trans_loss_raw`/`trans_loss`，以及完整 T/R pair 的对应 `*_valid`
  指标。base rotation 对比会保存并恢复 BatchNorm buffer；Adapter-only 下冻结动作头的
  BatchNorm 使用 checkpoint running statistics，避免原模型 buffer 漂移。主配置保持
  `oracle_valid_only_loss=False`，使用固定 batch 分母，避免 8-GPU 梯度累积时低
  coverage micro-batch 被放大；valid-only 开关保留用于消融，并修正了 DDP rank 间计数；
- 新增版本化 `rlbench_o2_semantic_roles.yaml`，覆盖 18 个任务的唯一 T/R、variation、
  固定顺序和成功条件；在线 `RLBenchGTOracleProvider` 从四视角 simulator handle mask
  生成 object/site prior，并输出 role audit、manifest 与分类统计；
- `eval.sh` 支持 `ORACLE_PROVIDER=none|rlbench_gt`、`ORACLE_STRICT=1` 和 expert
  keypoint replay。离线 `rewrite_replay_with_semantic_roles.py` 只重写 Oracle/audit 字段，
  不修改 baseline transition；
- 本地完成补丁格式、Python/Bash 语法与静态接线检查；新增 PyTorch/RLBench 回归测试需在
  服务器环境执行。

最小版本暂不处理以下复杂情况：

- 不加入 prior dropout、错误实例扰动、外部置信度或不确定性建模；当前 gate 只由
  GT T/R 三维关系特征学习；
- 不训练 Qwen/SAM/predicted-instance provider，也不把 Oracle 结果当作部署结果；
- 不实现多候选启发式消歧；当前 phase/角色严格来自 task/variation/success condition。
  success sensor/dummy site 已支持，但不加入第三个 Tool 通道；
- 不修改 Gemma、vision tower、rotation、gripper、collision 等原网络参数；这些冻结
  动作头参与 forward/backward，将联合动作 loss 的梯度传给新增 Adapter；
- 不在本地 Windows 环境完成 PyTorch 前向/反向和 RLBench online smoke test。本地缺少
  PyTorch；对应测试已编写，需在服务器 `bridgevla` 环境执行；
- online evaluation 已接入 `RLBenchGTOracleProvider`，每个时刻直接提供
  `oracle_target_object_points` 与 `oracle_reference_object_points`。仍需在目标服务器的
  vendored RLBench/TTM 上对全部 variation 跑 strict reset 和 closed-loop smoke test；任何
  selector/层级不一致应修 YAML/provider，不能回退邻近关系猜测。

上述项目是后续增强，不阻塞当前 adapter-only O2 训练。若轻量实验没有稳定收益，不应
提前增加这些复杂机制。

当前 eval 可视化已经支持逐视角保存 `o2_target_prior`、`o2_reference_prior`、
`o2_prior`、`o2_raw` 和 `o2_fused` 及其
RGB overlay，目录为
`<visualize_root_dir>/<task>/episode_<N>/<language_goal>/step<N>/{mvt1,mvt2}/`。
必须显式传入 `--visualize`；若该 step 没有 Oracle 输入，会保存
`o2_unavailable.txt`，而不是伪造 prior 图。

## 2. Oracle 等级与泄漏边界

| ID | 信息 | 用途 |
| --- | --- | --- |
| B0 | 无 prior | 原始 BridgeVLA baseline |
| O1 | 所有当前合法 target/reference 实体 | 主实验，最干净 |
| O2 | 当前应操作的精确实例 | 单列，包含实例选择 oracle |
| O3-V | 当前 RGB-D 中可见的真实 interaction site | 主实验 |
| O3-F | simulator 完整 site，包括遮挡部分 | 理论上界，单列 |
| O4 | 专家下一 keyframe 位置高斯图 | 仅诊断，不得作为 relevant-object 主结果 |

主实验允许读取当前及历史 observation、instruction、task/variation、当前 gripper 和任务谓词。禁止读取未来帧、专家下一动作、下一 waypoint、episode 最终结果。

例如 `stack_blocks`：O1 应标出所有尚未放置且合法的方块；若根据专家轨迹只标下一步抓取的 `block2`，应归为 O2/O4，不能写成 O1。

## 3. 数据流与代码归属

```text
RLBench RGB / point cloud / handle mask
                 ↓
OracleRoleProvider：handle → target/reference/site
                 ↓
LabeledPointCloud：xyz + rgb + role bits
                 ↓
BridgeVLA 原有 preprocessing
                 ↓
同一 renderer、同一三视图像素坐标
        ┌────────┴────────┐
        ↓                 ↓
BridgeVLA raw logits  Oracle instance heatmap
        └────────┬────────┘
                 ↓
zero-init learned residual fusion
                 ↓
BridgeVLA 原 action decoder
```

Oracle 首版应直接实现于 BridgeVLA 仓库，因为它需要 PyRep scene handles、BridgeVLA preprocessing 和 renderer。本仓库只提供设计参考：[`task_schema.py`](task_schema.py)、[`dynamic_role_reasoning.py`](dynamic_role_reasoning.py) 和 [`CONSERVE3D_IDEA_OVERVIEW_CN.md`](CONSERVE3D_IDEA_OVERVIEW_CN.md)。

## 4. 开工前定位 BridgeVLA 插入点

```bash
git checkout bridgevla
git rev-parse HEAD
rg -n heatmap|trans.*logit|translation finetune/RLBench
rg -n render|orthographic|point.*cloud finetune/RLBench
rg -n decode|act\(|agent|eval.*episode finetune/RLBench
rg -n replay|dataset|augmentation|transform finetune/RLBench
```

必须找到点云 preprocessing、三视图 renderer、decode 前 translation logits、offline/online observation 入口。prior 必须在 translation decode 前融合，不要在 3D action 生成后做最近点吸附。

## 5. 最小模块与接口

```text
oracle_prior/
  role_registry.py       # task/variation → handle selector
  role_provider.py       # 当前状态 → active handles
  labeled_points.py      # mask + point cloud → point roles
  prior_renderer.py      # labeled points → three-view prior
  fusion.py              # translation logit fusion
  leakage_guard.py       # 禁止未来信息
  metrics.py             # paired metrics
tools/
  eval_oracle_prior_offline.py
  eval_oracle_prior_online.py
```

role 使用 bit flags，因为 site 同时属于 entity：

```python
ROLE_TARGET    = 1 << 0
ROLE_REFERENCE = 1 << 1
ROLE_SITE      = 1 << 2
ROLE_ATTACHED  = 1 << 3
```

核心契约：

```python
@dataclass
class LabeledPointCloud:
    xyz: Tensor          # [N, 3]
    rgb: Tensor          # [N, 3]
    handle_id: Tensor    # [N]
    role_bits: Tensor    # [N]
    site_score: Tensor   # [N], [0, 1]

@dataclass
class OraclePriorViews:
    target: Tensor       # [B, 3, H, W]
    reference: Tensor    # [B, 3, H, W]
    site: Tensor         # [B, 3, H, W]
    active: Tensor       # [B, 3, H, W]
```

所有 crop、filter、sampling 和 augmentation 必须对 `xyz/rgb/handle/role` 使用相同索引和几何变换。

## 6. RLBench truth 与 role registry

```python
CameraConfig(
    rgb=True,
    depth=True,
    point_cloud=True,
    mask=True,
    masks_as_one_channel=True,
)
```

RGB 编码 mask 使用 `handle = R + G*256 + B*256*256`；若已是单通道 handle mask，不要再次解码。官方参考：

- <https://github.com/stepjam/RLBench/blob/master/rlbench/observation_config.py>
- <https://github.com/stepjam/RLBench/blob/master/rlbench/backend/observation.py>

registry 示例：

```yaml
tasks:
  stack_blocks:
    task_family: pick_place_repeated
    target_selectors: [{name_regex: block.*}]
    reference_selectors: [{name_regex: stack.*|base.*}]
    site_strategy: {type: top_surface, surface_band_m: 0.015}
```

reset 后解析 name→handle；0 匹配或意外多匹配必须报错；variation override 显式配置；registry hash 写入结果。

## 7. Site 与三视图 prior

site 生成优先级：独立 shape handle → entity 几何子区域 → 完整 simulator mesh（O3-F）→ 专家下一 action point（O4）。

top surface 示例：

```python
z_threshold = torch.quantile(entity_xyz[:, 2], 0.95) - surface_band_m
is_site = is_entity & (xyz[:, 2] >= z_threshold)
```

prior renderer 必须使用 BridgeVLA 相同的 workspace bounds、normalization、augmentation、view matrices、分辨率与像素约定。禁止直接 resize RLBench 相机 mask。

hard mask `M` 转 soft prior：

\[
P_v(x)=\exp\left(-\frac{d(x,M_v)^2}{2\sigma^2}\right)
\]

要求 `P∈[0,1]`、mask 内最大值为 1、空 prior 有 fallback；测试 `sigma_px ∈ {4,8,16}`。

阶段选择：未持有物体时使用 target/site；持有后使用 reference/site；press/open/turn 优先 site。分别评测 inferred phase 与 oracle phase。

## 8. Translation heatmap 融合

设冻结 PaliGemma 的视觉特征为 `X`，Target 与 Reference 的 GT instance heatmap
分别为 `P_T`、`P_R`。不再使用固定
`alpha/floor` 约束，而训练低秩 feature adapter 与多尺度 residual head：

    X_adapt = X + A_phi([X, downsample(P_T), downsample(P_R)])
    L_raw = up0(X_adapt)
    L_fused = L_raw + F_theta([L_raw, P_T, P_R,
                               L_raw * P_T, L_raw * P_R, P_T * P_R])

`A_phi` 和 `F_theta` 的输出层必须零初始化，因此启用 O2 后的初始输出严格等于
baseline；Oracle 无效样本的两段 residual 必须强制为零。训练输出同时保留
`trans_raw`、`oracle_target_prior`、`oracle_reference_prior`、合并检查用的
`oracle_instance_prior` 和实际参与 loss/decode 的 `trans`。

推荐主实验冻结原 BridgeVLA，只训练 adapter 与 fusion；fusion-only 作为最小对照，
完整动作 head 联合训练仅作为补充。三种设置必须分开报告，不能把重新微调整网的收益
归因于 prior。

## 9. 评测 protocol

### 离线 replay

同一 keyframe 分别解码原 logits 和 fused logits，与专家下一 translation 比较。报告：

- mean/median 3D error；
- PCK@2cm、PCK@5cm；
- correct-entity/site；
- rescue：baseline 错、oracle 对；
- harm：baseline 对、oracle 错；
- 按 task、phase、oracle level 分组结果。

### 在线 paired evaluation

同一个 task、variation、seed 运行：

```text
B0   baseline
O1   entity-set + inferred phase
O2   exact instance + inferred phase
O3V  visible site + inferred phase
O3P  visible site + oracle phase
O3F  full-state site + oracle phase
O4   action-point，仅诊断
```

pilot 每任务 25 个 paired episodes；正式建议至少 `25 episodes × 3 seeds`。保持 checkpoint、instruction、最大步数、renderer、decoder 和其他 heads 一致，报告 success、steps、wrong-entity/site、rescue/harm、failure taxonomy 和 paired bootstrap 95% CI。

## 10. 测试与验收

必须测试：

- registry 0/多匹配报错，O1 不访问专家下一实例；
- point attributes 经过 crop/sample 后仍对齐；
- 合成 cube/plane 三视图投影正确；
- prior 范围正确，空 prior 无 NaN；
- fusion 零初始化时 fused logits 与 raw logits 完全一致；
- fusion 最后一层在第一个训练 step 能收到非零梯度；
- `evidence_time <= action_time`，O4 必须显式授权；
- 一个 replay batch 可生成 RGB/prior views；
- 三视图 overlay 对齐；
- O2 关闭时 action 与旧 baseline 差异 `<=1e-6`；
- online 两 episode smoke test 可完成；
- prior 关闭后 baseline 独立运行。

Go/No-Go 建议：hard-task 离线 error 相对下降 ≥10% 或 PCK@5cm 提升 ≥5pp；在线 hard-task 提升 ≥10pp 或总体 ≥5pp；简单任务 harm ≤2pp。

## 11. 实施顺序与后续

1. 固定 BridgeVLA commit、checkpoint、RLBench 环境和 baseline seeds；
2. 实现 handle mask、registry 和 O1；
3. 实现 point roles 与三视图 renderer，人工验证对齐；
4. 插入 zero-init learned fusion，验证初始 parity；
5. 先只训练 fusion head，再跑 offline paired evaluation；
6. 跑 online B0/O1/O2；
7. 扩展 O3-V/O3-F；
8. 最后加入隔离的 O4。

Oracle 达到 Go 标准后，保持相同 prior 接口，依次替换 provider：

```text
SimulatorOracleProvider
  → CurrentProjectProvider（SAM/Qwen/3D fusion）
  → LearnedStudentProvider
  → Persistent World/Repair Provider
```

结果解释：O1 提升说明相关实体选择有价值；只有 O3 提升说明应重点做 interaction-site；O3-F 显著高于 O3-V 说明需要持久世界记忆；translation 改善但 success 不变，则瓶颈已转移到 rotation/gripper/collision。

## 12. 从 Oracle 训练迁移到可部署 prior

当前 O2 已采用 learned fusion，但训练和评估输入仍是 GT instance，因此仍属于
privileged Oracle 上界。验证信息有价值后，保持 fusion 接口不变，把 provider
依次替换为 predicted instance；再混合 truth、noisy、predicted prior，并使用
`30%~50%` prior dropout 提升容错。可进一步扩展
`target/reference/site` 三角色热图与末端关系 token。具体设计见
[`BRIDGEVLA_ROLE_RELATION_PRIOR_DESIGN_CN.md`](BRIDGEVLA_ROLE_RELATION_PRIOR_DESIGN_CN.md)。
