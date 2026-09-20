# Object-conditioned Policy：实现与边界

[精简设计](role-relation-prior.md) · [联合训练](../experiments/object-conditioned-joint.md) · [代码索引](../reference/code-map.md)

> 更新：2026-09-20。单帧条件化已提供 opt-in 代码；数值、训练和闭环收益仍须在目标环境验收。

## 1. 最小实现

```text
共享 scene feature + 同次 VLM 的 instruction context
  → 无序 slots → soft T/R tokens/maps 与 Reference NULL posterior
  → 原 relation hidden + anchor query z
  → 最终 action feature → translation 与 R/G/C
```

不增加独立 PhaseEncoder、phase class loss、relation graph 或第二个 VLM。
`z` 是现有 anchor query 的条件表示；隐式阶段行为由动作监督学习，不保证阶段可辨识。

### Feature 路由

`OracleRelationAnchorFeatureAdapter.forward_with_anchor()` 保留旧输出接口：
`translation_features, shared_features, anchor`。

- 默认模式：translation 使用 anchor-enhanced feature，R/G/C local 使用旧 shared feature，global 使用原始 feature。
- 新共享模式：translation、R/G/C local 和重新池化的 global 全部使用 anchor-enhanced feature。
- base diagnostic 单独使用原始 global/local feature，不混入最终 feature。
- 推理从最终 translation 解码 waypoint，随后在该位置采样 R/G/C；没有 post-hoc fusion。

仅切换 shared flag 的 GT 对照不改变角色机制。internal-slot 的新共享模式才启用 soft role conditioning。

### Semantic roles 与 action anchor

Target/Reference 描述任务关系，translation heatmap 解码的是当前动作锚点；锚点可位于
Target、Reference、接触点或自由空间，不能反向定义 T/R GT。测试诊断同时报告 adapter 前的
`base` 与实际执行的 `final` waypoint。配置中 `phase=-1` 的干扰候选只参与锚点归因，不能
成为有效语义 Target。纯诊断不覆盖 policy 输入；可选的两次前向模式允许 simulator residual
跟随 `trans_base` 选中的 Target，并用 gripper 周期锁定对象，但该对象属于预测条件而非 GT。具体字段和命令见
[Semantic-GT 指南](../guides/semantic-gt.md#测试期-heatmap-action-anchor-归因)。

### Instruction 与当前状态

`pool_instruction_context()` 从同一次 VLM hidden state 的非图像有效 token 池化，兼容左右 padding。
特殊 token 不参与。小 projection 位于原 adapter / slot predictor 内。

新网络入口为 `current_state[B,3]`；旧 `oracle_relation_state` / `relation_state` 参数继续兼容，
但不能同时传入两种名称。目前三维状态是 gripper open 与两维 finger state，不是 relation 或 phase GT。
replay 的目标 `gripper_pose` 只能用于动作监督，不能当作当前机器人状态。

### Soft geometry

`soft_role_geometry()` 在当前 stage 的 rendered XYZ 上计算 role-map 加权中心与标准差 spread：

- 保留地图到 geometry 的梯度；hard top-k 点云仍用于兼容和可视化。
- 屏蔽 background、非有限 XYZ 和不可用 role。
- spread 是可见支持的分布，不是完整物体 bbox/size。
- coarse/refine 各自在自身 render 坐标系计算，不跨 stage 混用 geometry。

Reference confidence 低不再连带关闭有效 Target。几何不可用使用 unknown 表示；
语义 NULL 由独立 posterior 软混合。两者不能互相代替。没有 memory 时，该分支不能可靠定位全遮挡物体。

## 2. Teacher 与数据

Replay schema 由 `finetune/RLBench/utils/dataset.py::create_replay()` 声明，
加载入口是 `finetune/RLBench/utils/get_dataset.py::get_dataset()`。

`_derive_role_presence()` 在磁盘读取时从已有 `kind` 派生数值标签，不修改 replay：
`object/site` 表示角色存在，`none` 表示 Reference absence。
缺少审计字段或 Target 为 `none` 的终止占位，不提供 presence 监督。

Internal-slot 模式的 GT maps、valid、points 在 `MVTSingle` 前与 policy 输入隔离。
presence 仅在 `RVTAgent._object_slot_auxiliary_losses()` 中使用。
NULL loss 监督实际推理 posterior；不可用几何不成为 negative existence label。
正向可见支持可训练 slot objectness，但不把 objectness 称为语义存在概率。

当前 semantic replay 只有已选 T/R，适合 role-map 监督，不能证明通用 object discovery、
完整 slot 身份、跨帧 tracking 或独立 visibility。GT site 体积也不等于传感器可见表面；
预测角色地图与可见 XYZ 的 site grounding 精度必须单独检查。

## 3. 训练与验收

GT joint 冻结 vision tower、projector 和 Gemma 前 6 层；internal-slot joint 冻结 vision tower、
Gemma 前 18 层但训练 projector。两者都训练其余 Gemma、action decoder 和各自 object 模块。
复用现有分组 LR：非 Gemma `4e-5`，Gemma `1e-5`。

从统一 baseline checkpoint 使用 `--init_checkpoint`。原参数名保留，新增输出 residual 零初始化；
新增模块的参数可缺失。训练 checkpoint 记录两个 conditioning 开关；旧路由不能 optimizer-resume
到新路由。同架构继续训练才使用 resume。

必须先运行 GT 对照与三-seed paired closed-loop gate。四组使用同数据、初始化、解冻范围、
训练步数及评估 episodes；base diagnostic loss 不是独立训练的 baseline。
CI 工具不丢弃失败、不接受不匹配 episodes，不替代训练预算核对或真实机器人安全验证。

`z` 是否利用 objects/instruction，可用 object swap、Reference→NULL、指令替换等干预测试检查。
这些测试只是条件依赖证据，不足以宣称 causal reasoning。
正常演示没有失败状态覆盖时，不能承诺坍塌恢复。完整命令和测试见[实验页](../experiments/object-conditioned-joint.md)。

## 4. 后续扩展

### 两角色短时 memory

仅在短遮挡/observation aliasing 确认后考虑。保留 T/R token、位置分布、confidence 与 age；
不可见时不更新成 NULL，不补全不可见表面。同一 control query 的 coarse/refine 只更新一次，
episode reset 清空。需要短窗口数据与 identity consistency 验收，当前代码没有跨 query memory。

### Object-layered Orthographic Refine

只处理虚拟投影 z-buffer 竞争与小物体分辨率，不解决原始点云缺失。

```text
global coarse top/front/right
 → soft T/R point membership
 → separate T/R refine layers + global context
 → feature-level fusion
 → shared full-action decoder
```

保留 global 障碍物上下文，不为每个 object 重跑 VLM，不在输出 heatmap 后融合。
优先比较仅加入 mask/depth channels；只有这一步不足才独立 rasterize。
object-centered scale 需上下界与可信 geometry；可见 spread 不能直接当完整 extent。
NULL 不渲染 Reference layer；unknown/低置信 geometry 不强行生成 crop。

| 观测瓶颈 | 扩展是否有用 |
| --- | --- |
| 点存在于输入点云但在虚拟投影被覆盖 | 分层投影可能改善 |
| 全局投影中的小物体分辨率不足 | bounded object-local scale 可能改善 |
| 表面没有被任何真实相机观测 | 分层投影无效，需要真实新观测或 temporal evidence |

显式进度、pair search 和 learned risk 也只在对应失败模式确认后增加。
