# OHT v423 数据迁移实施方案：Baseline 与 Predicted Object Assistance

日期：2026-10-09。

状态：已实施首版 OHT 数据审计、RGB-D/动作缓存、baseline、内部 role queries 与外部预测缓存/在线 wrapper、训练入口、绝对 TCP 推理服务及评估接口，并完成合成数据 CPU 集成验证。运行方式与实际文件对应见 [OHT 实施指南](../../finetune/OHT/README.md) 和第 17 节。深度/相机轴已按用户补充调查接入；未读取服务器实际数据，未运行完整 CUDA 训练或 IsaacLab 闭环，服务器全量几何验证、角色标注和仿真适配仍待完成。本文前 16 节保留设计背景，其中未列入第 17 节的规划内容不代表已经实现。现成 H-VLA/IsaacLab 的路径、启动脚本和结果来自用户提供的信息，未远程验证。

## 1. 结论与实施边界

BridgeVLA 的点云渲染、VLM、绝对末端位姿动作解码器，以及 O2 relation/anchor 和内部角色预测模块可以复用。迁移不是把数据路径替换成 LeRobot 路径：需要完成米制深度解码、相机/TCP 标定、观测与动作重建、角色数据契约、机器人执行和独立评估。

推荐顺序：

1. 数据审计和坐标/深度验证。
2. 优先核查现成 IsaacLab client 的观测、控制和任务协议，再做无模型专家目标位姿执行，证明控制与数据契约成立。
3. OHT baseline 单任务小样本，再四任务联合。
4. OHT T/R teacher 和可选 Oracle 诊断，判断辅助几何是否有用。
5. 内部 role queries 作为首条 predicted assistance 主线。
6. 如需独立感知管线，再实现外部 predicted objects；两条路线分开报告。
7. 固定预算、三训练 seeds、配对闭环与遮挡/扰动评估。

Baseline 和 assistance 共用观测、相机、动作定义、控制器、训练划分和语言协议。Oracle 结果单独报告，不能作为无 GT 预测方法的成绩。

## 2. 当前代码依据与复用范围

| 当前入口 | 已实现内容 | OHT 迁移处理 |
| --- | --- | --- |
| [RLBench dataset](../../finetune/RLBench/utils/dataset.py) | 当前观测到下一关键点的绝对 EE 位姿、旋转分桶和夹爪标签；replay schema | 复用张量约定，提取通用 schema；不伪造 RLBench Demo |
| [RLBench constants](../../finetune/RLBench/utils/peract_utils_rlbench.py) | 四个固定相机名、128 图像尺寸、RLBench bounds、4 维 low_dim | 新增 OHT 配置，通过构造参数传入 |
| [RLBench train](../../finetune/RLBench/train.py) | 训练/DDP/优化器、checkpoint、启动检查 | 复用训练机制，替换 loader、任务和数据契约 |
| [RVTAgent](../../finetune/bridgevla/models/bridgevla_agent.py) | update/act、外部预测字段选择、内部角色辅助监督 | 解耦 RLBench/GemBench 导入，保留 O2 前向 |
| [MVT](../../finetune/bridgevla/mvt/mvt.py) | 点云渲染、coarse/refine、角色变换 | 保留三虚拟视角和已有裁剪/归一化 |
| [oracle_prior](../../finetune/bridgevla/models/oracle_prior.py) | relation/anchor、slots、role_queries | 内外部预测都可复用；名称含 oracle 不代表所有模式都读取 GT |
| [object-conditioning guide](../guides/object-conditioning.md) | 数据字段、配置和训练/推理隔离 | 保留语义原则，新增 OHT 数据来源和审核 |
| [semantic contract](../../finetune/RLBench/utils/semantic_contract.py) | RLBench schema、stored handles、sim_replay/demo_events | 不能直接改 YAML 冒充 OHT；新增 dataset-specific contract |

当前 agent 在模块顶层导入 RLBench 工具，RLBench dataset 还导入 Observation、get_stored_demo 等；即使读预生成 replay，OHT 环境仍可能被迫安装 RLBench。第一阶段应把观测预处理和 schema 提取到不依赖 simulator 的公共模块，旧 RLBench API 保留兼容包装。

当前仓库没有 OHT loader、OHT predictor wrapper 或通用 detector/segmentor。外部 predicted YAML 定义的是接收接口，不是可自动运行的物体预测系统。

## 3. 新版说明带来的条件和待核查项

已说明五路 480×640 RGB-D、逐帧相机外参、世界 EE 位姿、关节顺序与分段指令。2026-10-09 按用户提供的已运行转换/训练代码复核后，修正早期说明中的深度与 EE 顺序假设：raw camera/EE 默认 wxyz，分别显式配置（若参考运行覆盖为 xyzw，这里也须匹配）；输出始终 world/TCP/xyzw。`ee_pos_world` 已含控制器 TCP 偏移。

### 3.1 深度视频契约（2026-10-09 补充调查）

深度为 **无损 HEVC gray12le**，数值编码不能从像素格式推断。用户本次提供的 `info.json` 仅列 RGB，无深度量化参数；参考转换器的 log fallback 不是 writer 契约。仓库默认更正为数据方此前调查给出的 `raw × 0.001 m` 射线距离，0/4095 无效：raw=897 应为 0.897 m，此前 log 默认值误解为约 1.215 m。仍支持独立核实的 quantized writer；若启用 metadata，必须具备每相机完整量化字段，缺失报错。配置、区别和迁移见 [OHT 运行说明](../../finetune/OHT/README.md#3-仿真数据契约与构建共用缓存)。

不需要 raw sidecar 或重新采集。读取器原生读取灰度平面并检查像素格式；仓库配置按原始帧序号配对，独立时间戳流可显式选择 PTS。禁止将数值深度转换成 RGB 再解码。相机世界姿态为 USD/OpenGL，使用 `T_world_optical = T_world_usd @ diag(1,-1,-1,1)`。

内参优先读取 `meta/camera_intrinsics.json`，文件不存在时读 `meta/info.json.camera_intrinsics`；夹爪端点来自 dataset-global stats/info，解析结果与源文件 hash 绑定 contract。`depth.kind=ray` 来自数据方说明；参考 `dataset.py` 将反投影委托给未提供的 `pointcloud_transforms.py`，本轮不把 ray/Z 的一致性当作已验证。错误解码可造成跨相机变形，不证明所有旋转错位都源于此；须检查有效配置及真实单帧，多相机外参/同步仍需复核。旧 XYZ 不能靠改 contract 修复，原始数据/audit 可复用。

### 3.2 四元数与相机坐标约定

原始 `observation.*_extrinsic` 默认 `[x,y,z,qw,qx,qy,qz]`，是 USD/OpenGL camera→world。`camera_pose_matrix()` 按 `camera_quaternion_order` 转内部 xyzw；若显式配置 `camera_extrinsic_direction: world_to_camera`，先对完整 4×4 求逆，再右乘 optical→sensor，默认得到 `T_world_optical = T_world_usd @ diag(1,-1,-1,1)`。不在世界系左乘翻轴，不只转置 R。`world_tcp_poses()` 独立按 `ee_quaternion_order` 转换 EE；输出和内部 geometry 为 xyzw。缓存相机外参为 4×4 optical→world。

`link_to_tcp` 保持单位阵，禁止重复加偏移。行向量投影 `(P_world-t) @ R_world_optical` 不再额外转置。schema 升级为 `oht_bridgevla_v2`，旧 XYZ/rotation/gripper 标签须在新目录重建；原始数据/audit 可保留，角色缓存随新 contract 重建，不手改合同或恢复旧 optimizer。

参考代码对照（实现均在 `finetune/OHT/data/`）：

| 提供代码中的函数/链路 | 本轮处理 |
|---|---|
| `_depth_video_spec()` / `_dequantize_depth_mm()` | `video.decode_depth()` 仍支持已核实的量化公式；`source_config.resolve_dataset_config()` 不再静默套 log fallback，当前导出按数据方毫米契约显式解码 |
| `_resolve_calibration_for_view()` / `_resize_intrinsic()` | 读取真实 metadata K；RGB-D 严格步长采样，K 同步缩放，不照搬 TFDS 方形 resize |
| `_canonicalize_extrinsic_pose7()` / optical conversion | 原始 wxyz/camera→world/OpenGL；只翻轴一次，BridgeVLA 保留 world 坐标 |
| `_canonicalize_new_gripper_to_legacy_physical()` / `binarize_gripper_hysteresis_with_diff()` | 使用原始 motor 端点直接归一化 open01，移植因果滞回+差分；不绕经旧 TFDS 的物理开度区间 |
| `proprio_absolute` / `proprio_relative` | 复用“观测重建标签”，未来 GT keypoint 的 pose 与 gripper 均来自实测；raw action 全部仅诊断 |
| `_filter_indices_by_ee()` / 左右 camera aliases | 不删除纯旋转/夹爪变化；五个物理相机独立命名和取外参，不把 left/right 当同一相机 |
| `dataset.py → pointcloud_transforms` | 下游公共反投影模块未提供，保留显式 ray/Z，不能宣称完整端到端等价 |

NVIDIA 相机 API 区分 distance_to_image_plane 和 distance_to_camera，不能对两者直接使用同一 Z-depth 公式。

### 3.3 文档与数据内部一致性

- 右轮典型位置的 y 为 +0.183，而右侧 INSERT 路点附近为 -0.183；可能来自不同 episode/随机化/对象或示例，必须用同一 episode 核对，不能直接据说明取轴中心。
- drop_label/is_settle 是采集标签，不自动等价于 online success；示例 held_obj 在 Drop 段仍写车轮，释放事件以实际时序交叉确认。
- metadata.pose 是 degree Euler 且可能含 keep；不能忽略 keep 掩码，或把目标路点直接当已到达观测。
- joint_vel 的相关性异常可能涉及采样时间、延迟或记录源，不能仅凭相关系数确定原因。关键点首版使用位置轨迹和事件，速度由时间戳差分核查。
- PD 比例和速度统计来自示例 episode，不当作 400 episodes 的全量事实。

## 4. 阶段 A：全量数据审计与划分

拟新增 tools/audit_oht_dataset.py。

1. 以实际 Parquet 文件建立 episode 清单；不以有重复的 episodes.jsonl 行数计数。
2. 唯一键为 task_name + episode_index；四目录内的 task_index 都为 0，联合训练时显式映射到 0–3。
3. 直接读取完整 Parquet schema，核查长度、dtype、finite、frame_index/timestamp 单调性、视频路径和 PTS。
4. 验证成功 metadata、终止/成功列、日志的完成事件是否一致；失败/部分成功单独登记，首版 BC 使用确认成功的示范。
5. 审计指令、夹爪事件、关节限位、EE/FK 一致性、物体名称与段边界；记录排除原因，不静默修补。
6. 固定 train/val/test manifest，建议每任务 80/10/10，共 320/40/40 episodes。若存在重复初始场景、采集重试或相同轨迹，应按这些组一起分配，不跨集合。
7. 若希望声称未知布局泛化，另按初始布局/资产条件分组留出；普通 episode 随机划分不证明这种泛化。
8. 在 train 集确定归一化/采样参数；val 调阈值、预算和控制超参数；test 仅做固定最终评估。

交付 audit.json、episodes.jsonl、split_manifest.json 和异常样本列表。每个产物记录原始路径、schema、版本及必要摘要。

验收：有效 episode 全部可追溯，任务 ID 不冲突，划分无组泄漏，所有缺失/失败都有明确处理。

## 5. 阶段 B：观测、标定和空间契约

拟新增 finetune/OHT/data/{reader,video,geometry,observation}.py 与 camera_calibration.yaml。

内部统一：

- policy 坐标：world，长度 m，时间 s，角度 rad，四元数 xyzw。
- 相机内参 K 按实际图像宽高读取；外参定义为 T_world_optical，禁止只写模糊的 extrinsic。
- 动作参考点定义为明确 TCP；如数据记录的是 flange/tool link，使用固定 T_link_tcp 转换。物体坐标原点不默认是几何中心。

对 Z-depth：

    p_optical = Z * inverse(K) * [u, v, 1]^T
    p_world   = T_world_optical * [p_optical, 1]^T

若记录为 ray distance，按单位射线恢复三维。先屏蔽无效深度，再融合；腕部相机每帧使用对应外参。v423 仓库默认 `video_alignment: frame_index`，遵循已运行参考转换器：原始 RGB/depth 解码帧 i 与 Parquet pose i 配对，严格核对整段帧数，不用 FPS 计算索引。独立时间戳流显式用 `timestamp` 最近 PTS 模式，旧配置缺字段保持该行为。此修正防止引用时间偏移带来的旋转错配，不证明导出 pose 已正确同步；ray/Z 仍需公共反投影模块核对。用 `tools/diagnose_oht_geometry.py` 导出单帧逐相机与来源着色融合图及实际参数，不重建已有缓存、不自动配准；命令见 OHT README。

五路传感器融合后仍渲染现有三虚拟视角；不需要把 VLM 改为五路物理相机直接输入。物理相机数与 MVT num_img 不是同一个参数。

首版保留当前 organized RGB/point_cloud batch 接口，图像尺寸改为配置项；先在原始分辨率验证几何，再选择下采样。深度/点坐标使用一致像素采样并更新 K，不对跨物体边缘的深度做普通 RGB 双线性插值。

已实现可选手工背景过滤 `point_cloud_filter`：仓库 YAML 启用经验 ROI `[-0.3,-0.8,0.4,1.2,0.8,1.9]`，旧配置缺少此块仍关闭。它覆盖已记录的坐标示例，须预览四任务抓取/放置/高位姿态验证，不能视作全量安全边界。世界坐标保留框减去排除框并集，独立于 `scene_bounds` 和动作标签。共享 `data/point_filter.py`，缓存构建、训练输入、在线 predictor/agent 及全局/局部预览一致；排除 XYZ 记 NaN，原始 RGB/depth 与像素布局不改。单相机可空，全场过滤为空则拒绝执行/完成缓存。参数纳入 data profile/契约，更改后在新目录重建 replay/角色缓存；原始数据/audit 可复用。配置与安全边界见 [OHT 运行说明](../../finetune/OHT/README.md#3-仿真数据契约与构建共用缓存)，不预设实际墙壁排除框，也不删除所有平面或按未来 GT goal 筛点。

约 150 万原始点/帧无需全部进 renderer。缓存抽取帧/共享观测，按空间与颜色一致性采样或保留局部细节。不要把 60 万帧的五路 float32 RGB-D/XYZ 全部复制为多个 replay transition；先估算磁盘、点数、加载吞吐和显存。

world bounds 由训练观测和动作工作区确定并留显式余量；先统计原 RLBench bounds 的越界率。不得把越界 GT 标签静默夹到边界。

验收：四任务分别叠加点云、EE、物体与目标区域；固定物体跨相机重合；腕部运动时环境不漂移；报告标定误差与深度量化误差。插入容差由任务实际要求决定，不能用统一的厘米级阈值自动通过。

## 6. 阶段 C：动作重建、关键点和语言协议

### 6.1 动作标签

对当前观测 t，选下一关键点 k(t)>t：

    action_target(t) = [p_TCP_world(k), q_TCP_world_xyzw(k), gripper_open_target(k)]

这是 8 维绝对目标，与当前 replay action_shape=(8,) 对齐。训练另提供 gripper_pose[7]、rot_grip_action_indicies[4] 和现有 translation 标签。模型输出的 ignore_collisions 另行适配，不是 OHT 原始 action 的第八维。

保留 PD 和 OSC 全部有效轨迹；不使用原 action[:6]，不按 7.9 全局缩放。ee(t+1) 可以作为密集目标对照，但首版优先 keypoint/horizon：60 Hz 相邻运动可能太小，直接照搬下一帧标签容易产生近似停留策略。

metadata 路点作为事件匹配与目标验证来源；先把路点/日志映射到实际轨迹帧。目标主标签取真实到达位姿，理想路点标签另立实验。

### 6.2 关键点

默认 `keypoints.method: gripper`：`actions.keypoints()` **只提取实测夹爪滤波后二值状态的切换帧**，不生成停稳、指令/运动跨度或人工终帧 goal。目标取切换帧的实测 TCP 与新夹爪状态；这是开闭事件，不是抓取/释放成功 GT。

`replay.build()` 先在完整轨迹提取事件，再按 sample_stride 采样输入并保留非最后事件帧作为输入；每个输入指向严格更晚的下一事件。最后一次开闭及其后的尾段无未来事件，不生成样本。无事件 episode 在解码视频前跳过，列表记入 `complete.json.skipped_episodes`；全无事件则拒绝完成空缓存。

保留两个对照模式：`bridgevla` 为开闭/停稳/终帧，带原版 4 帧 stopped buffer 和终帧相邻点去重；OHT 停稳用相邻 TCP 平移/旋转速度及夹爪稳定窗口，初值 `0.01 m/s`、`5°/s`，不是未经核实的关节速度。`geometric` 另含指令/位移/转角/帧距补点，旧配置未设置 method 时维持此行为。切换模式要替换外部 YAML 的整个 keypoints 块或直接使用新版仓库配置，在新目录重建 replay/teacher；原始数据和 audit 可复用。

开闭事件模式仍沿用 BridgeVLA 的未来绝对动作标签，但不是完整 RLBench 关键帧启发式。它不监督释放后撤退或无开闭的插拔/路径转折；需用专家端点验证执行与覆盖，不能承诺直接跨越中间路径可避障。不跨 episode 生成目标，决策步预算按实际关键点数量设置，不继承硬编码 episode_length=25。

### 6.3 夹爪与低维状态

- open=1、close=0。
- raw action 包括夹爪维度都不可信，只保留审计统计；标签完全来自 `observation.state[6]` 的实测开合。
- 用 dataset-global motor 端点得到 open01，再用参考代码的因果“滞回＋差分”得到 observed binary。当前输入用当前实测量，未来 keypoint 用该帧的实测二值标签；在线使用同一 `gripper_step()`，episode reset 清空滤波状态。接触后未完全闭合不等于抓取失败，差分规则可在中间开度识别闭合动作，但不是物体 grasp GT。
- 两指负 rad 不能直接送进现有 extract_obs：该函数裁到 [0,0.04]，会把负值全部变为 0。使用 OHT 专用归一化/标定；若保留旧输入尺度，显式定义兼容映射，不能宣称那就是物理米制开度。
- 首版 low_dim=[measured_open, left_opening_feature, right_opening_feature, time_feature]。time_feature 可先固定 0，训练和推理一致；若启用，只由在线已执行步数/固定预算计算，不用真实剩余轨迹长度。
- 全部关节角和当前 EE 输入作为后续独立增强，不能只给 assistance 加这些信息再与旧 baseline 比较。

### 6.4 两种语言与评估协议

主实验建议使用每个 episode 固定的 task-level goal，例如 Assemble the left wheel onto the OHT axle。此时预测策略从当前观测推断阶段，分段 instruction/held_obj/日志仅作为 teacher 或审计。

可另做 segment-level 指令实验，便于定位单技能能力；但必须规定指令供应者和阶段切换来源。按专家 instruction_id 或轨迹帧自动切换属于 privileged/teacher-assisted 评估，应独立报告。

不得 baseline 用总任务指令、assistance 用 GT 当前段指令；也不得依据 held_obj GT、metadata 下一路点或 GT phase 为预测模型选择 T/R。

## 7. 阶段 D：统一执行器与无模型专家验收

拟新增 finetune/OHT/runtime/{isaac_env,executor,success}.py。用户补充了已有 H-VLA/IsaacLab 测试入口，应优先复用客户端的环境/reset/执行/success 机制，再增加 BridgeVLA 协议适配，而非重新建立一套仿真任务；需先按第 16 节核查实际脚本。

在训练前先用真实专家 keypoint 测试绝对目标位姿执行：

1. world TCP → 机器人基座/控制器要求的参考点与四元数顺序。
2. IK/轨迹规划，连续种子/限位与解分支处理；可用 metadata.ik_joint 验证 IK，但不作为 policy 推理输入。
3. 大范围移动与局部接近的执行设置；保持所有实验共用同一策略和参数。
4. pose 到达、夹爪事件、接触稳定、超时和失败明确记录；不传送机械臂位姿代替执行。
5. 一个 policy step 是目标位姿调用，内部可执行多个 physics steps；仿真 60 Hz 不等于要求 VLM 60 Hz。
6. 控制模式若切换，使用在线可得的目标距离、测量和通用规则；不能读取专家 segment 决定 PD/OSC。

没有 OHT ignore_collisions 标签。首版保留张量兼容，但新增独立 collision loss mask/weight=0，执行器忽略预测 collision bit，使用固定明确的规划规则；不要伪造标签并宣称模型学会碰撞预测。add_rgc_loss=False 会同时关闭 rotation/gripper，不适合仅关闭 collision。

先执行保存的密集轨迹验证场景与控制；再执行重建稀疏关键点，比较两者成功率、末端误差、抓取/插入事件。若稀疏专家执行失败，先调整关键点/执行器，不能用模型弥补这一接口错误。

成功判定：装配检查目标轮与轴的相对位置/姿态及释放后的保持；拆卸检查轮已离轴、落在托盘有效区域且稳定。容差、home 是否必需、仿真超时固定到协议。GT 可用于环境评分，评分字段禁止进入预测 observation。

阶段验收：四任务的专家目标均可稳定执行，误差在实际任务要求内，失败原因可追溯。不预先承诺成功率数字。

## 8. 阶段 E：OHT Baseline

拟新增 baseline.yaml、train.py、eval.py 和独立启动脚本。配置从原始 BridgeVLA baseline 出发，关闭全部 object prior/predictor：

~~~yaml
use_oracle_objects: False
use_predicted_objects: False
oracle_prior_adapter_rank: 0
oracle_relation_gated_adapter: False
oracle_relation_anchor_rank: 0
object_slots:
  enabled: False
object_conditioning:
  shared_action_features: False
  use_context: False
rvt:
  object_prior_mode: none
  oracle_prior_mode: none
  oracle_prior_relation: False
~~~

以上仅示意已有模型开关，完整 OHT YAML 还需数据、动作、执行器和 collision loss 新参数；不是当前可运行配置。

训练顺序：

1. 每任务抽取一个 episode 检查 batch；单步训练检查 finite loss、梯度、checkpoint。
2. 单任务 5–10 episodes 做小样本拟合；验证平移/旋转/夹爪误差确实下降，而不是只看到总 CE 下降。
3. 在未参与拟合的 episode 上做 open-loop 与闭环 smoke。
4. 四任务 task-uniform 联合训练；固定有效 batch 和 optimizer steps，记录训练样本数与关键点分布。
5. 使用同一个公开/已取得的预训练初始化方案；跨机器人 RLBench checkpoint 只作 initialization，不能宣称零样本适配。

先冻结 vision tower、采用现有内存配置，按 OHT val 调预算；不把 bs=48 或原服务器参数当作新硬件可行值。需要域适应时分阶段解冻，并给 matched baseline 相同解冻范围。

初始关闭大幅 SE(3) augmentation，几何验证后再开小幅增强。所有角色点、当前输入 pose（若新增）、目标 pose、相机衍生几何使用相同变换；含“左/右”等语言时核查变换是否保留语义。

当前默认 72 rotation classes 为每轴 5° 分桶；平移精度由 bounds、renderer 与 coarse/refine 共同决定，不等同于 replay 的 100 voxel 网格。先测实际任务容差。若精度不足，缩小合理工作区、检查 stage_two，再独立比较更细旋转/连续旋转头或局部控制。改变旋转类别须同步 peract.num_rotation_classes、mvt.num_rot/feat_dim 和 checkpoint 加载策略。

## 9. 阶段 F：OHT Target/Reference Teacher

OHT 只有四物体中心和姿态，未提供实例 mask、mesh 尺寸或轴 site 几何，不能直接满足当前 O2 表面点/区域点输入。

拟新增 roles.yaml、role_manifest.py、role_teacher.py、validate_roles.py。

| 阶段 | Target | Reference 建议 | 必须核查 |
| --- | --- | --- | --- |
| assemble 取轮 | 当前待操作新车轮 | 有实际意义且已定义的支撑区域；否则按协议定义 NULL | 顶层轮是否能与堆叠其他轮区分 |
| assemble 搬运/插入 | 同一被搬运轮 | 左/右轴插入 site | 轴原点、轴向、有效插入区域 |
| disassemble 接近/拔出 | 指定侧现有轮 | 对应轴 site | 被夹持后 Target ID 不随物理移动改变 |
| disassemble 搬运/放置 | 同一拆下的轮 | 托盘承接区域 | 托盘可能移动，不能永久使用首帧位置 |
| home/无交互尾段 | 无活动 Target | 按定义无 Reference | Target NULL head 当前未实现，屏蔽角色监督但保留动作学习 |

这张表是需审核的建议规则，不是已验证 GT。不要从未来 EE waypoint 的最近物体反推 T/R；关系区域不必等于 action waypoint。

Teacher 生成优先级：

1. 原始同步实例 mask + 米制深度：提取当前可见物体表面点。
2. 相同资产/状态重建并渲染实例 mask：先证明与保存 RGB/深度配准，再使用；不能混用 simulator ID namespace。
3. 已知 mesh + 物体 pose：生成完整几何属于 CAD/privileged geometry；若用于当前可见 teacher，应渲染并核查遮挡，结果按来源分开。
4. 仅有中心：可做 center-only 诊断，但不是实例表面点 GT；不能重复中心 512 次冒充几何。

shaft site 从真实资产/标定和局部坐标定义，托盘从区域边界定义；新轮/已装轮中心不能替代轴口。site 表示任务空间区域，不宣称其 prior 是严格可见实例分割。

语义存在、几何可用、标签已知必须分开。被遮挡且应存在的 Reference 是 unknown geometry，不是 NULL。

新增 OHT contract，记录 schema、role_config hash、坐标/TCP/camera/depth 版本、geometry_source、episode/frame、teacher_valid/present_known 和划分。当前 RLBench contract 固定 stored handles 和有限 phase_source；不要伪装为 rlbench_o2_semantic_roles_v2，也不要关闭审核绕过。

可选 Oracle 诊断只回答“角色几何正确时是否有潜在收益”。Oracle 不收益不意味着所有预测路线必然无效，但应先检查 teacher、接口、精度和训练范围，再扩大训练。

## 10. 阶段 G1：内部 Predicted Object Assistance（推荐主线）

选内部两个有序 role queries，而非首先训练全场景实例 detector。理由是当前动作模块需要当前 T/R，四任务具有明确交互角色，且仓库已实现这一结构。是否优于 slots 仍需实验。

前向：

    当前 RGB-D → 场景点云 → coarse 三视角/VLM features
    + task instruction + 当前夹爪测量
    → Target/Reference queries → soft role maps/tokens/geometry
    → relation/anchor → coarse waypoint
    → refine（继承当前步角色）→ 绝对 TCP pose + gripper

训练 teacher 只进入 role-map/presence loss，动作前向使用预测 maps；推理不需要 GT 物体 pose、held_obj、instance ID、GT phase 或 GT 角色点。

以 [rlbench_o2_role_queries.yaml](../../finetune/RLBench/configs/rlbench_o2_role_queries.yaml) 为结构参考，新增 oht_role_queries.yaml：

- object_prior_mode=o2_internal_slots，predictor_type=role_queries，num_slots=2。
- shared_action_features/use_context=True，inherit_coarse_roles/preserve_role_tokens=True。
- inherit 需要 stage_two=True、XYZ correlation channels 和 anchor；完整配置必须校验这些依赖，不能只复制部分 flags。
- 当前 use_oracle_objects=True 是读取训练 teacher 的 schema 选择，不表示允许 Oracle 进入动作前向；OHT 入口明确区分 teacher_loading 与 policy_source。
- reference NULL head 只监督真正语义 NULL；低 confidence 或无点不标为 NULL。当前无 Target NULL head，home 阶段用 loss mask。
- 当前 role-query confidence 主要表示 map support，不能直接解释为校准的身份置信度。

训练：

1. 从同一 OHT baseline checkpoint 初始化 predictor/adapter 新权重，重新建立 optimizer；不是 resume 旧 optimizer。
2. 冻结 backbone/action heads，先验证角色损失、teacher 支持、梯度和辅助前向稳定；这是 warm-up，不代表已有 action 提升。
3. 解冻同样的 action heads/选定 VLM 层，action loss + role-map loss + Reference NULL loss 联合训练。
4. 全部正常动作样本保留 action loss；role teacher 无效的样本只屏蔽对应辅助项。
5. 先验证 coarse/refine 继承一致性，再分析无遮挡、局部遮挡与插入阶段。当前没有跨步 object memory，完全遮挡性能需实际测量。

参考损失：

    L = L_translation + lambda_R L_rotation + lambda_G L_gripper
        + lambda_role L_role_map + lambda_null L_reference_NULL

collision 首版 mask 掉。权重由 val 选择，所有额外训练预算记录。若 teacher 暂时无法获得，可做 action-only predictor 诊断，但不能声称已经学出可靠物体语义。

## 11. 阶段 G2：外部 Predicted Object Assistance（独立路线）

适用于已有或希望独立部署 detector/segmentor 的情况。当前仓库没有这个预测器，必须新增训练/推理 wrapper。

当前观测 RGB → 物体/区域候选 mask → 用当前米制深度恢复 XYZ → 语言/测量条件的 T/R 选择 → relation/anchor → action。

对象识别与任务角色选择是两个步骤：检测出 wheel/plate 不能自动决定左右任务、当前搬运轮或 shaft site。shaft 还可能需要区域检测/姿态估计/CAD 注册；任何在线配准都必须来自可部署信息，不能直接取 simulator GT。

每角色输出：

| 字段后缀 | 类型和含义 |
| --- | --- |
| predicted_{target,reference}_object_points | float32 [512,3]，与 scene 一致的 world XYZ |
| predicted_{target,reference}_object_valid | bool，当前几何可用 |
| predicted_{target,reference}_present | bool，预测语义角色存在 |
| predicted_{target,reference}_confidence | float32，预测可靠性 |

点数不足可确定性重采样并保留原始有效点数审计；无有效几何时字段仍存在、valid=False，不能用 GT 填补。记录预测器、checkpoint、prompt/阈值、相机、depth 版本和置信度来源。

实施：

1. predictor 只在 train 训练，val 调参；如使用监督训练预测器，优先用 out-of-fold/cross-fitting 给 policy train 生成预测，降低训练过拟合的理想预测与部署预测之间的差异。
2. 离线导出与 replay sample_frame 对齐的 predicted fields，绝不使用下一关键点/未来帧定位对象。
3. 新 OHT loader 全量校验字段和 provenance；保留现有 use_predicted_objects=True、use_oracle_objects=False、object_prior_mode=o2_predicted_relation。
4. 从与其他方法相同的 baseline checkpoint 训练 adapter，再进行匹配预算的 action 微调。
5. 在线 wrapper 在每次 act 前对当前 RGB-D 运行同一个 predictor，使用相同预处理和置信度规则；reset 清理所有缓存。
6. 单独测试预测缺失、低置信度、错轮、reference 遮挡与超时；报告 residual 实际启用比例。

现有外部路径的重要边界：bridgevla_agent.py::_select_oracle_prior_points 会在 present=True 但 reference invalid/低置信度时关闭 Target 的有效性，即关闭整对 residual；内部 soft 路径的行为不同。strict=True 是缺字段报错，不是低置信度自动报错。

第一版保留该行为并报告每阶段 pair coverage，避免把新 fallback 机制同时混入主对照。如果要改成可靠 Target 单独辅助，作为独立消融，新增显式 presence/geometry 语义测试，不把 unknown Reference 当 NULL。

## 12. 实验矩阵与公平对照

| 编号 | 方法 | 动作前向 object 来源 | 目的 |
| --- | --- | --- | --- |
| B0 | OHT baseline | 无 | 基础性能 |
| B1 | baseline 继续训练 | 无 | 匹配 assistance 的额外优化预算/解冻范围 |
| G | Oracle T/R + relation/anchor | GT | privileged 诊断，单报 |
| P1 | 内部 role queries assistance | 当前观测预测 | 推荐主方法 |
| P2 | 外部 predicted assistance | 独立 predictor | 感知模块化对照 |
| A1 | P1 同 checkpoint，关闭 residual | 无 | 该模型对辅助信息的依赖；不等同于 B0/B1 |
| A2 | P1 不继承 coarse roles | 内部预测 | 本步角色一致性消融 |
| A3 | P1 不使用 role teacher loss | 内部预测 | 区分辅助监督和结构收益 |
| A4 | predicted T/R 扰动或角色交换 | 受控错误预测 | 判断收益是否依赖正确角色 |

P1/P2 各自与 B1 用同一初始化、有效 batch、optimizer steps、相机和控制器比较；若 routes、训练范围不同，结果属于整套方案收益，不能归因于“预测更准”一个因素。内部与外部使用的 shared route 差异另作解释或补齐匹配对照。

三个训练 seeds 是训练随机性；每个 seed 在相同 scene/reset cases 上配对评估。没有完整 simulator 初始状态时，不能仅凭保存的 episode 编号宣称复现了同一闭环场景。

报告：

- 四任务单独成功率和 macro 平均；完成次数/要求次数；配对差值与置信区间。
- 抓取、拔出/插入、托盘放置、撤离失败；IK、限位、执行超时和碰撞失败。
- world/TCP 平移误差、旋转 geodesic 误差、夹爪事件误差；插入区域的轴向/径向误差。
- teacher 定义对应的 T/R map 指标、presence/NULL、有效几何/pair coverage；site prior 不称作严格 mask IoU。
- policy inference、predictor、renderer、controller 分别耗时及端到端耗时；峰值显存。
- 推理预测 crop 下的动作指标，不只看训练 GT crop/waypoint 条件下 CE。

测试集规模按有效划分与新仿真 cases 确定；每任务 10 个离线 test episodes 的统计可能较宽，应报告不确定性。固定测试完成列表，双方都中断不能算完成配对实验。

## 13. 建议代码目录与交付顺序

以下全部为拟新增或拟重构：

~~~text
finetune/OHT/
  configs/
    dataset.yaml
    camera_calibration.yaml
    roles.yaml
    baseline.yaml
    oracle_diagnostic.yaml
    role_queries.yaml
    predicted_external.yaml
  data/
    reader.py
    video.py
    geometry.py
    observation.py
    keypoints.py
    replay.py
    contract.py
    role_manifest.py
    role_teacher.py
  runtime/
    isaac_env.py
    executor.py
    success.py
    predicted_wrapper.py
  train.py
  eval.py
finetune/bridgevla/data/
  replay_schema.py
  observations.py
tools/
  audit_oht_dataset.py
  build_oht_replay.py
  validate_oht_replay.py
  validate_oht_roles.py
  export_oht_predictions.py
tests/
  test_oht_geometry.py
  test_oht_action_labels.py
  test_oht_dataset_split.py
  test_oht_contract.py
  test_oht_predicted_inputs.py
~~~

优先交付的变更：

| 顺序 | 交付内容 | 验收门槛 |
| --- | --- | --- |
| 1 | audit、split manifest、depth/pose 标定证明 | 全量索引通过；米制几何可用 |
| 2 | 通用 schema/预处理解耦、OHT reader 和 replay | 无 RLBench simulator 也能加载；张量/语言/动作对齐 |
| 3 | executor/success、专家执行报告 | 专家目标在四任务可执行 |
| 4 | baseline train/eval、matched continuation | 小样本拟合与 held-out 闭环成立 |
| 5 | OHT roles/teacher/contract、Oracle 诊断 | 几何/语义审核通过 |
| 6 | role queries assistance | GT 隔离、梯度、闭环完整 |
| 7 | external predictor/cache/wrapper | 离线与在线一致、pair coverage 和缺失行为明确 |
| 8 | 全实验矩阵与报告 | 固定预算、完整测试、可追溯统计 |

只有 1–4 完成，才可称为“支持 OHT baseline”；只有 5–6 或 5–7 完成且预测推理不读 GT，才可称为“支持 OHT predicted object assistance”。

## 14. 必需验证

围绕迁移风险编写数值/集成测试，不以 YAML 文本检查替代实际运行：

1. K/外参/光学轴/xyzw-wxyz/TCP 往返，已知点投影及跨相机对齐。
2. 视频 PTS、任务 ID 和 episode 边界；重复 episodes.jsonl 不生成重复样本。
3. 任意扰动 raw action 七维不改变关键点、观测和动作标签；夹爪由实测 motor 状态及端点得到，不把负角裁成全零。
4. metadata keep、角度单位、关键点顺序、动作时间偏移和缺失帧处理。
5. SE(3) 对 scene、teacher/predicted points、动作及新增当前 pose 保持同变换。
6. 预测 inference 删除/扰动 GT object/held_obj/phase 不改变动作；baseline 不读取 object 字段。
7. 训练中在保持观测、动作标签/teacher-forced crop 和随机性不变时，只扰动 role teacher，动作前向预测应不变，辅助 loss 可改变。不能拿改变 GT crop 导致动作变化误判为 teacher 泄漏。
8. internal teacher invalid/unknown 不变 NULL；home 不产生伪 Target 正例；coarse/refine 坐标一致。
9. external 缺字段 strict 报错；valid=False 与低 confidence 按既有语义处理；reference unavailable 关闭整对行为可复现。
10. 单步 GPU 训练、checkpoint/init/resume 和单 episode Isaac smoke；数据/标定/预测器版本变化拒绝错误恢复。
11. 专家目标执行、完整测试 case 计数、配对统计和失败日志。
12. 网络请求/响应的 episode、step、timestamp、动作坐标和表示一致；拒绝旧 episode 响应，reset 后清除 action chunk/角色缓存；双服务路由和阶段切换有可审计日志。

现有 tests 回归用于保护 RLBench 行为；目标 Linux/CUDA/Isaac 环境仍需实际执行上述验收。本文未宣称已经完成任何 OHT 实验。

## 15. 实施前需取得的输入

这些是数据/环境依赖，并非要求用户先批准方案：

- 一个完整 episode 的 Parquet、RGB/深度视频、info.json、metadata 和采集日志；随后全量数据访问。
- depth writer/编码说明或原始米制深度。
- 实际版本的 Isaac Sim 场景/资产、机器人 URDF/USD、TCP/link 定义、相机导出代码与基座姿态。
- shaft/plate site 定义；实例 mask 或可配准的资产几何/补标途径。
- reset、控制器、success API 与可复现初始场景；训练 GPU 和可用预训练权重路径。

先用上述输入消除深度和执行接口的不确定性，再决定精度升级、外部感知模型或跨步 memory。它们不应在第一版迁移中同时成为变量。

## 16. 补充：接入已有 H-VLA 2.0 / IsaacLab OHT 测试环境

本节依据用户提供的测试 README，不包含远程代码审查或运行结果复现。连接凭据不写入文档、配置或命令。

### 16.1 已提供的环境与入口

| 用途 | 用户提供的路径/脚本 | 接入时的处理 |
| --- | --- | --- |
| H-VLA 工程 | /data/xf.peng/code/HVLA/ | 参考 server 的请求/响应格式与模型加载，不覆盖既有工程 |
| IsaacLab 工程 | /data/xf.peng/code/IsaacLab/ | 优先复用 task、reset、观测和执行 |
| H-VLA 环境 | /data/xf.peng/miniforge3/envs/hvla/ | 既有 H-VLA 服务使用；BridgeVLA 依赖需单独核对 |
| IsaacLab 环境 | /data/xf.peng/miniforge3/envs/isaaclab_oht/ | 保留仿真环境，模型环境与之分离 |
| H-VLA checkpoint 根目录 | /data/xf.peng/model/HVLA2/oht_v423/ | 用于原方法参考，不是可直接加载的 BridgeVLA 权重 |
| Task 1 server | shell/hvla2_server_8004.sh | README 指定端口 8004 |
| Task 1 client | run_play_hvla2_v2_oht_task2_8004.sh | 虽含 task2，按实际实现确认任务，不自行改名 |
| Task 2 servers | shell/hvla2_server_8000.sh、shell/hvla2_server_8001.sh | README 要求先启动两个服务；具体路由尚待核查 |
| Task 2 client | run_play_hvla2_v2_oht_task2_8001.sh | 核查是否在内部调用另一个服务、由谁切换 |

用户 README 要求先启动所需 servers，再启动 client。BridgeVLA 测试采用新增脚本和可配置端口，避免占用/替换正在使用的服务；不能只修改模型路径后把 H-VLA server 当作 BridgeVLA server。

### 16.2 现有结果的解释边界

| 用户报告的配置 | Task 1 | Task 2 |
| --- | --- | --- |
| relative_eef | 19/20 | 0/20 |
| joint | 未报告 | 0/20 |
| stage-wise joint + relative_eef | 未报告 | 14/20 |

这些数字属于 H-VLA，不能视作 BridgeVLA baseline 或独立复现。每组 20 次的结果值得作为调试依据，但尚缺 seeds、初始场景配对、失败归因和阶段切换细节。

结果提示 Task 2 的动作表示、阶段路由或相关训练配置可能是重要因素；不能据表格断定失败完全由控制器导致，或承诺 BridgeVLA 也会得到同样收益。文档中的 Task 1/Task 2 与 v423 的四任务目录映射尚未知，必须先查 task 注册和启动参数。

### 16.3 接入前的只读代码核查清单

按照上述 shell 脚本追踪 Python 入口和配置，记录：

1. Task/env ID、左右侧、装配/拆卸、子任务边界；reset seed、随机化参数和 success 定义。
2. server 使用的具体 checkpoint、normalization/action statistics、推理 horizon、action chunk 长度与重规划间隔。
3. 请求是否包含五路 RGB、米制深度、内外参、测量 EE/TCP/关节/夹爪；相机名称、分辨率、编码和 timestamp。仅 RGB 的旧协议需要扩展，不能满足 BridgeVLA 点云输入。
4. relative_eef 是增量 pose、速度还是归一化网络输出；平移坐标、旋转表示/body-vs-base、TCP、单位、scale/clamp、夹爪编码及命令应用次数。
5. joint 是绝对目标位置、关节增量还是速度；关节顺序和 angle wrap、限位、IK 种子及执行频率。
6. Task 2 双服务如何路由：两个不同模型、ensemble、级联、fallback 或其他机制；不能仅凭端口推断。
7. 阶段切换来自 policy 预测、在线测量规则、GT held_obj/phase，还是固定专家时间表；记录进入 observation 与仅用于评分的字段。
8. 同步/异步 transport、超时重试、reset 和 action chunk 缓存；旧响应是否可能在新 episode 执行。

交付 protocol.md 与 task_mapping.json，记录源文件/配置版本。以上内容未知时继续完成离线审计，不启动猜测动作格式的闭环。

### 16.4 推荐 BridgeVLA 部署结构

~~~text
IsaacLab client（现有环境）
  → 当前 RGB-D + 标定 + 测量状态 + goal + episode/step/timestamp
  → BridgeVLA server（独立模型环境，新增入口）
       baseline：不读取 object 字段
       internal：预测 T/R → relation/anchor
       external：运行 predictor → predicted fields → relation/anchor
  ← 绝对 TCP 目标 + 夹爪目标 + action schema + 请求标识
  → client 侧动作适配器 → 现有机器人执行 → 重新观测
~~~

三种方法共用协议、客户端和执行器。请求应明确 policy 可读字段，不把全部 simulator state 无筛选透传；GT 可留在单独 teacher/scoring 通道。每个响应绑定 episode/step，并处理预测和执行之间的时间延迟。

先独立确认健康状态和 schema，再用 hold/小范围已知目标及专家目标验证。不要复用旧 H-VLA action chunk 缓存：BridgeVLA 的下一关键点绝对目标与原模型的一串连续增量不是等价接口。

### 16.5 保留绝对位姿策略，适配 relative_eef 执行

首版保持第 6 节的绝对 TCP 目标，不因 client 名称改成直接学习原始 action。client 若要求 relative_eef，使用当前测量 TCP 与目标位姿构造误差：

    delta_p_base = p_target_base - p_current_base
    R_error_body = transpose(R_current_base) * R_target_base
    R_error_base = R_target_base * transpose(R_current_base)

根据实际 controller 选择其中一种旋转误差，再转换为其 axis-angle/Euler/quaternion 等表示，应用已验证的 scale/clamp 和积分时间。若接口是速度控制，还需按闭环控制律生成速度，不能直接把位移差当速度。

持续跟踪同一绝对目标时，按执行时刻的当前测量重新计算误差，避免反复积分旧 delta。姿态和参考点先统一后求差，不对四元数分量直接相减。由适配器读取当前 EE 不等于给策略增加 EE 输入；所有比较方法使用相同适配器信息。

长距离移动也可以先从预测 TCP 通过在线 IK/cuRobo 生成关节轨迹；这是共享执行器。它与“模型直接预测 joint，再切换另一 relative_eef 模型”是两种不同方案，不能混为同一 hybrid 实验。

### 16.6 Task 2 的控制变量与新增对照

建议先跑统一 absolute-pose policy + 已验证执行器。若 Task 2 仍受移动/接触阶段执行瓶颈影响，再增加共同的 hybrid executor，对 baseline 与 assistance 同时启用。切换规则只能用声明的在线可得信息；若暂用 GT phase，应标为 privileged-stage 诊断。

| 控制设置 | BridgeVLA baseline | Predicted assistance | 实验问题 |
| --- | --- | --- | --- |
| E：同一绝对目标→EEF 执行器 | B-E | P-E | 同控制条件下物体辅助是否有效 |
| H：同一绝对目标→关节规划/局部 EEF 混合执行器 | B-H | P-H | 换执行器后辅助是否仍有效 |

主要比较 P-E 对 B-E、P-H 对 B-H；控制器收益比较 B-H 对 B-E。P-H 对 B-E 只能解释整套方案收益。H-VLA 的历史 19/20、14/20 放在单独参考表，复现并匹配评估条件后才作模型横向比较。

若最终决定实现双模型 learned-joint + learned-relative_eef，需新增关节动作头/模型、对应监督、训练预算和可部署 router；这超出仅复用当前 BridgeVLA 绝对目标头的首版迁移，单列后续阶段。不要把原 H-VLA joint server 静默接入后称作纯 BridgeVLA baseline。

### 16.7 更新后的实施优先级

1. 确认两个 README 中 Task 定义的对应关系，核查上述启动脚本、客户端、模型/action 配置。
2. 沿现成采集/相机代码核查 depth 编码、光学轴、TCP 和归一化；完成离线数据审计。
3. 固定协议，新增 BridgeVLA server 和客户端适配器；验证 hold、小范围目标、专家稀疏目标。
4. 训练 B-E baseline；必要时先用同策略输出验证 B-H，消除执行瓶颈。
5. 在完全相同执行器上加入内部/外部 predicted assistance；记录角色覆盖、阶段路由和端到端耗时。
6. 固定 reset cases、语言与阶段信息、预算，完成控制×物体辅助对照及三训练 seeds 评估。

上述路径和环境信息降低了重建仿真环境的工作量，但尚未消除离线深度可恢复性、teacher 几何和动作协议的不确定性。


## 17. 首版实施状态（2026-10-08）

本节覆盖已写入工作区的代码；验证边界以此为准。

| 模块 | 实际入口 | 完成内容 |
| --- | --- | --- |
| 原始数据审计 | tools/audit_oht_dataset.py | 全列 Parquet、四任务唯一身份、轨迹/场景分组、质量报告和固定划分 |
| 观测/动作缓存 | tools/build_oht_replay.py | 参考帧序号配对（可选 PTS）、显式米制 depth、逐帧外参、world XYZ/TCP、未来关键点及夹爪标签；可按间隔保存 RGB-D、全局/GT 局部三视图及 TCP 诊断 PNG |
| 缓存预检 | tools/validate_oht_replay.py | contract/文件哈希、split 隔离、观测 shape、教师/预测命名空间及覆盖 |
| 角色教师 | tools/build_oht_role_teacher.py | 显式可见表面 mask 或 site_region，区分 present/known/valid/NULL；可保存 T/R 叠加预览 |
| 外部预测缓存 | tools/predict_oht_objects.py | module:factory 插件、当前观测白名单、provenance、逐 episode reset |
| 三种训练模式 | finetune/OHT/train.py 与 configs | baseline / role_queries / predicted_external，任务均匀采样、累积、DDP、checkpoint/严格续训 |
| 推理服务 | finetune/OHT/server.py | 版本化 absolute world TCP/xyzw 协议、typed tensors、episode 重置与过期请求拒绝 |
| 执行转换 | finetune/OHT/runtime/executor.py | 显式 base/body 相对位置与旋转向量误差、限幅；非完整 IsaacLab 控制器 |
| 评估 | finetune/OHT/eval.py | held-out 离线误差与仿真插件闭环、失败纳入分母、cases/执行器契约记录 |
| 通用 Agent 解耦 | bridgevla/data/observations.py 等 | 通用预处理、仿真/数据增强依赖延迟加载、OHT 碰撞损失置零，其他默认权重保留 1 |

验证使用 12 个合成 OHT episode、真实 Parquet/MP4/米制深度，生成 60 条 transitions。覆盖 baseline batch、教师与预测缓存、无 GT 推理隔离、真实 RVTAgent 梯度累积和优化器更新、HTTP 本地收发及闭环失败计数。渲染器/VLM 用轻量 CPU 替身，完整模型及真实仿真不在此次验证范围。现有 role queries、跨尺度继承、角色特征保留和辅助损失测试亦已回归。

构建 buffer/teacher 时可用 `--visualize-every N`，默认关闭，`1` 覆盖每个生成样本；可指定 `--visualize-output-dir`。PNG 与缓存数据分开：全局三视图保留最多 20 万点，以 3×3 像素绘制改善显示空洞，并标出 GT 局部范围；局部三视图围绕下一 GT keypoint 各轴 ±0.20 m，明确标注 **GT-centered refine diagnostic**，不是模型真实二阶段输出。均为 CPU 诊断投影，不修改缓存 XYZ；site 点投影不代表可见性已验证。参数、颜色和输出路径统一见 [OHT 运行说明](../../finetune/OHT/README.md#生成时可视化)。

尚未完成且需要现场信息的工作：

1. 真实 v423 全量读取、跨相机几何及工作区覆盖检查；无需重新标定。按提供代码解析 gray12le linear/log 量化、metadata K/夹爪端点、显式 camera/EE 顺序；TCP 不重复偏移。需补公共 `pointcloud_transforms.py` 核对 ray/Z，并在服务器复核 frame 965 腕部参考投影 `(313.0,378.6)` 及多帧几何。
2. 从仿真或标注工具批量导出语义角色 masks/site；首版消费显式标注，未自动实现数据集角色路由和 mesh 重建。
3. 完整 CUDA/PaliGemma/point-renderer smoke 与三 seed 训练，没有新 OHT 成功率。
4. 核查 H-VLA/IsaacLab 脚本及 Task1/2 映射、真实采集和控制 API、EEF/IK/hybrid 执行器、专家目标回放和配对闭环。
5. 新的外部感知模型/权重：已有严格插件接口，内部 role queries 复用仓库现有真实网络；未提供未经训练的替代检测器。
6. 自动扰动/遮挡场景生成与统计置信区间仍属后续实验工作。

本轮尝试连接用户提供服务器的 SSH 22 端口超时，未读取或修改远端。没有把 H-VLA checkpoint 当作 BridgeVLA 权重，未把历史 H-VLA 成绩记作本次结果。

本地验证结果：20 项 OHT 集成测试、137 项相关既有回归测试和 6 项子测试通过；8 个新增命令入口及 Python 编译检查通过。
