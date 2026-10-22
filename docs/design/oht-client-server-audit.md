# OHT client/server 与参考脚本对照检查

检查日期：2026-10-22。检查范围为本仓库 HTTP client/server、用户提供的
`oht-test-t1v4-gtinfer-client.py`、`play_hvla2_v2_oht.py`，以及 `1022/` 的运行日志。
参考文件仅用于核对仿真与控制定义；BridgeVLA 继续采用自己的训练观测和动作协议。
本次没有运行真实 CUDA checkpoint 或 IsaacLab 仿真，下面区分代码确认、日志现象与待验证条件。

## 已确认的缺口与修复

| 项目 | 检查结果 | 当前处理 |
| --- | --- | --- |
| 完整场景与资产 | 参考 T1V4 在 `OSC-v2` ID 下显式实例化完整 V4 config；仅使用注册 ID 无法保证相同夹爪、控制器或资产。旧 run 未记录实际 USD，不能证明当时用错版本。 | 工作区已有 `--scene-preset oht-v4`，校验配置来源、完整 V4 类与实际 USD；不单独替换一个 USD。实际配置写入 run 与诊断报告。 |
| EE 测量参考点 | 旧原生 runtime 固定读取裸 body；仓库 OHT 数据约定为已含控制器偏移的记录点，两者在非单位 `body_offset` 时不同。参考 JEPA helper 读取裸 body，不能直接据此覆盖 BridgeVLA 数据约定。 | 工作区已有 `--ee-pose-source controller/body`。原生 CLI 默认 controller，`link_to_tcp` 与 body_offset 分别只应用一次。真实 checkpoint 的记录点仍需用采集代码/数据验证。 |
| 注册 Camera 位姿 | IsaacLab Camera 的 `update_latest_camera_pose=False` 默认返回初始化位姿；仅刷新图像不能保证移动相机外参更新。 | 原生 Camera runtime 打开位姿更新。USD 相机分支继续从实时物理 body 计算腕部位姿；用户旧 run 使用 USD，故该缺口尚不能解释那次运动。 |
| 环境超时 | 参考 T1V4 在创建环境前关闭 `terminations.time_out`，我们的原生路径此前保留它，长 rollout 可能提前终止并自动 reset。 | 原生配置仅关闭命名为 `time_out` 的项，保留其他终止条件；客户端 `max_steps` 控制策略请求上限。`scene.timeout_disabled` 记录是否执行了此操作。 |
| 动作协议验证 | 客户端此前从 `mode` 等字段恢复配置，没有完整验证 frame、四元数顺序、维度、参考点、单位和夹爪定义。正确的自有服务不会因此产生错位，但不兼容服务可能被误接受。 | `/model_info.action_spec` 必须与本地动作描述一致，在启动仿真前拒绝不兼容语义。 |
| 夹爪命令与标定 | `+1/-1` 命令与配置一致，不代表配置的物理开闭方向与 checkpoint 标定一致。 | 分别记录 `verified` 与 `calibration_direction_matches`。实测命令对应目标之和的开闭方向与训练端点相反时，保存诊断并在 HTTP reset/predict 和控制 env.step 前退出；不自动反转。旧服务不提供端点时保持 null。 |
| 未来关键点与夹爪时序 | 旧 policy 每次只做一步 OSC，第一步就执行未来关键点的夹爪状态；逐帧 GT 正常不能验证这一时序。 | policy 默认 waypoint：固定目标、连续到位、再切换夹爪并等待稳定。失败保留命令并停止。dense GT 保持单步。 |
| 同初态对照 | 普通 env.reset 与 GT 记录关节恢复不是同一初态。 | policy 可显式指定 initial-state 数据，只初始化一次，后续仍用实时观测和模型动作；另提供 GT keypoints 对照。 |

注册 Camera 的默认行为和 OSC 定义已核对
[IsaacLab CameraCfg](https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.2.1/source/isaaclab/isaaclab/sensors/camera/camera_cfg.py)、
[OSC ActionTerm](https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.2.1/source/isaaclab/isaaclab/envs/mdp/actions/task_space_actions.py)。
部署机器若使用修改过的 SDK，以其实际实现及诊断为准。

## 已核对一致的执行链路

- `eef` 返回 8 维 `[world_xyz, quat_xyzw, open01]`；客户端先还原控制点目标，再变换到 robot-root，构造 7 维 OSC 输入。没有直接把 EEF8 当作 OSC7。
- HTTP 机器人/相机位姿按 xyz+wxyz 传输；进入几何运算时显式转换为 xyzw。EEF 标签离散 Euler 与推理解码均使用相同的 180 度偏移定义。
- `relative_eef` 按该请求的实测 TCP 解码一次；多个控制子步保持同一个绝对目标。参考脚本对原始轨迹增量的“从上一期望目标积分”属于另一种动作定义，不能替换 BridgeVLA 的训练语义。
- OSC 用实时 robot-root 变换，物理误差先限幅，再除以 live `position_scale/orientation_scale`。参考 T1V4 的单步默认面向相邻动作；BridgeVLA policy 的未来关键点现在按 waypoint 到位流程执行，不能据该默认值推断所有动作都只需一个控制步。
- 正常浮点二值夹爪 `open01` 经 `2*g-1` 转换；正值打开、负值关闭。训练观测使用物理关节位置之和，经 checkpoint 标定归一化；不把控制符号当作物理开口。见 [BinaryJointAction](https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.2.1/source/isaaclab/isaaclab/envs/mdp/actions/binary_joint_actions.py)。
- 原生相机输入为米制 Z-depth、实时 K、OpenGL 位姿。服务仅转换一次 OpenGL→optical；不重复执行离线深度视频的 log/量化解码，不沿用 JEPA 的 q99 或 H-VLA 的相机坐标动作。
- joint/relative_joint 必须走关节位置环境及执行器，不能送入 OSC。关节名称缺省时，仍要求实际 arm term 顺序与训练六关节顺序一致。

## `1022/` 日志能说明什么

实际记录为 eef8/world/xyzw、`env_config=null`、`initialization=env-reset`、`score=none`。
50 步输出中：

- 四元数始终为 `[0, 0, -0.6755902171, 0.7372773290]`。
- 每一步的三个旋转控制分量都在绝对值 0.15 的限幅处。
- 第 29→30 步预测目标位置跳变约 2.688 m。
- 每一步夹爪符号都符合 `2*g-1`，没有发现编码层面的反号。

持续限幅并不证明物理姿态没有变化，也不能单独证明 TCP/四元数有错。
日志没有执行前后实测位姿或实际 USD，无法将上述现象归因为某一个配置错误。
目标跳变也提示需要检查在线几何与模型预测本身，不能仅调整夹爪或延长训练。

## 仍需在仿真服务器验证

1. 排查配置时可按[运行指南的日志与排错](../guides/oht-vla-server.md#日志与排错)使用 V4 `--dry-run`，检查实际 USD、body_offset、观测→控制点变换和夹爪物理端点。需要更新并重启服务才能获得新增训练端点信息。
2. 核对 checkpoint 所用采集代码记录的是 body 还是带偏移的控制点。HTTP `ee_link_pose` 是历史字段名，不能靠名字判定参考点；选择 `--ee-pose-source` 必须与该 checkpoint 对齐。
3. 核对启动状态。参考 T1V4 paired 路径会从选定训练帧恢复六关节与夹爪；我们的旧 run 只做 `env.reset()`，不等价。可使用已有 `--initialization task` 与 task initializer 恢复明确的状态；不能把参考脚本的 GT 机器人状态当作在线策略观测替代品。物体状态也需独立对齐。
4. 先短程执行并查看 action trace 的 `execution`、`measured_before/after`、`errors_before/after`，核对到位、夹爪稳定和停止原因。使用同初态 GT keypoints 检查执行器；不能以增加 streaming 子步数替代夹爪时序或路径处理。
5. 对同一初始状态下的实时 RGB-D 与训练缓存检查点云对齐、工作区覆盖及目标位置，再比较模型的平移/旋转输出。协议测试不证明真实渲染、传感器标定、checkpoint 质量或闭环成功率。

## 自动验证

本次运行场景、控制对齐、原生 runtime、独立客户端、HTTP 与服务端六组测试：139 passed。
覆盖非零 env origin、非单位 base 旋转、非单位 body/TCP 偏移、两种 SDK 四元数顺序、
移动 Camera 更新、夹爪相反标定、错误 action_spec、会话/reset，以及非法请求不推进状态。
测试使用 CPU 替身和合成数据；实际服务器尚未验收。

新增 waypoint、同初态初始化与 GT 关键点对照的实现与限制见[关键点执行说明](../guides/oht-waypoint-execution.md)。提供的 H-VLA 客户端明确把 `raw_key_actions` 用于高层目标显示，`raw_actions` 用于逐步动作块执行，并有默认关爪锁存；不能把两者统称为 EEF 而忽略目标时间尺度。直接 OSC 没有碰撞规划器，gripper-only checkpoint 缺失的中间路径仍需独立验证。
