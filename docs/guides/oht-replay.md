# OHT GT 轨迹回放

入口为 `tools/oht_bridgevla_client.py --mode replay`，完整运行命令见[运行指南](oht-vla-server.md#oht-gt-轨迹回放)。它在 IsaacLab 中执行我们自己的原始 OHT 轨迹，不连接模型服务、不加载 checkpoint。

`--replay-*` 参数只用于 `--mode replay`；在线策略模式下显式传入会报错，避免参数被忽略后继续请求模型服务。

## 数据与动作定义

`--replay-root` 的目录结构：

```text
<raw-root>/
  disassemble_left/lerobot_dataset/data/chunk-000/episode_000000.parquet
  disassemble_left/lerobot_dataset/meta/...
  assemble_left/...
  ...
```

使用 `--task` 与 `--replay-episode` 选择唯一原始轨迹。数据检查与训练共用 `reader.validate_episode()`；标定解析、wxyz/xyzw、`link_to_tcp` 及夹爪迟滞也复用训练实现。

```text
源帧 i：初始/实测状态
  → 源帧 i+1 的实测 TCP + 滤波后二值夹爪
  → 绝对 EEF8 [world_xyz, quat_xyzw, open01]
  → IsaacLabBridge.execute（与在线策略相同）
  → robot-root 残差、物理限幅、OSC scale 补偿、env.step
```

这是实测轨迹的 TCP 目标跟踪，不是原始 OSC 控制输入的逐字节复现。Parquet `action` 不作为可信控制量；`replay-source-v3` 的样本是抽样后的未来关键点，也不用于完整轨迹回放。当前 GT 回放固定为绝对 `eef`，不解码参考脚本的 HDF 10D 动作。

## 参数

| 参数 | 含义 |
| --- | --- |
| `--replay-root` | 原始 OHT 数据根目录，必填 |
| `--replay-episode` | 数据集 episode ID，默认 0；不同于回放重复次数 |
| `--replay-start-frame` | 默认 0，必须至少还有一个后续帧 |
| `--episodes` | 重复回放所选轨迹的次数，默认 1；每次重新初始化 |
| `--max-steps` | 最多执行的源帧转换数，默认回放全部剩余转换 |
| `--replay-data-config` | 自定义数据标定 YAML，默认仓库 `finetune/OHT/configs/dataset.yaml`；仍解析该数据源的 metadata |
| `--replay-contract` | 从现有训练 replay 缓存读取已解析标定，与 `--replay-data-config` 互斥；不读取缓存中的 GT 动作或图像 |
| `--replay-joint-names` | 六个关节名，顺序与记录的 `observation.state[:6]` 一致 |
| `--replay-gripper-joint-names` | 两个夹爪关节名，顺序与 `observation.gripper_joints` 一致 |
| `--replay-cameras` | 可选：同时绑定并刷新数据配置中的实时相机。默认只获取机器人状态，不要求传感器图像 |
| `--replay-object-map` | 可选：按明确映射恢复记录物体位姿，格式见下文 |
| `--initialization` | 默认 `trajectory`；也可显式 `env-reset` 或配置好的 `task`，起始 TCP 仍需通过对齐检查 |
| `--replay-initial-position-tolerance` | 起始 TCP 允许误差，默认 0.005 m |
| `--replay-initial-rotation-tolerance` | 起始 TCP 允许姿态误差，默认 0.05 rad |
| `--dry-run` | 只读取数据、校验控制周期并保存启动诊断；不恢复轨迹关节、不执行动作 |

需要 `pyarrow` 和 `PyYAML`，以及已有客户端的 NumPy/SciPy 与 IsaacLab 环境。回放不解码原始 MP4、深度视频或加载模型依赖。无界面运行使用 `--headless`，在窗口观察时去掉该参数。

## 初始化与时间

默认在每次回放开始时恢复所选源帧的六个机械臂关节和两个实测夹爪关节，并清零其速度。写入仿真、刷新 FK 后，核对实际 TCP 与记录 TCP；资产、base、关节顺序或 `ee-pose-source` 不匹配时，在第一个控制步前退出并保存错误。初始写入只进行一次，不会逐帧 teleport 机器人。

初态限位检查使用 SDK 提供的实际关节限位，兼容新旧字段名，不把学习用的 soft limit 子区间当作物理边界。诊断中的 `joint_limit_source` 记录采用的字段，SDK 不提供硬限位时为 null。定义见 [IsaacLab ArticulationData](https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.2.1/source/isaaclab/isaaclab/assets/articulation/articulation_data.py)。

未指定关节名称时按各 action term 的实际关节顺序映射，必须与采集顺序一致；有差异时显式提供上述关节名称参数。`controller/body` 选择按源数据的实际记录点设置，`link_to_tcp` 仅应用一次。

每个记录间隔执行一个控制步；要求所选区间内的时间差与 `env.step_dt` 一致，并且 `--max-substeps 1`。不自动跳帧、插值、重复积分或改变控制周期。检查失败时应选择与采集一致的完整环境配置。记录中没有恢复所需的完整动态状态，所以即使起始 pose 一致，也不能承诺复现原采集的每个物理状态。

## 可选物体恢复

默认物体使用环境 reset 状态，日志明确标为 `object_state=environment_reset`；单纯恢复机器人不代表整个装配场景已对齐。

若已确认记录物体的顺序、姿态约定和真实 `env.scene` 刚体键，可以提供如下 JSON：

```json
{
  "quaternion_order": "wxyz",
  "assets": {
    "实际场景刚体键": 0
  }
}
```

键必须是实际注册的刚体，值为原始 `observation.objects_pos/objects_quat` 中对应对象的零起始索引；不能凭 `wheel` 等名称猜测顺序。物体记录位置按 env-origin-relative world 解释，姿态按显式的 `quaternion_order` 转换到安装的 SDK 顺序，物体参考点须与场景刚体 root 一致。

初始化写入前会拒绝机器人或其他 articulation，也拒绝通过多个键重复选择同一个场景对象；这些错误不会改写机器人或物体状态。

该参数要求 `--initialization trajectory`，只在起始帧恢复物体 root 位姿并清零速度，不会重建装配约束、抓取关系或中间帧的接触状态。复杂约束初始化继续使用自己项目的 task initializer。

## 输出与判定

- `run.json`：`mode=replay`、`model_id=null`、原始 Parquet 路径/哈希、标定来源/哈希、起点和物体映射；`server_queries=false`。
- `control_diagnostics.json`：复用在线执行器的场景、TCP、夹爪与控制配置诊断。
- `episode_*.json`：初始化关节映射、起始 TCP 对齐误差、每步 `source_frame/target_frame` 与记录时间、GT 目标、实际命令及执行前后误差；`action_source=gt_trajectory`。
- `summary.json`：轨迹耗尽为 `trajectory_end`，人为长度上限为 `max_steps`，环境终止为 `environment_done`，异常为 `error`。

记录数据中的 `next.success` 不作为本次仿真成功判定。默认 `success=null`；若需评估成功率，使用项目中实际有效的 `--score task` 成功条件。CPU 测试验证数据/执行接口，真实 IsaacLab 中的 FK、传感器和接触行为仍需服务器验证。
