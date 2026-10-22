# OHT 关键点执行与同初态对照

BridgeVLA 的 OHT replay 标签是未来关键点，不是下一物理帧的 OSC 输入。policy 默认采用 waypoint：

```text
一次模型预测 → 按本次实测状态解码一次绝对目标
  → 保持原夹爪状态，持续跟踪同一目标
  → 位置/姿态连续到位
  → 切换本次预测的夹爪状态，并继续保持机械臂目标
  → 夹爪稳定 → 下一次预测
```

关爪允许稳定在物体接触位置，不要求达到空夹爪的闭合端点；但必须有已闭合或实际闭合运动的证据，不能把完全不动的开口夹爪当成成功。稳定仅表示执行阶段完成，不表示物体已被抓住。目标漂移、停滞、超时或窗口关闭后停止本次运行，保留部分命令，不重发模型请求。

| 参数 | 默认值与含义 |
| --- | --- |
| `--execution-mode` | auto：policy 与 GT keypoints 为 waypoint；GT frame 为 streaming |
| `--waypoint-max-steps` | 600，每个目标的物理步数上限 |
| `--waypoint-position-tolerance` | 0.005 m；需按装配精度设置，例如 0.001 |
| `--waypoint-rotation-tolerance` | 0.05 rad |
| `--waypoint-arrival-steps` | 3，连续到位确认次数 |
| `--gripper-min-steps` | 10，切换夹爪后的最少控制步数 |
| `--gripper-stable-steps` | 5，夹爪开口变化稳定窗口 |
| `--waypoint-stall-steps` | 100，无足够进展时停止 |
| `--max-steps` | 模型或 GT 目标数量，不是物理步数 |

`--max-substeps`、`--joint-hold-steps` 仅用于显式 streaming；waypoint 通过单步接口跟踪，joint/relative_joint 需要匹配的关节位置环境。自定义 joint executor 若用于 waypoint，必须提供 `step(absolute_target)` 单步接口。相对 EEF/joint 始终从该次请求的测量值解码一次，不重复累加 delta。

## 相同记录初态运行策略

在 IsaacLab 服务器运行。以下假设原始数据和训练 replay 缓存在该机器可读；训练缓存至少需要完整、匹配的 `contract.json`、`complete.json`、`samples.jsonl`，初始化不读取缓存图像。

```bash
cd /home/yiwei/IsaacLab
python /home/yiwei/BridgeVLA/tools/oht_bridgevla_client.py \
  --isaaclab-root "$PWD" \
  --server http://109.105.46.126:8004 \
  --task disassemble_left --scene-preset oht-v4 --camera-source usd \
  --ee-pose-source controller --execution-mode waypoint \
  --initial-state-root /home/yiwei/BridgeVLA/oht_curobo_pd_v423 \
  --initial-state-contract /data/oht/replay-source-v3 \
  --initial-state-episode 0 --initial-state-frame 0 \
  --episodes 1 --max-steps 20 \
  --output logs/bridgevla-policy-paired-01 --device cuda:0
```

`--initial-state-root` 自动选择 trajectory 初始化，只在每个 episode 开始恢复记录的六关节/两指位置和零速度。标定、任务、profile、可选 contract SHA 及起始 TCP 必须匹配；随后依然采集实时 RGB-D、请求模型，不使用记录的未来动作或成功标签。也可用 `--initial-state-data-config` 指定标定，省略时使用仓库默认并解析数据 metadata。

物体默认使用环境 reset 状态。需要恢复记录物体时，使用 `--initial-state-object-map`，JSON 格式与 [replay-object-map](oht-replay.md#可选物体恢复) 相同；这不能恢复装配约束、历史接触或抓取关系。另有 `--initial-state-joint-names`、`--initial-state-gripper-joint-names`、`--initial-state-position-tolerance`、`--initial-state-rotation-tolerance`。建议选夹爪已稳定开/闭的初始帧；过渡开口的迟滞历史在新 HTTP session 中不会自动恢复。

## GT 关键点对照

使用同一原始 episode、初始帧和训练 contract，跳过模型，只执行该配置生成的未来关键点：

```bash
cd /home/yiwei/IsaacLab
python /home/yiwei/BridgeVLA/tools/oht_bridgevla_client.py \
  --mode replay --replay-targets keypoints \
  --isaaclab-root "$PWD" \
  --replay-root /home/yiwei/BridgeVLA/oht_curobo_pd_v423 \
  --replay-contract /data/oht/replay-source-v3 \
  --task disassemble_left --replay-episode 0 --replay-start-frame 0 \
  --scene-preset oht-v4 --ee-pose-source controller \
  --output logs/bridgevla-gt-keypoints-01 --device cuda:0
```

它按实际训练配置提取 gripper/geometric/bridgevla 关键点，不插入未来密集路径，不伪造末尾目标。源帧时间只作来源记录，目标跟踪耗时由实际控制决定。默认完整执行所有剩余关键点，`--max-steps` 可限制数量。

原来的 `--mode replay` 默认 `--replay-targets frame`，仍严格逐帧 `i→i+1`、一次控制步、校验源记录周期。两种回放不可混用执行模式。

若逐帧 GT 正常、GT 关键点仍失败，继续检查关键点执行和中间路径；若 GT 关键点正常而同初态 policy 失败，再比较在线图像/点云和模型目标。默认 `score=none` 只记录执行结果，正式成功率仍需实际任务成功条件。

## 路径与参考脚本

当前 waypoint 仍为直接 OSC/关节目标跟踪，**没有碰撞规划器**。延长跟踪或修复夹爪时序，不能恢复 gripper-only 标签省略的抬升、绕障、对准路径。`run.json.motion_path` 明确记录此限制。已有 checkpoint 不会因修改本地 keypoints YAML 而自动学到这些路径；需要规划器，或按确认过的新关键点配置重建 replay 并重新训练。

提供的 `play_hvla2_v2_oht.py` 区分可视化用 `raw_key_actions` 与实际执行的 `raw_actions` 动作块，相对动作为训练保留时间轴上的相邻状态；它还默认启用关爪锁存。其固定 OSC 目标跟踪函数存在，但 `hvla2_osc_track_max_steps` 默认 1，不能把该脚本的动作块逐步执行直接套用到 BridgeVLA 未来关键点。
