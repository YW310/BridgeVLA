[文档索引](../README.md) · [数据与标定](data.md) · [训练与离线评估](training.md)

# OHT 推理、执行与回放

本页统一说明 VLA server、IsaacLab 在线执行、记录初态与 GT 回放。
当前 V4 chunk 优先使用新 `/predict` 服务和原生客户端；V3 单目标与旧 `/act` 的兼容边界另列。
数据构建、相机离线诊断见[数据说明](data.md)，模型配置、动作头训练与 open-loop 指标见[训练说明](training.md)。

命令面向 Linux VLA/IsaacLab 服务器；路径、IP、环境名按实际安装替换。
两台机器同步同一版本代码并激活各自环境，更新 server 代码或 checkpoint 后重启服务。
客户端每次使用尚不存在的输出目录。CPU 协议测试不能代替真实 CUDA、传感器几何与闭环验收。

导航：[启动与连接](#启动与连接) · [HTTP/观测](#HTTP与观测协议) · [执行](#单目标与chunk执行) ·
[记录初态](#记录初态) · [GT 回放](#GT回放) · [排错与验收](#排错与验收)。

<a id="启动与连接"></a>
<a id="安装与启动"></a>

## 启动与连接

| 用途 | 入口 | 适用输出 |
| --- | --- | --- |
| 原生 IsaacLab 在线推理 | `finetune.OHT.vla_server` + `tools/oht_bridgevla_client.py` | 单目标及 V4 EEF chunk |
| 旧 adapter 闭环评估 | `finetune.OHT.server` + `finetune.OHT.eval closed-loop` | 四种单目标动作；不支持 chunk |
| 原始 GT 回放 | `tools/oht_bridgevla_client.py --mode replay` | 无模型/HTTP；frame 或 keypoints |

新服务一个进程加载一个 checkpoint，四个任务通过 instruction/task/profile 选择；
不按任务各加载一份模型。加载、预热、predict、reset 在一条工作线程串行执行，Uvicorn workers=1。
服务端只预测；控制器、初始化、环境步进和成功评分由客户端负责。

### VLA server

先按[模型环境安装说明](../guides/installation.md#oht-数据与模型环境)准备 PyTorch/CUDA、PaliGemma 和 point-renderer；`requirements-server.txt` 仅补充 HTTP 服务依赖。在模型服务器的 BridgeVLA 根目录执行：

```bash
python -m pip install -r finetune/OHT/requirements-server.txt
GPU_ID=1 bash finetune/OHT/vla_server.sh \
  --model-path /path/to/v4-run/model_last.pth \
  --device cuda:0 --host 0.0.0.0 --port 8004 \
  --visualize-every 10 --output-dir logs/oht_vla_server_8004
```

`--model-path` 接受 checkpoint 文件或含 model_last.pth 的 run 目录；`--checkpoint` 为互斥别名。
`GPU_ID=1` 选物理卡 1，进程内仍用 cuda:0。`serve.sh` 是同一服务的兼容启动入口。
模型结构、数据契约、动作、相机和渲染配置从 checkpoint 读取，不另传一份训练 YAML。

| 可选项 | 配置 |
| --- | --- |
| 环境变量 | OHT_MODEL_PATH、OHT_HOST、OHT_PORT、OHT_DEVICE、OHT_SERVER_OUTPUT；CLI 优先 |
| 预训练资源 | `--pretrain-path /path/to/BridgeVLA-pretrain`；基础模型/processor 资源须可访问或已缓存 |
| 离线加载 | 资源齐备后设置 HF_HUB_OFFLINE=1、TRANSFORMERS_OFFLINE=1 |
| 解释器 | PYTHON_BIN=/path/to/env/bin/python |
| 预热 | `--warmup-replay /path/to/replay-v4 --warmup-split val --warmup-index 0`；契约须匹配，预热后 reset |
| 外部物体预测 | 单目标 predicted_external 所需 predictor/provenance 须与 checkpoint 记录一致 |

<a id="分机部署vla-server-与-isaaclab-client"></a>

### 连接检查

在仿真服务器执行，IP 填模型服务器实际地址：

```bash
curl http://109.105.46.126:8004/health
curl http://109.105.46.126:8004/model_info
```

`/health` 应 ready=true；`/model_info.observation_formats` 应含 bridgevla_rgbd_v1。
两机须允许访问 TCP 8004。完整动作契约读 model_info，不从 health 推断。

<a id="独立-bridgevla-客户端"></a>

### IsaacLab client

在已安装 OHT 场景的 IsaacLab 根目录运行：

```bash
cd /home/yiwei/IsaacLab
python /home/yiwei/BridgeVLA/tools/oht_bridgevla_client.py \
  --isaaclab-root "$PWD" --server http://109.105.46.126:8004 \
  --task disassemble_left \
  --env-id Isaac-Deploy-Reach-UR16e-DualEE-AllScene-OSC-v2 \
  --scene-preset oht-v4 --camera-source usd --ee-pose-source controller \
  --execution-mode auto --seed 42 --episodes 1 --max-steps 20 \
  --output logs/bridgevla-policy-01 --headless --device cuda:0
```

auto 按 checkpoint 选择 chunk 或 waypoint。示例最多 20 次预测，默认 env-reset、score=none，
因此 success=null。原生客户端只支持一个仿真环境，不提供多客户端交错共享同一服务的会话。
去掉 headless 可显示窗口；HTTP/打包在后台，等待时主线程最多约 20 Hz render，不推进物理。
关闭窗口后丢弃迟到动作；GUI 响应频率不等于策略动作频率。

启动文件可复制到 IsaacLab 目录，复制后增加 `--bridgevla-root /home/yiwei/BridgeVLA`；
直接使用仓库脚本无需设置 PYTHONPATH。`--headless --device` 等参数透传给 runtime。
需要额外任务注册模块时使用可重复的 `--task-import module`。

<a id="关键参数"></a>

### 关键参数

| 参数 | 默认值 / 约束 |
| --- | --- |
| `--isaaclab-root` | 优先于 ISAACLAB_ROOT，再回退含 isaaclab.sh 的当前目录 |
| `--scene-preset` | registry；oht-v4 加载完整 V4 场景并核对 USD，仅适用于 OSC/EEF |
| `--env-config`、`--expected-robot-usd` | 前者指定完整环境配置类，后者只校验资产路径，不替换资产 |
| `--ee-pose-source` | controller：body pose × body_offset；body：裸 link，须匹配采集参考点 |
| `--ee-link`、`--quaternion-order` | EE 默认取 OSC body_name；SDK 四元数默认 wxyz |
| `--camera-source` | auto 优先完整注册 Camera；usd 使用现有 USD/Replicator 相机；sensors 使用注册 Camera |
| `--camera-map` | 完整且互异的 checkpoint 相机名→sensor/USD 名或路径映射 |
| `--data-profile` | 自动匹配服务公布的 task 或普通/`-seg` 任务目录名；唯一且无任务限制的 profile 也自动选择；多个候选时从 model_info.raw_observation.data_profiles 显式选择 |
| `--episodes / --max-steps` | policy 默认 20 / 500；replay 默认 1 / 全部剩余条目 |
| `--max-translation / --max-rotation` | OSC 每轴物理限幅 0.05 m / 0.15 rad，限幅后除以 live scale |
| `--dry-run` | 保存启动诊断，不恢复记录初态、不检查恢复后 TCP、不执行 HTTP reset/predict 或控制步 |
| `--keep-open-on-error` | 启动检查或 episode 出错后保存记录，环境已创建时保留 GUI，不再控制；headless 不等待，错误仍以非零状态退出 |
| `--log-interval` | 10；1 逐请求打印，0 关闭周期日志，执行失败仍打印摘要 |

场景预设 oht-v4 与训练 replay/checkpoint 的 chunk V4 是两个版本概念。
注册 ID 的 OSC-v2 后缀不能证明实际 USD 版本。joint 模式选匹配的 registry 关节位置环境，
不能直接用上述 OSC 预设。body_offset 与 link_to_tcp 各只应用一次。

### 普通版与分割版 data profile

四个任务均支持普通目录 `<task>/lerobot_dataset` 和分割目录 `<task>-seg/lerobot_dataset`，例如
`disassemble_left/lerobot_dataset` 与 `disassemble_left-seg/lerobot_dataset`。
客户端优先按服务公布的 task 匹配，未声明 task 时解析这两种目录名；仅有一个且无任务限制的
profile（如 `default`、`.`、`lerobot_dataset`）也可自动选择。

同一 checkpoint 中普通版与分割版都匹配当前任务时，必须显式指定 `--data-profile`，不会任选一项。
先在 IsaacLab 服务器查询当前 checkpoint 公布的完整名称：

```bash
curl -fsS http://109.105.46.126:8004/model_info | python -c '
import json, sys
for name, spec in json.load(sys.stdin)["raw_observation"]["data_profiles"].items():
    print(name, "task=", spec.get("task"))
'
```

在上面的 client 启动命令中增加 `--data-profile disassemble_left/lerobot_dataset`（普通版），
或 `--data-profile disassemble_left-seg/lerobot_dataset`（分割版）；名称必须存在于查询结果中，
并与实际部署所需的训练数据标定一致。该参数选择 checkpoint 中保存的数据标定，不切换模型权重。

旧代码漏识别 `-seg` 后缀，可能在进入仿真前报 `Select --data-profile explicitly from /model_info`。
此兼容修复不改变数据或模型格式，无需重建 replay 或重新训练；部署时同步两端代码并重启服务。

<a id="HTTP与观测协议"></a>

## HTTP 与观测协议

新 HTTP 外层版本为 bridgevla_predict_v1；旧 /act 结构不同，不能只更改端口后混用。
原生客户端自动读取 model_info、协商动作和 chunk schema、reset，并校验返回身份。

### 接口、身份与串行会话

| 接口 | 内容 |
| --- | --- |
| GET /health | ready、busy、episode_failed、error 等轻量状态，不触发前向 |
| GET /model_info | model_id、contract_sha256、action_spec、相机、观测 schema、raw profiles、请求限额；chunk 另含 chunk_spec |
| POST /reset | 新 episode ID，清除策略/predictor 状态及 raw 夹爪历史 |
| POST /predict | 验证一次当前观测并返回一次预测 |
| GET /visualization、/visualization/latest | 诊断页面及最近完成的采样/writer 状态 |
| GET /visualization/files/{entry_id}/{filename} | 内部 ID 和文件白名单限定的 PNG/NPZ/JSON |

model_id 默认由 checkpoint 路径、大小、mtime 的摘要生成，也可手动指定；
contract_sha256 标识训练数据契约。两者都不是权重内容哈希，不能证明两个部署的权重字节相同。
跨实验核对实际 checkpoint 文件，并保存部署信息。

reset/predict 共用单请求准入，busy 返回 429，不积压控制请求；health 不等待 CUDA 前向。
step 从 0 连续递增，非负有限 measurement timestamp 严格递增，同一 episode 的 instruction 固定。
首次 predict 可自动建立新会话；之后切换必须 reset 到未使用过的 episode ID。
最近 4096 个已处理 request_id 去重，step/episode 检查不依赖该窗口。
输入/会话错误不推进 step；模型内部失败后 episode 标记失败，须使用新 ID reset。
客户端不自动重试。超时或断连不取消后台 CUDA 工作；忙状态释放后用新 episode reset，
不能重发结果不确定的旧动作。

<a id="请求"></a>

### 请求与 typed tensor

reset JSON 必须恰好包含 api_version、request_id、episode_id、model_id、contract_sha256。
api_version 固定为 bridgevla_predict_v1，模型/契约标识取自 model_info。
predict 在上述字段之外必须包含：

| 字段 | 约束 |
| --- | --- |
| step、timestamp | 连续请求步与递增测量时间 |
| instruction | 非空且 episode 内固定 |
| observation_format | bridgevla_prepared_v1 或 bridgevla_rgbd_v1 |
| observation | 下述格式的完整输入，不接受 teacher/未来标签 |
| prediction_schema | chunk 必须为 bridgevla_oht_eef_chunk_v1；单目标省略 |

tensor 封装为 `{"dtype": "...", "shape": [...], "data": "<Base64>"}`。
data 为连续 C-order 原始字节，浮点 little-endian；支持 uint8、float32、float64，
具体字段另校验 dtype/shape。JSON 数字拒绝 NaN/Inf；tensor 中无效深度/XYZ 按契约处理。
HTTP body 默认上限 32 MiB，含 JSON/Base64 开销；解码后 tensor 总量也独立限制为 32 MiB。

### raw 与 prepared

原生客户端使用 raw：observation 恰好含 task、data_profile、tensors；
profile 必须来自服务声明。每个所需相机的四项和三项机器人状态均须完整：

| raw tensor | shape / 约定 |
| --- | --- |
| `<camera>_rgb` | uint8 `[H,W,3]` |
| `<camera>_depth` | float `[H,W]`，米制 Z-depth；无效值可用 NaN，拒绝 Inf |
| `<camera>_intrinsics` | float `[3,3]`，当前源分辨率 K |
| `<camera>_pose` | float `[7]`，OpenGL sensor→env-origin-relative world，xyz+wxyz |
| `joint_positions` | float `[6]`，记录关节顺序，rad |
| `ee_link_pose` | float `[7]`，同一 world 的 xyz+wxyz；服务应用一次 profile.link_to_tcp |
| `gripper_position` | float `[1]`，实测两指关节位置之和，不是二值控制命令 |

相机名使用 checkpoint 声明，数量/尺寸不写死。RGB/depth 必须逐像素对齐；
源 H/W 须分别是 checkpoint image_size 的整数倍，不做上采样。
K 对应源图，pose 对应本次采样。服务转换 OpenGL→optical、反投影和整数采样，
不重复执行离线深度视频的量化/log 解码，也不自动推断单位或四元数顺序。
raw 所有动作模式都要求上述机器人字段；单目标 EEF 不因此额外把 joints/TCP 输入动作头。
夹爪历史按因果规则计算，仅在推理成功后提交；episode 内不能切换 format/task/profile。

prepared observation 直接是字段名→tensor 的映射，以 observation_schema 声明为准：
CHW RGB、训练契约的 metric depth、组织化 world XYZ、相机 K/T 和对应状态。
prepared 外参为 T_world_from_optical，不能把 raw OpenGL pose 直接当作它。
无像素对应的稀疏点云不能替代组织化 XYZ。通过 shape 校验不证明真实标定正确。

<a id="响应"></a>

### 响应与错误

两类响应都含 API/模型/契约标识、request_id、episode_id、step、timestamp、action_spec、
timing_ms={decode,prepare,policy,total}。返回的身份必须与请求一致。

| checkpoint | 预测字段 |
| --- | --- |
| 单目标 | action，维度/单位/参考点由 action_spec 声明 |
| V4 EEF chunk | actions[K,8]、独立 key_action[8]、time_offsets[K]、chunk_spec；没有 action |

chunk_spec 声明 schema、horizon K、execute_horizon H、action_dt、
frame=world、quaternion_order=xyzw、reference_point=tcp、gripper=1=open,0=close、action_size=8。
time_offsets 固定为 dt,2dt,…,Kdt。执行的是 actions 前 H 项，key_action 不是轨迹替代物。

| HTTP 状态 | 含义 |
| --- | --- |
| 400 / 422 | 无效 JSON / 字段、观测或 prediction schema 不匹配 |
| 409 / 429 / 503 | 模型/会话冲突 / busy / 未就绪 |
| 413 / 415 / 500 | 请求体过大 / 非 JSON Content-Type / 推理内部失败 |

<a id="单目标与chunk执行"></a>
<a id="动作定义"></a>

## 单目标与 chunk 执行

world 坐标统一按 OHT 训练契约解释，原生采集使用 env-origin-relative world。
EEF 为米、旋转为弧度或 xyzw 四元数，夹爪为 0 关 / 1 开。
HTTP 机器人/相机 pose 的 wxyz、SDK 的 quaternion-order 与动作 xyzw 是不同层次。
OSC 将 TCP 目标还原到控制点，再变换到实时 robot-root，构造 7 维残差输入；
不会把 EEF8 直接当作 OSC7。

### V4 chunk

当前 chunk 仅支持 baseline + absolute EEF。每项是 [world_xyz, quat_xyzw, open01]，
预测必须有限且四元数为单位旋转。客户端按 chunk_spec 协商，不执行 GT/教师路径。
action_dt 必须是 env.step_dt 的正整数倍；首段从本次实测 TCP 开始，
相邻 tick 间线性插值位置、SLERP 姿态；夹爪到该项 tick 才切换，中间保持上一命令。
执行前 H 项后丢弃后缀、重新观察，不等待关键点到位或夹爪稳定。
每个控制步检查跟踪误差；超限、硬限位异常或任务/环境停止后不继续后缀。

周期比按现有浮点容差与最近的正整数比较。例如记录的 `action_dt=0.01666666666666572`
和仿真的 `env.step_dt=1/60` 都对应一个控制步；比值略小于 1 时也接受，不修改 checkpoint
或重建 replay。真正短于控制周期或非整数倍且超出容差的配置仍在启动时拒绝，错误会列出两个周期及比值。

| 参数 | 默认值 / 作用 |
| --- | --- |
| --chunk-execute-horizon | 默认用服务的 H；覆盖值须在 1..K 内 |
| --chunk-max-tracking-translation | 0.10 m，跟踪误差上限 |
| --chunk-max-tracking-rotation | 0.5 rad，姿态误差上限 |
| --execution-mode | auto 自动选 chunk；chunk checkpoint 拒绝显式 waypoint/streaming |

### 单目标与 waypoint

| 模式 | 输出与请求参考 |
| --- | --- |
| eef | absolute TCP xyz + xyzw + grip，8 维 |
| relative_eef | delta_xyz + rotvec + grip，7 维；相对于本次实测 TCP |
| joint | 六个绝对关节角 + grip，7 维，rad |
| relative_joint | 六个关节差 + grip，7 维；相对于本次实测 joints |

相对 EEF 默认 world 平移、body 旋转；body 旋转为 log(R_current^T R_target)。
world/base 采用相应坐标系的旋转误差，base 必须使用 checkpoint 声明的固定 world_from_base，
不猜基座朝向。四元数不直接相减，相对量不是每秒速度。
相对动作只按对应请求的当前测量解码一次，后续控制步跟踪同一个绝对目标。

waypoint 先保持请求时实测二值夹爪状态，跟踪固定目标；机械臂连续到位后切换预测夹爪，
并保持该机械臂目标，待夹爪稳定且机械臂仍到位才请求下一目标。
关爪可在接触后的部分开口处完成，但须有已闭合或实际闭合运动证据；开爪须达到开口阈值。
稳定不证明抓取成功。夹爪阶段短暂漂移时继续追踪并保留夹爪命令，不自动重新开爪；
持续停滞或预算耗尽才失败，评分成功或环境终止可提前结束。

| waypoint 参数 | 默认值 |
| --- | --- |
| --waypoint-max-steps | 600，approach + gripper 两阶段总物理步数 |
| --waypoint-position-tolerance / --waypoint-rotation-tolerance | 0.005 m 欧氏位置误差 / 0.05 rad SO(3) 夹角 |
| --joint-target-tolerance | 0.01 rad，六关节最大绝对误差 |
| --waypoint-arrival-steps | 3，连续控制步到位 |
| --gripper-min-steps | 10，夹爪阶段至少执行步数，即使开闭目标未变 |
| --gripper-stable-steps | 5，保留 6 个开口读数，归一化极差 ≤ 0.002 |
| --waypoint-stall-steps | 100，无足够机械臂/夹爪进展时停止 |

max-substeps、joint-hold-steps 用于 streaming，waypoint 只调用单步接口；
自定义 joint executor 须暴露 step(absolute_target)。到位阈值与 OSC 每轴限幅不同。
直接 OSC/关节跟踪没有碰撞规划器；延长预算不能补回稀疏标签省略的抬升、绕障与对准路径。
H-VLA 参考的 raw_actions 动作块与显示用 raw_key_actions 时间尺度不同，不能据其 EEF 名称直接替换。

### 关节顺序与限位

训练向量顺序固定为 observation.state[:6]；joint_names 声明该顺序，不重排训练数组。
原生 arm_ids/JointPositionExecutor 按名称映射到实际关节 ID 与 term 顺序；省略名称时要求顺序已一致。
joint/relative_joint 需 JointPositionAction 环境，不能送入 OSC。
模型 joint_limits 会限制预测，执行器仍独立检查仿真 soft_joint_pos_limits，越界停止；
初态恢复检查硬限位，两者不同。joint_wrap 只影响相对标签/预测的最短角差，
不会把执行误差/限位检查改为周期等价判断。详细训练设置见[训练说明](training.md)。

### 旧 /act 兼容入口

旧服务只提供 /health 与 /act，接收 prepared 在线字段，不接受新 raw envelope；EEF schema 为 bridgevla_oht_absolute_tcp_v1，
其余三模式为 bridgevla_oht_action_v2。旧请求包含 schema、episode、step、timestamp、goal、
contract_sha256、observation；向量模式还必须有匹配的 action_spec。响应回传同一身份/契约及 action，
EEF 另声明 world/xyzw/tcp，向量模式回传 action_spec。GET /health 提供 schema/action_spec。
旧服务以新 episode 的 step 0 自动 reset，没有新接口的 /reset；runtime.transport.Client 只连接 /act。

```bash
python -m finetune.OHT.server \
  --checkpoint /path/to/single-target-joint-run/model_last.pth --port 8010
```

若使用旧 adapter 的 eval closed-loop，在另一终端从 BridgeVLA 根目录执行：

```bash
python -m finetune.OHT.eval closed-loop \
  --config finetune/OHT/configs/joint.yaml --server http://127.0.0.1:8010 \
  --replay /path/to/replay-v3 --environment my_isaac_oht:create \
  --cases /path/to/eval-cases.jsonl --executor-id verified-joint-v1 \
  --output /path/to/new-evaluation.json
```

my_isaac_oht:create 是需自行实现的 factory(contract)，提供 reset(case)、step(action)；
JSONL case 含唯一 id、task、seed。config 必须对应 checkpoint 动作/坐标，不是改 YAML 就能换动作头。
原生新客户端不使用此入口；V4 chunk 也不支持该旧闭环接口。
此 adapter 按完整 cases 列表循环，失败/超时仍计入分母，和下文原生 CLI 的 attempted 口径不同。

<a id="记录初态"></a>
<a id="相同记录初态运行策略"></a>

## 记录初态

在[原生 client 命令](#独立-bridgevla-客户端)中增加以下参数，保留 execution-mode auto，
并为本次配对使用新的输出目录：

```text
--initial-state-root /path/to/original-oht-data
--initial-state-data-config /path/to/dataset-used-for-this-checkpoint.yaml
--initial-state-episode 0 --initial-state-frame 0
```

记录初态与 GT 均需 pyarrow/PyYAML 及可读的 raw metadata，不解码原始视频。
initial-state-root 自动选择 trajectory，不能同时选择 env-reset/task/collection。
episode/frame 默认 0，frame 可为最后一帧。每次先 env.reset，再恢复选定帧六关节/两指并清零其速度，
刷新 FK、检查 TCP，然后 reset HTTP session，使用实时观测和模型动作。
不逐帧 teleport，也不拿记录未来动作驱动策略；恢复初态不要求原数据周期等于在线控制周期。
优先选夹爪稳定开/闭的帧；新 HTTP session 不继承记录中过渡开口的迟滞历史。

<a id="标定来源与校验边界"></a>

### 标定来源与校验边界

| 来源 | 行为 |
| --- | --- |
| initial-state-data-config | 完整数据 YAML，顶层含 cameras/gripper/link_to_tcp；不是训练 YAML 或 contract.json，仍按 raw dataset 解析 metadata |
| 两种标定参数均省略 | 读当前仓库 configs/dataset.yaml，不会从 checkpoint 恢复原 YAML |
| initial-state-contract | 替代 data-config 的 V3 replay 路径，两者互斥；需完整匹配的 contract.json、complete.json、samples.jsonl，不读取缓存图像，并要求 SHA 与服务契约一致 |
| V4 checkpoint | 当前初态 loader 拒绝 V4 contract；使用已核对的数据 YAML，未提供 contract 时不要求本地 SHA 与 V4 服务相等 |

自动比对 task/profile、link_to_tcp、gripper.open/close、scene_bounds、image_size；
矩阵/边界/尺寸绝对容差 1e-6，端点 1e-8，均 rtol=0。
相机 K/T、原始 EE 四元数约定、夹爪迟滞、关键点配置不在初态比对字段中，须独立核对。
数据 YAML 的 ee_quaternion_order 解释 Parquet；CLI quaternion-order 解释安装的 SDK，不能互相替代。
起始 TCP 默认阈值为 0.005 m / 0.05 rad，由 initial-state-position-tolerance/rotation-tolerance 调整。
initial-state-joint-names 和 initial-state-gripper-joint-names 可明确记录顺序。

相同 contract SHA 不证明当前 raw 文件、完整场景或模型权重相同；run.initial_state 记录本次文件/标定哈希。
dry-run 不恢复状态或运行恢复后的 TCP 门槛；起始对齐失败会写 episode 错误并在第一次预测前停止。
数据/标定预检可能在创建输出目录前失败，应保留终端错误。
对象默认是环境 reset 状态；可用 initial-state-object-map 按[明确映射](#可选物体恢复)恢复刚体，
但不能重建装配约束、接触历史和抓取关系。

<a id="GT回放"></a>
<a id="oht-gt-轨迹回放"></a>

## GT 回放

GT 不连接 VLA server、不加载模型，使用原始 Parquet 实测 TCP，不执行原始 action 列。
replay-root 应含 `<task>/lerobot_dataset/data/chunk-*/episode_*.parquet`，不是 replay-v3/v4 缓存。
使用 task + replay-episode 选择唯一轨迹；当前固定绝对 EEF，不解码参考 HDF 10D 动作。

```bash
cd /home/yiwei/IsaacLab
python /home/yiwei/BridgeVLA/tools/oht_bridgevla_client.py \
  --mode replay --isaaclab-root "$PWD" \
  --replay-root /home/yiwei/BridgeVLA/oht_curobo_pd_v423 \
  --replay-data-config /path/to/dataset-used-for-this-checkpoint.yaml \
  --task disassemble_left --replay-episode 0 --replay-start-frame 0 \
  --scene-preset oht-v4 --ee-pose-source controller --seed 42 \
  --output logs/bridgevla-gt-frame-01 --headless --device cuda:0
```

<a id="初始化与时间"></a>

### frame 与 keypoints

| 设置 | 行为 |
| --- | --- |
| replay-targets frame（默认） | i→i+1 的实测 TCP + 滤波夹爪，streaming 一次控制步；所选区间记录时间差须等于 env.step_dt，max-substeps 必须为 1 |
| replay-targets keypoints | 提取 gripper/geometric/bridgevla 未来终点，以 waypoint 执行；源时间戳仅作来源，不约束跟踪时长 |
| replay-episode / replay-start-frame | 均默认 0；回放起点必须至少有一帧后续记录 |
| episodes / max-steps | 默认回放一次直到结束；max-steps 限制 frame 转换数或 keypoint 数 |
| replay-contract | 可替代 replay-data-config，只接受完整 V3 replay contract；不支持 V4 contract |
| replay-cameras | 默认关闭；开启后绑定/刷新数据配置全部实时相机，不解码原始视频 |
| initialization | 默认 trajectory，也可 env-reset 或已配置 task；起始 TCP 均须对齐 |

<a id="gt-关键点对照"></a>

GT keypoint 对照在上述命令增加 `--replay-targets keypoints` 并更换 output。
单目标 checkpoint 可将 data-config 换为 `--replay-contract /path/to/replay-v3`，
以使用实际训练关键点配置；GT 端不连接模型，不会检查其与 checkpoint SHA 一致。
不插入原始密集路径或合成末尾目标；这不是 V4 chunk 的定时对照。
逐帧 GT 正常但 keypoints 失败时检查终点执行/中间路径；keypoints 正常再比较同初态 policy 观测与预测。

默认每次恢复所选帧机械臂/夹爪，写仿真后核对关节和实际 TCP。SDK 有硬限位时据其检查，
不将 soft limit 当物理边界；缺少硬限位信息时诊断为 null。
replay-joint-names/replay-gripper-joint-names 可指定记录顺序；
replay-initial-position-tolerance/rotation-tolerance 默认 0.005 m / 0.05 rad。
replay-data-config 默认读仓库当前 dataset.yaml 并解析 metadata。replay-* 只用于 mode replay，
policy 显式传入会报错。完整动态状态未恢复，即使 TCP 相同也不能承诺逐物理状态复现。

<a id="可选物体恢复"></a>

### 可选物体恢复

replay-object-map / initial-state-object-map 使用显式 JSON：

```json
{"quaternion_order": "wxyz", "assets": {"实际场景刚体键": 0}}
```

值是 observation.objects_pos/objects_quat 的零起始索引，键须是实际 env.scene 刚体；
按记录 quaternion_order 转换 SDK 顺序，位置按 env-origin-relative world 加当前 origin。
仅起始写 root pose、清零速度，不恢复约束/接触；不能猜物体索引或参考点。
机器人、其他 articulation 或重复别名映射会在状态写入前拒绝。映射要求 trajectory 初始化，
复杂接触初态须由项目自己的初始化方案保证。

<a id="排错与验收"></a>

## 排错与验收

<a id="时间预算与停止结果"></a>
<a id="任务初始化与成功率评估"></a>

### 时间预算与停止结果

HTTP 等待只 render，不推进物理；measurement timestamp = common_step_counter × step_dt。
waypoint 一次请求最多 waypoint_max_steps × step_dt 仿真秒；
chunk 正常执行 H × action_dt，另有采集/网络/推理墙钟开销。
原生 CLI 的 timeout 默认 120 s，是 HTTP 网络超时参数，不是整个 rollout 墙钟期限。
比较时同时记录请求数、commands/物理步数、仿真时长和墙钟耗时；相同 max-steps 不等于同物理预算。

四任务为 assemble_left/right、disassemble_left/right。task 选 instruction/profile，不保证场景初态。
task-config 使用[实际任务 YAML](../../finetune/OHT/configs/isaaclab_tasks.example.yaml)，不是模型/环境配置；
initialization task 要求有效 reset_assets 或 initializer_factory，并与 trajectory 恢复互斥。
score task 要求真实 success_term/evaluator_factory，或用 success-term 指定实际环境成功条件。
未配置评分时 success=null；next.success、terminated/truncated 不自动成为本次仿真成功。

| stop_reason | 含义 |
| --- | --- |
| success | 有效评分器报告成功，可在目标未走完前提前结束 |
| environment_done | 环境终止/截断，不自动等于成功 |
| trajectory_end / max_steps | GT 耗尽 / 请求或条目预算耗尽，不等于任务成功 |
| waypoint_stalled / waypoint_timeout / waypoint_error / waypoint_application_stopped | 保存已收集的执行记录并结束运行，不再请求目标或尝试后续 episode |
| chunk_tracking_error / chunk_joint_limit、error（含 HTTP 异常） | 停止控制并记录错误；迟到响应不执行、不自动重试 |

原生 CLI 的 summary.success_rate = successes/attempted；执行错误后剩余计划记为 unattempted，不在该分母中。
必须核对 planned/attempted/unattempted 和固定 case 覆盖；不能只比较两份 summary 百分比。
配对需固定 raw 文件/帧、资产、对象恢复、seed、控制周期、阈值和评分，并检查实际恢复结果。
同一 raw episode 重复 N 次不是 N 个独立 case；只恢复机器人也不保证物体或接触初态相同。

<a id="日志与排错"></a>

### 日志与排错

| 文件 / 现象 | 检查 |
| --- | --- |
| server info.json / requests.jsonl | checkpoint 路径、部署/契约身份、请求状态、轮转摘要和分段耗时 |
| client startup.json | 启动阶段与 starting/ready/failed 状态；客户端记录 profile、执行模式、chunk_spec 和环境控制周期，捕获的启动错误含 error/traceback，在关闭 Isaac Sim 前保存 |
| client run.json / control_diagnostics.json | 实际 USD、scene/config 来源、TCP 变换、控制尺度、顺序、夹爪物理端点及是否关闭 time_out |
| episode_*.json | action、实际 commands、measured_before/after、errors_before/after、stop_reason |
| execution | waypoint 阶段/到位/夹爪；chunk consumed_actions/requested_actions、control_substeps、discarded_suffix；streaming 无该字段 |
| joint_diagnostics | 实际关节、硬/软限位来源和余量；缺失硬限位为 null |
| camera_observation | 本次源分辨率、K、pose 和外参来源；USD 优先有效渲染 view，回退 USD/物理 body × 相机局部变换 |
| client_timing_ms | capture/pack/serialize/http/predict/request/execute/total；存在包含关系，不能相加 |
| server timing_ms | decode/prepare/policy/total；policy 含结果回传的墙钟耗时，不是 CUDA kernel 时间 |
| waypoint 停滞 / chunk 超差 | 区分目标错误、限位、接触和跟踪问题；不要仅翻转夹爪或加长训练/预算 |
| 409 / 429 / 503 | 身份/会话不符 / busy / 未就绪；结果不确定时待空闲后新 episode reset |

窗口在 `native env=... cameras=...` 后退出且没有预测记录时，先读 `startup.json`。
这行只代表原生环境与相机已建立，后续还有环境 reset、overlay、控制 bridge、chunk 时序和控制诊断检查。
捕获的 Python 启动异常会先打印并保存，再关闭仿真；环境已返回客户端时，可增加 `--keep-open-on-error`
保留 GUI。若进程被终止或发生原生崩溃，文件可能停在 starting，需结合退出码及 Kit 日志末尾判断，
不能仅根据材质/相机 warning 或历史 minidump 提示认定本次崩溃原因。

GT replay 默认不绑定相机，所以不带 `--replay-cameras` 的 replay 正常，不能验证在线 RGB-D 路径。
可先用在线 client 命令配合新输出目录和 `--dry-run` 检查启动（不预测、不执行控制，TCP overlay 也关闭）；
通过后去掉 dry-run、保留 `--no-visualize` 测试真实策略，区分启动检查、overlay 与预测/执行阶段。
`--no-visualize` 只关闭仿真中的 TCP overlay，保留 GUI 和模型所需的 RGB-D 相机。

夹爪正开/负关的命令验证与物理端点方向验证分开记录；相反标定会在 reset/predict/env.step 前停止，
不自动反号。ee-pose-source 必须按 checkpoint 采集定义选择，不能从历史字段名 ee_link_pose 猜参考点。
注册 Camera 开启实时位姿更新；USD 外参检查记录实际来源，不硬编码 90° 补偿。
历史日志中的固定旋转、长期限幅和大目标跳变不能单独证明某个配置是根因。

<a id="在线可视化"></a>

### 在线可视化与验收

浏览器打开 server 的 /visualization：同次请求的交互点云、原始 RGB-D/目标投影、
coarse/refine 热图和可用对象 prior；原生 raw 保留传感器分辨率，prepared 标记为已处理观测。
relative EEF 解码绝对目标；joint 只画实测 TCP/辅助热图，不伪造目标 FK；chunk 默认标首个执行目标。
深度显示按分位拉伸，数值以 NPZ 米制数组为准；点云/目标标记不是碰撞规划路径。
默认每 episode 第 0、10、20…次采样，visualize-every=0 关闭；visualize-keep 默认 100。
CPU writer 最多两个任务，满时跳过；PNG/NPZ/JSON 完成后发布 latest，可视化错误不重跑模型。
每次启动独立 visualizations/run-* 目录；原始 RGB-D 会增加磁盘与拷贝开销。
GUI 默认显示 TCP/目标轨迹，no-visualize 关闭，visualize-history 默认 200；
采集前清除标记，快照完成后恢复，防止其进入模型图像。缺 debug-draw 时降级继续。
latest 可能早于当前 step，应按 episode/step 对照，不把旧诊断当作刚执行的动作。

<a id="检查与-replay-烟测"></a>

在 BridgeVLA 根目录使用匹配 replay 做 HTTP 烟测，无其他客户端占用该服务：

```bash
python tools/smoke_oht_vla_server.py \
  --server http://109.105.46.126:8004 --replay /path/to/replay-v4 \
  --split val --index 0
```

烟测只发送缓存观测/返回预测，不执行动作，不等于 raw 相机链路或 GT 回放验收。
先用 dry-run 排查完整场景/TCP/夹爪，再核对实时 RGB-D 与训练点云；
随后按相同初态对照 dense GT、单目标 GT keypoints 或 chunk 时序执行，最后检验 policy 与任务评分。
CUDA 前向、传感器/接触和闭环成功率仍须真实服务器验证；CPU 替身不提供这些结论。

实现入口：[vla_server.py](../../finetune/OHT/vla_server.py)、
[isaaclab_client.py](../../finetune/OHT/isaaclab_client.py)、[runtime](../../finetune/OHT/runtime)。
协议与执行回归集中在 test_oht_vla_server、test_oht_isaaclab_http、test_oht_chunk_pipeline、
test_oht_waypoint_executor 和 test_oht_policy_initialization；本页不将历史测试数量当作当前验收成绩。
