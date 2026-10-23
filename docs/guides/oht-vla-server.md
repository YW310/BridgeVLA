# OHT VLA server 与 IsaacLab 客户端

先启动模型服务器，再在 IsaacLab 仿真服务器启动客户端。两台机器均需同步 BridgeVLA 代码并激活各自环境；更新服务代码后需重启服务。以下路径按实际安装位置替换。

## 安装与启动

在 **VLA 模型服务器**的 BridgeVLA 根目录执行：

```bash
python -m pip install -r finetune/OHT/requirements-server.txt

GPU_ID=1 bash finetune/OHT/vla_server.sh \
  --model-path /home/yiwei/project/BridgeVLA/finetune/OHT/runs/baseline-eef-8gpu-s0/model_last.pth \
  --device cuda:0 \
  --host 0.0.0.0 --port 8004 \
  --visualize-every 10 \
  --output-dir logs/oht_vla_server_8004
```

`--model-path` 可为 checkpoint 文件或含 `model_last.pth` 的训练目录。`GPU_ID=1` 选择物理 GPU 1，进程内仍使用 `cuda:0`。`serve.sh` 是同一服务的兼容启动入口。

| 可选配置 | 用法 |
| --- | --- |
| 环境变量启动 | `OHT_MODEL_PATH=/path/to/model_last.pth OHT_HOST=0.0.0.0 OHT_PORT=8004 bash finetune/OHT/vla_server.sh` |
| 预训练资源 | 按加载环境增加 `--pretrain-path /path/to/BridgeVLA-pretrain`；PaliGemma/processor 资源需可访问或已缓存 |
| 离线加载 | 资源齐备后设置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` |
| 外部物体预测 | `--predictor` 与 `--predictor-provenance` 必须匹配 checkpoint 的训练记录 |
| 预热 | 增加 `--warmup-replay /data/oht/replay-source-v3 --warmup-split val --warmup-index 0`；replay 契约须匹配 |
| 指定解释器 | `PYTHON_BIN=/path/to/env/bin/python` |

命令行参数优先于对应环境变量。服务使用单进程、单模型，不用 `torchrun`；多个模型分别使用不同端口和日志目录。

V4 chunk checkpoint 使用同一启动入口，训练与离线评估见 [V4 说明](../../finetune/OHT/README.md#v4关键目标--连续-chunk-训练与评估)。`/model_info.chunk_spec` 声明 K、dt 和建议 H；新版客户端会确认 chunk schema，旧单动作客户端不能将其当作关键目标执行。

## 分机部署：VLA server 与 IsaacLab client

在 **IsaacLab 仿真服务器**检查连接：

```bash
curl http://109.105.46.126:8004/health
curl http://109.105.46.126:8004/model_info
```

服务端绑定 `0.0.0.0`，客户端填写服务端实际 IP。两台机器需网络互通并允许访问 TCP `8004`；`/health` 应返回 `ready=true`，`/model_info.observation_formats` 应包含 `bridgevla_rgbd_v1`。

## IsaacLab 实时 RGB-D 客户端

### 独立 BridgeVLA 客户端

启动脚本：[tools/oht_bridgevla_client.py](../../tools/oht_bridgevla_client.py)。主逻辑：[finetune/OHT/isaaclab_client.py](../../finetune/OHT/isaaclab_client.py)。

在仿真服务器激活能运行 OHT 的 IsaacLab 环境，在其根目录执行以下命令。客户端启动仿真、连接 VLA 服务并执行策略；示例运行 1 个 episode，最多请求 20 个未来关键点。

```bash
cd /home/yiwei/IsaacLab
python /home/yiwei/BridgeVLA/tools/oht_bridgevla_client.py \
  --isaaclab-root "$PWD" \
  --server http://109.105.46.126:8004 \
  --task disassemble_left \
  --env-id Isaac-Deploy-Reach-UR16e-DualEE-AllScene-OSC-v2 \
  --scene-preset oht-v4 --camera-source usd \
  --ee-pose-source controller \
  --execution-mode auto \
  --episodes 1 --max-steps 20 \
  --output logs/bridgevla-v4-run-01 \
  --headless --device cuda:0
```

**每次运行使用尚不存在的输出目录。** 默认 `--initialization env-reset --score none`，因此 `success=null`；这不是任务成功率评估。

policy 默认按 waypoint 执行：固定本次解码的绝对目标，先保持当前夹爪状态并跟踪到位，连续确认后才开闭夹爪；等待夹爪稳定后再请求下一目标。每目标最多 600 个物理控制步，停滞/超时会记录原因并停止。逐帧 GT replay 仍按原始时间线执行。详细参数、同初态策略验证和 GT 关键点对照见[关键点执行说明](oht-waypoint-execution.md)。

去掉 `--headless` 即可显示窗口。客户端默认在后台处理 HTTP reset/predict（包括观测打包），等待期间由主线程以最多约 20 Hz 调用 [IsaacLab render 接口](https://github.com/isaac-sim/IsaacLab/blob/v2.2.1/source/isaaclab/isaaclab/sim/simulation_context.py)刷新窗口；headless 下不做额外等待渲染。等待预测时不推进物理，收到预测后才开始跟踪目标；关闭窗口后丢弃未执行的预测。窗口响应与策略动作频率是两项指标，动作频率仍受采集、传输和推理耗时影响。

### 在线可视化

正常启动 server 和 client 后，浏览器打开 **[http://109.105.46.126:8004/visualization](http://109.105.46.126:8004/visualization)**。页面每秒检查最新采样，包含以下内容，均来自同一次请求，不使用 GT：

- **交互 3D**：多相机 RGB-D 重建的彩色点云、相机位置/视锥、当前 TCP（绿色）、实际动作 waypoint（红色）和热图 waypoint（青色）。拖动旋转，右键或 Shift 拖动平移，滚轮缩放；可切换相机点云、点大小、重置视角或暂停刷新。更新数据时保留当前观察角度。
- **原始 sensor 图像**：各相机传入的原分辨率 RGB 和米制深度显示图，叠加相同 waypoint 的透视投影。超出视野、在相机后方、被深度遮挡或深度无效时显示对应状态，不将点硬裁到边缘。原图和数值深度可下载 NPZ。
- **模型三视图**：原有 coarse/refine RGB、预测 heatmap、动作/热图 TCP 投影和可用的 Target/Reference prior。

原生 IsaacLab 客户端使用 raw RGB-D 协议，sensor 面板保留传感器分辨率，不使用模型缩采样图冒充原图；prepared 协议只能显示已处理的观测，页面会标记 `prepared_observation`。3D 使用 env-origin-relative world 米制坐标，每份快照最多 24,000 点，按相机均衡采样；它用于观察本次观测和预测，不表示碰撞规划路径。relative EEF 先还原绝对目标；joint 模式仅有实测 TCP 和辅助热图点，不伪造关节目标的 FK。

3D 的 waypoint 标记置前显示，便于从点云中定位；是否被遮挡以原始相机面板的投影状态为准。深度颜色按每相机 2–98% 分位拉伸，数值深度以 NPZ 中的米制数组为准。

server 默认 `--visualize-every 10`，每个 episode 的第 0、10、20…步采集；改为 `1` 可逐步采集，`0` 关闭。导出复用本次模型前向，CPU 拼图/写盘异步进行，队列满时跳过可视化采样。默认 `--visualize-keep 100`，每次 server 启动在 `<output-dir>/visualizations/run-<时间>-<ID>/` 下保留最近 100 份快照，每份含模型 PNG/NPZ、`sensors.png`、`sensors.npz`、`scene.json` 和 `prediction.json`。原始 RGB-D 增加磁盘用量，采集仍有 CPU 拷贝和 GPU→CPU 传输开销，可按耗时调大间隔或减少保留份数。网页和 3D 查看器无需外网资源。

带窗口运行 client 时，TCP 标记默认开启：绿色为实测 TCP 和轨迹，红色为本次解码后的绝对 EEF 目标及目标轨迹，坐标轴表示姿态。使用 `--no-visualize` 关闭，`--visualize-history 200` 控制轨迹长度；headless 自动跳过。相对 EEF 只解码一次，joint 模式仅绘制实测 TCP，不伪造目标 FK。标记在采集 RGB-D 前清除、快照完成后恢复；`run.json.visualization` 记录显示后端与可用状态。若 SDK 缺少 debug-draw，客户端提示后继续执行，浏览器热图仍可使用。

HTTP `/predict` 的采样响应包含 `visualization` 状态/文件链接，不直接携带图片。`GET /visualization/latest` 提供最近完成的诊断及写盘状态；浏览器显示的 step 可能早于当前控制 step，需按 episode/step 对照日志。服务代码更新后需重启 server，新客户端也需同步代码。

### 关键参数

| 参数 | 说明 |
| --- | --- |
| `--isaaclab-root` | 含 `isaaclab.sh` 的实际工程目录。优先于 `ISAACLAB_ROOT`，再回退到含该文件的当前目录 |
| `--env-id` | 默认 `Isaac-Deploy-Reach-UR16e-DualEE-AllScene-OSC-v2`；自有任务按实际注册 ID 设置 |
| `--scene-preset oht-v4` | 加载完整 V4 配置，校验配置来源及 `<ISAACLAB_ROOT>/oht/assets_v4/ALL/Collected_all/all.usd`，不回退 V2；仅适用于 OSC/EEF |
| `--scene-preset registry` | 默认值；使用 `--env-id` 的注册配置，或用 `--env-config module:ConfigClass` 显式指定。`OSC-v2` 的 ID 后缀不能证明实际资产版本 |
| `--expected-robot-usd` | 校验完整配置使用的 USD 路径，不替换资产 |
| `--ee-pose-source controller` | 原生 CLI 默认值，测量 body pose × body_offset，适用于训练记录已含控制器偏移的情况 |
| `--ee-pose-source body` | 测量裸 body；采集定义为裸 link，或 profile 已包含完整 body→TCP 偏移时使用，避免重复偏移 |
| `--ee-link` | 默认 OSC 的 body_name；关节位置环境需按实际配置指定 |
| `--quaternion-order` | 安装的 IsaacLab 状态/配置四元数顺序，默认 `wxyz` |
| `--camera-source` | `usd` 使用现有 USD 相机；`sensors` 使用注册 Camera；`auto` 优先完整注册相机。必须提供 checkpoint 所需全部视角 |
| `--camera-map file.json` | checkpoint 相机名→sensor 名/USD prim 名或精确路径的完整映射 |
| `--data-profile` | 任务无法唯一匹配时，从 `/model_info.raw_observation.data_profiles` 显式选择 |
| `--execution-mode` | 默认 auto：单关键点采用 waypoint，V4 chunk 采用 chunk，逐帧 GT 采用 streaming |
| `--chunk-execute-horizon` | V4 可覆盖本次执行 H，必须在 1..K 内；其余后缀丢弃后重新观察 |
| `--chunk-max-tracking-translation/rotation` | V4 跟踪误差上限，默认 0.10 m / 0.5 rad；超限停止并记录 |
| `--waypoint-max-steps` | 默认 600，每个关键点允许的物理控制步上限 |
| `--max-substeps` | 仅用于 streaming，默认 1；dense GT 必须为 1，waypoint 使用独立的 max-steps 上限 |
| `--max-translation / --max-rotation` | OSC 每轴物理限幅，默认 0.05 m / 0.15 rad，限幅后再除以控制器尺度 |
| `--dry-run` | 可选：只保存启动诊断，不执行策略动作 |
| `--keep-open-on-error` | 有窗口运行时，episode 失败后保存日志、停止控制并保留画面；关窗或 Ctrl+C 后退出，仍返回错误码 1。headless 不等待 |
| `--log-interval` | 默认 10；1 为每步打印，0 关闭周期日志；waypoint 失败始终打印目标、阶段、物理步数、夹爪状态及最终误差 |
| `--visualize / --no-visualize` | 默认显示 GUI 中的 TCP、EEF 目标和轨迹；与服务端 heatmap 开关独立 |
| `--visualize-history` | 默认 200，GUI 中每条轨迹保留的点数 |
| `--task-import module` | 额外导入自有任务注册模块，可重复 |

`ee-pose-source` 必须匹配当前 checkpoint 的采集定义，不能只根据 `eef/relative_eef` 选择。joint/relative_joint 需要 `JointPositionAction` 环境，使用 `registry` 选择匹配配置；不能使用上面的 V4 OSC 预设。

不清楚 IsaacLab 根目录时，在仿真服务器查找：

```bash
find "$HOME" /data -maxdepth 6 -type f -name isaaclab.sh 2>/dev/null
```

若需要启动文件放在 IsaacLab 根目录：

```bash
cp /home/yiwei/BridgeVLA/tools/oht_bridgevla_client.py ./oht-bridgevla-client.py
```

随后将上述命令的脚本路径改为 `./oht-bridgevla-client.py`，增加 `--bridgevla-root /home/yiwei/BridgeVLA`，其余参数保持一致。直接使用仓库内脚本时无需设置 `PYTHONPATH`。

### 任务初始化与成功率评估

支持 `assemble_left`、`assemble_right`、`disassemble_left`、`disassemble_right`。`--task` 选择 instruction/profile，场景布局和初始状态由环境或 task initializer 决定。

这是可选的初始化/评分配置，正常运行使用前面的命令即可，默认 `--initialization env-reset --score none`。若希望 policy 与已有 GT replay 使用同一记录初态，在正常命令中增加 `--initial-state-root /home/yiwei/BridgeVLA/oht_curobo_pd_v423 --initial-state-episode 0 --initial-state-frame 0`；自动采用 trajectory 初始化，后续仍用实时传感器和模型动作。标定校验、显式物体恢复与完整命令见[同初态对照](oht-waypoint-execution.md#相同记录初态运行策略)。

`--task-config` 指向用户实际创建的任务 YAML，用于配置重置资产、初始化器和成功条件；它不是 IsaacLab 根目录、环境配置或模型配置。仓库提供[任务配置模板](../../finetune/OHT/configs/isaaclab_tasks.example.yaml)，需按仿真环境填写后才能用于评估。

| 评估选项 | 使用条件 |
| --- | --- |
| `--task-config` | 指向已按实际环境配置的 YAML 文件；不提供时使用默认任务设置 |
| `--initialization task` | YAML 中至少配置有效的 `reset_assets` 或 `initializer_factory` |
| `--score task` | 配置实际注册的 `success_term` 或 `evaluator_factory`；也可用 `--success-term` 指定实际环境成功条件 |

模板中的成功条件名称需要仿真环境真实提供，不能仅复制模板就得到成功率。普通 `env.reset()` 不等同于参考脚本的训练帧状态恢复，环境 terminated/truncated 也不自动等于成功。

## OHT GT 轨迹回放

在 IsaacLab 服务器运行，不需要 VLA server 或 checkpoint。`--replay-root` 指向原始数据根目录，其中应有 `<task>/lerobot_dataset/data/chunk-*/episode_*.parquet`，不是 `replay-source-v3` 缓存目录。

```bash
cd /home/yiwei/IsaacLab
python /home/yiwei/BridgeVLA/tools/oht_bridgevla_client.py \
  --mode replay \
  --isaaclab-root "$PWD" \
  --replay-root /home/yiwei/BridgeVLA/oht_curobo_pd_v423 \
  --task disassemble_left --replay-episode 0 \
  --scene-preset oht-v4 --ee-pose-source controller \
  --output logs/bridgevla-gt-replay-01 \
  --headless --device cuda:0
```

默认恢复所选轨迹起始帧的机械臂/夹爪，逐帧将下一帧实测 TCP 作为绝对 EEF 目标，沿用在线执行器；不执行原始 `action` 列。默认回放一次直到轨迹结束，可用 `--replay-start-frame N` 选择起点、`--max-steps N` 限制长度。默认不绑定相机；去掉 `--headless` 可在仿真窗口观察运动。

标定默认读取仓库 `dataset.yaml` 并解析源数据 metadata，也可用 `--replay-contract /data/oht/replay-source-v3` 复用已有训练缓存标定。逐帧回放要求记录帧间隔与环境 `step_dt` 一致、`--max-substeps 1`。需要检验未来关键点执行时，在该命令中增加 `--replay-targets keypoints --replay-contract /data/oht/replay-source-v3`，换用新的输出目录；自动使用 waypoint 执行，不强制源帧周期匹配。物体默认保留环境 reset 状态；物体映射、关节顺序和诊断字段见 [OHT replay 说明](oht-replay.md)。

## 检查与 replay 烟测

在 BridgeVLA 根目录、能读取匹配 replay 且没有其他客户端使用服务时执行：

```bash
python tools/smoke_oht_vla_server.py \
  --server http://109.105.46.126:8004 \
  --replay /data/oht/replay-source-v3 \
  --split val --index 0
```

此命令仅发送缓存观测并输出预测，不启动 IsaacLab、不执行动作，也不是 GT 动作回放。

## 日志与排错

可选配置排查：在正常运行命令中增加 `--dry-run` 并换用新输出目录，仅保存场景/TCP/夹爪诊断后退出，不执行策略动作。正常运行无需此参数。

| 文件或现象 | 检查内容 |
| --- | --- |
| 服务端 `info.json / requests.jsonl` | checkpoint、模型/契约、请求状态及分段耗时 |
| 客户端 `run.json / control_diagnostics.json` | 实际 USD、TCP 变换、控制尺度、关节顺序、夹爪标定 |
| 客户端 `episode_*.json / summary.json` | 预测、实际命令、执行前后位姿及误差、stop_reason、成功统计 |
| 每步 `camera_observation` | 发给策略的相机位姿、K、原始分辨率与外参来源；USD 分支优先用有效的 `cameraViewTransform`，不可用时回退 USD/实时物理位姿 |
| 每步 `client_timing_ms` | `capture` 相机/状态采集，`pack` 数组/Base64 打包，`serialize` JSON 编码，`http` 完整 HTTP 往返及回包解析，`predict` 后台预测调用总耗时，`request` 主线程等待预测耗时（含刷新），`execute` 控制执行，`total` 该步总耗时；单位均为 ms，字段包含关系不能直接相加 |
| 每步 `timing_ms` | 服务端 `decode / prepare / policy / total`，与客户端计时分别保留；`--log-interval 1` 可逐步打印两者 |
| 每步 `execution` | waypoint 状态、approach/gripper 物理步数、固定目标、夹爪状态与最终误差；`joint_diagnostics` 记录实际关节位置、硬/软限位及余量，缺失的硬限位为 null，不以软限位代替；streaming 没有此字段 |
| `stop_reason=max_steps` | 已达到策略/关键点请求数上限；物理控制步数看 commands 和 execution.steps |
| `waypoint_stalled / waypoint_timeout` | 目标未完成，保留已执行命令并停止后续预测；查看终端 `waypoint stopped` 的 step、phase、errors，以及 episode 最后一条 action.execution；默认日志间隔不会隐藏失败摘要 |
| 输出停止摘要后窗口关闭 | 默认由客户端在保存失败记录后主动关闭环境并返回 1；有窗口检查时在运行命令增加 `--keep-open-on-error`，不会继续推进物理或请求策略 |
| 运动异常或长期限幅 | 对比 `measured_before/after` 与 `errors_before/after`，区分目标跳变与执行跟踪问题；不要仅翻转夹爪符号 |
| HTTP 409 / 429 / 503 | 分别检查模型/会话冲突、服务 busy、服务未就绪；请求结果不确定时不重发旧动作，待空闲后重置 episode |
| 退出时 `Invalid NodeObj` | 区分相机清理错误与运行期间错误，检查 episode 是否已完成；其他 cleanup warning 保留排查 |
| 在线 heatmap 可视化 | 打开 `/visualization`，检查 `/model_info.visualization` 和 `/visualization/latest`；离线 GT 对比见 [OHT 评估说明](../../finetune/OHT/README.md#8-离线与闭环评估) |

客户端自动处理 `/model_info`、`/reset`、`/predict` 和会话校验；服务端支持 prepared/raw 两种观测格式。协议字段与实现见[服务设计](../design/oht-vla-server.md#4-http-接口)，坐标、TCP 与参考脚本差异见[对照检查](../design/oht-client-server-audit.md)，动作模式见[动作模式指南](oht-action-modes.md)。

CPU 回归使用仿真接口替身和合成数据；真实 CUDA 模型、传感器几何与闭环成功率需在服务器验证。旧参考 helper 和自定义 runtime 的参数可用下列命令查询：

```bash
python tools/oht_bridgevla_client.py --help
bash finetune/OHT/vla_server.sh --help
```
