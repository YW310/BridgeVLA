# OHT FastAPI VLA server

本入口提供 `GET /health`、`GET /model_info`、`POST /reset`、`POST /predict`。支持 prepared 观测及 IsaacLab 原始 RGB-D；模型结构、物体辅助模式、相机及渲染方向从 checkpoint 获取；server 返回模型输出，IsaacLab client 执行。它参考 H-VLA2 的服务组织方式，但不是现有 H-VLA2 `/predict` client 的逐字段兼容替代品。

## 安装与启动

在现有 BridgeVLA 模型/CUDA 环境中安装 Web 依赖：

```bash
python -m pip install -r finetune/OHT/requirements-server.txt
```

从仓库根目录启动：

```bash
bash finetune/OHT/serve.sh \
  --model-path /data/oht/runs/baseline-s0 \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 8010 \
  --output-dir logs/oht_vla_server_8010
```

`--model-path` 可指向 checkpoint 文件，或包含 `model_last.pth` 的 run 目录；`--checkpoint` 为互斥别名。`--model_path`、`--output_dir` 也可使用。`serve.sh` 是兼容入口，实际调用同目录下的 `vla_server.sh`；也可直接运行 `python -m finetune.OHT.vla_server`。

参考 H-VLA2 shell 的环境变量启动方式：

```bash
GPU_ID=1 \
OHT_MODEL_PATH=/data/oht/runs/baseline-s0/model_last.pth \
OHT_HOST=0.0.0.0 \
OHT_PORT=8004 \
OHT_SERVER_OUTPUT=logs/oht_vla_server_8004 \
bash finetune/OHT/vla_server.sh
```

| 环境变量 | 默认值 / 作用 |
| --- | --- |
| `BRIDGEVLA_ROOT` | 根据脚本位置定位当前仓库 |
| `PYTHON_BIN` | `python`；可指向 Conda 环境解释器的完整路径 |
| `GPU_ID` | 未设置时保留已有 `CUDA_VISIBLE_DEVICES`；设置后覆盖可见 GPU |
| `OHT_MODEL_PATH` | 无默认模型；必须设置它或传 `--model-path` / `--checkpoint` |
| `OHT_DEVICE` | `cuda:0` |
| `OHT_HOST` / `OHT_PORT` | `127.0.0.1` / `8010` |
| `OHT_SERVER_OUTPUT` | `logs/oht_vla_server` |
| `OHT_PRETRAIN_PATH` | 可选 BridgeVLA 热图预训练目录，对应 `--pretrain-path` |
| `HF_HUB_OFFLINE` | `0`；设置 `1` 启用 Hugging Face 离线加载 |
| `TRANSFORMERS_OFFLINE` | 未单独设置时跟随 `HF_HUB_OFFLINE` |

命令行参数优先于对应环境变量，其他参数原样透传。例如环境变量设置 `OHT_PORT=8004`，传 `--port 8011` 后实际使用 8011。`GPU_ID=1` 让物理 GPU 1 成为进程内的逻辑 `cuda:0`；不要因此把默认 device 改为 `cuda:1`。脚本不自动激活 Conda，可先激活环境，或设置 `PYTHON_BIN=/home/yiwei/env/bin/python`。

脚本自动进入所选仓库，模型/输出等相对路径按仓库根目录解释；检查 server 文件、Python 解释器、checkpoint，以及显式预训练目录中的 `model.safetensors.index.json`。启动时打印最终参数、GPU 可见性和离线设置；`--help` 不要求模型，也不创建日志目录。它将 `PYTHONPATH` 固定为该仓库及其 `finetune` 目录，启用无缓冲输出，最后通过 `exec` 启动 Python。

该脚本没有沿用 H-VLA2 的 `dataset_statistics.json` / `tokenizer/` 检查，也没有新增 `--use_bf16`：这些不是当前 BridgeVLA 加载器的接口。只有预先缓存好 PaliGemma 和 processor 等资源时，才设置两个 offline 开关为 `1`；默认不强制离线。

模型加载仍调用现有 `load_policy()`，基础 PaliGemma/processor 权重需已缓存或可访问。按现有环境需要传入 `--pretrain-path`。external predictor checkpoint 另传 `--predictor` 与 `--predictor-provenance`，必须匹配训练记录。不要将 H-VLA checkpoint 直接传给此入口。

可选预热：

```bash
bash finetune/OHT/serve.sh \
  --model-path /data/oht/runs/baseline-s0/model_last.pth \
  --warmup-replay /data/oht/replay-source-v3 \
  --warmup-split val --warmup-index 0 \
  --port 8010 --output-dir logs/oht_vla_server_8010
```

预热 replay 必须与 checkpoint 契约一致；只将观测和 instruction 传入模型，预热后 reset。模型加载、预热、推理、reset 均在同一条专用工作线程执行。Uvicorn 固定 `workers=1`，不使用 `torchrun`。多模型使用不同进程、端口和输出目录。

## 分机部署：VLA server 与 IsaacLab client

IsaacLab 可以在另一台服务器运行，模型权重和 BridgeVLA CUDA 环境只需放在 VLA server 上。默认 `127.0.0.1` 仅用于本机访问；分机部署时，在启动命令中显式绑定 VLA 服务器用于通信的 IP。

例如 VLA 服务器 IP 为 `192.168.1.10`，在该服务器运行（替换为真实 IP 和模型路径）：

```bash
GPU_ID=1 \
OHT_MODEL_PATH=/data/oht/runs/baseline-s0/model_last.pth \
OHT_HOST=192.168.1.10 \
OHT_PORT=8004 \
bash finetune/OHT/vla_server.sh
```

`OHT_HOST` 必须是 VLA 服务器自身的地址；多网卡时选择 IsaacLab 服务器可路由到的地址。在 IsaacLab 所在服务器测试：

```bash
VLA_SERVER_URL=http://109.105.46.126:8004
curl "${VLA_SERVER_URL}/health"
curl "${VLA_SERVER_URL}/model_info"
```

IsaacLab client 向 `${VLA_SERVER_URL}/predict` 发送观测和 instruction，使用 `/reset` 开始新 episode。客户端填写服务端实际 IP，不填 `localhost` 或客户端自己的 IP。两台服务器需网络互通，且服务端 TCP `8004` 端口允许连接。分机部署支持 prepared 或 raw RGB-D 请求，不要求共享模型文件；replay 烟测使用 prepared 格式，另需能读取匹配的 replay。实时客户端使用 raw RGB-D，启动方式见下文。

## 检查与 replay 烟测

```bash
curl http://127.0.0.1:8010/health
curl http://127.0.0.1:8010/model_info
```

`/model_info` 返回 API/model 标识、checkpoint 路径、数据契约 hash、观测字段及 shape/dtype、相机名、训练 MVT 配置和输出描述。默认 model ID 根据启动时 checkpoint 绝对路径/大小/mtime 派生，不是权重内容的加密摘要；可用 `--model-id` 设置部署名。

在能访问同一 replay 的终端运行：

```bash
python tools/smoke_oht_vla_server.py \
  --server http://127.0.0.1:8010 \
  --replay /data/oht/replay-source-v3 \
  --split val --index 0
```

工具检查服务器与 replay 契约，读取一个已校验的 NPZ 观测，创建新 episode，调用 `/reset` 和 `/predict`，输出动作及耗时。它不会调用 IsaacLab，也不会执行动作；请在没有其他 client 使用该服务时运行。instruction 来自所选 sample 的 `goal`，teacher/GT 动作不发送给 server。

## 请求契约

固定 API 版本为 `bridgevla_predict_v1`，支持 `bridgevla_prepared_v1` 和 `bridgevla_rgbd_v1` 两种观测格式。先取得 `/model_info`，将其中的 `model_id` 和 `contract_sha256` 带入请求。原有 prepared 请求保持兼容。

`POST /reset` 的完整外层字段：

```json
{
  "api_version": "bridgevla_predict_v1",
  "request_id": "case-001-reset",
  "episode_id": "case-001",
  "model_id": "<model_info 返回值>",
  "contract_sha256": "<model_info 返回值>"
}
```

`POST /predict` 在相同身份字段上增加：

```json
{
  "api_version": "bridgevla_predict_v1",
  "request_id": "case-001-0",
  "episode_id": "case-001",
  "model_id": "<model_info 返回值>",
  "contract_sha256": "<model_info 返回值>",
  "step": 0,
  "timestamp": 0.0,
  "instruction": "<实际任务指令>",
  "observation_format": "bridgevla_prepared_v1",
  "observation": {}
}
```

`observation` 必须填充 `/model_info.observation_schema` 中全部且仅这些字段。每个数组封装为 `{dtype, shape, data}`，`data` 是数组字节的 Base64；可复用 `finetune.OHT.runtime.transport.pack_observation()`，不用先将数组转换成嵌套数值列表。

- RGB 为 uint8 CHW；depth 为米制训练契约定义的深度；XYZ 是 world 空间、与像素对齐的组织化点云。
- K 为对应图像分辨率内参，T 为 `T_world_from_optical`。不能直接将未转换的 USD/OpenGL 外参放入该字段。
- 浮点字段声明 float32，也接受 float64 并转成 float32；无效 depth/XYZ 的 NaN 按已有契约处理，其余状态及标定必须有效。
- 相机与尺寸从 checkpoint 获取，缺失相机报错，不自动复制其他视图。额外 teacher、标签和未知字段会被拒绝。
- 原始 RGB-D 使用下文单独的 `bridgevla_rgbd_v1` 格式；prepared 格式不接收原始传感器字段、`json_numpy` 字符串、H-VLA2 `proprio` 或稀疏点云。prepared 观测的构造参考 OHT README 中现有 IsaacLab 适配说明。
- 请求外层字段固定，不接受未声明选项；DDIM、CFG 等 H-VLA2 运行时参数不会被静默忽略。

返回值包含同一 request/episode/step/timestamp、`action`、checkpoint 的 `action_spec` 和 `timing_ms`。不返回伪造的 normalized/fused action 或置信度。客户端应先校验响应标识，再使用动作。

## 会话、并发与错误

首次 predict 可以从 step 0 自动建立 episode；之后切换任务/episode，必须显式 `/reset` 到新的 episode ID。同一 episode 的 step 从 0 连续递增，测量时间戳严格递增，instruction 保持一致。最近 4096 个已执行 request ID 去重，step/episode 检查独立于该窗口。

只允许一个正在处理的推理/reset，忙时返回 429，不排队；health/model_info 不经过模型工作线程，推理中仍可查询。请求超时或 HTTP 协程取消不会取消后台 GPU 工作；工作线程完成前仍保持 busy。结果不确定时不要重发同一步，待 busy=false 后用新 episode ID reset。推理异常后当前 episode 标记为失败，也需新 episode/reset。

| 状态码 | 含义 |
| --- | --- |
| 400/422 | 无效 JSON、字段、观测或版本 |
| 409 | 模型/契约不匹配、重复请求、episode/step/instruction 冲突 |
| 413 | 实际 HTTP body 超过限制；默认 32 MiB，可用 `--max-request-bytes` 调整 |
| 415 | Content-Type 不是 application/json |
| 429 | 模型正在推理或 reset |
| 503 | 服务未就绪或正在退出 |
| 500 | 推理/reset 内部异常，详细 traceback 仅写服务日志 |

输入校验失败不推进 step，也不调用模型。退出时停止接收新工作并等待已运行推理完成。旧的 `/act` 服务保留原接口，需单独运行，不能用旧 `runtime.transport.Client` 直接调用新入口。

## IsaacLab 实时 RGB-D 客户端

独立入口是仓库自带的 `tools/oht_bridgevla_client.py`，通过 HTTP 调用 BridgeVLA。默认使用已实现的原生工厂 `runtime/isaaclab_factory.py`，自行启动 Isaac Sim、加载 OHT 环境、绑定相机并执行 rollout，**无需 `--reference-client`、`my_oht_runtime.py` 或自定义项目模板**。底层模块入口 `python -m finetune.OHT.isaaclab_client` 同样默认使用该工厂。IsaacLab 工程仍提供实际场景资产、任务注册和控制器配置。

连接顺序为：两台机器同步本次 BridgeVLA 代码 → 在模型服务器启动 VLA server → 在 IsaacLab 机器用 `curl` 检查 `/health` 和 `/model_info` → 启动下列客户端。模型服务器启动和分机 IP/端口配置见上文“分机部署”；`/model_info.observation_formats` 应包含 `bridgevla_rgbd_v1`，更新代码后需重启旧服务才能启用新格式。

### 独立 BridgeVLA 客户端

在**仿真服务器**激活能运行 OHT 的 IsaacLab 环境，在 IsaacLab 根目录启动：

```bash
cd /home/yiwei/IsaacLab
python /home/yiwei/BridgeVLA/tools/oht_bridgevla_client.py \
  --isaaclab-root "$PWD" \
  --server http://109.105.46.126:8004 \
  --task disassemble_left \
  --env-id Isaac-Deploy-Reach-UR16e-DualEE-AllScene-OSC-v2 \
  --scene-preset oht-v4 \
  --camera-source usd \
  --episodes 1 --max-steps 10 \
  --output logs/bridgevla-native-smoke \
  --headless --device cuda:0
```

启动文件保存在 BridgeVLA 仓库内时，会自动找到自己的仓库根目录，无需设置 `PYTHONPATH`。也可将这一个启动文件复制到 IsaacLab 根目录，随后执行同样的参数，额外指定 `--bridgevla-root /home/yiwei/BridgeVLA`，或设置环境变量 `BRIDGEVLA_ROOT`。它仍调用仓库中的 BridgeVLA 实现，因此仿真服务器需要该仓库和 NumPy/SciPy 等客户端依赖，不需要模型权重、VLM/CUDA 推理依赖、JEPA、ZeroMQ 或原参考 Python 文件。`--help` 不启动 Isaac Sim。

HTTP 连接的是模型服务，客户端在自身进程中创建仿真，不会附着到另一个已打开的 Isaac Sim 进程。原生入口使用 AppLauncher 启用相机后才导入仿真/任务模块，再通过 `gym.make()` 创建单环境。先校验 `/model_info`，按 checkpoint 动作模式检查环境控制器；不会自动将 OSC 环境改为关节位置环境，也不会修改 PD 增益或力矩上限。

- `--env-id`：实际已注册的 OHT 环境 ID；默认使用注册配置。
- `--scene-preset registry|oht-v4`：默认 `registry` 保留已注册场景；本文对齐参考 T1V4 的命令显式选择 `oht-v4`。V4 预设加载完整 `UR16eDualEEAllSceneOSCReachEnvCfgV4`，校验配置模块位于所选 IsaacLab 工程内、机器人 USD 指向 `<ISAACLAB_ROOT>/oht/assets_v4/ALL/Collected_all/all.usd` 且文件存在、EE body 为 `base_link_0`、OSC 为 `pose_rel`。任何不匹配都报错，不退回注册配置或 V2。该预设为 OSC 场景，不适用于 joint checkpoint。
- `--expected-robot-usd /absolute/path/all.usd`：自定义场景时校验实际配置引用的本地 USD 路径；不会替换 `spawn.usd_path`。应加载匹配的完整环境配置，避免只换 USD 而保留错误的夹爪、控制器和 EE 定义。
- `--env-config module:ConfigClass`：显式替换该环境的配置类。参考 T1V4 使用的类为 `isaaclab_tasks.manager_based.manipulation.deploy.reach.config.ur_16e.all_scene_osc_env_cfg:UR16eDualEEAllSceneOSCReachEnvCfgV4`；`--scene-preset oht-v4` 已自动选择并校验该类，无需再手工指定。使用自定义配置时保留 `--scene-preset registry` 并提供 `--env-config`，可另加 `--expected-robot-usd` 校验资产。
- `--task-import module`：额外导入 OHT 环境注册模块，可重复；内置已导入 `isaaclab_tasks`。
- `--ee-link name`：训练 `link_to_tcp` 对应的真实机器人 link；默认读取 OSC 的 `body_name`。关节位置环境没有 `body_name` 时必须显式提供。
- `--ee-pose-source controller|body`：原生工厂默认 `controller`，观测 EE 位姿为 `body pose × arm_action.body_offset`，匹配本仓库 OHT 数据配置中“记录位姿已含 TCP 偏移”的约定。执行器不会再添加相同偏移。导出器记录裸 link，或 profile 的 `link_to_tcp` 已含完整 body→TCP 偏移时，显式选择 `body`，避免重复转换。直接嵌入 `IsaacLabRuntime` 的旧调用仍默认 `body`，需按数据契约显式选择。
- `--quaternion-order wxyz|xyzw`：已安装 IsaacLab 的状态及配置四元数顺序，默认 `wxyz`（2.x）；使用其他约定的版本需显式设置。
- `--camera-source auto|sensors|usd`：`auto` 默认优先使用完整的已注册 Camera sensors，否则从现有 USD camera prim 创建 Replicator render products。`sensors` 强制使用已注册 sensors；`usd` 强制使用 USD 相机。
- `--camera-map file.json`：覆盖 checkpoint 逻辑相机名到 sensor 名/USD prim 名/绝对 prim 路径的全部映射。默认 USD 名为逻辑名大写，重名必须提供精确路径；缺少右侧等必需相机时明确报错。
- `--camera-width 640 --camera-height 480 --camera-warmup-ticks 8`：USD RGB-D 的渲染尺寸及只渲染的预热次数；预热不执行 `env.step()`。
- `--dry-run`：启动环境和相机，保存 `run.json`、`control_diagnostics.json` 后退出，不发 `/predict`、不执行控制用的 `env.step()`。夹爪检查只调用 action term 的 `process_actions` 并恢复原值，不调用 `apply_actions`。
- `--log-interval N`：默认首步和每 10 步输出预测/目标、跟踪误差及夹爪物理状态；`1` 每步输出，`0` 关闭。

USD 模式只读取 RGB、`distance_to_image_plane` 和 pinhole `camera_params`；深度直接是米制 Z-depth，不从 pointcloud 生成。机器人挂载相机以 USD 中的固定 body→camera 变换结合实时物理 body pose 求世界位姿；wrist 无法绑定机器人 body 时拒绝运行，避免使用停留在旧位置的 USD wrist pose。相机矩阵始终按 OpenGL 约定发送，在服务端转换一次。实现参照 [AppLauncher 官方说明](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/00_sim/launch_app.html)和 [Replicator 官方 API](https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.13.30/source/extensions/omni.replicator.core/docs/API.html)。

如需文件放在 IsaacLab 根目录，可按以下方式复制启动文件，随后不再用原参考脚本：

```bash
cp /home/yiwei/BridgeVLA/tools/oht_bridgevla_client.py ./oht-bridgevla-client.py
python ./oht-bridgevla-client.py \
  --bridgevla-root /home/yiwei/BridgeVLA --isaaclab-root "$PWD" \
  --scene-preset oht-v4 \
  --server http://109.105.46.126:8004 --task disassemble_left \
  --camera-source usd --episodes 1 --max-steps 10 \
  --output logs/bridgevla-copied-smoke --headless --device cuda:0
```

四个任务统一使用本仓库 `data/common.py` 中的 instruction；任务配置不能覆盖 instruction、模型输入、相机标定或动作坐标系。原生入口使用所选环境的 reset；`--task` 决定 instruction/profile，不会自动改变场景资产布局，装配/拆卸分布需由相应环境或下文 initializer 配置保证。不注入参考脚本的 GT 状态、固定第 300 帧、q99 或原 JEPA 模型策略。

已有特殊环境的项目仍可通过 `--runtime-factory module:factory` 覆盖内置工厂，或在现有仿真进程中直接构造 `IsaacLabRuntime(env, cameras, ...)` 和 `IsaacLabBridge`。自定义 factory 签名仍为 `create_runtime(*, args, model_info)`，返回原生 runtime；`--reference-client` 仅保留为旧 helper 兼容入口。两者互斥，均未指定时使用仓库自带的内置工厂。

首次使用采用上面的短 rollout，默认 `--initialization env-reset --score none`；`summary.json` 的 success/success_rate 为 null。四个任务的初始化和评分通过自己的配置接入，可复制并修改 [任务配置示例](../../finetune/OHT/configs/isaaclab_tasks.example.yaml)：

```bash
python /home/yiwei/BridgeVLA/tools/oht_bridgevla_client.py \
  --isaaclab-root /home/yiwei/IsaacLab \
  --scene-preset oht-v4 \
  --server http://109.105.46.126:8004 --task assemble_left \
  --task-config /home/yiwei/isaaclab_tasks.yaml --score task \
  --episodes 20 --max-steps 500 --output logs/assemble-left-eval
```

示例中的 `assemble_left_success` 等名称是待接入环境的条件名，必须在 `env.termination_manager` 中实际注册。缺失条件或将 timeout 用作成功条件会报错。没有环境成功条件时，可配置 `evaluator_factory: my_module:create_evaluator`，签名为 `factory(env=..., runtime=..., task=...)`，返回实现 `reset(episode_index=...)` 和 `update(step=..., executed_gripper_cmd=...) -> bool` 的对象。评分器只记录结果及结束 episode，不向策略输入 GT、不切换模型或控制阶段。环境的 terminated/truncated 会停止当前 episode，但不会单独视为成功。

`--initialization env-reset` 使用环境原有 reset。增加 `--initialization task` 后，按所选任务的 `reset_assets` 恢复已注册 `env.scene` 刚体在 runner 初始 reset 后的位置/姿态并清零速度；可再用 `initializer_factory` 实现任务特有的机器人或资产随机化。该 factory 同样接收 `env/runtime/task`，返回 `reset(episode_index=..., seed=...)` 对象。必须至少配置 reset_assets 或 initializer_factory；USD 裸 prim、装配约束和采集分布需要由自己的 initializer 处理。四个任务各自声明设置，不复用左轮拆卸成功几何阈值。

### 使用现有参考脚本启动

已有 `oht-test-t1v4-gtinfer-client.py` 时，使用这个入口，无需 `my_oht_runtime.py` 或额外的自定义项目。在 IsaacLab 服务器中激活原参考代码可运行的环境，可以从 BridgeVLA 根目录启动，通过 `--isaaclab-root` 指定仿真工程；本机客户端不需要 VLA checkpoint。下面的 `/home/yiwei/...` 均需替换为该服务器上的实际路径，`--reference-client` 指向参考 Python 文件本身：

IsaacLab 根目录是**仿真服务器**上含 `isaaclab.sh` 的工程目录。`--reference-client` 只指定参考文件，工程根目录由 `--isaaclab-root` 单独指定。如果原参考脚本就是从该目录运行，在原来可正常运行它的终端执行 `pwd -P` 即可获得路径。不清楚安装位置时，可在仿真服务器执行：

```bash
find "$HOME" /data -maxdepth 6 -type f -name isaaclab.sh 2>/dev/null
```

输出文件所在目录就是候选根目录；多个副本时，选择原参考代码实际使用、含 OHT 任务和资产的那份，并激活与它匹配的 IsaacLab 环境。路径选择优先级为 **`--isaaclab-root` > 环境变量 `ISAACLAB_ROOT` > 启动目录本身含 `isaaclab.sh`**。三者都未设置时保持原有行为，不扫描文件系统猜测安装位置。

选定根目录后，客户端会在加载参考脚本/调用原生 factory **之前**校验 `isaaclab.sh`，设置 `ISAACLAB_ROOT`、将根目录加入 Python 模块搜索路径，并切换到该目录，使相对资产路径与从 IsaacLab 根目录启动时一致。无需再手动 `export ISAACLAB_ROOT`；参考 collection helper 会读取设置后的值。此参数不替代正确安装的 IsaacLab/OHT Python 包，也不会自动切换 Conda 环境。

BridgeVLA 已知路径参数 `--reference-client`、`--task-config`、`--camera-map` 和 `--output` 的相对路径按**启动命令所在目录**解析，不随内部目录切换而改变。额外透传给 runtime 的路径参数仍由它按 IsaacLab 工作目录解释。所用绝对根目录会输出到启动日志及 `run.json`；退出或异常时恢复调用方的工作目录、`ISAACLAB_ROOT` 和 Python 搜索路径。

```bash
cd /home/yiwei/BridgeVLA
python -m finetune.OHT.isaaclab_client \
  --isaaclab-root /home/yiwei/IsaacLab \
  --reference-client /home/yiwei/oht-test-t1v4-gtinfer-client.py \
  --server http://109.105.46.126:8004 \
  --task disassemble_left \
  --env-id Isaac-Deploy-Reach-UR16e-DualEE-AllScene-OSC-v2 \
  --episodes 1 --max-steps 10 \
  --output logs/bridgevla-http-smoke \
  --headless --device cuda:0
```

需要旧参考脚本的左轮拆卸评估时，使用新输出目录，增加 `--initialization collection --score disassemble-left --episodes 20 --max-steps 500`。此兼容评分开关仅支持 `disassemble_left`，依赖参考脚本原有采集 helper/配置；不自动注入第 300 帧或 GT state。其他任务使用自己的 `--score task` 配置。

客户端从 `/model_info` 获取所需相机。原生 factory 必须提供逻辑名完全匹配的相机字典；兼容入口默认映射为同名大写 USD prim（例如 `global_right -> GLOBAL_RIGHT`），可用 `--camera-map camera-map.json` 覆盖全部映射。两者均绑定 checkpoint 所需全部相机，缺失即报错。采集 uint8 HWC RGB、metric `distance_to_image_plane` Z-depth、实时 K 和 OpenGL sensor→world 位姿；位置统一为相对 `env_origins[0]` 的 world 坐标，在线协议四元数统一为 wxyz。每次查询前 render-only 刷新；无效深度转 NaN。参考相机 warmup 在最终 episode reset 前执行，不在首帧测量时额外推进物理。

每个 raw 请求的 `observation_format` 为 `bridgevla_rgbd_v1`，`observation` 结构为：

```text
task: assemble_left / assemble_right / disassemble_left / disassemble_right
data_profile: /model_info.raw_observation.data_profiles 中的精确键
tensors: { 每项均使用现有 dtype/shape/Base64 封装
  joint_positions: float32[6]                  # 实测关节，不补零、不做 q99
  ee_link_pose: float32[7]                     # xyz+wxyz；server 应用 link_to_tcp
  gripper_position: float32[1]                 # 实测夹爪关节位置之和
  <camera>_rgb: uint8[H,W,3]
  <camera>_depth: float32[H,W]                  # 米制 Z-depth，不传 PNG 编码值
  <camera>_intrinsics: float32[3,3]             # 对应本次 H,W 的 K
  <camera>_pose: float32[7]                     # OpenGL sensor→world xyz+wxyz
}
```

`task` 与 `data_profile` 用于选择 checkpoint 内已保存的逐数据源配置；按任务能唯一匹配时自动选择，否则显式指定 `--data-profile`。服务端使用该 profile 的夹爪标定和因果迟滞、depth limits/kind、毫米舍入、点云过滤和图像尺寸，基于实时 K/pose 反投影，再按训练整数步长采样。OpenGL→optical 仅转换一次；relative EEF 所需 TCP 由实测 link 位姿与 profile 的 `link_to_tcp` 得到。在线 metric depth 不再次执行离线视频的 log/量化解码，不沿用 JEPA 的 q99、7→8 补零或单图 PIL resize。若启用 ray depth，服务端先将 Z-depth 换为 ray distance。原始相机通常 480×640，五相机 RGB-D 请求约十余 MiB，仍受 `/model_info.max_request_bytes` 和 tensor 总量限制约束。

同一 episode 的 format、task、data_profile 和 instruction 固定；切换需要 `/reset` 和新 episode ID。服务端在推理成功后才提交夹爪迟滞状态；非法输入不会消耗 step 或修改历史，reset 会清空历史。

EEF 执行先按 `action_spec` 解码为绝对 TCP 目标，relative EEF 的锚点是该次请求的实测 TCP；随后用 `link_to_tcp` 逆变换得到 EE link 目标，应用 OSC 的 `body_offset`，再转换到实时 robot-root 坐标。跟踪时每次读取当前 link 位姿，计算 base-frame 残差，先按 `--max-translation`（默认每轴 0.05m）和 `--max-rotation`（默认每轴 0.15rad）裁剪，再除以 OSC 配置中的 `position_scale/orientation_scale`。夹爪 open01 映射成环境的 +1/-1。默认 `--max-substeps 1`；增加子步时跟踪同一个固定目标，不重复叠加同一预测增量。动作 term 必须为 OSC，采用 `pose_rel` 控制目标，原生 runtime 的 `ee_link` 须与 `body_name` 一致；使用自定义 `task_frame_rel_path` 的环境需先提供相应适配，内置执行器会明确拒绝。

其中“应用 body_offset”按观测参考点处理：`controller` 观测已经是控制点，观测→控制点变换为单位阵；`body` 观测才需要应用配置的偏移。HTTP 的历史字段名 `ee_link_pose` 保留，但其内容须由上述来源选择与训练记录参考点对齐；profile 的 `link_to_tcp` 仅应用一次。模型 `eef` 输出 8 维 `[world_xyz, quat_xyzw, open01]`，IsaacLab OSC 实际接收 7 维 `[base_dxyz, base_rotvec, gripper_sign]`，不能将这两种数组直接互换。

### 运动异常时先检查控制契约

先使用新输出目录运行诊断：

```bash
cd /home/yiwei/IsaacLab
python /home/yiwei/BridgeVLA/tools/oht_bridgevla_client.py \
  --isaaclab-root "$PWD" --server http://109.105.46.126:8004 \
  --scene-preset oht-v4 \
  --task disassemble_left --camera-source usd \
  --ee-pose-source controller --dry-run \
  --output logs/bridgevla-control-check --headless --device cuda:0
```

检查 `control_diagnostics.json` 中的 `body_to_observation`、`link_to_tcp`、`observation_to_control`、实时 root 变换、关节顺序和夹爪目标。`gripper.verified=true` 仅说明环境的 `process_actions(+1/-1)` 与其配置的 open/close 目标一致，不保证配置的物理开口、训练标定或成功抓取已验证。标准 [IsaacLab BinaryJointAction](https://raw.githubusercontent.com/isaac-sim/IsaacLab/v2.2.1/source/isaaclab/isaaclab/envs/mdp/actions/binary_joint_actions.py) 为正值打开、负值关闭；BridgeVLA 的 `1=open` 映射为 `+1`，不应未经验证直接翻转符号。

`gripper.calibration_direction_matches` 单独检查命令目标之和的物理开闭方向与训练端点是否相同；相反或退化时在动作执行前退出，不自动修改夹爪语义。端点数值差异仍需结合实际开口与数据范围判断。客户端启动时完整校验 `action_spec` 的坐标、四元数、单位、参考点、维数和夹爪定义。原生环境关闭 `terminations.time_out`，由客户端限制 rollout 长度，其他终止条件保留；注册 Camera 传感器开启实时位姿更新。具体证据、与参考脚本的区别和未验证项见[client/server 对照检查](../design/oht-client-server-audit.md)。

新版服务 `/model_info` 的 raw profiles 会显示训练的物理夹爪 open/close 端点及 scene bounds，需更新并重启服务后获得；旧服务缺少这些元信息时报告为 null。若环境目标之和与训练端点明显不符，应核对 V2/V4 场景和夹爪配置，而不是把物理关节值、open01 和命令符号混为一谈。一次 `--max-substeps 1` 只推进一个控制步，不能据此认定已到达模型预测的未来关键点；绝对目标跳变或长期处于限幅，需要结合执行前后的误差确认原因。

正常 rollout 的每条 action 现在还保存 `measured_before`、`measured_after`、`errors_before`、`errors_after`：实测 EE 为 xyz+wxyz，TCP 为 xyz+xyzw，夹爪为物理关节位置之和。它们用于区分模型目标异常、坐标/参考点不匹配和控制器跟踪不足。

joint/relative_joint 模式使用内置 `JointPositionExecutor`，要求环境的 `arm_action` 为绝对 `JointPositionAction`，不能送入 OSC 七维口。执行器读取实时 scale/offset，反算原始控制量并校验 action transform 是否发生额外裁剪；目标超出 soft joint limits 时在步进前报错。checkpoint 配置有 `joint_names` 时显式按名称映射，否则必须由环境保证 arm term 顺序与训练六关节顺序一致。控制向量按 action manager 的实际 term 顺序排列，夹爪为单维 +1 开/-1 关。执行器沿用环境现有 PD/驱动配置，通过 `env.step()` 步进。

relative_joint 按请求时的实测关节还原绝对目标一次。`--joint-hold-steps` 默认 1，`--joint-target-tolerance` 默认 0.01rad；可选 `--joint-max-step` 限制每控制步向目标移动的关节位移，默认不限制。多次跟踪时保持同一个物理目标，不重复累加 delta。需要其他驱动方式时可用 `--joint-executor module:factory` 覆盖，`factory(env, runtime)` 返回 `execute(absolute_joint_target, after_step=callback)`；返回实际控制量列表，每步调用 callback，终止时设置 `last_done=True`。

输出目录拒绝覆盖，`run.json` 保存任务、instruction、模型/契约标识、action_spec、profile 和执行/评分配置；`episode_*.json` 保存原始预测、解码后的固定目标、实际控制量及服务耗时，`summary.json` 保存结果。评分时超时、异常和未成功的已尝试 episode 都留在分母；异常后停止控制，不自动重试请求，未执行的 episode 单列为 unattempted。额外参数传入原生 factory 的 `args.runtime_args` 或兼容参考脚本；`--help` 不启动 Isaac Sim。

## 日志与验证

客户端输出如 `episode=0 steps=10 success=None stop_reason=max_steps`，表示按 `--max-steps 10` 完成了 10 次策略请求/动作执行后结束；`--score none` 下成功状态保持 null，不代表任务失败或已成功。`stop_reason` 同时写入 episode 和 summary JSON，可为 `max_steps`、`success`、`environment_done` 或 `error`。任务成功率需要上文的显式 `--score task` 和实际环境成功条件。

若退出时出现 `Invalid NodeObj object in Py_Node in getAttributes`，应检查它是否发生在 episode 完成后的相机清理阶段。这与运行期间 `/predict` 或动作执行报错不同。多相机现在先 detach 全部 annotators，再销毁全部 render products，与 [Isaac Sim 官方多相机示例](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/replicator_tutorials/tutorial_replicator_getting_started.html)的释放顺序一致。部分 Kit 版本若仍返回已失效的 annotator NodeObj，退出时仅汇总一次并继续释放；其他 detach/destroy 异常仍显示带相机和操作名的 warning。此改动需在实际 Isaac Sim 环境重跑确认。

`config=registry` 表示使用 `--env-id` 的已注册配置，没有显式选择 V4。环境 ID 后缀 `OSC-v2` 也不能单独证明实际资产版本：参考 T1V4 同样使用这个 ID，但额外传入完整 V4 config。现在启动日志会显示 scene preset、配置类/来源文件及实际 robot USD；`run.json` 和 `control_diagnostics.json` 的 `scene` 字段保存同样信息及资产校验结果。旧日志没有 USD 路径时不能据此确认资产版本，需重跑诊断。重跑使用新的 `--output` 目录；客户端拒绝覆盖已有结果。

输出目录包含 `info.json` 和 `requests.jsonl`。info 记录模型、相机、配置、Asia/Shanghai 启动时间和依赖版本；逐请求日志记录标识、状态和耗时，不保存完整 RGB-D/点云。默认每份日志 10 MiB，保留 3 份轮转备份。

离线 `python -m finetune.OHT.eval open-loop ...` 默认开启三视图、预测 heatmap 和 GT/TCP 投影可视化，每 50 个样本保存一次；`--visualize-every 1` 保存每个样本，`--visualize-every 0` 关闭。可选 `--visualize-output-dir` 指定新图片目录；默认是报告路径去掉扩展名再加 `_visualizations/`。评估结束后打开该目录的 `index.html`；完整命令及 PNG/NPZ 字段见 [OHT README 的离线评估说明](../../finetune/OHT/README.md#8-离线与闭环评估)。可视化开关属于离线评估，服务的 `/predict` 响应仍按上述动作协议返回。

`timing_ms.decode` 是 JSON 解码，`prepare` 是张量解码/过滤/校验，`policy` 是模型调用及输出回传的墙钟耗时，首次调用还包含 episode reset；`total` 是服务端处理时间，不包含 client 网络往返和动作执行时间。

```bash
python -m pytest tests/test_oht_vla_server.py -q
python -m pytest tests/test_oht_server_launcher.py -q
python -m pytest tests/test_oht_isaaclab_http.py tests/test_oht_native_runtime.py tests/test_oht_standalone_client.py -q
```

测试使用合成真实 replay，覆盖模型仅加载一次、12 种动作头/辅助模式组合的 Policy 一致性、非法输入隔离、取消后的 busy、并发拒绝、reset、退出等待、轮转日志及真实 Uvicorn HTTP 烟测。原生运行时测试用仿真接口替身覆盖四任务×四动作模式的 HTTP 联调、相机四元数顺序、关节顺序/scale/offset/限位、相对目标固定及终止停止。VLM/光栅化和 IsaacLab 运行时在 CPU 集成测试中使用小替身；完整 CUDA checkpoint、实际场景条件和 IsaacLab 闭环仍需在服务器验证。
