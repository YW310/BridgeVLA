# OHT VLA server 与 IsaacLab 客户端

先启动模型服务器，再在 IsaacLab 仿真服务器启动客户端。两台机器均需同步 BridgeVLA 代码并激活各自环境；更新服务代码后需重启服务。以下路径按实际安装位置替换。

## 安装与启动

在 **VLA 模型服务器**的 BridgeVLA 根目录执行：

```bash
python -m pip install -r finetune/OHT/requirements-server.txt

GPU_ID=1 bash finetune/OHT/vla_server.sh \
  --model-path /data/oht/runs/baseline-s0/model_last.pth \
  --device cuda:0 \
  --host 0.0.0.0 --port 8004 \
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

在仿真服务器激活能运行 OHT 的 IsaacLab 环境，在其根目录执行以下命令。客户端启动仿真、连接 VLA 服务并执行策略；示例运行 1 个 episode，最多请求并执行 500 次策略动作。

```bash
cd /home/yiwei/IsaacLab
python /home/yiwei/BridgeVLA/tools/oht_bridgevla_client.py \
  --isaaclab-root "$PWD" \
  --server http://109.105.46.126:8004 \
  --task disassemble_left \
  --env-id Isaac-Deploy-Reach-UR16e-DualEE-AllScene-OSC-v2 \
  --scene-preset oht-v4 --camera-source usd \
  --ee-pose-source controller \
  --episodes 1 --max-steps 500 \
  --output logs/bridgevla-v4-run-01 \
  --headless --device cuda:0
```

**每次运行使用尚不存在的输出目录。** 默认 `--initialization env-reset --score none`，因此 `success=null`；这不是任务成功率评估。

去掉 `--headless` 即可显示窗口。客户端默认在后台处理 HTTP reset/predict（包括观测打包），等待期间由主线程以最多约 20 Hz 调用 [IsaacLab render 接口](https://github.com/isaac-sim/IsaacLab/blob/v2.2.1/source/isaaclab/isaaclab/sim/simulation_context.py)刷新窗口；headless 下不做额外等待渲染。等待时不推进物理，收到预测后才执行原有控制步；关闭窗口后丢弃未执行的预测。窗口响应与策略动作频率是两项指标，动作频率仍受采集、传输和推理耗时影响。

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
| `--max-substeps` | 默认 1；每次模型目标最多跟踪多少控制步，多子步不重复累加预测增量 |
| `--max-translation / --max-rotation` | OSC 每轴物理限幅，默认 0.05 m / 0.15 rad，限幅后再除以控制器尺度 |
| `--dry-run` | 可选：只保存启动诊断，不执行策略动作 |
| `--log-interval` | 默认 10；1 为每步打印，0 关闭 |
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

这是可选的初始化/评分配置，正常运行使用前面的命令即可，默认 `--initialization env-reset --score none`。

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

标定默认读取仓库 `dataset.yaml` 并解析源数据 metadata，也可用 `--replay-contract /data/oht/replay-source-v3` 复用已有训练缓存标定。回放要求记录帧间隔与环境 `step_dt` 一致、`--max-substeps 1`。物体默认保留环境 reset 状态；物体映射、关节顺序和诊断字段见 [OHT replay 说明](oht-replay.md)。

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
| 每步 `client_timing_ms` | `capture` 相机/状态采集，`pack` 数组/Base64 打包，`serialize` JSON 编码，`http` 完整 HTTP 往返及回包解析，`predict` 后台预测调用总耗时，`request` 主线程等待预测耗时（含刷新），`execute` 控制执行，`total` 该步总耗时；单位均为 ms，字段包含关系不能直接相加 |
| 每步 `timing_ms` | 服务端 `decode / prepare / policy / total`，与客户端计时分别保留；`--log-interval 1` 可逐步打印两者 |
| `stop_reason=max_steps` | 已达到策略请求上限；一次控制步不保证到达未来关键点 |
| 运动异常或长期限幅 | 对比 `measured_before/after` 与 `errors_before/after`，区分目标跳变与执行跟踪问题；不要仅翻转夹爪符号 |
| HTTP 409 / 429 / 503 | 分别检查模型/会话冲突、服务 busy、服务未就绪；请求结果不确定时不重发旧动作，待空闲后重置 episode |
| 退出时 `Invalid NodeObj` | 区分相机清理错误与运行期间错误，检查 episode 是否已完成；其他 cleanup warning 保留排查 |
| heatmap 可视化 | 使用离线 `eval open-loop`，见 [OHT 评估说明](../../finetune/OHT/README.md#8-离线与闭环评估)；HTTP `/predict` 返回动作 |

客户端自动处理 `/model_info`、`/reset`、`/predict` 和会话校验；服务端支持 prepared/raw 两种观测格式。协议字段与实现见[服务设计](../design/oht-vla-server.md#4-http-接口)，坐标、TCP 与参考脚本差异见[对照检查](../design/oht-client-server-audit.md)，动作模式见[动作模式指南](oht-action-modes.md)。

CPU 回归使用仿真接口替身和合成数据；真实 CUDA 模型、传感器几何与闭环成功率需在服务器验证。旧参考 helper 和自定义 runtime 的参数可用下列命令查询：

```bash
python tools/oht_bridgevla_client.py --help
bash finetune/OHT/vla_server.sh --help
```
