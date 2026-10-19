# OHT FastAPI VLA server

本入口提供 `GET /health`、`GET /model_info`、`POST /reset`、`POST /predict`。模型结构、物体辅助模式、相机及渲染方向从 checkpoint 获取；server 返回模型输出，IsaacLab client 执行。它参考 H-VLA2 的服务组织方式，但不是现有 H-VLA2 `/predict` client 的逐字段兼容替代品。

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

命令行参数优先于对应环境变量，其他参数原样透传。例如环境变量设置 `OHT_PORT=8004`，传 `--port 8011` 后实际使用 8011。`GPU_ID=1` 让物理 GPU 1 成为进程内的逻辑 `cuda:0`；不要因此把默认 device 改为 `cuda:1`。脚本不自动激活 Conda，可先激活环境，或设置 `PYTHON_BIN=/path/to/env/bin/python`。

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

固定 API 版本为 `bridgevla_predict_v1`，首版观测格式为 `bridgevla_prepared_v1`。先取得 `/model_info`，将其中的 `model_id` 和 `contract_sha256` 带入请求。

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
- 当前不支持 raw RGB-D、`json_numpy` 字符串、H-VLA2 `proprio` 或稀疏点云自动转换。prepared 观测的构造参考 OHT README 中现有 IsaacLab 适配说明。
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

## 日志与验证

输出目录包含 `info.json` 和 `requests.jsonl`。info 记录模型、相机、配置、Asia/Shanghai 启动时间和依赖版本；逐请求日志记录标识、状态和耗时，不保存完整 RGB-D/点云。默认每份日志 10 MiB，保留 3 份轮转备份。

`timing_ms.decode` 是 JSON 解码，`prepare` 是张量解码/过滤/校验，`policy` 是模型调用及输出回传的墙钟耗时，首次调用还包含 episode reset；`total` 是服务端处理时间，不包含 client 网络往返和动作执行时间。

```bash
python -m pytest tests/test_oht_vla_server.py -q
python -m pytest tests/test_oht_server_launcher.py -q
```

测试使用合成真实 replay，覆盖模型仅加载一次、12 种动作头/辅助模式组合的 Policy 一致性、非法输入隔离、取消后的 busy、并发拒绝、reset、退出等待、轮转日志及真实 Uvicorn HTTP 烟测。VLM/光栅化在 CPU 集成测试中使用小替身；完整 CUDA checkpoint 和 IsaacLab 闭环仍需在服务器验证。
