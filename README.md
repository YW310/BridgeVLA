<div align="center">

# BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models

A 3D VLA framework that aligns the input and output within a shared 2D space in both pre-training and fine-tuning, enabling strong data efficiency and achieves impressive performance in both basic and generalization settings.


[\[📄Paper\]](https://arxiv.org/abs/2506.07961)  [\[🏠Project Page\]](https://bridgevla.github.io/)  [\[📊Dataset\]](https://huggingface.co/datasets/LPY/BridgeVLA)  [\[🤗Checkpoints\]](https://huggingface.co/datasets/LPY/BridgeVLA)


</div>


## 🔥 News
* **`2026.04.06`** 🌟 We release [MV-VDP](https://lpy1219.github.io/MV-VDP-Web/), a spatio-temporal aware video action model that leverages the similar projection and back-projection strategy. It demonstrates great data-efficiency, robustness, generalization and interpretability. We invite you to check it out!
* **`2025.09.20`** 🌟 BridgeVLA was accepted by NeurIPS 2025! 🥳🥳🥳
* **`2025.06.15`** 🌟 We are proud to introduce BridgeVLA, a model designed to bridge the gap between VLM backbone and VLA by aligning input and output within a shared 2D space during both pre-training and fine-tuning.

## 👀 Contents

- [Model Overview](#model-overview)
- [Installation](#installation)
- [Training](#training)
  - [RLBench 8×40 GB 训练](#rlbench-8x40)
  - [Raw → Replay 生成](#rlbench-raw-replay)
  - [Oracle 3D 物体 Replay](#oracle-replay)
  - [严格 Semantic-GT T/R](#semantic-gt-roles)
  - [O2 Target/Reference 训练](#o2-training)
  - [训练日志与实时 Loss](#rlbench-training-logs)
- [Evaluation](#evaluation)
- [Experimental Results](#experimental-results)
- [TODO](#todo)
- [Acknowledgement](#acknowledgement)
- [Contact](#contact)
- [Citation](#citation)


<a id=model-overview></a>

## 📋 Model Overview
As illustrated in the following figure, BridgeVLA employs a dual-phase training recipe. During pre-training, it is trained to predict 2D heatmaps on object detection datasets. During fine-tuning, point clouds are projected into multiple 2D images as inputs to the VLM backbone. The model is trained to predict 2D heatmaps for estimating the translational action and other action components. **This design aligns the input and output within a shared 2D space in both pre-training and fine-tuning.**
![](./assets/network.png)
<a id=installation></a>

## 🛠️ Installation
1. **Clone this repository and navigate to the BridgeVLA folder:**
```bash
git clone https://github.com/BridgeVLA/BridgeVLA.git
cd BridgeVLA
```

2. **Install the required package:**
```bash
conda create -n bridgevla python=3.9 -y
conda activate bridgevla

# For pre-training
cd pretrain
bash ./pretrain_install.sh

# For RLBench fine-tuning
cd finetune/RLBench
bash ./install_rlbench.sh

# For Colosseum fine-tuning
cd finetune/Colosseum
bash ./install_colosseum.sh

# For GemBench fine-tuning
cd finetune/GemBench
bash ./install_gembench.sh
```
3. Note: To avoid potential conflicts between different simulation benchmarks, we suggest creating separate virtual environments for each benchmark. Also, our model is built upon [Paligemma](https://huggingface.co/google/paligemma-3b-pt-224), which is a gated repo. Therefore, you should first be authenticated to access it.
<a id=training></a>

## 🚀 Training
If you want to reproduce our results, please use the same training hyperparameters in the config file. **Do not forget to modify the corresponding saving path in the file before running the following code.**
1. **Pre-training:**
We use the object detection data in the RoboPoint dataset to pre-train the model. We upload the data and checkpoints [here](https://huggingface.co/datasets/LPY/BridgeVLA/tree/main/). With the `pretrain/pretrain.py` file, you can do three things:
* `visualiztion`: This function is used to visualize the pre-training dataset.
* `pre-training`: This function is used to pre-train the Paligemma model on the dataset .
* `evaluation`: This function is used to test the pre-trained checkpoints.
What you need to do is to modify the checking branch in the file and then run the following code:
```bash
cd pretrain
bash pretrain.sh --branches BRANCH_OPTION --config_path PATH_TO_CONFIG_FILE --json_detection_path PATH_TO_DETECTION_JSON --image_folder PATH_TO_IMAGE_FOLDER
```
2. **RLBench Fine-tuning:** To finetune on RLBench, you should first prepare the dataset. You can generate the train and test data yourself by following the instructions in [PerAct repository](https://github.com/peract/peract?tab=readme-ov-file#data-generation). Or you can directly download the dataset we generate to fully reproduce our results [here](https://huggingface.co/datasets/LPY/BridgeVLA_RLBench_TRAIN_DATA/tree/main). To improve the data loading speed, we will first convert the raw data into replay buffer. The training code will automatically do that if it does not find the replay buffer. Meanwhile, you can also directly download the replay buffer we preprocess [here](https://huggingface.co/datasets/LPY/BridgeVLA_RLBench_TRAIN_BUFFER/tree/main). After the data is ready, you can use the `finetune/RLBench/train.sh` file to finetune the model. Please run the following code:
```bash
cd finetune/RLBench
bash train.sh --exp_cfg_path  configs/rlbench_config.yaml \
              --exp_note debug \
              --freeze_vision_tower \
              --log_dir exp/RLBench \
              --load_pretrain \
              --pretrain_path  LPY/BridgeVLA/checkpoints/RLBench/model_80.pth 
```

<a id=rlbench-8x40></a>

### RLBench fine-tuning on one 8 x 40 GB node

The reduced-hardware profiles keep the paper-scale effective batch size of
192 with gradient accumulation. The recommended profile uses batch size 2 per
GPU on 8 GPUs and accumulates 12 micro-batches before each optimizer update.

Run the four-task trend reproduction (20,000 optimizer steps):

```bash
cd finetune/RLBench
GPUS_PER_NODE=8 bash train_8x40.sh \
    --exp_cfg_path configs/rlbench_trend_8x40.yaml \
    --exp_note trend_seed0 \
    --freeze_vision_tower \
    --log_dir exp/RLBench \
    --load_pretrain \
    --pretrain_path PATH_TO_2D_HEATMAP_PRETRAINED_MODEL \
    --save_initial_checkpoint \
    --save_optimizer_state
```

For the full 18-task, 83,300-step schedule, replace the config with
`configs/rlbench_full_8x40.yaml`. Checkpoints include the completed optimizer
step when `--save_optimizer_state` is enabled, so the same command can resume
with `--resume /path/to/model_last.pth`. Prebuilt replay buffers are strongly
recommended; replay generation is not part of the distributed training run.

Evaluate all 18 tasks with one isolated simulator process per GPU:

```bash
python eval_parallel.py \
    --model-folder PATH_TO_CHECKPOINT_FOLDER \
    --eval-datafolder PATH_TO_RLBENCH_EVAL_DATA \
    --model-name model_80.pth \
    --gpus 0,1,2,3,4,5,6,7 \
    --eval-episodes 25
```

The runner creates a unique run directory, merges the 18 task CSV files, and
writes `summary.json` with the macro success rate. It does not record videos.


<a id=rlbench-raw-replay></a>

### RLBench Raw → Replay 独立生成

`tools/generate_rlbench_replay.py` 使用与训练入口完全相同的
`create_replay()` / `fill_replay()` 实现，把已保存的 RLBench raw episode 转换为标准
BridgeVLA/YARR replay。它会完成关键帧发现、demo augmentation、动作离散化、四相机观测、
RN50 CLIP 语言特征、`replay_info.npy` 和 final-observation sentinel。请在已安装
RLBench/PyRep/YARR/peract_colab 的 `bridgevla` 环境中，从仓库根目录运行。

先仅检查任务和 episode：

```bash
python tools/generate_rlbench_replay.py \
    --raw-data-dir LPY/BridgeVLA_RLBench_TRAIN_DATA \
    --output-dir LPY/BridgeVLA_RLBench_TRAIN_Buffer \
    --split train \
    --task stack_blocks \
    --start-episode 0 \
    --num-demos 100 \
    --demo-augmentation-every-n 10 \
    --device cuda:0 \
    --dry-run
```

确认路径和 episode 数量后移除 `--dry-run` 正式生成。输出位于
`LPY/BridgeVLA_RLBench_TRAIN_Buffer/stack_blocks`，可直接作为下一节 Oracle 增强脚本的
`--replay-dir`。如果 `--raw-data-dir` 已经指向 `.../train`，程序也会自动识别，
不需要再次拼接 `train`。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--raw-data-dir PATH` | 必填 | 数据集根目录、`train` split、单任务目录或 `episodes` 目录。 |
| `--output-dir PATH` | 必填 | Replay 输出根目录；每个任务写入 `PATH/<task>`，必须与 raw 路径相互独立。 |
| `--split NAME` | `train` | raw 根目录下的 split 名。 |
| `--task NAME` | `all` | 任务名、逗号分隔任务名或 `all`；可重复传入。 |
| `--start-episode N` | `0` | 第一个 raw episode 编号。 |
| `--num-demos N` | 从起点到最后 | 连续 episode 数量；若范围内缺号会在生成前报错。 |
| `--demo-augmentation-every-n N` | `10` | 每隔 N 个 raw 帧创建一个 demo augmentation replay 子序列，与原训练默认值一致。 |
| `--no-demo-augmentation` | 关闭 | 每个 episode 只从起始帧生成一个子序列，数据更少且与默认训练数据分布不同。 |
| `--device DEVICE` | `auto` | CLIP 特征设备；`auto` 优先 `cuda:0`，也可指定 `cpu` 或其他 CUDA 卡。 |
| `--clip-model NAME_OR_PATH` | `RN50` | OpenAI CLIP 名称或本地权重路径；首次使用名称时可能需要下载权重。 |
| `--batch-size N` | `1` | YARR replay schema 的 batch size；不影响生成的 transition 内容。 |
| `--replay-capacity N` | `300000` | 内部 UniformReplayBuffer 容量。 |
| `--dry-run` | 关闭 | 只发现并检查输入，不加载 CLIP、不创建输出。 |
| `--skip-existing` | 关闭 | 校验已有任务 replay 后跳过；不会补写不完整目录。 |
| `--overwrite` | 关闭 | 删除并重建已存在的精确任务输出；与 `--skip-existing` 互斥。 |

生成时先写入 `<output-dir>/.<task>.raw_to_replay.tmp`。只有 replay 文件连续、
`replay_info.npy` 长度一致且普通 transition/final sentinel 校验通过后，才会原子改名为
正式任务目录。若程序中断，临时目录会保留用于检查；确认无需保留后使用
`--overwrite` 重新生成。

<a id=oracle-replay></a>

### RLBench Oracle 3D 物体 Replay 数据准备

> 本节导航：[参数表](#oracle-parameters) · [关键行为与检查](#oracle-checks) ·
> [O2 Target/Reference 训练](#o2-training)

`tools/augment_replay_with_oracle_objects.py` 可直接为已有 BridgeVLA replay 追加
RLBench GT instance 点云，无需重新采集数据或重建原始 replay。脚本用每个
`N.replay` 的 `episode_idx` 和 `sample_frame` 找到四路相机 mask，经
`rgb_handles_to_mask` 解码后与 replay 点云逐像素对齐，并跨视角合并相同 handle。

```text
N.replay[episode_idx, sample_frame]
  -> episode{episode_idx}/{camera}_mask/{sample_frame}.png
  -> front / left_shoulder / right_shoulder / wrist 点云融合
  -> 固定尺寸 Oracle instance 张量
```

新增字段：

```text
oracle_object_points   [MAX_OBJECTS, NUM_POINTS, 3]  float32
oracle_object_centers  [MAX_OBJECTS, 3]              float32
oracle_object_sizes    [MAX_OBJECTS, 3]              float32
oracle_object_ids      [MAX_OBJECTS]                 int32
oracle_object_valid    [MAX_OBJECTS]                 bool
oracle_object_roles    [MAX_OBJECTS]                 int8
```

`oracle_object_roles` 与 ID/valid 使用相同 slot：`0=unknown/padding`、
`1=target`、`2=reference`。启用时序匹配后，角色按夹爪抓取周期计算：闭爪位置附近被
抓取的 object 是 `target`，松爪/放置位置附近除 target 外最近的 object 是
`reference`。同一抓取周期内 T/R 的稳定 object ID 固定不变，只在下一个抓取周期重新
选择；放置位置附近没有有效候选时 reference 为空。
启用时序匹配后，同一刚性物体的多个 raw mask handle 会先合并点云，再占用一个 slot；
`oracle_object_ids` 使用组内最小 handle ID 作为稳定代表，原始成员保存在 task handle
JSON 的 `object_groups` 和 `group_by_handle` 中。

`sample_frame` 是当前 observation 对应的 raw 帧；`keypoint_frame` 不参与 Oracle
对齐。`terminal == -1` 是 YARR final-observation sentinel，写入全 invalid 的填充张量；
可视化时会通过上一条 replay 的 `next_keypoint_frame` 恢复最终 raw 帧。

推荐先在命令末尾保留 `--dry-run` 检查结果，确认后移除它进行正式写入。以下示例使用
默认开启的严格源对齐检查；发现异常会先写 manifest 再停止，不会静默生成错误 instance。
只有已经确认少量异常样本可以退回 baseline 时，才额外添加 `--skip-invalid-frames`：

```bash
python tools/augment_replay_with_oracle_objects.py \
    --replay-dir LPY/BridgeVLA_RLBench_TRAIN_Buffer \
    --raw-data-dir LPY/BridgeVLA_RLBench_TRAIN_DATA/train \
    --output-dir LPY/BridgeVLA_RLBench_TRAIN_TASK_OBJECT_Buffer \
    --detect-robot-handles \
    --robot-detection-frames 128 \
    --robot-detection-stride 5 \
    --robot-detection-window 200 \
    --robot-motion-threshold 0.1 \
    --robot-link-motion-threshold 0.001 \
    --robot-adjacency-distance 0.10 \
    --temporal-id-matching \
    --task-detection-frames 16 \
    --task-prior-filter \
    --filter-thin-planes \
    --min-object-points 1 \
    --max-objects 32 \
    --num-points 512 \
    --workers 8 \
    --cache-frames 256 \
    --visualize-every 100 \
    --visualize-output-dir oracle_visualizations \
    --visualize-objects-only \
    --thin-plane-max-thickness 0.02 \
    --validate-source-alignment \
    --dry-run
```

如果多任务生成中断，不要添加 `--overwrite`。使用与原来完全相同的参数重新运行，
仅在末尾增加 `--resume`（并确保没有 `--dry-run`）：

    python tools/augment_replay_with_oracle_objects.py [原来的全部参数] --resume

工具的正式 replay 使用临时文件校验后原子改名，因此 `--resume` 会跳过输出目录中
已完成的 `*.replay`；只留下 `*.replay.tmp` 或尚未生成正式文件的条目会重新处理。
已完整完成的 task 会在 episode detection 前直接跳过，部分完成的 task 也只检测和处理
剩余 replay 所涉及的 episode。续跑必须保持原 replay、raw data 和过滤参数不变；
如果需要修改生成参数，应改用新的输出目录或显式 `--overwrite` 全量重建。

<a id=oracle-parameters></a>

#### 参数表

| 类别 | 参数 | 默认值 | 说明 |
| --- | --- | --- | --- |
| 输入 | `--replay-dir PATH` | 必填 | 原始 BridgeVLA replay 目录；可为单任务目录或包含多个任务的根目录。 |
| 输入 | `--raw-data-dir PATH` | 必填 | RLBench raw data 的 `train` 目录或可解析到 episode 的上级目录。 |
| 输入 | `--task NAME` | `all` | 任务名、逗号分隔任务名或 `all`；可重复传入。 |
| 输出 | `--output-dir PATH` | 无 | 写入新的 Oracle replay 目录；非 dry-run 时必须与 `--in-place` 二选一。 |
| 输出 | `--in-place` | 关闭 | 原地修改 replay；与 `--output-dir` 互斥，建议优先使用新目录。 |
| 输出 | `--resume` | 关闭 | 断点续跑：跳过已原子写完的正式 replay，只生成缺失文件；与 `--overwrite` 互斥。 |
| 输出 | `--overwrite` | 关闭 | 重新处理并覆盖已有输出 replay/元数据；与 `--resume` 互斥。 |
| 输出 | `--durable-write` | 关闭 | 临时文件重命名前执行 `fsync`；更安全，但网络盘上更慢。 |
| 张量 | `--max-objects N` | `32` | 每帧固定的最大 instance 槽位数；超出部分会截断。 |
| 张量 | `--num-points N` | `512` | 每个 instance 的固定采样点数；不足时有放回采样。 |
| 张量 | `--min-object-points N` | `20` | 跨相机融合并移除 NaN/Inf 后少于该点数的实例会删除；高召回检查可设为 `1`。 |
| 几何过滤 | `--filter-thin-planes` | 关闭 | 用抗噪主平面内点比例删除大薄平面；几何判定优先于当前帧 target/reference。 |
| 几何过滤 | `--thin-plane-max-thickness METRES` | `0.010` | 点到拟合平面的最大内点距离；默认允许主体平面具有约 1 cm 深度噪声。 |
| 几何过滤 | `--thin-plane-min-extent METRES` | `0.30` | 平面内两个方向都至少达到该尺寸时才删除；小于约 30 cm 的托盘、支架面等中小型平面默认保留。 |
| 几何过滤 | `--thin-plane-min-inlier-ratio RATIO` | `0.80` | 至少该比例的点落在平面距离带内才删除；默认允许最多约 20% 深度离群点。提高该值会更保守。 |
| 几何过滤 | `--filter-thin-planes-all-roles` | 默认行为 | 兼容旧命令；现在 target/reference 薄平面也默认删除。 |
| 几何过滤 | `--preserve-role-thin-planes` | 关闭 | 仅当任务确实包含薄片状相关物体时，选择保留 target/reference 薄平面。 |
| 张量 | `--camera NAME` | 四路相机 | 指定相机，可重复传入；默认 `front`、`left_shoulder`、`right_shoulder`、`wrist`。 |
| 排除 | `--exclude-object-id ID` | `0` | 精确排除 decoded handle，可重复传入；`--exclude-robot-id` 是同义参数。 |
| 单帧先验 | `--task-prior-filter` | 关闭 | 按下一关键动作距离排序，并删除明显大平面背景；默认高召回，不按半径删除远处实例。 |
| 单帧先验 | `--task-prior-strict` | 关闭 | 配合 `--task-prior-filter` 删除交互半径外实例；召回率更低，仅在明确需要激进筛选时使用。 |
| 单帧先验 | `--task-prior-radius METRES` | 按任务 | 覆盖 18 个任务配置中的交互半径。 |
| 单帧先验 | `--task-prior-max-instances N` | 高召回不限；strict 按任务 | 限制先验保留的 simulator handle 数。 |
| 单帧先验 | `--task-prior-background-extent METRES` | `0.60` | 两个轴均达到该尺度时视为明显桌面/地面。 |
| 时序匹配 | `--temporal-id-matching`（兼容旧名 `--temporal-task-filter`） | 关闭 | 建立稳定 `handle ID → object group → slot`；以闭爪处物体为 T、松爪放置位置最近的其他物体为 R，并在同一抓取周期固定 T/R；不删除当前帧可见实例。 |
| 时序任务 | `--task-detection-frames N` | `16` | 每个 episode 均匀抽取的最大检测帧数；需覆盖闭爪和松爪边界，长 episode 建议提高到 `24` 或 `32`。 |
| 时序匹配 | `--task-handle-cache-dir PATH` | `<output-dir>/<task>/task_handle_maps` | episode 稳定 slot 与 task handle JSON 缓存；显式 PATH 作为根目录并追加 task 名。 |
| 时序任务 | `--refresh-task-handle-cache` | 关闭 | 忽略已有 task handle JSON 并重新检测。 |
| 机器人 | `--detect-robot-handles` | 关闭 | 检测 wrist 稳定的夹爪 seed，并沿持续邻接的运动学链扩展到机械臂及静止底座；只使用第一次闭合前的前缀，避免把被抓物体当作机器人。 |
| 机器人 | `--robot-detection-frames N` | `64` | 自适应扩展时最多读取的 raw 证据帧数；正常有运动时通常只读取初始窗口的 21 帧。 |
| 机器人 | `--robot-detection-stride N` | `5` | raw 帧采样间隔；默认依次读取 `0, 5, 10, ...`。 |
| 机器人 | `--robot-detection-window N` | `100` | 从 raw 帧 0 开始的初始闭区间；运动不足时才在该窗口之后继续扩展。 |
| 机器人 | `--robot-motion-threshold METRES` | `0.02` | 只控制自适应 raw 采样：夹爪最大位移达到该值后不再扩展采样；不直接改变 arm 判定。 |
| 机器人 | `--robot-link-motion-threshold METRES` | `0.008` | 第一段机械臂 link 需要达到的最小位移；漏掉低幅运动 link 时可尝试 `0.002` 或 `0.001`。 |
| 机器人 | `--robot-adjacency-distance METRES` | `0.05` | 两个 robot handle 的 AABB 最大连接间隙；分段机械臂链断开时可尝试 `0.08`，过大会增加误删风险。 |
| 机器人 | `--robot-handle-cache-dir PATH` | `<output-dir>/<task>/robot_handle_maps` | episode robot handle JSON 缓存；显式 PATH 作为根目录并追加 task 名。 |
| 机器人 | `--refresh-robot-handle-cache` | 关闭 | 忽略已有 robot handle JSON 并重新检测。 |
| 性能 | `--refresh-replay-metadata-cache` | 关闭 | 强制重建 replay 元数据索引；仅在同名 `.replay` 被原地改写时使用，日常运行不要添加。 |
| 检查 | `--validate-source-alignment` | 开启 | 默认严格比较 raw/replay RGB，并比较 raw depth 重建 XYZ 与 replay point cloud；异常时停止，防止静默生成错位 instance。 |
| 检查 | `--no-validate-source-alignment` | 关闭 | 为旧流程关闭内容校验以提高速度；只建议在已离线确认 replay/raw 完全一致时使用。 |
| 检查 | `--alignment-rgb-tolerance VALUE` | `1.0` | RGB 单通道允许的最大差值，按 `0..255` 尺度计算。 |
| 检查 | `--alignment-point-cloud-tolerance METRES` | `0.002` | raw 重建点云与 replay 点云逐像素误差的 P95 上限。 |
| 检查 | `--alignment-min-finite-ratio RATIO` | `0.95` | 每个相机 replay 点云的最小有限点比例，同时用于要求 raw/replay 可比较像素比例。 |
| 容错 | `--skip-invalid-frames` | 关闭 | raw 帧越界或源对齐校验失败时保留 replay，并写入 `valid=False` 的空 Oracle；仅用于明确接受 baseline fallback 的容错实验。 |
| 性能 | `--workers N` | `1` | replay 线程数；建议从 `4` 或 `8` 测试，过高会增加内存和网络盘竞争。 |
| 性能 | `--cache-frames N` | `128` | 相同 raw 帧 Oracle 结果的 LRU 容量；`0` 禁用，内存有限时降低。 |
| 性能 | `--seed N` | `0` | 控制确定性点采样和 dry-run 抽样。 |
| 性能 | `--no-progress` | 关闭 | 关闭 tqdm 进度条。 |
| 检查 | `--dry-run` | 关闭 | 只验证和可视化，不写 replay，也不新建/覆盖 handle 缓存。 |
| 检查 | `--dry-run-samples N` | `5` | 未指定间隔可视化时，dry-run 随机检查的普通 transition 数。 |
| 可视化 | `--visualize-index N` | 无 | 保存指定 replay index；与 `--visualize-every` 互斥。 |
| 可视化 | `--visualize-every N` | `0` | 每隔 N 个排序后的 replay 保存一张 PNG；`0` 关闭。 |
| 可视化 | `--visualize-output-dir PATH` | `oracle_visualizations` | PNG 输出目录，不设置时也会自动创建该默认目录。 |
| 可视化 | `--visualize-objects-only` | 关闭 | 隐藏点云面板中的灰色完整场景，仅绘制保留实例；不影响上排 RGB。 |

<a id=oracle-checks></a>

#### 关键行为与检查

- 推荐同时启用 `--detect-robot-handles`、`--temporal-id-matching` 和
  `--task-prior-filter`：先排除机器人，再建立 episode 级稳定 slot，最后进行单帧
  排序和背景清理。旧参数名 `--temporal-task-filter` 保持兼容，但不再执行硬过滤。
- 时序检测会把交互/邻接 handle 排在 episode slot 映射前部，并把其他观测到的
  handle 追加到稳定映射；`rejected_dynamic_handles` 仅作为诊断和优先级证据，不会由
  temporal 模式删除。某个稳定 handle 在当前帧不可见时保留该 slot，写入
  `valid=False`；其他可见 handle 会使用剩余 slot。
- 时序流程先按持续空间邻接和多帧相对位姿合并 raw handle；`max_instances` 限制的是 group
  数而不是 raw handle 数。episode 内其他帧的夹爪开闭、物体随动、动作邻近和静态接触会
  形成 task group 及 `role_cycles`。每个周期在闭爪时选择附近被抓取的 group 作为 T，
  在后续松爪时选择离夹爪放置位置最近的其他 group 作为 R；从该周期开始到松爪帧，所有
  replay observation 都查询同一组 T/R，不会因夹爪移动或当前帧距离变化而切换。周期外或
  未检测到完整开闭事件时，才使用时序先验与当前帧几何的兼容回退；reference 始终可选且
  每周期最多一个。检测结果保存在 task handle JSON 的 `role_cycles` 字段，运行日志中的
  `cycles=` 依次显示 `[start_frame, end_frame, target_id, reference_id]`，其中
  `reference_id=-1` 表示没有 R。若抓取任务日志为 `cycles=[]`，可使用
  `--task-detection-frames 32`（长 episode 可继续提高）并添加
  `--refresh-task-handle-cache` 重新检测。low-dim observation 使用最多 8 个 episode 的
  LRU 缓存并自动淘汰。
- 刚性分组要求两个 handle 在至少 75% 的共同可见证据中可用、80% 以上持续邻接，且
  多帧中心间距离标准差不超过 1 cm。方向和幅度一致的共同运动可以合并；若两者一直静止，
  边界长期紧密接触也可以合并。所有持续兼容关系最终按连接图的连通分量合并，因此支架的
  多个末端区域即使彼此不直接接触，只要都稳定连接到同一中心/底座，也会形成一个 object
  group。仅在任务后期才接触支架的杯子不满足全时段持续邻接要求，不会并入支架。逐帧角色
  只对分组后的 object group 计算，不对 raw handle 单独赋值。
  普通第一阶段始终使用 2 cm，避免把静止邻近物体在整幅场景中误合并。`place_cups` 只有在
  多帧证据先找到 reference 支架种子后，才允许从该种子向其他静止、相对位姿稳定的区域做
  6 cm 二阶段结构扩展；target 和发生明显运动的杯子不会进入该扩展。
- Robot 检测使用 raw observation 中的当前夹爪位姿、`gripper_open`、GT mask 和 raw
  depth。depth 会用同帧相机内外参重建世界坐标点云，与 mask 像素严格对齐。夹爪 seed 以
  wrist 图像稳定性为主，并允许夹爪旋转造成的世界坐标偏移、部分遮挡及距离离群；夹爪
  handle 不必在第 0 帧可见，在 episode 早期窗口内首次出现（如 raw frame 10）时，会以
  它自己的首个可见帧检查 wrist 邻接和后续随动。严格评分没有 seed 时会从 wrist 稳定候选
  恢复 seed。只在外部相机可见、wrist mask 中不可见的夹爪 link 也允许在早期首次出现；
  arm 扩展会使用它与 seed 的首个共同可见帧，而不再强制要求 raw frame 0 可见。arm 扩展先确认紧邻夹爪的移动 link，
  再沿 episode 早期已连接且持续邻接的运动学链扩展，因此可覆盖运动较少的机械臂底座。
  默认在 `0–100` raw 帧内每隔 5 帧取样；若夹爪相对第 0 帧的最大位移不足 2 cm，则继续
  按相同间隔向后扩展，直到运动足够、达到帧数上限、episode 结束或夹爪第一次闭合。
  `--robot-motion-threshold` 只影响这里的采样长度。机械臂是否进入 `arm_handles` 由
  `--robot-link-motion-threshold`、`--robot-adjacency-distance`、可见率和持续邻接共同决定。
  低幅 link 只要达到独立的绝对运动阈值，就不会再仅因运动量小于夹爪的 25% 被当作静止物体。
  若一直静止会打印 `motion-based robot evidence is weak`，提示运动证据不足。检测只使用
  第一次闭合之前的前缀，因此不会把随后被夹起并跟随夹爪运动的任务物体当成机械臂。
  夹爪持续运动而某实例基本静止时，该实例不会判为机器人；证据不足的实例只进入
  `ambiguous_handles`，不会加入 `excluded_object_ids`。task prior 和时序任务筛选会
  使用 replay 的下一关键动作，因此属于 action-conditioned Oracle 离线标注，不是
  无标签推理阶段的公平筛选器。整个流程不调用 Qwen 或 SAM。
- Replay 文件编号按写入顺序排列，但 `sample_frame` 是稀疏关键帧，不保证 raw 帧号
  连续；`terminal == -1` 分隔的是 replay 子序列。Demo augmentation 可能让同一个
  `episode_idx` 出现多段子序列。Robot 只借助 replay metadata 确定 episode，检测证据
  直接来自 episode 前部等间隔的 raw 帧；检测出的稳定 handle ID 会应用到后续整个
  episode。task 仍执行 episode 级 replay 均匀采样，并按
  `(episode_idx, sample_frame, next_keypoint_frame)` 保留不同动作边。
- 提供 `--output-dir` 时，Task 缓存位于
  `<output-dir>/<task>/task_handle_maps/episode_NNNN.json`，robot 缓存位于
  `<output-dir>/<task>/robot_handle_maps/episode_NNNN.json`。显式指定对应的
  `--*-handle-cache-dir PATH` 时使用 `PATH/<task>/episode_NNNN.json`，避免多任务间
  episode 文件重名。Robot cache 会记录 raw 采样间隔、窗口、帧数上限、采样运动阈值、
  link 运动阈值和邻接距离；
  修改这些参数时会自动失效。修改 task 检测帧数、半径或实例限制后仍应使用
  `--refresh-task-handle-cache`；修复前生成的旧版 task/robot 缓存也会自动失效。
- 缺少可用的 `replay_info.npy` 时，首次运行必须读取每个 `.replay` 的 metadata，并写入
  `.oracle_replay_metadata_v2.npz`。缓存会根据 replay 文件名、大小和修改时间自动失效；
  提供 `--output-dir` 时，缓存位于对应的输出 task
  目录（包括 dry-run）；未提供输出目录或使用 `--in-place` 时才写入输入 replay 目录。
  以后运行会显示
  `replay metadata disk cache hit`，同一次运行中 robot/task 共用索引时显示
  `memory cache hit`。新增、删除或重命名 replay 会自动使索引失效；若原地改写同名文件，
  使用一次 `--refresh-replay-metadata-cache`。如果日志提示无法保存索引，需要检查 replay
  目录写权限，否则下次仍会全量扫描。
- 如果日志显示 `episode detection: replay segments`，说明当前目录存在长度匹配的
  `replay_info.npy`。程序只用它划分 replay segment，并直接从选中的 `.replay` 读取
  `episode_idx`、`sample_frame` 和 `next_keypoint_frame`；这条路径不读取
  `.oracle_replay_metadata_v2.npz`，所以 `--refresh-replay-metadata-cache` 不会修复这里的
  帧越界。注意 `85.replay` 中的 `85` 只是 replay 写入序号，实际 raw episode 仍由文件内的
  `episode_idx` 决定。
- 如果提示 `Replay/raw frame mismatch` 或
  `Cannot read current gripper_pose for frame N`，且 raw episode 中确实没有第 `N` 帧：仅当
  前面的日志是 `replay metadata disk cache hit` 或 `memory cache hit` 时，先使用一次
  `--refresh-replay-metadata-cache`；如果日志是 `replay segments`，或者刷新后仍然越界，则
  对应 `.replay` 与 raw episode 确实不匹配。这通常表示 replay 与 raw data 来自不同版本，
  或 raw episode 不完整。应改用生成 replay 时的同一批 RLBench demonstrations，或基于当前
  raw data 重新生成 replay。不要把越界帧强行截到最后一帧，也不要直接改写
  `sample_frame`，否则会造成图像、点云、夹爪状态和动作监督错位。
- 若确认只有少量异常样本，并允许它们退回 baseline，可添加
  `--skip-invalid-frames`：程序保留 replay 的连续结构，为越界样本写入
  `valid=False` 的空 Oracle，并输出 `ignored_invalid_frames` 与
  `ignored_invalid_replays`。这属于容错绕过，不会修复数据配对关系。
- 源对齐检查默认开启。图像来自 raw `*_rgb/{sample_frame}.png`，instance mask 来自
  raw `*_mask/{sample_frame}.png`，而训练点云来自 `.replay` 的
  `*_point_cloud`；程序现在会检查三者是否对齐。缺少点云字段、整帧或任一 mask instance
  的有限点比例不足、RGB 不一致，或 raw depth 重建点云的整帧/逐实例 P95 误差超过阈值，
  都会判为无效，因此小物体点云单独缺失也不会被全局统计掩盖。
- 每个 task 的检查结果写入输出目录下的 `invalid_alignment_manifest.json`；即使严格模式
  在第一个异常处退出，manifest 也会先原子落盘。dry-run 时写入
  `<visualize-output-dir>/<task>/invalid_alignment_manifest.json`。最终汇总中的
  `alignment_invalid_frames` 是检测到的唯一异常帧数。
- raw 数据没有单独的 point-cloud 文件夹属于正常现象：点云由 depth PNG、
  `low_dim_obs.pkl` 中的相机内参/外参和 near/far 在线重建。若 depth PNG 或相机参数缺失，
  则该帧无法可靠生成 Oracle instance，不应只因为 RGB 存在就继续训练。
- `dry-run` 会打印 `excluded_object_ids`、`no_finite_point_object_ids`、
  `small_object_ids`、`task_prior_filtered_object_ids`、
  `temporal_filtered_object_ids` 和 `truncated_object_ids`。命令行时序匹配不再产生
  `temporal_filtered_object_ids`；若缺失 ID 不在其他列表中，说明它在当前帧所选相机
  的 GT mask 中不可见。
- 每张可视化 PNG 使用两排八个面板：上排为 front、left shoulder、right shoulder、
  wrist RGB；下排为 3D、XY、XZ、YZ 点云。相同 episode/handle ID 跨帧颜色固定；图中
  不直接显示可能达到千万级的 simulator handle，而按 episode 稳定 slot 映射为从 `1`
  开始的连续小编号。上排利用同帧 GT mask 为下排实际保留的 ID 绘制半透明同色框：
  unknown 只显示小编号（例如 `3`），target/reference 分别简写为 `T_3` / `R_4`。
  底层 `oracle_object_ids` 和 handle 缓存仍保留真实 ID，不影响跨帧匹配与训练数据。
  3D 图例和三个正交视图采用相同标签；某个实例在当前相机不可见时不画框。
  点云使用固定米制
  场景边界；缺失 RGB 显示
  `RGB unavailable`，不会中断生成。
- 默认先写 `.tmp`、回读验证后原子重命名，不覆盖原始数据。缓存会自动淘汰已完成的
  旧帧；进度条中的 `cache_hits`、`cache_misses` 和 `cache_entries` 可用于检查效果。

**突然只剩一个物体时：**先比较启动日志中的 `slots=[...]` 和图片标题里的
`episode_idx/sample_frame`。`--max-objects 32` 是 episode 稳定映射的容量上限，不会
补回当前帧不可见的实例。

- 如果 `task handles` 只有一个，但 `slots` 有多个，其他实例仍会保留；task handles
  现在只决定容量不足时的优先级，不再构成白名单。旧版按白名单删除实例的 task cache
  会通过版本号自动失效。
- 如果某个 replay 只剩一个，检查 dry-run 输出：`excluded_object_ids` 表示被
  robot/手工 ID 排除，`task_prior_filtered_object_ids` 表示被显式单帧先验删除，
  `small_object_ids` 表示点数不足，`thin_plane_object_ids` 表示被显式薄平面规则删除，
  `protected_thin_plane_object_ids` 表示几何上是薄平面但因 target/reference 角色而保留，
  `no_finite_point_object_ids` 表示 mask 可见但点云
  全为 NaN/Inf。缺失 ID 不在任何列表时，说明它在当前帧所选相机的 GT mask 中不可见
  或被完全遮挡；不再归因于 temporal 匹配。
- 若 robot 检测仍有疑似误判，检查
  `<output-dir>/<task>/robot_handle_maps/episode_NNNN.json`：只有
  `gripper_handles` 和
  `arm_handles` 会硬删除；`grasped_handles` 是被抓物体保护集合，
  `ambiguous_handles` 仅供诊断。旧版 robot cache 会自动失效；重新生成 robot 结果后
  也应刷新 task cache。
- 图片标题切换到新的 `episode_idx` 时，会改用另一份 episode cache；这不是同一
  episode 内 ID 突变。`sentinel=True` 对应 `terminal == -1` 填充 transition，保存的
  Oracle 张量为空是预期行为。

可先提高 robot/task 采样密度和交互半径重新检查；dry-run 只实时重检，不写入新缓存，
确认后需移除 `--dry-run` 才会保存结果。robot 结果会影响后续 task slot，因此刷新顺序
是先 robot、再 task：

```bash
--detect-robot-handles \
--robot-detection-frames 64 \
--robot-detection-stride 5 \
--robot-detection-window 100 \
--robot-motion-threshold 0.02 \
--robot-link-motion-threshold 0.002 \
--robot-adjacency-distance 0.08 \
--refresh-robot-handle-cache \
--temporal-id-matching \
--task-detection-frames 32 \
--task-prior-radius 0.30 \
--refresh-task-handle-cache \
--dry-run \
--visualize-index 100
```

同一 `episode_idx` 的多个 replay 子序列共用 slot 映射；切换到新的 `episode_idx` 时
使用独立映射。训练随机采样 replay 时不依赖上一条 transition。

训练加载 Oracle replay 时，通过 `--train_replay_storage_dir` 显式指定 Oracle
输出根目录，并启用与数据准备阶段一致的张量尺寸：

    bash train.sh --train_replay_storage_dir /path/to/augmented_replay \
        --exp_cfg_opts 'use_oracle_objects True oracle_max_objects 32 oracle_num_points 512' \
        [其他训练参数]

use_oracle_objects 默认为 False，因此原始非 Oracle replay 的加载行为保持不变。

<a id=semantic-gt-roles></a>

### 严格 Semantic-GT Target/Reference

正式 O2 upper-bound 不再使用最近距离、运动幅度、Qwen 或时域 ID 猜测角色。唯一语义
契约是 `finetune/RLBench/configs/rlbench_o2_semantic_roles.yaml`：Target 是当前未完成
子目标中必须直接接触、抓取或控制的实体；Reference 是该子目标终止条件中与 Target
构成空间关系的唯一物体或 site。单物体关节任务没有 Reference。一个语义实体可合并
多个 simulator handles，phase 只在 live RLBench 成功条件满足后推进。

| 任务类型 | Target / Reference | phase 规则 |
| --- | --- | --- |
| 单关节 | `open_drawer`、`push_buttons`、`turn_tap`：T 为源码指定的可动部件，R 不存在 | 对应 joint condition 满足 |
| 单次放置 | `close_jar`、`light_bulb_in`、`meat_off_grill`、`place_shape_in_shape_sorter`、`place_wine_at_rack_location`、`put_groceries_in_cupboard`、`put_item_in_drawer`、`put_money_in_safe`、`slide_block_to_color_target` | variation 决定唯一 T/R；detector 和需要时的释放条件满足 |
| 顺序操作 | `place_cups`、`stack_blocks`、`stack_cups` | 固定源码顺序；空抓、错误张合和其他物体移动不推进 |
| 工具任务 | `reach_and_drag`：stick/target；`sweep_to_dustpan_of_size`：broom/dustpan site | 不新增第三个 Tool 通道 |
| 几何选择 | `insert_onto_square_peg`：ring/与 `success_centre` 对齐的 pillar | 四个 detector 同时满足 |

Reference `kind=object` 时从四视角 GT handle mask 提取完整点云；`kind=site` 时读取
success sensor/dummy 的 world position，并重复到 `oracle_num_points` 后走原有 Gaussian
projection。site 只用于 Oracle upper-bound，manifest 和 replay audit 字段都会显式记录
`kind=site`。

正式生成前先对全部 variation 做 strict reset 审计（不需要 checkpoint）：

```bash
cd finetune/RLBench
python validate_semantic_roles.py \
    --output-dir semantic_role_validation \
    --headless
```

任一对象选择器无法解析、T/R 混入 robot handle 或层级不满足契约时立即报错；成功时
输出 `variation_role_audit.json`、逐 variation 首帧 audit 图和 provider 统计。

#### 1. 用 simulator GT 生成 phase/handle manifest

在 RLBench 环境中回放保存的 expert keypoints。每个 episode 只调用一次
`reset_to_demo`，provider 直接查询 task 对象属性、层级、variation、success condition 和
四视角 GT mask：

```bash
cd finetune/RLBench
TASKS="close_jar place_cups push_buttons stack_blocks" \
MODEL_FOLDER=/path/to/baseline_checkpoint_folder \
MODEL_NAME=model_80.pth \
EXP_CFG_PATH=configs/rlbench_config.yaml \
EVAL_DATAFOLDER=/path/to/BridgeVLA_RLBench_TRAIN_DATA/train \
EVAL_EPISODES=100 \
EPISODE_LENGTH=50 \
REPLAY_GROUND_TRUTH=1 \
SAVE_VIDEO=0 \
ORACLE_PROVIDER=rlbench_gt \
ORACLE_STRICT=1 \
ORACLE_DEBUG=1 \
bash eval.sh
```

manifest 生成只回放 expert action，不调用 policy，因此可以使用已有 baseline checkpoint，
不依赖尚未训练的 O2 checkpoint；模型仅用于复用现有 eval 启动入口。

对 18 个任务可把 `TASKS` 设为 `finetune/bridgevla/utils/rvt_utils.py` 中的完整任务列表。
若 expert keypoint 数超过 `EPISODE_LENGTH`，离线重写器会拒绝不完整 manifest，不能静默
沿用最后一个 phase。每个 checkpoint/task 的输出位于：

- `.../eval/<task>/rlbench_gt/<model>/semantic_oracle/semantic_role_manifests/<task>/episode_N.json`；
- `oracle_provider_stats.json`：区分 `mapping_errors`、`not_visible_*` 和 `no_reference`；
- `semantic_role_audits/<task>/episode_N/role_audit_step_000.png`：首帧四视角 overlay、
  原图、instance/T/R mask、三正交 T/R 点云以及 phase condition 状态。

#### 2. 只重写 Oracle 字段，生成 semantic-GT buffer

```bash
python tools/rewrite_replay_with_semantic_roles.py \
    --replay-dir LPY/BridgeVLA_RLBench_TRAIN_Buffer \
    --raw-data-dir LPY/BridgeVLA_RLBench_TRAIN_DATA/train \
    --manifest-dir /path/to/model/eval \
    --output-dir LPY/BridgeVLA_RLBench_SEMANTIC_GT_Buffer \
    --task all \
    --max-objects 32 \
    --num-points 512 \
    --cache-frames 128 \
    --cache-episodes 2 \
    --resume
```

工具保留 action、图像、点云、语言、`episode_idx/sample_frame` 和其他 baseline 字段；只
替换六个 Oracle tensor，并增加不输入网络的审计字段：schema version、phase ID、T/R
semantic name、kind、原始 handle 集合及各角色 valid。输出中的 T/R 使用固定小 slot ID
`0/1`，不会把上千万的 simulator handle 当作显示 ID；真实 handle 仍保存在 audit 字段。

严格行为如下：

- live mask 与保存 mask 在相同 manifest frame 的 handle 体系不一致：立即停止并报告
  `mapping_error`，禁止用邻近实例代替；
- 角色正确但当前四个相机均不可见：该角色 `valid=False` 并计入 `not_visible`；
- 任务定义没有 R：计入 `no_reference`，不是异常，网络的 R residual 为零；
- raw/replay frame 越界：立即停止，不截断到最后一帧，也不生成伪点云；
- `--resume` 只跳过已经原子写完的 replay；`--overwrite` 与它互斥。
- `--cache-frames` 与 `--cache-episodes` 都是有界 LRU；默认最多保留 128 个 Oracle
  帧和 2 个 episode 的 low-dim 数据，不会随已处理 episode 数持续增长。

#### 3. 正式 semantic-GT O2 训练

```bash
cd finetune/RLBench
bash train.sh \
    --exp_cfg_path configs/rlbench_o2_semantic_gt.yaml \
    --train_replay_storage_dir /path/to/BridgeVLA_RLBench_SEMANTIC_GT_Buffer \
    --init_checkpoint /path/to/baseline/model_80.pth \
    --train_oracle_adapter_only
```

`rlbench_o2_semantic_gt.yaml` 设置 `oracle_semantic_audit=True`；旧启发式 buffer 必须继续
使用 `rlbench_o2_gt_instance.yaml`（audit schema 默认关闭）。两类 buffer/checkpoint 不应
混在同一实验目录。semantic mapping 是 privileged GT，结果只能解释为 Oracle 上界。

<a id=o2-training></a>

### O2：训练 Target/Reference 实例 GT

> 本节导航：[推荐 Adapter + Fusion](#o2-adapter-fusion) ·
> [仅 Fusion](#o2-fusion-only) · [完整动作网络](#o2-full-action) ·
> [开启 T/R relation](#o2-relation-switch) · [同 batch loss 对比](#o2-loss-comparison) ·
> [代码插入位置](#o2-code-path) ·
> [训练可视化](#o2-training-visualization) · [代码测试](#o2-tests)

O2 不把 GT heatmap 作为固定 mask 或手工 logit 约束。主配置会同时选择唯一的
Target 与 Reference，按固定顺序组成双通道三视角 prior `[P_T, P_R]`；两组点云经过
完全相同的增强、归一化和投影，因此通道间保留当前状态下的空间关系。双通道 prior
下采样后，通过低秩 feature adapter 注入 PaliGemma 的 2048 维视觉特征。主配置还会
用共享 PointNet-style MLP 编码 T/R 完整 3D 点集，并将 pooled feature、中心、尺度和
相对位移用于 gated FiLM；随后由多尺度 residual fusion 融合 translation logits、
两个 prior 及交互项。adapter
与 fusion 输出层均为零初始化，因此训练开始时与 baseline 完全一致；Oracle
任一角色缺失或不唯一时，relation residual 整体关闭并强制回退原始路径。
推荐配置中 Adapter 输出同时进入 translation、rotation、gripper 和 collision 分支；
六项动作 loss 联合更新新增 Adapter，translation 的 logit Fusion 仍只接收 translation
梯度。原 BridgeVLA（包括原动作头）保持冻结，因此可训练规模仍约 22.4 万参数。

<a id=o2-adapter-fusion></a>

#### 推荐主实验：Relation-gated Adapter + Fusion 联合动作 loss（约 22.4 万参数）

冻结整个原 BridgeVLA，只训练新增 feature adapter 和 fusion：

```bash
bash train.sh \
    --exp_cfg_path configs/rlbench_o2_semantic_gt.yaml \
    --train_replay_storage_dir /home/yiwei/project/BridgeVLA/LPY/BridgeVLA_RLBench_SEMANTIC_GT_Buffer \
    --init_checkpoint /home/yiwei/project/BridgeVLA/LPY/BridgeVLA/checkpoints/RLBench/model_80.pth \
    --train_oracle_adapter_only
```

该配置使用双通道 Target/Reference prior、rank=16、hidden=64，并显式编码 3D
T/R relation；两阶段模型精确训练 223,878 个参数。关闭 relation-gated adapter 后，
旧 Adapter + Fusion 仍为 220,548 个参数。
主配置设置 `oracle_adapter_translation_only=False` 和 `peract.add_rgc_loss=True`：
T/R-adapted feature 同时供 translation 与 R/G/C 动作头使用，`trans_loss`、三个
rotation loss、`grip_loss` 和 `collision_loss` 都参与反向传播。原动作头虽被冻结，
梯度仍可穿过它们更新 Adapter；Fusion 只位于 translation logits 路径上。
启动时应看到 `Enable joint Oracle action losses...` 和
`Total trainable parameters: 223,878`，且 R/G/C loss 不再为零。
本次主配置已从 translation-only 改为联合动作目标；旧 O2 checkpoint 若直接 resume，
会在中途改变优化目标，不应与原曲线视为同一实验。主实验请从 baseline checkpoint
重新 `--init_checkpoint`。专用 YAML 已改用新的
`exp_id: rlbench_o2_gt_instance_joint_action`，避免日志目录混合。

如需恢复旧的 translation-only 消融，可覆盖：

```bash
--exp_cfg_opts 'oracle_adapter_translation_only True peract.add_rgc_loss False'
```

<a id=o2-fusion-only></a>

#### 最小消融：仅 Fusion

该设置只训练 logit fusion，不训练 feature adapter，不作为推荐主实验：

```bash
bash train.sh \
    --exp_cfg_path configs/rlbench_o2_semantic_gt.yaml \
    --train_replay_storage_dir /home/yiwei/project/BridgeVLA/LPY/BridgeVLA_RLBench_SEMANTIC_GT_Buffer \
    --init_checkpoint /home/yiwei/project/BridgeVLA/LPY/BridgeVLA/checkpoints/RLBench/model_80.pth \
    --train_oracle_fusion_only
```

<a id=o2-full-action></a>

#### 补充实验：完整动作网络联合训练（约 0.54B 参数）

只有确实需要完整微调动作网络、但不 fine-tune Gemma 和视觉塔时才使用：

```bash
bash train.sh \
    --exp_cfg_path configs/rlbench_o2_semantic_gt.yaml \
    --train_replay_storage_dir /home/yiwei/project/BridgeVLA/LPY/BridgeVLA_RLBench_SEMANTIC_GT_Buffer \
    --init_checkpoint /home/yiwei/project/BridgeVLA/LPY/BridgeVLA/checkpoints/RLBench/model_80.pth \
    --freeze_language_model \
    --freeze_vision_tower
```

专用配置文件为
正式配置文件为 `finetune/RLBench/configs/rlbench_o2_semantic_gt.yaml`，集中配置 semantic
audit schema、Oracle replay
shape、adapter rank、多尺度 fusion、relation 模式和 heatmap sigma。checkpoint
路径以及 `--train_oracle_adapter_only` / `--train_oracle_fusion_only` /
`--freeze_language_model` 属于单次运行策略，
仍通过命令行指定。临时修改单个值时，仍可在配置文件之后使用
`--exp_cfg_opts 'tasks stack_blocks rvt.oracle_prior_strict True'` 覆盖。

<a id=o2-relation-switch></a>

#### 开启 Target/Reference relation 输入

O2 专用配置已经默认开启双通道输入，因此使用该 YAML 时不需要额外添加命令行参数：

```yaml
oracle_relation_gated_adapter: True
oracle_adapter_translation_only: False

peract:
  add_rgc_loss: True

rvt:
  oracle_prior_relation: True
  oracle_log_base_loss: True
  oracle_valid_only_loss: False
```

如果使用其他实验配置，可在命令行显式开启：

```bash
bash train.sh \
    --exp_cfg_path configs/rlbench_config.yaml \
    --exp_cfg_opts 'oracle_relation_gated_adapter True oracle_adapter_translation_only False peract.add_rgc_loss True rvt.oracle_prior_relation True' \
    [其他训练参数]
```

如需保留双通道 heatmap、但恢复旧的无显式 3D relation Adapter，可覆盖为：

```bash
--exp_cfg_opts 'oracle_relation_gated_adapter False'
```

如需进一步恢复 Target/Reference 二选一单 prior，必须同时关闭 gated adapter：

    --exp_cfg_opts 'oracle_relation_gated_adapter False rvt.oracle_prior_relation False'

`rvt.oracle_prior_relation` 决定 Oracle 输入采用双通道 T/R 还是旧单 prior；
`oracle_relation_gated_adapter` 决定 feature adapter 是否显式编码 3D T/R relation；
`oracle_adapter_translation_only` 决定 Adapter 输出是否只进入 translation decoder；
`peract.add_rgc_loss` 决定 rotation、gripper、collision loss 是否加入总目标；
`rvt.oracle_valid_only_loss` 决定 translation 优化是否忽略不完整 T/R pair。
`--train_oracle_adapter_only` 决定冻结范围和可训练模块。这些开关作用不同，推荐主实验
同时使用 O2 专用 YAML 和 `--train_oracle_adapter_only`。

<a id=o2-loss-comparison></a>

#### Object prior 与原方法的同 batch loss 对比

O2 专用配置默认启用 `rvt.oracle_log_base_loss=True`。一次 PaliGemma 前向后，代码
复用同一组视觉 feature、GT waypoint、数据增强和动作标签，构造无梯度 baseline 支路与
可训练 O2 支路；不会为了对比重复运行 PaliGemma。`total_loss_base` 与 `total_loss`
因而是逐 batch 配对指标：

```mermaid
flowchart LR
    A[同一 replay batch<br/>图像 点云 指令 动作标签] --> B[冻结的 PaliGemma<br/>共享视觉 feature x]
    B --> C{配对分支}
    C --> D[Baseline 支路<br/>原 feature x<br/>no grad]
    C --> E[O2 支路<br/>T/R 点云和 heatmap<br/>Relation Adapter]
    D --> F[原 up0<br/>trans base]
    D --> G[原动作头<br/>rot grip collision base]
    E --> H[up0<br/>trans raw]
    H --> I[T/R Logit Fusion<br/>trans fused]
    E --> J[原动作头<br/>rot grip collision]
    F --> K[trans_loss_base]
    G --> L[R/G/C base losses]
    I --> M[trans_loss]
    J --> N[R/G/C losses]
    K --> O[total_loss_base<br/>仅监控]
    L --> O
    M --> P[total_loss<br/>反向传播]
    N --> P
    O --> Q[total_loss_gain<br/>base minus O2]
    P --> Q
    P --> R[更新 Adapter 和 Fusion<br/>原 BridgeVLA 保持冻结]
```

主配置采用相同的六项等权交叉熵：

```text
total_loss_base = trans_loss_base
                + rot_loss_x_base + rot_loss_y_base + rot_loss_z_base
                + grip_loss_base + collision_loss_base

total_loss = trans_loss
           + rot_loss_x + rot_loss_y + rot_loss_z
           + grip_loss + collision_loss

total_loss_gain = total_loss_base - total_loss
```

`total_loss_gain > 0` 表示加入 object prior 后当前 batch 的完整动作 loss 更低；
`total_loss_gain_pct` 是相对于 `total_loss_base` 的百分比。TensorBoard 会记录所有
base/O2 分量，tqdm 实时显示 `total_loss`、`total_loss_base`、gain 和 gain percentage。
总 loss 容易受三项 rotation loss 主导，因此报告结果时还应逐项对比，而不能只看总和。
这里的 `total_loss_base` 表示同一 batch 上“冻结 baseline checkpoint、不使用 object
prior”的配对诊断，不代表 baseline 又训练了相同步数。正式结果至少还要在固定验证集和
closed-loop evaluation 上比较原始 baseline checkpoint 与 O2 checkpoint；如果要进一步
排除“额外训练步数/新增参数”本身的影响，还需另设相同可训练预算的无 prior control。
当前配对指标主要用于降低训练期 batch 波动和定位收益来自哪项 loss。

完整指标如下：

| 指标 | 位置 | 是否参与反向传播 |
| --- | --- | --- |
| `total_loss_base` | Adapter 前六项动作 loss 之和 | 否，仅监控 |
| `total_loss` | O2 六项动作 loss 之和 | 是，主优化目标 |
| `total_loss_gain` / `total_loss_gain_pct` | baseline 减 O2 | 否，派生对比指标 |
| `trans_loss_base` | Adapter 前、全 batch | 否，仅监控 |
| `trans_loss_raw` | Adapter 后、Fusion 前、全 batch | 否，仅监控 |
| `trans_loss` | Adapter + Fusion 后、全 batch | 是，总目标的一部分 |
| `rot_loss_x/y/z_base` | Adapter 前的三个旋转 loss | 否，仅监控 |
| `rot_loss_x/y/z` | Adapter 后的三个旋转 loss | 是，总目标的一部分 |
| `grip_loss_base` / `collision_loss_base` | Adapter 前的离散动作 loss | 否，仅监控 |
| `grip_loss` / `collision_loss` | Adapter 后的离散动作 loss | 是，总目标的一部分 |
| `trans_loss_base_valid` | Adapter 前、仅完整 T/R pair | 否，仅监控 |
| `trans_loss_raw_valid` | Adapter 后、Fusion 前、仅完整 T/R pair | 否，仅监控 |
| `trans_loss_valid` | Adapter + Fusion 后、仅完整 T/R pair | 仅 `oracle_valid_only_loss=True` 时作为优化目标 |

Base 对比只额外执行无梯度的 `up0` 和小型动作头前向；代码在计算 base rotation
时保存并恢复 BatchNorm buffer，避免诊断支路改变训练状态。它不会建立反向图、不会
重复 PaliGemma，也不会改变模型参数。Adapter-only 下冻结动作头的 BatchNorm 还会固定
使用 checkpoint running statistics，不再因联合 loss 前向而悄悄更新 buffer。如果更重视
吞吐量、暂时不需要该诊断，可关闭：

```bash
--exp_cfg_opts 'rvt.oracle_log_base_loss False'
```

主配置使用 `rvt.oracle_valid_only_loss=False`。因为无效 pair 的 residual 已被
mask 为零，它们不会给 Adapter/Fusion 产生错误梯度；固定 batch 分母还能避免低
coverage micro-batch 被过度放大，适合当前 8-GPU + gradient accumulation 设置。
`trans_loss_valid` 仍会输出用于诊断。只有做高 coverage 的 valid-only 消融时才建议：

    --exp_cfg_opts 'rvt.oracle_valid_only_loss True'

该可选路径已按所有 DDP rank 的有效样本总数修正单个 micro-batch 的梯度归一化；
但多个 accumulation micro-batch 的有效数仍可能不同，因此不作为主实验默认值。

init_checkpoint 只初始化模型权重，adapter/fusion 保持零初始化，epoch 和 optimizer 从头开始；
继续已开始的 O2 训练则使用 resume_checkpoint。两者不能同时指定。
旧版 fusion-only checkpoint 不含 adapter/multiscale 权重，不能直接作为新版配置的
resume_checkpoint；请重新从 baseline 使用 init_checkpoint，或将 adapter rank 设为
0、关闭 multiscale fusion 后继续旧结构。

主配置中的 `rvt.oracle_prior_relation=True` 会同时输入唯一 T/R，
`oracle_prior_active_role` 在该模式下不参与选择；它仅用于兼容旧的单 prior 模式。
在线评估器可直接同时提供 `oracle_target_object_points [B,P,3]`、
`oracle_reference_object_points [B,P,3]` 及对应可选 valid，或提供完整的
`oracle_object_points/valid/roles`。relation 模式不接受旧的单个
`oracle_active_object_points` 作为有效 O2 输入，因为它无法表达 T/R 关系。训练日志同时记录
全 batch 的 trans_loss_base/trans_loss_raw/trans_loss，以及完整 pair 对应的
trans_loss_base_valid/trans_loss_raw_valid/trans_loss_valid；
fusion-only 模式下 trans_loss_raw 就是原 heatmap。oracle_prior_strict 默认为
False：T/R 任一缺失或存在多个候选时，该样本回退 trans_raw；
`oracle_target_coverage`、`oracle_reference_coverage` 和 `oracle_prior_coverage`
分别监控两个角色及完整 pair 的有效比例。数据审计时可
显式设置 rvt.oracle_prior_strict True，使异常样本直接报错。

use_oracle_objects=False、rvt.oracle_prior_mode=none 均为默认值；此时不创建
adapter/fusion 参数、不要求 Oracle 字段，旧 replay、旧 checkpoint 和原始前向路径保持
不变。O2 在训练和评估时均使用 GT 实例，属于 privileged Oracle 上界，不应作为
无 GT 的部署结果报告。

<a id=o2-code-path></a>

#### O2 代码插入位置

O2 不修改 Gemma 或视觉塔权重。feature adapter 插在 PaliGemma 视觉特征与
`up0` translation decoder 之间；多尺度 fusion 插在 translation logits 输出之后、
translation loss 和坐标 decode 之前。核心实现位于
`finetune/bridgevla/mvt/mvt_single.py::forward` 和
`finetune/bridgevla/mvt/mvt.py::_apply_oracle_instance_prior`：

    x_base = x
    x_o2 = oracle_feature_adapter(x, prior, valid, relation_points)
    trans_base = up0(x_base)                 # no grad，仅对比
    action_base = action_head(x_base)         # no grad，仅对比
    trans = up0(x_o2)
    action = action_head(x_o2)                # R/G/C loss 更新 Adapter
    raw_logits = stage_out['trans']
    stage_out['trans_raw'] = raw_logits.detach()
    stage_out['trans'] = fusion(raw_logits, prior, valid)

完整执行路径如下：

1. `finetune/RLBench/utils/dataset.py::create_replay` 注册 Oracle points、
   valid、roles 等 replay 字段；
2. `finetune/bridgevla/models/bridgevla_agent.py::_select_oracle_prior_points`
   同时选择唯一 Target 和 Reference，并固定输出顺序为 `[T,R]`；
3. `bridgevla_agent.py::update` 临时把两组 instance points 展平并拼入场景点云，使其和
   场景及动作标签执行相同 SE(3) augmentation，完成后立即拆开；
4. `mvt.py::_build_oracle_instance_prior` 将两组完整实例点分别投影成对应 stage
   坐标系的三视图 prior `[B,V,2,H,W]`；Stage 2 的 relation descriptor 也使用同一
   局部平移和缩放坐标系；
5. MVT1/MVT2 分别先由 `OracleRelationGatedFeatureAdapter` 编码完整 T/R 点集，
   以 pooled point feature、中心、尺度及相对位移生成 gated FiLM；主配置将 adapted
   feature 同时送入 translation 与 R/G/C 分支，再由 `OraclePriorFusion` 对 translation
   logits 做多尺度 residual 融合；
6. 同一 forward 内，原 feature 经过无梯度的 base decoder/action heads，adapted feature
   经过可训练 O2 路径；`bridgevla_agent.py::update` 分别计算 `total_loss_base` 和
   `total_loss`。主配置联合优化六项动作 loss，完整 T/R pair 的 `trans_loss_valid`
   仍仅作 coverage 对齐后的诊断指标。

Fusion head 定义在
`finetune/bridgevla/models/oracle_prior.py::OraclePriorFusion`。双通道主配置的
基础输入为 `[L_raw,P_T,P_R]`，多尺度分支额外接收
`[L_raw*P_T,L_raw*P_R,P_T*P_R]`；最后一层零初始化。因此启用 O2
后的初始 translation 输出与 baseline 完全相同。

正式 closed-loop 评测仍使用 `finetune/RLBench/eval.sh`。三组必须使用相同 task、demo
编号和 episode 数：

```bash
# 1. 原始 baseline checkpoint
TASKS="place_cups" MODEL_FOLDER=/path/to/baseline MODEL_NAME=model_80.pth \
EXP_CFG_PATH=configs/rlbench_config.yaml ORACLE_PROVIDER=none bash eval.sh

# 2. O2 checkpoint，但关闭 prior，测同一 checkpoint 的 raw 分支
TASKS="place_cups" MODEL_FOLDER=/path/to/o2 MODEL_NAME=model_50.pth \
EXP_CFG_PATH=configs/rlbench_o2_semantic_gt.yaml ORACLE_PROVIDER=none bash eval.sh

# 3. O2 checkpoint + simulator semantic-GT T/R fusion
TASKS="place_cups" MODEL_FOLDER=/home/yiwei/project/BridgeVLA/finetune/RLBench/train/rlbench_o2_gt_instance_joint_action/08_28_17_12 MODEL_NAME=model_last.pth \
EXP_CFG_PATH=configs/rlbench_o2_semantic_gt.yaml \
ORACLE_PROVIDER=rlbench_gt ORACLE_STRICT=1 ORACLE_DEBUG=1 bash eval.sh
```

也可以使用 O2 专用包装脚本；它默认加载 semantic-GT 配置并启用 strict/debug：

```bash
TASKS="place_cups" \
MODEL_FOLDER=/path/to/o2 MODEL_NAME=model_50.pth \
EVAL_DATAFOLDER=/path/to/BridgeVLA_RLBench_EVAL_DATA \
bash eval_o2.sh

# 同一个 O2 checkpoint 的 raw 分支对照
ORACLE_PROVIDER=none TASKS="place_cups" \
MODEL_FOLDER=/path/to/o2 MODEL_NAME=model_50.pth bash eval_o2.sh
```

`ORACLE_PROVIDER=none` 会显式关闭 O2 prior 选择，而不是依赖“字段缺失后自动回退”；
`ORACLE_PROVIDER=rlbench_gt` 会让 CoppeliaSim 直接输出四视角 one-channel handle mask，
provider 提取点云后立刻移除 mask，mask 不进入 BridgeVLA policy。O2 config 加载 checkpoint
时使用严格 `state_dict`，缺失 adapter/fusion 权重会报错，不再 `strict=False` 静默继续。
评测入口会兼容 NumPy 2 对 `uint8 * 256` 的严格溢出检查，并在解码保存的 RLBench
RGB handle mask 前转换到安全整数类型；`eval.sh` 遇到 Python 异常会立即返回非零状态，
不会再把失败任务打印为 `Completed` 或生成误导性的空汇总结果。

若同时需要模型 heatmap 可视化，运行 `eval.sh` 时设置
`VISUALIZE=1 VISUALIZE_ROOT_DIR=exp/RLBench_O2_vis`。

每个 step 的 `mvt1/` 和 `mvt2/` 目录会保存：

- `original_N.png`、`gray_N.png`、`overlay_N.png`：输入视图、最终 fused
  translation heatmap 和其最大值位置；
- `o2_target_prior_N.png`、`o2_target_prior_overlay_N.png`：GT Target prior；
- `o2_reference_prior_N.png`、`o2_reference_prior_overlay_N.png`：GT Reference prior；
- `o2_prior_N.png`、`o2_prior_overlay_N.png`：两者逐像素最大值的合并检查图；
- `o2_raw_N.png`、`o2_raw_overlay_N.png`：原始 BridgeVLA heatmap；
- `o2_fused_N.png`、`o2_fused_overlay_N.png`：融合后 heatmap。

完整路径为
`<visualize_root_dir>/<task>/episode_<N>/<language_goal>/step<N>/{mvt1,mvt2}/`。
只设置 `--visualize_root_dir` 不会启用模型可视化。semantic role 首帧 audit 由
`ORACLE_DEBUG=1` 单独控制；provider 统计和 manifest 无论是否启用图片都会写入
`<eval_log>/semantic_oracle/`。`ORACLE_STRICT=1` 约束的是 task/variation/handle 映射错误；
正确实体暂时不可见会记录 `valid=False`，无 Reference 的任务则记录 `no_reference`。

<a id=o2-training-visualization></a>

#### O2 训练中间可视化

训练样本可视化由实验 YAML 控制，默认配置关闭；O2 配置示例已开启：

    train_visualization:
      enabled: True
      interval: 500
      save_png: True
      tensorboard: True
      output_dir: train_visualizations

interval 使用 optimizer step，而不是梯度累积的 micro-step。每次只采集 rank 0
最后一个 micro-batch 的第一个样本，并分别生成 MVT1/MVT2 拼图。每张拼图包含
Input、经过当前 SE(3) 增强和投影后的 GT translation heatmap、
Target prior、Reference prior、合并 prior、Raw pred 和 Fused pred。

当 save_png=True 时，图片保存到
`<log_dir>/<output_dir>/step_XXXXXXXX/{mvt1,mvt2}.png`。当
tensorboard=True 时，必须同时使用 --log_backend tensorboard，图片显示在
TensorBoard 的 train_visualization/mvt1 和 train_visualization/mvt2 下。
两种输出可以独立关闭；可视化未命中的 step 不会拷贝训练张量到 CPU。

<a id=o2-tests></a>

#### O2 训练代码测试

在服务器的 `bridgevla` 环境、仓库根目录运行：

    python -m unittest tests.test_oracle_prior tests.test_o2_joint_action_loss tests.test_rlbench_training_utils tests.test_rlbench_training_visualization -v
    python -m pytest tests/test_o2_semantic_roles.py tests/test_replay_extra_fields.py -q

`tests.test_oracle_prior` 检查固定 `[T,R]` 选择、缺失角色回退、双通道实例点投影、adapter/fusion
零初始化 identity、无效 prior 回退、反向梯度和训练 GT/pred 张量拆分；
`tests.test_o2_joint_action_loss` 检查联合动作配置，以及无梯度 base 动作支路不会改变
BatchNorm 状态；
`tests.test_rlbench_training_utils` 检查 fusion-only、adapter-only 冻结范围和
batch/optimizer-step 规划；`tests.test_rlbench_training_visualization` 检查
PNG 与 TensorBoard 拼图输出。
`tests/test_o2_semantic_roles.py` 检查 18 任务配置覆盖、只读 RGB handle mask 解码、
多 handle 实体合并、顺序 phase 的完成/释放门控、`no_reference` 与 strict selector 错误。
`tests/test_replay_extra_fields.py` 检查 semantic audit metadata 保留在磁盘 replay 中但不会
进入训练 batch，同时缺失训练必需字段仍会立即报错。

确认专用 YAML 能被项目 YACS 配置系统加载：

    PYTHONPATH=finetune python -c "from bridgevla.config import get_cfg_defaults; c=get_cfg_defaults(); c.merge_from_file('finetune/RLBench/configs/rlbench_o2_semantic_gt.yaml'); assert c.use_oracle_objects and c.oracle_semantic_audit and c.rvt.oracle_prior_mode == 'o2_gt_instance'; print(c)"

配置打印结果还应包含 `rvt.oracle_prior_relation: True`、
`oracle_adapter_translation_only: False` 和 `peract.add_rgc_loss: True`。上述测试属于代码级检查；
正式训练前仍应在 GPU 上运行一个真实 replay batch 的 forward/backward，并确认
T/R 两项 coverage、`oracle_prior_coverage`、`total_loss_base`、`total_loss` 和
`total_loss_gain` 均能正常输出。

训练默认只在 `model_*.pth` 中保存 `epoch` 和 `model_state`，适用于评估与
推理，不保存体积较大的 Adam optimizer state。如果需要完整恢复优化器以继续
训练，启动训练时显式添加 `--save_optimizer_state`。轻量 checkpoint 仍可直接
传给 RLBench `eval.py`；使用轻量 checkpoint 执行 `--resume` 时只恢复模型
权重，优化器会重新初始化。

<a id=rlbench-training-logs></a>

### RLBench 训练日志与实时 Loss

训练日志可通过 `--log_backend` 在 TensorBoard、W&B 和关闭指标记录之间切换。
默认使用 TensorBoard，不需要账号或网络连接；无论选择哪种后端，主进程都会在
tqdm 进度条中实时显示 total、translation、rotation、gripper 和 collision loss。

默认 TensorBoard 模式等价于：

    bash train.sh [其他参数] --log_backend tensorboard

全部标量 loss 和 learning rate 会写入当前实验目录下的 `tensorboard` 子目录。

启动训练后，在另一个终端执行：

    tensorboard --logdir /path/to/experiment/tensorboard --port 6006

然后在浏览器打开 `http://localhost:6006`。如果训练运行在远程服务器，可使用
SSH 端口转发：

    ssh -L 6006:localhost:6006 user@server

默认每 10 个 iteration 额外输出一行纯文本 loss，便于保存 shell 日志。可以
调整为每 50 步输出：

    bash train.sh [其他参数] --loss_print_interval 50

使用 `--loss_print_interval 0` 可关闭纯文本 loss，但 tqdm 和选定的日志后端
仍会继续工作。TensorBoard 默认每 10 秒刷新一次，可通过
`--tensorboard_flush_secs` 调整。

需要切换回 W&B 在线记录时：

    bash train.sh [其他参数] --log_backend wandb --wandb_project BridgeVLA

可选使用 `--wandb_entity ENTITY` 指定团队或用户。服务器无法联网时，可以先写入
本地 W&B 离线目录，之后再执行 `wandb sync`：

    bash train.sh [其他参数] --log_backend wandb --wandb_mode offline

完全关闭 TensorBoard/W&B 指标记录可使用 `--log_backend none`；tqdm 实时 loss
和由 `--loss_print_interval` 控制的纯文本 loss 不受影响。

3. **COLOSSEUM Fine-tuning:** For COLOSSEUM, we fine-tune the model with the training dataset provided by the [COLOSSEUM challenge](https://huggingface.co/datasets/colosseum/colosseum-challenge/tree/main). Similarly, our training code will first convert the raw data into replay buffer. You can also directly download the replay buffer we preprocess [here](https://huggingface.co/datasets/LPY/BridgeVLA_COLOSSEUM_TRAIN_BUFFER/tree/main). Then, you can use the `finetune/Colosseum/train.sh` file to finetune the model. Please run the following code:
```bash
cd finetune/Colosseum
bash train.sh --exp_cfg_path  configs/colosseum_config.yaml \
              --exp_note debug \
              --freeze_vision_tower \
              --log_dir PATH_TO_LOG_DIR \
              --load_pretrain \
              --pretrain_path  PATH_TO_PRETRAINED_MODEL
```
4. **GemBench Fine-tuning:** To finetune on GemBench, you should first download the dataset from [here](https://huggingface.co/datasets/rjgpinel/GEMBench/tree/main). The structure of GemBench is different from RLBench and COLOSSEUM. We did not use replay buffer and did not do demo augmentation. You can use the `finetune/GemBench/train.sh` file to finetune the model. Please run the following code:
```bash
cd finetune/GemBench
bash train.sh --exp_cfg_path  configs/gembench_config.yaml \
              --exp_note debug \
              --freeze_vision_tower \
              --log_dir PATH_TO_LOG_DIR \
              --load_pretrain \
              --pretrain_path  PATH_TO_PRETRAINED_MODEL
```
<a id=evaluation></a>

## 🧪 Evaluation
1. **RLBench Evaluation:** To evaluate on RLBench, you can just run the following code:
```bash
cd finetune/RLBench
bash eval.sh # Please modify the evaluated tasks and the checkpoint path in the file.
```
2. **COLOSSEUM Evaluation:** To evaluate on COLOSSEUM, you should first preprocess the eval data as the original format is not suitable for our data loading. Run the following code to preprocess them. Or you can directly download the cleaned data we have tided from [here](https://huggingface.co/datasets/LPY/BridgeVLA_COLOSSUM_EVAL_DATA/tree/main).
```bash
cd finetune/Colosseum
python3   cleanup_script.py   LPY/COLOSSEUM_EVAL_DATA/
```
After cleaning the eval dataset, you can run the following code to evaluate the model:
```bash
cd finetune/Colosseum
bash eval.sh  VARIATION LOG_NAME MODEL_EPOCH MODEL_FOLDER
```

COLOSSEUM requires to evaluate on all the variation factors. We provide the  `Colosseum/cal_statics.py` to compute the per task success rate on each variation factor. Just replace the results folder path in the file and run the following code:
```bash
cd finetune/Colosseum
python3 cal_statics.py
```
Note: During the evaluation of Variations 1 and 6, three tasks—“close laptop lid,” “wipe desk,” and “insert onto peg”—occasionally encountered errors in certain evaluation episodes. These issues stem from problems within the evaluation data itself. I have contacted the COLOSSEUM authors, who have confirmed the issue and plan to address it in a future update. In the meantime, I adopted the following workaround: I recorded only the successful trials and repeated the evaluation until I had collected 25 successful runs for each of these tasks. These 25 valid trials were then used to compute the final performance metrics.

3. **GemBench Evaluation:** To evaluate on GemBench, you should first launch the server. Run the following code:
```bash
cd finetune/GemBench
bash run_server.sh  MODEL_EPOCH  MODEL_BASE_PATH
```
After lanuching the server, you can run the following code to evaluate the model:
```bash
cd finetune/GemBench
bash run_client.sh  SEED MODEL_EPOCH
```
The results are saved as `results.json`, which record the success status of each trial. We provide the `GemBench/cal_results.py` to compute the average success rates of each task in each setting for each seed. Just replace the results folder path in the file and run the following code:
```bash
cd finetune/GemBench
python3 cal_results.py
```
<a id=experimental-results></a>

## 📈 Experimental Results
BridgeVLA's performance on three simulation benchmarks is shown in the following table:
### RLBench Task Success Rates (Part 1)

| Model                          | Avg. Success (%) ↑ | Avg. Rank ↓ | Close Jar      | Drag Stick     | Insert Peg     | Meat off Grill | Open Drawer    | Place Cups     | Place Wine     | Push Buttons   |
|--------------------------------|--------------------|-------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|----------------|
| Image-BC (CNN)                 | 1.3                | 9.3         | 0.0            | 0.0            | 0.0            | 0.0            | 0.0            | 4.0            | 0.0            | 0.0            |
| Image-BC (ViT)                 | 1.3                | 9.7         | 0.0            | 0.0            | 0.0            | 0.0            | 0.0            | 0.0            | 0.0            | 0.0            |
| C2F-ARM-BC                     | 20.1               | 8.7         | 24.0           | 24.0           | 4.0            | 20.0           | 20.0           | 0.0            | 8.0            | 72.0           |
| HiveFormer                     | 45.3               | 6.9         | 52.0           | 76.0           | 0.0            | **100.0**      | 52.0           | 0.0            | 80.0           | 84.0           |
| PolarNet                       | 46.4               | 6.5         | 36.0           | 92.0           | 4.0            | **100.0**      | 84.0           | 0.0            | 40.0           | 96.0           |
| PerAct                         | 49.4               | 6.3         | 55.2±4.7       | 89.6±4.1       | 5.6±4.1        | 70.4±2.0       | 88.0±5.7       | 2.4±3.2        | 44.8±7.8       | 92.8±3.0       |
| Act3D                          | 65.0               | 4.3         | 92.0           | 92.0           | 27.0           | 94.0           | 93.0           | 3.0            | 80.0           | 99.0           |
| RVT                            | 62.9               | 4.4         | 52.0±2.5       | 99.2±1.6       | 11.2±3.0       | 88.0±2.5       | 71.2±6.9       | 4.0±2.5        | 91.0±5.2       | **100.0±0.0**  |
| 3D Diffuser Actor              | 81.3               | 2.5         | 96.0±2.5       | **100.0±0.0**  | 65.6±4.1       | 96.8±1.6       | 89.6±4.1       | 24.0±7.6       | 93.6±4.8       | 98.4±2.0       |
| RVT-2                          | 81.4               | 2.5         | **100.0±0.0**  | 99.0±1.7       | 40.0±0.0       | 99.0±1.7       | 74.0±11.8      | 38.0±4.5       | **95.0±3.3**   | **100.0±0.0**  |
| **BridgeVLA (Ours)**           | **88.2**           | **1.9**     | **100.0±0.0**  | **100.0±0.0**  | **88.0±2.8**   | **100.0±0.0**  | **100.0±0.0**  | **58.4±10.0**  | 88.0±2.8       | 98.4±2.2       |

### RLBench Task Success Rates (Part 2)

| Model                          | Put in Cupboard   | Put in Drawer    | Put in Safe      | Screw Bulb      | Slide Block     | Sort Shape      | Stack Blocks    | Stack Cups      | Sweep to Dustpan | Turn Tap        |
|--------------------------------|-------------------|------------------|------------------|-----------------|-----------------|-----------------|-----------------|------------------|------------------|-----------------|
| Image-BC (CNN)                 | 0.0               | 8.0              | 4.0              | 0.0             | 0.0             | 0.0             | 0.0             | 0.0              | 0.0              | 8.0             |
| Image-BC (ViT)                 | 0.0               | 0.0              | 0.0              | 0.0             | 0.0             | 0.0             | 0.0             | 0.0              | 0.0              | 16.0            |
| C2F-ARM-BC                     | 0.0               | 4.0              | 12.0             | 8.0             | 16.0            | 8.0             | 0.0             | 0.0              | 0.0              | 68.0            |
| HiveFormer                     | 32.0              | 68.0             | 76.0             | 8.0             | 64.0            | 8.0             | 8.0             | 0.0              | 28.0             | 80.0            |
| PolarNet                       | 12.0              | 32.0             | 84.0             | 44.0            | 56.0            | 12.0            | 4.0             | 8.0              | 52.0             | 80.0            |
| PerAct                         | 28.0±4.4          | 51.2±4.7         | 84.0±3.6         | 17.6±2.0        | 74.0±13.0       | 16.8±4.7        | 26.4±3.2        | 2.4±2.0          | 52.0±0.0         | 88.0±4.4        |
| Act3D                          | 51.0              | 90.0             | 95.0             | 47.0            | 93.0            | 8.0             | 12.0            | 9.0              | 92.0             | 94.0            |
| RVT                            | 49.6±3.2          | 88.0±5.7         | 91.2±3.0         | 48.0±5.7        | 81.6±5.4        | 36.0±2.5        | 28.8±3.9        | 26.4±8.2         | 72.0±0.0         | 93.6±4.1        |
| 3D Diffuser Actor              | **85.6±4.1**      | 96.0±3.6         | 97.6±2.0         | 82.4±2.0        | **97.6±3.2**    | 44.0±4.4        | 68.3±3.3        | 47.2±8.5         | 84.0±4.4         | **99.2±1.6**    |
| RVT-2                          | 66.0±4.5          | 96.0±0.0         | 96.0±2.8         | **88.0±4.9**    | 92.0±2.8        | 35.0±7.1        | **80.0±2.8**    | 69.0±5.9         | **100.0±0.0**   | 99.0±1.7        |
| **BridgeVLA (Ours)**           | 73.6±4.6          | **99.2±1.8**     | **99.2±1.8**     | 87.2±6.6        | 96.0±2.8        | **60.8±7.7**    | 76.8±8.7        | **81.6±3.6**     | 87.2±1.8         | 92.8±3.3        |

### COLOSSEUM Task Success Rates (Part 1)

| Models                  | Average ↑ | Avg. Rank ↓ | All Perturbations      | MO-COLOR          | RO-COLOR          | MO-TEXTURE        | RO-TEXTURE        | MO-SIZE          |
|-------------------------|-----------|-------------|------------------------|-------------------|-------------------|-------------------|-------------------|------------------|
| R3M-MLP                 | 0.8       | 5.71        | 0.6                   | 0.4              | 0.0               | 0.0               | 0.0              | 1.8             |
| MVP-MLP                 | 1.6       | 5.0         | 0.8                   | 1.2              | 0.0               | 0.4               | 0.0              | 4.44            |
| PerAct                  | 27.9      | 3.71        | 7.2                   | 24.0             | 29.2              | 28.8              | 17.71            | 35.6            |
| RVT                     | 35.4      | 3.28        | 6.4                   | 26.0             | 31.3              | 44.8              | 41.1             | 35.3            |
| RVT-2                   | 56.7      | 1.92        | 15.6 ± 0.8            | 53.0 ± 0.9       | 54.6 ± 0.6        | 59.7 ± 0.7        | 56.7 ± 1.4       | 60.9 ± 0.9      |
| **BridgeVLA (Ours)**    | **64.0**  | **1.07**    | **18.7 ± 2.2**        | **60.5 ± 1.1**   | **63.8 ± 0.1**    | **63.5 ± 1.5**    | **68.4 ± 3.3**   | **69.3 ± 1.0**  |

### COLOSSEUM Task Success Rates (Part 2)

| Models                  | RO-SIZE          | Light Color       | Table Color        | Table Texture      | Distractor         | Background Texture | RLBench          | Camera Pose      |
|-------------------------|------------------|-------------------|--------------------|--------------------|--------------------|--------------------|------------------|------------------|
| R3M-MLP                 | 0.0              | 1.0               | 1.4                | 0.2                | 1.6                | 1.2                | 2.0              | 0.8              |
| MVP-MLP                 | 0.0              | 1.6               | 1.6                | 1.0                | 3.8                | 2.2                | 2.0              | 2.6              |
| PerAct                  | 29.3             | 29.1              | 30.4               | 23.2               | 27.1               | 33.5               | 39.4             | 36.3             |
| RVT                     | 40.5             | 34.0              | 30.0               | 45.2               | 18.8               | 46.4               | 53.4             | 42.2             |
| RVT-2                   | 53.4 ± 1.5       | 58.0 ± 1.1        | 62.6 ± 0.9         | 56.6 ± 0.9         | **60.8 ± 0.5**     | 68.7 ± 1.1         | 68.8 ± 1.3       | 64.4 ± 0.5       |
| **BridgeVLA (Ours)**    | **61.7 ± 0.8**   | **69.7 ± 1.2**    | **75.7 ± 0.9**     | **71.3 ± 0.7**     | 51.8 ± 1.5         | **74.8 ± 1.0**     | **73.1 ± 0.2**   | **73.8 ± 0.3**   |

### Performance on GemBench

| Method                         | Avg.  | L1              | L2              | L3              | L4             |
|--------------------------------|-------|-----------------|-----------------|-----------------|----------------|
| Hiveformer                     | 30.4  | 60.3±1.5        | 26.1±1.4        | 35.1±1.7        | 0.0±0.0        |
| PolarNet                       | 38.4  | 77.7±0.9        | 37.1±1.4        | 38.5±1.7        | 0.1±0.2        |
| 3D diffuser actor              | 44.0  | 91.9±0.8        | 43.4±2.8        | 37.0±2.2        | 0.0±0.0        |
| RVT-2                          | 44.0  | 89.1±0.8        | 51.0±2.3        | 36.0±2.2        | 0.0±0.0        |
| 3D-LOTUS                       | 45.7  | **94.3±1.4**    | 49.9±2.2        | 38.1±1.1        | 0.3±0.3        |
| 3D-LOTUS++                     | 48.0  | 68.7±0.6        | 64.5±0.9        | 41.5±1.8        | **17.4±0.4**   |
| **BridgeVLA (Ours)**           | **50.0** | 91.1±1.1        | **65.0±1.3**    | **43.8±1.2**    | 0.0±0.0        |


<a id=todo></a>

## 📅 TODO 

- [x] Release the pre-training code
- [x] Release the training & evaluation code of RLBench
- [x] Release the training & evaluation code of COLOSSEUM
- [x] Release the training & evaluation code of GemBench
- [x] Release the pre-training data
- [x] Release the checkpoints

</details>

<a id=acknowledgement></a>

## 🙏 Acknowledgement
We stand on the shoulders of giants, and our work in developing BridgeVLA has been inspired and empowered by the remarkable open source projects in the field. We would like to extend our heartfelt gratitude to each of these initiatives and their dedicated developers.
- [PerAct](https://peract.github.io/)
- [RVT-2](https://robotic-view-transformer-2.github.io/)
- [Palligemma](https://huggingface.co/blog/paligemma)
- [RLBench](https://github.com/stepjam/RLBench/tree/master)
- [GemBench](https://www.di.ens.fr/willow/research/gembench/)
- [COLOSSEUM](https://robot-colosseum.github.io/)
- [RoboPoint](https://github.com/wentaoyuan/RoboPoint)

<a id=contact></a>

## ✉️ Contact
If you have any questions about the code, please contact peiyan.li@cripac.ia.ac.cn.
<a id=citation></a>

## 📝 Citation

```bibtex
@misc{li2025bridgevla,
    title={BridgeVLA: Input-Output Alignment for Efficient 3D Manipulation Learning with Vision-Language Models},
    author={Peiyan Li and Yixiang Chen and Hongtao Wu and Xiao Ma and Xiangnan Wu and Yan Huang and Liang Wang and Tao Kong and Tieniu Tan},
    year={2025},
    eprint={2506.07961},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```


