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

- [Model Overview](#Model-Overview)
- [Installation](#Installation)
- [Training](#Training)
- [Evaluation](#evaluation)
- [Experimental Results](#experimental-results)
- [TODO](#TODO)
- [Acknowledgement](#Acknowledgement)
- [Contact](#Contact)
- [Citation](#Citation)


## 📋 Model Overview
As illustrated in the following figure, BridgeVLA employs a dual-phase training recipe. During pre-training, it is trained to predict 2D heatmaps on object detection datasets. During fine-tuning, point clouds are projected into multiple 2D images as inputs to the VLM backbone. The model is trained to predict 2D heatmaps for estimating the translational action and other action components. **This design aligns the input and output within a shared 2D space in both pre-training and fine-tuning.**
![](./assets/network.png)
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
### RLBench Oracle 3D 物体 Replay 数据准备

为了进行 BridgeVLA + Oracle 3D Object Tokens 实验，可以在不重新生成原始
BridgeVLA replay buffer 的前提下，为已有的逐 transition replay 文件追加 GT
物体点云。工具位于 tools/augment_replay_with_oracle_objects.py。

每个 N.replay 都是一个独立的 pickle transition 字典。脚本使用其中的
episode_idx 和 sample_frame 定位当前观测对应的原始 RLBench 帧：

    N.replay[episode_idx] + N.replay[sample_frame]
        -> episode{episode_idx}/{camera}_mask/{sample_frame}.png
        -> RLBench rgb_handles_to_mask 解码实例 ID
        -> 与 N.replay[{camera}_point_cloud] 按像素对齐
        -> 跨 front、left_shoulder、right_shoulder、wrist 相机合并同一实例
        -> 固定尺寸 Oracle 物体点云

追加的字段为：

    oracle_object_points   # [MAX_OBJECTS, NUM_POINTS, 3], float32
    oracle_object_centers  # [MAX_OBJECTS, 3], float32
    oracle_object_sizes    # [MAX_OBJECTS, 3], float32
    oracle_object_ids      # [MAX_OBJECTS], int32
    oracle_object_valid    # [MAX_OBJECTS], bool

sample_frame 表示当前 replay observation 的原始帧。keypoint_frame 表示前一个
关键帧，Oracle 对齐不会使用它。terminal == -1 的文件是 YARR 的最终观测
sentinel，其 episode/frame 元数据未定义，因此脚本只为它写入全 invalid 的填充
张量。

处理 stack_blocks：

    python tools/augment_replay_with_oracle_objects.py \
        --replay-dir LPY/BridgeVLA_RLBench_TRAIN_Buffer \
        --raw-data-dir LPY/BridgeVLA_RLBench_TRAIN_DATA/train \
        --task stack_blocks \
        --output-dir LPY/BridgeVLA_RLBench_ORACLE_Buffer \
        --max-objects 32 \
        --num-points 512 \
        --min-object-points 20 \
        --workers 8 \
        --cache-frames 256

处理 replay 根目录下的全部任务：

    python tools/augment_replay_with_oracle_objects.py \
        --replay-dir LPY/BridgeVLA_RLBench_TRAIN_Buffer \
        --raw-data-dir LPY/BridgeVLA_RLBench_TRAIN_DATA/train \
        --task all \
        --output-dir LPY/BridgeVLA_RLBench_ORACLE_Buffer

建议首先执行 dry-run。该模式会随机选择若干 transition，打印 task、replay
index、episode_idx、sample_frame、实例 ID、物体中心、点数、张量形状及有限值
检查结果，但不会写入文件：

    python tools/augment_replay_with_oracle_objects.py \
        --replay-dir LPY/BridgeVLA_RLBench_TRAIN_Buffer \
        --raw-data-dir LPY/BridgeVLA_RLBench_TRAIN_DATA/train \
        --task stack_blocks \
        --dry-run \
        --dry-run-samples 5

可视化指定 transition：

    python tools/augment_replay_with_oracle_objects.py \
        --replay-dir LPY/BridgeVLA_RLBench_TRAIN_Buffer \
        --raw-data-dir LPY/BridgeVLA_RLBench_TRAIN_DATA/train \
        --task stack_blocks \
        --dry-run \
        --visualize-index 100 \
        --visualize-output-dir oracle_visualizations

可视化不再依赖图形界面弹窗，而是保存为 PNG。上述命令生成：

    oracle_visualizations/stack_blocks_replay_100.png

若不传 `--visualize-output-dir`，默认也会保存到当前工作目录下新建的
`oracle_visualizations` 文件夹；目标文件夹不存在时会自动创建。
每张 PNG 会从同一个 raw `episode_idx/sample_frame` 读取
`front_rgb`、`left_shoulder_rgb`、`right_shoulder_rgb` 和 `wrist_rgb`，因此可以直接
对照不同相机中的原始场景与筛选后的三维实例。某个 RGB 文件缺失时只在对应面板
显示 `RGB unavailable`，不会中断 replay 生成。

按排序后的 replay 文件间隔绘制整个任务目录，例如每 100 个文件保存一张：

    python tools/augment_replay_with_oracle_objects.py \
        --replay-dir LPY/BridgeVLA_RLBench_TRAIN_Buffer \
        --raw-data-dir LPY/BridgeVLA_RLBench_TRAIN_DATA/train \
        --task stack_blocks \
        --dry-run \
        --visualize-every 100 \
        --visualize-output-dir oracle_visualizations

配合 `--dry-run` 时，只计算并检查这些间隔抽样文件，不会写入 replay。去掉
`--dry-run` 时仍会迁移整个 replay 目录，并在处理过程中每隔 100 个文件保存一张
图片。`--visualize-index` 与 `--visualize-every` 不能同时使用。

默认行为会生成新的 Oracle replay 副本，不会覆盖原始文件。每个 replay 先写入
N.replay.tmp，重新加载并验证原字段及 Oracle 字段后，再原子重命名。已有输出
默认会被拒绝；只有显式传入 --overwrite 才会替换输出。若确实需要修改原目录，
必须显式使用 --in-place。默认不对每个文件调用 fsync，以避免网络盘上的严重
性能损失；如需要每个临时文件在重命名前强制落盘，可显式添加 --durable-write。
处理过程中默认显示每个任务的 tqdm 进度、速度和预计剩余时间；后台运行时可用
--no-progress 关闭进度条。可通过 --workers N 使用线程池并行处理不同 replay
文件，例如 --workers 8。线程数过高会增加内存占用和网络盘 I/O 竞争，建议从
4 或 8 开始测试；默认值 1 保持原来的串行行为。

脚本默认使用 --cache-frames 128 缓存最近完成的原始帧 Oracle 结果，上面的推荐
命令将容量提高到 256。缓存键为 task、episode_idx 和 sample_frame；多个 replay
transition 指向同一帧时，会复用 mask 解码、跨视角融合和点采样结果。多线程
同时请求同一帧时也只计算一次。进度条中的 cache_hits 和 cache_misses 可用于
确认缓存效果。可根据可用内存调整容量，或使用 --cache-frames 0 禁用缓存。同一
原始帧的随机采样由 task、episode_idx、sample_frame 和 --seed 决定，不受
replay index 或线程完成顺序影响。

oracle_object_sizes 保存融合点云的轴对齐包围盒尺寸，即每个坐标轴上的
max(points) - min(points)。脚本默认使用 --min-object-points 20，实例在跨相机
合并并移除 NaN/Inf 后少于 20 个点时会被过滤。进度条中的 filtered 为累计过滤
实例数，dry-run 会打印当前帧被过滤的数量。若希望保留此前的最宽松行为，可使用
--min-object-points 1。

RLBench mask 是 RGB 编码的 simulator handle，不能直接把 RGB 像素值当作实例
ID。脚本调用 RLBench 自带的 rgb_handles_to_mask 解码，仅默认移除已确认的背景
ID 0。离线数据没有提供 handle 到物体名称的映射，因此不会自动移除机器人；
如已知某个 handle，可重复传入 --exclude-object-id。

### 生成 Oracle 时同步执行任务先验筛选

无需重新运行 RLBench 或重新生成原始 replay。添加 `--task-prior-filter` 后，脚本会在
GT mask 与点云融合完成、写入 `oracle_object_*` 之前，根据当前 transition 中监督的
下一关键动作 `gripper_pose[:3]` 对实例排序。18 个 BridgeVLA 任务的默认动作类型、
交互半径和最大 handle 数定义在 `tools/rlbench_task_object_priors.py`。明显覆盖两个大
坐标轴的桌面/地面会作为背景移除；其余实例按点云包围盒到动作位置的距离排序。
默认采用高召回模式，距离超过交互半径的实例不会仅因此被删除，避免多物体任务中
只保留下一步动作附近的一个物体。

    python tools/augment_replay_with_oracle_objects.py \
        --replay-dir LPY/BridgeVLA_RLBench_TRAIN_Buffer \
        --raw-data-dir LPY/BridgeVLA_RLBench_TRAIN_DATA/train \
        --task stack_blocks \
        --output-dir LPY/BridgeVLA_RLBench_TASK_OBJECT_Buffer \
        --task-prior-filter \
        --min-object-points 1 \
        --max-objects 32 \
        --num-points 512 \
        --workers 8 \
        --visualize-objects-only \
        --cache-frames 256

    python tools/augment_replay_with_oracle_objects.py \
        --replay-dir LPY/BridgeVLA_RLBench_TRAIN_Buffer \
        --raw-data-dir LPY/BridgeVLA_RLBench_TRAIN_DATA/train \
        --task stack_blocks \
        --detect-robot-handles \
        --refresh-robot-handle-cache \
        --robot-detection-frames 8 \
        --robot-handle-cache-dir robot_handle_maps \
        --task-prior-filter \
        --min-object-points 1 \
        --max-objects 32 \
        --visualize-index 100 \
        --visualize-objects-only \
        --dry-run

建议先配合 `--dry-run --visualize-every N` 检查结果。可使用
`--task-prior-radius`、`--task-prior-max-instances` 和
`--task-prior-background-extent` 覆盖默认值。进度条和 dry-run 输出中的
`prior_filtered` 表示被任务先验删除的 handle 数。如确实需要旧式的激进空间筛选，
显式添加 `--task-prior-strict`；此时才会删除交互半径之外的 handle，并使用任务默认
的最大 handle 数。也可以只传 `--task-prior-max-instances N` 主动限制高召回结果。

dry-run 会同时打印 `excluded_object_ids`、`no_finite_point_object_ids`、
`small_object_ids`、`task_prior_filtered_object_ids` 和 `truncated_object_ids`，分别对应
ID/机械臂排除、mask 可见但点云全为 NaN/Inf、点数阈值、任务先验以及
`--max-objects` 截断。若某个 handle 不在这些列表，也不在当前帧保留 ID 中，说明它
在所选相机的当前帧 GT mask 中不可见，而不是过滤器删除。

若只想查看筛选后保留的彩色 object，不显示灰色完整场景点云，可添加：

    --visualize-objects-only

该参数只影响保存的 PNG，不修改 `oracle_object_*` 或其他 replay 字段。
可视化颜色由解码后的 handle ID 确定，因此同一 episode 中相同 handle ID 在所有
帧里始终使用相同颜色，不再随 slot 或绘制顺序变化。图片标题同时显示
`episode_idx` 和 `sample_frame`，用于避免误将不同 episode 的颜色或 ID 作比较。
所有 PNG 固定使用 BridgeVLA RLBench 的米制场景边界
`[-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]`，并保持 xyz 等比例显示；因此跨帧不会因
Matplotlib 自动缩放而改变物体的视觉尺度。该设置只影响绘图，不缩放或修改保存的
Oracle 点云。

每张 PNG 使用两排八个面板：上排依次显示 front、left shoulder、right shoulder、
wrist 四个 RGB 相机视角；下排显示同一批点云的 3D 透视图，以及三个正交投影：
俯视 `XY`（沿 Z 轴观察）、正视 `XZ`（沿 Y 轴观察）和侧视 `YZ`（沿 X 轴观察）。
四个点云视角共享固定场景尺度和 handle ID 颜色；`--visualize-objects-only` 只隐藏
下排点云中的灰色完整场景，不会隐藏上排 RGB 图像。

离线 GT mask 只保存数字 handle，没有 `handle -> object name` 语义映射，因此不能
仅凭任务名称严格证明某个 handle 是机械臂。确认机器人 ID 后，可重复使用
`--exclude-robot-id ID`（它是 `--exclude-object-id` 的别名）进行精确排除：

    --exclude-robot-id 23 --exclude-robot-id 24 --exclude-robot-id 25

该模式只筛选保留的 GT handle，不生成 target/reference 角色标签；默认不传
`--task-prior-filter` 时仍保存全部有效 GT instance，与原 Oracle 实验保持兼容。

注意：这里使用的 `gripper_pose` 是下一关键动作的监督值，因此该自动模式属于
**action-conditioned Oracle 上界/离线标注**，不能直接作为无标签推理阶段的公平
筛选器。若要求完全不使用未来动作，需要额外提供离线 `handle -> object name` 或
人工 handle 白名单；只有数字 mask ID 和任务名称时无法无歧义地完成该语义映射。

#### Episode 级时序任务 handle 白名单

若高召回模式仍保留无关实例，添加 `--temporal-task-filter`。脚本会为每个 episode
抽取最多 `--task-detection-frames` 帧，将 raw observation 中的当前夹爪位置与 replay
中的下一关键动作位置组成完整动作轨迹，然后生成一份供该 episode 所有帧共用的
handle 白名单：

- 静止物体只要曾靠近抓取、放置或操作位置就会保留；不要求它发生运动。
- 推动、按压、拉动等未闭合夹爪的交互，只要移动期间靠近夹爪或动作位置也会保留。
- 多帧显著移动但移动前后始终没有任何夹爪/动作邻近证据的 handle 会记录在
  `rejected_dynamic_handles`，不会进入白名单。
- 与已确认交互 handle 持续相邻的静态目标、容器或组合结构额外保留一跳；不会递归
  扩散到整个场景。
- `robot_handles` 会在时序任务检测之前排除，因此建议两个检测参数同时启用。

    python tools/augment_replay_with_oracle_objects.py \
        --replay-dir LPY/BridgeVLA_RLBench_TRAIN_Buffer \
        --raw-data-dir LPY/BridgeVLA_RLBench_TRAIN_DATA/train \
        --task stack_blocks \
        --output-dir LPY/BridgeVLA_RLBench_TASK_OBJECT_Buffer \
        --detect-robot-handles \
        --robot-detection-frames 8 \
        --robot-handle-cache-dir robot_handle_maps \
        --temporal-task-filter \
        --task-detection-frames 16 \
        --task-handle-cache-dir task_handle_maps \
        --task-prior-filter \
        --min-object-points 1 \
        --max-objects 32 \
        --visualize-every 100 \
        --visualize-objects-only

时序结果缓存为 `task_handle_maps/<task>/episode_NNNN.json`。JSON 包含
`task_handles`、`interaction_handles`、`adjacent_handles`、
`rejected_dynamic_handles`、`background_handles` 和 `sampled_frames`。修改任务半径、
最大实例数或检测帧数后，应添加 `--refresh-task-handle-cache` 重新检测。dry-run 会
读取缓存但不会写入新缓存。运行进度及 dry-run 中的 `temporal_filtered` /
`temporal_filtered_object_ids` 表示当前帧被 episode 白名单删除的 handle。

该功能不调用 Qwen、SAM，也不重新运行 RLBench；它直接在生成 Oracle replay 前使用
现有 GT mask、点云和动作轨迹完成离线筛选。若 episode 很长或存在很多轮操作，可将
`--task-detection-frames` 提高到 24 或 32，以降低漏过短暂交互的概率。

### 自动检测并排除机械臂和夹爪

不需要重新运行或重新采集 RLBench。添加 `--detect-robot-handles` 后，脚本会在生成
Oracle 前对每个 episode 最多抽取若干个时间上均匀分布的 replay 帧，并读取原始
`low_dim_obs.pkl[sample_frame].gripper_pose` 中的**当前**夹爪位置。检测器只把首个
采样帧就在 wrist mask 中可见、靠近夹爪，并且跨帧保持稳定“物体中心－夹爪位置”
偏移的 handle 作为夹爪种子；机械臂 link 除了需要持续运动和相邻，还必须在 episode
早期就已和夹爪/机械臂链连接。因此只在抓取后才靠近或随机械臂运动的任务物体，
不会被误加入 `robot_handles`。

    python tools/augment_replay_with_oracle_objects.py \
        --replay-dir LPY/BridgeVLA_RLBench_TRAIN_Buffer \
        --raw-data-dir LPY/BridgeVLA_RLBench_TRAIN_DATA/train \
        --task stack_blocks \
        --output-dir LPY/BridgeVLA_RLBench_TASK_OBJECT_Buffer \
        --detect-robot-handles \
        --robot-detection-frames 8 \
        --robot-handle-cache-dir robot_handle_maps \
        --task-prior-filter \
        --min-object-points 1 \
        --visualize-every 100 \
        --visualize-objects-only

检测结果按 task/episode 缓存，例如：

    robot_handle_maps/stack_blocks/episode_0000.json

JSON 中保存 `gripper_handles`、`arm_handles`、合并后的 `robot_handles`、置信度和
参与检测的 `sampled_frames`。后续运行默认直接复用缓存；需要重新检测时添加
`--refresh-robot-handle-cache`。dry-run 可以读取已有缓存，但不会新建或覆盖缓存。
自动结果仍可与重复传入的 `--exclude-robot-id ID` 精确黑名单同时使用。
检测缓存带有算法版本；旧版 `wrist_pose_temporal_adjacency_v1` JSON 会自动失效，
对应 episode 将按新版规则重新检测并在非 dry-run 时覆盖为 v2，不需要手动删除。

检测启动阶段会优先读取 replay 目录中的轻量 `replay_info.npy` 划分 episode，不再
为分组而打开全部大型 `.replay`；每个未缓存 episode 只读取最多
`--robot-detection-frames` 个 replay。已有 JSON 的 episode 会跳过帧加载。dry-run
只检测本次抽样/可视化涉及的 episode。若旧目录没有有效 `replay_info.npy`，脚本会
回退为逐 replay 元数据扫描，并显示 `robot detection: replay metadata scan` 进度条。

该检测不使用 replay 中的下一动作 `gripper_pose`，因此不会引入未来动作标签；它
读取的是 raw episode 中当前观测帧的夹爪位姿。由于这是无对象名称的几何时域
启发式，建议先用 `--dry-run --visualize-every N --visualize-objects-only` 检查各任务
的 JSON 和可视化，再处理完整训练集。

如果场景中的有效实例多于 max-objects，脚本优先保留点数最多的实例；少于
num-points 的实例会有放回采样，多于该数量时无放回采样。所有 NaN 和 Inf 点均
会在采样前移除。

训练加载 Oracle replay 时，需要将
finetune/RLBench/utils/peract_utils_rlbench.py 中的
TRAIN_REPLAY_STORAGE_DIR 指向 Oracle 输出目录，并启用与数据准备阶段一致的张量
尺寸：

    bash train.sh \
        --exp_cfg_opts 'use_oracle_objects True oracle_max_objects 32 oracle_num_points 512' \
        [其他训练参数]

use_oracle_objects 默认为 False，因此原始非 Oracle replay 的加载行为保持不变。
当前改动负责准备和加载 Oracle 字段，不包含将这些字段转换为 object tokens 的
策略网络结构。

训练默认只在 `model_*.pth` 中保存 `epoch` 和 `model_state`，适用于评估与
推理，不保存体积较大的 Adam optimizer state。如果需要完整恢复优化器以继续
训练，启动训练时显式添加 `--save_optimizer_state`。轻量 checkpoint 仍可直接
传给 RLBench `eval.py`；使用轻量 checkpoint 执行 `--resume` 时只恢复模型
权重，优化器会重新初始化。

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


## 📅 TODO 

- [x] Release the pre-training code
- [x] Release the training & evaluation code of RLBench
- [x] Release the training & evaluation code of COLOSSEUM
- [x] Release the training & evaluation code of GemBench
- [x] Release the pre-training data
- [x] Release the checkpoints

</details>

## 🙏 Acknowledgement
We stand on the shoulders of giants, and our work in developing BridgeVLA has been inspired and empowered by the remarkable open source projects in the field. We would like to extend our heartfelt gratitude to each of these initiatives and their dedicated developers.
- [PerAct](https://peract.github.io/)
- [RVT-2](https://robotic-view-transformer-2.github.io/)
- [Palligemma](https://huggingface.co/blog/paligemma)
- [RLBench](https://github.com/stepjam/RLBench/tree/master)
- [GemBench](https://www.di.ens.fr/willow/research/gembench/)
- [COLOSSEUM](https://robot-colosseum.github.io/)
- [RoboPoint](https://github.com/wentaoyuan/RoboPoint)

## ✉️ Contact
If you have any questions about the code, please contact peiyan.li@cripac.ia.ac.cn.
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


