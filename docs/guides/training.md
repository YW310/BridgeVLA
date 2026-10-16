[文档索引](../README.md) · [项目首页](../../README.md)

> 命令不在 docs/ 下执行。带 cd 的独立示例从仓库根目录开始；其后命令沿用该目录。替换所有示例路径后再运行。

<a id=training></a>

# Training

If you want to reproduce our results, please use the same training hyperparameters in the config file. **Do not forget to modify the corresponding saving path in the file before running the following code.**

## Pre-training

We use the object detection data in the RoboPoint dataset to pre-train the model. We upload the data and checkpoints [here](https://huggingface.co/datasets/LPY/BridgeVLA/tree/main/). With the `pretrain/pretrain.py` file, you can do three things:

| `--branches` | Behavior |
| --- | --- |
| `1` | Visualize the pre-training dataset |
| `2` | Pre-train PaliGemma; vision tower is frozen by this entry |
| `3` (default) | Test the pre-trained checkpoints |

Choose the branch explicitly; no source edit is needed for branch selection:

```bash
bash pretrain/pretrain.sh --branches BRANCH_OPTION --config_path PATH_TO_CONFIG_FILE --json_detection_path PATH_TO_DETECTION_JSON --image_folder PATH_TO_IMAGE_FOLDER
```

## RLBench Fine-tuning

Prepare the dataset using [PerAct data generation](https://github.com/peract/peract?tab=readme-ov-file#data-generation), or download the [raw demonstrations](https://huggingface.co/datasets/LPY/BridgeVLA_RLBench_TRAIN_DATA/tree/main) / [prebuilt replay](https://huggingface.co/datasets/LPY/BridgeVLA_RLBench_TRAIN_BUFFER/tree/main). The standard training entry can convert raw data when replay is missing; for distributed runs, prepare it first with the [standalone replay generator](replay.md).

Run from the repository root after configuring the simulator environment in [installation](installation.md).
Use `train_8x40.sh` for any single-node GPU count; its name does not require 8 GPUs.
The old RLBench `train.sh` hardcodes server paths and two GPUs, and forwards unquoted `$@`,
which splits `--exp_cfg_opts` and paths containing spaces. It is not the recommended entry.

The epoch budget comes from the YAML and `--exp_cfg_opts` unless `--epochs` is explicitly supplied.
An explicit `--epochs` overrides that budget and is recorded in the saved configuration; on resume it remains the total target epoch count.
Older runs silently used the CLI default of 100 even when their YAML said 50. To continue such a run to its original target, supply `--epochs 100` explicitly.

```bash
cd finetune/RLBench
GPUS_PER_NODE=2 bash train_8x40.sh --exp_cfg_path configs/rlbench_config.yaml \
              --exp_note debug \
              --freeze_vision_tower \
              --log_dir exp/RLBench \
              --load_pretrain \
              --pretrain_path PATH_TO_2D_HEATMAP_PRETRAINED_MODEL
```

<a id="rlbench-config-datasets"></a>

### RLBench config 与数据目录

配置位于 `finetune/RLBench/configs/`，决定所需字段；实际 replay 根目录由 `--train_replay_storage_dir` 指定，不按配置名自动选择。O2 必须显式传入；baseline 也建议显式指定，避免使用旧服务器默认路径。

| Config（省略 `rlbench_` 前缀及 `.yaml`） | 对应 replay 目录 |
| --- | --- |
| `config`、`config_small`、`full_8x40`、`trend_8x40` | `BridgeVLA_RLBench_TRAIN_Buffer`；调试可用 `BridgeVLA_RLBench_TINY_Buffer` |
| `o2_gt_instance` | `BridgeVLA_RLBench_TRAIN_TASK_OBJECT_Buffer`；调试可用 `BridgeVLA_RLBench_TINY_TASK_OBJECT_Buffer`，T/R 为启发式角色 |
| `o2_semantic_gt`、`o2_semantic_gt_joint`、`o2_semantic_gt_relation_anchor` | `BridgeVLA_RLBench_SEMANTIC_GT_MASK_VERIFIED_Buffer`，需通过语义契约验收 |
| `o2_internal_slots` | 推荐同一 Semantic-GT buffer；旧配置未强制语义审计，也允许启发式物体 buffer，但监督质量不同 |
| `o2_internal_slots_joint`、`o2_internal_slots_cross_scale`、`o2_role_queries` | 同一 Semantic-GT buffer，需通过语义契约验收；GT 仅用于辅助监督 |
| `o2_predicted_objects` | 另行准备含 `predicted_*` 字段的外部预测 replay；不能直接以 Oracle 字段替代 |
| `o2_semantic_roles` | 不是训练配置，是 T/R 角色定义与契约文件 |

`config_small` 不自动绑定 TINY 数据。按用户提供的目录快照（2026-10-16），TINY buffers 仍有 18 个任务目录，实际 episode/transition 数需统计；`BridgeVLA_RLBench_TASK_OBJECT_Buffer` 只有 `close_jar`，仅用于对应单任务实验（设置 `tasks=close_jar`）。目录存在不代表已完整生成或验收通过。

`BridgeVLA_RLBench_TRAIN_DATA/train` 是 raw → replay 的输入，`BridgeVLA_RLBench_EVAL_DATA` 用于闭环评估；`BridgeVLA/checkpoints` 用于初始化，`BridgeVLA/pretrain_data` 用于预训练，`semantic_role_visualizations` 不参与训练。生成与验收见 [标准 Replay](replay.md)、[启发式实例增强](oracle-replay.md)及 [Semantic-GT](semantic-gt.md)；训练／预测输入隔离见 [O2 指南](object-conditioning.md#数据与角色契约)。

<a id=rlbench-8x40></a>

### RLBench fine-tuning on one 8 x 40 GB node

The reduced-hardware profiles keep the paper-scale effective batch size of
192 with gradient accumulation. The current trend YAML uses tasks: all and
batch size 4 per GPU. Check the configuration and launch logs for the actual batch plan.

Run the trend profile (20,000 optimizer steps; all tasks by default):

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

Optimizer-continuous resume requires `--save_optimizer_state`; otherwise weights resume with a fresh optimizer and a warning.
Checkpoints do not restore a full RNG/loader snapshot, so this is not a bit-for-bit continuation.

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
It currently exposes no Oracle provider, O2 config override, or episode-resume flags;
use [O2 evaluation](object-conditioning.md#closed-loop评估) for those runs.

<a id=rlbench-training-logs></a>

### RLBench 训练日志与实时 Loss

训练日志可通过 `--log_backend` 在 TensorBoard、W&B 和关闭指标记录之间切换。
默认使用 TensorBoard，不需要账号或网络连接；无论选择哪种后端，主进程都会在
tqdm 进度条中实时显示 total、translation、rotation、gripper 和 collision loss。

默认 TensorBoard 模式等价于：

    GPUS_PER_NODE=2 bash train_8x40.sh [其他参数] --log_backend tensorboard

全部标量 loss 和 learning rate 会写入当前实验目录下的 `tensorboard` 子目录。

启动训练后，在另一个终端执行：

    tensorboard --logdir /path/to/experiment/tensorboard --port 6006

然后在浏览器打开 `http://localhost:6006`。如果训练运行在远程服务器，可使用
SSH 端口转发：

    ssh -L 6006:localhost:6006 user@server

默认每 10 个 iteration 额外输出一行纯文本 loss，便于保存 shell 日志。可以
调整为每 50 步输出：

    GPUS_PER_NODE=2 bash train_8x40.sh [其他参数] --loss_print_interval 50

使用 `--loss_print_interval 0` 可关闭纯文本 loss，但 tqdm 和选定的日志后端
仍会继续工作。TensorBoard 默认每 10 秒刷新一次，可通过
`--tensorboard_flush_secs` 调整。

需要切换回 W&B 在线记录时：

    GPUS_PER_NODE=2 bash train_8x40.sh [其他参数] --log_backend wandb --wandb_project BridgeVLA

可选使用 `--wandb_entity ENTITY` 指定团队或用户。服务器无法联网时，可以先写入
本地 W&B 离线目录，之后再执行 `wandb sync`：

    GPUS_PER_NODE=2 bash train_8x40.sh [其他参数] --log_backend wandb --wandb_mode offline

完全关闭 TensorBoard/W&B 指标记录可使用 `--log_backend none`；tqdm 实时 loss
和由 `--loss_print_interval` 控制的纯文本 loss 不受影响。

## COLOSSEUM Fine-tuning

COLOSSEUM/GemBench launchers still contain path/GPU/port placeholders and unquoted `$@`.
Adapt the scripts before use; multi-word overrides need quoted forwarding, or call `torchrun ... train.py` directly.
These launchers are not covered by the portable RLBench wrapper.

Use the [COLOSSEUM training dataset](https://huggingface.co/datasets/colosseum/colosseum-challenge/tree/main) or the [prebuilt replay](https://huggingface.co/datasets/LPY/BridgeVLA_COLOSSEUM_TRAIN_BUFFER/tree/main). From the repository root, after adapting the launcher:

```bash
cd finetune/Colosseum
bash train.sh --exp_cfg_path  configs/colosseum_config.yaml \
              --exp_note debug \
              --freeze_vision_tower \
              --log_dir PATH_TO_LOG_DIR \
              --load_pretrain \
              --pretrain_path  PATH_TO_PRETRAINED_MODEL
```
## GemBench Fine-tuning

Download the [GemBench dataset](https://huggingface.co/datasets/rjgpinel/GEMBench/tree/main). This branch does not use RLBench replay or its demo augmentation. From the repository root, after adapting the launcher:

```bash
cd finetune/GemBench
bash train.sh --exp_cfg_path  configs/gembench_config.yaml \
              --exp_note debug \
              --freeze_vision_tower \
              --log_dir PATH_TO_LOG_DIR \
              --load_pretrain \
              --pretrain_path  PATH_TO_PRETRAINED_MODEL
```

> 预训练前还需设置 pretrain/pretrain.sh 内的端口、GPU 数和节点数；该脚本自身包含 cd pretrain，请从仓库根目录执行 bash pretrain/pretrain.sh 并传入上述参数。
