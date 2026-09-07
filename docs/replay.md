[文档索引](README.md) · [项目首页](../README.md)

> 命令不在 docs/ 下执行。带 cd 的独立示例从仓库根目录开始；其后命令沿用该目录。替换所有示例路径后再运行。

<a id=rlbench-raw-replay></a>

# RLBench Raw → Replay 独立生成

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
`LPY/BridgeVLA_RLBench_TRAIN_Buffer/stack_blocks`，可直接作为[Oracle 增强脚本](oracle-replay.md)的
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
