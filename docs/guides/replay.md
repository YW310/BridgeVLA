[文档索引](../README.md) · [项目首页](../../README.md)

> 命令不在 docs/ 下执行。带 cd 的独立示例从仓库根目录开始；其后命令沿用该目录。替换所有示例路径后再运行。

<a id=rlbench-raw-replay></a>

# RLBench Raw → Replay 独立生成

`tools/generate_rlbench_replay.py` 使用与训练入口完全相同的
`create_replay()` / `fill_replay()` 实现，把已保存的 RLBench raw episode 转换为标准
BridgeVLA/YARR replay。它会完成关键帧发现、demo augmentation、动作离散化、四相机观测、
RN50 CLIP 语言特征、`replay_info.npy` 和 final-observation sentinel。请在已安装
RLBench/PyRep/YARR/peract_colab 的 `bridgevla` 环境中，从仓库根目录运行。

先仅检查任务目录、episode 编号连续性和输出/raw 路径是否独立；dry-run 不读取每帧 RGB-D、不检查 CUDA/CLIP 或已有输出内容：

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
`LPY/BridgeVLA_RLBench_TRAIN_Buffer/stack_blocks`。[Oracle 增强脚本](oracle-replay.md)的
`--replay-dir` 可传这个单任务目录，也可传其父目录：前者将增强文件直接写到 `--output-dir`，后者写到 `--output-dir/<task>`。如果 `--raw-data-dir` 已经指向 `.../train`，程序也会自动识别，
不需要再次拼接 `train`。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--raw-data-dir PATH` | 必填 | 数据集根目录、`train` split、单任务目录或 `episodes` 目录。 |
| `--output-dir PATH` | 必填 | Replay 输出根目录；每个任务写入 `PATH/<task>`，必须与 raw 路径相互独立。 |
| `--split NAME` | `train` | raw 根目录下的 split 名。 |
| `--task NAME` | `all` | 任务名、逗号分隔任务名或 `all`；可重复传入。`all` 发现实际存在的任务，不要求固定 18 个；start-episode 和显式 num-demos 应用于每个选定任务，任一任务缺号都会使预检失败。 |
| `--start-episode N` | `0` | 第一个 raw episode 编号。 |
| `--num-demos N` | 从起点到最后 | 连续 episode 数量；若范围内缺号会在生成前报错。 |
| `--demo-augmentation-every-n N` | `10` | 每隔 N 个 raw 帧创建一个 demo augmentation replay 子序列，与原训练默认值一致。 |
| `--no-demo-augmentation` | 关闭 | 每个 episode 只从起始帧生成一个子序列，数据更少且与默认训练数据分布不同。 |
| `--device DEVICE` | `auto` | CLIP 特征设备；`auto` 优先 `cuda:0`，也可指定 `cpu` 或其他 CUDA 卡。 |
| `--clip-model NAME_OR_PATH` | `RN50` | OpenAI CLIP 名称或本地权重路径；首次使用名称时可能需要下载权重。 |
| `--batch-size N` | `1` | YARR replay schema 的 batch size；不影响生成的 transition 内容。 |
| `--replay-capacity N` | `300000` | 内部 UniformReplayBuffer 容量。 |
| `--dry-run` | 关闭 | 只发现并检查输入，不加载 CLIP、不创建输出。 |
| `--skip-existing` | 关闭 | 通过下述结构抽查后跳过已有任务；不比较源数据/生成参数，不补写不完整目录。 |
| `--overwrite` | 关闭 | 删除并重建已存在的精确任务输出；与 `--skip-existing` 互斥。 |

生成时先写入 `<output-dir>/.<task>.raw_to_replay.tmp`。结构检查确认 replay 从 0 连续编号、
`replay_info.npy` 长度一致，并抽查首个普通 transition 的必要 metadata 和最后一个 final sentinel
的 terminal 一致性；通过后才改名为正式任务目录。这不是逐文件内容审计，也不证明 raw/replay
逐像素对齐；追加 Oracle 时仍应保留严格源对齐检查。

若程序中断，临时目录会保留用于检查；确认无需保留后使用 `--overwrite` 重新生成。
该选项先删除已有任务输出和对应临时目录，重建失败时旧任务目录不会自动恢复。
更换 raw 数据、CLIP 权重或 augmentation 参数时使用新输出目录，不能依赖 `--skip-existing`
判断已有 replay 是否来自相同配置。正式运行即使全部任务将被跳过，也会先加载 CLIP。

本工具的 `--dry-run` 不落盘；Oracle 增强工具的同名选项会实际解码、审计并可能写诊断文件，见[对应参数说明](oracle-replay.md#oracle-parameters)。两者都不能把抽查结果当作全量内容验收。
