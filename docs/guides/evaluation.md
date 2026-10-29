[文档索引](../README.md) · [项目首页](../../README.md)

> 下列 Bash/Python 示例面向已安装对应仿真依赖的 Linux 环境；带 cd 的独立代码块从仓库根目录开始。
> 示例路径须替换。固定 episode/seed/预算后再运行：策略失败和超时计入分母，环境/数据异常另列；
> 不允许重复直到收集足够成功。GT 准入采用[三-seed 配对 CI](object-conditioning.md#闭环统计与准入)。

<a id=evaluation></a>

# Evaluation

## RLBench

`finetune/RLBench/eval.sh` 已支持环境变量配置，无需为常规任务/路径选择修改脚本。
默认仅运行 place_cups、25 episodes；需要全任务时显式设置 `TASKS=all`。

```bash
MODEL_FOLDER=/path/to/checkpoints \
MODEL_NAME=model_60.pth \
EVAL_DATAFOLDER=/path/to/BridgeVLA_RLBench_EVAL_DATA \
TASKS="place_cups stack_blocks" \
EVAL_EPISODES=25 START_EPISODE=0 EPISODE_LENGTH=25 \
DEVICE=0 SAVE_VIDEO=0 \
bash finetune/RLBench/eval.sh
```

脚本切换到自身目录。模型配置、预训练资源和 CoppeliaSim/PyRep 环境仍需齐备。
DISPLAY 使用实际可用的显示/Xvfb 配置，不沿用脚本中机器专属的默认地址。
输出位于 `MODEL_FOLDER/eval/<task>/<oracle-provider>/<model-name-without-pth>/`；
检查 runtime log、逐 episode 记录和 completed/requested episodes，而不只读取汇总百分比。
同一路径续跑需显式 `EVAL_RESUME=1`；更换 checkpoint、配置或任务预算时使用独立实验目录。

多 GPU 评估见 [8×40GB 说明](training.md#rlbench-8x40)；
Oracle/GT replay 与训练后模型评估的差异见 [O2 评估](object-conditioning.md#closed-loop评估)。

## COLOSSEUM

先阅读 `finetune/Colosseum/cleanup_script.py`：它会原地删除 stack_cups 下的 ZIP，
并将名字以 `_13` 结尾目录中的内容移动到 variation0。先备份或在副本上处理，
也可下载[已整理数据](https://huggingface.co/datasets/LPY/BridgeVLA_COLOSSUM_EVAL_DATA/tree/main)。

```bash
cd finetune/Colosseum
python3 cleanup_script.py /path/to/COLOSSEUM_EVAL_DATA
```

运行前修改 `eval.sh` 的仓库、数据、显示和输出路径。该脚本还会卸载
opencv-python/opencv-contrib-python 并安装 opencv-python-headless；
环境已就绪时先移除这些安装步骤，避免每次评估改动环境。
位置参数依次为 variation、日志 trial 名、模型 epoch、模型目录：

```bash
cd finetune/Colosseum
bash eval.sh VARIATION LOG_NAME MODEL_EPOCH /path/to/checkpoints
```

脚本对缺失的 task/variation 数据直接跳过；退出成功不证明覆盖完整。
确认预定任务、全部 variation 和 episode 覆盖后，修改 `cal_statics.py` 中的结果路径并运行：

```bash
cd finetune/Colosseum
python3 cal_statics.py
```

历史说明对 Variations 1/6 的 close laptop lid、wipe desk、insert onto peg 曾采用
“重复至收集 25 个 successful trials”的 workaround。这不能作为新流程，也不能据此
认定历史表格使用标准固定 episode 分母。`cal_statics.py` 只聚合已有 task/model CSV，
不审计完整预定任务集；历史数值应待原始日志核实口径后解释。

## GemBench

先准备 jq，并修改 run_server.sh/run_client.sh 的仓库、数据、输出路径和一致的端口
（脚本当前均为 13003）。run_client.sh 每次执行 `sudo apt-get install -y jq`，
环境准备完成后应移除这一步。先启动服务，保持进程运行：

```bash
cd finetune/GemBench
bash run_server.sh MODEL_EPOCH /path/to/model_base
```

在另一终端启动客户端：

```bash
cd finetune/GemBench
bash run_client.sh SEED MODEL_EPOCH
```

客户端将结果追加为 JSON-lines `result.json`，不是一个 JSON 数组。
`cal_results.py` 按 task-list 顺序切分为每任务 20 条，仅检查各 split 总行数，
不校验逐条 task/episode 身份，缺少整个 split 时还可能跳过。
聚合前确认无重复追加、任务顺序一致、所有预定 split/seed 齐全；
其 seed 平均为已有 split 成功率的等权平均，并非跨全部 episode 加权，也不自动汇总跨 seed 方差。

设置 `cal_results.py` 的结果目录后运行：

```bash
cd finetune/GemBench
python3 cal_results.py
```

OHT 的单目标/chunk open-loop 指标见 [OHT 离线评估](../oht/training.md#离线与闭环评估)，
HTTP 与原生 IsaacLab 执行见[服务指南](../oht/runtime.md#启动与连接)。
离线位姿/轨迹误差不能作为闭环任务成功率。
