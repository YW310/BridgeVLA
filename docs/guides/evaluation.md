[文档索引](../README.md) · [项目首页](../../README.md)

> 命令不在 docs/ 下执行。带 cd 的独立示例从仓库根目录开始；其后命令沿用该目录。替换所有示例路径后再运行。

<a id=evaluation></a>

> 新评估固定 episodes：所有有效策略失败计入分母，环境/数据异常单独记录；禁止重复测试直到收集足够成功。
> 下文 COLOSSEUM 的 successful trials 是历史说明，原始筛选口径不能仅凭文字确认，不据此重解释历史结果。
> GT 准入采用[三-seed 配对 CI](object-conditioning.md#闭环统计与准入)。

# Evaluation

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
历史说明曾对 Variations 1/6 的 close laptop lid、wipe desk、insert onto peg 采用“重复至收集 25 个 successful trials”的 workaround。
这不应作为新评估流程，也不能据此认定历史表格采用标准固定 episode 成功率。`cal_statics.py` 只聚合已有 task/model CSV，
不验证完整预定任务集；新实验先检查 task/variation/episode 覆盖，再统计。历史数值保持原样，待原始日志确认口径。

3. **GemBench Evaluation:** First provision `jq` and adapt the scripts' repository/data/output paths and server port.
`run_client.sh` currently invokes `sudo apt-get install -y jq` on every run; review or remove that installer step before evaluation.
Then launch the server:

```bash
cd finetune/GemBench
bash run_server.sh  MODEL_EPOCH  MODEL_BASE_PATH
```
After lanuching the server, you can run the following code to evaluate the model:
```bash
cd finetune/GemBench
bash run_client.sh  SEED MODEL_EPOCH
```
The client writes JSON-lines `result.json`, not a single JSON array. `cal_results.py` assumes task-list order and exactly 20 trials per task;
it checks total line counts but does not verify per-record task/episode identity, and missing split files can be skipped.
Before aggregation, verify task order, no duplicated append runs, and all requested splits/seeds. Set the results path and run:

```bash
cd finetune/GemBench
python3 cal_results.py
```

RLBench 多 GPU 评估见 [8×40GB 说明](training.md#rlbench-8x40)；Oracle 对照见 [O2 评估](object-conditioning.md#closed-loop评估)。
