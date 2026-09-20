[文档索引](../README.md) · [项目首页](../../README.md)

> 命令不在 docs/ 下执行。带 cd 的独立示例从仓库根目录开始；其后命令沿用该目录。替换所有示例路径后再运行。

<a id=evaluation></a>

> 新评估固定 episodes：所有有效策略失败计入分母，环境/数据异常单独记录；禁止重复测试直到收集足够成功。
> 下文 COLOSSEUM 的 successful trials 是历史说明，原始筛选口径不能仅凭文字确认，不据此重解释历史结果。
> GT 准入采用[三-seed 配对 CI](../experiments/object-conditioned-joint.md#3-闭环准入)。

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

RLBench 多 GPU 评估见 [8×40GB 说明](training.md#rlbench-8x40)；Oracle 对照见 [O2 评估](../experiments/o2-training.md#o2-code-path)。
