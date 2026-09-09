[文档索引](../README.md) · [项目首页](../../README.md)

> 命令不在 docs/ 下执行。带 cd 的独立示例从仓库根目录开始；其后命令沿用该目录。替换所有示例路径后再运行。

<a id=installation></a>

# Installation

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
(cd pretrain && bash ./pretrain_install.sh)

# For RLBench fine-tuning
(cd finetune/RLBench && bash ./install_rlbench.sh)

# For Colosseum fine-tuning
(cd finetune/Colosseum && bash ./install_colosseum.sh)

# For GemBench fine-tuning
(cd finetune/GemBench && bash ./install_gembench.sh)
```
3. Note: To avoid potential conflicts between different simulation benchmarks, we suggest creating separate virtual environments for each benchmark. Also, our model is built upon [Paligemma](https://huggingface.co/google/paligemma-3b-pt-224), which is a gated repo. Therefore, you should first be authenticated to access it.

> 安装面向 Linux / Bash；执行前修改安装脚本中的硬编码路径，并配置本机 CoppeliaSim 环境变量。
