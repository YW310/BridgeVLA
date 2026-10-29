[文档索引](../README.md) · [项目首页](../../README.md)

> 命令不在 docs/ 下执行。带 cd 的独立示例从仓库根目录开始；其后命令沿用该目录。替换所有示例路径后再运行。

<a id=installation></a>

# Installation

## OHT 数据与模型环境

以下命令从仓库根目录执行。仅做 audit、replay 构建、几何诊断或角色标注时，安装数据依赖即可，不需要启动 RLBench / CoppeliaSim：

```bash
python -m pip install -r finetune/OHT/requirements-data.txt
```

OHT 模型训练和离线推理需要 CUDA、PaliGemma 资源和编译后的 point-renderer。仓库 `requirements-model.txt` 要求先单独安装与本机 CUDA 匹配的 `torch==2.5.1` / `torchvision==0.20.1`，然后执行：

```bash
python -m pip install -r finetune/OHT/requirements-model.txt
python -m pip install --no-build-isolation -e finetune/bridgevla/libs/point-renderer
python -m pip install tensorboard
```

编译 point-renderer 需要可用的 CUDA toolkit 和 C++ 编译器，只有驱动或 CPU PyTorch 不够。TensorBoard 是训练默认日志后端，当前 requirements 文件未列入它；使用 `--log-backend none` 时可省略。IsaacLab 客户端另用其仿真环境，见 [OHT 运行指南](../oht/runtime.md#启动与连接)。

准备 PaliGemma 的访问授权与本地缓存；BridgeVLA 热图预训练目录不能替代基础模型/processor 资源。OHT 数据、训练和预检入口见 [OHT README](../../finetune/OHT/README.md)。下列旧安装模板主要面向预训练与模拟 benchmark。

## 预训练与 RLBench / COLOSSEUM / GemBench

1. **Clone this repository and navigate to the BridgeVLA folder:**
```bash
git clone https://github.com/BridgeVLA/BridgeVLA.git
cd BridgeVLA
```

2. **Install the required package:** Review and adapt the selected install script before running it; these are environment templates, not portable installers.

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

安装面向 Linux / Bash，不直接适用于 Windows。先处理以下前提：

- 修改脚本中所有 `cd`、下载目录和 `/PATH_TO_*`；`pretrain_install.sh` 自带 `cd BridgeVLA/finetune`，按上面的工作目录原样执行会找错路径。
- 部分脚本使用 `sudo apt-get`、升级/卸载依赖；先审查系统权限与现有环境，不在已有训练环境中盲跑。
- RLBench 安装脚本固定部分模拟器/包版本，但 `xformers`、Pytorch3D 等仍有滚动依赖，不是完整 lockfile。实验前保存实际包版本、CUDA/driver、模拟器与 RLBench/PyRep commit；本地语法检查不证明它们兼容。

便携训练入口不会替你配置 simulator。激活环境后设置真实路径：

```bash
export COPPELIASIM_ROOT=/absolute/path/to/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
```

需要渲染时配置本机可用 `DISPLAY`，或在 Linux 使用 Xvfb；不沿用脚本中的远端 display/IP。
训练脚本选择与路径问题见[训练](training.md#rlbench-fine-tuning)，未修复的代码问题集中于[项目审查](../reference/code-map.md#项目审查2026-09-29)。
