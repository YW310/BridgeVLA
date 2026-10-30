# 文档索引

[项目首页](../README.md) · [代码索引与运行限制](reference/code-map.md)

完整阅读入口：[图解完整手册（单页 HTML）](handbook.html)。它将 docs 全部 Markdown、项目首页和 OHT README 整理为连贯主题章节，包含八张理解图、完整算法交互、正文、命令、表格及论文链接。合并重复导航、收起长示例，手机端宽表转为逐项阅读；固定浅色，可离线阅读和打印。原 Markdown 继续作为维护源，入口说明与历史内容分区保留。

更新完整手册：先 `python -m pip install -r tools/requirements-docs.txt`，再运行 `python tools/build_docs_handbook.py`；加 `--check` 检查是否同步。生成器自动发现 docs 新增的 Markdown，并将文档互链转换为页内跳转；本地图像内嵌，代码/配置链接仍指向仓库文件，论文外链仍需联网。

先看[交互算法总览（HTML）](design/algorithm-overview.html)：固定浅色、离线可读，包含输入输出维度、带符号流程图、O2/anchor 与 Action chunk 的完整公式和函数。顶部可切换路径、点击节点；下方可连续阅读全部步骤与状态条件前后对比，也可折叠章节或打印。

可编辑源文件为 `assets/algorithm-overview.fragment.html`；修改后在仓库根目录运行 `python tools/render_algorithm_overview.py`，用 `--check` 检查生成页面是否同步。

## OHT

OHT 的数据指南、模型设计、部署说明和交接记录统一合并为以下三份，按流程阅读即可。

| 文档 | 内容 |
| --- | --- |
| [数据](oht/data.md) | V4/V3 构建、划分与标注、相机/深度/ROI、几何与点云诊断 |
| [训练与模型](oht/training.md) | 已确认基线与排查经验、连续轨迹模型、训练/续训/评估、单关键点与物体辅助 |
| [部署与回放](oht/runtime.md) | 模型服务、HTTP 协议、IsaacLab、初态、chunk/waypoint 执行、GT 回放与验收 |

当前轨迹基线为 V4 + `regression_state`；用户运行确认及证据边界见[已确认基线与交接](oht/training.md#已确认基线与交接)。V4 chunk 使用 baseline，不接入 role cache；单关键点和 RLBench replay 使用各自的数据契约。

<a id="按任务阅读"></a>

## 通用环境与 RLBench / O2

| 任务 | 文档 |
| --- | --- |
| 安装与基础复现 | [安装](guides/installation.md) → [训练](guides/training.md) → [评估](guides/evaluation.md) |
| RLBench 数据 | [Raw → Replay](guides/replay.md) → [Oracle 实例增强](guides/oracle-replay.md) → [Semantic-GT](guides/semantic-gt.md) |
| O2 配置、训练、闭环与诊断 | [O2 操作指南](guides/object-conditioning.md) |

<a id="核心文档设计依据执行"></a>

## 设计与参考

| 文档 | 内容 |
| --- | --- |
| [统一设计与研究方向](design/role-relation-prior.md) | O2 当前架构、OHT 研究动机与候选方向、后续 memory / 真机规划 |
| [代码索引](reference/code-map.md) | 对应函数、配置入口、当前运行限制与验证入口 |
| [论文调研](research/object-centric-policy-memory.md) | 方法依据、监督前提与适用边界 |
| [结果与发布记录](experiments/results.md) | 原始实验表与历史记录 |

<a id="先分清实现与计划"></a>

设计中的后续规划不等于已实现能力，历史结果不作为当前配置验收。Oracle-GT 属于上界/诊断路线，预测动作前向不得读取 GT T/R；当前限制集中在[代码索引](reference/code-map.md#当前运行限制)。

## 命令约定

运行示例默认使用 Linux / Bash，从仓库根目录或指定目录执行。新终端需重新设置示例变量，并替换机器路径；带 `cd` 的独立示例从仓库根目录开始。各文档按主题维护一份命令，避免复制后产生分歧。

仓库根目录运行 `python tools/check_docs.py` 可离线检查本地链接与章节锚点；它不执行示例或访问外部网站。
