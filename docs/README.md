# 文档索引

[返回项目首页](../README.md)

## 目录层次

```text
docs/
├── README.md                         # 总入口与阅读顺序
├── guides/                           # 使用指南与数据准备
│   ├── installation.md
│   ├── training.md
│   ├── replay.md
│   ├── oracle-replay.md
│   ├── semantic-gt.md
│   └── evaluation.md
├── experiments/                      # 实验操作与结果
│   ├── o2-training.md
│   └── results.md
├── design/                           # 研究方案（非实现承诺）
│   ├── role-relation-prior.md        # 精简主线与实施顺序
│   └── role-relation-details.md      # 公式、接口与验收细节
└── handoff/                          # 实现背景与交接
    └── oracle-prior.md
```

## 1. 使用指南

- 环境准备：[安装与依赖](guides/installation.md)。
- 复现 baseline：[训练、8×40GB 与日志](guides/training.md) → [评估](guides/evaluation.md)。
- 准备数据：[Raw → Replay](guides/replay.md) → [Oracle 实例增强](guides/oracle-replay.md) → [严格 Semantic-GT 角色](guides/semantic-gt.md)。

## 2. 实验与结果

- Oracle 上界实验：[Semantic-GT 数据](guides/semantic-gt.md) → [O2 训练、消融与评估](experiments/o2-training.md)。
- 论文结果与发布记录：[结果与历史](experiments/results.md)。

O2 在训练和评估时使用 GT 实例，不能作为无 GT 的部署结果报告。启发式 Oracle 角色与严格 semantic-GT 标注须区分。

## 3. 研究设计

[精简设计](design/role-relation-prior.md)：ARE 主线、当前 phase 的候选与物体视角、pair heatmap、cost 决策及最小实现。

[详细设计与验收](design/role-relation-details.md)：公式、监督、接口字段、相关工作、任务例子与验收反例；按需查阅。

该文档描述下一阶段研究方案；请以其中的实现边界为准，不将方案视为已完成实现。

## 4. 实现交接

[Oracle prior 交接](handoff/oracle-prior.md)：实验动机、实现位置、历史状态和后续扩展。

交接文档保留原有日期与历史语境。实际运行参数以当前脚本、配置和使用／实验指南为准。

## 命令约定

命令在仓库根目录或明确指定的代码目录执行，**不在文档目录下执行**。
带 `cd` 的独立示例默认从仓库根目录开始；同一示例中的后续命令沿用其工作目录。
替换示例中的 `PATH_TO_*`、`/path/to/*` 和机器专用绝对路径后再运行。
