# 文档索引

[返回项目首页](../README.md)

## 推荐阅读顺序

- 复现 baseline：[安装](installation.md) → [训练](training.md) → [评估](evaluation.md)。
- 准备数据：[Raw → Replay](replay.md) → [Oracle 实例增强](oracle-replay.md) → [严格 Semantic-GT 角色](semantic-gt.md)。
- 运行 Oracle 上界实验：[Semantic-GT](semantic-gt.md) → [O2 训练、消融与评估](o2-training.md)。
- 查阅结果：[论文结果与发布记录](results.md)。

## 研究与实现边界

[Phase / relation / object-centric views 研究设计](../BRIDGEVLA_ROLE_RELATION_PRIOR_DESIGN_CN.md)
讨论候选选择、遮挡、物体视角、pair heatmap 与 cost 等扩展；请以其中的实现边界和验收条件为准，不将方案视为已完成实现。

[Oracle prior 实现交接](../BRIDGEVLA_ORACLE_PRIOR_CODING_HANDOFF_CN.md)
保留历史实现背景。实际运行参数以当前脚本、配置及上述专题说明为准。

## 命令约定

命令在仓库根目录或明确指定的代码目录执行，**不在 `docs/` 下执行**。
带 `cd` 的独立示例默认从仓库根目录开始；同一示例中的后续命令沿用其工作目录。
替换示例中的 `PATH_TO_*`、`/path/to/*` 和机器专用绝对路径后再运行。
