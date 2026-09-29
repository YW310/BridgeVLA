# 文档索引

[返回项目首页](../README.md)

## 按任务阅读

| 你要做什么 | 入口 | 下一步 |
| --- | --- | --- |
| 复现 BridgeVLA | [安装](guides/installation.md) | [训练](guides/training.md) → [评估](guides/evaluation.md) |
| 准备训练数据 | [Raw → Replay](guides/replay.md) | [实例增强](guides/oracle-replay.md) → [Semantic-GT](guides/semantic-gt.md) |
| 运行 O2：GT / anchor / 外部预测 / slots | [O2 操作指南](guides/object-conditioning.md) | 同页完成配置选择、训练、闭环、日志与可视化 |
| 理解设计与实现 | [统一设计](design/role-relation-prior.md) | 同页查看架构、接口、memory 与真机约束；函数见[代码索引](reference/code-map.md) |
| 查看论文与结果 | [论文调研](research/object-centric-policy-memory.md) | [结果与发布记录](experiments/results.md) |

## 核心文档：设计、依据、执行

| 文档 | 负责什么 |
| --- | --- |
| [统一设计](design/role-relation-prior.md) | 整体架构、当前接口与数据、本轮调整、实验顺序、后续 memory、真机要求；不再拆独立子设计 |
| [论文 survey](research/object-centric-policy-memory.md) | 原论文、监督前提、方法边界和设计依据 |
| [O2 操作指南](guides/object-conditioning.md) | 所有现有 O2 模式的配置、训练、闭环准入与诊断；每项操作只维护一次 |
| [项目审查与代码索引](reference/code-map.md) | 对应函数、配置、已确认运行风险与验收清单，不重复设计正文 |

`guides/` 维护环境、数据和运行说明，`experiments/` 只保留历史结果；设计与 survey 各一份。
原 O2 分篇正文已合并，不另保留命令副本；实际运行以当前脚本和配置为准。
当前代码审查及未修问题见[项目审查](reference/code-map.md#项目审查2026-09-29)。

## 先分清实现与计划

单帧 slots、soft role 条件化和共享动作特征已有 opt-in 代码，闭环收益仍需验证。
最终混合角色图直接监督、两个语义 queries、跨尺度继承、temporal memory 与恢复机制尚未实现。
现有 YAML 不会因为设计文档更新而自动切换到计划架构。

Oracle-GT 是上界/诊断路线；预测策略的动作前向不得读取 GT T/R。
测试期让 simulator 对象跟随 base heatmap 的 residual 模式属于预测条件化，不是真正的 Oracle Target GT。

## 命令约定

命令在仓库根目录或示例指定的代码目录执行，不在 `docs/` 执行。
同一示例沿用前面的 `cd`；替换 `PATH_TO_*`、`/path/to/*` 和机器专用路径后再运行。
