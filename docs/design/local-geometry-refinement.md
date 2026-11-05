# 观测约束的局部几何动作精修

[文档索引](../README.md) · [代码与数据契约](../reference/code-map.md)

## 研究问题与当前交付

目标是检验：原始 RGB-D 已观测到、但投影表示未充分保留的局部几何，是否能改善下一关键帧动作目标。基准模型完整保留原有 coarse-to-fine 预测。新分支以候选夹爪位姿定义交互区域，用几何证据预测小幅位姿修正。

目前提供独立的 CPU 离线原型：几何场构建、零初始化精修头、缓存训练和未使用 episode 上的位姿误差评估。它未接入现有在线动作路径，未训练机器人数据，也未完成 CUDA/模拟器验收。合成缓存只用于验证软件链路，不能作为任务成功率证据。现有对象条件策略与 OHT chunk 的配置及动作路径不因本原型改变。

| 文件 | 职责 |
| --- | --- |
| [local_geometry.py](../../finetune/bridgevla/models/local_geometry.py) | 多视角米制深度投影、局部几何证据与已知区域比例 |
| [local_geometry_refiner.py](../../finetune/bridgevla/models/local_geometry_refiner.py) | 有界局部平移/旋转修正，位姿损失和误差指标 |
| [train_local_geometry_refiner.py](../../tools/train_local_geometry_refiner.py) | 显式 NPZ 缓存验证、合成样例、训练和独立测试集评估 |

## 与 OHT action chunk 共用的模块

公共几何层位于 [bridgevla/geometry](../../finetune/bridgevla/geometry/__init__.py)，不依赖任何数据集 schema 或策略配置。

| 共用文件 | 使用方与职责 |
| --- | --- |
| [numpy_ops.py](../../finetune/bridgevla/geometry/numpy_ops.py) | OHT 数据与运行时、精修缓存入口共用刚体/内参校验、xyzw 四元数和深度几何基础操作；纯 NumPy 路径不加载 Torch |
| [torch_ops.py](../../finetune/bridgevla/geometry/torch_ops.py) | OHT chunk decoder、局部证据场与精修头共用坐标变换、四元数/6D 旋转、SO(3) 指数及旋转矩阵误差 |

OHT 的 `data.geometry` 与 `chunk_model.quaternion_matrix/rotation_6d` 保留原导入入口，底层实现来自公共层。模型参数仍由原 decoder/refiner 持有；共用层只提供无参数运算。

Torch 数学运算的热路径只检查形状、dtype 类型与 device 等元数据，有限性与刚体检查在显式 validator 或原模型边界执行。点变换遵循 Torch 原生类型提升和 autocast，支持 FP32 实测 TCP 与 BF16 网络输出共同参与运算；CPU BF16 回归覆盖该接口，GPU 性能仍需目标环境实测。

调用层继续管理各自的坐标基准、有效掩码与损失尺度：OHT chunk 的测量 TCP 和本原型的预测候选夹爪是不同的局部坐标系。Spatial chunk 的 `delta_pose` 平移在初始 TCP 坐标系累计，旋转逐步右乘；精修器则修正单个动作目标。OHT 旋转 MSE 对九个矩阵元素取均值，精修旋转项按自己的尺度归一化，均由原损失入口定义。

OHT 的同 z-buffer XYZ/coverage token 池化、世界轴 TCP crop 与本原型的深度证据网格分别构建。coverage 表示可见表面覆盖，证据网格的 known/free 表示射线观测支持，二者各有有效性契约。序列时间、padding、flow velocity loss 和 checkpoint 配置属于 action chunk 业务层。

## 几何与动作定义

所有位置单位为米。相机使用 OpenCV 光学轴：X 向右、Y 向下、Z 向前；像素中心坐标为整数，深度是相机 Z，而非径向距离。`camera_to_world` 为光学坐标到世界坐标的刚体变换，`candidate_pose` 为候选夹爪坐标到世界坐标的变换。

网格中心位于候选夹爪附近，张量顺序为 `[B,C,D,H,W]`，后三轴分别对应局部 Z/Y/X。对每个体素，将世界点投回各原始相机，以最近邻采样深度，避免在深度边缘插值制造假表面。

设深度差为 `delta = sampled_depth - camera_z`，阈值为 `tau`：

- `delta > tau`：该视角支持自由空间；
- `abs(delta) <= tau`：该视角支持近表面，不能据此推断完整实体内部；
- `delta < -tau`：在该表面后方，该视角没有提供该位置的几何证据；
- 图外、相机后方、无效深度：未知。

五通道固定为：归一化的已观测投影距离、近表面支持比例、自由空间支持比例、已知标记、自由/近表面混合标记。支持比例以相机总数为分母；投影距离只融合提供自由/近表面证据的视角；未知位置距离为零并且已知标记为零。混合标记只记录不同视角支持不一致，不直接判定为传感器错误。

局部动作使用明确的解耦组合：

```text
p_final = p_candidate + R_candidate @ translation_local
R_final = R_candidate @ Exp_SO3(rotation_vector)
```

局部平移和旋转向量分别限制模长；这不是把平移向量直接作为完整 SE(3) 指数的切空间平移。最后输出层零初始化，初始动作等于基准动作；完全没有已知几何的样本保持基准位姿。夹爪开合与原 collision flag 不由此原型重定义。

训练损失对最终位姿使用平移 SmoothL1 与旋转矩阵差异，避免训练时直接对零误差执行不稳定的 `acos`。离线报告分别计算平移米制误差与旋转弧度误差。

## 缓存契约与防泄漏

输入为 `allow_pickle=False` 可读取的 NPZ，每个文件只包含一个明确的数据来源和冻结基准版本。相机标定/深度必须先与原始有序点云进行反投影核验，不能仅修改元数据就宣称已完成坐标转换。

| 字段 | 形状或值 |
| --- | --- |
| `schema` | 标量 Unicode：`local_geometry_refinement_v1` |
| `source` | 标量 Unicode：`synthetic`、`rlbench` 或 `oht` |
| `baseline_id` | 标量 Unicode，冻结基准权重及配置的可追溯标识 |
| `camera_model` | 标量 Unicode：`opencv_z_meters` |
| `candidate_source` | 真实缓存为 `policy_prediction`；合成缓存为 `synthetic_perturbation` |
| `depth` | 浮点 `[N,V,H,W]`，米制相机 Z；无效深度允许为 NaN/0 |
| `intrinsics` | 浮点 `[N,V,3,3]`，与 depth 尺寸、像素坐标一致 |
| `camera_to_world` | 浮点 `[N,V,4,4]` |
| `candidate_pose` | 浮点 `[N,4,4]`，冻结基准完整两阶段的世界系预测 |
| `target_pose` | 浮点 `[N,4,4]`，专家下一关键帧，仅用于损失/指标 |
| `context` | 浮点 `[N,C]`，同次观测与指令的冻结语义/状态特征 |
| `episode_id` | Unicode `[N]`，包含数据集/任务/variation/demo，允许同 episode 多帧 |
| `sample_id` | 唯一 Unicode `[N]`，包含 episode 和当前帧身份 |

训练、验证、测试按 episode 隔离，不能随机打散同一示范的帧。训练入口检查两份缓存的数据来源、基准标识和 context 维度一致，并拒绝 episode/sample 交叉；测试入口拒绝训练或验证用过的 episode 与 sample ID。元数据不能自动证明缓存内容无泄漏，真实导出仍需审查前向输入。

本地 RLBench 对接有四个具体边界：

1. [dataset.py](../../finetune/RLBench/utils/dataset.py) 中 `gripper_pose` 保存的是**专家下一关键帧**。它只能转成 `target_pose`，不能用作裁剪中心或当前 TCP；局部中心必须来自冻结模型 `act()` 的预测。
2. 原 replay 的 `low_dim_state` 包含夹爪、双指关节与时间，不提供当前 TCP。需要当前末端位姿时，应导出独立的观测字段。
3. 离线 replay 保存 depth/K/外参，但在线 [create_obs_config](../../finetune/bridgevla/libs/peract/helpers/utils.py) 当前关闭深度。闭环接入前需启用并确认米制深度，或验证由有序点云回算相机 Z 的独立适配器。
4. [bridgevla_agent.py](../../finetune/bridgevla/models/bridgevla_agent.py) 的 SE(3) 增广只变换点云/标签。构建真实缓存时应使用一致的原始世界坐标，或同步变换全部相机/候选/标签，不能混用两套坐标。

真实 replay 到该 NPZ 契约的自动导出器、在线 hook 和同预算 RGB-D 对照头尚未实现；需要可用权重与数据后再接入。实验缓存只读取用户明确生成的 NPZ，不自动扫描或反序列化旧 replay pickle。

## 可运行的合成验证

以下从仓库根目录运行。Windows 当前环境可将 `python` 替换成 `.\.venv-oht\Scripts\python.exe`。示例仅验证 CPU 软件链路；三份缓存分别生成不同 episode，形状是单相机看到的凹陷圆盘。

```bash
python tools/train_local_geometry_refiner.py synthetic --output-dir tmp/local-geometry-pilot
python tools/train_local_geometry_refiner.py train --train-cache tmp/local-geometry-pilot/train.npz --validation-cache tmp/local-geometry-pilot/validation.npz --output tmp/local-geometry-pilot/refiner.pt --epochs 10 --resolution 24
python tools/train_local_geometry_refiner.py evaluate --cache tmp/local-geometry-pilot/test.npz --checkpoint tmp/local-geometry-pilot/refiner.pt
```

训练使用固定轮数，保存最后一轮；验证误差仅用于诊断。调整超参数后需要保持独立测试集。checkpoint 绑定几何参数、context 维度、来源、冻结基准标识和训练/验证 episode。默认拒绝覆盖已有缓存或 checkpoint，重跑请使用新的输出目录。

```bash
python -m pytest -q tests/test_local_geometry.py tests/test_local_geometry_refiner.py tests/test_local_geometry_pilot.py
python tools/check_docs.py docs/design/local-geometry-refinement.md
```

## 本机软件验证记录

2026-11-05，Windows、Python 3.12.13、PyTorch 2.5.1 CPU。针对性测试为 **55 passed、35 subtests passed**，覆盖几何坐标、遮挡/未知、残差梯度与边界、缓存划分以及训练/保存/重载/测试链路。

使用默认合成生成器的 train/validation/test 各 32/12/12 个样本，resolution=24、seed=0，训练 10 轮。未使用测试 episode 的离线位置误差如下；合成样例没有旋转误差，不能据此验证旋转修正效果。

| 合成测试指标 | 候选位姿 | 精修后 |
| --- | --- | --- |
| 平均位置误差 | 6.00 mm | 5.16 mm |
| P90 位置误差 | 8.71 mm | 9.05 mm |

平均值降低而尾部略有变差，所以本结果只证明软件链路可运行和局部头可学习，不支持鲁棒性或机器人成功率提升的结论。本地运行产物位于 `tmp/local-geometry-pilot-20261105/`，`run-summary.json` 保存完整离线指标；该临时目录不进入版本控制。

## 真实实验顺序与退出条件

先抽查插孔、形状块入槽和杯架任务的失败帧，确认原始深度里存在可辨认的关键间隙，并把全局对象选错与局部位姿误差分开统计。随后使用同一冻结基准生成候选缓存。

| 对照 | 要排除的解释 |
| --- | --- |
| 原始完整 BridgeVLA | 原有局部放大被人为删除 |
| 同 ROI、相近参数量/算力的 RGB-D crop 精修头 | 收益只是来自额外局部网络 |
| 去掉已知/自由空间通道的局部几何精修头 | 观测状态是否真的必要 |
| 相同预算下更高分辨率的原有局部预测 | 收益只是来自更高空间采样密度 |

最终报告任务成功率、最终定位误差、非预期碰撞/卡滞和端到端延迟；同一任务初态、示范数与训练预算配对，使用多个训练 seed。离线位姿误差下降仅作为闭环准入证据。

若同预算 RGB-D crop 等效，则该路线只支持工程改进；若只有理想深度有效，则优先处理传感器/标定；若精修只改善离线误差却不改善闭环，需检查分布偏移与执行器，而不是继续叠加重构网络。TSDF、局部坐标和残差头均不是单独的新贡献，论文需要建立观测状态保留与具体错误减少之间的证据。
