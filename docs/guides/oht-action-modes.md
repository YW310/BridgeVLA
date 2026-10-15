# OHT 动作模式配置

训练配置新增 action_mode，可选 **eef / relative_eef / joint / relative_joint**。它与物体辅助 mode 独立组合，baseline、role_queries 和 predicted_external 均支持四种动作模式。

## 配置与训练

现有 baseline.yaml 默认 eef，旧配置省略 action_mode 也按 eef 处理：

~~~yaml
extends: baseline.yaml
action_mode: joint
~~~

可直接使用以下文件：

| 配置 | 输出 |
| --- | --- |
| finetune/OHT/configs/baseline.yaml | eef |
| finetune/OHT/configs/relative_eef.yaml | relative_eef |
| finetune/OHT/configs/joint.yaml | joint |
| finetune/OHT/configs/relative_joint.yaml | relative_joint |

例如：

~~~bash
python -m finetune.OHT.train \
  --config finetune/OHT/configs/joint.yaml \
  --replay /data/oht/replay-v3 \
  --output /data/oht/runs/joint-s0 \
  --validate-only
~~~

去掉 --validate-only 并提供项目所需预训练权重，即进入 CUDA 训练。先完成数据预检和短步训练，再正式训练。完整模型/GPU/IsaacLab 四模式尚未现场验证。

若要角色预测辅助加关节输出，在 configs 目录创建配置：

~~~yaml
extends: role_queries.yaml
action_mode: relative_joint
~~~

训练命令额外提供 --role-cache，与现有角色训练流程一致。外部物体辅助同理继承 predicted_external.yaml。配置文件相对继承路径以该 YAML 所在目录为准。

## 动作定义

所有模式最后一维均为夹爪二值状态：0 关，1 开。关节顺序固定为原数据 observation.state 的前六个元素；joint_names 仅声明名称，不执行重排。

| 模式 | 动作向量 | 标签来源 |
| --- | --- | --- |
| eef | world TCP xyz + xyzw quaternion + grip，共 8 维 | 未来关键点的实测 TCP |
| relative_eef | delta_xyz + rotation_vector + grip，共 7 维 | 当前实测 TCP 到同一未来 TCP 的位姿差 |
| joint | q1…q6 + grip，共 7 维 | 未来关键点实测六关节位置，单位 rad |
| relative_joint | delta_q1…delta_q6 + grip，共 7 维 | 同一未来关节位置减当前实测位置，单位 rad |

原始 action 前六维不参与这些标签。未来终点由现有 keypoints 配置确定，因此相对动作表示**到未来关键点的位移/转角目标**，不是每秒速度，也不保证是单个仿真步的位移。

relative_eef 默认平移采用 world 系，旋转采用当前 TCP body 系的旋转向量，单位分别为米与弧度。四元数分量不直接相减。

~~~yaml
action_mode: relative_eef
action:
  translation_frame: world  # world / body / base
  rotation_frame: body     # world / body / base
~~~

若使用 base，必须显式提供固定的 world_from_base 4×4 刚体变换。body 平移为 R_current^T * (p_target-p_current)；body 旋转为 log(R_current^T * R_target)。world/base 旋转使用对应坐标系下的左乘旋转误差。不会猜测机器人基座朝向。

关节角默认直接相减。只有确认某关节连续旋转时才填写 joint_wrap，让该维使用最短角差：

~~~yaml
action_mode: relative_joint
action:
  joint_wrap: []  # 确认连续关节后填写其 0..5 索引；默认不 wrap
  joint_names: null  # 可选六个名称，与 observation.state[:6] 顺序一致
  joint_limits: null  # 可选六组 [min,max]，单位 rad
~~~

joint_limits 与 joint_wrap 不同时启用。给出关节限位后，训练拒绝越界 GT；在线关节预测按该限位限制目标。必须使用真实机器人参数，仓库不提供猜测限位。

## 观测与缓存

新构建的 replay 额外保存当前 joint_positions[6]、tcp_pose[7]，以及独立标签 joint_action[7]。数据契约记录 action_targets_version 和 joint_count；内容继续由 index/observation 哈希绑定。

同一份新 replay 可训练四种模式，无需按动作模式重复构建 RGB-D 缓存。未来关节标签只用于训练，在线输入白名单不会包含 joint_action：

- eef 保留已有输入行为，不增加机器人 pose/joint 输入。
- relative_eef 额外要求当前 tcp_pose，world/xyzw。
- joint、relative_joint 额外要求当前 joint_positions，rad。

已通过当前数据契约校验的旧 EEF 缓存和 checkpoint 仍按 EEF 处理；旧缓存若缺当前机器人状态/未来关节标签，其他模式会明确要求重新 build，不会用零值代填。原始数据、audit 和相机标定可以复用。重建缓存后须同步重建绑定旧 contract hash 的教师/预测缓存。

## 模型与损失

eef 使用原有热图位置、分桶旋转和夹爪动作头。其余三种模式新增真实的回归动作头，读取相同的物体辅助后全局/局部动作特征及当前机器人状态，直接预测六维控制量与一个夹爪 logit。

新 OHT 默认配置显式设置 mvt.final_dim=2048，与当前 VLM 的全局/局部 2048 维特征一致。旧 RVT 默认值 64 不能直接用于此 VLM 的分类头输入。向量模式冻结未用于控制输出的旧分类头，从旧配置初始化时允许这些分类头尺寸不同；EEF 模式仍严格检查分类头尺寸，不会静默忽略。

训练损失为：

- 六维动作的 Smooth L1。
- 夹爪 BCE。
- 辅助 EEF translation heatmap 损失，维持 coarse/refine 的目标定位。
- 已启用的角色图/Reference NULL 等原有辅助损失。

向量模式不会把旧旋转/夹爪分类损失加入优化目标；日志保留原有诊断项，实际动作损失见 vector_action_loss、vector_gripper_loss、eef_aux_translation_loss。内部教师仍只用于角色监督，动作前向不输入教师几何。

可选训练参数：

~~~yaml
action:
  hidden_dim: 256
  translation_scale: 1.0  # 米，回归归一化尺度
  rotation_scale: 1.0     # 弧度
  joint_scale: 1.0        # 弧度
  auxiliary_translation_weight: 1.0
~~~

这些 scale 是训练归一化尺度，输出时还原物理单位，**不是控制器 gain**。辅助定位权重必须大于 0。向量模式要求 transform_augmentation=false，防止独立改变场景/TCP 却保持真实关节状态不变。adapter_only 时新动作头也保持可训练。

checkpoint 保存完整动作配置。严格加载/续训拒绝模式或动作设置不一致。显式 --init-checkpoint 可迁移共享骨干；动作设置改变时重新初始化回归头，必须重新训练。不能把一个已经训练好的 EEF checkpoint 仅改 config 就当作 joint 模型。

`--resume` 还要求解析继承后的其余训练配置一致，仅允许变更 `optimizer_steps`、`checkpoint_interval` 和 `num_workers`；学习率、batch size、梯度累积、冻结设置和 seed 等不能变更。replay 契约、role cache 和卡数也必须一致。已达到总步数时需增加 `optimizer_steps`；GPU RNG 尚未恢复，不承诺逐位复现。

使用 [OHT full 8×40 epoch 配置](../../finetune/OHT/configs/oht_full_8x40.yaml) 时，可额外调整 `epochs`、`max_optimizer_steps` 和 `checkpoint_every_epochs`；`train_iter`、`global_batch_size` 和模型/动作设置仍须一致。关节/相对模式继承该配置时必须设置 `agent.transform_augmentation: false`。完整预算、冻结策略、loss/epoch 日志及保存方式见 [OHT README](../../finetune/OHT/README.md)。

`train_visualization` 仅控制训练图像诊断，四种动作模式均支持，可在续训时调整；它不参与优化器连续性指纹。关节模式显示的预测定位热图属于辅助 EEF 头，不能当作关节预测图。

## 推理与执行

服务自动使用训练 checkpoint 内的动作模式，不在启动时覆盖头语义：

~~~bash
python -m finetune.OHT.server \
  --checkpoint /data/oht/runs/joint-s0/model_last.pth \
  --port 8010
~~~

EEF 保持原 absolute_tcp_v1 协议；其余模式使用 bridgevla_oht_action_v2。GET /health 返回 action_spec，包含模式、维度、单位、关节顺序或相对坐标约定。客户端和服务端模式/坐标不匹配时拒绝请求。

~~~python
from finetune.OHT.config import load
from finetune.OHT.action_modes import resolve_action
from finetune.OHT.runtime.transport import Client
from finetune.OHT.runtime.executor import execution_target

options = resolve_action(load("finetune/OHT/configs/joint.yaml"))
client = Client(
    "http://127.0.0.1:8010", contract["sha256"], action_config=options
)
# observation 包含相机观测、low_dim，以及当前 joint_positions。
command = client.act(observation, goal, episode_id, step, measurement_timestamp)
absolute_joint_target = execution_target(
    command, options, current_joints=observation["joint_positions"]
)
~~~

相对 EEF 可用 execution_target(..., current_tcp=observation["tcp_pose"]) 得到绝对 TCP 目标；绝对模式直接返回原命令。仿真也可以直接消费相匹配的原生模式。

上面的 server 与 client 均为 `joint`。切换到 `relative_joint` 时，必须同时使用该模式训练的 checkpoint 和对应训练配置；相对关节命令调用同一个 `execution_target(..., current_joints=...)` 解码，不能仅替换客户端 YAML。

相对命令以对应请求的当前测量为参考，应转换一次并跟踪该目标；不能在跟踪同一目标时重复累加旧 delta。实际 PD/OSC 的归一化、控制周期、夹爪映射和轨迹跟踪仍由 IsaacLab adapter 实现。此改动没有假设现有 H-VLA client 与新协议兼容。

## 评估

open-loop 自动使用 checkpoint 动作模式。EEF 类报告位置/旋转误差；relative_eef 先恢复对应绝对姿态再计算 SO(3) 误差。关节类报告 rad 单位 MAE/RMSE，所有模式报告夹爪准确率。

closed-loop 显式提供对应训练 config，下面与上面的 `joint` server 匹配：

~~~bash
python -m finetune.OHT.eval closed-loop \
  --config finetune/OHT/configs/joint.yaml \
  --server http://127.0.0.1:8010 --replay /data/oht/replay-v3 \
  --environment my_isaac_oht:create --cases /data/oht/eval-cases.jsonl \
  --executor-id verified-joint-v1 \
  --output /data/oht/eval/joint.json
~~~

environment factory 收到的 contract 包含 action_spec，adapter 必须按该模式解释 step(action)。省略 --config 保持旧 EEF 行为。报告记录 action_spec，失败/超时继续计入分母。

## 验证范围

tests/test_oht_action_modes.py 覆盖四类标签、九种相对 EEF 坐标组合、关节 wrap/限位、四动作 × 三物体模式的真实 Agent 更新与推理、HTTP 协商、checkpoint 初始化/不匹配拒绝，以及旧 EEF 缓存兼容。CPU 骨干与渲染器使用小替身，未运行完整 CUDA 或真实机器人闭环。
