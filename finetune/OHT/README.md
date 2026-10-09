# OHT v423 实施与运行

本目录提供 OHT 数据审计、观测/动作缓存、baseline、内部 T/R role queries、可选外部 predicted objects、训练、推理服务和评估入口。命令均从 **BridgeVLA 仓库根目录** 执行。设计背景见 [迁移方案](../../docs/design/oht-data-migration.md)。

已用合成 OHT 数据完成 CPU 集成验证，包括真实 Parquet、MP4 解码、深度点云、真实 Agent 更新及 HTTP 请求。真实 v423 数据、完整 CUDA 模型和 IsaacLab 闭环尚未运行；提供的远程服务器 SSH 连接超时。以下是已实现入口的使用方式，不能据此视为已有 OHT 成功率。

## 1. 环境

数据处理不依赖 RLBench、GemBench、IsaacLab 或 CUDA：

~~~bash
python -m pip install -r finetune/OHT/requirements-data.txt
~~~

工具使用 Python >=3.9。完整训练/推理另需项目已有的 BridgeVLA CUDA 环境，包括 PyTorch、Transformers、PaliGemma 权重和 point-renderer 扩展；版本参考 [原安装说明](../../docs/guides/installation.md)。CPU 测试使用 Python 3.12、PyTorch 2.5.1，不能据此保证其他依赖组合兼容。

--pretrain-path 指 **BridgeVLA 热图预训练权重目录**，与现有 MVT 加载方式相同；H-VLA 的 oht_v423 checkpoint 不能直接作为 BridgeVLA checkpoint。模型构造仍需可访问或已缓存的 google/paligemma-3b-pt-224 基础权重。

若独立准备 OHT 模型环境，先安装与机器 CUDA 匹配的 PyTorch 2.5.1 / torchvision 0.20.1，再执行以下命令。point-renderer 构建需要相匹配的 CUDA toolkit 和编译器；本轮未验证其 GPU 编译。

~~~bash
python -m pip install -r finetune/OHT/requirements-model.txt
python -m pip install --no-build-isolation -e finetune/bridgevla/libs/point-renderer
~~~

OHT 配置显式启用 use_point_renderer，关闭 SE(3) 增强，因此无需为 OHT 安装 RLBench 或加载其 augmentation 的 PyTorch3D 依赖。

## 2. 审计与固定划分

~~~bash
python tools/audit_oht_dataset.py \
  --root /common-data-32t/data/robot_data/oht_curobo_pd_v423 \
  --output /data/oht/audit-seed0.json --seed 0
~~~

直接扫描四个任务的 data/chunk-*/episode_*.parquet，忽略不完整的 LeRobot features 描述和重复的 episodes.jsonl。任务身份为任务名加 episode index。检查必需列、形状、时间单调性、夹爪命令、相机引用文件、四元数及成功标记，记录 EE 范围、时间间隔、原始 action 零值率等。无成功标记或 metadata 明确失败的 episode 排除并报告；存在坏 episode 时命令返回非零，审查报告后再决定是否使用保留的 episode。

默认按每任务 80/10/10 的目标比例划分。相同轨迹和显式关联场景必须在同一集合，组过大时实际数量可偏离目标。可添加 --groups scene-groups.json：

~~~json
{"assemble_left/0": "scene-A", "disassemble_left/12": "scene-A"}
~~~

这里只是映射格式示例，不能推断真实场景对应关系。没有显式场景映射时，只能检测所用状态/EE 轨迹完全相同的重复，不能保证不同轨迹的同场景已隔离。audit 检查视频引用和时间戳格式；实际视频 PTS 对齐在缓存构建时检查。

## 3. 标定与构建共用缓存

复制 [dataset.yaml](configs/dataset.yaml) 为新的本地配置，填写：

- scene_bounds：现场验证的世界系边界，顺序 xmin,ymin,zmin,xmax,ymax,zmax。
- link_to_tcp：数据 EE link 到策略 TCP 的 4×4 刚体变换；若参考点相同，显式填写单位阵。
- 每相机 optical_to_sensor：将 optical 坐标点变换到数据相机 pose 所指 sensor 坐标系，不能因为数据四元数是 xyzw 就假设单位阵。
- depth.encoding、depth.kind：分别填写实际编码和 z-depth/ray-distance 类型。

默认 image_size=[120,160] 对原始 480×640 做严格 4 倍步长采样，K 同步缩放。这是输入 RGB-D 尺寸，模型的虚拟渲染图像仍用现有 MVT 配置。

建议使用可校准的米制深度 sidecar：

~~~yaml
depth:
  encoding: metric
  kind: z
  path_pattern: "{task}/metric_depth/{episode:06d}/{camera}/{frame:06d}.npy"
  limits: [0.001, 10.0]
~~~

每个 NPY 是与原 RGB 对齐的浮点 H×W **米制**深度。整数 PNG/TIFF 可用 scaled_integer，必须显式给出 scale（米/整数单位）、可选 offset 和 invalid_values。无 sidecar 时只支持 exporter 已明确证明可恢复的 linear_channel MP4 编码，必须声明 channel/scale/offset。普通可视化深度和未知 H.264 编码不能恢复可靠几何，不应填猜测值绕过检查。

~~~bash
python tools/build_oht_replay.py \
  --root /common-data-32t/data/robot_data/oht_curobo_pd_v423 \
  --manifest /data/oht/audit-seed0.json \
  --config /data/oht/dataset-calibrated.yaml \
  --output /data/oht/replay-v1 --sample-stride 10 \
  --visualize-every 100 \
  --visualize-output-dir /data/oht/previews/replay-v1

python tools/validate_oht_replay.py --replay /data/oht/replay-v1
~~~

实现行为：

- 按 Parquet 引用时间戳选择最近视频 PTS，超出容差即报错；腕部 pose 按当前帧处理。
- 米制 RGB-D → optical XYZ → world XYZ；无效点用 NaN，进入 Agent 时按边界过滤。
- 用测得的世界系 EE 轨迹与 TCP 变换重建下一关键点绝对目标，**不使用原始 action 前六维作标签**。
- 原始 action 最后一维仅重建夹爪意图：+1 关、−1 开、0 保持；网络输出为 0 关、1 开。
- 关键点包含夹爪/指令边界前后帧、位移/转角/帧距阈值与终帧；在线不输入 instruction_id。
- 语言为统一任务目标，low_dim 为当前测量夹爪和两指兼容特征。collision 标签仅占位，损失权重固定 0。
- contract.json、samples.jsonl、观测 NPZ、complete.json 分开存储，校验哈希、未来目标关系及分组划分。

缓存构建使用新目录；失败目录没有有效 complete 标记，不能当作完成缓存。工具拒绝覆盖已有输出。建议先在保留相同目录结构的小样本副本上完成审计/构建；缓存需要额外磁盘空间，应根据实际样本率估算。

### 生成时可视化

`--visualize-every N` 按每个 episode **生成的样本数**保存 PNG，包含第一个样本，随后每隔 N 个；不是按原始视频帧计数。`1` 显示每个生成样本，默认 `0` 关闭。可省略 `--visualize-output-dir`，此时保存到 `<output>/visualizations/<task>/<episode六位>/<frame六位>.png`。预览文件存在时拒绝覆盖。

每张图包含缓存分辨率的各相机 RGB、米制 depth、XY/XZ/YZ 彩色点云及当前/未来 TCP：青色为当前 TCP，品红色为下一关键点动作目标。depth 蓝色近、红色远，标注每相机当前有效深度范围（米），黑色表示无效；不同图的深度颜色范围可能不同。三视图固定使用 `scene_bounds` 世界坐标范围，显示点最多均匀抽取 20,000 个，属于 CPU 诊断投影，不是模型的虚拟 renderer。

PNG 单独输出，不新增 observation/label 字段，也不改变 buffer contract 或训练样本。无需模型、CUDA 或图形桌面；仅使用数据环境已有的 NumPy/Pillow。基础 buffer 没有 T/R 标注，角色预览在下一节的 teacher 构建时生成。

## 4. Baseline 训练

先做 CPU 数据预检：

~~~bash
python -m finetune.OHT.train \
  --config finetune/OHT/configs/baseline.yaml \
  --replay /data/oht/replay-v1 --output /data/oht/check-only \
  --validate-only
~~~

然后在 BridgeVLA GPU 环境做两步 smoke，再正式训练：

~~~bash
python -m finetune.OHT.train \
  --config finetune/OHT/configs/smoke.yaml \
  --replay /data/oht/replay-v1 --output /data/oht/runs/smoke \
  --pretrain-path /path/to/bridgevla-heatmap-pretrain

python -m finetune.OHT.train \
  --config finetune/OHT/configs/baseline.yaml \
  --replay /data/oht/replay-v1 --output /data/oht/runs/baseline-s0 \
  --pretrain-path /path/to/bridgevla-heatmap-pretrain
~~~

多卡将 python -m finetune.OHT.train 换成 torchrun --nproc_per_node=2 -m finetune.OHT.train，保留其余参数。

默认每卡 batch 2、累积 8、2,000 个优化步，任务均匀采样；有效 batch = 每卡 batch × 累积次数 × 卡数。这是初始配置，尚未经真实 OHT 收敛或显存验证。比较方法保持相同预算、冻结设置、数据划分和 seeds。

输出 config.json、contract.json、metrics.jsonl、model_last.pth 及最终步 checkpoint。--resume 恢复严格匹配的模型/优化器和采样位置，允许增加总步数；必须保持 replay、role cache、配置和卡数一致。GPU RNG 尚未恢复，不承诺逐位复现。--init-checkpoint 仅初始化权重；只允许物体 adapter/predictor 的增删或尺寸变化，骨干不匹配时报错。

## 5. 内部 predicted assistance：首选路线

[role_queries.yaml](configs/role_queries.yaml) 复用现有两角色预测器：由当前渲染特征与任务上下文预测 Target/Reference，在 coarse/refine 间继承几何与角色 token。教师只监督角色图和 Reference NULL，推理时不输入 GT 对象。

每个 replay sample 的教师需要显式标注。annotations.jsonl 的一行示例：

~~~json
{"id":"assemble_left/000000/000000","target":{"present":true,"known":true,"source":"visible_surface","mask_path":"masks/assemble_left-000000-000000.npz"},"reference":{"present":false,"known":true,"source":"none"}}
~~~

Reference=NULL **仅示范格式**，实际任务必须逐阶段定义正确角色。NPZ 以相机名为 key，值是与缓存同尺寸的 bool mask；取当前可见表面 world XYZ 后重采样。原尺寸 mask 应按缓存相同的整数步长采样。

可用 site_region 标注接收位置：提供 world_from_site 4×4 和 size 三维正长度，生成带类型声明的 site 区域点。这是特权训练标注，不能把物体中心伪装成表面，也不能当作在线预测结果。

| 语义状态 | present | known | source |
| --- | --- | --- | --- |
| 已知语义缺失 | false | true | none |
| 存在但遮挡/几何不可用 | true | true | unknown |
| 角色分配未确认 | 显式填写 | false | unknown |

未知和无可见几何不产生伪表面监督。教师构建器不会从 objects_pos/held_obj/instruction_id 自动猜测语义，也未实现原始 mesh 或仿真 segmentation 的批量导出；需从已验证的标注/采集端提供。

~~~bash
python tools/build_oht_role_teacher.py \
  --replay /data/oht/replay-v1 --annotations /data/oht/annotations.jsonl \
  --output /data/oht/teachers-v1 --point-count 512 \
  --visualize-every 100

python tools/validate_oht_replay.py \
  --replay /data/oht/replay-v1 --mode role_queries \
  --role-cache /data/oht/teachers-v1

python -m finetune.OHT.train \
  --config finetune/OHT/configs/role_queries.yaml \
  --replay /data/oht/replay-v1 --role-cache /data/oht/teachers-v1 \
  --output /data/oht/runs/role-queries-s0 \
  --init-checkpoint /data/oht/runs/baseline-s0/model_last.pth
~~~

Teacher 使用相同的可视化参数，默认输出到 `<teacher-output>/visualizations/`。Target 为绿色、Reference 为蓝色、重叠为黄色；标注 mask 区域显示 `30% 原始 RGB + 70% 角色颜色`，背景保留 RGB。`site_region` 显示区域点轮廓，不能当作已验证可见的 mask。图中同时列出角色的 source、present、known 与几何有效性，NULL/unknown 不伪造点。旧 replay 未缓存当前 TCP，teacher 预览明确显示 `not cached`，未来 TCP 仍来自动作标签。

上例表示从已训练 baseline 继续训练的辅助实验，有额外训练预算。公平对照应让 baseline 从相同初始 checkpoint 继续相同优化步数，或两者均从相同预训练初始化各训同样预算。正式实验用三个训练 seeds，不应把继续训练收益全部归因于物体辅助。

## 6. 外部 predicted objects：可选路线

实现了严格的缓存和在线 wrapper 接口，**未附带新的检测/分割模型或真实预测权重**。应用方提供 module:factory，factory 无参返回 callable：

~~~python
class Predictor:
    def reset(self):
        pass  # 可选；清除 episode 状态

    def __call__(self, observation, goal):
        # observation 仅含当前 low_dim、每相机 RGB/depth/world XYZ/K/外参。
        # 使用真实预测结果；不读取仿真对象状态、教师或未来帧。
        return role_fields(
            "predicted", points, valid, present, confidence=confidence
        )
~~~

导入 role_fields 自 finetune.OHT.data.role_cache。points 为有限 float32 [2,512,3] world XYZ，valid/present 为 bool [2]，confidence 为 [2] 且范围 [0,1]。顺序固定 Target/Reference。缺失几何应 valid=false；语义存在不能因为遮挡改成 present=false。模型既有门控会在必需 Reference 不可用时禁用整对辅助。

provenance YAML 至少含 model_sha256（真实感知权重 SHA256）和 training_split（训练数据说明），工具附加 factory 并写入缓存/训练 checkpoint。它记录调用方声明，不能独立证明检测器未用测试集。若感知器在 OHT 上训练，只能使用训练划分。

~~~bash
python tools/predict_oht_objects.py \
  --replay /data/oht/replay-v1 --predictor my_oht_predictor:create \
  --provenance /data/oht/predictor-provenance.yaml \
  --output /data/oht/predictions-v1

python -m finetune.OHT.train \
  --config finetune/OHT/configs/predicted_external.yaml \
  --replay /data/oht/replay-v1 --role-cache /data/oht/predictions-v1 \
  --output /data/oht/runs/predicted-external-s0 \
  --init-checkpoint /data/oht/runs/baseline-s0/model_last.pth
~~~

训练缓存须覆盖训练样本；验证时各模式检查对应集合的缓存覆盖。离线预测按缓存采样帧顺序、episode 开始时 reset，在线按控制采样频率调用；有历史状态的预测器必须对齐两者采样策略，否则应先用无状态预测器。在线服务要求 factory/provenance 与训练缓存完全匹配。

## 7. 推理服务与 IsaacLab 接入

~~~bash
python -m finetune.OHT.server \
  --checkpoint /data/oht/runs/baseline-s0/model_last.pth \
  --host 127.0.0.1 --port 8010
~~~

role queries 使用同一命令。external checkpoint 额外提供 --predictor、--predictor-provenance。GET /health 返回 ready 和 contract hash。

服务协议为 bridgevla_oht_absolute_tcp_v1，与 H-VLA 现有协议不同。请求含 episode、step、测量时间戳、固定 goal、contract hash、typed tensors；输出 [x,y,z,qx,qy,qz,qw,gripper_open]，**world/TCP/xyzw**。同一 episode 从 step0 开始严格递增，时间戳递增；新 episode 清除策略和感知状态。超时不自动重试同一控制请求。一个 server 顺序服务一个 client。

仿真适配器通过现有函数构造与训练一致的观测：

~~~python
from finetune.OHT.data.actions import low_dim
from finetune.OHT.data.observation import camera_observation
from finetune.OHT.runtime.transport import Client

observation = {"low_dim_state": low_dim(measured_open_fraction, finger_joints)}
for name in data_config["cameras"]:
    observation.update(camera_observation(
        name, rgb[name], metric_depth[name], sensor_pose_world_xyzw[name], data_config
    ))
client = Client("http://127.0.0.1:8010", contract["sha256"])
absolute_target = client.act(observation, goal, episode_id, control_step, simulation_timestamp)
~~~

返回绝对目标交给共同执行器。runtime/executor.py 中 relative_eef 提供 world→base/body 位移与旋转向量误差，并要求显式位移/旋转限幅。真实 client 的尺度、积分周期、axis-angle/Euler 约定、夹爪转换、目标到达策略和在线 IK 必须由 IsaacLab 端核对实现；该函数不是完整控制器。

没有复制 H-VLA server、猜测两个端口的模型分工或修改远端脚本。保留用户提供的 Task1/2 脚本名，尤其 Task1 的 client 名中仍含 task2；真实映射和 joint+relative_eef 路由待远程核查。

## 8. 离线与闭环评估

~~~bash
python -m finetune.OHT.eval open-loop \
  --checkpoint /data/oht/runs/baseline-s0/model_last.pth \
  --replay /data/oht/replay-v1 --split val \
  --output /data/oht/eval/baseline-val.json
~~~

离线报告逐任务位置误差、旋转误差、夹爪准确率和模型耗时；这些不能代替闭环成功率。内部角色模式没有教师输入，即使 evaluation 数据行存在标签也会被策略白名单过滤。

闭环工具通过 --environment module:factory 接入本机 IsaacLab 适配器。factory(contract) 返回：

- reset(case) → {observation, goal, timestamp}，按固定 seed 重置并恢复执行器状态。
- step(absolute_action8) → 下一 {observation, goal, timestamp, done, success}；终止时必须显式给出 bool success。
- 可选 close() 释放环境。

适配器拥有实际 EEF / joint+EEF 执行逻辑；GT 仅用于 reset 和 success 评分，不能放入策略/感知输入。先用专家重建目标完成执行器校验，再比较模型。

~~~bash
python -m finetune.OHT.eval closed-loop \
  --server http://127.0.0.1:8010 --replay /data/oht/replay-v1 \
  --environment my_isaac_oht:create --cases /data/oht/eval-cases.jsonl \
  --executor-id verified-eef-executor-v1 --max-steps 500 \
  --output /data/oht/eval/baseline-eef-s0.json
~~~

case JSONL 每行含唯一 id、task、seed；id 每次运行需使用新 episode 标识或重启 server。所有方法用同一份 cases、执行器版本和时间预算，分别报告 B-E/P-E，必要时 B-H/P-H。异常、超时和未完成 episode 均保留在成功率分母。报告记录 cases hash、契约和执行器 ID，便于检查对照条件。

用户提供的 H-VLA 19/20 与 14/20 是历史参考；本实现尚无真实 OHT baseline 或 assistance 成绩。

## 9. 本地验证

~~~bash
python -m pytest -q tests/test_oht_migration.py
~~~

覆盖 12 个合成 episode → 60 条 transitions、五相机、视频 PTS、米制 depth、划分检查、教师/预测缓存、无 GT 推理隔离、真实 Agent 梯度累积与零碰撞损失、HTTP 协议、闭环失败计数和续训采样。backbone/render 使用 CPU 小替身，未验证完整 PaliGemma/point-renderer GPU 前向。另运行现有角色预测、跨尺度继承、辅助损失、前向与优化器回归测试。

验证结果：OHT 集成测试 20 项通过，相关既有回归测试 137 项及 6 项子测试通过，合计 157 项及 6 项子测试。所有新增命令的帮助入口与 Python 编译检查通过。
