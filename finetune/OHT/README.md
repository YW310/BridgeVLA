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

直接扫描四个任务的 data/chunk-*/episode_*.parquet，忽略不完整的 LeRobot features 描述和重复的 episodes.jsonl。任务身份为任务名加 episode index。检查必需列、形状、时间单调性、相机引用文件、四元数及成功标记，记录 EE 范围、时间间隔、原始 action 零值率和夹爪命令分布（仅诊断，不要求可信）等。无成功标记或 metadata 明确失败的 episode 排除并报告；存在坏 episode 时命令返回非零，审查报告后再决定是否使用保留的 episode。

每个 episode 内相同视频路径只做一次路径安全/文件存在性检查，所有帧的引用与时间戳仍逐一验证，Parquet SHA-256 和分组划分不变。CLI 默认在 stderr 显示 episode 进度、单集/累计耗时及有效/无效数；`--quiet` 关闭进度，stdout 始终保留最终 JSON 摘要。已有输出会在扫描前立即拒绝覆盖。原始数据未变、仅修改相机解码配置时，可复用已有有效 audit，无需重跑。

默认按每任务 80/10/10 的目标比例划分。相同轨迹和显式关联场景必须在同一集合，组过大时实际数量可偏离目标。可添加 --groups scene-groups.json：

~~~json
{"assemble_left/0": "scene-A", "disassemble_left/12": "scene-A"}
~~~

这里只是映射格式示例，不能推断真实场景对应关系。没有显式场景映射时，只能检测所用状态/EE 轨迹完全相同的重复，不能保证不同轨迹的同场景已隔离。audit 检查视频引用和时间戳格式；实际视频 PTS 对齐在缓存构建时检查。

## 3. 仿真数据契约与构建共用缓存

以用户提供的已运行 `convert_oht_lerobot_v21_to_tfds.py → transforms.py → dataset.py` 为本轮读取参考，无需重新标定或采集。使用 [dataset.yaml](configs/dataset.yaml)；实际参数及 metadata SHA256 写入 contract 的 `source_data_configs`，各任务分别读取，不串用左右相机参数。

- 内参来自各 `<task>/lerobot_dataset/meta/camera_intrinsics.json`，支持大小写相机名和 `cameras.<name>.intrinsic`；缺失时报错，不再猜局部相机焦距。使用 YAML 内参时设 `intrinsics_source: config` 并填完整 K。
- 原始 camera/EE 四元数默认 **wxyz**，分别由 `camera_quaternion_order` / `ee_quaternion_order` 指定；策略输出仍为 **world/TCP/xyzw**。参考转换器的默认值与早期“EE 为 xyzw”说明冲突；若成功运行时用了 `--input-quat-order xyzw`，这里也应显式设 xyzw，不能从数值猜格式。物体四元数未用于当前动作标签。
- `ee_pos_world` 直接作为已含控制器偏移的 TCP，`link_to_tcp` 保持单位阵，不重复添加约 12 cm 偏移。
- 夹爪从实测 `observation.state[6]` 读取，用数据集全局 stats/info 的 open=min、close=max 归一化。反向端点可用 `gripper.source: config` 显式指定；不按 episode 重估端点。采用参考 `binarize_gripper_hysteresis_with_diff()` 的因果规则：open01 大于 0.9 为 open、小于 0.1 为 close，中间根据 ±0.05 差分切换，否则保持上一状态。**raw action 全部仅用于审计，不生成监督。**

`scene_bounds` 默认设为 `[-0.5, -1.0, 0.3, 1.5, 1.0, 2.0]`，顺序 xmin,ymin,zmin,xmax,ymax,zmax，单位米；覆盖 v423 说明中的物体与路点示例并留余量。这是初始策略工作区，不是全量统计结果；训练前检查实际 EE/物体覆盖与动作越界，越界时调整配置，不静默裁剪 GT。audit 的 EE 范围仅是参考，不能替代完整物体范围。

手工去除墙壁等背景可配置 `point_cloud_filter`，不改变动作空间。仓库 YAML 已启用以下**经验保留区**，覆盖已记录的物体/TCP 示例并留余量，不是全量验证过的安全范围或实测墙壁边界：

~~~yaml
point_cloud_filter:
  enabled: true
  keep_bounds: [-0.3, -0.8, 0.4, 1.2, 0.8, 1.9]
  exclude_boxes: []
~~~

先预览四个任务的抓取、放置和最高抬升时刻，确认物体、夹爪及 Reference 未被误删，再用于正式训练；不能保证这个 ROI 去掉所有墙壁。`keep_bounds: null` 不额外裁剪，`enabled: false` 关闭手工过滤。`exclude_boxes` 可填多个排除框，点落入任意一个即移除；未确认墙壁位置前保持空列表。均为 world 米制 `[xmin,ymin,zmin,xmax,ymax,zmax]`，下界包含、上界不包含。`point_filter.point_cloud_mask()` 在投影前筛选 XYZ，被排除点记为 NaN，保留 RGB-D 像素对应关系；原始 RGB、米制 depth、TCP/动作/角色 presence 标签不改。原始相机图仍显示墙壁，只有正交点云图去除它。

`camera_observation()` 将过滤后的 XYZ 写入 replay；训练沿用这些点进入 coarse/refine。在线 `Policy.act()` 在外部 predictor 和 agent 前使用相同过滤，CPU 全局/GT 局部预览也共用规则。单相机过滤为空允许；全场无工作区点则报错，不退回未过滤输入。可见面 teacher 若因此没有有效点，保留角色 present，仅将几何标为不可用，不改成 NULL。

二阶段的局部缩放/坐标变换没有改，也不在局部坐标中再次套用世界 ROI；但它继承过滤后的点云，被删的表面无法恢复。过滤还可能改变 coarse 预测及后续裁剪中心。当前不是“仅 coarse 过滤、refine 读原始点云”的模式，必须给操作局部范围留余量。

过滤配置写入各 data profile 和 replay/checkpoint 契约；旧配置缺少此块仍关闭。修改参数需在新目录重建 replay 及 teacher/预测缓存，再进行匹配的训练/推理；原始数据与 audit 可复用。只修改 YAML 不会改变旧缓存或已加载 checkpoint，不要手改 contract 来绕过一致性检查。

原始相机外参为 camera→world、USD/OpenGL 轴，按显式顺序转换到内部 xyzw。五个相机的 `optical_to_sensor` 均为 `diag(1,-1,-1,1)`：`T_world_optical = T_world_usd @ optical_to_sensor`。缓存外参是 4×4 optical→world；投影使用 `(P_world - t) @ R_world_optical`，不再次转置或翻轴。

深度为无损 HEVC `gray12le`，但 **12-bit 像素不等于毫米值**。参考转换器从 `meta/info.json` 的 `features.observation.depth.<camera>.info` 读取固定量化参数（`video.*` 或普通 key）。设 `n=q/qmax`：对数逆变换为 `exp(n*(log(max+shift)-log(min+shift))+log(min+shift))-shift`，线性为 `n*(max-min)+min`。YAML 默认与参考代码一致：min=0.01、max=10、shift=3.5、use_log=true、qmax=4095；metadata 中存在的字段覆盖默认值。

`depth.path_pattern: null` 通过 Parquet Path/Timestamp 读取原生灰度视频，不经过 RGB；`0` 为无效，**qmax=4095 对应 depth_max，不自动屏蔽**。默认对数编码下 q=1024 约为 1.416 m，并非 1.024 m。参考 TFDS 为 PNG 四舍五入到毫米；这里缓存浮点米制，不额外舍入。

`depth.kind: ray` 暂沿用数据方描述，也支持显式 `z`。提供的 `dataset.py` 将实际反投影交给未提供的 `pointcloud_transforms.py`，因此本轮没有声称已与它验证 ray/Z 等价；gray12le 或量化公式本身不能决定深度类型。

仓库配置更新不会修改此前复制的 `dataset-calibrated.yaml`。构建始终读取 `--config` 指定的文件；请同步所需字段，保留本地已有的正确值。

**旧缓存迁移**：本轮 schema 为 `oht_bridgevla_v2`；v1 replay/checkpoint 不作为已修复数据使用。用新配置在新目录（如 `replay-source-v2`）重建 replay，再重建 teacher/预测缓存。原始视频、Parquet 和已有 audit 可复用，不删除旧数据、不重采集；修改旧 contract 或只重画 PNG 无法修复旧 XYZ/rotation/gripper 标签，也不能沿用旧 optimizer resume。

数据方提供的 frame 965 腕部投影参考值为原图 `(313.0, 378.6)`，在默认 4 倍步长缓存中应约为 `(78.25, 94.65)`。本地没有该原始样本，仍需在服务器复核多帧、多相机，不以“落在图内”代替几何对齐验证。

默认 image_size=[120,160] 对原始 480×640 做严格 4 倍步长采样，K 同步缩放。这是输入 RGB-D 尺寸，模型的虚拟渲染图像仍用现有 MVT 配置。

当前默认深度配置：

~~~yaml
depth:
  encoding: quantized
  pixel_format: gray12le
  metadata: true
  depth_min: 0.01
  depth_max: 10.0
  shift: 3.5
  use_log: true
  qmax: 4095
  kind: ray
  invalid_values: [0]
  path_pattern: null
  limits: [0.001, 10.0]
~~~

读取器检查实际像素格式，保留 12-bit 数值及视频 PTS；旧 PyAV 的 gray12le 数组兼容路径直接读取含行 padding 的 uint16 平面。仍兼容浮点 NPY（metric）、整数 PNG/TIFF（scaled_integer）和显式指定通道/scale/offset 的旧 linear_channel 编码。普通可视化视频不能据此当作米制深度。

~~~bash
python tools/build_oht_replay.py \
  --root /common-data-32t/data/robot_data/oht_curobo_pd_v423 \
  --manifest /data/oht/audit-seed0.json \
  --config finetune/OHT/configs/dataset.yaml \
  --output /data/oht/replay-source-v2 --sample-stride 10 \
  --visualize-every 100 \
  --visualize-output-dir /data/oht/previews/replay-source-v2

python tools/validate_oht_replay.py --replay /data/oht/replay-source-v2
~~~

实现行为：

- 按 Parquet 引用时间戳选择最近视频 PTS，超出容差即报错；腕部 pose 按当前帧处理。
- 米制 RGB-D → optical XYZ → world XYZ；无效点用 NaN，进入 Agent 时按边界过滤。
- 用未来关键点的实测 world TCP 位姿与实测二值夹爪状态生成全部标签；**原始 action 的所有分量均不参与标签生成**。网络夹爪输出为 0 关、1 开。
- 不照搬参考转换器“仅按 EE 平移删静止帧”的采样，以免删掉原地夹爪/旋转动作。
- 默认 `keypoints.method: gripper`：**只取实测夹爪经因果滤波后的二值开闭切换帧**。不按停稳、指令、位移/转角补中间 goal，也不补终帧；开闭切换并不代表物体已成功抓住/释放。
- `--sample-stride` 只控制输入观测采样；每个输入指向严格晚于它的下一开闭事件，事件输入帧也保留用于预测再下一事件。最后一次切换及其后尾段没有未来事件，不生成样本；无切换的 episode 跳过并记录在 `complete.json.skipped_episodes`，全无事件则报错，不写完成标记。
- `bridgevla` 对照模式仍保留开闭/停稳/终帧，停稳初值为 `0.01 m/s`、`5°/s`（实际 dt）；旧配置未指定 method 时仍为 `geometric`（指令/位移/转角/帧距）。切换须用新版仓库 YAML 或替换外部 YAML 的整个 `keypoints` 块，再建新 replay/teacher；原始数据/audit 可复用。
- 预览里的 goal 是未来动作 TCP，不是语言任务目标。只按开闭取点不再监督释放后的撤退或无夹爪切换的路径转折，不能据此保证这些端点可安全直接执行。
- 语言为统一任务目标，low_dim 为当前测量夹爪和两指兼容特征。collision 标签仅占位，损失权重固定 0。
- contract.json、samples.jsonl、观测 NPZ、complete.json 分开存储，校验哈希、未来目标关系及分组划分。

缓存构建使用新目录；失败目录没有有效 complete 标记，不能当作完成缓存。工具拒绝覆盖已有输出。建议先在保留相同目录结构的小样本副本上完成审计/构建；缓存需要额外磁盘空间，应根据实际样本率估算。

### 生成时可视化

`--visualize-every N` 按每个 episode **生成的样本数**保存 PNG，包含第一个样本，随后每隔 N 个；不是按原始视频帧计数。`1` 显示每个生成样本，默认 `0` 关闭。可省略 `--visualize-output-dir`，此时保存到 `<output>/visualizations/<task>/<episode六位>/<frame六位>.png`。预览文件存在时拒绝覆盖。

每张图包含缓存分辨率的各相机 RGB、米制 depth，以及两排 XY/XZ/YZ 彩色点云：青色为当前 TCP，品红色为下一 GT 关键点。depth 蓝色近、红色远，标注当前有效范围（米），黑色表示无效；不同图的深度颜色范围可能不同。

每个 RGB 面板下方显示 TCP/goal 的 `in view`、`outside image`、`behind camera` 或 `unavailable`，以及缓存像素坐标/光学 Z。画外点不强制移到边缘；`in view` 仅代表在视锥内，不代表没有被遮挡。

- 全局三视图使用 `scene_bounds`，最多显示 200,000 个点（默认五相机 120×160 的全部有效点都在预算内）。3×3 像素小面积绘制按深度处理重叠，仅改善显示空洞；品红色方框标出局部立方体的投影范围。
- 局部三视图以 GT keypoint 为中心，各轴 ±0.20 m，显示米制坐标范围。先从完整有效点云选择局部点，再独立限制显示点数，避免全局抽样漏掉小物体；没有观测点时明确提示，不补造几何。

局部图标注 **GT-centered refine diagnostic (NOT model stage2)**：它不是模型 coarse 预测或带噪训练中心产生的二阶段视图。模型真实 coarse/refine renderer 图应在训练/推理前向中另行导出。两排都显示抽样前后点数；缓存的 4 倍步长采样不因预览变密而恢复到原始分辨率。

PNG 单独输出，不新增 observation/label 字段，也不改变 buffer contract 或训练样本。无需模型、CUDA 或图形桌面；仅使用数据环境已有的 NumPy/Pillow。基础 buffer 没有 T/R 标注，角色预览在下一节的 teacher 构建时生成。

## 4. Baseline 训练

先做 CPU 数据预检：

~~~bash
python -m finetune.OHT.train \
  --config finetune/OHT/configs/baseline.yaml \
  --replay /data/oht/replay-source-v2 --output /data/oht/check-only \
  --validate-only
~~~

然后在 BridgeVLA GPU 环境做两步 smoke，再正式训练：

~~~bash
python -m finetune.OHT.train \
  --config finetune/OHT/configs/smoke.yaml \
  --replay /data/oht/replay-source-v2 --output /data/oht/runs/smoke \
  --pretrain-path /path/to/bridgevla-heatmap-pretrain

python -m finetune.OHT.train \
  --config finetune/OHT/configs/baseline.yaml \
  --replay /data/oht/replay-source-v2 --output /data/oht/runs/baseline-s0 \
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
  --replay /data/oht/replay-source-v2 --annotations /data/oht/annotations.jsonl \
  --output /data/oht/teachers-v1 --point-count 512 \
  --visualize-every 100

python tools/validate_oht_replay.py \
  --replay /data/oht/replay-source-v2 --mode role_queries \
  --role-cache /data/oht/teachers-v1

python -m finetune.OHT.train \
  --config finetune/OHT/configs/role_queries.yaml \
  --replay /data/oht/replay-source-v2 --role-cache /data/oht/teachers-v1 \
  --output /data/oht/runs/role-queries-s0 \
  --init-checkpoint /data/oht/runs/baseline-s0/model_last.pth
~~~

Teacher 使用相同的可视化参数，默认输出到 `<teacher-output>/visualizations/`。Target 为绿色、Reference 为蓝色、重叠为黄色；标注 mask 区域显示 `30% 原始 RGB + 70% 角色颜色`，背景保留 RGB。`site_region` 只显示区域点轮廓。图中列出角色 source/present/known/几何有效性，NULL/unknown 不伪造点。v2 在样本索引中保存当前 TCP，仅供诊断；teacher 预览可同时显示当前 TCP 和未来 goal，该字段不进入训练 batch。

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
  --replay /data/oht/replay-source-v2 --predictor my_oht_predictor:create \
  --provenance /data/oht/predictor-provenance.yaml \
  --output /data/oht/predictions-v1

python -m finetune.OHT.train \
  --config finetune/OHT/configs/predicted_external.yaml \
  --replay /data/oht/replay-source-v2 --role-cache /data/oht/predictions-v1 \
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
import numpy as np
from finetune.OHT.data.actions import low_dim, gripper_step
from finetune.OHT.data.observation import camera_observation
from finetune.OHT.runtime.transport import Client

# Select the source_data_configs profile matching the online task/dataset.
data_config = contract["source_data_configs"][source_profile]["data_config"]
# episode reset: previous_fraction = previous_binary = None
g = data_config["gripper"]
measured_open_fraction = float(np.clip((g["close"] - measured_motor_position) / (g["close"] - g["open"]), 0, 1))
binary = gripper_step(measured_open_fraction, previous_fraction, previous_binary, g)
observation = {"low_dim_state": low_dim(measured_open_fraction, binary_state=binary)}
previous_fraction, previous_binary = measured_open_fraction, binary
for name in data_config["cameras"]:
    observation.update(camera_observation(
        name, rgb[name], metric_depth[name], sensor_pose_world_wxyz[name], data_config
    ))
client = Client("http://127.0.0.1:8010", contract["sha256"])
absolute_target = client.act(observation, goal, episode_id, control_step, simulation_timestamp)
~~~

上例的 pose 是与 Parquet 相同的 USD camera→world 原始 wxyz 格式，需匹配 checkpoint 的 `data_config.camera_quaternion_order`。若在线 SDK 返回 xyzw，应在调用前显式重排为 wxyz；已生成的 optical→world 缓存矩阵不能再次做这一步。

使用 contract 中已解析的 K/端点，不直接把含 null 的原始 YAML 交给 `camera_observation()`。`metric_depth[name]` 必须已按同一深度契约解码成米，不能再次解量化。low_dim 的两个开度特征为 `open01 × 0.04` 的兼容量，不是已标定的真实两指米制距离；逐帧二值夹爪采用与训练相同的因果规则，并在 episode reset 清空状态。

返回绝对目标交给共同执行器。runtime/executor.py 中 relative_eef 提供 world→base/body 位移与旋转向量误差，并要求显式位移/旋转限幅。真实 client 的尺度、积分周期、axis-angle/Euler 约定、夹爪转换、目标到达策略和在线 IK 必须由 IsaacLab 端核对实现；该函数不是完整控制器。

没有复制 H-VLA server、猜测两个端口的模型分工或修改远端脚本。保留用户提供的 Task1/2 脚本名，尤其 Task1 的 client 名中仍含 task2；真实映射和 joint+relative_eef 路由待远程核查。

## 8. 离线与闭环评估

~~~bash
python -m finetune.OHT.eval open-loop \
  --checkpoint /data/oht/runs/baseline-s0/model_last.pth \
  --replay /data/oht/replay-source-v2 --split val \
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
  --server http://127.0.0.1:8010 --replay /data/oht/replay-source-v2 \
  --environment my_isaac_oht:create --cases /data/oht/eval-cases.jsonl \
  --executor-id verified-eef-executor-v1 --max-steps 500 \
  --output /data/oht/eval/baseline-eef-s0.json
~~~

case JSONL 每行含唯一 id、task、seed；id 每次运行需使用新 episode 标识或重启 server。所有方法用同一份 cases、执行器版本和时间预算，分别报告 B-E/P-E，必要时 B-H/P-H。异常、超时和未完成 episode 均保留在成功率分母。报告记录 cases hash、契约和执行器 ID，便于检查对照条件。

用户提供的 H-VLA 19/20 与 14/20 是历史参考；本实现尚无真实 OHT baseline 或 assistance 成绩。

## 9. 本地验证

~~~bash
python -m pytest -q tests/test_oht_point_filter.py tests/test_oht_keypoints.py tests/test_oht_audit.py tests/test_oht_dataset_config.py tests/test_oht_depth_video.py tests/test_oht_migration.py tests/test_oht_visualization.py tests/test_oht_source_config.py
~~~

覆盖 12 个合成 episode → 60 条 transitions、五相机、视频 PTS、米制 depth、划分检查、教师/预测缓存、无 GT 推理隔离、真实 Agent 梯度累积与零碰撞损失、HTTP 协议、闭环失败计数和续训采样。backbone/render 使用 CPU 小替身，未验证完整 PaliGemma/point-renderer GPU 前向。另运行现有角色预测、跨尺度继承、辅助损失、前向与优化器回归测试。

2026-10-09：174 项 OHT 测试分批通过（NumPy 1.26.4 / PyArrow 19.0.1）。覆盖已启用经验 ROI 的文档坐标示例保留、手工 XYZ 保留/排除框、边界与空场景、训练/在线/预览一致性及 RGB/GT 不变、gripper-only 关键帧、无人工终帧/尾段、无事件跳过/空缓存拒绝、BridgeVLA 事件/原版同等停稳信号对照、实际 dt 与纯旋转、旧几何模式、严格未来目标、metadata K/逐相机量化/端点与 hash、EE 顺序、实测夹爪因果处理、raw action 七维扰动不影响 replay、v1 拒绝，以及原生 gray12 视频→replay→teacher 预览、诊断字段不进入 batch。先前合跑出现视频库内存分配失败，本轮限制数值库线程并分批验证。

另已直接抽取提供转换器的纯数值函数，对照 linear/log 各 4096 个深度码值；最大差约 0.504 mm（参考 PNG 毫米舍入及浮点差异），相机变换/K 缩放和原始夹爪端点归一化通过对照。该对照不表示参考链路二次归一化后的夹爪标签或完整点云流程完全一致。

均为本地合成数据/纯函数验证，未读取服务器 v423 全量数据，也未验证完整 CUDA/VLM/真实闭环。缺少公共反投影模块，ray/Z 尚需核对；audit 的服务器实际提速尚未测量。
