# OHT v423 实施与运行

本目录提供 OHT 数据审计、观测/动作缓存、baseline、内部 T/R role queries、可选外部 predicted objects、训练、推理服务和评估入口。命令均从 **BridgeVLA 仓库根目录** 执行。设计背景见 [迁移方案](../../docs/design/oht-data-migration.md)。

完整重建 replay 和对象预览见[完整重新生成 replay](#完整重新生成-replay)。原始数据未变时可复用已有 audit，直接从构建步骤开始。

已用合成 OHT 数据完成 CPU 集成验证，包括真实 Parquet、MP4 解码、深度点云、真实 Agent 更新及 HTTP 请求。真实 v423 数据、完整 CUDA 模型和 IsaacLab 闭环尚未验证。

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

- 内参优先来自各 `<task>/lerobot_dataset/meta/camera_intrinsics.json`；该文件不存在时读取 `meta/info.json.camera_intrinsics`（用户本次提供的格式）。支持大小写相机名和 `cameras.<name>.intrinsic`；缺失/非法时报错，不猜焦距，也不静默绕过非法 sidecar。使用 YAML 内参时设 `intrinsics_source: config` 并填完整 K。
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

原始相机外参默认为 camera→world、USD/OpenGL 轴，按显式顺序转换到内部 xyzw。`camera_pose_matrix()` 与参考转换器一致：先解析 `camera_quaternion_order`，若 `camera_extrinsic_direction: world_to_camera` 则对完整 4×4 求逆，再右乘 `optical_to_sensor`。默认方向 `camera_to_world`，五个相机的光学变换均为 `diag(1,-1,-1,1)`：`T_world_optical = T_world_usd @ optical_to_sensor`。不要将翻轴矩阵左乘到世界坐标，也不要只转置 R 而忘记平移。若来源已经是 optical camera→world，显式使用 identity，禁止再翻轴。缓存外参始终是 4×4 optical→world；投影使用 `(P_world - t) @ R_world_optical`，不再次转置或翻轴。camera 与 EE 的四元数配置独立，不能靠改动作旋转修复点云。

深度为无损 HEVC `gray12le`；**像素格式不能决定数值编码**。当前按用户指定的参考代码读取：原始像素是量化码，先 linear/log 反量化，再按参考 TFDS 转换器舍入为 uint16 毫米，最后转回米；使用 `kind: z` 针孔反投影。`0` 为无效码，`qmax=4095` 是有效 far plane 码，默认解为 10 m，再由参考点云的 `0 < Z <= 3 m` 范围过滤。不能直接把原始码乘 0.001，也不能将 4095 当作无效原始码。

**以参考链路为准**：`metadata: reference` 从每个相机的 `features.observation.depth.<camera>.info` 读取量化参数，`video.*` key 优先于普通 key；缺失字段采用 YAML 中与参考 `_depth_video_spec()` 一致的默认值：`depth_min=0.01`、`depth_max=10.0`、`shift=3.5`、`use_log=true`、`qmax=4095`、`pixel_format=gray12le`。RGB-only info 或没有 info 文件时也沿用这些默认值，不从 RGB codec 推断深度。默认 raw=897 解为 **1.215 m**，raw=1500 解为 **2.249 m**。此前 `raw × 0.001` 的默认配置与此链路不一致；即使外参正确且表面高度一致，也可能造成 XY 融合分离。

`encoding: quantized` 时设 `n=q/qmax`，log 逆变换为 `exp(n*(log(max+shift)-log(min+shift))+log(min+shift))-shift`，linear 为 `n*(max-min)+min`。默认 `round_to_mm: true` 匹配参考 float32 运算及 PNG 毫米舍入；设为 false 可保留亚毫米浮点深度。`metadata: true` 仍为严格模式，要求每相机完整的 `depth_min/depth_max/shift/use_log/qmax/pix_fmt`，缺失即报错；`metadata: false` 只用显式 YAML 参数。已确认的原始毫米导出仍可单独配置 `encoding: scaled_integer`、`metadata: false`、`scale: 0.001`，但不再作为本次参考链路的默认值。每任务实际解析出的参数及源文件 hash 写入 contract，不能仅改 contract 来重解释旧 XYZ。

`depth.path_pattern: null` 通过 Parquet 引用读取原生灰度视频，不经过 RGB；按下文 `video_alignment` 配对帧。补充的 `pointcloud_transforms.py` 明确使用 `Z=depth_m`、`X=(u-cx)*Z/fx`、`Y=(v-cy)*Z/fy`，因此默认 `depth.kind: z`，不归一化射线。将 Z-depth 按此前的 `ray` 处理会随像素离主点的距离拉近表面，造成跨相机融合变形。只有独立确认来源为相机到点的欧氏距离时才显式使用 `ray`。

从旧 ray 或直接毫米解码配置迁移时，替换实际 `--config` 文件中的整个 depth 块，同时检查 `cameras.<name>.depth` 的旧覆盖值。旧 replay/checkpoint 中记录的编码、kind 和已保存 XYZ 不随仓库默认值改变；在新目录重建 replay，再重建依赖的 teacher/预测缓存并训练匹配的 checkpoint。不要手改旧 contract 或把旧米制 depth/XYZ 仅重画成预览就当作修复。

仓库配置更新不会修改此前复制的 `dataset-calibrated.yaml`。构建始终读取 `--config` 指定的文件；请同步所需字段，保留本地已有的正确值。

当前 schema 为 `oht_bridgevla_v2`；下文的 `replay-source-v3` 只是新的输出目录名，不代表 schema 升级。旧缓存的几何或标签修复后，需要重建依赖的 teacher/预测缓存并重新训练，不能沿用旧 optimizer resume。

数据方提供的 frame 965 腕部投影参考值为原图 `(313.0, 378.6)`，在默认 4 倍步长缓存中应约为 `(78.25, 94.65)`。本地没有该原始样本，仍需在服务器复核多帧、多相机，不以“落在图内”代替几何对齐验证。

默认 image_size=[120,160] 对原始 480×640 做严格 4 倍步长采样，K 同步缩放。这是输入 RGB-D 尺寸，模型的虚拟渲染图像仍用现有 MVT 配置。

当前默认深度配置：

~~~yaml
depth:
  encoding: quantized
  pixel_format: gray12le
  metadata: reference
  depth_min: 0.01
  depth_max: 10.0
  shift: 3.5
  use_log: true
  qmax: 4095
  round_to_mm: true
  kind: z
  invalid_values: [0]
  path_pattern: null
  limits: [0.001, 3.0]
~~~

读取器检查实际像素格式，保留 12-bit 数值及视频 PTS；旧 PyAV 的 gray12le 数组兼容路径直接读取含行 padding 的 uint16 平面。仍兼容浮点 NPY（metric）、整数 PNG/TIFF（scaled_integer）和显式指定通道/scale/offset 的旧 linear_channel 编码。普通可视化视频不能据此当作米制深度。

### 完整重新生成 replay

使用当前仓库配置，从原始 RGB-D 重新生成 replay、近似对象点云和预览。已有有效的 `/data/oht/audit-seed0.json` 可直接复用；输出目录 `/data/oht/replay-source-v3` 必须尚不存在。再次重建时更换新目录名，保留旧结果用于对照。

~~~bash
python tools/build_oht_replay.py \
  --root /common-data-32t/data/robot_data/oht_curobo_pd_v423 \
  --manifest /data/oht/audit-seed0.json \
  --config finetune/OHT/configs/dataset.yaml \
  --output /data/oht/replay-source-v3 \
  --sample-stride 10 \
  --object-cloud-config finetune/OHT/configs/object_clouds.yaml \
  --visualize-every 1

python tools/validate_oht_replay.py --replay /data/oht/replay-source-v3
~~~

构建结束并通过校验后，打开 **`/data/oht/replay-source-v3/preview/index.html`**。PNG 保留原有场景面板，底部增加各对象点云区域；HTML 上方显示 PNG，下方提供点云交互、视角和帧选择。

- `--sample-stride 10` 控制哪些原始帧进入 replay；完整重建仍沿用该采样策略，如需更密的 replay 可减小步长。
- `--visualize-every 1` 为每个生成的 replay 样本输出 PNG/HTML；改为 `100` 可减少预览数量和开销，对象 PLY/NPZ 仍逐样本生成。
- `--object-cloud-config` 启用对象截取和对象预览。只传 `--visualize-every` 会生成基础场景 PNG，没有对象区域和 HTML 统一入口。
- 示例省略 `--visualize-output-dir`，所有诊断文件统一放在 replay 下的 `preview`；显式指定该参数会将 PNG/HTML/入口输出到指定的新目录，对象数据仍保存在 replay 下。

旧 PNG/HTML 不会随代码更新自动变化。请打开本次新目录；复制到本机时保留整个 `preview` 目录及相对路径。

若旧版本在首帧报 `FileExistsError: Use a new object cloud directory: .../preview/...`，是 PNG 先建目录与对象导出冲突。更新到包含写入顺序修复的代码后，用新的 replay 输出目录重新构建；失败目录没有 `complete.json`，不能用于训练。无需重新审计原始数据。

实现行为：

- 仓库默认 `video_alignment: frame_index`：与参考 `_read_video_selected_rgb()` / `_read_depth_video_selected()` 一样，原始 Parquet 第 i 行严格配对 RGB 第 i 个解码帧、depth 第 i 个解码帧和第 i 行相机外参；不是抽样后的第 i 帧，也不从 FPS 推算。逐流核对解码总数等于 Parquet 行数（含未抽样尾段），缺帧/多帧即失败、不写 complete。可避免时间戳引用偏移时旋转相机的视频帧与姿态错配，但不能修复导出端已经错配的 pose。
- 独立时间戳流可显式设 `video_alignment: timestamp`，按引用选择最近 PTS，超出 `video_timestamp_tolerance` 报错；旧配置缺少 alignment 字段仍保留此行为。切换对齐模式需新建 replay/teacher，修改 YAML 不会修复旧 XYZ；原始数据/audit 可复用。
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

`--visualize-every N` 按每个 episode **生成的样本数**保存 PNG，包含第一个样本，随后每隔 N 个；不是按原始视频帧计数。`1` 显示每个生成样本，默认 `0` 关闭。可省略 `--visualize-output-dir`，此时保存到 `<output>/preview/<task>/<episode六位>/<frame六位>/preview.png`。预览文件存在时拒绝覆盖。

每张图包含缓存分辨率的各相机 RGB、米制 depth，以及两排 XY/XZ/YZ 彩色点云：青色为当前 TCP，品红色为下一 GT 关键点。depth 蓝色近、红色远，标注当前有效范围（米），黑色表示无效；不同图的深度颜色范围可能不同。

每个 RGB 面板下方显示 TCP/goal 的 `in view`、`outside image`、`behind camera` 或 `unavailable`，以及缓存像素坐标/光学 Z。画外点不强制移到边缘；`in view` 仅代表在视锥内，不代表没有被遮挡。

- 全局三视图使用 `scene_bounds`，最多显示 200,000 个点（默认五相机 120×160 的全部有效点都在预算内）。全局/局部点云均以单像素绘制，不扩张相邻像素，重叠点按深度处理；TCP/goal 标记大小不变，品红色方框标出局部立方体的投影范围。
- 局部三视图以 GT keypoint 为中心，各轴 ±0.20 m，显示米制坐标范围。先从完整有效点云选择局部点，再独立限制显示点数，避免全局抽样漏掉小物体；没有观测点时明确提示，不补造几何。

局部图标注 **GT-centered refine diagnostic (NOT model stage2)**：它不是模型 coarse 预测或带噪训练中心产生的二阶段视图。模型真实 coarse/refine renderer 图应在训练/推理前向中另行导出。两排都显示抽样前后点数；缓存的 4 倍步长采样不因预览变密而恢复到原始分辨率。

PNG 单独输出，不新增 observation/label 字段，也不改变 buffer contract 或训练样本。无需模型、CUDA 或图形桌面；仅使用数据环境已有的 NumPy/Pillow。基础 buffer 没有 T/R 标注，角色预览在下一节的 teacher 构建时生成。

#### 生成 replay 时同步提取并预览对象点云

上面的完整重建命令已启用 `--object-cloud-config`，对**每个生成的 replay 样本**按当前帧对象中心同步导出近似点云，不按未来 goal 帧截取。

对象配置与相机/训练配置分开，示例见 [`configs/object_clouds.yaml`](configs/object_clouds.yaml)。`objects: null` 表示全部记录对象；初始 `default_size: [0.20, 0.20, 0.20]` 为 world XYZ 全长（米）的可调整猜测，不是物体尺寸 GT。需要不同尺寸或只提取部分对象时，复制该 YAML 并改为例如：

~~~yaml
source: depth
default_size: [0.20, 0.20, 0.20]
objects:
  - index: 0
    name: object_0
    size: [0.30, 0.20, 0.20]
  - index: 1
    size: [0.15, 0.15, 0.25]
~~~

配置按对象索引应用到每个样本，名称语义需先核实；指定索引在当前帧不存在时构建失败，不猜测替代对象。`source: depth` 用已缓存米制 depth/K/外参重建 ROI 前点云，`xyz` 则只截取缓存中仍有效的 XYZ。框随当前中心平移，保持 world 轴对齐，不随四元数旋转；没有实例 mask，框内仍可能混入背景或邻近物体。

每个样本保存至 `<replay>/preview/<task>/<episode>/<frame>/`：逐对象 `object_<index>.ply`、`object_<index>.npz` 和 `object-clouds.json`；索引行的 `object_clouds` 字段指向报告。空框保留零点文件及 `valid: false`；当前帧没有任何对象且选择全部时，保存空对象列表。

默认目录结构如下（预览采样帧才含 PNG/HTML；对象文件覆盖每个生成样本）：

~~~text
preview/
  index.html
  <task>/<episode>/<frame>/
    preview.png
    point-cloud-roi.html
    object-clouds.json
    object_0.ply
    object_0.npz
~~~

`--visualize-every N` 同时控制 PNG 和交互 HTML 的频率，按每个 episode 的生成样本数计数、包含首个样本。`1` 为每个样本生成预览，`0` 只保存对象点云、不生成预览。默认 PNG、`point-cloud-roi.html` 与对象文件同目录，统一入口为 `<replay>/preview/index.html`。HTML 上方显示当前帧完整 PNG，下方是可旋转的对象点云交互区，自动勾选配置对象并恢复名称/尺寸；可继续勾选其他对象和调整尺寸。页面优先将框内点纳入显示抽样，报告/PLY/NPZ 保留缓存分辨率下的全部有效框内点。显式 `--visualize-output-dir` 会同时改变 PNG、HTML 和入口页的位置；PLY/NPZ/报告仍在 replay 的 `preview` 中，保证索引可携带校验。

**对象点云直接放在同一张 PNG 的底部**，无需另开 HTML 才能看到。上方原有相机 RGB/depth、全局场景与 GT 局部 XY/XZ/YZ 面板保持原布局；下方为每个选中对象增加一排 XY/XZ/YZ 点云，按该对象完整 world 轴对齐框适配范围，并显示中心、名称和框。按对象索引着色，重叠框成员为白色；颜色表示近似框成员关系，不是实例或 T/R 标签。每排记录完整 `source` 点数、工作区内 `world` 点数和 GT 局部框内 `local` 点数，空框明确标注并保留空三视图。对象点从缓存完整源单独选择/限制显示预算，不受场景抽样影响；`source: depth` 可显示旧 ROI 之外的有效深度点，即使背景 XYZ 已被过滤。PNG 高度随选中对象数增加，保存的观测和动作标签保持原样。

对象文件在构建过程中逐帧写出，可复制已写完的样本预览目录（包含 PNG 和 HTML）在本机查看；整批结束后报告/HTML 绑定最终 contract 哈希，生成统一入口 `<replay>/preview/index.html`，再写 `complete.json`。若在构建途中打开过页面，结束后重新复制/打开最终 HTML 再下载绑定配置。构建失败不生成完成标记。

启用后会增加逐帧 PLY/NPZ 存储，`--visualize-every` 仅减少预览数量，不减少对象点云导出。配置和对象报告路径纳入 contract/索引校验，启用前后的 contract 哈希会不同；原 observation NPZ、动作标签和策略输入保持一致。近似对象点云只作诊断，不自动进入教师或训练前向。

#### HTML 统一查看、角度与帧选择

构建时启用 `--object-cloud-config` 和非零 `--visualize-every` 后，打开 `<replay>/preview/index.html`，即可在同一页面选择 episode、实际帧号，或用上一帧/下一帧、滑条切换。列表只包含已生成 HTML 的样本；`--visualize-every 1` 覆盖每个 replay 样本，不能显示未采样进 replay 的原视频帧。复制到本机时保留整个 `preview` 目录及其相对路径，可离线双击入口，无需服务。入口页先显示当前帧 PNG（底部含对象区域），再显示下方交互点云；单独打开一帧 HTML 也保持这个顺序。

顶部提供 3D/XY/XZ/YZ 和适配当前帧按钮；点云支持左键旋转、右键平移、滚轮缩放，并显示水平/俯仰角度。默认切帧保留旋转、缩放、平移和相机/显示开关；取消“切帧保留视角”后使用新帧默认视图。同一 episode 内保留对象勾选、名称、尺寸框以及 ROI 编辑，跨 episode 独立保存对象/ROI 设置。对象中心、姿态、TCP、GT 和下载配置的帧/校验值始终取当前帧，切帧不会套用旧帧位姿。设置只保留在当前打开页面中，刷新重置；要留存尺寸/ROI，可下载对应 JSON/YAML。

HTML 中对象与场景共用一张点云画布，“对象框成员”颜色按索引区分，重叠框为白色；可切回 RGB/相机颜色、调整背景透明度或只显示对象框内点。对象列表可勾选全部记录对象的中心、名称和姿态轴；轴需选择正确四元数顺序。

上方 PNG 是生成时配置对应的静态预览；下方勾选对象和调整尺寸仅更新交互点云，不重写 PNG。顶部“查看下方交互点云”可直接跳到交互区。

如果修改代码后看起来没变化，先检查生成命令是否带 `--object-cloud-config`，以及打开的是否为本次新输出。旧 PNG/HTML 不会随代码更新自动改变；旧 replay 未记录对象配置时，下面的独立导出命令需显式增加对象 YAML。

仅更新可视化、无需重算 replay 时，可为已有**完整 replay** 导出新的 PNG/HTML。已执行上面的完整重建则不需要再运行此命令：

~~~bash
python tools/export_oht_replay_viewer.py \
  --replay /data/oht/replay-source-v3 \
  --output /data/oht/replay-source-v3/preview/refresh \
  --object-cloud-config finetune/OHT/configs/object_clouds.yaml \
  --every 1 --max-points 40000
~~~

打开 `/data/oht/replay-source-v3/preview/refresh/index.html`。输出目录必须尚不存在，每帧 PNG/HTML 保存到 `<output>/<task>/<episode>/<frame>/`。`--every N` 按各 episode 的 replay 样本数选帧，包含首个；浏览器每次仅加载一帧。默认源为缓存米制 depth/K/外参，可用 `--source xyz` 改为缓存 XYZ。导出校验已完成 replay 和观测 NPZ，不改写缓存，也不额外导出 PLY/NPZ。

#### 单帧几何排错（可选）

需要检查多相机融合参数时，使用 `samples.jsonl` 中真实存在的 sample ID。以下命令同时输出几何诊断和离线点云 HTML，目录必须尚不存在：

~~~bash
python tools/diagnose_oht_geometry.py \
  --replay /data/oht/replay-source-v3 \
  --sample-id assemble_left/000000/000000 \
  --output /data/oht/replay-source-v3/preview/geometry-frame0 \
  --html --html-source depth --max-points 40000
~~~

工具不修改缓存，输出各物理相机三视图、`fused_rgb.png`、按来源相机着色的 `fused_camera_colors.png`、`geometry.json`（实际缓存 K、optical→world 外参、有效点数、契约配置）和 `point-cloud-roi.html`。先检查静止桌面/夹具是否重合，再看运动部分；偏移时核对深度编码、`kind: z`、相机参数和同步，不用 ICP 掩盖参数错误。该单帧工具不生成统一入口，也不替代上面的对象预览构建命令。

默认要求完整 buffer。**仅诊断时**可加 `--allow-incomplete`，不读取/要求 `complete.json`；仍检查 contract 自身哈希、配置、已绑定的索引哈希，以及所选样本的 NPZ 哈希。训练、教师生成和正式评估不支持此开关，完整性要求不变。

#### 交互确定点云范围

构建生成的 HTML 或上述单帧 `point-cloud-roi.html` 均可用于选择范围。左键旋转、右键/Shift 拖动平移、滚轮缩放；支持 XY/XZ/YZ、相机开关、RGB/相机着色、点大小及点选世界坐标，离线使用无需启动服务。

选择并修改 `scene_bounds`、`keep_bounds` 或排除框的六个世界坐标，实时查看保留点及框；可添加/删除排除框，复制或下载 YAML 片段，再人工合并进实际 `--config` 文件。上下界与 `point_cloud_filter` 一致：下界包含、上界不包含；非法框不更新且禁止导出。`scene_bounds` 是策略工作区，不只是显示裁剪；它影响动作归一化及数据/checkpoint 契约，不能随意覆盖已有模型的配置。相机开关、点大小和显示抽样不写入训练配置。

`--html-source depth` 推荐用于确定范围：直接将缓存的**米制 depth + K + optical→world 外参**重新反投影，仅用于显示，不再解码深度、不套旧 ROI，也不写回缓存。它能查看被旧 ROI 删除但深度仍有效的点；PNG/geometry.json 仍描述原缓存 XYZ。默认 `--html-source xyz` 则只显示缓存剩余有限 XYZ，不额外裁剪，也无法恢复其 NaN 点。页面明确标出来源。

**限制**：两种来源都受缓存分辨率、有效深度范围和原始内外参/同步正确性限制，不能补造未观测表面，不能修复历史错误深度编码。扩大范围不必为了预览先重建 buffer；确定新过滤/工作区配置后，再重建受影响的训练缓存，不能只改旧 contract。默认最多显示 40,000 点，可调至 60,000；保留计数仅对应显示抽样，不是全量覆盖证明。工具不配准点云、不修改 buffer；TCP/未来 GT goal 仅作诊断，并提示超出当前框。至少检查四任务、多时刻、最高抬升与操作夹具，再确定统一范围。

#### 显示任务相关对象

新构建的 replay 在 `samples.jsonl` 的 `diagnostic_objects` 中保存**当前观测帧**记录的全部对象中心与原始四元数及源 Parquet 路径/哈希，不增加 observation NPZ 或训练 batch 字段。对象数量按 `objects_pos` 的每 3 个位置值与 `objects_quat` 的每 4 个四元数值匹配，不固定为四个；两者数量必须一致，同一 episode 内保持一致。HTML 默认勾选全部对象，按索引列出 `object_0…object_{N-1}`，提供“全选对象”“清空对象选择”和每项复选框；仅显示勾选对象的中心、名称及已启用的姿态轴。还可修改显示名称、缩放到所选对象、调淡背景点，并点击中心查看 world XYZ 和原始四元数。中心在当前框外仍显示并提示。清单只包含数据中记录了位姿的对象，不自动发现未记录的场景物体。对象选择、名称和背景透明度只作用于页面，不改变点云保留计数、ROI YAML 或缓存；改名仅在当前页面有效。

参考代码没有确认对象索引与语义名称的对应关系，也没有确认 `objects_quat` 的顺序，因此不自动猜测车轮/托盘或 Target/Reference。姿态轴默认隐藏；核实导出端约定后，在页面的“对象四元数顺序”中独立选择 `wxyz` 或 `xyzw`（不继承 camera/EE 设置）。轴为对象局部 x 红/y 绿/z 蓝，长度可调，坐标直接使用 world 位姿，不套相机 optical 翻轴。对象中心不是表面点或轴口 site；实例 mask 和角色点仍使用下文教师接口。

仅旧 replay 缺少对象元数据时，导出工具可同时增加 `--object-root /common-data-32t/data/robot_data/oht_curobo_pd_v423` 和 `--object-manifest /data/oht/audit-seed0.json`，清单必须是构建该 replay 时使用的 audit。单帧诊断还需启用 `--html`。读取器核对 audit、Parquet 哈希及当前帧，绑定不符则拒绝输出；不改写缓存，也不猜测缺失位姿。新重建流程无需这些补充参数。

#### 没有实例 mask 时近似提取 object 点云

HTML 支持以当前记录的对象中心为中心，设置 **world X/Y/Z 轴对齐框的全长（米）**。每轴初始 0.20 m 只是可调整的预览值，不代表实物尺寸；不依赖对象四元数，不随姿态轴顺序旋转。若记录原点不在几何中心，需要调整框尺寸以覆盖物体。此方法截取的是框内观测点，可能包含桌面/邻近物体，不能作为实例分割 GT；重叠框允许共享点，不自动将点分配给最近对象。

1. 打开当前帧 HTML，勾选所需对象，调整每项的 X/Y/Z 全长。
2. 开启“显示近似截取框”和“仅显示勾选对象框内点云”，检查物体是否被完整覆盖；点击“缩放到所选对象”可将整个截取框纳入视图。框边界为下界包含、上界不包含；点云仍受当前相机开关、工作区显示和背景透明度控制。
3. 若要按页面调整后的尺寸重新导出该帧对象文件，点击“下载对象截取配置 JSON”，将 `oht-object-boxes.json` 传回服务器，再运行下列命令。构建时已生成默认对象文件，无需重复提取；JSON 绑定当前 sample/frame、点源和缓存校验值，不修改 replay 或 ROI YAML。

~~~bash
python tools/extract_oht_object_clouds.py \
  --replay /data/oht/replay-source-v3 \
  --sample-id assemble_left/000000/000000 \
  --boxes /data/oht/oht-object-boxes.json \
  --output /data/oht/replay-source-v3/preview/object-clouds-frame0
~~~

旧 replay 缺少对象位姿时，在提取命令中同样增加 `--object-root` 和 `--object-manifest`，无需重建点云。仅诊断未完成 replay 可显式加 `--allow-incomplete`；训练完整性要求不变。对象配置绑定当前帧，不能直接用于另一帧或另一份缓存；重新预览相应样本后再导出。

输出为每个选中对象的 `object_<index>.ply`（world XYZ + RGB）、`object_<index>.npz`（`xyz[N,3]`、`rgb[N,3]`、`camera_ids[N]`），以及 `object-clouds.json`（框范围、每相机点数、校验值和 `approximate: true`）。没有框内点时输出空点云并记录 `valid: false`，不补造物体。输出目录必须是新目录。

**HTML 是显示抽样，导出命令读取缓存分辨率下的全部有效点**，不受 HTML 点数预算、相机开关或页面工作区显示过滤影响，也不做额外去重/降采样。JSON 的 `source: xyz` 使用已缓存点，无法恢复旧 ROI 的 NaN；`source: depth` 使用缓存米制深度/K/外参重建，可恢复旧 ROI 删掉但深度仍有效的点。全量导出不等于恢复原始分辨率或完整物体表面，不能修复深度/标定/同步错误。提取结果仅作近似对象诊断，不自动进入角色教师或策略输入。

#### 构建未完成时

`samples.jsonl` 在 buffer 完成时才写入。若尚无索引，直接查看已经写完的 NPZ：

~~~bash
python tools/diagnose_oht_geometry.py \
  --replay /data/oht/replay-source-v3 \
  --observation observations/assemble_left/000000/000000.npz \
  --data-profile assemble_left/lerobot_dataset \
  --allow-incomplete --html --html-source depth \
  --output /data/oht/replay-source-v3/preview/geometry-building-frame
~~~

NPZ 路径相对于 replay，不能越出该目录；`--data-profile` 必须替换成 `contract.json.source_data_configs` 中对应数据源的键，多源时必须显式提供，单源可省略。`--observation` 与 `--sample-id` 二选一；它只依赖已有 contract 和 NPZ，不依赖索引或完成标记，也不读取原始视频。

直接 NPZ 模式不猜测帧时序、TCP 或 goal：HTML 标注未知，PNG 仅保留全局三视图，不生成 GT-centered 局部图。页面和 `geometry.json.validation` 明确记录仅诊断、未验证整批完成；NPZ 的实际哈希仅作本次快照记录，不冒充索引校验。文件仍在写入、缺少相机字段或标定不合法时拒绝输出，稍后换新输出目录重试。不会补写 `complete.json`/索引或使未完成 buffer 可用于训练。

## 4. Baseline 训练

先做 CPU 数据预检：

~~~bash
python -m finetune.OHT.train \
  --config finetune/OHT/configs/baseline.yaml \
  --replay /data/oht/replay-source-v3 --output /data/oht/check-only \
  --validate-only
~~~

然后在 BridgeVLA GPU 环境做两步 smoke，再正式训练：

~~~bash
python -m finetune.OHT.train \
  --config finetune/OHT/configs/smoke.yaml \
  --replay /data/oht/replay-source-v3 --output /data/oht/runs/smoke \
  --pretrain-path /path/to/bridgevla-heatmap-pretrain

python -m finetune.OHT.train \
  --config finetune/OHT/configs/baseline.yaml \
  --replay /data/oht/replay-source-v3 --output /data/oht/runs/baseline-s0 \
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
  --replay /data/oht/replay-source-v3 --annotations /data/oht/annotations.jsonl \
  --output /data/oht/teachers-v1 --point-count 512 \
  --visualize-every 100

python tools/validate_oht_replay.py \
  --replay /data/oht/replay-source-v3 --mode role_queries \
  --role-cache /data/oht/teachers-v1

python -m finetune.OHT.train \
  --config finetune/OHT/configs/role_queries.yaml \
  --replay /data/oht/replay-source-v3 --role-cache /data/oht/teachers-v1 \
  --output /data/oht/runs/role-queries-s0 \
  --init-checkpoint /data/oht/runs/baseline-s0/model_last.pth
~~~

Teacher 使用相同的可视化参数，默认输出到 `<teacher-output>/preview/`。Target 为绿色、Reference 为蓝色、重叠为黄色；标注 mask 区域显示 `30% 原始 RGB + 70% 角色颜色`，背景保留 RGB。`site_region` 只显示区域点轮廓。图中列出角色 source/present/known/几何有效性，NULL/unknown 不伪造点。v2 在样本索引中保存当前 TCP，仅供诊断；teacher 预览可同时显示当前 TCP 和未来 goal，该字段不进入训练 batch。

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
  --replay /data/oht/replay-source-v3 --predictor my_oht_predictor:create \
  --provenance /data/oht/predictor-provenance.yaml \
  --output /data/oht/predictions-v1

python -m finetune.OHT.train \
  --config finetune/OHT/configs/predicted_external.yaml \
  --replay /data/oht/replay-source-v3 --role-cache /data/oht/predictions-v1 \
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
  --replay /data/oht/replay-source-v3 --split val \
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
  --server http://127.0.0.1:8010 --replay /data/oht/replay-source-v3 \
  --environment my_isaac_oht:create --cases /data/oht/eval-cases.jsonl \
  --executor-id verified-eef-executor-v1 --max-steps 500 \
  --output /data/oht/eval/baseline-eef-s0.json
~~~

case JSONL 每行含唯一 id、task、seed；id 每次运行需使用新 episode 标识或重启 server。所有方法用同一份 cases、执行器版本和时间预算，分别报告 B-E/P-E，必要时 B-H/P-H。异常、超时和未完成 episode 均保留在成功率分母。报告记录 cases hash、契约和执行器 ID，便于检查对照条件。

用户提供的 H-VLA 19/20 与 14/20 是历史参考；本实现尚无真实 OHT baseline 或 assistance 成绩。

## 9. 本地验证

数据迁移、构建时对象导出与 HTML 帧导航的快速回归：

~~~bash
python -m pytest -q \
  tests/test_oht_migration.py \
  tests/test_oht_object_cloud_build.py \
  tests/test_oht_replay_viewer.py
~~~

HTML/诊断测试覆盖抽样/RGB/相机对齐、旧 ROI 点的显示重建、无缓存改写、离线导出、非法范围，以及未完成 buffer 的显式隔离、直接 NPZ、缺失标签、多源选择和损坏文件拒绝。可选浏览器 smoke test 需安装 Playwright，并设置 `OHT_BROWSER_EXECUTABLE` 为本机 Chromium/Chrome/Edge 可执行文件；使用独立测试 profile，不复用用户浏览器数据。未设置时仅跳过这一项浏览器测试，不影响 HTML 导出使用。

覆盖 12 个合成 episode → 60 条 transitions、五相机、视频 PTS、米制 depth、划分检查、教师/预测缓存、无 GT 推理隔离、真实 Agent 梯度累积与零碰撞损失、HTTP 协议、闭环失败计数和续训采样。backbone/render 使用 CPU 小替身，未验证完整 PaliGemma/point-renderer GPU 前向。另运行现有角色预测、跨尺度继承、辅助损失、前向与优化器回归测试。

其余 `tests/test_oht_*.py` 按修改范围运行，覆盖相机约定、参考 float32 反量化/毫米舍入、默认 Z-depth、跨相机表面偏移、ROI、标签、训练和诊断隔离等。上述验证基于本地合成数据与纯函数，真实跨相机对齐仍需用服务器重建后的缓存复核。
