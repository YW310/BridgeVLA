# OHT v423 实施与运行

动作模式可通过训练配置中的 action_mode 选择 eef / relative_eef / joint / relative_joint，并与三种物体辅助模式组合；标签、模型头、协议和运行步骤见 [四种动作模式配置](../../docs/guides/oht-action-modes.md)。省略该配置保持 EEF 兼容行为。

本目录提供 OHT 数据审计、观测/动作缓存、baseline、内部 T/R role queries、可选外部 predicted objects、训练、推理服务和评估入口。命令均从 **BridgeVLA 仓库根目录** 执行。设计背景见 [迁移方案](../../docs/design/oht-data-migration.md)。

下一版 [关键目标 + 连续 action chunk baseline](../../docs/design/key-goal-action-chunk-baseline.md) 已接通 V4 数据、联合/冻结关键目标两种训练、离线评估、HTTP 与客户端执行，见 [V4 训练与评估](#v4关键目标--连续-chunk-训练与评估)。默认沿用 full 的 Gemma 前缀层冻结，联合训练其后续层、关键目标头和 decoder。单关键点配置仍使用原路径；旧 checkpoint 用于初始化关键目标模型，不能直接 resume 成 chunk 策略。真实 CUDA/IsaacLab 验收待运行。

新连续数据使用 [V4 完整重建](#v4完整重建关键目标与连续-chunk)；现有单关键点 replay 和对象预览见[完整重新生成 replay](#完整重新生成-replay)。原始数据未变时可复用已有 audit，直接从构建步骤开始。

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
  --output /common-data-32t/data/robot_data/oht_curobo_pd_v423/audit-seed0.json --seed 0
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

现有单关键点缓存 schema 为 `oht_bridgevla_v2`，目录名 `replay-source-v3` 不代表 schema 升级。新连续 chunk 缓存使用独立 schema `oht_bridgevla_v4`，建议目录名 `replay-source-v4`。几何、标签或 schema 改变后不能沿用旧 optimizer resume；旧 teacher/预测缓存也不能仅改 hash 后绑定新缓存。

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

### V4：完整重建关键目标与连续 chunk

V4 从原始 Parquet 和 RGB-D 视频重新生成全部所选输入帧，不复制或复用 V3 的观测 NPZ。原始数据不需重新采集；已有有效 audit 可复用，以保持相同 episode/场景划分。输出目录必须尚不存在。

~~~bash
python tools/build_oht_replay.py \
  --root /common-data-32t/data/robot_data/oht_curobo_pd_v423 \
  --manifest /common-data-32t/data/robot_data/oht_curobo_pd_v423/audit-seed0.json \
  --config finetune/OHT/configs/dataset.yaml \
  --output /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v4 \
  --replay-version 4 \
  --chunk-horizon 16 --action-frame-stride 1 \
  --sample-stride 10 --visualize-every 100

python tools/validate_oht_replay.py \
  --replay /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v4 --workers 4
~~~

`--root` 指原始四任务数据的父目录，不是 `replay-source-v3`。若之前使用本地标定 YAML，继续用已核查的文件替换 `--config`，不要丢掉本地标定。

| 参数 / 字段 | V4 含义 |
| --- | --- |
| `--chunk-horizon 16` | 每个输入最多监督 16 个未来实测目标 |
| `--action-frame-stride 1` | 每个动作前进 1 个原始帧，与输入采样步长分开 |
| `--action-dt` | 可选，显式期望动作时间间隔；省略时从原始时间戳推导并固定整个缓存的间隔 |
| `--sample-stride 10` | 输入观测每隔 10 帧抽样，另保留非终帧关键事件；不是动作的 dt |
| `--visualize-every 100` | 每个 episode 的首个及每隔 100 个样本保存 PNG/HTML，将有效 GT chunk 叠加在 sensor RGB 和点云投影图上；不改变数据标签 |
| `--object-cloud-config ...` | 可选，增加近似对象点云导出；V4 基础 HTML 不依赖此配置，不是训练 role teacher |

V4 同时保存下一关键目标和未来连续轨迹，动作取实测 TCP/关节/夹爪，不使用原始动作命令或关键点间直线插值。目标时间为 `t+dt ... t+K*dt`；实际 `dt`、stride、K、标定、audit 和索引摘要写入 contract。不同 episode 的采样率不匹配时构建失败，时间缺口或粗采样丢失完整夹爪切换的窗口会跳过并记录原因。`--action-timestamp-atol/rtol` 可设时间检查容差；不能用放宽容差掩盖不同采样率。

最后关键点之后的尾段、无夹爪切换的 episode 也生成有效连续样本。没有下一关键目标时 `key_valid=false`；episode 尾部不足 K 项使用 `action_valid` 屏蔽，不复制最后帧充当真实停留。真实静止、旋转和夹爪动作保持原数据时序。

~~~text
replay-source-v4/
  contract.json
  samples.jsonl
  complete.json
  observations/<task>/<episode>/<frame>.npz
  targets/<task>/<episode>/<frame>.npz
  preview/index.html                        # 整批构建完成后生成
  preview/<task>/<episode>/<frame>/preview.png
  preview/<task>/<episode>/<frame>/point-cloud-roi.html
~~~

观测包含实测 `tcp_pose`、`joint_positions` 和 `gripper_open01`；独立 targets NPZ 保存 `key_action/key_joint_action/key_valid/key_frame`、`action_chunk[K,8]`、`joint_chunk[K,7]`、`action_frames`、`action_timestamps` 和 `action_valid`。位置是 world 米、TCP 四元数是 xyzw、夹爪 1 开 0 关。每个 payload 都有独立哈希，完整校验会检查观测、目标、mask、时间、边界和组划分。

构建进度写 stderr，包含当前 episode、样本数、跳过窗口数、实际 dt 和耗时；`--quiet` 隐藏进度。`complete.json` 记录各划分数量、原始 dt 统计与跳过原因。失败目录不写完成标记；重跑使用新的目录。

读取入口为 `OHTChunkDataset`，batch 中 chunk 为 `[B,1,K,8]`，历史轴与未来轴分开。V4 必须使用下方的 chunk 配置；单关键点配置和旧 role-cache 工具仍拒绝 V4。

构建预览中，青色为当前 TCP，品红色为有效的下一关键目标，橙色为有效的真实未来 chunk。轨迹直接叠加在原始 sensor RGB 图（使用缓存相机内外参进行透视投影，按面板尺寸缩放）以及全局/局部 XY/XZ/YZ 点云投影图上，不额外生成独立轨迹面板。连线仅连接实测采样点，不表示规划出的中间路径；相机投影不进行遮挡判定。尾部 padding 不绘制，无未来关键目标时仍显示有效 chunk。仅为启用 preview 后选中的样本生成图，预测与 GT 的对比图仍由训练/离线评估生成。

V4 开启 `--visualize-every N` 后，采样帧同时生成 PNG 和逐帧 HTML，无需 `--object-cloud-config`。整批完成、网页绑定最终 contract 后生成 `preview/index.html`，随后写入 `complete.json`；构建途中可能已有逐帧页面但尚无统一入口。`--visualize-output-dir` 可将这些文件放到指定预览目录；`0` 关闭预览。旧代码生成的 PNG 不会自动补出 HTML，请用下面命令导出。

已有完整 V4 可单独刷新 PNG/HTML，不重建数据、不改写观测或标签：

~~~bash
OHT_ROOT=/common-data-32t/data/robot_data/oht_curobo_pd_v423
python tools/export_oht_replay_viewer.py \
  --replay "$OHT_ROOT/replay-source-v4" \
  --output "$OHT_ROOT/replay-source-v4/preview/chunk-refresh-01" \
  --every 100 --max-points 40000
~~~

打开输出目录的 `index.html`，轨迹显示在页面上方的 PNG 预览中。`--every` 按每个 episode 的缓存样本计数，包含首个；输出目录必须尚不存在。导出会核对选中样本的观测/目标哈希和 chunk 标签有效性，不读取原始视频。

### 分割增强数据：语义 ID 转 target/reference

`oht_curobo_pd_v423-seg` 是重新采集的单任务数据，目录为 `disassemble_left-seg/lerobot_dataset`，工具将其任务身份规范为 `disassemble_left`。不能与旧数据按 episode 编号配对或给旧 replay 补 mask；需单独 audit、构建新的 replay。原版与 `-seg` 同名任务同时出现在一个根目录时会拒绝读取，避免轨迹身份冲突。

`--root` 指向只含原始数据的根目录，**不需要预先准备 contract.json、replay 或 audit.json**。这些由脚本从头生成。`--output` 是新建的流水线输出目录，最终训练/导出使用其中的 `replay-source-v4/`；其 `contract.json` 和 `complete.json` 必须由构建步骤产生。脚本会核对这两个文件再进入校验阶段。

可直接运行完整生成脚本（默认 V4，包含关键目标、连续稠密 action chunk 和分割角色 mask）：

```bash
ROOT=/media/34.92/common-data-32t/data/robot_data/oht_curobo_pd_v423-seg
bash finetune/OHT/prepare_seg_data.sh \
  --root "$ROOT" --output "$ROOT/bridgevla-seg-v4-run01" \
  --chunk-horizon 16 --action-frame-stride 1
```

默认脚本依次执行 audit → V4 replay → 完整校验 → seg-role annotations → role-preview（预览开启时），任一步失败立即停止。V4 targets 保存下一关键目标、未来实测 TCP/夹爪 `action_chunk[K,8]`、关节/夹爪 `joint_chunk[K,7]` 及有效 mask、帧号和时间戳；不使用原始 action 命令或关键点间插值。默认每个观测监督未来 16 个连续原始帧，`--sample-stride 10` 只控制输入观测抽样，不会将 action stride 改成 10。原始 60 fps 数据的动作间隔约为 1/60 s，实际间隔由原始时间戳推导并校验。原始数据不覆盖，`--output` 必须为新目录；目录内保存 `dataset.yaml`、`seg_roles.yaml` 配置快照和 `pipeline.json` 执行状态，各工具直接打印进度。只包含 disassemble_left 的数据无需补其他任务。

默认 `--sample-stride 10 --visualize-every 100 --workers 4 --point-count 512`。可用 `--config /path/to/dataset.yaml --mapping /path/to/seg_roles.yaml` 指定已确认的标定/角色配置，`--dry-run` 查看命令且不创建文件。disassemble_left 的角色在全部阶段固定：target 为左轮 ID 1，reference 为放置盘 ID 4。也可直接执行 `python tools/prepare_oht_seg_data.py`，参数相同；Bash 入口可通过 `PYTHON_BIN` 指定解释器。

正常运行会先打印原始数据根目录、解释器、输出目录和 replay 路径，并在启动 audit 前写出 `pipeline.json`。若当前输出目录没有任何 JSON，检查终端最早的启动/路径错误及 `--dry-run`；尚未生成的 replay 不能用于训练或 mask 导出。`--root "$ROOT"` 中的变量必须已设置，空字符串会立即报错。配置快照失败也会记录在 `pipeline.json` 的 `failed_stage=setup` 中。

仅需要原有单关键点 replay 和 3D role teacher 时，显式选择 V3：

```bash
bash finetune/OHT/prepare_seg_data.sh \
  --root "$ROOT" --output "$ROOT/bridgevla-seg-v3-run01" \
  --replay-version 3
```

V3 在 mask 导出后继续生成 teacher。V4 在导出 mask 后，按 `--visualize-every` 生成 `role-preview/index.html`：target 左轮为绿色、reference 放置盘为蓝色、重叠为黄色，叠加在 sensor RGB、全局/局部点云投影及交互式 3D 点云上；PNG 保留橙色稠密轨迹。`replay-source-v4/preview/index.html` 是构建时生成的原始预览，不含后续角色 mask。`--visualize-every 0` 同时关闭两个预览。V4 **不运行不兼容的 teacher 步骤**，也不表示 chunk 训练已接入角色监督。以下为各步骤的独立命令。

先在 BridgeVLA 根目录构建新数据缓存（以下为单关键点版本；chunk 数据使用上方 V4 构建参数和新输出目录）：

```bash
SEG_ROOT=/media/34.92/common-data-32t/data/robot_data/oht_curobo_pd_v423-seg
python tools/audit_oht_dataset.py --root "$SEG_ROOT" --output "$SEG_ROOT/audit.json"
python tools/build_oht_replay.py \
  --root "$SEG_ROOT" --manifest "$SEG_ROOT/audit.json" \
  --config finetune/OHT/configs/dataset.yaml \
  --output "$SEG_ROOT/replay-source-v3" --sample-stride 10

python tools/export_oht_seg_roles.py \
  --root "$SEG_ROOT" --manifest "$SEG_ROOT/audit.json" \
  --replay "$SEG_ROOT/replay-source-v3" \
  --mapping finetune/OHT/configs/seg_roles.yaml \
  --output "$SEG_ROOT/seg-role-annotations"

python tools/build_oht_role_teacher.py \
  --replay "$SEG_ROOT/replay-source-v3" \
  --annotations "$SEG_ROOT/seg-role-annotations/annotations.jsonl" \
  --output "$SEG_ROOT/role-teacher" --point-count 512 --visualize-every 100
```

`dataset.yaml` 的深度、相机和夹爪参数仍需匹配采集定义。该根目录只有一个任务，audit 报告其余任务缺失属预期；坏 episode 仍需检查。

类别 ID：`0=robot, 1=wheel_left, 2=wheel_right, 3=wheel_stack, 4=plate, 5=shaft_left, 6=shaft_right, 255=背景/未标注`。其中原采集名 `shaft_1`（ID 5）已确认是左轴，`shaft_2`（ID 6）是右轴。这是语义分割，不能区分同类别的多个实例。disassemble_left 使用固定映射 `target: [1]`（左轮）、`reference: [4]`（放置盘），所有阶段保持一致。通用导出器仍允许其他任务按**当前帧** `instruction_id` 配置 `phases`，不会使用未来关键帧的阶段。`[]` 仅用于已确认语义为空的角色，`unknown` 表示未标注，二者不同。

旧标注若为 `reference: unknown`，已有 replay 和稠密动作标签可直接复用；使用更新后的 `--mapping finetune/OHT/configs/seg_roles.yaml` 将分割标注重新导出到新目录。原运行目录的 `seg_roles.yaml` 是历史快照，不会随仓库配置更新。单关键点版本依赖旧标注的 teacher 也需重新生成。

已有 V4 数据只刷新角色标注和网页，无需重建 replay（输出目录均需全新）：

```bash
ROOT=/media/34.92/common-data-32t/data/robot_data/oht_curobo_pd_v423-seg
OUT="$ROOT/bridgevla-seg-v4-run01"
python tools/export_oht_seg_roles.py \
  --root "$ROOT" --manifest "$OUT/audit.json" \
  --replay "$OUT/replay-source-v4" \
  --mapping finetune/OHT/configs/seg_roles.yaml \
  --output "$OUT/seg-role-annotations-wheel-plate"

python tools/export_oht_replay_viewer.py \
  --replay "$OUT/replay-source-v4" \
  --role-annotations "$OUT/seg-role-annotations-wheel-plate/annotations.jsonl" \
  --output "$OUT/role-preview-wheel-plate" --every 100
```

打开 `$OUT/role-preview-wheel-plate/index.html`。已按正确映射生成 mask 时可直接执行第二条命令，指向已有的 annotations.jsonl。预览验证标注与 replay 的契约和 annotation 哈希；新版标注还记录并校验 mask 哈希。旧标注没有 mask 哈希时仍可诊断预览，`role_preview.json` 明确记录源哈希是否齐备及本次读取的 mask 哈希。任何预览都不修改 observation、稠密动作标签或训练契约。

导出逐帧读取 `observation.seg.*` 引用的实际路径（包括非标准 `seg/`），按原始帧序号对齐，校验流长度、RGB/seg 引用时间戳和原始分辨率。只接受原生灰度 `uint8` 精确 ID，不经 RGB 转换或插值；与 RGB-D 一样做整数步长缩采样。小目标可能因缩采样消失，可提高 replay 的 `image_size` 后重建；不能用插值恢复标签。角色存在性来自映射，完全遮挡时保留 present/known，空 mask 对应无有效 3D geometry。

输出包含每个角色各视角的 bool mask NPZ、`annotations.jsonl` 及源视频/映射/契约哈希的 `provenance.json`；teacher 通过缓存深度点云选取可见表面，不凭分割补全被遮挡几何。预览叠加在 sensor 图和点云投影视图上。mask 和阶段标签是离线监督，不加入策略 observation；部署的 predicted object assistance 仍需由可部署的预测器提供角色几何，不能把 GT 分割当作预测结果。

导出器也接受 V4 `--replay` 并生成同样的 2D 标注。**V4 chunk 训练目前不接收 role cache**；上面的 teacher 命令及现有 role_queries/predicted_external 训练路径适用于单关键点缓存。输出目录需全新；失败后的不完整目录不能作为完成产物。

### V4：关键目标 + 连续 chunk 训练与评估

默认 `stage: joint` 可不加载已训练的 OHT EEF checkpoint，直接联合学习关键目标和 chunk；也可用兼容 EEF checkpoint 初始化。`stage: frozen_key` 冻结整个关键目标网络，仍要求 `--init-checkpoint` 或 `--resume`。默认冻结 Gemma 前 6 层、视觉编码器和 multimodal projector，训练后续 Gemma 层、关键目标头和 chunk decoder。chunk 分支只读取当前观测、实测 proprioception 和预测关键目标；GT 裁剪仅用于独立的关键目标监督分支。

已生成 `bridgevla-seg-v4-run01` 时，在仓库根目录直接训练，无需 EEF checkpoint：

~~~bash
OUT=/common-data-32t/data/robot_data/oht_curobo_pd_v423-seg/bridgevla-seg-v4-run01
GPUS_PER_NODE=8 bash finetune/OHT/train_8x40.sh \
  --config finetune/OHT/configs/oht_chunk_v4_8x40.yaml \
  --replay "$OUT/replay-source-v4" \
  --log-dir "$OUT/runs" --exp-id oht-seg-v4-joint-s0
~~~

`OUT` 改为实际生成目录（含 `/media/34.92` 等实际挂载前缀）。未传 `--init-checkpoint` / `--pretrain-path` 时使用 PaliGemma 基础模型，BridgeVLA 热图/动作头和 chunk decoder 从新初始化训练；可增加 `--pretrain-path /path/to/BridgeVLA-pretrain` 加载热图预训练资源。该参数不是 PaliGemma 普通模型目录。跳过 EEF checkpoint 仍需完整可访问或已缓存的 PaliGemma 模型及 processor；离线缓存完整时在启动前设置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。启动打印 `Chunk initialization: stage=joint source=paligemma_base` 或 `bridgevla_pretrain`。这种初始化的早期预测裁剪尚不准确，关键目标 GT 分支仍独立训练；不能据此保证与已训练 EEF 初始化相同的收敛速度或闭环性能。分割角色标签仍未接入此 chunk baseline。

数据校验及可选的兼容 EEF 权重初始化命令如下：

~~~bash
python -m finetune.OHT.train \
  --config finetune/OHT/configs/oht_chunk_v4_8x40.yaml \
  --replay /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v4 --validate-only

GPUS_PER_NODE=8 bash finetune/OHT/train_8x40.sh \
  --config finetune/OHT/configs/oht_chunk_v4_8x40.yaml \
  --replay /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v4 \
  --init-checkpoint /home/yiwei/project/BridgeVLA/finetune/OHT/runs/baseline-eef-8gpu-s0/model_last.pth \
  --log-dir /common-data-32t/data/robot_data/oht_curobo_pd_v423/runs --exp-id key-chunk-v4-s0
~~~

可选的 `--init-checkpoint` 替换为实际旧 EEF 模型，其解析后的相机、TCP、夹爪、图像尺寸及 data profiles 必须与新 V4 一致；新 chunk decoder 随机初始化。joint 不传该参数时按上一命令联合训练。新 run 继续使用时间戳目录、epoch、全局 batch、AdamW、TensorBoard 和模型保存机制。`trainable_parameters.json` 列出可训练层。全局刚体增强关闭，避免破坏实测关节与 TCP 的关系。

~~~yaml
freeze_language_model: false         # 使用下方的按层数冻结
freeze_gemma_prefix_layers: 6        # 冻结索引 0..5；可按需要调整
freeze_multimodal_projector: true
freeze_vision_tower: true
chunk_policy:
  enabled: true
  stage: joint                      # frozen_key 则冻结整个关键目标网络
  key_loss_weight: 1.0
  position_weight: 5.0               # chunk 平移项权重；未指定时为 1.0
~~~

联合损失为 `chunk_loss + key_loss_weight * key_loss`；key loss 沿用关键目标的 heatmap、旋转和夹爪监督，只计算 `key_valid=true` 的样本。每批先计算独立的 GT 监督分支，再通过预测目标裁剪提取 chunk 条件；两条路径共享模型权重，GT 特征不会送入 decoder。关键目标的离散解码及裁剪位置保持 detach；chunk loss 可沿预测分支更新未冻结的 Gemma 后续层，但不穿过 argmax。冻结模块保持 eval，BatchNorm 运行统计固定，仿射参数仍遵循冻结规则。`lm_head/embed_tokens` 沿用原有冻结规则；若显式设置 `freeze_language_model=true`，则覆盖层数设置并冻结整个语言模型。

chunk 损失为 `position_weight * L_pos + rotation_weight * L_rotation + gripper_weight * L_gripper`。`oht_chunk_v4_8x40.yaml` 将 `position_weight` 设为 5，其他两项为 1，`position_scale` 保持 0.25。未指定 `position_weight` 的旧配置仍按 1 解析；`chunk_position_loss` 记录原始值，`chunk_position_weighted_loss` 记录加权值。权重 5 是待验证的对照设置，需通过物理位置误差和轨迹图判断效果。

**相邻位移监督实验。** [oht_chunk_v4_displacement_8x40.yaml](configs/oht_chunk_v4_displacement_8x40.yaml) 继承上述 pos=5 配置，增加 `displacement_weight: 1.0`、`displacement_scale: 0.01`。总损失另加 `displacement_weight * L_displacement`：第一项比较预测/真实的 `p₁−当前实测 TCP`，后续比较各自的 `pₖ−pₖ₋₁`，使用按 `displacement_scale` 归一化的 SmoothL1。0.01 的单位为米，只控制辅助 loss 的归一化，不改变模型输出尺度或动作执行。它监督真实运动的方向和幅度，不是让轨迹更平滑，也不要求 chunk 终点等于远处关键目标。该设置仍需实验验证。

仅计算有效的当前→第一帧及连续有效帧对，不跨越无效标签；末尾 padding 不参与梯度。默认 `displacement_weight: 0.0` 保留旧训练目标；所有配置默认 `displacement_scale: 0.01`。V4 已有当前 TCP 和连续未来标签，无需重建数据。用上一实验的 chunk checkpoint 初始化可保留 encoder/decoder 权重，但改变监督需新建 run：

~~~bash
GPUS_PER_NODE=8 bash finetune/OHT/train_8x40.sh \
  --config finetune/OHT/configs/oht_chunk_v4_displacement_8x40.yaml \
  --replay "$OUT/replay-source-v4" \
  --init-checkpoint /path/to/previous-chunk-run/model_last.pth \
  --log-dir "$OUT/runs" --exp-id oht-seg-v4-pos5-displacement-s0
~~~

`OUT` 使用本节前面的实际数据生成目录。TensorBoard/JSONL 增加原始 `chunk_displacement_loss`、加权后的 `chunk_displacement_weighted_loss`，以及每个有效位移的平均欧氏误差 `chunk_displacement_error_m`、保持当前位置的对照 `chunk_stationary_displacement_error_m`（均为米/步，不是 m/s）。调用训练入口时，即使辅助权重为 0 也计算这些位移诊断，便于与原 baseline 对比；统计仍为 rank 0 本地 microbatch 均值。判断效果时同时检查绝对位置误差、位移误差和轨迹图，避免只优化局部位移而产生整体位置偏差。

**相对 TCP 关键目标条件实验。** [oht_chunk_v4_relative_goal_8x40.yaml](configs/oht_chunk_v4_relative_goal_8x40.yaml) 独立继承 pos=5 baseline，设置 `chunk_policy.goal_conditioning: relative_tcp`，不默认叠加位移辅助 loss。decoder 的目标 token 使用 `inverse(T_current_tcp) @ T_predicted_key`：平移为 `R_currentᵀ(p_key−p_current) / position_scale`，旋转为 `inverse(q_current) * q_key`（xyzw、单位四元数并统一符号），夹爪仍为预测关键目标的 0/1。只使用当前实测 TCP 和预测关键目标，训练与在线推理共用同一路径。

当前本体状态、预测关键目标的返回值、chunk 输出和训练标签仍为世界坐标绝对 EEF；相对条件不改变旋转输出或客户端执行方式，也不引入历史状态。已有 V4 数据无需重建。旧配置未设置此项时继续使用 `absolute`，旧 checkpoint 的推理行为保持一致。该修改是否改善方向或位置误差需实验验证。

~~~bash
GPUS_PER_NODE=8 bash finetune/OHT/train_8x40.sh \
  --config finetune/OHT/configs/oht_chunk_v4_relative_goal_8x40.yaml \
  --replay "$OUT/replay-source-v4" \
  --init-checkpoint /path/to/previous-chunk-run/model_last.pth \
  --log-dir "$OUT/runs" --exp-id oht-seg-v4-pos5-relative-goal-s0
~~~

相对/绝对条件切换必须用 `--init-checkpoint` 新建 run：保留 encoder 和 chunk decoder 的其他权重，仅重新初始化 `decoder.goal` 投影层，optimizer 从新开始；不能 `--resume` 接续旧语义训练。联合训练也可省略初始化 checkpoint 从头训练。服务端直接读取新 checkpoint 内的配置，无需增加客户端参数。若要组合位移实验，可在相对目标配置中显式增加 `displacement_weight: 1.0`、`displacement_scale: 0.01` 后新建独立 run；单项对照应保持两次训练的 loss 权重一致。

**显式三维条件与 flow matching。** 新增 [三维回归配置](configs/oht_chunk_v4_spatial_8x40.yaml) 和 [flow 配置](configs/oht_chunk_v4_flow_8x40.yaml)。两者共用原始渲染表面的 XYZ/有效性、视角与尺度标识、TCP 局部坐标、三维相对 attention 和未来物理时间，且均开启位移监督；flow 另以向量场损失学习整段 pose，默认 8 次积分。推理输出仍为世界坐标绝对 EEF，HTTP/IsaacLab 客户端无需修改。

~~~bash
OUT=/common-data-32t/data/robot_data/oht_curobo_pd_v423-seg/bridgevla-seg-v4-run01
GPUS_PER_NODE=8 bash finetune/OHT/train_8x40.sh \
  --config finetune/OHT/configs/oht_chunk_v4_flow_8x40.yaml \
  --replay "$OUT/replay-source-v4" \
  --init-checkpoint /path/to/previous-chunk-run/model_last.pth \
  --log-dir "$OUT/runs" --exp-id oht-seg-v4-flow-s0
~~~

`OUT` 按实际生成目录设置。三维回归对照将 config 改为 `oht_chunk_v4_spatial_8x40.yaml`，另用独立 `--exp-id`。新配置沿用当前 chunk baseline 的 epoch、batch、AdamW、视觉/projector 冻结和 Gemma 前缀层数；已有 V4 不需重建。旧模型初始化新结构时保留 encoder、重新初始化 decoder/optimizer，须新 run，不能 `--resume`；joint 也可不传 `--init-checkpoint`。当前未加入本体历史或跨 chunk 延续。

flow 的 loss 包含带噪 GT 重建，因此 `chunk_*_loss` 不代表部署误差。轨迹图与 `chunk_position_error_m`、`chunk_displacement_error_m`、`chunk_prefix_1_position_error_m`、`chunk_prefix_<H>_position_error_m` 均使用独立部署采样，不读取未来标签；另记 `chunk_reconstruction_position_error_m`。`/model_info.chunk_policy` 可核对实际配置；默认 `flow_inference_seed: 0` 固定采样便于复现，设为 `null` 则随机。算法与坐标细节见[设计说明](../../docs/design/oht-spatial-flow-chunk.md)。实际精度、吞吐与显存仍需 GPU 验证。

可在派生配置中将 `stage` 改为 `frozen_key` 做对照。从 frozen_key 切换到 joint、改变 loss 权重或冻结策略时，使用 `--init-checkpoint` 新建 run，不复用旧优化器；仅阶段/loss 权重变化时保留已训练的 encoder 和 decoder 权重。相同配置才能 `--resume`。两种阶段共用 V4 数据，无需因此重新构建。

新配置默认 K=16、建议执行 H=4。可将模型 K 减小到 cache horizon 以内做对照，无需重建同一 V4 数据；超过监督 horizon 则拒绝训练。TensorBoard 记录总损失、关键目标各项损失、masked chunk 位置/旋转/夹爪损失，以及预测关键目标热图和 chunk 的 XY/XZ/YZ 预测与实测未来路径。预测条件复用逐样本部署路径；joint 还增加一次有效关键目标子批的监督 forward，吞吐和显存需在实际 GPU 上测量。

训练会打印实际 run 目录。将该目录填写为环境变量后执行：

~~~bash
CHUNK_RUN=/common-data-32t/data/robot_data/oht_curobo_pd_v423/runs/实际训练目录
python -u -m finetune.OHT.eval open-loop \
  --checkpoint "$CHUNK_RUN/model_last.pth" \
  --replay /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v4 --split val \
  --output /common-data-32t/data/robot_data/oht_curobo_pd_v423/eval/key-chunk-v4-val.json

GPU_ID=1 bash finetune/OHT/vla_server.sh \
  --model-path "$CHUNK_RUN/model_last.pth" \
  --device cuda:0 --host 0.0.0.0 --port 8004
~~~

离线评估按未来 offset 报告位置、旋转、夹爪和样本数量，关键目标误差独立统计；默认导出路径 PNG/NPZ 和预测热图。它仍不是任务成功率。

实时客户端沿用 [IsaacLab 启动命令](../../docs/guides/oht-vla-server.md#isaaclab-实时-rgb-d-客户端)。`--execution-mode auto` 会根据 `/model_info.chunk_spec` 选择 chunk 执行；可增加 `--chunk-execute-horizon 4` 覆盖 H。`--max-steps` 仍为策略请求次数，日志另记每次消费的动作数量和控制步数。

响应 schema 为 `bridgevla_oht_eef_chunk_v1`，返回 `key_action`、`actions[K,8]`、`time_offsets`；不返回单动作 `action`。请求需通过 `prediction_schema` 确认输出格式，新版客户端和 smoke 工具自动处理。客户端校验身份、K、dt 和坐标，再按时间消费前缀并丢弃后缀。夹爪按各 tick 切换，不等待远处关键点到位。当前要求 action_dt 为仿真 control dt 的整数倍，网络等待时只刷新窗口、不推进物理。

默认跟踪误差上限为 0.10 m / 0.5 rad，可通过 `--chunk-max-tracking-translation/rotation` 设置；超限停止并写日志，不反复执行不可达目标。在线页面显示关键目标热图、首个动作投影和 3D chunk 路径；红色前缀按服务端建议 H 绘制，客户端覆盖 H 后以本地执行日志为准。该版本不包含碰撞规划。

### 完整重新生成 replay

以下为现有单关键点版本。使用当前仓库配置，从原始 RGB-D 重新生成 replay、近似对象点云和预览。已有有效的 `/common-data-32t/data/robot_data/oht_curobo_pd_v423/audit-seed0.json` 可直接复用；输出目录 `/common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3` 必须尚不存在。再次重建时更换新目录名，保留旧结果用于对照。

~~~bash
python tools/build_oht_replay.py \
  --root /common-data-32t/data/robot_data/oht_curobo_pd_v423 \
  --manifest /common-data-32t/data/robot_data/oht_curobo_pd_v423/audit-seed0.json \
  --config finetune/OHT/configs/dataset.yaml \
  --output /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3 \
  --sample-stride 10 \
  --object-cloud-config finetune/OHT/configs/object_clouds.yaml \
  --visualize-every 1

python tools/validate_oht_replay.py --replay /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3 --workers 4
~~~

`validate` 会检查全部划分的全部样本，逐个计算观测 NPZ 哈希、解压并校验相机/状态/标签；它是 CPU/磁盘任务，不使用 GPU，不能仅按 400 个 episode 估算耗时。`--workers 4` 使用四个线程并行检查，默认仍为串行 `--workers 1`，校验范围相同。进度写到 stderr，显示当前划分的完成样本数、samples/s 和预计剩余秒数，约每 5 秒更新（开始/结束也输出）；最终 JSON 仍写 stdout。`--quiet` 仅关闭进度。机械盘、共享存储或内存紧张时可降为 `--workers 1` 或 `2`；更多线程不保证更快。预览图片、HTML 和对象诊断点云不参与此项训练 replay 校验。

构建结束并通过校验后，打开 **`/common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3/preview/index.html`**。PNG 保留原有场景面板，底部增加各对象点云区域；HTML 上方显示 PNG，下方提供点云交互、视角和帧选择。

- `--sample-stride 10` 控制哪些原始帧进入 replay；完整重建仍沿用该采样策略，如需更密的 replay 可减小步长。
- `--visualize-every 1` 为每个生成的 replay 样本输出 PNG/HTML；改为 `100` 可减少预览数量和开销，对象 PLY/NPZ 仍逐样本生成。
- 本段单关键点版本使用 `--object-cloud-config` 启用对象截取和 HTML 对象预览；只传 `--visualize-every` 时仅生成基础 PNG。上文 V4 版本开启预览即可生成基础 HTML 入口。
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

每张图包含缓存分辨率的各相机 RGB、米制 depth，以及两排 XY/XZ/YZ 彩色点云：青色为当前 `EE-pos`，品红色为下一 GT 关键点。当前 OHT 配置的 `link_to_tcp` 为单位阵，青色位置就是原始 `observation.ee_pos_world`；PNG/HTML 均显示 `EE-pos`，缓存字段仍为 `current_tcp`。若配置了额外 TCP 偏移，显示名称保留 `TCP`，不把偏移后的位置标成原始 EE。depth 蓝色近、红色远，标注当前有效范围（米），黑色表示无效；不同图的深度颜色范围可能不同。

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

带对象配置的 HTML 默认勾选“仅显示勾选对象框内点云”，隐藏框外背景。统一入口顶部和单帧页面均提供“恢复完整场景”，点击后显示所有相机的场景抽样点、恢复点透明度并缩放到场景范围，保留对象选择、名称、尺寸、姿态轴和 ROI 配置；重新勾选“仅显示勾选对象框内点云”可返回物体视图。完整场景仍受所选点源和 HTML 抽样预算限制；缓存 XYZ 已被裁掉的点需改用 depth 源重新导出预览。

上方 PNG 是生成时配置对应的静态预览；下方勾选对象和调整尺寸仅更新交互点云，不重写 PNG。顶部“查看下方交互点云”可直接跳到交互区。

如果修改代码后看起来没变化，先检查生成命令是否带 `--object-cloud-config`，以及打开的是否为本次新输出。旧 PNG/HTML 不会随代码更新自动改变；旧 replay 未记录对象配置时，下面的独立导出命令需显式增加对象 YAML。

仅更新可视化、无需重算 replay 时，可为已有**完整 replay** 导出新的 PNG/HTML。已执行上面的完整重建则不需要再运行此命令：

~~~bash
python tools/export_oht_replay_viewer.py \
  --replay /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3 \
  --output /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3/preview/refresh \
  --object-cloud-config finetune/OHT/configs/object_clouds.yaml \
  --every 1 --max-points 40000
~~~

打开 `/common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3/preview/refresh/index.html`。输出目录必须尚不存在，每帧 PNG/HTML 保存到 `<output>/<task>/<episode>/<frame>/`。`--every N` 按各 episode 的 replay 样本数选帧，包含首个；浏览器每次仅加载一帧。默认源为缓存米制 depth/K/外参，可用 `--source xyz` 改为缓存 XYZ。导出校验已完成 replay 和观测 NPZ，不改写缓存，也不额外导出 PLY/NPZ。

#### 单帧几何排错（可选）

需要检查多相机融合参数时，使用 `samples.jsonl` 中真实存在的 sample ID。以下命令同时输出几何诊断和离线点云 HTML，目录必须尚不存在：

~~~bash
python tools/diagnose_oht_geometry.py \
  --replay /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3 \
  --sample-id assemble_left/000000/000000 \
  --output /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3/preview/geometry-frame0 \
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

仅旧 replay 缺少对象元数据时，导出工具可同时增加 `--object-root /common-data-32t/data/robot_data/oht_curobo_pd_v423` 和 `--object-manifest /common-data-32t/data/robot_data/oht_curobo_pd_v423/audit-seed0.json`，清单必须是构建该 replay 时使用的 audit。单帧诊断还需启用 `--html`。读取器核对 audit、Parquet 哈希及当前帧，绑定不符则拒绝输出；不改写缓存，也不猜测缺失位姿。新重建流程无需这些补充参数。

#### 没有实例 mask 时近似提取 object 点云

HTML 支持以当前记录的对象中心为中心，设置 **world X/Y/Z 轴对齐框的全长（米）**。每轴初始 0.20 m 只是可调整的预览值，不代表实物尺寸；不依赖对象四元数，不随姿态轴顺序旋转。若记录原点不在几何中心，需要调整框尺寸以覆盖物体。此方法截取的是框内观测点，可能包含桌面/邻近物体，不能作为实例分割 GT；重叠框允许共享点，不自动将点分配给最近对象。

1. 打开当前帧 HTML，勾选所需对象，调整每项的 X/Y/Z 全长。
2. 开启“显示近似截取框”和“仅显示勾选对象框内点云”，检查物体是否被完整覆盖；点击“缩放到所选对象”可将整个截取框纳入视图。框边界为下界包含、上界不包含；点云仍受当前相机开关、工作区显示和背景透明度控制。
3. 若要按页面调整后的尺寸重新导出该帧对象文件，点击“下载对象截取配置 JSON”，将 `oht-object-boxes.json` 传回服务器，再运行下列命令。构建时已生成默认对象文件，无需重复提取；JSON 绑定当前 sample/frame、点源和缓存校验值，不修改 replay 或 ROI YAML。

~~~bash
python tools/extract_oht_object_clouds.py \
  --replay /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3 \
  --sample-id assemble_left/000000/000000 \
  --boxes /common-data-32t/data/robot_data/oht_curobo_pd_v423/oht-object-boxes.json \
  --output /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3/preview/object-clouds-frame0
~~~

旧 replay 缺少对象位姿时，在提取命令中同样增加 `--object-root` 和 `--object-manifest`，无需重建点云。仅诊断未完成 replay 可显式加 `--allow-incomplete`；训练完整性要求不变。对象配置绑定当前帧，不能直接用于另一帧或另一份缓存；重新预览相应样本后再导出。

输出为每个选中对象的 `object_<index>.ply`（world XYZ + RGB）、`object_<index>.npz`（`xyz[N,3]`、`rgb[N,3]`、`camera_ids[N]`），以及 `object-clouds.json`（框范围、每相机点数、校验值和 `approximate: true`）。没有框内点时输出空点云并记录 `valid: false`，不补造物体。输出目录必须是新目录。

**HTML 是显示抽样，导出命令读取缓存分辨率下的全部有效点**，不受 HTML 点数预算、相机开关或页面工作区显示过滤影响，也不做额外去重/降采样。JSON 的 `source: xyz` 使用已缓存点，无法恢复旧 ROI 的 NaN；`source: depth` 使用缓存米制深度/K/外参重建，可恢复旧 ROI 删掉但深度仍有效的点。全量导出不等于恢复原始分辨率或完整物体表面，不能修复深度/标定/同步错误。提取结果仅作近似对象诊断，不自动进入角色教师或策略输入。

#### 构建未完成时

`samples.jsonl` 在 buffer 完成时才写入。若尚无索引，直接查看已经写完的 NPZ：

~~~bash
python tools/diagnose_oht_geometry.py \
  --replay /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3 \
  --observation observations/assemble_left/000000/000000.npz \
  --data-profile assemble_left/lerobot_dataset \
  --allow-incomplete --html --html-source depth \
  --output /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3/preview/geometry-building-frame
~~~

NPZ 路径相对于 replay，不能越出该目录；`--data-profile` 必须替换成 `contract.json.source_data_configs` 中对应数据源的键，多源时必须显式提供，单源可省略。`--observation` 与 `--sample-id` 二选一；它只依赖已有 contract 和 NPZ，不依赖索引或完成标记，也不读取原始视频。

直接 NPZ 模式不猜测帧时序、TCP 或 goal：HTML 标注未知，PNG 仅保留全局三视图，不生成 GT-centered 局部图。页面和 `geometry.json.validation` 明确记录仅诊断、未验证整批完成；NPZ 的实际哈希仅作本次快照记录，不冒充索引校验。文件仍在写入、缺少相机字段或标定不合法时拒绝输出，稍后换新输出目录重试。不会补写 `complete.json`/索引或使未完成 buffer 可用于训练。

## 4. Baseline 训练

这里训练的是 **BridgeVLA on OHT baseline**：`mode: baseline` 关闭物体辅助，默认 `action_mode: eef` 预测世界系绝对 TCP 位姿和夹爪。无需提供 teacher、外部预测器或 `--role-cache`。

以下 Bash 命令从 **BridgeVLA 仓库根目录** 执行；正式训练需要第 1 节的 Linux/CUDA 模型环境，数据预检可在 CPU 环境执行。先设置实际路径：

~~~bash
OHT_REPLAY=/common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3
BRIDGEVLA_PRETRAIN=/path/to/bridgevla-heatmap-pretrain
~~~

`OHT_REPLAY` 指已经完整构建的 replay，目录中应有 `complete.json`、`contract.json` 和 `samples.jsonl`。`BRIDGEVLA_PRETRAIN` 必须替换为实际的 **BridgeVLA 热图预训练权重目录**，包含 `model.safetensors.index.json` 和对应分片；模型构造还需已缓存或可访问的 PaliGemma 基础权重，详见第 1 节。原始数据目录或单独的 preview 目录不能直接作为训练 replay。

**第一步：CPU 数据预检。**

~~~bash
python -m finetune.OHT.train \
  --config finetune/OHT/configs/baseline.yaml \
  --replay "$OHT_REPLAY" \
  --output /common-data-32t/data/robot_data/oht_curobo_pd_v423/check-only \
  --validate-only
~~~

输出 `valid: true` 后继续；`--validate-only` 不启动模型，也不创建训练输出目录。

若刚刚已对同一份未修改的 replay 成功运行 `validate_oht_replay.py --workers 4`，且本次使用默认 EEF baseline，可直接进行第二步，避免再次扫描训练集。其他动作设置或辅助模式仍需按对应配置/role cache 预检。

**第二步：两步 GPU 短测。**

~~~bash
python -m finetune.OHT.train \
  --config finetune/OHT/configs/smoke.yaml \
  --replay "$OHT_REPLAY" \
  --output /common-data-32t/data/robot_data/oht_curobo_pd_v423/runs/smoke \
  --pretrain-path "$BRIDGEVLA_PRETRAIN"
~~~

短测使用 batch 1、累积 1、2 个优化步。确认 loss 有限且成功生成 `model_last.pth`，再进行正式训练。

**第三步：单卡正式训练。**

~~~bash
python -m finetune.OHT.train \
  --config finetune/OHT/configs/baseline.yaml \
  --replay "$OHT_REPLAY" \
  --output /common-data-32t/data/robot_data/oht_curobo_pd_v423/runs/baseline-s0 \
  --pretrain-path "$BRIDGEVLA_PRETRAIN"
~~~

上述 `--output` 固定输出目录均需尚不存在；重复实验时更换目录名，中断恢复使用 `--resume`。需要与 RLBench 一样自动按时间分目录时，使用下面的 `--log-dir`。

**实验名、备注和时间目录。** `--log-dir ROOT`（也接受 RLBench 风格的 `--log_dir`）将模型、配置、TensorBoard 和 PNG 保存在 `ROOT/train/<exp_id>/<exp_note>/MM_DD_HH_MM/`。默认实验名为配置中的 `exp_id` 或训练 YAML 文件名（去掉后缀），也可用 `--exp-id` / `--exp_id` 覆盖；备注使用 `--exp-note` / `--exp_note`，为空时省略该层。未指定 `--output` 或 `--log-dir` 时默认根目录为 `exp/OHT`。

时间采用 Asia/Shanghai（UTC+08:00），目录命名与 RLBench 相同；同一分钟重复启动自动追加 `_01`、`_02` 等后缀，不覆盖已有 run。rank 0 创建目录并向全部 GPU 广播同一个路径。启动时打印 `Run directory: ...`，`run.json` 记录实际绝对路径、包含年份/秒/时区的创建时间、实验名、备注和启动参数。

~~~text
exp/OHT/train/oht_full_8x40/baseline_s0/10_15_14_30/
├── run.json
├── config.json
├── contract.json
├── training_plan.json
├── metrics.jsonl
├── tensorboard/
├── train_visualizations/
└── model_last.pth                 # 以及按 epoch/step 保存的 checkpoint
~~~

`--output DIR` 继续表示精确固定目录，与 `--log-dir` 互斥。`--resume /path/to/model_last.pth` 自动写回 checkpoint 所在目录，保留原始 `run.json` 的创建时间，不生成新时间目录；可省略输出参数，保留原命令中的 `--log-dir` 也会沿用原 run。若同时保留 `--output`，它必须指向 checkpoint 所在目录。`--validate-only` 不创建任何 run 目录。服务器出现 `/home/...` 与挂载真实路径不同的情况时，以打印的绝对保存路径为准。

**多卡正式训练。** [train_8x40.sh](train_8x40.sh) 与 RLBench 同名脚本采用相同的 torchrun 启动参数和环境变量约定，默认单机 8 卡，使用 `--standalone`/DDP。OHT 使用 `-m finetune.OHT.train`，脚本自动进入仓库根目录，完整透传 OHT 训练参数。沿用前面设置的 replay 和预训练路径：

~~~bash
bash finetune/OHT/train_8x40.sh \
  --config finetune/OHT/configs/baseline.yaml \
  --replay "$OHT_REPLAY" \
  --pretrain-path "$BRIDGEVLA_PRETRAIN" \
  --log-dir /common-data-32t/data/robot_data/oht_curobo_pd_v423/runs --exp-note baseline-eef-8gpu-s0
~~~

用 `GPUS_PER_NODE=4 CUDA_VISIBLE_DEVICES=0,1,2,3` 作为上述 `bash` 命令的前缀即可改为四卡；两卡设 `GPUS_PER_NODE=2` 并选择两个 GPU。与 RLBench 脚本一致，`PYTHONWARNINGS` 默认 `ignore`（可用环境变量覆盖），`TRANSFORMERS_VERBOSITY` 设为 `error`，`MASTER_PORT` 默认 `29500` 并传给 torchrun；但 PyTorch 的 `--standalone` 会将 rendezvous endpoint 设为 `localhost:0`，实际端口自动分配，不能依赖 `MASTER_PORT` 固定该端口。脚本透传 batch/动作配置、`--role-cache`、`--init-checkpoint`、`--resume` 等训练参数，相对参数路径按仓库根目录解析。训练超参数仍由 OHT YAML 控制，启动脚本一致不代表训练配置已对齐 RLBench。`8x40` 沿用参考脚本的名称，不会自动检测显存或保证 40 GB 显存足够。

默认 seed 0、学习率 `4e-5`、每卡 batch 2、累积 8、2,000 个优化步，任务均匀采样；有效 batch = 每卡 batch × 累积次数 × 卡数，单卡为 **16**。这是初始配置，尚未经真实 OHT 收敛或显存验证。比较方法保持相同预算、冻结设置、数据划分和 seeds。

使用默认配置时，四卡有效 batch 为 **64**，八卡为 **128**；若要保持单卡的有效 batch **16**，在新训练配置中将四卡的 `accumulation_steps` 设为 **2**、八卡设为 **1**。不要在严格续训时更换卡数或累积设置。

**对齐 RLBench full 8×40 的正式配置。** [configs/oht_full_8x40.yaml](configs/oht_full_8x40.yaml) 对齐本仓库 [RLBench full 8×40](../RLBench/configs/rlbench_full_8x40.yaml) 的训练配方和 [rvt2.yaml](../bridgevla/mvt/configs/rvt2.yaml) 的 MVT 开关。原有 `baseline.yaml` 保留按优化步训练的初始配置；选择 full 预设时显式指定：

~~~bash
GPUS_PER_NODE=8 bash finetune/OHT/train_8x40.sh \
  --config finetune/OHT/configs/oht_full_8x40.yaml \
  --replay "$OHT_REPLAY" \
  --pretrain-path "$BRIDGEVLA_PRETRAIN" \
  --log-dir exp/OHT --exp-note baseline_s0
~~~

| 设置 | full 8×40 预设 |
| --- | --- |
| 训练预算 | `epochs: 100`，`train_iter: 160000`，最多 83,300 优化步 |
| batch | 每卡 4，全局有效 batch 固定 192；八卡自动累积 6 次，四卡 12 次，两卡 24 次 |
| 优化器 | AdamW，学习率 `8e-5`，weight decay `1e-6`，固定学习率 |
| 冻结 | 视觉塔、前 6 层 Gemma、multimodal projector、lm_head/embed_tokens |
| 模型 | 224/14、三视图、两阶段、convex upsampling、`rot_ver: 1`，其余 MVT 开关跟随 rvt2 |
| 增强 | 启用场景/EEF 联合变换，平移范围 `[0.125,0.125,0.125]`，绕 Z 旋转范围 45° |
| 内存策略 | 高效 PaliGemma 前向开启；gradient checkpointing、GPU 图像预处理、FlashAttention 2 关闭 |
| 日志和保存 | 保留每步 loss/学习率和 epoch 进度；首步及每 500 步记录 coarse/refine 图像；每 10 epochs 和训练结束保存 checkpoint |

这里的 epoch 沿用 RLBench 的**采样预算**定义：每轮优化步数为 `floor(train_iter / global_batch_size)`，即 `floor(160000 / 192) = 833`，每轮实际采样 159,936 次。仍采用任务均衡、有放回采样，不表示每轮恰好遍历 replay 一次；即使训练集只有 40,634 条，full 预算也保持不变。若希望按 replay 样本数定义每轮预算，可在新的自定义配置中设 `train_iter: null`，此时改为 `floor(训练样本数 / 全局 batch)`。全局 batch 必须能被每卡 batch × 卡数整除；`accumulation_steps: null` 表示自动计算。

启动时打印实际预算、冻结策略和可训练/冻结参数数量，并保存 `training_plan.json`。JSONL 每步包含 `epoch`、`epoch_step`、`epoch_progress` 和原有 loss；epoch 边界也打印到终端。full 预设保存 `model_epoch_10.pth`、`model_epoch_20.pth` 等周期快照、`model_last.pth`，结束时另存 `model_step_83300.pth`。checkpoint 包含已完成的优化步、完整 epoch 数和轮内位置。`--epochs N` 可覆盖 epoch 配置中的总目标轮数；`max_optimizer_steps` 仍为上限，要延长已达到上限的训练，应同时提高该上限或在配置中设为 `0`。日志横轴仍以优化步对齐，epoch_progress 是额外曲线。

OHT 保留自己的相机/坐标/夹爪/语言和数据契约，collision loss 为 0；不能将本预设称为未经改动的 RLBench 实验。相对 EEF/关节模式仍要求关闭几何增强：自定义配置继承 full 后设置对应 `action_mode` 及 `agent.transform_augmentation: false`。辅助模式可在继承 full 的配置中设置 `mode` 和角色参数，并使用对应 role cache；沿用 full 训练预算进行公平对照。切换到 full 会改变冻结和动作头等设置，需要新的训练输出，从预训练权重初始化；不能严格 resume 旧 baseline 的 checkpoint。该预设仍需在实际 GPU 上确认显存和收敛。

**预训练加载尺寸不匹配。** OHT 配置显式使用 `mvt.img_size: 224`、`mvt.img_patch_size: 14`，与 BridgeVLA 预训练和 PaliGemma 的 16×16 图像 token 网格一致。若日志出现 `ConvexUpSample net_mask.2` 的 `1764` 与 `1089` 不匹配，说明仍在使用旧的 `220/11` 渲染配置；更新 baseline 配置及自定义配置中的覆盖项后重新启动，保留严格权重加载。其他 OHT 预设继承 baseline，因此同步生效。这是模型渲染尺寸，不是缓存相机图像尺寸，已有有效 replay 无需重建或重复校验。初始化失败可能已经创建输出目录；本次失败没有可恢复的训练 checkpoint，重新运行时使用新的输出目录。

**选择其他动作模式的 OHT baseline。** 替换上述命令的 `--config` 即可，这些配置均保持 `mode: baseline`：

| 动作模式 | 配置 | 网络输出 |
| --- | --- | --- |
| `eef`（默认） | [configs/baseline.yaml](configs/baseline.yaml) | 世界系绝对 TCP 位姿 + 夹爪 |
| `relative_eef` | [configs/relative_eef.yaml](configs/relative_eef.yaml) | 相对 TCP 位移/旋转向量 + 夹爪 |
| `joint` | [configs/joint.yaml](configs/joint.yaml) | 六关节绝对位置 + 夹爪 |
| `relative_joint` | [configs/relative_joint.yaml](configs/relative_joint.yaml) | 六关节增量 + 夹爪 |

短测与正式训练需使用相同的 `action_mode`；现成 `smoke.yaml` 默认 EEF，其他模式短测需同步设置该字段并保留两步预算。新 replay 可用于四种模式；旧缓存若缺当前 TCP/关节状态或未来关节标签，预检会要求重建。相对坐标、单位和配置继承见 [四种动作模式指南](../../docs/guides/oht-action-modes.md)。

每个实际 run 目录输出 `run.json`、`config.json`、`contract.json`、`training_plan.json`、`metrics.jsonl`、`tensorboard/events.out.tfevents.*`、`model_last.pth` 及最终步 checkpoint。默认 baseline 每 250 个优化步更新 `model_last.pth`，完成 2,000 步后另存 `model_step_2000.pth`；full 的保存周期见上文。后续 `--init-checkpoint`、服务、评估或日志转换中的模型/run 路径应替换为实际打印的 run 目录；下文固定路径示例也可通过旧 `--output` 方式生成。

**TensorBoard 训练曲线。** 训练默认启用 `--log-backend tensorboard`；训练环境需安装 `tensorboard`（`python -m pip install tensorboard`）。数据构建、校验和 `--validate-only` 不需要该依赖。模型和 checkpoint 初始化成功后创建 `<output>/tensorboard/`，并打印日志路径。每个优化步写入 `train/<指标名>`（包括 total/translation/rotation/gripper 等实际返回的 loss）及 `train/learning_rate`；有多个优化器参数组时额外记录各组学习率。loss 与 JSONL 一致，为梯度累积内的平均值；多卡仅 rank 0 写入，指标是 rank 0 的本地统计，没有跨卡平均。训练图像通过下面的配置启用；闭环成功率仍需单独评估。

**TensorBoard 中间图像。** `oht_full_8x40.yaml` 默认开启；兼容旧行为的 `baseline.yaml` 默认关闭。在训练 YAML 中配置：

~~~yaml
train_visualization:
  enabled: true
  interval: 500
  save_png: true
  tensorboard: true
  output_dir: train_visualizations
~~~

启用后记录第 1 个优化步及每个 `interval` 的倍数步，在 TensorBoard 的 Images 中查看 `train_visualization/mvt1`（coarse）和 `train_visualization/mvt2`（refine）。图中包含该次训练实际渲染的输入 RGB、增强/投影后用于 loss 的 GT 定位热图、模型预测定位热图；模型返回角色监督、Target/Reference prior 或 relation anchor 时也显示对应面板。每次显示 rank 0 最后一个累积 microbatch 的首个样本，并记录 task 和 instruction。关节模式的定位热图来自辅助 EEF 头，不代表预测关节角或闭环轨迹。

PNG 同时保存为 `<output>/train_visualizations/step_00000001_mvt1.png`、`..._mvt2.png` 等；相对 `output_dir` 按训练输出目录解析。`save_png: false` 可仅写 TensorBoard；只保存 PNG 时设置 `tensorboard: false`，可配合 `--log-backend none`。同时关闭两个输出需将 `enabled` 设为 false。只在采集步请求真实前向载荷，不额外运行模型，不为其余 rank/microbatch 复制图像；新增开销是 CPU 传输、拼图和图像写入，可增大 `interval` 控制。图像只用于训练诊断，含 GT 的诊断内容不会成为额外策略输入。

V4 chunk 训练另有 `train_visualization/chunk`：红线是预测，绿线是同一样本的实测未来 TCP，白色十字是当前 TCP。两条线均只绘制 `action_valid` 有效的未来时刻，使用世界坐标及各面板内等比例轴；图中显示 sample ID、步数、采样间隔，以及有效时刻的平均/最大/末点位置误差（mm）。`save_png: true` 时同步保存 `chunk_step_<step>.png` 和 `.npz`，NPZ 包含本次 loss 前向的原始预测、标签、有效性、当前 TCP 和样本信息，便于逐点核对。

标量 `train/chunk_position_error_m` 是有效未来时刻的平均欧氏位置误差，`train/chunk_stationary_error_m` 是始终保持当前 TCP 的对照误差；两者单位为米，沿用上述本地 microbatch 累积平均规则。`chunk_position_loss` 则是按 `position_scale` 归一化后的 SmoothL1，不能直接当作米误差；loss 下降不等同于轨迹准确，需结合这两个物理误差及图中单样本误差判断。

已运行的旧进程不会自动新增图像；更新代码并从兼容 checkpoint 恢复后生效。`train_visualization` 仅控制诊断，可在续训时调整，不参与优化器连续性指纹；旧 JSONL 没有保存模型图像，历史标量转换工具无法补出这些中间结果。

默认每 10 秒刷新一次，可用 `--tensorboard-flush-secs 1` 缩短；保存 checkpoint 时也刷新，正常结束或 Python 异常退出时关闭 writer。用 `--log-backend none` 关闭 event 写入，仍保存 `metrics.jsonl`。这些命令行日志选项不改变训练配置/续训指纹。`--resume` 在同一日志目录从 checkpoint 的下一步继续写入，并隐藏旧日志中晚于该 checkpoint 的标量，避免回退后出现重复曲线。

~~~bash
tensorboard --logdir exp/OHT/train --port 6006
~~~

服务器运行 TensorBoard 时，在本机建立 SSH 端口转发后访问 `http://localhost:6006`；例如 `ssh -L 6006:127.0.0.1:6006 yiwei@serv92`。`--logdir` 可指向日志根目录下的 `train/` 来聚合实验和时间目录，也可指向单次 run 的 `tensorboard/`；使用自定义 `--log-dir` 或旧固定输出时替换为对应路径。

**旧训练只有 JSONL 时补导曲线。** 更新代码不会为已运行的旧进程自动补写 event；使用 [export_oht_tensorboard.py](../../tools/export_oht_tensorboard.py) 转换已有完整日志，无需重跑训练：

~~~bash
python tools/export_oht_tensorboard.py --run /common-data-32t/data/robot_data/oht_curobo_pd_v423/runs/baseline-eef-8gpu-s0
~~~

默认写入 `<run>/tensorboard-from-jsonl/`，也可用 `--output` 指定新目录；拒绝覆盖已有导出，不修改源 JSONL、checkpoint 或原 TensorBoard 日志。按优化步恢复已有标量；遇到续训步数回退时剔除旧分支的后续记录。JSONL 没有保存的图像、学习率或真实历史时间不能凭空补出。若训练进程正在追加文件，使用完整 JSONL 快照或在其结束后转换，截断的末行会报错。

`--resume` 恢复严格匹配的模型、优化器和采样位置。续训指纹检查解析继承后的训练配置，仅允许变更 `optimizer_steps`、`checkpoint_interval` 和 `num_workers`；学习率、batch size、梯度累积、冻结设置、seed、物体辅助模式和动作设置等其余配置必须一致，replay 契约、role cache 和卡数也必须一致。已达到总步数时需先增加 `optimizer_steps`。GPU RNG 尚未恢复，不承诺逐位复现。

`train_visualization` 是独立的诊断设置，允许续训时调整。scalar 日志后端/刷新间隔仍由命令行控制。

epoch 配置还允许变更 `epochs`、`max_optimizer_steps` 和 `checkpoint_every_epochs` 来延长预算或调整保存周期；`train_iter`、`global_batch_size`、模型/冻结/优化设置及卡数等必须保持一致。续训从 checkpoint 后的下一优化步继续，不重新开始整轮；若同时配置 epoch 与 step 预算会被拒绝，epoch 预设中的 `optimizer_steps` 和 `checkpoint_interval` 必须为 `null`。

`--init-checkpoint` 仅迁移权重，不恢复优化器或采样位置；动作配置变化时重新初始化回归头，需重新训练，其他兼容性限制见 [动作模式指南](../../docs/guides/oht-action-modes.md)。

## 5. 内部 predicted assistance：首选路线

[role_queries.yaml](configs/role_queries.yaml) 复用现有两角色预测器：由当前渲染特征与任务上下文预测 Target/Reference，在 coarse/refine 间继承几何与角色 token。教师监督角色图、Reference NULL 及可见表面上的 T/R 分离，推理时不输入 GT 对象。

`role_loss.diversity: 0.01` 控制 coarse 分离项，`role_loss.refine_diversity: 0.01` 控制启用 `object.refine_role_masks: true` 后的局部分离项（该 head 默认关闭）。在可靠 Reference 表面抑制 Target 响应，反向同理；仅监督当前渲染可见、几何一致且角色 footprint 不重叠的点，不把 NULL、遮挡或未知区域当负例。TensorBoard 中分别查看 `object_slot_diversity_loss` / `object_refine_diversity_loss` 和对应的 `*_diversity_supervised_pixels`；像素数为 0 时 loss 为 0 属正常跳过。角色允许共享同一对象不同区域时应关闭分离项或明确互斥标注，具体边界见[分离监督说明](../../docs/guides/object-conditioning.md#tr-分离监督与日志)。

已有有效 replay/teacher cache 无需重新生成；使用新配置在新目录训练，既有 checkpoint 的预测不会因新增 loss 自动改变。这里仍是单关键点 `role_queries` 路径，V4 chunk 训练暂不接收 role cache。

每个 replay sample 的教师需要显式标注。annotations.jsonl 的一行示例：

~~~json
{"id":"assemble_left/000000/000000","target":{"present":true,"known":true,"source":"visible_surface","mask_path":"masks/assemble_left-000000-000000.npz"},"reference":{"present":false,"known":true,"source":"none"}}
~~~

Reference=NULL **仅示范格式**，实际任务必须按明确的角色定义填写。NPZ 以相机名为 key，值是与缓存同尺寸的 bool mask；取当前可见表面 world XYZ 后重采样。原尺寸 mask 应按缓存相同的整数步长采样。

可用 site_region 标注接收位置：提供 world_from_site 4×4 和 size 三维正长度，生成带类型声明的 site 区域点。这是特权训练标注，不能把物体中心伪装成表面，也不能当作在线预测结果。

| 语义状态 | present | known | source |
| --- | --- | --- | --- |
| 已知语义缺失 | false | true | none |
| 存在但遮挡/几何不可用 | true | true | unknown |
| 角色分配未确认 | 显式填写 | false | unknown |

未知和无可见几何不产生伪表面监督。教师构建器不会从 objects_pos/held_obj/instruction_id 自动猜测语义，也未实现原始 mesh 或仿真 segmentation 的批量导出；需从已验证的标注/采集端提供。

~~~bash
python tools/build_oht_role_teacher.py \
  --replay /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3 --annotations /common-data-32t/data/robot_data/oht_curobo_pd_v423/annotations.jsonl \
  --output /common-data-32t/data/robot_data/oht_curobo_pd_v423/teachers-v1 --point-count 512 \
  --visualize-every 100

python tools/validate_oht_replay.py \
  --replay /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3 --mode role_queries \
  --role-cache /common-data-32t/data/robot_data/oht_curobo_pd_v423/teachers-v1

python -m finetune.OHT.train \
  --config finetune/OHT/configs/role_queries.yaml \
  --replay /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3 --role-cache /common-data-32t/data/robot_data/oht_curobo_pd_v423/teachers-v1 \
  --output /common-data-32t/data/robot_data/oht_curobo_pd_v423/runs/role-queries-s0 \
  --init-checkpoint /common-data-32t/data/robot_data/oht_curobo_pd_v423/runs/baseline-s0/model_last.pth
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
  --replay /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3 --predictor my_oht_predictor:create \
  --provenance /common-data-32t/data/robot_data/oht_curobo_pd_v423/predictor-provenance.yaml \
  --output /common-data-32t/data/robot_data/oht_curobo_pd_v423/predictions-v1

python -m finetune.OHT.train \
  --config finetune/OHT/configs/predicted_external.yaml \
  --replay /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3 --role-cache /common-data-32t/data/robot_data/oht_curobo_pd_v423/predictions-v1 \
  --output /common-data-32t/data/robot_data/oht_curobo_pd_v423/runs/predicted-external-s0 \
  --init-checkpoint /common-data-32t/data/robot_data/oht_curobo_pd_v423/runs/baseline-s0/model_last.pth
~~~

训练缓存须覆盖训练样本；验证时各模式检查对应集合的缓存覆盖。离线预测按缓存采样帧顺序、episode 开始时 reset，在线按控制采样频率调用；有历史状态的预测器必须对齐两者采样策略，否则应先用无状态预测器。在线服务要求 factory/provenance 与训练缓存完全匹配。

## 7. 推理服务与 IsaacLab 接入

**GT 轨迹回放。** `tools/oht_bridgevla_client.py --mode replay --replay-root <原始OHT根目录>` 可在 IsaacLab 中回放原始 Parquet 的逐帧实测 TCP 与夹爪，复用在线 EEF 执行器，不需要 VLA server 或 checkpoint。默认恢复起始机械臂/夹爪状态，完整回放所选 episode；物体可用显式映射恢复。运行命令见[OHT GT 轨迹回放](../../docs/guides/oht-vla-server.md#oht-gt-轨迹回放)，参数与输出见[回放说明](../../docs/guides/oht-replay.md)。

仿真客户端启动脚本为 [tools/oht_bridgevla_client.py](../../tools/oht_bridgevla_client.py)。正常运行会启动 IsaacLab 仿真、连接 VLA 服务并执行策略动作，完整命令见[独立 BridgeVLA 客户端](../../docs/guides/oht-vla-server.md#独立-bridgevla-客户端)。对齐参考 T1V4 的 EEF 示例显式使用 `--scene-preset oht-v4 --ee-pose-source controller`；若 checkpoint 的采集记录为裸 body，需按其数据定义改选 `body`。

**FastAPI VLA server。** 新入口提供 `/health`、`/model_info`、`/reset`、`/predict`，复用 checkpoint 绑定的 Policy，支持 prepared 观测、串行推理、请求日志和 replay 烟测：

~~~bash
python -m pip install -r finetune/OHT/requirements-server.txt
bash finetune/OHT/serve.sh \
  --model-path /common-data-32t/data/robot_data/oht_curobo_pd_v423/runs/baseline-s0 \
  --host 0.0.0.0 --port 8010 \
  --output-dir logs/oht_vla_server_8010

python tools/smoke_oht_vla_server.py \
  --server http://127.0.0.1:8010 \
  --replay /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3 --split val --index 0
~~~

`--model-path` 可为 checkpoint 或含 `model_last.pth` 的 run 目录。`serve.sh` 调用 [vla_server.sh](vla_server.sh)，也支持参考 H-VLA2 shell 的环境变量方式：`GPU_ID=1 OHT_MODEL_PATH=/path/to/model_last.pth OHT_PORT=8004 bash finetune/OHT/vla_server.sh`。CLI 参数优先，默认不强制 HF 离线；基础模型/processor 资源仍需由 BridgeVLA 加载器读取。

烟测工具仅请求模型，不执行动作；请在没有其他 client 使用时运行。该入口参考 H-VLA2 服务组织方式，但不逐字段兼容其 `json_numpy/proprio` 请求。完整环境变量、观测契约、reset、错误处理、预热和日志见 [VLA server 运行指南](../../docs/guides/oht-vla-server.md)，实现背景见 [设计文档](../../docs/design/oht-vla-server.md)。CPU/HTTP 集成已验证，真实 CUDA/IsaacLab 仍需服务器验收。

**原有 `/act` 服务。** 以下入口和客户端保留原协议，与新 FastAPI 服务是独立进程；同时运行时使用不同端口：

~~~bash
python -m finetune.OHT.server \
  --checkpoint /common-data-32t/data/robot_data/oht_curobo_pd_v423/runs/baseline-s0/model_last.pth \
  --host 127.0.0.1 --port 8010
~~~

上例启动默认 `eef` baseline。其他动作模式使用各自训练的 checkpoint；服务从 checkpoint 自动读取动作设置，不能在启动时将已有 EEF 模型改为 joint。role queries 使用同一命令；external checkpoint 额外提供 `--predictor`、`--predictor-provenance`。`GET /health` 返回 `ready`、contract hash 和 `action_spec`。

`eef` 使用 `bridgevla_oht_absolute_tcp_v1`；其他三种模式使用 `bridgevla_oht_action_v2`，请求携带 `action_spec`。客户端的动作设置必须与 checkpoint 一致，包括坐标系、关节顺序等；不匹配会被拒绝。两种协议均与 H-VLA 现有协议不同。

| 动作模式 | 服务返回的命令 | 额外的当前测量输入 |
| --- | --- | --- |
| `eef` | `[x,y,z,qx,qy,qz,qw,gripper_open]`，world/TCP/xyzw | 无 |
| `relative_eef` | `[dx,dy,dz,rx,ry,rz,gripper_open]`，位移 m、旋转向量 rad，坐标系由配置决定 | `tcp_pose`: float32 `[7]`，world/TCP/xyzw |
| `joint` | `[q1,…,q6,gripper_open]`，关节位置 rad | `joint_positions`: float32 `[6]` |
| `relative_joint` | `[dq1,…,dq6,gripper_open]`，关节增量 rad | `joint_positions`: float32 `[6]` |

夹爪为 `[0,1]` 开度。请求含 episode、step、测量时间戳、固定 goal、contract hash、typed tensors。同一 episode 从 step 0 开始严格递增，时间戳递增；新 episode 清除策略和感知状态。超时不自动重试同一控制请求。一个 server 顺序服务一个 client。

仿真适配器通过现有函数构造与训练一致的观测。下面默认配置与上面的 EEF server 匹配；使用其他模式时，配置必须与所启动 checkpoint 的训练动作设置一致（自定义设置也需保留）：

~~~python
import numpy as np
from finetune.OHT.config import load
from finetune.OHT.action_modes import resolve_action
from finetune.OHT.data.actions import low_dim, gripper_step
from finetune.OHT.data.observation import camera_observation
from finetune.OHT.runtime.executor import execution_target
from finetune.OHT.runtime.transport import Client

# Initialize once, using the training config matching the server checkpoint.
options = resolve_action(load("finetune/OHT/configs/baseline.yaml"))
client = Client(
    "http://127.0.0.1:8010", contract["sha256"], action_config=options
)
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
if options["mode"] in ("joint", "relative_joint"):
    # Measured radians in the same six-joint order as training.
    observation["joint_positions"] = np.asarray(measured_joint_positions, dtype=np.float32)
elif options["mode"] == "relative_eef":
    # Already transformed to the training TCP reference point; world/xyzw.
    observation["tcp_pose"] = np.asarray(measured_tcp_world_xyzw, dtype=np.float32)
command = client.act(observation, goal, episode_id, control_step, simulation_timestamp)
absolute_target = execution_target(
    command, options,
    current_tcp=observation.get("tcp_pose"),
    current_joints=observation.get("joint_positions"),
)
~~~

上例的 `sensor_pose_world_wxyz` 是与 Parquet 相同的 USD camera→world 原始 wxyz 格式，需匹配 checkpoint 的 `data_config.camera_quaternion_order`。若在线 SDK 返回 xyzw，应在调用 `camera_observation()` 前显式重排为 wxyz；已生成的 optical→world 缓存矩阵不能再次做这一步。该相机四元数约定与 `tcp_pose` 的 xyzw 约定不同。

使用 contract 中已解析的 K/端点，不直接把含 null 的原始 YAML 交给 `camera_observation()`。`metric_depth[name]` 必须已按同一深度契约解码成米，不能再次解量化。low_dim 的两个开度特征为 `open01 × 0.04` 的兼容量，不是已标定的真实两指米制距离；逐帧二值夹爪采用与训练相同的因果规则，并在 episode reset 清空状态。

`tcp_pose` 必须是训练使用的 TCP 位姿，不能把未经 `link_to_tcp` 转换的 EE link 位姿直接填入，也不能重复应用该偏移。关节测量必须匹配训练的六关节顺序。所有输入均来自当前观测，不包含未来动作、教师或仿真 GT 对象状态。

`execution_target()` 将相对命令按本次请求的测量值转换一次，得到绝对 TCP 目标（8 维）或绝对关节目标（7 维）；绝对模式原样返回。执行器跟踪该目标时不能重复累加旧 delta。适配器也可直接消费对应模式的原生命令。`runtime/executor.py` 中另一个函数 `relative_eef()` 用于将绝对 TCP 目标转换为有显式限幅的 base/body 控制误差，与模型的相对动作解码用途不同。真实 client 的尺度、积分周期、axis-angle/Euler 约定、夹爪转换、目标到达策略和在线 IK 必须由 IsaacLab 端核对实现；这些函数不是完整控制器。

没有复制 H-VLA server、猜测两个端口的模型分工或修改远端脚本。保留用户提供的 Task1/2 脚本名，尤其 Task1 的 client 名中仍含 task2；真实映射和 joint+relative_eef 路由待远程核查。

## 8. 离线与闭环评估

在线 policy 现在默认采用未来关键点执行：固定目标到位后开闭夹爪、确认稳定后再预测，超时/停滞即停止。逐帧 GT replay 保留原始时序；新增 GT keypoints 与记录初态 policy 对照，命令见[关键点执行说明](../../docs/guides/oht-waypoint-execution.md)。直接 OSC 尚不提供碰撞路径规划。

**在线预测可视化。** VLA server 默认每 10 步导出同一次请求的诊断，运行客户端时打开 `http://109.105.46.126:8004/visualization`。页面支持旋转/缩放/平移 3D 彩色点云，显示相机与实际动作/热图 waypoint；各原始 sensor 的 RGB、深度图叠加 waypoint 透视投影，同时保留 coarse/refine heatmap。raw 请求保留原始分辨率，prepared 请求明确标记为已处理观测。带窗口的 IsaacLab client 默认显示实测 TCP、EEF 目标与轨迹；采集前清除标记，避免进入策略图像。开关、下载、保留数量和运行方式见 [在线可视化](../../docs/guides/oht-vla-server.md#在线可视化)。在线诊断不含 GT；下方离线评估可用于 GT 对比。

实时 IsaacLab 对接使用独立客户端 `tools/oht_bridgevla_client.py`，可从 IsaacLab 根目录直接启动，也可复制启动文件后用 `--bridgevla-root` 指定仓库。底层入口 `python -m finetune.OHT.isaaclab_client` 默认使用自带的原生工厂，无需参考脚本或自定义 runtime 模板；支持 `--isaaclab-root`、注册环境/显式配置类及已有 sensor/USD 相机绑定。服务端按 checkpoint 准备 RGB-D 观测，客户端支持 EEF/relative EEF 的 OSC 跟踪，以及 joint/relative_joint 的关节位置执行。四个任务使用本仓库 instruction，可分别配置初始化和成功条件；默认 `--score none` 不输出成功率，显式 `--score task --task-config ...` 才进行任务评分。运行命令、任务配置、raw 字段及控制器要求见 [IsaacLab 实时 RGB-D 客户端](../../docs/guides/oht-vla-server.md#isaaclab-实时-rgb-d-客户端)。

~~~bash
python -m finetune.OHT.eval open-loop \
  --checkpoint /common-data-32t/data/robot_data/oht_curobo_pd_v423/runs/baseline-s0/model_last.pth \
  --replay /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3 --split val \
  --output /common-data-32t/data/robot_data/oht_curobo_pd_v423/eval/baseline-val.json
~~~

离线评估自动读取 checkpoint 的动作模式：EEF 类逐任务报告位置/旋转误差（相对 EEF 先还原绝对位姿），关节类报告 rad 单位 MAE/RMSE；所有模式报告夹爪准确率和模型耗时。这些不能代替闭环成功率。内部角色模式没有教师输入，即使 evaluation 数据行存在标签也会被策略白名单过滤。

离线评估默认向 stderr 输出阶段日志，并每 10 秒更新当前阶段、已完成样本数、速度和预计剩余时间（完成首个样本后才有速度估计）。`--progress-interval 5` 可改为 5 秒，`--quiet` 关闭这些提示；最终摘要仍输出到 stdout，逐样本指标保存在 JSON。诊断时先在上述命令中加 `--limit 8`，并使用新的输出名，例如 `baseline-val-smoke.json`；这是前 8 个样本的检查，不能代表完整验证集或覆盖所有任务。确认后去掉 `--limit` 运行正式评估。

若长时间停在 `Cameras: [...]`，请查看其后 `[OHT eval]` 的 `stage`：`build_agent` 是模型构建，`apply_checkpoint` 是应用 OHT 权重，`prepare_replay` 是读取 replay 索引，`read_sample` 是校验/解压当前 NPZ，`predict` 是当前样本推理。首次 EEF 推理还会建立三维热图反投影缓存，可能比后续样本慢；日志中的心跳只表示评估进程仍可报告状态，不证明 GPU 运算正在推进。结合另一终端的 `nvidia-smi` 检查利用率、显存和其他进程。`Loading original paligemma model` 是构造基础网络时的提示，后续成功出现 `model_ready` 才表示 OHT checkpoint 已应用。`hash_checkpoint` 是结果写出前计算权重文件 SHA256，文件较大时也可能耗时。评估期间请保持 checkpoint 文件不变，训练仍在更新 `model_last.pth` 时应选用已保存的固定步数 checkpoint。

**离线预测可视化。** CLI 默认开启，每 50 个样本导出一次（样本索引 `0,50,100,...`）；`--visualize-every N` 调整间隔，`--visualize-every 1` 导出每个样本，`--visualize-every 0` 关闭。不改变评估样本数量。先检查 8 个样本时可用：

~~~bash
python -u -m finetune.OHT.eval open-loop \
  --checkpoint finetune/OHT/runs/baseline-eef-8gpu-s0/model_last.pth \
  --replay /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3 --split val --limit 8 \
  --visualize-every 1 \
  --visualize-output-dir /common-data-32t/data/robot_data/oht_curobo_pd_v423/eval/baseline-val-images \
  --output /common-data-32t/data/robot_data/oht_curobo_pd_v423/eval/baseline-val-with-images.json
~~~

报告和图片目录均要求使用新路径。不指定图片目录时，默认保存到报告路径去掉扩展名再加 `_visualizations/`。完整验证可去掉 `--limit`，省略 `--visualize-every` 使用默认间隔 50；固定间隔不保证覆盖每个子任务。评估结束后打开图片目录的 `index.html`，各 `sample_000000/` 子目录包含：

- `mvt1.png` / `mvt2.png`：同一次实际推理的 coarse/refine 三视图 RGB、GT 点高斯图、预测热图、GT/动作 TCP/热图 TCP 投影对比；模型提供预测侧 Target/Reference prior 或 relation anchor 时增加相应列。单阶段模型只输出 `mvt1`。
- `mvt1.npz` / `mvt2.npz`：RGB、未经显示归一化的逐视图热图概率、world→pixel 仿射矩阵、GT/预测像素坐标及可用的 prior。PNG 颜色按每视图 min/max 拉伸，不能据颜色比较不同样本置信度。GT 图是用于观察的 `sigma=2px` 高斯，不是重新计算的训练 loss 标签；GT 超出当前 refine 视野时不裁到边缘，记录 `gt_in_view=false`。
- `prediction.json`：样本 ID、任务、instruction、动作配置、预测/GT 动作、世界坐标 TCP 和误差。绿色圈为 GT，红色叉为动作 TCP，青色叉为热图解码 TCP；relative EEF 先按配置还原到世界坐标。非 EEF 模式的热图来自辅助 EEF 头，图中标明 auxiliary；关节模式不伪造 FK 位置，因此没有红色动作 TCP 标记。

导出只采集本次前向结果，不额外调用模型；GT 在推理完成后投影绘图，不进入策略输入。抽样步的 `latency_seconds` 包含诊断载荷的 CPU 拷贝和投影采集，PNG/NPZ 写盘在推理计时之外；测速时用 `--visualize-every 0` 关闭可视化。本接口用于离线评估，独立于 `train_visualization`，不新增 TensorBoard events 或 HTTP 响应图片。底层调用可用 `Policy.act(..., visualization_callback=callback)` 接收 label-free CPU NumPy 载荷；`open_loop(..., visualizer=EvaluationVisualizer(path, every=N))` 接入相同导出器。

闭环工具通过 `--environment module:factory` 接入本机 IsaacLab 适配器。`factory(contract)` 收到的运行时契约包含 `action_spec`，返回：

- reset(case) → {observation, goal, timestamp}，按固定 seed 重置并恢复执行器状态。
- step(command) → 下一 {observation, goal, timestamp, done, success}；输入是上表对应模式的原生命令，EEF 为 8 维，其余为 7 维，工具不自动解码相对动作。终止时必须显式给出 bool success。
- 可选 close() 释放环境。

适配器根据 `contract["action_spec"]` 解释命令；相对模式可调用 `execution_target()`，参考状态必须与该次请求中的测量一致。`reset` 和 `step` 返回的 observation 需包含上表要求的当前 TCP/关节字段。适配器拥有实际 EEF / joint 执行逻辑；GT 仅用于 reset 和 success 评分，不能放入策略/感知输入。先用专家重建目标完成执行器校验，再比较模型。

下面与第 7 节的默认 EEF server 匹配，显式指定训练配置：

~~~bash
python -m finetune.OHT.eval closed-loop \
  --config finetune/OHT/configs/baseline.yaml \
  --server http://127.0.0.1:8010 --replay /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3 \
  --environment my_isaac_oht:create --cases /common-data-32t/data/robot_data/oht_curobo_pd_v423/eval-cases.jsonl \
  --executor-id verified-eef-executor-v1 --max-steps 500 \
  --output /common-data-32t/data/robot_data/oht_curobo_pd_v423/eval/baseline-eef-s0.json
~~~

评估其他模式时，同步更换 server checkpoint、`--config`、已验证的执行器 ID 和输出路径；例如 joint 模式使用 `joint-s0/model_last.pth` 与 `configs/joint.yaml`，完整命令见 [动作模式指南](../../docs/guides/oht-action-modes.md)。省略 `--config` 会采用旧 EEF 默认值，不会自动从远端 server 选择模式。

case JSONL 每行含唯一 id、task、seed；id 每次运行需使用新 episode 标识或重启 server。所有方法用同一份 cases、执行器版本和时间预算，分别报告 B-E/P-E，必要时 B-H/P-H。异常、超时和未完成 episode 均保留在成功率分母。报告记录 cases hash、契约、`action_spec` 和执行器 ID，便于检查对照条件。

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
