# OHT 数据诊断：相机、深度与点云

正常构建和训练从 [OHT README](../../finetune/OHT/README.md) 开始。本页用于核对标定、深度编码、ROI 和预览工具；所有命令从 BridgeVLA 根目录执行。

- [相机、深度与工作区契约](#相机深度与工作区契约)
- [生成时可视化](#生成时可视化)
- [单帧几何排错](#单帧几何排错可选)
- [构建未完成时](#构建未完成时)

## 相机、深度与工作区契约

以用户提供的已运行 `convert_oht_lerobot_v21_to_tfds.py → transforms.py → dataset.py` 为当前读取参考，无需重新标定或采集。使用 [dataset.yaml](../../finetune/OHT/configs/dataset.yaml)；实际参数及 metadata SHA256 写入 contract 的 `source_data_configs`，各任务分别读取，不串用左右相机参数。

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

`depth.path_pattern: null` 通过 Parquet 引用读取原生灰度视频，不经过 RGB；按本页 `video_alignment` 配对帧。补充的 `pointcloud_transforms.py` 明确使用 `Z=depth_m`、`X=(u-cx)*Z/fx`、`Y=(v-cy)*Z/fy`，因此默认 `depth.kind: z`，不归一化射线。将 Z-depth 按此前的 `ray` 处理会随像素离主点的距离拉近表面，造成跨相机融合变形。只有独立确认来源为相机到点的欧氏距离时才显式使用 `ray`。

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

## 视频与帧对齐

- 仓库默认 `video_alignment: frame_index`：与参考 `_read_video_selected_rgb()` / `_read_depth_video_selected()` 一样，原始 Parquet 第 i 行严格配对 RGB 第 i 个解码帧、depth 第 i 个解码帧和第 i 行相机外参；不是抽样后的第 i 帧，也不从 FPS 推算。逐流核对解码总数等于 Parquet 行数（含未抽样尾段），缺帧/多帧即失败、不写 complete。可避免时间戳引用偏移时旋转相机的视频帧与姿态错配，但不能修复导出端已经错配的 pose。
- 独立时间戳流可显式设 `video_alignment: timestamp`，按引用选择最近 PTS，超出 `video_timestamp_tolerance` 报错；旧配置缺少 alignment 字段仍保留此行为。切换对齐模式需新建 replay/teacher，修改 YAML 不会修复旧 XYZ；原始数据/audit 可复用。


## 生成时可视化

`--visualize-every N` 按每个 episode **生成的样本数**保存 PNG，包含第一个样本，随后每隔 N 个；不是按原始视频帧计数。`1` 显示每个生成样本，默认 `0` 关闭。可省略 `--visualize-output-dir`，此时保存到 `<output>/preview/<task>/<episode六位>/<frame六位>/preview.png`。预览文件存在时拒绝覆盖。

每张图包含缓存分辨率的各相机 RGB、米制 depth，以及两排 XY/XZ/YZ 彩色点云：青色为当前 `EE-pos`，品红色为下一 GT 关键点。当前 OHT 配置的 `link_to_tcp` 为单位阵，青色位置就是原始 `observation.ee_pos_world`；PNG/HTML 均显示 `EE-pos`，缓存字段仍为 `current_tcp`。若配置了额外 TCP 偏移，显示名称保留 `TCP`，不把偏移后的位置标成原始 EE。depth 蓝色近、红色远，标注当前有效范围（米），黑色表示无效；不同图的深度颜色范围可能不同。

每个 RGB 面板下方显示 TCP/goal 的 `in view`、`outside image`、`behind camera` 或 `unavailable`，以及缓存像素坐标/光学 Z。画外点不强制移到边缘；`in view` 仅代表在视锥内，不代表没有被遮挡。

- 全局三视图使用 `scene_bounds`，最多显示 200,000 个点（默认五相机 120×160 的全部有效点都在预算内）。全局/局部点云均以单像素绘制，不扩张相邻像素，重叠点按深度处理；TCP/goal 标记大小不变，品红色方框标出局部立方体的投影范围。
- 局部三视图以 GT keypoint 为中心，各轴 ±0.20 m，显示米制坐标范围。先从完整有效点云选择局部点，再独立限制显示点数，避免全局抽样漏掉小物体；没有观测点时明确提示，不补造几何。

局部图标注 **GT-centered refine diagnostic (NOT model stage2)**：它不是模型 coarse 预测或带噪训练中心产生的二阶段视图。模型真实 coarse/refine renderer 图应在训练/推理前向中另行导出。两排都显示抽样前后点数；缓存的 4 倍步长采样不因预览变密而恢复到原始分辨率。

PNG 单独输出，不新增 observation/label 字段，也不改变 buffer contract 或训练样本。无需模型、CUDA 或图形桌面；仅使用数据环境已有的 NumPy/Pillow。基础 buffer 没有 T/R 标注，分割角色预览见[数据准备与标注](oht-data-preparation.md#分割角色标注)。

### 生成 replay 时同步提取并预览对象点云

在[单关键点构建](oht-single-keypoint.md#单关键点数据)或 V4 构建命令中增加 `--object-cloud-config finetune/OHT/configs/object_clouds.yaml`，即可对每个生成的 replay 样本按当前帧对象中心同步导出近似点云，不按未来 goal 帧截取。

对象配置与相机/训练配置分开，示例见 [`configs/object_clouds.yaml`](../../finetune/OHT/configs/object_clouds.yaml)。`objects: null` 表示全部记录对象；初始 `default_size: [0.20, 0.20, 0.20]` 为 world XYZ 全长（米）的可调整猜测，不是物体尺寸 GT。需要不同尺寸或只提取部分对象时，复制该 YAML 并改为例如：

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

### HTML 统一查看、角度与帧选择

V4 开启非零 `--visualize-every` 即生成基础 HTML；单关键点构建需同时启用 `--object-cloud-config`。对象点云图层由该对象配置开启。打开 `<replay>/preview/index.html`，可选择 episode、实际帧号，或用上一帧/下一帧、滑条切换。列表只包含已生成 HTML 的样本；`--visualize-every 1` 覆盖每个 replay 样本，不能显示未采样进 replay 的原视频帧。复制到本机时保留整个 `preview` 目录及其相对路径，可离线双击入口。入口页先显示当前帧 PNG，再显示交互点云。

顶部提供 3D/XY/XZ/YZ 和适配当前帧按钮；点云支持左键旋转、右键平移、滚轮缩放，并显示水平/俯仰角度。默认切帧保留旋转、缩放、平移和相机/显示开关；取消“切帧保留视角”后使用新帧默认视图。同一 episode 内保留对象勾选、名称、尺寸框以及 ROI 编辑，跨 episode 独立保存对象/ROI 设置。对象中心、姿态、TCP、GT 和下载配置的帧/校验值始终取当前帧，切帧不会套用旧帧位姿。设置只保留在当前打开页面中，刷新重置；要留存尺寸/ROI，可下载对应 JSON/YAML。

HTML 中对象与场景共用一张点云画布，“对象框成员”颜色按索引区分，重叠框为白色；可切回 RGB/相机颜色、调整背景透明度或只显示对象框内点。对象列表可勾选全部记录对象的中心、名称和姿态轴；轴需选择正确四元数顺序。

带对象配置的 HTML 默认勾选“仅显示勾选对象框内点云”，隐藏框外背景。统一入口顶部和单帧页面均提供“恢复完整场景”，点击后显示所有相机的场景抽样点、恢复点透明度并缩放到场景范围，保留对象选择、名称、尺寸、姿态轴和 ROI 配置；重新勾选“仅显示勾选对象框内点云”可返回物体视图。完整场景仍受所选点源和 HTML 抽样预算限制；缓存 XYZ 已被裁掉的点需改用 depth 源重新导出预览。

上方 PNG 是生成时配置对应的静态预览；下方勾选对象和调整尺寸仅更新交互点云，不重写 PNG。顶部“查看下方交互点云”可直接跳到交互区。

如果修改代码后看起来没变化，先检查生成命令是否带 `--object-cloud-config`，以及打开的是否为本次新输出。旧 PNG/HTML 不会随代码更新自动改变；旧 replay 未记录对象配置时，下面的独立导出命令需显式增加对象 YAML。

仅更新可视化、无需重算 replay 时，可为已有完整 replay 导出新的 PNG/HTML。已在构建时生成所需预览则不必重复运行：

~~~bash
python tools/export_oht_replay_viewer.py \
  --replay /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3 \
  --output /common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3/preview/refresh \
  --object-cloud-config finetune/OHT/configs/object_clouds.yaml \
  --every 1 --max-points 40000
~~~

打开 `/common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3/preview/refresh/index.html`。输出目录必须尚不存在，每帧 PNG/HTML 保存到 `<output>/<task>/<episode>/<frame>/`。`--every N` 按各 episode 的 replay 样本数选帧，包含首个；浏览器每次仅加载一帧。默认源为缓存米制 depth/K/外参，可用 `--source xyz` 改为缓存 XYZ。导出校验已完成 replay 和观测 NPZ，不改写缓存，也不额外导出 PLY/NPZ。

### 单帧几何排错（可选）

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

### 交互确定点云范围

构建生成的 HTML 或上述单帧 `point-cloud-roi.html` 均可用于选择范围。左键旋转、右键/Shift 拖动平移、滚轮缩放；支持 XY/XZ/YZ、相机开关、RGB/相机着色、点大小及点选世界坐标，离线使用无需启动服务。

选择并修改 `scene_bounds`、`keep_bounds` 或排除框的六个世界坐标，实时查看保留点及框；可添加/删除排除框，复制或下载 YAML 片段，再人工合并进实际 `--config` 文件。上下界与 `point_cloud_filter` 一致：下界包含、上界不包含；非法框不更新且禁止导出。`scene_bounds` 是策略工作区，不只是显示裁剪；它影响动作归一化及数据/checkpoint 契约，不能随意覆盖已有模型的配置。相机开关、点大小和显示抽样不写入训练配置。

`--html-source depth` 推荐用于确定范围：直接将缓存的**米制 depth + K + optical→world 外参**重新反投影，仅用于显示，不再解码深度、不套旧 ROI，也不写回缓存。它能查看被旧 ROI 删除但深度仍有效的点；PNG/geometry.json 仍描述原缓存 XYZ。默认 `--html-source xyz` 则只显示缓存剩余有限 XYZ，不额外裁剪，也无法恢复其 NaN 点。页面明确标出来源。

**限制**：两种来源都受缓存分辨率、有效深度范围和原始内外参/同步正确性限制，不能补造未观测表面，不能修复历史错误深度编码。扩大范围不必为了预览先重建 buffer；确定新过滤/工作区配置后，再重建受影响的训练缓存，不能只改旧 contract。默认最多显示 40,000 点，可调至 60,000；保留计数仅对应显示抽样，不是全量覆盖证明。工具不配准点云、不修改 buffer；TCP/未来 GT goal 仅作诊断，并提示超出当前框。至少检查四任务、多时刻、最高抬升与操作夹具，再确定统一范围。

### 显示任务相关对象

新构建的 replay 在 `samples.jsonl` 的 `diagnostic_objects` 中保存**当前观测帧**记录的全部对象中心与原始四元数及源 Parquet 路径/哈希，不增加 observation NPZ 或训练 batch 字段。对象数量按 `objects_pos` 的每 3 个位置值与 `objects_quat` 的每 4 个四元数值匹配，不固定为四个；两者数量必须一致，同一 episode 内保持一致。HTML 默认勾选全部对象，按索引列出 `object_0…object_{N-1}`，提供“全选对象”“清空对象选择”和每项复选框；仅显示勾选对象的中心、名称及已启用的姿态轴。还可修改显示名称、缩放到所选对象、调淡背景点，并点击中心查看 world XYZ 和原始四元数。中心在当前框外仍显示并提示。清单只包含数据中记录了位姿的对象，不自动发现未记录的场景物体。对象选择、名称和背景透明度只作用于页面，不改变点云保留计数、ROI YAML 或缓存；改名仅在当前页面有效。

参考代码没有确认对象索引与语义名称的对应关系，也没有确认 `objects_quat` 的顺序，因此不自动猜测车轮/托盘或 Target/Reference。姿态轴默认隐藏；核实导出端约定后，在页面的“对象四元数顺序”中独立选择 `wxyz` 或 `xyzw`（不继承 camera/EE 设置）。轴为对象局部 x 红/y 绿/z 蓝，长度可调，坐标直接使用 world 位姿，不套相机 optical 翻轴。对象中心不是表面点或轴口 site；实例 mask 和角色点使用[物体辅助的教师接口](oht-single-keypoint.md#内部角色预测)。

仅旧 replay 缺少对象元数据时，导出工具可同时增加 `--object-root /common-data-32t/data/robot_data/oht_curobo_pd_v423` 和 `--object-manifest /common-data-32t/data/robot_data/oht_curobo_pd_v423/audit-seed0.json`，清单必须是构建该 replay 时使用的 audit。单帧诊断还需启用 `--html`。读取器核对 audit、Parquet 哈希及当前帧，绑定不符则拒绝输出；不改写缓存，也不猜测缺失位姿。新重建流程无需这些补充参数。

### 没有实例 mask 时近似提取 object 点云

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

### 构建未完成时

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
