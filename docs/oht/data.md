[文档索引](../README.md) · [训练与评估](training.md) · [服务与执行](runtime.md)

# OHT 数据、标注与几何诊断

当前主线是 **原始 RGB-D → V4 实测轨迹缓存 → 训练**。所有命令从 BridgeVLA 仓库根目录执行，替换示例路径后运行；输出目录使用新名字，不覆盖既有数据。

导航：[准备数据](#数据准备) · [契约与划分](#数据契约与划分) · [标注与对象](#标注与对象点云) · [几何诊断](#几何与可视化诊断) · [单关键点](#单关键点数据) · [迁移边界](#迁移边界)。

## 数据准备

仅做数据处理与 CPU 预览时安装：

```bash
python -m pip install -r finetune/OHT/requirements-data.txt
```

统一设置后续路径。`ROOT` 是各任务 `lerobot_dataset` 的父目录；训练读取 `REPLAY`，不是 `ROOT` 或 `OUT`：

```bash
ROOT=/common-data-32t/data/robot_data/oht_curobo_pd_v423-seg
OUT="$ROOT/bridgevla-seg-v4-run01"
REPLAY="$OUT/replay-source-v4"
MANIFEST="$OUT/audit.json"
```

| 已有数据 | 下一步 |
| --- | --- |
| 完整 V4 | 先核对[数据契约](#数据契约与划分)并完整校验，再进入[训练](training.md) |
| 分割原始数据 | 运行下方完整流水线 |
| 普通非分割数据，或需要场景分组 | 修改上述路径，走手动构建 |
| V4 完整，只改角色映射或图片 | [刷新标注](#分割角色标注)或[导出预览](#几何与可视化诊断)，复用 observation/动作 |
| 单关键点或 role teacher 实验 | [单关键点数据](#单关键点数据)，不使用 V4 chunk replay |

### 分割数据完整流水线

```bash
bash finetune/OHT/prepare_seg_data.sh \
  --root "$ROOT" --output "$OUT" \
  --chunk-horizon 16 --action-frame-stride 1
```

依次执行 **audit → V4 replay → 完整校验 → 分割角色标注 → 角色预览**，任一步失败即停止。原始目录无需预先包含 audit、replay 或 contract。`OUT` 必须尚不存在；任务目录 `disassemble_left-seg` 会规范为 `disassemble_left`，只有这一个任务也可运行。新旧同名任务目录不能混用，也不能按 episode 编号给旧轨迹补分割。

| 参数 | 默认值与作用 |
| --- | --- |
| `--config` | [dataset.yaml](../../finetune/OHT/configs/dataset.yaml)，相机/深度/状态/ROI 配置 |
| `--mapping` | [seg_roles.yaml](../../finetune/OHT/configs/seg_roles.yaml)，T/R 语义映射 |
| `--chunk-horizon` / `--action-frame-stride` | 16 / 1；未来动作数量 / 原始帧间隔 |
| `--sample-stride` | 10；只控制输入观测采样 |
| `--visualize-every` | 100；每个 episode 按生成样本数取预览，含首个；0 关闭预览 |
| `--workers` / `--point-count` | 4 个校验线程 / 512 点；point-count 仅用于 V3 teacher |
| `--replay-version` | 4；设为 3 时另生成单关键点 role teacher |
| `--dry-run` | 仅打印命令，不解码、不创建输出 |

脚本在 `OUT` 保存配置快照和 `pipeline.json`，记录完成阶段或失败原因。可用 `PYTHON_BIN=/path/to/python` 指定解释器，或直接调用 `python tools/prepare_oht_seg_data.py`。修改仓库 YAML 不会更新已有快照。

### 手动构建 V4

普通数据可把 `ROOT` 改为 `/common-data-32t/data/robot_data/oht_curobo_pd_v423`，并重新设置对应的新 `OUT/REPLAY/MANIFEST`。已完成完整流水线时不用重复执行：

```bash
python tools/audit_oht_dataset.py \
  --root "$ROOT" --output "$MANIFEST" --seed 0

python tools/build_oht_replay.py \
  --root "$ROOT" --manifest "$MANIFEST" \
  --config finetune/OHT/configs/dataset.yaml \
  --output "$REPLAY" --replay-version 4 \
  --chunk-horizon 16 --action-frame-stride 1 \
  --sample-stride 10 --visualize-every 100

python tools/validate_oht_replay.py --replay "$REPLAY" --workers 4
```

原始数据未变且 audit 有效时复用 `MANIFEST`，跳过第一条命令。audit/build 均拒绝覆盖输出。缺失其他任务的提示可属于单任务输入的正常情况；坏 episode、缺成功标记或明确失败仍需处理 audit 报告。audit 返回非零时完整流水线停止。

完整流水线的主要产物为：

```text
OUT/
  audit.json, pipeline.json, dataset.yaml, seg_roles.yaml
  replay-source-v4/
    contract.json, samples.jsonl, complete.json
    observations/<task>/<episode>/<frame>.npz
    targets/<task>/<episode>/<frame>.npz
    preview/index.html
  seg-role-annotations/
  role-preview/index.html
```

手动路线只生成指定的 audit/replay，不自动复制配置或导出分割角色。

## 数据契约与划分

### 划分与完整性

audit 按任务目标比例 80/10/10 确定性分组，合并 `trajectory_hash` 相同的重复轨迹；实际比例受组大小影响。**不同轨迹是否来自同一场景不会自动识别。** 需要场景隔离时，准备以下 JSON，并给手动 audit 增加 `--groups /path/to/scene-groups.json`：

```json
{"assemble_left/0":"scene-a","disassemble_left/5":"scene-a"}
```

key 是规范任务名与未补零 episode 编号，value 是共享场景组名。同组不跨 train/val/test。完整流水线暂不暴露 `--groups`，因此有此需求时走手动路线，再导出标注。普通 episode 留出不等于未知布局泛化；归一化只从 train 拟合，val 调参，test 固定评估。

训练要求有效的 `contract.json`、样本索引和 `complete.json`，并校验绑定哈希。失败目录没有完成标记，应换新目录重建，不手写 contract/complete 绕过检查。`validate_oht_replay.py` 读取全部非空划分的全部 NPZ；训练的 `--validate-only` 只检查 train，不能替代它。

校验是 CPU/磁盘任务，默认 `--workers 1`；共享盘或内存紧张时用 1–2，更多线程不保证更快。约每 5 秒向 stderr 输出进度与 ETA，`--quiet` 关闭进度，最终 JSON 写 stdout。PNG/HTML 和近似对象点云不属于训练 replay 内容校验。

### V4 时间与实测标签

V4 schema 为 `oht_bridgevla_v4`，从原始 Parquet 和 RGB-D 重新构建，不复制 V3 observation。复用前核对缓存 horizon 不小于模型 horizon（当前 16），`action_dt` / frame stride 符合目标时间尺度，各数据源几何和状态约定正确；仅切换 regression/delta/flow 动作头可复用相同 V4。

| 字段 | 含义 |
| --- | --- |
| `tcp_pose`、`joint_positions`、`gripper_open01` | 当前实测状态，属于 observation |
| `key_action`、`key_valid` | 下一关键目标与监督有效性 |
| `action_chunk[K,8]` | 未来实测 world TCP / xyzw / 二值夹爪 |
| `joint_chunk[K,7]` | 同批未来六关节与夹爪标签；不代表已有 joint chunk 模型 |
| `action_frames`、`action_timestamps`、`action_valid` | 实际未来帧、时刻与有效 mask |

观测与目标独立保存并绑定哈希。未来时刻为 `t+dt … t+K*dt`；`dt` 从原始时间戳推导并写入 contract，也可用 build 的 `--action-dt` 声明期望间隔。时间容差参数为 `--action-timestamp-atol`（默认 `1e-5`）和 `--action-timestamp-rtol`（默认 `1e-4`）。不同采样率、时间缺口或粗采样漏掉完整夹爪事件的窗口会被拒绝/跳过并记录。

标签来自实测 TCP、关节与因果滤波夹爪；**raw action 所有维度只用于审计**，不用原命令、全局缩放或 keypoint 直线插值制造监督。保留真实静止、纯旋转和夹爪变化。无下一 key 的尾段可保留 `key_valid=false`；不足 K 项用 `action_valid` 屏蔽，不重复最后一帧制造停留。

夹爪从 `observation.state[6]` 读取：dataset-global stats/info 的 open=min、close=max，归一化为 open01。反向端点可用 `gripper.source: config` 显式指定，不按 episode 重估。因果规则为 open01>0.9 开、<0.1 闭，中间由 ±0.05 差分切换，否则保持上一状态；模糊首帧默认 `initial_state: 1`。标签 open=1、close=0；接触后未闭到底不等于抓取失败。在线使用同一滤波，并在 episode reset 清状态。

### 相机、坐标与深度

五个物理相机为 `global_left/global_right/local_left/local_right/wrist`。默认输入 `image_size: [120,160]`，对原始 480×640 严格 4 倍步长采样，RGB/depth/XYZ 同步取像素、K 同步缩放；不在深度边缘做 RGB 双线性插值。五个物理相机仍融合为模型的三个虚拟视角，不能把它们直接当作 MVT 的 num_img。

内参逐任务读取：优先 `<task>/lerobot_dataset/meta/camera_intrinsics.json`，不存在才用 `meta/info.json.camera_intrinsics`；支持大小写相机名和 `cameras.<name>.intrinsic`。非法 sidecar 会报错，不猜焦距。显式 YAML K 使用 `intrinsics_source: config`。实际参数及 metadata SHA-256 写入 `source_data_configs`，左右相机不串用。

raw camera/EE 四元数默认 **wxyz**，分别由 `camera_quaternion_order` / `ee_quaternion_order` 控制；参考运行若显式用了 xyzw，必须相应配置，不能从数值猜顺序。策略与标签始终为 **world / TCP / 米 / xyzw**。`ee_pos_world` 已含控制器 TCP 偏移，`link_to_tcp` 保持单位阵，不再加约 12 cm。

raw 相机默认是 USD/OpenGL camera→world。若 `camera_extrinsic_direction: world_to_camera`，先求完整 4×4 的逆，再右乘 `optical_to_sensor`；默认五相机均为：

```text
T_world_optical = T_world_usd @ diag(1,-1,-1,1)
p_optical = Z * inverse(K) @ [u,v,1]
p_world = R_world_optical @ p_optical + t
```

缓存外参为 optical→world；行向量投影用 `(P_world-t) @ R_world_optical`。不能在世界系左乘翻轴或只转置 R。若源本来就是 optical→world，使用 identity optical 变换，不再翻轴。camera 与 EE 顺序独立，改动作旋转不能修复点云。

默认深度来自无损 HEVC `gray12le`，像素格式不决定数值编码。当前 [dataset.yaml](../../finetune/OHT/configs/dataset.yaml) 为：

```yaml
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
```

`metadata: reference` 优先各相机 `features.observation.depth.<camera>.info` 的量化参数，`video.*` key 优先；缺字段使用上面参考默认值，不从 RGB codec 推断。`metadata: true` 要求完整 `depth_min/depth_max/shift/use_log/qmax/pix_fmt`，`false` 只用 YAML。`path_pattern: null` 按 Parquet 引用读取原生灰度视频，不转 RGB；读取器检查像素格式并保留 12-bit 值/PTS。

令 `n=q/qmax`：log 反量化为 `exp(n*(log(max+shift)-log(min+shift))+log(min+shift))-shift`，linear 为 `n*(max-min)+min`。默认 float32 运算后舍入 uint16 毫米再转米；raw=897→**1.215 m**、1500→**2.249 m**。0 无效；4095 是有效 far-plane 码，解为 10 m 后再被 3 m 深度范围过滤，不能直接当无效码。

`kind: z` 使用 `X=(u-cx)*Z/fx`、`Y=(v-cy)*Z/fy`，不归一化射线。只有独立确认源为欧氏距离才用 `ray`。已确认的毫米导出可设 `encoding: scaled_integer`、`metadata: false`、`scale: 0.001`；浮点 NPY metric、整数 PNG/TIFF、显式 channel/scale/offset 的旧 linear_channel 也有兼容入口，不能据此把普通可视化视频当米制深度。

默认 `video_alignment: frame_index` 严格配对原始 Parquet 第 i 行、RGB 第 i 解码帧、depth 第 i 解码帧和该行外参，不按 FPS 推算。各流解码总数须等于 Parquet 行数，包括未抽样尾段；缺帧/多帧失败，不写 complete。独立时间戳流才设 `timestamp`，按最近 PTS 匹配，超出默认 `video_timestamp_tolerance: 0.0084333333` 报错；旧配置缺 alignment 字段仍走 timestamp。

### 工作区与点云过滤

```yaml
scene_bounds: [-0.5, -1.0, 0.3, 1.5, 1.0, 2.0]
point_cloud_filter:
  enabled: true
  keep_bounds: [-0.3, -0.8, 0.4, 1.2, 0.8, 1.9]
  exclude_boxes: []
```

均为 world 米制 `[xmin,ymin,zmin,xmax,ymax,zmax]`，下界包含、上界不包含。scene_bounds 影响动作空间/归一化；keep_bounds 是经验保留区，不是全量实测边界。必须覆盖四任务抓取、放置、最高抬升及夹具/Reference，不能静默裁剪 GT。

`keep_bounds: null` 不额外裁剪，`enabled: false` 关闭过滤，旧配置缺该块时默认关闭；落入任意 exclude_boxes 的点被删除。过滤前保留像素对应关系，排除 XYZ 写 NaN，原 RGB/depth/TCP/动作/presence 不改。缓存、训练 coarse/refine、在线 predictor/agent 前及 CPU 预览共用规则；refine 继承过滤后的点，不会从原始点云恢复表面。单相机可空，全场无工作区点则报错。teacher 无可见几何时保留 present，不改成 NULL。

## 标注与对象点云

### 分割角色标注

当前映射对 `disassemble_left` 全阶段固定 **Target=左轮 ID 1、Reference=放置盘 ID 4**：

| ID | 语义 |
| --- | --- |
| 0 | robot |
| 1 / 2 | 左轮 / 右轮 |
| 3 / 4 | wheel_stack / plate |
| 5 / 6 | 左轴 / 右轴（shaft_1 / shaft_2） |
| 255 | 背景或未标注 |

这是语义分割，不能区分同类别多实例。导出器读取当前实际 `observation.seg.*` 引用，检查流长、时间戳、原始分辨率和灰度 uint8 ID，再按 RGB-D 同整数步长采样，不插值标签。输出 bool mask NPZ、`annotations.jsonl`、绑定视频/映射/replay 的 `provenance.json`。通用 phases 可用当前 `instruction_id`，不读取未来阶段。

修改映射后只刷新标注/预览，复用动作与 observation；下列命令沿用分割数据的 ROOT/OUT/REPLAY/MANIFEST：

```bash
python tools/export_oht_seg_roles.py \
  --root "$ROOT" --manifest "$MANIFEST" --replay "$REPLAY" \
  --mapping finetune/OHT/configs/seg_roles.yaml \
  --output "$OUT/seg-role-annotations-wheel-plate"

python tools/export_oht_replay_viewer.py \
  --replay "$REPLAY" \
  --role-annotations "$OUT/seg-role-annotations-wheel-plate/annotations.jsonl" \
  --output "$OUT/role-preview-wheel-plate" --every 100
```

打开新 `role-preview-wheel-plate/index.html`。Target 绿、Reference 蓝、重叠黄，有效未来轨迹橙。构建时 preview 不会自动包含后导出的 mask；已有正确 annotations 时只执行第二条。**V4 chunk 不接收 role cache，导出标注不等于模型启用物体辅助。** GT mask 只作训练标签/诊断，不进入在线策略观测。

### 近似对象点云

在 V3/V4 build 命令增加 `--object-cloud-config finetune/OHT/configs/object_clouds.yaml`，对每个生成样本提取**当前帧**记录对象，不按未来 goal 截取。自定义 YAML 例：

```yaml
source: depth
default_size: [0.20, 0.20, 0.20]
objects:
  - index: 0
    name: object_0
    size: [0.30, 0.20, 0.20]
```

`objects: null` 选择全部记录对象；指定索引不存在会失败。尺寸为 world XYZ 轴对齐框全长（米），0.20 只是猜测，不随对象四元数旋转；框内可能混入背景/邻近物体，重叠允许共享点，不能当实例分割 GT 或自动 T/R。

`source: depth` 用缓存米制 depth/K/外参重建旧 ROI 前点云；`xyz` 只用缓存剩余有限 XYZ。每样本在 `<replay>/preview/<task>/<episode>/<frame>/` 保存 `object_<index>.ply`、NPZ 和 `object-clouds.json`；NPZ 为 `xyz[N,3] / rgb[N,3] / camera_ids[N]`。空框保存零点和 valid=false，无对象时列表为空。

`--visualize-every` 只减少 PNG/HTML，不减少逐样本对象文件。对象配置/报告绑定 contract 与索引；诊断不进入训练 batch。PNG 在场景下方增加对象三视图，标明 source/world/local 点数；HTML 可调整框。下载当前帧 `oht-object-boxes.json` 后，可独立导出调整结果：

```bash
SAMPLE=disassemble_left/000000/000000  # 替换成 samples.jsonl 中实际 ID
python tools/extract_oht_object_clouds.py \
  --replay "$REPLAY" --sample-id "$SAMPLE" \
  --boxes /path/to/oht-object-boxes.json \
  --output "$OUT/object-clouds-adjusted-01"
```

JSON 绑定 sample/frame、点源和缓存哈希，不能用于另一帧。导出读取缓存分辨率下全部有效点，不受 HTML 显示预算/相机开关影响，也不恢复原始分辨率或未观测表面。对象中心/原始四元数来自 `diagnostic_objects`；对象数量由位置/四元数长度匹配，不固定四个。索引语义与 objects_quat 顺序须核实，姿态轴默认隐藏，不能直接继承 camera/EE 顺序。

## 几何与可视化诊断

### 生成与刷新预览

build 的 `--visualize-every N` 按每 episode 生成样本数计数，含首个，默认 0；显式 `--visualize-output-dir` 改变 PNG/HTML 位置，对象 PLY/NPZ 仍在 replay 的 preview。V4 非零频率就有基础 HTML；V3 需同时启用 object-cloud-config 才有对象 HTML，单独频率只生成基础 PNG。

完整缓存可随时生成新预览，不重读原视频、不修改 replay：

```bash
python tools/export_oht_replay_viewer.py \
  --replay "$REPLAY" --output "$OUT/preview-refresh-01" \
  --every 100 --max-points 40000
```

打开新目录 `index.html`，复制到本机时保留完整目录和相对路径。`--every` 也按 replay 样本计数；没有未采样原帧。默认点源取已有对象配置，否则 depth，可用 `--source xyz` 切换；旧缓存要看对象时加 `--object-cloud-config`。独立导出只生成 PNG/HTML，不另导出 PLY/NPZ。旧图片不会随代码更新变化。

PNG 包含 sensor RGB/depth 和全局/局部 XY/XZ/YZ：青色当前 EE-pos/TCP、品红下一 GT key、橙色有效 chunk，padding 不画。无额外 TCP 偏移时 EE-pos 即 raw ee_pos_world。全局最多 200000 点；局部以 GT key 为中心、每轴 ±0.20 m，标注 **GT-centered refine diagnostic (NOT model stage2)**，不是模型真实 coarse/refine 前向。depth 蓝近红远，各图范围可变，不能只凭颜色比较距离。

RGB 面板列出投影坐标/光学 Z 和 in view/outside image/behind camera；in view 不证明未遮挡。单像素点云按深度处理，不补造几何。预测与 GT 的比较图由[训练/评估](training.md)生成。

HTML 支持 3D/XY/XZ/YZ、左键旋转、右键或 Shift 拖动平移、滚轮缩放、相机与点源着色、点选 world XYZ。默认切帧保留视角；同 episode 保留对象/ROI 编辑，跨 episode 独立。刷新重置设置，保存时下载 JSON/YAML；编辑只更新交互点云，不重写 PNG。

带对象配置时默认只显示选中框内点，点击“恢复完整场景”返回当前点源的场景抽样。ROI 编辑显示 scene_bounds/keep_bounds/排除框；非法框禁止导出。下载后人工合并到实际 --config 文件，显示开关不改训练配置。点数预算默认 40000、最多 60000；保留计数是显示抽样，不是全量覆盖证明。

### 单帧几何排错

从 `samples.jsonl` 选择真实 ID，再运行；输出必须为新目录：

```bash
SAMPLE=disassemble_left/000000/000000
python tools/diagnose_oht_geometry.py \
  --replay "$REPLAY" --sample-id "$SAMPLE" \
  --output "$OUT/geometry-frame-01" \
  --html --html-source depth --max-points 40000
```

输出每物理相机三视图、`fused_rgb.png`、来源相机着色 `fused_camera_colors.png`、`geometry.json`（实际 K、optical→world、有限点数、配置/验证范围）和 `point-cloud-roi.html`，不生成整批入口页、不改缓存。

`--html-source depth` 重新反投影缓存的米制深度，不重新反量化、不套旧 ROI，仅用于显示；可找回旧 ROI 外仍有有效深度的点。PNG/geometry.json 仍描述原缓存 XYZ。默认 xyz 只显示剩余有限点。两者都不能修复历史深度、外参或同步错误，也不能补全遮挡表面。

旧缓存缺少对象位姿时，viewer/diagnose/extract 可同时加 `--object-root "$ROOT" --object-manifest "$MANIFEST"`；manifest 必须是构建该 replay 的 audit。diagnose 还需 `--html`。工具核对 audit/Parquet 哈希与当前帧，绑定不符拒绝，不改写缓存。

### 构建未完成时

仅诊断可用 `--allow-incomplete`，不要求 complete；已有索引时仍检查绑定哈希。没有 `samples.jsonl` 时直接查看已写完 NPZ，替换路径及 data-profile 为 contract 中实际值：

```bash
python tools/diagnose_oht_geometry.py \
  --replay "$REPLAY" \
  --observation observations/disassemble_left/000000/000000.npz \
  --data-profile disassemble_left-seg/lerobot_dataset \
  --allow-incomplete --html --html-source depth \
  --output "$OUT/geometry-building-01"
```

NPZ 路径相对 replay，不能越界；`--observation` 与 `--sample-id` 二选一。多源必须指定 `source_data_configs` 中真实键，单源可省略。此模式只依赖已有 contract/NPZ，不猜帧时序、TCP 或 goal，PNG 仅有全局图；文件仍写入或字段非法时失败。不会补写索引/complete，训练、teacher 和正式评估不能用此开关绕过完整性。

### 几何验收

先查静态桌面/夹具跨相机是否重合，再看腕部运动时静态环境是否稳定；偏移时依次核对 depth 编码/kind、相机 K/轴方向和逐帧同步，不用 ICP 掩盖配置错误。检查四任务、多时刻、抓取/放置/最高抬升与 Reference，确认 ROI 不误删必要几何，动作不越界。

既有数据说明给出的 frame 965 腕部投影为原图 `(313.0,378.6)`，4 倍步长缓存约 `(78.25,94.65)`；本地未用该真实帧复核。落在图内不能代替多相机/多帧对齐验收。

## 单关键点数据

单关键点目录惯称 `replay-source-v3`，schema 实际为 **`oht_bridgevla_v2`**，与 V4 不互换。复用同一有效 audit，使用新输出：

```bash
KEY_REPLAY="$OUT/replay-source-v3"
python tools/build_oht_replay.py \
  --root "$ROOT" --manifest "$MANIFEST" \
  --config finetune/OHT/configs/dataset.yaml \
  --output "$KEY_REPLAY" --replay-version 3 --sample-stride 10 \
  --object-cloud-config finetune/OHT/configs/object_clouds.yaml \
  --visualize-every 100
python tools/validate_oht_replay.py --replay "$KEY_REPLAY" --workers 4
```

分割原始数据也可运行 `prepare_seg_data.sh --root "$ROOT" --output "$ROOT/bridgevla-seg-v3-run01" --replay-version 3`，自动生成 replay-source-v3、seg-role-annotations 和 role-teacher；OUT 必须是新目录。

默认 `keypoints.method: gripper` 只取实测滤波后二值夹爪切换帧，不加停稳、指令补点或人工末帧。输入严格指向下一事件；最后事件及其后尾段不生成单关键点样本。无事件 episode 在解码前跳过并记入 complete.skipped_episodes；全无事件拒绝生成空缓存。它不监督释放后撤退，也不代表抓取/释放成功。

对照 `bridgevla` 模式另含开闭/停稳/终帧，保留 4 帧 stopped buffer 与相邻终帧去重；OHT 停稳初值 `0.01 m/s`、`5°/s`，基于实测 TCP 差分。`geometric` 还含指令/位移/转角/帧距补点，旧 YAML 缺 method 时保留此行为。更换模式需更新整个 keypoints 块并重建。

单关键点 `low_dim_state` 实际为 `[observed_binary, open01*0.04, open01*0.04, 0]`；中间两项是兼容开度特征，不是实测米制两指开度，最后时间项固定 0。不能把负 rad 关节直接按旧 [0,0.04] 规则裁剪，也不能把 GT 剩余轨迹长度加入在线输入。V4 另有实测 TCP/q6/夹爪条件。

role teacher 接受显式可见 mask 或定义好的 site，不从 objects_pos/held_obj/未来路点自动猜角色。annotations 一行格式示例：

```json
{"id":"disassemble_left/000000/000000","target":{"present":true,"known":true,"source":"visible_surface","mask_path":"masks/sample.npz"},"reference":{"present":false,"known":true,"source":"none"}}
```

上例 NULL 只示范格式，实际按任务定义标注。NPZ 以相机名为键、值为缓存同尺寸 bool mask；原尺寸 mask 按 RGB-D 同步步长采样。`site_region` 提供 world_from_site 4×4 与三维正 size，属于特权训练区域，不是可见表面。presence/known/geometry_valid 独立：已知缺失 false/true/none，存在但几何不可用 true/true/unknown，角色未确认 known=false。遮挡不改为 NULL，不补造表面。

```bash
python tools/build_oht_role_teacher.py \
  --replay "$KEY_REPLAY" --annotations /path/to/annotations.jsonl \
  --output "$OUT/role-teacher-new" --point-count 512 --visualize-every 100
python tools/validate_oht_replay.py \
  --replay "$KEY_REPLAY" --mode role_queries --role-cache "$OUT/role-teacher-new"
```

分割流水线已有正确 teacher 时无需重建。预览默认在 teacher-output/preview，列出 source/present/known/几何有效性；visible_surface 叠加 30% RGB+70% 角色色，site 只画区域轮廓。teacher 当前 TCP/未来 goal 只用于诊断，不进入策略。后续 role_queries、外部预测与训练覆盖要求见[训练](training.md)。

## 迁移边界

| 改动 | 数据与 checkpoint 处理 |
| --- | --- |
| 只刷新 PNG/HTML 或调整诊断对象框 | 新建预览/对象输出，不改 observation/动作 |
| 只改分割角色映射 | 重导 annotations/预览；V3 辅助路线另重建 teacher，V4 observation/稠密动作可复用 |
| 改深度编码、Z/ray、K/外参、四元数、同步、ROI/bounds、夹爪或关键点规则 | 新目录重建 replay 及依赖 teacher/预测缓存；原始数据与有效 audit 可复用 |
| 改 V4 horizon、动作间隔或标签/schema | 重新按目标契约构建，不通过改目录名/哈希伪装迁移 |
| 只换兼容的 chunk 动作头 | 复用契约匹配 V4；初始化/resume 规则见[训练](training.md) |

构建读取实际 --config 文件；仓库更新不会修改已复制的 dataset-calibrated.yaml。迁移时检查 per-camera depth 覆盖，不能只改 contract 或重画错误 XYZ。几何/标签/schema 变化后不沿用旧 optimizer resume，也不只改 teacher hash 强绑新缓存。

OHT 使用独立 reader/dataset、provenance 与 contract，不伪造 RLBench Demo，也不把 RLBench semantic-handle YAML 当 OHT 契约。单关键点 low_dim 与 V4 TCP/q6/夹爪条件不同；公平对照须匹配实际输入。主任务指令固定时，分段 instruction/held_obj/metadata 路点只作 teacher/审计；按 GT instruction_id 自动切换属于 privileged-stage 协议，单独报告。

验收顺序是数据/几何→同初始化的实测轨迹与专家目标执行→训练→配对闭环。相同划分、相机、语言、初始化、预算和执行器下比较 baseline/assistance，Oracle 单独报告；测试留出、合成 CPU 检查或单次 forward 都不能代替完整 CUDA/IsaacLab 验收。客户端、成功判据和时序执行见[运行](runtime.md)。

H-VLA 的远端路径、checkpoint 和历史成绩来自既有用户说明，未在本地复现，不能作为 BridgeVLA 成绩或权重。当前 regression_state 的用户确认与仍缺少的三 seed/定量闭环边界见[训练与评估](training.md)。
