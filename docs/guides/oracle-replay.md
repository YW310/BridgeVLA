[文档索引](../README.md) · [项目首页](../../README.md)

> 命令不在 docs/ 下执行。带 cd 的独立示例从仓库根目录开始；其后命令沿用该目录。替换所有示例路径后再运行。

<a id=oracle-replay></a>

# RLBench Oracle 3D 物体 Replay 数据准备

> 本节导航：[参数表](#oracle-parameters) · [关键行为与检查](#oracle-checks) ·
> [O2 Target/Reference 训练](../experiments/o2-training.md#o2-training)

`tools/augment_replay_with_oracle_objects.py` 可直接为已有 BridgeVLA replay 追加
RLBench GT instance 点云，无需重新采集数据或重建原始 replay。脚本用每个
`N.replay` 的 `episode_idx` 和 `sample_frame` 找到四路相机 mask，经
`rgb_handles_to_mask` 解码后与 replay 点云逐像素对齐，并跨视角合并相同 handle。

```text
N.replay[episode_idx, sample_frame]
  -> episode{episode_idx}/{camera}_mask/{sample_frame}.png
  -> front / left_shoulder / right_shoulder / wrist 点云融合
  -> 固定尺寸 Oracle instance 张量
```

新增字段：

```text
oracle_object_points   [MAX_OBJECTS, NUM_POINTS, 3]  float32
oracle_object_centers  [MAX_OBJECTS, 3]              float32
oracle_object_sizes    [MAX_OBJECTS, 3]              float32
oracle_object_ids      [MAX_OBJECTS]                 int32
oracle_object_valid    [MAX_OBJECTS]                 bool
oracle_object_roles    [MAX_OBJECTS]                 int8
```

`oracle_object_roles` 与 ID/valid 使用相同 slot：`0=unknown/padding`、
`1=target`、`2=reference`。启用时序匹配后，角色按夹爪抓取周期计算：闭爪位置附近被
抓取的 object 是 `target`，松爪/放置位置附近除 target 外最近的 object 是
`reference`。同一抓取周期内 T/R 的稳定 object ID 固定不变，只在下一个抓取周期重新
选择；放置位置附近没有有效候选时 reference 为空。
启用时序匹配后，同一刚性物体的多个 raw mask handle 会先合并点云，再占用一个 slot；
`oracle_object_ids` 使用组内最小 handle ID 作为稳定代表，原始成员保存在 task handle
JSON 的 `object_groups` 和 `group_by_handle` 中。

`sample_frame` 是当前 observation 对应的 raw 帧；`keypoint_frame` 不参与 Oracle
对齐。`terminal == -1` 是 YARR final-observation sentinel，写入全 invalid 的填充张量；
可视化时会通过上一条 replay 的 `next_keypoint_frame` 恢复最终 raw 帧。

推荐先在命令末尾保留 `--dry-run` 检查结果，确认后移除它进行正式写入。以下示例使用
默认开启的严格源对齐检查；发现异常会先写 manifest 再停止，不会静默生成错误 instance。
只有已经确认少量异常样本可以退回 baseline 时，才额外添加 `--skip-invalid-frames`：

```bash
python tools/augment_replay_with_oracle_objects.py \
    --replay-dir LPY/BridgeVLA_RLBench_TRAIN_Buffer \
    --raw-data-dir LPY/BridgeVLA_RLBench_TRAIN_DATA/train \
    --output-dir LPY/BridgeVLA_RLBench_TRAIN_TASK_OBJECT_Buffer \
    --detect-robot-handles \
    --robot-detection-frames 128 \
    --robot-detection-stride 5 \
    --robot-detection-window 200 \
    --robot-motion-threshold 0.1 \
    --robot-link-motion-threshold 0.001 \
    --robot-adjacency-distance 0.10 \
    --temporal-id-matching \
    --task-detection-frames 16 \
    --task-prior-filter \
    --filter-thin-planes \
    --min-object-points 1 \
    --max-objects 32 \
    --num-points 512 \
    --workers 8 \
    --cache-frames 256 \
    --visualize-every 100 \
    --visualize-output-dir oracle_visualizations \
    --visualize-objects-only \
    --thin-plane-max-thickness 0.02 \
    --validate-source-alignment \
    --dry-run
```

如果多任务生成中断，不要添加 `--overwrite`。使用与原来完全相同的参数重新运行，
仅在末尾增加 `--resume`（并确保没有 `--dry-run`）：

    python tools/augment_replay_with_oracle_objects.py [原来的全部参数] --resume

工具的正式 replay 使用临时文件校验后原子改名，因此 `--resume` 会跳过输出目录中
已完成的 `*.replay`；只留下 `*.replay.tmp` 或尚未生成正式文件的条目会重新处理。
已完整完成的 task 会在 episode detection 前直接跳过，部分完成的 task 也只检测和处理
剩余 replay 所涉及的 episode。续跑必须保持原 replay、raw data 和过滤参数不变；
如果需要修改生成参数，应改用新的输出目录或显式 `--overwrite` 全量重建。

<a id=oracle-parameters></a>

## 参数表

| 类别 | 参数 | 默认值 | 说明 |
| --- | --- | --- | --- |
| 输入 | `--replay-dir PATH` | 必填 | 原始 BridgeVLA replay 目录；可为单任务目录或包含多个任务的根目录。 |
| 输入 | `--raw-data-dir PATH` | 必填 | RLBench raw data 的 `train` 目录或可解析到 episode 的上级目录。 |
| 输入 | `--task NAME` | `all` | 任务名、逗号分隔任务名或 `all`；可重复传入。 |
| 输出 | `--output-dir PATH` | 无 | 写入新的 Oracle replay 目录；非 dry-run 时必须与 `--in-place` 二选一。 |
| 输出 | `--in-place` | 关闭 | 原地修改 replay；与 `--output-dir` 互斥，建议优先使用新目录。 |
| 输出 | `--resume` | 关闭 | 断点续跑：跳过已原子写完的正式 replay，只生成缺失文件；与 `--overwrite` 互斥。 |
| 输出 | `--overwrite` | 关闭 | 重新处理并覆盖已有输出 replay/元数据；与 `--resume` 互斥。 |
| 输出 | `--durable-write` | 关闭 | 临时文件重命名前执行 `fsync`；更安全，但网络盘上更慢。 |
| 张量 | `--max-objects N` | `32` | 每帧固定的最大 instance 槽位数；超出部分会截断。 |
| 张量 | `--num-points N` | `512` | 每个 instance 的固定采样点数；不足时有放回采样。 |
| 张量 | `--min-object-points N` | `20` | 跨相机融合并移除 NaN/Inf 后少于该点数的实例会删除；高召回检查可设为 `1`。 |
| 几何过滤 | `--filter-thin-planes` | 关闭 | 用抗噪主平面内点比例删除大薄平面；几何判定优先于当前帧 target/reference。 |
| 几何过滤 | `--thin-plane-max-thickness METRES` | `0.010` | 点到拟合平面的最大内点距离；默认允许主体平面具有约 1 cm 深度噪声。 |
| 几何过滤 | `--thin-plane-min-extent METRES` | `0.30` | 平面内两个方向都至少达到该尺寸时才删除；小于约 30 cm 的托盘、支架面等中小型平面默认保留。 |
| 几何过滤 | `--thin-plane-min-inlier-ratio RATIO` | `0.80` | 至少该比例的点落在平面距离带内才删除；默认允许最多约 20% 深度离群点。提高该值会更保守。 |
| 几何过滤 | `--filter-thin-planes-all-roles` | 默认行为 | 兼容旧命令；现在 target/reference 薄平面也默认删除。 |
| 几何过滤 | `--preserve-role-thin-planes` | 关闭 | 仅当任务确实包含薄片状相关物体时，选择保留 target/reference 薄平面。 |
| 张量 | `--camera NAME` | 四路相机 | 指定相机，可重复传入；默认 `front`、`left_shoulder`、`right_shoulder`、`wrist`。 |
| 排除 | `--exclude-object-id ID` | `0` | 精确排除 decoded handle，可重复传入；`--exclude-robot-id` 是同义参数。 |
| 单帧先验 | `--task-prior-filter` | 关闭 | 按下一关键动作距离排序，并删除明显大平面背景；默认高召回，不按半径删除远处实例。 |
| 单帧先验 | `--task-prior-strict` | 关闭 | 配合 `--task-prior-filter` 删除交互半径外实例；召回率更低，仅在明确需要激进筛选时使用。 |
| 单帧先验 | `--task-prior-radius METRES` | 按任务 | 覆盖 18 个任务配置中的交互半径。 |
| 单帧先验 | `--task-prior-max-instances N` | 高召回不限；strict 按任务 | 限制先验保留的 simulator handle 数。 |
| 单帧先验 | `--task-prior-background-extent METRES` | `0.60` | 两个轴均达到该尺度时视为明显桌面/地面。 |
| 时序匹配 | `--temporal-id-matching`（兼容旧名 `--temporal-task-filter`） | 关闭 | 建立稳定 `handle ID → object group → slot`；以闭爪处物体为 T、松爪放置位置最近的其他物体为 R，并在同一抓取周期固定 T/R；不删除当前帧可见实例。 |
| 时序任务 | `--task-detection-frames N` | `16` | 每个 episode 均匀抽取的最大检测帧数；需覆盖闭爪和松爪边界，长 episode 建议提高到 `24` 或 `32`。 |
| 时序匹配 | `--task-handle-cache-dir PATH` | `<output-dir>/<task>/task_handle_maps` | episode 稳定 slot 与 task handle JSON 缓存；显式 PATH 作为根目录并追加 task 名。 |
| 时序任务 | `--refresh-task-handle-cache` | 关闭 | 忽略已有 task handle JSON 并重新检测。 |
| 机器人 | `--detect-robot-handles` | 关闭 | 检测 wrist 稳定的夹爪 seed，并沿持续邻接的运动学链扩展到机械臂及静止底座；只使用第一次闭合前的前缀，避免把被抓物体当作机器人。 |
| 机器人 | `--robot-detection-frames N` | `64` | 自适应扩展时最多读取的 raw 证据帧数；正常有运动时通常只读取初始窗口的 21 帧。 |
| 机器人 | `--robot-detection-stride N` | `5` | raw 帧采样间隔；默认依次读取 `0, 5, 10, ...`。 |
| 机器人 | `--robot-detection-window N` | `100` | 从 raw 帧 0 开始的初始闭区间；运动不足时才在该窗口之后继续扩展。 |
| 机器人 | `--robot-motion-threshold METRES` | `0.02` | 只控制自适应 raw 采样：夹爪最大位移达到该值后不再扩展采样；不直接改变 arm 判定。 |
| 机器人 | `--robot-link-motion-threshold METRES` | `0.008` | 第一段机械臂 link 需要达到的最小位移；漏掉低幅运动 link 时可尝试 `0.002` 或 `0.001`。 |
| 机器人 | `--robot-adjacency-distance METRES` | `0.05` | 两个 robot handle 的 AABB 最大连接间隙；分段机械臂链断开时可尝试 `0.08`，过大会增加误删风险。 |
| 机器人 | `--robot-handle-cache-dir PATH` | `<output-dir>/<task>/robot_handle_maps` | episode robot handle JSON 缓存；显式 PATH 作为根目录并追加 task 名。 |
| 机器人 | `--refresh-robot-handle-cache` | 关闭 | 忽略已有 robot handle JSON 并重新检测。 |
| 性能 | `--refresh-replay-metadata-cache` | 关闭 | 强制重建 replay 元数据索引；仅在同名 `.replay` 被原地改写时使用，日常运行不要添加。 |
| 检查 | `--validate-source-alignment` | 开启 | 默认严格比较 raw/replay RGB，并比较 raw depth 重建 XYZ 与 replay point cloud；异常时停止，防止静默生成错位 instance。 |
| 检查 | `--no-validate-source-alignment` | 关闭 | 为旧流程关闭内容校验以提高速度；只建议在已离线确认 replay/raw 完全一致时使用。 |
| 检查 | `--alignment-rgb-tolerance VALUE` | `1.0` | RGB 单通道允许的最大差值，按 `0..255` 尺度计算。 |
| 检查 | `--alignment-point-cloud-tolerance METRES` | `0.002` | raw 重建点云与 replay 点云逐像素误差的 P95 上限。 |
| 检查 | `--alignment-min-finite-ratio RATIO` | `0.95` | 每个相机 replay 点云的最小有限点比例，同时用于要求 raw/replay 可比较像素比例。 |
| 容错 | `--skip-invalid-frames` | 关闭 | raw 帧越界或源对齐校验失败时保留 replay，并写入 `valid=False` 的空 Oracle；仅用于明确接受 baseline fallback 的容错实验。 |
| 性能 | `--workers N` | `1` | replay 线程数；建议从 `4` 或 `8` 测试，过高会增加内存和网络盘竞争。 |
| 性能 | `--cache-frames N` | `128` | 相同 raw 帧 Oracle 结果的 LRU 容量；`0` 禁用，内存有限时降低。 |
| 性能 | `--seed N` | `0` | 控制确定性点采样和 dry-run 抽样。 |
| 性能 | `--no-progress` | 关闭 | 关闭 tqdm 进度条。 |
| 检查 | `--dry-run` | 关闭 | 只验证和可视化，不写 replay，也不新建/覆盖 handle 缓存。 |
| 检查 | `--dry-run-samples N` | `5` | 未指定间隔可视化时，dry-run 随机检查的普通 transition 数。 |
| 可视化 | `--visualize-index N` | 无 | 保存指定 replay index；与 `--visualize-every` 互斥。 |
| 可视化 | `--visualize-every N` | `0` | 每隔 N 个排序后的 replay 保存一张 PNG；`0` 关闭。 |
| 可视化 | `--visualize-output-dir PATH` | `oracle_visualizations` | PNG 输出目录，不设置时也会自动创建该默认目录。 |
| 可视化 | `--visualize-objects-only` | 关闭 | 隐藏点云面板中的灰色完整场景，仅绘制保留实例；不影响上排 RGB。 |

<a id=oracle-checks></a>

## 关键行为与检查

- 推荐同时启用 `--detect-robot-handles`、`--temporal-id-matching` 和
  `--task-prior-filter`：先排除机器人，再建立 episode 级稳定 slot，最后进行单帧
  排序和背景清理。旧参数名 `--temporal-task-filter` 保持兼容，但不再执行硬过滤。
- 时序检测会把交互/邻接 handle 排在 episode slot 映射前部，并把其他观测到的
  handle 追加到稳定映射；`rejected_dynamic_handles` 仅作为诊断和优先级证据，不会由
  temporal 模式删除。某个稳定 handle 在当前帧不可见时保留该 slot，写入
  `valid=False`；其他可见 handle 会使用剩余 slot。
- 时序流程先按持续空间邻接和多帧相对位姿合并 raw handle；`max_instances` 限制的是 group
  数而不是 raw handle 数。episode 内其他帧的夹爪开闭、物体随动、动作邻近和静态接触会
  形成 task group 及 `role_cycles`。每个周期在闭爪时选择附近被抓取的 group 作为 T，
  在后续松爪时选择离夹爪放置位置最近的其他 group 作为 R；从该周期开始到松爪帧，所有
  replay observation 都查询同一组 T/R，不会因夹爪移动或当前帧距离变化而切换。周期外或
  未检测到完整开闭事件时，才使用时序先验与当前帧几何的兼容回退；reference 始终可选且
  每周期最多一个。检测结果保存在 task handle JSON 的 `role_cycles` 字段，运行日志中的
  `cycles=` 依次显示 `[start_frame, end_frame, target_id, reference_id]`，其中
  `reference_id=-1` 表示没有 R。若抓取任务日志为 `cycles=[]`，可使用
  `--task-detection-frames 32`（长 episode 可继续提高）并添加
  `--refresh-task-handle-cache` 重新检测。low-dim observation 使用最多 8 个 episode 的
  LRU 缓存并自动淘汰。
- 刚性分组要求两个 handle 在至少 75% 的共同可见证据中可用、80% 以上持续邻接，且
  多帧中心间距离标准差不超过 1 cm。方向和幅度一致的共同运动可以合并；若两者一直静止，
  边界长期紧密接触也可以合并。所有持续兼容关系最终按连接图的连通分量合并，因此支架的
  多个末端区域即使彼此不直接接触，只要都稳定连接到同一中心/底座，也会形成一个 object
  group。仅在任务后期才接触支架的杯子不满足全时段持续邻接要求，不会并入支架。逐帧角色
  只对分组后的 object group 计算，不对 raw handle 单独赋值。
  普通第一阶段始终使用 2 cm，避免把静止邻近物体在整幅场景中误合并。`place_cups` 只有在
  多帧证据先找到 reference 支架种子后，才允许从该种子向其他静止、相对位姿稳定的区域做
  6 cm 二阶段结构扩展；target 和发生明显运动的杯子不会进入该扩展。
- Robot 检测使用 raw observation 中的当前夹爪位姿、`gripper_open`、GT mask 和 raw
  depth。depth 会用同帧相机内外参重建世界坐标点云，与 mask 像素严格对齐。夹爪 seed 以
  wrist 图像稳定性为主，并允许夹爪旋转造成的世界坐标偏移、部分遮挡及距离离群；夹爪
  handle 不必在第 0 帧可见，在 episode 早期窗口内首次出现（如 raw frame 10）时，会以
  它自己的首个可见帧检查 wrist 邻接和后续随动。严格评分没有 seed 时会从 wrist 稳定候选
  恢复 seed。只在外部相机可见、wrist mask 中不可见的夹爪 link 也允许在早期首次出现；
  arm 扩展会使用它与 seed 的首个共同可见帧，而不再强制要求 raw frame 0 可见。arm 扩展先确认紧邻夹爪的移动 link，
  再沿 episode 早期已连接且持续邻接的运动学链扩展，因此可覆盖运动较少的机械臂底座。
  默认在 `0–100` raw 帧内每隔 5 帧取样；若夹爪相对第 0 帧的最大位移不足 2 cm，则继续
  按相同间隔向后扩展，直到运动足够、达到帧数上限、episode 结束或夹爪第一次闭合。
  `--robot-motion-threshold` 只影响这里的采样长度。机械臂是否进入 `arm_handles` 由
  `--robot-link-motion-threshold`、`--robot-adjacency-distance`、可见率和持续邻接共同决定。
  低幅 link 只要达到独立的绝对运动阈值，就不会再仅因运动量小于夹爪的 25% 被当作静止物体。
  若一直静止会打印 `motion-based robot evidence is weak`，提示运动证据不足。检测只使用
  第一次闭合之前的前缀，因此不会把随后被夹起并跟随夹爪运动的任务物体当成机械臂。
  夹爪持续运动而某实例基本静止时，该实例不会判为机器人；证据不足的实例只进入
  `ambiguous_handles`，不会加入 `excluded_object_ids`。task prior 和时序任务筛选会
  使用 replay 的下一关键动作，因此属于 action-conditioned Oracle 离线标注，不是
  无标签推理阶段的公平筛选器。整个流程不调用 Qwen 或 SAM。
- Replay 文件编号按写入顺序排列，但 `sample_frame` 是稀疏关键帧，不保证 raw 帧号
  连续；`terminal == -1` 分隔的是 replay 子序列。Demo augmentation 可能让同一个
  `episode_idx` 出现多段子序列。Robot 只借助 replay metadata 确定 episode，检测证据
  直接来自 episode 前部等间隔的 raw 帧；检测出的稳定 handle ID 会应用到后续整个
  episode。task 仍执行 episode 级 replay 均匀采样，并按
  `(episode_idx, sample_frame, next_keypoint_frame)` 保留不同动作边。
- 提供 `--output-dir` 时，Task 缓存位于
  `<output-dir>/<task>/task_handle_maps/episode_NNNN.json`，robot 缓存位于
  `<output-dir>/<task>/robot_handle_maps/episode_NNNN.json`。显式指定对应的
  `--*-handle-cache-dir PATH` 时使用 `PATH/<task>/episode_NNNN.json`，避免多任务间
  episode 文件重名。Robot cache 会记录 raw 采样间隔、窗口、帧数上限、采样运动阈值、
  link 运动阈值和邻接距离；
  修改这些参数时会自动失效。修改 task 检测帧数、半径或实例限制后仍应使用
  `--refresh-task-handle-cache`；修复前生成的旧版 task/robot 缓存也会自动失效。
- 缺少可用的 `replay_info.npy` 时，首次运行必须读取每个 `.replay` 的 metadata，并写入
  `.oracle_replay_metadata_v2.npz`。缓存会根据 replay 文件名、大小和修改时间自动失效；
  提供 `--output-dir` 时，缓存位于对应的输出 task
  目录（包括 dry-run）；未提供输出目录或使用 `--in-place` 时才写入输入 replay 目录。
  以后运行会显示
  `replay metadata disk cache hit`，同一次运行中 robot/task 共用索引时显示
  `memory cache hit`。新增、删除或重命名 replay 会自动使索引失效；若原地改写同名文件，
  使用一次 `--refresh-replay-metadata-cache`。如果日志提示无法保存索引，需要检查 replay
  目录写权限，否则下次仍会全量扫描。
- 如果日志显示 `episode detection: replay segments`，说明当前目录存在长度匹配的
  `replay_info.npy`。程序只用它划分 replay segment，并直接从选中的 `.replay` 读取
  `episode_idx`、`sample_frame` 和 `next_keypoint_frame`；这条路径不读取
  `.oracle_replay_metadata_v2.npz`，所以 `--refresh-replay-metadata-cache` 不会修复这里的
  帧越界。注意 `85.replay` 中的 `85` 只是 replay 写入序号，实际 raw episode 仍由文件内的
  `episode_idx` 决定。
- 如果提示 `Replay/raw frame mismatch` 或
  `Cannot read current gripper_pose for frame N`，且 raw episode 中确实没有第 `N` 帧：仅当
  前面的日志是 `replay metadata disk cache hit` 或 `memory cache hit` 时，先使用一次
  `--refresh-replay-metadata-cache`；如果日志是 `replay segments`，或者刷新后仍然越界，则
  对应 `.replay` 与 raw episode 确实不匹配。这通常表示 replay 与 raw data 来自不同版本，
  或 raw episode 不完整。应改用生成 replay 时的同一批 RLBench demonstrations，或基于当前
  raw data 重新生成 replay。不要把越界帧强行截到最后一帧，也不要直接改写
  `sample_frame`，否则会造成图像、点云、夹爪状态和动作监督错位。
- 若确认只有少量异常样本，并允许它们退回 baseline，可添加
  `--skip-invalid-frames`：程序保留 replay 的连续结构，为越界样本写入
  `valid=False` 的空 Oracle，并输出 `ignored_invalid_frames` 与
  `ignored_invalid_replays`。这属于容错绕过，不会修复数据配对关系。
- 源对齐检查默认开启。图像来自 raw `*_rgb/{sample_frame}.png`，instance mask 来自
  raw `*_mask/{sample_frame}.png`，而训练点云来自 `.replay` 的
  `*_point_cloud`；程序现在会检查三者是否对齐。缺少点云字段、整帧或任一 mask instance
  的有限点比例不足、RGB 不一致，或 raw depth 重建点云的整帧/逐实例 P95 误差超过阈值，
  都会判为无效，因此小物体点云单独缺失也不会被全局统计掩盖。
- 每个 task 的检查结果写入输出目录下的 `invalid_alignment_manifest.json`；即使严格模式
  在第一个异常处退出，manifest 也会先原子落盘。dry-run 时写入
  `<visualize-output-dir>/<task>/invalid_alignment_manifest.json`。最终汇总中的
  `alignment_invalid_frames` 是检测到的唯一异常帧数。
- raw 数据没有单独的 point-cloud 文件夹属于正常现象：点云由 depth PNG、
  `low_dim_obs.pkl` 中的相机内参/外参和 near/far 在线重建。若 depth PNG 或相机参数缺失，
  则该帧无法可靠生成 Oracle instance，不应只因为 RGB 存在就继续训练。
- `dry-run` 会打印 `excluded_object_ids`、`no_finite_point_object_ids`、
  `small_object_ids`、`task_prior_filtered_object_ids`、
  `temporal_filtered_object_ids` 和 `truncated_object_ids`。命令行时序匹配不再产生
  `temporal_filtered_object_ids`；若缺失 ID 不在其他列表中，说明它在当前帧所选相机
  的 GT mask 中不可见。
- 每张可视化 PNG 使用两排八个面板：上排为 front、left shoulder、right shoulder、
  wrist RGB；下排为 3D、XY、XZ、YZ 点云。相同 episode/handle ID 跨帧颜色固定；图中
  不直接显示可能达到千万级的 simulator handle，而按 episode 稳定 slot 映射为从 `1`
  开始的连续小编号。上排利用同帧 GT mask 为下排实际保留的 ID 绘制半透明同色框：
  unknown 只显示小编号（例如 `3`），target/reference 分别简写为 `T_3` / `R_4`。
  底层 `oracle_object_ids` 和 handle 缓存仍保留真实 ID，不影响跨帧匹配与训练数据。
  3D 图例和三个正交视图采用相同标签；某个实例在当前相机不可见时不画框。
  点云使用固定米制
  场景边界；缺失 RGB 显示
  `RGB unavailable`，不会中断生成。
- 默认先写 `.tmp`、回读验证后原子重命名，不覆盖原始数据。缓存会自动淘汰已完成的
  旧帧；进度条中的 `cache_hits`、`cache_misses` 和 `cache_entries` 可用于检查效果。

**突然只剩一个物体时：**先比较启动日志中的 `slots=[...]` 和图片标题里的
`episode_idx/sample_frame`。`--max-objects 32` 是 episode 稳定映射的容量上限，不会
补回当前帧不可见的实例。

- 如果 `task handles` 只有一个，但 `slots` 有多个，其他实例仍会保留；task handles
  现在只决定容量不足时的优先级，不再构成白名单。旧版按白名单删除实例的 task cache
  会通过版本号自动失效。
- 如果某个 replay 只剩一个，检查 dry-run 输出：`excluded_object_ids` 表示被
  robot/手工 ID 排除，`task_prior_filtered_object_ids` 表示被显式单帧先验删除，
  `small_object_ids` 表示点数不足，`thin_plane_object_ids` 表示被显式薄平面规则删除，
  `protected_thin_plane_object_ids` 表示几何上是薄平面但因 target/reference 角色而保留，
  `no_finite_point_object_ids` 表示 mask 可见但点云
  全为 NaN/Inf。缺失 ID 不在任何列表时，说明它在当前帧所选相机的 GT mask 中不可见
  或被完全遮挡；不再归因于 temporal 匹配。
- 若 robot 检测仍有疑似误判，检查
  `<output-dir>/<task>/robot_handle_maps/episode_NNNN.json`：只有
  `gripper_handles` 和
  `arm_handles` 会硬删除；`grasped_handles` 是被抓物体保护集合，
  `ambiguous_handles` 仅供诊断。旧版 robot cache 会自动失效；重新生成 robot 结果后
  也应刷新 task cache。
- 图片标题切换到新的 `episode_idx` 时，会改用另一份 episode cache；这不是同一
  episode 内 ID 突变。`sentinel=True` 对应 `terminal == -1` 填充 transition，保存的
  Oracle 张量为空是预期行为。

可先提高 robot/task 采样密度和交互半径重新检查；dry-run 只实时重检，不写入新缓存，
确认后需移除 `--dry-run` 才会保存结果。robot 结果会影响后续 task slot，因此刷新顺序
是先 robot、再 task：

```bash
--detect-robot-handles \
--robot-detection-frames 64 \
--robot-detection-stride 5 \
--robot-detection-window 100 \
--robot-motion-threshold 0.02 \
--robot-link-motion-threshold 0.002 \
--robot-adjacency-distance 0.08 \
--refresh-robot-handle-cache \
--temporal-id-matching \
--task-detection-frames 32 \
--task-prior-radius 0.30 \
--refresh-task-handle-cache \
--dry-run \
--visualize-index 100
```

同一 `episode_idx` 的多个 replay 子序列共用 slot 映射；切换到新的 `episode_idx` 时
使用独立映射。训练随机采样 replay 时不依赖上一条 transition。

训练加载 Oracle replay 时，通过 `--train_replay_storage_dir` 显式指定 Oracle
输出根目录，并启用与数据准备阶段一致的张量尺寸：

    bash train.sh --train_replay_storage_dir /path/to/augmented_replay \
        --exp_cfg_opts 'use_oracle_objects True oracle_max_objects 32 oracle_num_points 512' \
        [其他训练参数]

use_oracle_objects 默认为 False，因此原始非 Oracle replay 的加载行为保持不变。

> 上述启发式角色不是严格 semantic-GT。正式 O2 角色标注请继续阅读 [Semantic-GT](semantic-gt.md)。
