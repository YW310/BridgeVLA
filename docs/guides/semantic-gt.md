[文档索引](../README.md) · [项目首页](../../README.md)

> 命令不在 docs/ 下执行。带 cd 的独立示例从仓库根目录开始；其后命令沿用该目录。替换所有示例路径后再运行。

<a id=semantic-gt-roles></a>

# 严格 Semantic-GT Target/Reference

`demo_events` 的生成统计写入 `manifest_results.csv`，字段为生成覆盖率（百分比）、
已生成/请求 episode 数和逻辑 transition 数；TensorBoard 标签为 `manifest_coverage_<task>`。
`eval.sh` 合并为 `*_merged_manifest_results.csv`。覆盖率 100% 只代表 manifest 生成完成，
不是模型闭环成功率，也不代表几何验证通过。普通评估继续使用 `eval_results.csv` 和 `eval_<task>`。
单个 episode 也会输出数值统计；缺失指标不再以 `unknown` 字符串写入 TensorBoard。
若旧版在 TensorBoard 收尾时报错，已经逐 episode 原子保存的 manifest 仍保留，
无需仅为该日志错误重新生成它们。
批量运行中途在 episode `N` 停止时，推荐保留原命令并增加
`EVAL_RESUME=1 SAVE_VIDEO=0 VISUALIZE=0`。这是通用恢复开关：

- 标准 closed-loop 测试每完成一个 episode，就原子保存
  `eval/<task>/<provider>/<model>/episode_results/<task>/episode_N.json`。重启时只有
  checkpoint、实验/MVT 配置、数据目录、episode length、Oracle 关键参数和核心评估源码
  的签名完全一致才会显示 `[Evaluation][RESUME] ... skipped`；签名不同、文件损坏或
  字段无效会重跑。
- `MANIFEST_PHASE_SOURCE=demo_events` 时复用已验证的 manifest，显示
  `[Manifest][RESUME] ... skipped`。缺失、损坏、未完成、alignment 模式或当前 role
  YAML 摘要不一致的 episode 会重新生成。
- `sim_replay` expert-action manifest 不能安全地从普通评估结果恢复；需要可恢复生成时使用
  `demo_events`。
- resume 不支持同时保存视频或逐帧可视化，因为跳过的 episode 无法补回这些视觉产物。
  直接调用 `eval.py` 时 resume 仅支持单任务；启用 resume 后，`eval.sh` 会把
  `TASKS="all"` 展开成 18 个独立任务进程，避免跳过整个任务后 simulator task 状态错位。
  因而全任务恢复时会有逐任务重新加载模型的启动开销。

`MANIFEST_RESUME=1` 和 CLI `--manifest-resume` 仍是兼容别名，新命令统一使用
`EVAL_RESUME=1` / `--eval-resume`。旧标准测试没有逐 episode 日志，第一次开启 resume
仍需运行一次；之后才能自动跳过。也可手工使用
`START_EPISODE=4 EVAL_EPISODES=96`，其中 `EVAL_EPISODES` 是本次运行数量，不是终止下标。
Manifest resume 会检查已保存的 mask 指纹是否存在，但为了在 simulator 启动前快速跳过，
不会重新读取 raw mask 计算哈希；semantic replay 重写阶段仍会逐 episode 重算并严格比对。

批量 strict 生成不希望因单个对齐错误停止时，可同时设置
`MANIFEST_CONTINUE_ON_ERROR=1`。该开关只适用于 `demo_events`：失败 episode 不会生成
manifest，也不会伪造 handle 映射；错误记录原子写入
`semantic_oracle/manifest_failures/<task>/episode_N.json`，随后继续下一 episode。建议始终与
`EVAL_RESUME=1` 配合；修复对齐问题后重跑相同命令，完整 episode 被跳过，失败 episode
再次尝试，成功后旧 failure marker 自动删除。最终 `Generated Coverage` 小于 100% 就表示
仍有失败项，不能把该批 manifest 当作完整训练输入。

## 可选的 mask 身份验证（不代表点云几何通过）

`ORACLE_HANDLE_ALIGNMENT=verified` 仍为默认：同时检查多视角 mask 和 1 cm 点云距离。
若已配准视角的完整实例 mask 完全相同，但点云距离检查失败，可显式改为
`ORACLE_HANDLE_ALIGNMENT=mask_verified`（仅用于 `MANIFEST_PHASE_SOURCE=demo_events`）。
其余命令参数不变，无需关闭 `ORACLE_STRICT=1`。

新模式仍要求至少两个视角、每个支持视角至少 16 像素、完整实例 mask 的
precision/recall 均不低于 0.9，并且映射唯一。第三视角的小范围轮廓栅格化差异只记入
审计；单侧可见或 precision/recall 低于 0.5 的实质性冲突仍会否决。隐藏实体不能靠
这个模式猜测。
薄环等只在一个相机达到 16 像素的实体采用受限回退：候选必须全局唯一，并且只有一个
可见检查、至少 32 像素、mask precision/recall 均不低于 0.98、点云 P95 距离不超过
1 cm。报告以 `registered_mask_overlap_single_view_geometry` 标记；不满足任一条件仍拒绝。
顶层 `geometry_policy=audit_only_except_single_view_corroboration` 表示全局点云几何仍未
认证，但该受限回退确实使用当前实例的局部几何作为身份佐证。
对于由多个 simulator handles 组成的同一语义实体，若某个部件在所有已配准视角均少于
16 像素且没有候选映射，该部件以 `excluded_unobservable` 记入审计而不猜测 ID；只有
同一实体至少还有一个其他部件通过多视角映射时才允许生成。若整个实体不可观测，仍会
在 strict 模式中终止。
`reach_and_drag` 是明确例外：RLBench 的彩色 `target0` 不出现在保存的实例 mask 中，
因此 Reference 定义为其世界坐标中心 `site`，而不是伪造 object handle。Target 仍为
`stick`；训练和在线测试复用同一配置，均通过现有 Gaussian site prior 表示 T/R 关系。
点云偏差只审计，不阻止身份映射；不修正原始点云，也不保证所有 episode 都能通过。
报告和 manifest 使用独立的 `status=mask_verified`、`geometry_verified=false`；
逐候选视角包含 `geometry_passed` / `geometry_warnings`。

后续 `tools/rewrite_replay_with_semantic_roles.py` 默认拒绝这种 manifest。
检查几何审计并决定接受后，须显式添加 `--allow-mask-verified-handles`，
输出到独立 buffer。正式几何可信的 upper-bound 实验仍应先定位并修复深度偏差，
不能将 mask 身份验证通过描述为几何验证通过。

正式 O2 upper-bound 不再使用最近距离、运动幅度、Qwen 或时域 ID 猜测角色。唯一语义
契约是 `finetune/RLBench/configs/rlbench_o2_semantic_roles.yaml`：Target 是当前未完成
子目标中必须直接接触、抓取或控制的实体；Reference 是该子目标终止条件中与 Target
构成空间关系的唯一物体或 site。单物体关节任务没有 Reference。一个语义实体可合并
多个 simulator handles，phase 只在 live RLBench 成功条件满足后推进。

| 任务类型 | Target / Reference | phase 规则 |
| --- | --- | --- |
| 单关节 | `open_drawer`、`push_buttons`、`turn_tap`：T 为源码指定的可动部件，R 不存在 | 对应 joint condition 满足 |
| 单次放置 | `close_jar`、`light_bulb_in`、`meat_off_grill`、`place_shape_in_shape_sorter`、`place_wine_at_rack_location`、`put_groceries_in_cupboard`、`put_item_in_drawer`、`put_money_in_safe`、`slide_block_to_color_target` | variation 决定唯一 T/R；detector 和需要时的释放条件满足 |
| 顺序操作 | `place_cups`、`stack_blocks`、`stack_cups` | 固定源码顺序；空抓、错误张合和其他物体移动不推进 |
| 工具任务 | `reach_and_drag`：stick/target；`sweep_to_dustpan_of_size`：broom/dustpan site | 不新增第三个 Tool 通道 |
| 几何选择 | `insert_onto_square_peg`：ring/与 `success_centre` 对齐的 pillar | 四个 detector 同时满足 |

Reference `kind=object` 时从四视角 GT handle mask 提取完整点云；`kind=site` 时读取
success sensor/dummy 的 world position，并重复到 `oracle_num_points` 后走原有 Gaussian
projection。site 只用于 Oracle upper-bound，manifest 和 replay audit 字段都会显式记录
`kind=site`。

正式生成前先对全部 variation 做 strict reset 审计（不需要 checkpoint）：

```bash
cd finetune/RLBench
python validate_semantic_roles.py \
    --output-dir semantic_role_validation \
    --headless
```

任一对象选择器无法解析、T/R 混入 robot handle 或层级不满足契约时立即报错；成功时
输出 `variation_role_audit.json`、逐 variation 首帧 audit 图和 provider 统计。

## 1. 用 simulator GT 生成 phase/handle manifest

在 RLBench 环境中回放保存的 expert keypoints。每个 episode 只调用一次
`reset_to_demo`，provider 直接查询 task 对象属性、层级、variation、success condition 和
四视角 GT mask：

```bash
cd finetune/RLBench
TASKS="all" \
MODEL_FOLDER=/home/yiwei/project/BridgeVLA/checkpoints/RLBench  \
MODEL_NAME=model_80.pth \
EXP_CFG_PATH=/home/yiwei/project/BridgeVLA/finetune/RLBench/configs/rlbench_config.yaml \
EVAL_DATAFOLDER=/home/yiwei/project/BridgeVLA/LPY/BridgeVLA_RLBench_TRAIN_DATA/train \
EVAL_EPISODES=100 \
EPISODE_LENGTH=50 \
REPLAY_GROUND_TRUTH=1 \
GT_REPLAY_RETRIES=3 \
MANIFEST_PHASE_SOURCE=sim_replay \
SAVE_VIDEO=0 \
ORACLE_PROVIDER=rlbench_gt \
ORACLE_STRICT=1 \
ORACLE_DEBUG=1 \
bash eval.sh
```

manifest 生成只回放 expert action，不调用 policy，因此可以使用已有 baseline checkpoint，
不依赖尚未训练的 O2 checkpoint；模型仅用于复用现有 eval 启动入口。

`GT_REPLAY_RETRIES` 表示首次 expert replay 失败后，最多从同一个 `reset_to_demo`
完整重跑 episode 的次数，默认是 3；设为 0 可关闭。重试不会增加逻辑 episode 或任务切换
计数，Final Score 仍只统计每个 episode 最终采用的尝试。provider 会丢弃失败尝试的全部
entries，manifest 的 `generation_attempt` 从 1 开始记录最终采用的是第几次尝试。若全部重试仍失败，
保留最后一次失败 manifest，离线重写器会因最终 `completion_satisfied=False` 拒绝使用。

18 个任务均支持不重新执行动作的 stored-demo phase 模式：

```bash
TASKS="all" \
REPLAY_GROUND_TRUTH=1 \
MANIFEST_PHASE_SOURCE=demo_events \
EVAL_RESUME=1 \
MANIFEST_CONTINUE_ON_ERROR=1 \
ORACLE_HANDLE_ALIGNMENT=verified \
EVAL_DATAFOLDER=/home/yiwei/project/BridgeVLA/LPY/BridgeVLA_RLBench_TRAIN_DATA/train \
SAVE_VIDEO=0 \
ORACLE_PROVIDER=rlbench_gt \
ORACLE_STRICT=1 \
ORACLE_DEBUG=0 \
bash eval.sh
```

该模式不调用 simulator `step()`，因此没有 IK、路径规划或接触重放失败，也不需要重试。
18 个任务使用 YAML 中逐任务声明的严格策略：

| 策略 | 任务 | phase 边界 |
| --- | --- | --- |
| `single_success` | 除下列 4 个多阶段任务外的 14 个任务 | 成功 stored demo 的末帧；这些任务始终只有 phase 0 |
| `release_cycles` | `place_cups`、`stack_blocks`、`stack_cups` | 每次夹爪 close→open 完成一个固定顺序子目标；释放次数必须严格等于 phase 数 |
| `ordered_target_contact` | `push_buttons` | 按源码固定按钮顺序，在 expert keypoints 中用夹爪到对应 GT top-plate 的距离定位接触；要求边界严格递增且距离不超过 YAML 的 `max_distance` |

`push_buttons` 使用接触距离仅定位“何时切 phase”，不会用距离选择“哪个物体是
Target”；Target 顺序仍唯一来自任务源码和 YAML。原因是当前 legacy stored observation
没有 task joint state，无法离线读取 `button_joint >= 0.003`。manifest 会额外保存
`phase_strategy`、`phase_boundary_source`、`phase_boundary_frames`，按钮任务还保存
`contact_distances`，便于审计；任何次数、顺序、可见性或距离校验失败都会终止该 episode，
不会静默退回启发式角色。

manifest 和每个 entry 都记录 `phase_source=demo_events`。默认
`MANIFEST_PHASE_SOURCE=sim_replay` 保持原有在线 success-condition 行为。
生成前仍会执行一次 simulator reset，并将 live 首帧与 stored demo 第 0 帧的 T/R handle
可见性进行交叉检查；只有 manifest 中 `source_alignment_validated=true` 时，离线重写器
才接受该 demo-events 标注。
该模式日志中的 `Generated Coverage=100` 只表示原始 demo 通过事件校验并生成了完整
manifest，不表示重新执行动作获得了 100% closed-loop success。

生成训练 manifest 时必须显式指定对应的训练 raw 数据目录。前一个命令中的临时
环境变量不会自动保留给下一个命令；省略 EVAL_DATAFOLDER 会采用脚本默认值或
shell 已导出的值。

若出现首帧 live/stored 不一致，检查现在会在 phase 生成前失败，并列出 T/R 的
live handles、各相机保存 mask 中的匹配像素数、有效点数和主要 handle：

- stored_masks_missing：保存 observation 没有加载 mask。
- matching_mask_pixels_but_no_finite_point_cloud：mask 有对应实例，但缺点云或对应点均无效。
- live_role_handles_absent_from_stored_masks：保存 mask 中没有当前 simulator 的角色 handle；
  需检查数据目录、原始采集环境及名称到 handle 的映射。

reset_to_demo 恢复场景初始条件，并不能据此保证跨 simulator 会话的 handle 编号相同。
demo_events 默认启用 ORACLE_HANDLE_ALIGNMENT=verified。在首次生成 phase 前，收集
该 episode **所有 phase** 的 T/R 可渲染部件，建立一次 live→stored 映射，随后统一使用
stored handles 提取保存帧点云。不可渲染的物理部件、dummy/joint 不需要 mask ID；
首帧遮挡的可渲染部件仍必须有映射，不能默默忽略。

| 参数或检查 | 行为 |
| --- | --- |
| ORACLE_HANDLE_ALIGNMENT=verified | 默认；仅用于 demo_events，在线 policy 仍使用 live handles |
| ORACLE_HANDLE_ALIGNMENT=identity | 旧编号假设，仅用于已有同编号数据的兼容检查；不能解决编号错配 |
| ORACLE_HANDLE_MAP_DIR | 可选原始采集映射根目录，文件为 task/episode_N.json；显式指定后缺文件或内容不完整会报错 |
| 相机一致性 | 逐视角检查内外参（绝对容差 1e-4）和 mask/点云分辨率；无法配准的视角退出匹配并记录原因，其余视角继续验证 |
| 自动匹配证据 | 至少两个相机各有 16 个实例像素，mask 双向覆盖率均 ≥0.90，对应点的三维距离 P95 ≤1 cm，且 ≥95% 重合像素有有限点坐标 |
| 冲突处理 | 已配准相机的 mask/几何证据有矛盾、对应关系不唯一、多对一、部件缺失均拒绝；自动匹配仍要求至少两个相机支持每个部件 |

优先读取显式文件，其次读取 demo[0].misc.oracle_handle_metadata；都没有时才尝试上述
已标定多视角的 mask 对应。这是有几何证据的编号配准，**仍需检查真实数据的 audit**，
并不等价于原始采集时记录的身份真值。manifest 用 source 区分 acquisition_metadata 与
registered_masks；T/R 语义仍由任务配置决定。

例如 wrist 外参在 live reset 与 stored 第 0 帧之间不一致时，该视角不能做逐像素匹配；
可由其余已配准视角继续建立映射。evidence._registration 记录 used_cameras、
excluded_cameras，以及标定矩阵的最大元素差值。排除后少于两个有效视角会明确报错。
这里排除仅影响首帧编号配准；映射建立后，保存数据的四视角仍参与 T/R 点云提取。
有原始采集映射时沿用元数据路径的检查规则，至少保留一个已配准视角。

原始映射文件使用以下格式（数值仅示例，必须替换成采集时真实 ID，包含所有所需可渲染子部件）：

```json
{
  "schema_version": "rlbench_name_to_handle_v1",
  "task": "close_jar",
  "episode_idx": 0,
  "variation": 4,
  "name_to_handle": {"jar_lid0": 12345, "jar0": 12346}
}
```

映射元数据必须来自对应 episode；程序检查 task/variation/episode/schema，并拒绝可见区域
与声明矛盾的映射。对于完全不可见部件，依赖采集元数据的真实性。失败证据即使
ORACLE_DEBUG=0 也写入 semantic_oracle/handle_alignment/task/episode_N.json。

每个成功 episode 的 manifest 立即原子落盘；后续 episode 失败不会丢失之前已完成的文件。
manifest 的 handle_namespace=stored，handle_alignment 保存 live_to_stored 和各视角证据。
离线重写器直接使用 stored handles，并检查原始第 0 帧 mask 的 SHA-256 指纹，防止对另一份
数据应用映射。生成与重写必须使用相同 raw 数据及 mask 分辨率。无需重新生成 baseline replay。
source_alignment_validated 表示相应检查通过；它不替代真实 simulator episode 的验收。

对 18 个任务可把 `TASKS` 设为 `finetune/bridgevla/utils/rvt_utils.py` 中的完整任务列表。
若 expert keypoint 数超过 `EPISODE_LENGTH`，离线重写器会拒绝不完整 manifest，不能静默
沿用最后一个 phase。每个 checkpoint/task 的输出位于：

- `.../eval/<task>/rlbench_gt/<model>/semantic_oracle/semantic_role_manifests/<task>/episode_N.json`；
- `oracle_provider_stats.json`：区分 `mapping_errors`、`not_visible_*` 和 `no_reference`；
- `semantic_role_audits/<task>/episode_N/role_audit_step_000.png`：首帧四视角 overlay、
  原图、instance/T/R mask、三正交 T/R 点云以及 phase condition 状态。

## 2. 只重写 Oracle 字段，生成 semantic-GT buffer

以下命令在仓库根目录执行；若刚运行完上一节，请先返回仓库根目录。

```bash
python tools/rewrite_replay_with_semantic_roles.py \
    --replay-dir LPY/BridgeVLA_RLBench_TRAIN_Buffer \
    --raw-data-dir LPY/BridgeVLA_RLBench_TRAIN_DATA/train \
    --manifest-dir /path/to/model/eval \
    --output-dir LPY/BridgeVLA_RLBench_SEMANTIC_GT_Buffer \
    --task all \
    --max-objects 32 \
    --num-points 512 \
    --cache-frames 128 \
    --cache-episodes 2 \
    --resume
```

工具保留 action、图像、点云、语言、`episode_idx/sample_frame` 和其他 baseline 字段；只
替换六个 Oracle tensor，并增加不输入网络的审计字段：schema version、phase ID、T/R
semantic name、kind、原始 handle 集合、`oracle_phase_source` 及各角色 valid。输出中的 T/R 使用固定小 slot ID
`0/1`，不会把上千万的 simulator handle 当作显示 ID；真实 handle 仍保存在 audit 字段。

严格行为如下：

- live mask 与保存 mask 在相同 manifest frame 的 handle 体系不一致：立即停止并报告
  `mapping_error`，禁止用邻近实例代替；
- 角色正确但当前四个相机均不可见：该角色 `valid=False` 并计入 `not_visible`；
- 任务定义没有 R：计入 `no_reference`，不是异常，网络的 R residual 为零；
- raw/replay frame 越界：立即停止，不截断到最后一帧，也不生成伪点云；
- `--resume` 只跳过已经原子写完的 replay；`--overwrite` 与它互斥。
- `--cache-frames` 与 `--cache-episodes` 都是有界 LRU；默认最多保留 128 个 Oracle
  帧和 2 个 episode 的 low-dim 数据，不会随已处理 episode 数持续增长。

## 3. 正式 semantic-GT O2 训练

```bash
cd finetune/RLBench
bash train.sh \
    --exp_cfg_path configs/rlbench_o2_semantic_gt.yaml \
    --train_replay_storage_dir /path/to/BridgeVLA_RLBench_SEMANTIC_GT_Buffer \
    --init_checkpoint /path/to/baseline/model_80.pth \
    --train_oracle_adapter_only
```

`rlbench_o2_semantic_gt.yaml` 设置 `oracle_semantic_audit=True`；旧启发式 buffer 必须继续
使用 `rlbench_o2_gt_instance.yaml`（audit schema 默认关闭）。两类 buffer/checkpoint 不应
混在同一实验目录。semantic mapping 是 privileged GT，结果只能解释为 Oracle 上界。

训练模式、消融和评估见 [O2 实验](../experiments/o2-training.md)。
