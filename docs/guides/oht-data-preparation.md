# OHT 数据准备与分割标注

本页覆盖原始数据到 V4、标注刷新和预览。训练入口见 [OHT README](../../finetune/OHT/README.md#3-训练)，深度/相机/ROI 参数见[数据诊断](oht-data-diagnostics.md)。命令从 BridgeVLA 根目录执行。

## 选择路径

| 当前数据 | 操作 |
| --- | --- |
| 已有完整 `replay-source-v4` | 直接训练；代码或动作头更新不要求重建数据 |
| 只有原始 `disassemble_left-seg/lerobot_dataset` | 使用下面的完整生成脚本 |
| 普通非分割 OHT 原始数据 | 使用[手动 V4 构建](#从原始数据手动构建-v4) |
| V4 已完整，只需修改 T/R 映射 | [刷新分割标注](#只刷新标注与预览)，复用稠密动作 |
| 运行旧单关键点或 role teacher | 使用[单关键点指南](oht-single-keypoint.md)的 V3 路线 |

## 从分割原始数据生成 V4

```bash
ROOT=/common-data-32t/data/robot_data/oht_curobo_pd_v423-seg
OUT="$ROOT/bridgevla-seg-v4-run01"
bash finetune/OHT/prepare_seg_data.sh \
  --root "$ROOT" --output "$OUT" \
  --chunk-horizon 16 --action-frame-stride 1
```

`ROOT` 可以只含原始数据，无需已有 audit、replay 或 `contract.json`。`OUT` 必须为新目录；实际挂载在 `/media/34.92/...` 时只替换 `ROOT`。任务目录 `disassemble_left-seg` 会规范为 `disassemble_left`，只有该任务也可运行。新旧同名任务目录不能混在同一根目录，也不能按 episode 编号给旧轨迹补分割。

流水线依次执行 **audit → V4 replay → 完整校验 → 分割角色标注 → 角色预览**。任一步失败即停止；`pipeline.json` 记录执行状态和失败阶段。启动会打印 root、解释器、output 和 replay 路径。训练读取 `$OUT/replay-source-v4`，不是 `ROOT` 或 `OUT` 本身。

| 常用参数 | 默认值 / 含义 |
| --- | --- |
| `--config` | [dataset.yaml](../../finetune/OHT/configs/dataset.yaml)，可替换为已核查的本地标定配置 |
| `--mapping` | [seg_roles.yaml](../../finetune/OHT/configs/seg_roles.yaml)，保存 T/R 映射 |
| `--chunk-horizon` / `--action-frame-stride` | 16 / 1；每个输入监督未来连续 16 个原始帧 |
| `--sample-stride` | 10；只控制输入观测抽样，与未来动作间隔不同 |
| `--visualize-every` | 100；按每个 episode 生成样本数抽取预览，0 关闭两个预览 |
| `--workers` / `--point-count` | 4 / 512；校验线程数 / 角色点数设置 |
| `--dry-run` | 仅打印命令，不创建输出或解码数据 |

脚本在 `OUT` 保存 `dataset.yaml` 和 `seg_roles.yaml` 快照。仓库配置更新不会修改旧快照。可用 `PYTHON_BIN=/path/to/python` 指定解释器，或直接运行 `python tools/prepare_oht_seg_data.py`。

## 从原始数据手动构建 V4

普通 OHT 数据或需要分步运行时使用此路径；已完成上述流水线时无需重复执行。

```bash
RAW_ROOT=/common-data-32t/data/robot_data/oht_curobo_pd_v423
MANIFEST="$RAW_ROOT/audit-seed0.json"
REPLAY="$RAW_ROOT/replay-source-v4"

python tools/audit_oht_dataset.py \
  --root "$RAW_ROOT" --output "$MANIFEST" --seed 0

python tools/build_oht_replay.py \
  --root "$RAW_ROOT" --manifest "$MANIFEST" \
  --config finetune/OHT/configs/dataset.yaml \
  --output "$REPLAY" --replay-version 4 \
  --chunk-horizon 16 --action-frame-stride 1 \
  --sample-stride 10 --visualize-every 100

python tools/validate_oht_replay.py --replay "$REPLAY" --workers 4
```

原始数据未变且已有有效 audit 时，复用 `MANIFEST` 并跳过第一条命令；audit 和 replay 工具均拒绝覆盖已有输出。原始数据是各任务 `lerobot_dataset` 的父目录，不能把旧 replay 当作 `--root`。单任务数据的其他任务缺失提示属预期，坏 episode 仍需检查 audit 报告。

audit 按任务目标比例 80/10/10 固定组划分，同一轨迹及关联场景不跨 train/val/test；无成功标记或明确失败的 episode 排除并报告。构建保留这份划分，不因切换动作头重新随机分组。

V4 从原始 Parquet 和 RGB-D 视频重新构建，不复制 V3 observation。动作标签来自实测 TCP、关节与因果滤波夹爪，不使用 raw action 命令或关键点之间的直线插值。未来时刻为 `t+dt ... t+K*dt`，实际 `dt` 从原始时间戳推导并写入 contract；也可用 `--action-dt` 指定期望值。不同采样率、时间缺口和粗采样丢失完整夹爪事件的窗口会被拒绝或跳过并记录。

## 缓存内容与完整性

```text
OUT/
  audit.json
  pipeline.json
  dataset.yaml
  seg_roles.yaml
  replay-source-v4/
    contract.json
    samples.jsonl
    complete.json
    observations/<task>/<episode>/<frame>.npz
    targets/<task>/<episode>/<frame>.npz
    preview/index.html
  seg-role-annotations/
  role-preview/index.html
```

上面是完整生成脚本的目录；手动构建只生成指定的 replay。观测与目标文件独立保存并绑定哈希：

| 字段 | 含义 |
| --- | --- |
| `tcp_pose`、`joint_positions`、`gripper_open01` | 当前实测状态，只属于 observation |
| `key_action`、`key_valid` | 下一关键目标及其监督有效性 |
| `action_chunk[K,8]` | 未来实测 world EEF / xyzw / 二值夹爪 |
| `joint_chunk[K,7]` | 同批未来关节/夹爪标签；不表示已实现关节 chunk 模型 |
| `action_frames`、`action_timestamps`、`action_valid` | 实际未来帧、时刻及有效 mask |

没有下一关键目标的尾段仍可保留，`key_valid=false` 仅屏蔽关键目标损失。尾部不足 K 项通过 `action_valid` 屏蔽，不复制最后帧制造停留。真实静止、旋转和夹爪动作保留原时序。

训练缓存必须有有效的 `contract.json`、索引和 `complete.json`。失败目录没有完成标记，应换新目录重跑；不要手写 contract 或 complete 绕过校验。完整校验读取全部划分和全部 NPZ，是 CPU/磁盘任务；共享存储上可将 `--workers` 降至 1 或 2。

## 分割角色标注

`disassemble_left` 全部阶段固定 **target=左轮 ID 1，reference=放置盘 ID 4**。

| ID | 语义 |
| --- | --- |
| 0 | robot |
| 1 / 2 | 左轮 / 右轮 |
| 3 | wheel_stack |
| 4 | plate |
| 5 / 6 | 左轴 / 右轴；原始名字 shaft_1 / shaft_2 |
| 255 | 背景或未标注 |

这是语义分割，不能区分同类别多个实例。导出器逐帧读取实际 `observation.seg.*` 引用，检查流长度、时间戳、原始分辨率和灰度 uint8 ID，再按 RGB-D 相同整数步长缩采样，不通过插值产生标签。

输出包括各视角 bool mask NPZ、`annotations.jsonl` 和绑定视频/映射/replay 的 `provenance.json`。角色存在但完全遮挡时保留 present/known，空 mask 表示无可见几何；`[]` 仅用于已确认语义为空，`unknown` 表示未确认。通用映射可按当前帧 `instruction_id` 配置 phases，不读取未来阶段。

**当前 V4 chunk 训练不接收 role cache。** 这些 mask 用于标注检查和预览；单关键点 role_queries 的 teacher 路线见[物体辅助](oht-single-keypoint.md#内部角色预测)。GT mask 不进入在线策略观测。

## 只刷新标注与预览

已有完整 V4 时，修改映射只需重新导出标注和预览，稠密动作与 observation 可直接复用：

```bash
ROOT=/common-data-32t/data/robot_data/oht_curobo_pd_v423-seg
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

打开 `$OUT/role-preview-wheel-plate/index.html`。已按正确映射导出时只运行第二条并指向已有 annotations；输出使用新目录。页面将 target 画成绿色、reference 蓝色、重叠黄色，并保留橙色实测未来轨迹。构建时的 `replay-source-v4/preview/index.html` 不含后续导出的角色 mask。

只刷新稠密 GT 轨迹的 PNG/HTML，不需要原始视频：

```bash
python tools/export_oht_replay_viewer.py \
  --replay "$OUT/replay-source-v4" \
  --output "$OUT/chunk-preview-refresh-01" --every 100 --max-points 40000
```

轨迹叠加在 sensor RGB 和全局/局部点云投影上。青色是当前 TCP，品红色是下一 GT key，橙色是有效未来 chunk；padding 不画。它们是数据预览，预测与 GT 的对比图由训练或离线评估生成。更多对象点云、ROI 和未完成缓存诊断见[数据诊断](oht-data-diagnostics.md)。
