# OHT 单关键点与物体辅助

本页适用于旧单动作实验。连续 chunk 的当前运行命令见 [OHT README](../../finetune/OHT/README.md#3-训练)。所有命令从 BridgeVLA 根目录执行。

| 路线 | 配置 | 数据 |
| --- | --- | --- |
| 单关键点 EEF baseline | [oht_full_8x40.yaml](../../finetune/OHT/configs/oht_full_8x40.yaml) | 单关键点 replay，常用目录名 `replay-source-v3` |
| 内部 T/R role queries | [role_queries.yaml](../../finetune/OHT/configs/role_queries.yaml) | 同上，加 role teacher |
| 外部 predicted objects | [predicted_external.yaml](../../finetune/OHT/configs/predicted_external.yaml) | 同上，加预测缓存及真实 predictor |

这些路线不接受 V4 chunk replay。目录名 V3 对应现有单关键点 schema `oht_bridgevla_v2`；不要通过修改 contract 让两种缓存混用。

## 单关键点数据

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
- 本段单关键点版本使用 `--object-cloud-config` 启用对象截取和 HTML 对象预览；只传 `--visualize-every` 时仅生成基础 PNG。V4 版本开启预览即可生成基础 HTML 入口，见[数据准备](oht-data-preparation.md)。
- 示例省略 `--visualize-output-dir`，所有诊断文件统一放在 replay 下的 `preview`；显式指定该参数会将 PNG/HTML/入口输出到指定的新目录，对象数据仍保存在 replay 下。

旧 PNG/HTML 不会随代码更新自动变化。请打开本次新目录；复制到本机时保留整个 `preview` 目录及相对路径。

若旧版本在首帧报 `FileExistsError: Use a new object cloud directory: .../preview/...`，是 PNG 先建目录与对象导出冲突。更新到包含写入顺序修复的代码后，用新的 replay 输出目录重新构建；失败目录没有 `complete.json`，不能用于训练。无需重新审计原始数据。



分割原始数据也可直接走单关键点流水线，自动导出 mask 和 role teacher：

```bash
ROOT=/common-data-32t/data/robot_data/oht_curobo_pd_v423-seg
bash finetune/OHT/prepare_seg_data.sh \
  --root "$ROOT" --output "$ROOT/bridgevla-seg-v3-run01" --replay-version 3
```

此时后续训练的 replay 改为该输出下的 `replay-source-v3`，内部角色训练的 role cache 改为 `role-teacher`。

## 单关键点训练

```bash
OHT_REPLAY=/common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3
PRETRAIN=/home/yiwei/project/BridgeVLA/checkpoints/pretrain

GPUS_PER_NODE=2 bash finetune/OHT/train_8x40.sh \
  --config finetune/OHT/configs/oht_full_8x40.yaml \
  --replay "$OHT_REPLAY" --pretrain-path "$PRETRAIN" \
  --log-dir exp/OHT --exp-id oht-keypoint-eef-s0
```

当前 full 配置为 60 epochs、`train_iter=40000`、全局 batch 16、每卡 batch 8、AdamW 学习率 `8e-5`。冻结视觉塔、projector 和 Gemma 前 6 层，启用场景/EEF 联合变换增强。两卡自动累积 1 次，单卡 2 次；更多卡须在新配置中减小每卡 batch。epoch 是任务均衡、有放回采样的预算，不保证一轮恰好遍历数据一次。

`baseline.yaml` 保留早期按优化步训练的配方，`smoke.yaml` 是两步 GPU 短测。它们不是上述 full 预算。其他动作表示及关闭增强的要求见[动作模式指南](oht-action-modes.md)。初始化、严格 resume、TensorBoard 和输出目录规则见[训练说明](oht-training.md)。

## 内部角色预测

[role_queries.yaml](../../finetune/OHT/configs/role_queries.yaml) 复用现有两角色预测器：由当前渲染特征与任务上下文预测 Target/Reference，在 coarse/refine 间继承几何与角色 token。教师监督角色图、Reference NULL 及可见表面上的 T/R 分离，推理时不输入 GT 对象。

`role_loss.diversity: 0.01` 控制 coarse 分离项，`role_loss.refine_diversity: 0.01` 控制启用 `object.refine_role_masks: true` 后的局部分离项（该 head 默认关闭）。在可靠 Reference 表面抑制 Target 响应，反向同理；仅监督当前渲染可见、几何一致且角色 footprint 不重叠的点，不把 NULL、遮挡或未知区域当负例。TensorBoard 中分别查看 `object_slot_diversity_loss` / `object_refine_diversity_loss` 和对应的 `*_diversity_supervised_pixels`；像素数为 0 时 loss 为 0 属正常跳过。角色允许共享同一对象不同区域时应关闭分离项或明确互斥标注，具体边界见[分离监督说明](object-conditioning.md#tr-分离监督与日志)。

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

未知和无可见几何不产生伪表面监督。教师构建器不会从 objects_pos/held_obj/instruction_id 自动猜测语义，未提供 mesh 表面补全；分割采集数据可用[语义 ID 导出器](oht-data-preparation.md#分割角色标注)生成标注。

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

## 外部物体预测

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

## 评估与部署

```bash
RUN=/path/to/actual/keypoint/run
python -u -m finetune.OHT.eval open-loop \
  --checkpoint "$RUN/model_last.pth" --replay "$OHT_REPLAY" --split val \
  --output /path/to/new/keypoint-val.json
```

部署统一使用 [VLA server 与 IsaacLab 指南](oht-vla-server.md)。单关键点 policy 采用固定目标到位后开闭夹爪；GT 逐帧 replay 保留原时序，细节见[关键点执行说明](oht-waypoint-execution.md)。旧 `/act` 协议和自定义适配器见[服务设计](../design/oht-vla-server.md)与[动作模式指南](oht-action-modes.md)。
