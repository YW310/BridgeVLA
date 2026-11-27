# OHT 模型、训练与评估

[文档索引](../README.md) · [数据准备与契约](data.md) · [服务与执行](runtime.md)

先理解数据流与张量维度，可看[交互算法总览（HTML）](../design/algorithm-overview.html)，切换到 Chunk 回归或 Flow；对象条件分支与当前 V4 chunk 尚未接通。

当前主线为 **V4 → 关键目标 + 连续 EEF chunk → 离线评估 → IsaacLab**。所有命令从 BridgeVLA 仓库根目录执行，替换示例路径。数据构建/标注见 [data.md](data.md)，HTTP、客户端、GT 回放和 waypoint 执行见 [runtime.md](runtime.md)。

基于当前结果的后续研究动机、候选方向与实验门槛统一见[设计讨论](../design/role-relation-prior.md#研究动机与候选方向)；本页维护已实现模型与运行方法。

## 真机装轮训练与评估

真机 `assemble_wheel` 复用 V4 联合关键目标与 EEF chunk 训练。周期关节编码使用 `sin_cos`。从下列仿真 checkpoint 初始化时会加载 encoder，并重新初始化周期关节输入的 chunk decoder。真机夹爪使用源包测量端点归一化后的 `open01`。相机、标签和划分约定见[真机装轮数据](data.md#真机装轮数据)。

### 训练入口

| 配置 | 训练设置 |
| --- | --- |
| [16 GB 单卡](../../finetune/OHT/configs/oht_real_chunk_v4_periodic_1x16.yaml) | batch 1，累积 4 个 microbatch。更新 key head 的输出层和整个 chunk decoder，配置预算 1,000 次优化器更新 |
| [8×40 GB](../../finetune/OHT/configs/oht_real_chunk_v4_periodic_8x40.yaml) | 继承原周期编码联合训练设置和冻结范围。使用 `train_chunk_repair.sh`，通过 `CONFIG` 显式选用真机预设 |

```bash
# REPLAY 指向已完成并校验通过的真机 V4 replay
REPLAY=logs/oht-real-v4-20261127/pilot-lineage-replay
INIT=finetune/OHT/runs/oht_chunk_v4_delta_state_tcp_guided_8x40/model_epoch_30.pth
RUN=finetune/OHT/runs/oht-real-wheel-v4-periodic-1x16
CONFIG=finetune/OHT/configs/oht_real_chunk_v4_periodic_1x16.yaml

python -m finetune.OHT.cli validate --replay "$REPLAY" --workers 4
python -m finetune.OHT.train --config "$CONFIG" --replay "$REPLAY" --validate-only
python -m finetune.OHT.train --config "$CONFIG" --replay "$REPLAY" \
  --init-checkpoint "$INIT" --data-checksum once --output "$RUN"

python -m finetune.OHT.eval open-loop --checkpoint "$RUN/model_last.pth" \
  --replay "$REPLAY" --split val --samples-per-episode 24 --visualize-every 0 \
  --output "$RUN/val-open-loop.json"
```

只跑短测时，可在同一配置目录中新建 YAML，继承 `oht_real_chunk_v4_periodic_1x16.yaml` 并覆写 `optimizer_steps: 8`。八卡入口使用 `CONFIG=finetune/OHT/configs/oht_real_chunk_v4_periodic_8x40.yaml GPUS_PER_NODE=8 bash finetune/OHT/train_chunk_repair.sh "$REPLAY" "$RUN" "$INIT"`，可附加 `--check-only` 做 CPU 预检。短测或换卡应使用自己的新输出目录。评估 replay 须与 checkpoint 的契约一致，重分组后训练和比较都应使用新契约。

### 2026-11-27 单卡短测

在 RTX 5080 上用修正版 pilot 训练了 8 次优化器更新，每次累积 4 个真实观测。key 输出层和 decoder 均得到更新，PyTorch 峰值预留显存为 9.45 GiB。这个结果覆盖 16 GB 单卡的短训和权重重载。正式预算和八卡训练仍待实测。

同一个第 8 步 checkpoint 在 val/test 的每段录像中等距抽样 24 个观测。H4 对应名义 80 ms，表中误差对前 4 个有效动作步取平均。基线对每个动作步均保持当前 TCP 位置。

| split | 录像数 / 抽样观测数 | H4 平均位置误差 | 保持当前位姿 | 有标注的 key 预测越出工作区 |
| --- | ---: | ---: | ---: | ---: |
| val | 5 / 120 | 4.26 mm | 3.31 mm | 45 / 86 |
| test | 6 / 144 | 4.30 mm | 3.40 mm | 61 / 104 |

两组的 H4 位置误差均高于保持基线，测试集 key 平均位置误差为 0.76 m。两组的 H4 抽样均未包含目标开闭状态与当前状态不同的预测项。完整 16 步覆盖名义 320 ms，val/test 分别有 3/19 个此类预测项。pilot 的留出录像仍共享部分采集时的辅助参考，实机闭环成功率尚未测量。报告、训练日志和轨迹图位于本机 `logs/oht-real-v4-20261127/lineage-check/`。

## 2026-11-25：周期关节编码与训练入口修复

### 新实验使用连续的周期关节特征

新增 `chunk_policy.joint_encoding: sin_cos`。六个实测关节角直接按弧度计算
`sin(q)`、`cos(q)`，不先换到离训练中心最近的 `2π` 分支，因此没有该标量分支的切口。
训练、离线评估和 server 共用 `SpatialChunkDecoder` 的同一个状态编码函数，
编码方式保存到 checkpoint 配置，并由 server 的 `/model_info.chunk_policy.joint_encoding` 公布。

HTTP/replay 仍传原始 `TCP7 + q6 + gripper1` 共 14 个状态量；decoder 的状态投影输入改为
`identity_TCP7 + sin(q6)6 + cos(q6)6 + normalized_gripper1`，共 20 维。
`identity_TCP7` 是局部状态 token 的固定原点/单位姿态；实测 TCP 仍参与局部几何和动作还原。
周期特征不使用关节分位数的 center/scale，夹爪仍按已保存的夹爪统计归一化。
该编码把 `q` 与 `q + 2π` 视为同一周期位置，不表达机械臂绕圈次数；原始关节角仍用于
FK、控制和硬/软限位检查，模型不会改写这些实测值。

未声明 `joint_encoding` 的旧 checkpoint 默认 `scalar`，完整保留原有标量归一化和分支对齐语义。
新配置只适用于 spatial chunk。编码变化会改变输入含义和投影维度，不能用旧 run 的 `--resume`；
`--init-checkpoint` 只保留 encoder，整个 decoder 重新初始化，并在启动日志中打印原因。
这不是将旧 decoder 权重静默套进新输入，也不会让旧 epoch-30 checkpoint 自动获得修复。

新增 `oht_chunk_v4_delta_state_tcp_guided_periodic_8x40.yaml` 继承 calibrated 配置，
只增加周期编码，保留原 loss/LR，作为编码修复的独立对照。
`oht_chunk_v4_delta_state_tcp_guided_repair_8x40.yaml` 也启用周期编码，并保留下面的
motion loss、学习率和梯度裁剪候选。关节编码修正不要求重采全部原始轨迹；此前夹爪标定修正
仍须使用新建、全量校验通过的完整 replay。本次没有重建 replay，也没有启动训练。

这修复了已确认的输入切口，不等于已经解决早期渐进偏离或完成完整任务。
本次没有新增恢复示范、视觉增强或控制器改动，这些仍须分别验证。
验证结果：周期编码专项 36 项通过，完整 OHT 回归 1658 项通过、10 项跳过。
真实保存状态的边界特征变化从约 2.453 降至 0.000200；旧 epoch-30 的 decoder 输出保持兼容。
这些是编码与接口正确性检查，没有训练新的周期编码策略，也不代表新策略闭环任务通过。

2026-11-26 后续分析将默认训练入口收敛为 **periodic-only**：同一标定数据上只改变已确认
有缺陷的关节编码，保留原 loss/LR；组合 repair 通过 `CONFIG=...repair_8x40.yaml` 显式选择。
在 frame80/120 的偏离态中，仅由当前姿态变化运输原局部运动造成的 H4 位置差为
0.230/0.464 mm，模型局部增量改变项为 1.606/2.051 mm。两组真实短闭环的全部关节均未发生
换圈，说明早期预测恢复不足仍独立存在，不能认为修完周期编码就已解决。该分解是算术归因，
不是正确恢复动作的 oracle；不能据此认定某个 loss 是唯一原因。

### 先前诊断与训练候选

后续恢复诊断进一步缩小了问题范围。原 epoch-30 在四组一秒真实反馈对照中，
从示教 frame80/120 启动的最近路径误差约 1.5/2.8 mm；恢复原闭环偏离状态后，
误差由约 16.6/32.3 mm 增至 43.6/76.7 mm。两组偏离初态的前 16 个控制 tick
均未使用惯性正则化、平均对模型目标的跟踪误差仅 0.217/0.207 mm，但路径偏差已经增加。
这证明在这组初态下，预测目标与反馈恢复存在问题，不能归咎于单纯跟踪失败；
它没有证明某个 loss 是唯一根因。后半段仍有惯性正则化与非零跟踪误差，控制耦合尚在。
episode 0 属于训练集，四个短时 trial 不是 held-out 泛化或完整任务成功率。

训练修复还须检验对偏离状态的恢复监督与视觉稳定性。仅调 loss、学习率或裁剪梯度，
不保证学到未覆盖的恢复行为。恢复数据应包含同步的实际 RGB-D、关节/TCP 与可执行的
纠正轨迹，标签仍遵循实测后续 TCP 语义，不能用不一致的观测或未经执行验证的动作充当恢复样本。
目前没有生成恢复训练集，也没有启动新训练。

继续使用 K=16、H=4 的 delta-pose chunk；执行时只用模型输出，不改成关键目标规划器。
新的 `oht_chunk_v4_delta_state_tcp_guided_repair_8x40.yaml` 继承 calibrated + motion 配置，
使用上述 20 维周期状态投影，保持 TCP crop、真实 60 Hz 动作标签和联合 key/chunk 训练，另外把学习率降到
2e-5，并在完整梯度累积、DDP 同步后以 `max_grad_norm=1` 裁剪。非有限梯度会在 optimizer
更新前报错，日志 `chunk_grad_norm_before_clip` 记录裁剪前范数。旧配置默认不裁剪，旧模型推理不变。
这些是待训练、待验证的修复候选，不能据此宣称新策略已经正常完成任务。

旧目标的姿态 chordal MSE 未按角度尺度归一化；1° 的单轴误差约为 6.77e-5，
motion 配置按 1° 缩放后约为 0.222。尺度变化约 3283 倍，因此单独提供更保守的优化器配置。
这说明旧目标对小角度误差的惩罚弱，不能单凭损失公式认定它是闭环失败的唯一原因。
旧 replay 的训练锚点中，H=4 末点位移 ≥1 mm 的比例约 94.67%，同时位移 <1 mm 且
姿态变化 <1° 的比例约 5.33%；目前没有证据把失败归因于“大部分样本静止”，本次不加静止/运动重采样。

本次实际对照：稳定 OSC 的 GT 回放完成 2028 个控制 tick（33.8 s），平均位置跟踪误差
4.00 mm、平均姿态误差 0.291°；原 epoch-30 模型实时闭环 H=4 在约 17.9 s 阻塞，
H=16 在 4.45 s 因跟踪误差停止（末次请求时刻为 4.27 s）。这次单集 H=16 对照未改善轨迹，
不能据此推广到所有执行频率。GT 跟踪通过也不等于全部控制问题已解决或模型任务通过。
原测试约 2 s 时 TCP 已偏离整条示教空间路径 32.32 mm，当前前缀对模型目标的末点跟踪误差
仅 0.142 mm；首次请求力矩超限在 2.45 s。已发现的控制异常不能单独解释最初的偏离。
起始偏离的原因仍未隔离，loss、学习率和梯度裁剪均是未验证候选；单元测试通过不能证明其任务收益。
完整原始审计为 100 个有效 episode；按用户要求已停止新 replay 构建，没有启动训练。
`logs/oht-calibrated-replay-20261125/replay-source-v4` 是未完成产物，禁止用于训练。

后续执行时，先用上面的重建脚本生成一个**新的**完整 replay，再运行以下入口。
8×40 GB 是训练预算；当前单张 16 GB 仿真卡未验证能运行联合训练。

```bash
conda activate bridgevla
# 换成下一次重建成功、存在 complete.json 的目录；不要填写 raw 根目录。
REPLAY=finetune/OHT/replays/seg-calibrated-v4-20261125/replay-source-v4
INIT=finetune/OHT/runs/oht_chunk_v4_delta_state_tcp_guided_8x40/model_epoch_30.pth
NEW_RUN=finetune/OHT/runs/oht_tcp_guided_periodic_v1

# 仅展示命令，不读取 replay、不训练。
bash finetune/OHT/train_chunk_repair.sh "$REPLAY" "$NEW_RUN" "$INIT" --dry-run
# CPU：验证所有 split、完整性、端点、配置、训练归一化；不构建模型、不训练。
bash finetune/OHT/train_chunk_repair.sh "$REPLAY" "$NEW_RUN" "$INIT" --check-only
# 在相应训练机执行：重复检查通过后才启动新训练。
GPUS_PER_NODE=8 bash finetune/OHT/train_chunk_repair.sh "$REPLAY" "$NEW_RUN" "$INIT"
```

入口使用 `PYTHON_BIN` 对应环境里的 `torch.distributed.run`，避免误用其他 conda 环境的
`torchrun`。默认 periodic 配置只改变关节编码。组合 loss/LR/梯度裁剪实验须使用独立的新 run：

```bash
CONFIG=finetune/OHT/configs/oht_chunk_v4_delta_state_tcp_guided_repair_8x40.yaml \
  bash finetune/OHT/train_chunk_repair.sh "$REPLAY" \
  finetune/OHT/runs/oht_tcp_guided_repair_periodic_v1 "$INIT" --check-only

# 后续在相应训练机执行；本次未执行训练。
CONFIG=finetune/OHT/configs/oht_chunk_v4_delta_state_tcp_guided_repair_8x40.yaml \
  GPUS_PER_NODE=8 bash finetune/OHT/train_chunk_repair.sh "$REPLAY" \
  finetune/OHT/runs/oht_tcp_guided_repair_periodic_v1 "$INIT"
```

仍可用 `CONFIG=...calibrated_8x40.yaml` 跑仅修数据的旧标量编码对照，
或 `CONFIG=...motion_8x40.yaml` 跑旧标量编码的 motion 对照。`--check-only` 不加载大 checkpoint，
权重兼容性仍在实际初始化时检查。新的归一化若改变 decoder 输入/输出语义，现有逻辑会重建 decoder，
保留 encoder，并输出初始化原因；编码切换同样重建整个 decoder，不强行复制旧归一化或伪装成恢复训练。

验收顺序：新 replay 全量校验 → 完整 held-out episode 的 H=1/4/16 位置/姿态与静止基线对照
→ 同一 native 环境的 GT 回放 → 实时五相机闭环。真实闭环须检查轮子是否脱轴、搬运并释放到盘上，
同时保留阻塞/跟踪失败和关节力矩日志；仅没有异常或提前停止不算任务通过。

### 与旧单关键点模型的同观测对照

`baseline-eef-8gpu-s0/model_last.pth` 是下一夹爪事件的绝对 EEF 关键目标模型，
实际完成 100 epoch / 20,800 optimizer steps。用户反馈是轨迹更稳，但未完整完成任务。
本次把它用于相同 62 份新采集的五相机 RGB-D/物理状态；两模型分别按自身 contract
重算夹爪输入并保留各自点云 ROI。比较对象为旧单目标与 epoch-30 的 `key_action`，
不把关键目标和 H=4 连续动作误差混算，也没有运行旧模型完整真实闭环。

48 个关节 ±0.01 rad 扰动样本的关键目标位置变化均值为旧模型 1.66 mm、chunk 关键点头
1.27 mm，最大值为 3.87/3.16 mm；两者姿态及夹爪输出均未改变。同姿态单次重复渲染本身
也产生最高 3.16/2.23 mm 的关键点变化，因此不能据此声称统计显著优势。
这组早期观测未显示 chunk 关键目标头明显退化；已有真实反馈失败仍发生于连续 chunk
动作/执行反馈路径，不足以独立区分 decoder、输入分布、监督覆盖和控制耦合的贡献。

旧模型默认 waypoint 保持目标直到抵达后才更新，chunk 每四个控制 tick 重预测；
执行方式可能影响轨迹观感。旧模型还使用另一批 v423 原数据、四任务、空间增强、冻结 Gemma
前 6 层，chunk 则使用 v423-seg 单任务、关闭空间增强、冻结前 16 层。没有完成统一控制器的
任务对照和训练消融前，不能将差异唯一归因于 chunk 结构或某个 loss。新训练配置仍是候选。
原始推理、配对指标和可视化保存在
`logs/oht-epoch30-closedloop-20261125/baseline_comparison/`。

### 后续机理核查：关节分支切口与执行速度

已确认一个具体的输入编码问题。`spatial_chunk.py` 将关节角对齐到离训练中心最近的
2π 分支；q6 中心约 -2.58965 rad，切口在 +0.551943 rad。原真实闭环在
3.8667 s 首次跨过该切口，4.0 s 又切回；它不能解释切口前两秒的渐进偏离。
训练数据也有少量跨切口状态，15222 个训练锚点中仅 9 个进入换圈分支；这并非纯部署时新增的编码。

实际重渲染的两份状态中，q6 仅相差 0.0002 rad，TCP 差 0.033 mm / 0.011°，
关键目标完全相同，H=4 输出却相差 14.24 mm / 3.02°。固定全部视觉和其他状态、
只换 q6 数值，H=4 仍跳 13.84 mm / 3.09°，16 步夹爪均从开变关。
仅在 CPU 机理消融中令该处 q6 特征局部连续，H=4 跳变降到 0.0022 mm / 0.00063°，
夹爪翻转消失。因此这个切口能直接造成输出不连续；对整段闭环失败的贡献仍须修复后配对实测。
局部消融 hook 破坏全局 2π 编码约定，**没有部署，也不能直接作为在线修复**。

上述诊断阶段没有更改生产编码或训练新模型。随后已加入本节开头的 `sin_cos` 正式实现与
训练配置，训练和 server 使用同一输入契约，旧 checkpoint 继续按 `scalar` 处理。
不能把临时 hook 当作修复，也不能只改旧 checkpoint 的在线归一化；新编码尚未通过新模型训练
和真实闭环任务验收，早期偏离的唯一训练原因仍未确认。

另外已完成四组关闭速度前馈的一秒真实反馈对照，模型、初始 q/TCP/夹爪和稳定 OSC 均相同。
两个偏离初态的路径距离仍由 16.58/32.32 mm 增至 21.98/37.12 mm；FF-on 对应为
43.56/76.69 mm。FF-off 的实际路径速度降至 6.25/32.38 mm/s，FF-on 为 55.30/116.19 mm/s，
目标跟踪滞后变大。较小最终偏差包含低速与滞后影响，不能称恢复修复或直接据此关闭前馈。
首次预测仍有 0.053–0.402 mm 的 H4 差异，来自少量渲染变化；这不是逐像素相同输入的重复试验。

连续支路训练使用 prediction/eval 路径，配置中的 `img_aug_2` 只在独立 key-loss 支路开启；
replay/sampler 没有专门构建模型偏离后的恢复样本。新损失配置没有自动补足这些覆盖。
本次未发现 delta encode/decode、未来帧偏移或 regression 训练/部署路由硬错误；
现有逐帧 oracle 的各相位误差接近，也不支持仅将 sample_stride=10 视为根因。
机理消融、FF 实测与训练分布审计统一保存在
`logs/oht-epoch30-closedloop-20261125/decoder_audit/index.html`。

## 2026-11-20 轨迹排查后的处理顺序

当前应分别处理控制跟踪、派生数据标签和模型预测。客户端新增
`--chunk-velocity-feedforward`：为原生固定基座 OSC 提供相邻预测目标的速度参考，
不修改 checkpoint、IsaacLab 安装文件、控制增益或停止阈值。这项控制修正无需重训。
GT 回放的复测结果和使用边界见 [runtime.md](runtime.md#移动目标-osc-跟踪)。
它不能修正模型已经给错的目标，也不等于完整任务闭环通过。

### 新数据：保留原始录制，重建派生 replay

v423-seg 中的瞬态夹爪极值约 -0.095 被旧 metadata 标为全开，正常全开约 -0.04。
采用正常物理端点 open=-0.04、close=-0.006 重建，不改原始视频、Parquet 或已有 run。
这些端点针对本次 v423-seg；其他采集版本应先确认其正常开/关平台，不能照搬瞬态极值。

```bash
cd /media/34.49/common-data-4t/project/BridgeVLA
conda activate bridgevla

PYTHON_BIN=python bash finetune/OHT/prepare_seg_motion_replay.sh \
  /media/34.49/common-data-4t/dataset/robot_data/oht_curobo_pd_v423-seg \
  finetune/OHT/replays/seg-calibrated-v4-20261120
```

脚本依次审计原始 episode、从原始 RGB-D/Parquet 重建全部观测和标签、验证 replay。
输出必须是不存在的新目录；失败后保留审计产物，重新运行时另选新目录。
保持 K=16、动作 frame_stride=1、action_dt=1/60 s；观测 sample_stride=10，另保留夹爪关键帧。
不会修改 TCP 轨迹或把动作降到 6 Hz。新的夹爪开度、二值状态、关键帧/key 标签、
低维观测及 contract 一起重建，不能只改旧 replay 的 contract.json 或 server 端点。
训练会重新拟合 train split 的动作/本体归一化，不能复制旧 run 的归一化 JSON。

本次未发现必须重新采集全部原始轨迹的证据；GT 的高速跟踪偏差应单独用执行链处理。
正常开/关平台校正也不保证消除碰撞瞬态产生的所有夹爪标签跳变，应在新 replay 中复查释放阶段。
数据重建与完整训练耗时较长，这些命令是后续执行入口，不表示已经生成新 replay 或训练完模型。

### 是否重训与推荐命令

| 改动 | 是否需要重训 |
| --- | --- |
| OSC 目标速度前馈、诊断日志、GT chunk 回放 | 不需要；客户端可使用原 epoch-30 checkpoint |
| 正常夹爪端点、观测/关键帧/action 标签修正 | 需要在新 replay 上启动新 run |
| `joint_encoding: sin_cos` 连续周期编码 | 需要新 run；旧 chunk 初始化仅保留 encoder，整个 decoder 重置 |
| motion 候选的损失尺度/前 H 步权重 | 需要单独训练和消融，尚未验证收益 |

先用仅改变数据标定的 `oht_chunk_v4_delta_state_tcp_guided_calibrated_8x40.yaml`，
保留原模型结构和所有 loss。以下命令按 8 GPU 预算配置，应在相应训练机器运行；
本次单卡 16 GB 仿真机器未验证该训练预算。

```bash
GPUS_PER_NODE=8 bash finetune/OHT/train_8x40.sh \
  --config finetune/OHT/configs/oht_chunk_v4_delta_state_tcp_guided_calibrated_8x40.yaml \
  --replay finetune/OHT/replays/seg-calibrated-v4-20261120/replay-source-v4 \
  --init-checkpoint finetune/OHT/runs/oht_chunk_v4_delta_state_tcp_guided_8x40/model_epoch_30.pth \
  --output finetune/OHT/runs/oht_tcp_guided_calibrated_v1
```

使用 `--init-checkpoint` 启动新实验，不使用 `--resume` 恢复旧数据的 optimizer/步数。
新配置会拒绝端点不符的 replay。如果新归一化使 decoder 的参数语义变化，现有初始化逻辑
会保留 encoder 并重新初始化 decoder，以启动日志为准；不能保证旧 decoder 全部复用。
旧 run、旧 checkpoint、旧 replay 均保留用于对照。

### motion 损失候选：独立消融

`oht_chunk_v4_delta_state_tcp_guided_motion_8x40.yaml` 在上述 calibrated 基线之外调整
四个训练量：位置损失尺度 5 mm、逐步位移尺度 1 mm、旋转矩阵差尺度 1°、前 H 步权重 4。
旋转仍用平滑的 chordal MSE；1° 是旋转矩阵差的缩放参数，并非姿态误差上限。
默认 `rotation_loss_scale=1` 和 `prefix_loss_weight=1` 保留旧训练目标。
新参数不改变已有 checkpoint 的推理输出，当前没有证据证明它能修好早段模型轨迹偏离。

如需比较，另开 run：

```bash
GPUS_PER_NODE=8 bash finetune/OHT/train_8x40.sh \
  --config finetune/OHT/configs/oht_chunk_v4_delta_state_tcp_guided_motion_8x40.yaml \
  --replay finetune/OHT/replays/seg-calibrated-v4-20261120/replay-source-v4 \
  --init-checkpoint finetune/OHT/runs/oht_chunk_v4_delta_state_tcp_guided_8x40/model_epoch_30.pth \
  --output finetune/OHT/runs/oht_tcp_guided_motion_v1
```

离线 chunk 报告新增 `prefix_motion`，分别汇总 1/H/K 步的米制/角度误差、
保持实测位姿的基线，以及 `translation_m_skill_vs_stationary` / `rotation_degrees_skill_vs_stationary`。
skill = 1 − 模型平均误差 / 静止基线平均误差；正值优于基线，负值更差，基线接近零时为 null。
H 步误差按有效目标点逐点计算后平均，不是 H 步误差之和。
不能只用毫米误差或完整 K 均值验收；还需覆盖完整 episode、移动/转向/释放阶段，
完成 GT 控制回放、录制与实时观测对齐检查和真实闭环任务评分。
`--limit` 仅取 split 前 N 个观测，不代表均匀覆盖全部轨迹。

| 阅读目标 | 入口 |
| --- | --- |
| 已解决的配置、核心修改及排查经验 | [已确认基线与交接](#已确认基线与交接) |
| 状态条件、动作表示、损失与归一化 | [连续轨迹模型](#连续轨迹模型) |
| 启动、预算、续训、日志和评估 | [训练与评估](#训练与评估) |
| 旧单关键点、内部/外部对象辅助 | [单关键点与物体辅助](#单关键点与物体辅助) |
| EEF、相对 EEF、关节与相对关节头 | [单动作表示](#单动作表示) |

## 已确认基线与交接

更新日期：**2026-11-02**。2026-10-29 用户反馈 [oht_chunk_v4_regression_state_8x40.yaml](../../finetune/OHT/configs/oht_chunk_v4_regression_state_8x40.yaml) 的轨迹预测已有改善；后续在线测试仍报告抓取前碰边、动作抖动或跳变。当前将其作为回归调试基线，闭环任务成功尚未完成验收。

这是用户对实际运行的确认，反馈未附成功 run 路径、checkpoint 或完整定量报告，复现时应补存。结论不扩展为所有任务/样本的误差统计或闭环成功率；`delta_state`、`flow_state` 尚未获得相同确认，也没有独立消融可以将效果全部归因于下列某一项。

### 核心修改

| 修改 | 实际作用 |
| --- | --- |
| 确定性回归 | `decoder_type: regression`，直接预测 K 个未来 pose；每点相对同一次观测的 TCP，最终还原 world 绝对 EEF |
| 本体归一化与逐层注入 | train 拟合 q6，实测夹爪采用固定映射；状态进入初始动作 queries 和每层 AdaLN |
| 门控关键目标 | 相对 TCP 的预测 key 经可学习 gate 注入，初值 0.1；保留 coarse/refine 特征及预测 key 裁剪 |

回归头还继承 spatial/temporal 分支的三维 patch XYZ/有效性、未来物理时间、时序卷积、尺度分离、相邻位移和匹配 GT 的二阶差分监督。它不是旧 `displacement` 配置简单增大 loss 权重，也不是 `delta_state` 的逐步增量累积。

配置继承为 `regression_state → flow_state → flow_temporal → flow_scaled → flow → spatial → relative_goal → chunk_v4 → full → baseline`。继承文件名含 flow 不代表实际使用 flow；最终 `decoder_type=regression` 决定动作头。

此次修改沿用现有 V4，不需重建数据。K=16、建议 H=4，输出仍为 world 米/TCP/xyzw/1 开 0 关；joint 冻结视觉塔、projector 和 Gemma 前 16 层，后续层与动作头训练。当前 chunk 是 baseline，不接 role cache；导出 T/R 分割标注不等于启用物体辅助。

### 排查经验

1. 先建立可学好的确定性基线，再用相同数据、初始化来源和预算比较 flow/delta，复杂生成头不保证更准确。
2. 短期动作需直接读取实测状态；归一化 q6/夹爪、逐层条件、视觉和 key 是本次组合改动，尚未逐项归因。
3. loss 与物理误差分开：同时看完整轨迹、前 1/H 步米制误差、静止基线和同时间/同尺度轨迹图。
4. flow 带噪 GT 重建不能替代独立部署采样；回归使用相同部署预测路径。
5. 不把早期猜测写成根因：chunk 已读 coarse/refine，没有证据证明仅删除 refine 可解决；key gate 也不能切断 refine 裁剪依赖。

### 历史验证记录

实现提交为 `c8852a9`，此前文档整理为 `f9ed851`。交接记录报告实现阶段通过 312 项 CPU 回归、两组双进程 Gloo DDP 检查，并对照修改前版本检查旧配置输出兼容。这是历史记录，不表示本次文档合并重跑了这些测试。

2026-10-29 文档核对另有 30 项状态条件 CPU 回归通过；full 和三组 state 的实际预算解析也已核对。上述正确性检查、用户轨迹确认与完整 CUDA 吞吐/显存/定量/闭环验收分别记录，不互相替代。

模型入口为 [spatial_chunk.py](../../finetune/OHT/spatial_chunk.py)、[chunk_model.py](../../finetune/OHT/chunk_model.py) 和 [chunk_normalization.py](../../finetune/OHT/chunk_normalization.py)；状态条件与保存恢复回归见 [test_oht_chunk_state.py](../../tests/test_oht_chunk_state.py)。

## 连续轨迹模型

### 当前三组配置

| 配置 | 内部表示 | 生成方式 |
| --- | --- | --- |
| [regression_state](../../finetune/OHT/configs/oht_chunk_v4_regression_state_8x40.yaml) | 相对当前 TCP 的未来 pose | 确定性回归；轨迹改善有反馈，闭环待验证 |
| [delta_state](../../finetune/OHT/configs/oht_chunk_v4_delta_state_8x40.yaml) | 逐步平移/旋转增量 | 回归后累积，监督完整轨迹 |
| [delta_state_guided](../../finetune/OHT/configs/oht_chunk_v4_delta_state_guided_8x40.yaml) | 同 delta_state，增加独立空间 key token | 新增目标引导对照，尚未验证闭环收益 |
| [flow_state](../../finetune/OHT/configs/oht_chunk_v4_flow_state_8x40.yaml) | 相对当前 TCP 的未来 pose | conditional flow matching |

三组共用 coarse/refine 视觉、实测本体状态、相对 key、三维 attention、AdaLN、时序卷积与物理监督。当前只支持 `mode=baseline`、`action_mode=eef`，不接历史帧、role cache、异步执行或碰撞规划。

### 一次预测的数据流

1. 当前多视角 RGB-D/指令经 BridgeVLA，预测关键目标并产生 coarse 全局与 refine 局部特征；refine 围绕预测 key 裁剪。
2. 将视觉 patch 对应的可见表面 XYZ、有效性、view/scale 标识送入 decoder。
3. 当前 TCP 定义局部坐标，按 checkpoint 的 `joint_encoding` 编码 q6，归一化夹爪并与门控 key 注入动作 block。
4. 预测 K 个未来动作并还原 world EEF；客户端按实际 dt 消费前 H 项，执行规则见 [runtime.md](runtime.md)。

key 只是方向条件，可以位于 K 步窗口之外，不要求 chunk 末点到达它。第一点监督真实 `t+dt`，不强制等于当前 TCP。V4 标签、padding、缺帧及 no-key 尾段契约见 [data.md](data.md)。

### 本体状态条件与增量回归

`state_conditioning` 将当前 TCP 局部状态投影为 `s`，相对预测 key 投影为 `g`：

```text
c = time_embedding(tau) + s + sigmoid(key_gate_logit) * g
```

`c` 同时加入初始 queries 和各层 AdaLN 的 self-attention、cross-attention、FFN 调制。AdaLN 残差 gate 零初始化时，状态仍能通过初始 queries 影响输出。回归 tau=1；flow tau 是生成时间，与未来物理时间 `k*action_dt` 不同。

保留一个始终有效的状态 memory token；全部视觉 patch 无效时仍可读状态。原 state 配置的显式 key 只有上述 gate 路径；gate 初值 0.1，记录为 `chunk_key_condition_gate`。refine 仍依赖 key 裁剪，gate 接近 0 不等于网络完全不使用 key。

### 独立空间关键目标引导（新增对照）

`oht_chunk_v4_delta_state_guided_8x40.yaml` 继承 delta_state，仅启用
`chunk_policy.spatial_key_token=true`。原配置默认 false，旧 checkpoint 按原路径加载。

```text
预测 key pose/gripper → 相对当前 TCP 编码 → 独立 goal memory token
                                   └→ TCP-local XYZ / geometry_scale
coarse/fine 特征 + 实测状态 token ────────────┤
                                           ↓
                   每层三维 cross-attention → delta chunk → world TCP
```

目标 token 与场景 patch 使用相同的 TCP 局部几何坐标，读取的是本次模型预测的 key；
原有 `s + gate*g` queries/AdaLN 条件保留。新 token 的 attention logits 增加
`log(sigmoid(key_gate_logit)) + log(N)`，N 是其余有效 memory token 数（包括状态 token）。
这样只增加无效 patch 或改变有效 patch 数不会单凭数量稀释目标的 attention 先验；实际权重仍由
内容、空间 bias 和 gate 共同决定，gate=0.1 不代表实际 attention 固定为 10%。
门控趋近 0 时，新 token 的语义和 XYZ 路径一起关闭；视觉全无效时仍保留状态 token，避免全屏蔽 attention。

训练 loss、数据、K/H、冻结和预算继承 delta_state；不加入朝 key 的直线插值、每步距离递减或
窗口终点到达约束，因此允许示范中的绕行。预测离散 key/crop 仍 detach，新增条件不把 chunk
梯度反传到离散关键目标坐标。该实验需要新 run 训练，现有 V4 replay 可直接复用，无需重建数据。

旧 `joint_encoding: scalar` 使用 train 1%/99% 分位数的中点/半区间归一化 q6，scale 下限
0.1 rad；不裁剪分位区间外值，scale<π 的关节在模型特征中对齐到最接近训练中心的 2π 分支。
新 `sin_cos` 使用原始弧度的正弦和余弦，不做这项分支对齐或关节分位数缩放。
两者的实测夹爪均采用保存的 center=0.5、scale=0.5；原始测量和物理控制不变，
原始 TCP 状态保留用于坐标恢复。周期编码的新配置与兼容边界见[本次修复](#新实验使用连续的周期关节特征)。

### 几何与空间 attention

点云渲染在原 RGB 后附加 XYZ/覆盖通道，共享 z-buffer、splat 和抗锯齿，不重复渲染或修改 RGB/correlation/depth。patch 池化预乘 XYZ 与覆盖再相除，无有效表面则屏蔽；world 原点真实表面仍有效，token 顺序为 view→row→column。

coarse 撤销全局 cube 变换；refine 先撤销实际预测 crop，再撤销 cube。世界米坐标转为 `R_0ᵀ(x_world-p_0)`，按 `geometry_scale=0.25 m` 缩放。这统一 decoder 几何定义，不使上游 VLM 成为严格 SE(3) 等变模型。

各 attention head 从相对方向和对数距离生成三维 bias，视觉带 view/scale embedding，queries 带未来物理时间。动作与几何归一化尺度不同，位置比较乘 `action_position_scale/geometry_scale`；delta 先累积平移再比较。

### 动作表示

内部 pose 为 K×9（平移 3、连续旋转 6），另预测夹爪 logit。六维旋转正交化恢复 SO(3)，转单位 xyzw 四元数；推理拒绝退化旋转、非有限或非法动作。

| 表示 | 平移编码 | 旋转编码 |
| --- | --- | --- |
| `relative_pose` | `R_0ᵀ(p_k-p_0)/scale` | `R_0ᵀR_k` 的前两列 |
| `delta_pose` | `R_0ᵀ(p_k-p_(k-1))/scale` | `D_k=R_(k-1)ᵀR_k` 的前两列 |

relative 的各点相对同一当前 TCP；delta 平移轴仍为初始 TCP，恢复为 `p_k=p_0+R_0*scale*sum(d_1...d_k)`，旋转为 `R_k=R_0D_1...D_k`。不能直接累加 Euler/旋转向量。delta 当前只支持 spatial regression；公共输出始终是 `[x,y,z,qx,qy,qz,qw,open01]` 绝对 world TCP。

### 训练目标

joint 优化 `key_loss_weight*key_loss + chunk_loss`，key 默认权重 1。三组的物理监督作用于恢复后的完整世界轨迹：

| 项 | 权重/尺度 | 监督对象 |
| --- | --- | --- |
| 位置 Smooth L1 | 5 / 0.01 m | 各有效未来点 |
| 旋转 MSE | 1 | 世界旋转矩阵；四元数符号等价 |
| 夹爪 BCE | 1 | 对应未来 tick 二值状态 |
| 相邻位移 Smooth L1 | 1 / 0.01 m | 首段从实测 TCP 开始，后续连续有效帧 |
| 二阶差分 Smooth L1 | 0.1 / 0.001 m | 匹配 GT 位移变化，保留真实转弯 |

mask 屏蔽无效标签，位移/二阶差分不跨 padding 或缺失点。二阶差分单位为米/控制步²，不是 m/s²，也不惩罚曲率本身。输出前有 kernel=3 的 depthwise 时序卷积，残差零初始化。

flow 用高斯噪声 epsilon 和 tau 构造 `A_tau=(1-tau)*epsilon+tau*A_gt`，拟合 `v_theta=A_gt-epsilon`；平移/旋转 MSE 按上述权重加权，再乘 `flow_weight=1`。物理辅助监督作用于 `A_hat=A_tau+(1-tau)*v_theta`，向量场速度不是机器人速度。

当前 tau 为 `0.999*(1-U**(1/1.5))`，偏向高噪声端；推理 8 次 Euler 更新后另调用一次 field 计算干净端点夹爪，共 9 次 field。默认 seed 0 在每次请求生成同一初始噪声并扩展到 batch，使同条件结果不随 batch 大小改变，不修改全局 RNG；null 才逐样本随机采样。回归不计算 flow loss/积分。

#### 训练目标与部署采样的一致性诊断

训练与部署的差异有三处：(1) 重建项评估的是单步干净端点估计 `A_hat`，高噪声 tau 下近似条件均值，与 8 步 Euler 采样的误差不同；(2) 夹爪 logit 在训练时读取随机 tau 的带噪输入，部署时读取 tau=1 的采样终点，`beta_high_noise` 下 tau>0.9 的训练占比很低；(3) 无效（padding）tick 训练时输入纯噪声。`SpatialChunkDecoder.flow_reconstruction` 是训练与诊断共用的单步路径，`sample(memory, state, steps=, seed=)` 默认仍使用配置的步数和 seed。

[flow_diagnostics.py](../../finetune/OHT/flow_diagnostics.py) 的 `flow_consistency_report(decoder, context, target, valid)` 对 `ChunkAgent._context` 产生的上下文输出：各 tau 的单步重建误差、按训练 tau 分布加权的重建误差、1/2/4/8/16 步采样的位置/旋转/夹爪误差及多 seed 标准差、`reconstruction_minus_deployed_m`，以及夹爪在干净 GT 终点（上界）与部署采样下的准确率。它只度量差距，不替代闭环成功率。

`sampling_by_steps` 是 `--flow-steps` 指定步数的多 seed 扫描结果。部署指标另行调用 `sample(memory, state)`，使用 checkpoint 配置的 `flow_steps` 和 `flow_inference_seed`；即使配置步数不在扫描列表中，也始终评估。`deployed_position_error_m`、`gripper.deployed_accuracy` 和 `reconstruction_minus_deployed_m` 均基于这次部署采样；seed 为 `null` 时沿用部署的随机采样行为。

在 held-out replay 上对 checkpoint 运行（需 CUDA，沿用 `open-loop` 的加载与 contract 校验，报告为各样本平均）：

```bash
python -m finetune.OHT.eval flow-consistency --checkpoint model_last.pth --replay REPLAY \
  --split val --limit 200 --flow-steps 1 2 4 8 16 --noise-seeds 8 --output NEW_flow_consistency.json
```

上下文来自 `Policy.chunk_decoder_context`，与部署使用同一套观测过滤、key 预测和 TCP crop。replay 报告同时记录 `flow_steps` 和 `flow_inference_seed`，便于核对部署配置。

测试 [test_oht_flow_consistency.py](../../tests/test_oht_flow_consistency.py) 用精确的点质量边缘速度场验证：任意步数 Euler 采样都能还原 `encode_target` 的目标（绝对位姿与 `delta_pose` 两种表示），即 tau 方向、编码/解码与旋转复合约定一致。结论依赖真实 checkpoint 与 held-out 样本，尚未在实际模型上测量。

### joint 与 frozen_key

| 阶段 | 关键目标网络 | chunk 条件与监督 |
| --- | --- | --- |
| `joint`，当前默认 | 兼容参数联合训练，保留独立 key loss | 重新走预测 key/crop 路径，chunk loss 可更新未冻结视觉特征 |
| `frozen_key` | 从已训练 EEF checkpoint 加载并整体冻结 | inference-equivalent 特征提取，只有新 decoder 训练 |

joint 的 key 监督分支只选 `key_valid=true` 子批，沿用 GT crop/rotation conditioning；chunk 分支重新从当前观测预测 key，不共享 GT 选出的局部特征。GT waypoint、未来状态和 key_valid 不进入 chunk 条件；no-key 尾段只算 chunk，key loss 按全卡有效 key 数归一化，允许某 rank 没有 key。

两条路径位于同一外层 DDP forward，最后一次 backward/step。预测条件当前逐样本复用部署 `act`，并非整个 batch 仅两次 VLM，也尚未批量优化 backbone。冻结模块保持 eval、BatchNorm 运行统计固定，支持 singleton key 子批。

预测 key 的离散 pose/gripper 与 crop 位置 detach，chunk 可沿预测视觉特征反传，frozen_key 则全部 detach。关键目标 GT 仅独立监督，不能将 GT-crop 训练特征直接接入 chunk 后声称条件与部署一致。

### 归一化与 checkpoint

rank 0 按 train 样本索引等间隔选最多 4096 个样本，广播统计或错误；不是任务/episode 均衡随机采样，val/test 与无效 padding 不参与。

| 用途 | 设置/产物 |
| --- | --- |
| 动作平移编码、flow 位置噪声 | `action_position_scale: train_quantile`；当前 TCP 到有效未来点半径 99% 分位数，下限 0.01 m |
| attention 的 scene/key/query 坐标 | `geometry_scale: 0.25` |
| 世界位置 loss 容差 | `position_loss_scale: 0.01` |
| 动作统计 | `chunk_normalization.json`、`config.chunk_normalization` |
| q6/夹爪统计 | `proprio_normalization.json`、`config.chunk_proprio_normalization`，同时为 decoder buffer；`sin_cos` 的周期关节特征不使用关节分位数缩放 |

delta 也统计完整未来位移，不改成单步 delta 尺度。动作统计读小型 targets NPZ 与哈希绑定 current_tcp；本体统计只解压 q6/开度，但 checksum 仍读完整文件，启动打印进度。数值存入 checkpoint，部署无需 replay 或重新拟合。旧配置未分离尺度时回退 `position_scale=0.25`。

### 旧配置与消融

| 配置（均位于 `finetune/OHT/configs/`） | 主要变化 |
| --- | --- |
| `oht_chunk_v4_8x40.yaml` | 基础 token Transformer，位置权重 5，无 spatial/state 条件 |
| `oht_chunk_v4_displacement_8x40.yaml` | 基础配置加相邻位移监督 |
| `oht_chunk_v4_relative_goal_8x40.yaml` | key 改为相对 TCP，不自动叠加位移 loss |
| `oht_chunk_v4_spatial_8x40.yaml` | 相对 key、三维 attention、位移监督、回归头 |
| `oht_chunk_v4_flow_8x40.yaml` | spatial 加 flow matching |
| `oht_chunk_v4_flow_scaled_8x40.yaml` | 分离动作/几何/loss 尺度，高噪声 tau |
| `oht_chunk_v4_flow_temporal_8x40.yaml` | scaled 加时间 AdaLN、时序卷积、二阶差分 |

flow_state 再加本体归一化、逐层状态与 key gate，regression_state/delta_state 从这里分出。旧 checkpoint 仍按保存配置解释。基础 token decoder 与 spatial/state 坐标不同，不能混称相同结构。

对照固定数据划分、输入、初始化、seed、预算和执行器；K=1/H=1、K=16/H=1 与 H=4 分开测。删显式 key embedding 仍保留 key-crop，不是“全部无 key”；若检验全部 key 条件，应共用全局特征/裁剪规则。GT-key chunk 单列 privileged 上界，不以辅助 loss 或热图观感证明闭环收益。

时间采样参考 [pi0](https://arxiv.org/abs/2410.24164)，相关方法另见 [ChainedDiffuser](https://proceedings.mlr.press/v229/xian23a.html)、[3D Diffuser Actor](https://3d-diffuser-actor.github.io/)。时序卷积、物理差分与尺度分离为本仓库设计，不是原样复现。

## 训练与评估

### 环境与路径

使用已安装 PyTorch/CUDA、Transformers、PaliGemma 与 point-renderer 的 BridgeVLA 环境，见[安装说明](../guides/installation.md)。附加依赖：

```bash
python -m pip install -r finetune/OHT/requirements-model.txt
python -m pip install --no-build-isolation -e finetune/bridgevla/libs/point-renderer
```

本页独立设置变量，已有完整 V4 可直接训练：

```bash
ROOT=/common-data-32t/data/robot_data/oht_curobo_pd_v423-seg
OUT="$ROOT/bridgevla-seg-v4-run01"
PRETRAIN=/home/yiwei/project/BridgeVLA/checkpoints/pretrain
CONFIG=finetune/OHT/configs/oht_chunk_v4_regression_state_8x40.yaml
EXP=oht-seg-v4-regression-state-s0
```

`PRETRAIN` 是热图预训练目录，PaliGemma 模型/processor 仍需可访问或已完整缓存；离线缓存齐备时可设置 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。数据未构建时先按 [data.md](data.md)准备，不手工补 contract。

### 启动命令

```bash
GPUS_PER_NODE=2 bash finetune/OHT/train_8x40.sh \
  --config "$CONFIG" --replay "$OUT/replay-source-v4" \
  --pretrain-path "$PRETRAIN" \
  --log-dir "$OUT/runs" --exp-id "$EXP"
```

delta/flow 对照分别改为 `oht_chunk_v4_delta_state_8x40.yaml`、`oht_chunk_v4_flow_state_8x40.yaml`，并用新 EXP；初始化来源和 seed 保持一致。joint 不要求旧 OHT checkpoint，复用 encoder 时另加 `--init-checkpoint /path/to/model_last.pth`。

新增的空间 key guidance 对照可直接运行：

```bash
GPUS_PER_NODE=2 bash finetune/OHT/train_8x40.sh \
  --config finetune/OHT/configs/oht_chunk_v4_delta_state_guided_8x40.yaml \
  --replay "$OUT/replay-source-v4" --pretrain-path "$PRETRAIN" \
  --log-dir "$OUT/runs" --exp-id oht-seg-v4-delta-state-guided-s0
```

已有相同数据/归一化的 delta_state 权重时，可额外加
`--init-checkpoint /path/to/delta-state/model_last.pth` 做新实验：保留兼容 encoder/decoder 权重，
启用新空间 token 时将共享 key gate 重置为配置初值 0.1，避免继承旧实验已接近关闭的 gate。
这会重新开始 optimizer/采样；不能用旧实验的 `--resume` 来切换 guidance。
训练日志打印 `spatial_key_token=true`，server 从新 checkpoint 自动恢复此设置，client 动作协议不变。

### Chunk 使用当前 TCP 局部视图

`oht_chunk_v4_delta_state_tcp_guided_8x40.yaml` 将两种局部视图分开：关键目标分支仍使用 coarse 预测目标中心的 MVT2 裁剪；chunk 使用 coarse 全局 tokens、实测 TCP 中心的局部 tokens、归一化实测状态与带 gate 的预测 key 条件。TCP 局部视图复用现有视觉/Gemma 编码器权重，增加一次局部渲染与编码，只返回视觉 tokens，跳过额外的 heatmap/action heads；冻结层数沿用配置，额外计算与显存开销需在 GPU 上实测。

```bash
GPUS_PER_NODE=2 bash finetune/OHT/train_8x40.sh \
  --config finetune/OHT/configs/oht_chunk_v4_delta_state_tcp_guided_8x40.yaml \
  --replay "$OUT/replay-source-v4" --pretrain-path "$PRETRAIN" \
  --log-dir "$OUT/runs" --exp-id oht-seg-v4-delta-state-tcp-guided-s0
```

`chunk_policy.local_context` 有三个选项：

| 值 | Chunk 视觉输入 | 对应 guided 配置 |
| --- | --- | --- |
| `key_refine`（默认） | coarse + 目标中心 refine | `oht_chunk_v4_delta_state_guided_8x40.yaml` |
| `tcp` | coarse + 实测 TCP 中心局部视图 | `oht_chunk_v4_delta_state_tcp_guided_8x40.yaml` |
| `coarse_only` | 仅 coarse；保留状态和预测 key 引导 | `oht_chunk_v4_delta_state_coarse_guided_8x40.yaml` |

`tcp_crop_size_m: 0.4` 是以实测 TCP 为中心、与世界 XYZ 轴平行的立方体完整边长，即每轴 ±0.2 m；不是半径，也不随 workspace 大小变化。对当前观测点云在三个轴上裁剪，再投影并编码；几何随同一 z-buffer 对齐，逆变换到世界坐标后供 chunk 使用。空局部视图的 tokens 被有效性 mask 排除，仍可读取 coarse 和状态；这不提供碰撞保证。

新 TCP 路径的正交渲染相机沿光轴后移，保持 XY 投影不变，同时将立方体内的表面放到相机前方，再过滤非正或非有限深度；旧 key/refine 路径保留原相机与深度规则。空点云直接返回空白图和无效覆盖，避免零线程块 CUDA 启动。训练诊断 `chunk_local_valid_rate` 是局部有效 patch 比例，`chunk_local_empty_rate` 是整份局部输入为空的样本比例；TCP 模式下若后者长期较高，应先检查点云 ROI、TCP 坐标和裁剪尺寸。coarse-only 不记录这两项，因其主动移除了局部输入。

三组继续使用相同 V4 replay，不需要重建数据。可用相同数据与归一化的 delta/guided checkpoint 加 `--init-checkpoint` 开启新实验，兼容 encoder/decoder 权重保留；切换 `local_context` 或裁剪尺寸不能使用旧 run 的严格 `--resume`。server 按新 checkpoint 中保存的设置自动恢复，client 继续提供当前实测 TCP、关节和夹爪，动作协议不变。日志与 `/model_info.chunk_policy` 可核对 `local_context/tcp_crop_size_m`。

TensorBoard 的 `mvt2` 仍显示关键目标分支的目标中心裁剪，并非新 chunk 的 TCP 局部输入；`train_visualization/chunk` 比较实际 chunk 预测和未来轨迹。闭环收益需固定初始化、预算、seed 与执行器，对照成功率、碰撞和抖动，不能由裁剪图或训练 loss 单独判定。

TCP-guided 的 open-loop 每次读取专家记录的当前状态，预测前 K 个未来点；在线每次读取实际执行后的状态。
当前数据只监督专家轨迹上的观测，没有自动提供偏离轨迹后的恢复示范。K=16、H=4 在 60 Hz 下分别覆盖
约 0.267 s / 0.067 s，短窗准确并不证明能持续推进或恢复偏差。预测 key 是带 gate 的条件，不能保证 chunk
朝它前进，也不构成避碰约束。全局 224 像素视图可能掩盖毫米误差，应同时核对前 H 项数值误差和姿态。

方向或位置明显偏离时，先按[记录初态](runtime.md#记录初态)做一次同 episode/frame 的实时观测对照，
再比较 episode 中的原始 actions、实际下发 target 与实测 TCP：

| 现象 | 优先排查 |
| --- | --- |
| 同记录初态的第一次预测已经偏离 | 实际加载的 checkpoint、相机 K/T 与深度、TCP 定义、关节顺序/归一化、物体状态及模型预测误差 |
| 模型目标合理，实测运动明显偏离 | TCP 到 OSC 控制点的变换、控制基坐标、姿态及实际跟踪误差 |
| 前缀起初正常，滚动多次后逐渐偏离 | 执行误差、相邻 chunk 不一致、任务进度/接触状态变化及训练分布外的恢复能力 |

server 只按 checkpoint 保存的配置建模；核对 `/model_info.chunk_policy` 中
`decoder_type=regression`、`prediction_representation=delta_pose`、`local_context=tcp`、
`tcp_crop_size_m=0.4`、`spatial_key_token=true`。这些路径检查不代替真实仿真验收，
没有本次执行记录时不能从 open-loop 图单独确定碰撞根因。

核心审查还有三项边界：

- **未来状态与控制命令不同。** V4 `action_chunk` 是未来实测 TCP/夹爪状态，client 用 OSC 残差跟踪；
  它不是直接送进 `env.step` 的 7 维 OSC 命令，也不包含逆动力学或前馈控制。数据采集与在线跟踪的
  控制响应不同，可能在重规划后累积偏差；应先量化预测目标与实测响应，不能仅用状态轨迹拟合误差比较控制策略。
- **位置正确不代表姿态正确。** 三维位置图不显示整段旋转与夹爪时序。旋转矩阵逐元素 MSE
  约 0.0595 已可对应 30° 姿态误差；这不是 0.0595°。检查下方角度/夹爪指标，尤其实际执行的前 H 项。
- **状态捷径是待测风险。** 直接状态条件能绕过视觉 cross-attention；AdaLN 残差 gates 全零初始化时，
  新动作头的初始输出不依赖视觉 memory，gates 学开后才有该路径。初始化测试可复现这一点，
  但不能据此断言训练后的 checkpoint 忽略图像。须在实际权重上做固定状态的视觉条件对照，再决定是否改初始化或训练规则。

### 预算与卡数

| 当前三组共用设置 | 值 |
| --- | --- |
| `epochs` / `train_iter` | 60 / 40000 |
| `global_batch_size` / `batch_size` | 16 / 每卡 8 |
| `max_optimizer_steps` | 83300 |
| `checkpoint_every_epochs` | 10 |
| 优化器/学习率/weight decay | AdamW / `8e-5` / `1e-6`，固定学习率 |
| 冻结 | vision tower、projector、Gemma 前 16 层；lm_head/embed_tokens 沿用冻结 |
| 全局刚体增强 | 关闭；恢复前须定义机器人基座/关节等变契约 |

`train_iter` 是任务均衡、有放回的采样预算，不是数据集长度。一轮 `floor(40000/16)=2500` 优化步，实际目标 `min(60×2500,83300)=83300`，约 **33.32 个采样 epoch**；60 是配置目标，不是已完成轮数。跑满 60 轮需在新配置中将上限设为 0 或至少 150000。

`accumulation_steps: null` 自动计算：两卡累积 1、单卡 2；四卡需每卡 batch 4，八卡需 2，满足全局 batch 可被 `每卡 batch×卡数` 整除。显存需实测，严格 resume 不能更换卡数/batch。

脚本默认八卡，示例显式设两卡；名称 `8x40` 不自动调整 batch/显存。torchrun `--standalone` 自动分配 rendezvous endpoint，不能依赖 `MASTER_PORT` 固定它。

### 初始化与续训

| 操作 | 参数 | 恢复内容 |
| --- | --- | --- |
| 基础 joint | `--pretrain-path` 可指定热图预训练 | 不要求 OHT checkpoint，仍需 PaliGemma |
| 新实验初始化 | `--init-checkpoint /path/to/model_last.pth` | 兼容权重，optimizer/采样重新开始 |
| 原实验续训 | `--resume /path/to/model_last.pth` | 模型、optimizer、已完成采样位置 |

结构、`joint_encoding`、状态/本体归一化或动作尺度变化时 init 新 run，保留 encoder、重建整个 decoder；允许的 loss/采样设置变化可保留 decoder。`spatial_key_token` 只改变路由，可在 init 新 run 时保留兼容 decoder，首次启用时重置共享 key gate；边界由 `load_chunk_weights()` 检查。单关键点 checkpoint 只初始化匹配 data profiles/标定的 encoder，不能 resume 为 chunk。frozen_key 必须加载已训练网络。

```bash
RUN=/path/to/actual/run
GPUS_PER_NODE=2 bash finetune/OHT/train_8x40.sh \
  --config "$CONFIG" --replay "$OUT/replay-source-v4" \
  --pretrain-path "$PRETRAIN" --resume "$RUN/model_last.pth"
```

严格 resume 校验解析后配置、replay/role cache 契约、卡数、学习率、batch、冻结、动作和 seed。允许调预算、保存周期、num_workers 与独立可视化；epoch 的 `epochs/max_optimizer_steps/checkpoint_every_epochs` 与 step 的 `optimizer_steps/checkpoint_interval` 两套预算不同时设置。

采样器按完成 step 跳过抽样位置。checkpoint 同时保存每个 rank 的 Python/NumPy/torch CPU 与 CUDA RNG 状态（`rng_states`），续训时各 rank 在创建 DataLoader 迭代器之后恢复自己的状态，启动日志打印 `Resume RNG state restored`。DataLoader worker 进程内部的随机流不在其中，增强若发生在 worker 内则不承诺逐位复现；更早版本的 checkpoint 没有 `rng_states`，仍可续训但不恢复 RNG（日志为 `False`）。续训写回原 run，不创建新时间目录。

### 数据校验开销

V4 样本的 observation/targets NPZ 默认在每次读取时重算 SHA-256（严格，等价旧行为）。数据已由 `--validate-only` 校验后，可用 `--data-checksum once`（每个进程对每个文件只校验一次）或 `--data-checksum trusted`（跳过逐样本哈希）降低读取开销。`--validate-only` / `validate_all()` 始终严格校验，不受该选项影响；`trusted` 只应在缓存没有被改动时使用。

### 预检与短测

完整 replay 校验检查所有非空划分，训练预检仅检查 train，不能代替前者：

```bash
python tools/validate_oht_replay.py --replay "$OUT/replay-source-v4" --workers 4
WORLD_SIZE=2 python -m finetune.OHT.train \
  --config "$CONFIG" --replay "$OUT/replay-source-v4" --validate-only
```

预检不构建模型/DDP 或输出目录，但归一化会读取 train NPZ。未设 WORLD_SIZE 时通常按一张卡计算；八卡仍配每卡 8 会因全局 batch 16 不可整除而失败，CPU 数据检查不证明 GPU 配置正确。

短测新建 `finetune/OHT/configs/my_chunk_smoke.yaml`：

```yaml
extends: oht_chunk_v4_regression_state_8x40.yaml
epochs: 1
max_optimizer_steps: 2
```

用此配置、新 EXP 启动同一训练命令。现有 `smoke.yaml` 属旧单关键点，不能直接读 V4。

### 输出目录

`--log-dir ROOT` 创建 `ROOT/train/<exp_id>/<exp_note>/MM_DD_HH_MM/`，空备注省略，Asia/Shanghai 时间，同分钟追加后缀。rank 0 创建/广播并打印 `Run directory: ...`，以该路径为准。

目录保存 `run.json/config.json/contract.json/training_plan.json`、启用时的两个 normalization JSON、`metrics.jsonl`、`tensorboard/`、`train_visualizations/` 与 checkpoints。`--output DIR` 使用精确新目录，和 `--log-dir` 互斥；resume 回原目录。评估使用不再覆盖的固定快照，避免一边训练覆盖 `model_last.pth` 一边读它。

### 日志与可视化

TensorBoard 默认启用但需安装 `tensorboard`，model requirements 未列入；`--log-backend none` 关闭 events，JSONL 仍保存。

```bash
python -m pip install tensorboard
tensorboard --logdir "$OUT/runs/train" --port 6006
```

rank 0 写日志，通常是本地 microbatch/累积平均，不是跨卡均值；key 指标另按有效 key 归一化。`train_visualization` 控制 enabled/interval/save_png/tensorboard/output_dir，记录第 1 步和间隔步；可视化增加传输/写盘成本，可增大 interval。

OHT 与 RLBench 共用 `finetune/RLBench/training_utils.py` 的终端进度和指标格式：tqdm 显示进度、耗时、预计剩余时间、主要 loss 和学习率；定期输出 `[train] epoch=... iter=... step=... lr=...` 明细。OHT 支持 `--loss-print-interval 10`（也接受 `--loss_print_interval`），设为 `0` 仅关闭明细行。chunk 的进度摘要将位置误差显示为 `pos_mm/prefix1_mm`，明细仍保留原指标名和米单位，小数值 loss 使用有效数字避免误显示为零。OHT 的 `metrics.jsonl` 仍逐 optimizer step 保存完整指标，TensorBoard 标量与 checkpoint 步数不变；epoch/step 编号沿用各自训练器的约定。

Chunk 训练的 `train_visualization/mvt1`、`mvt2` 中，`Rendered RGB` 和 `Policy / heatmap TCP` 两列叠加完整 chunk 轨迹：红线为预测，绿线为同一组有效 tick 的未来观测，白叉为当前实测 TCP。TCP 标记列还保留红叉（第一步 chunk 预测 TCP）和青圈（heatmap 解码的关键目标 TCP）；`Prediction heatmap` 列显示原始热图底图。轨迹和标记均使用各 stage 的 `world_to_pixel` 投影，MVT2 已包含 refine 裁剪变换；图外点不绘制，不跨无效或图外点连线，二维投影不判断遮挡。启用 `save_png` 时同时保存对应 stage 的 NPZ，含原始热图、投影矩阵、完整轨迹世界/像素坐标、共享 `chunk_valid`、sample ID、step 和 dt，便于核对。未来标签只在预测后的绘图阶段使用。无需重建数据或更换配置；重启训练后新生成的 TensorBoard 图与 PNG 生效。独立的 `train_visualization/chunk` 继续提供等比例坐标和毫米误差。

`mvt1/mvt2` 对应实际 coarse/refine；chunk 图红色预测、绿色有效未来、白色当前 TCP，采用世界坐标、相同时间和等比例轴。PNG/NPZ 含预测、标签、mask、状态和 sample ID；GT 只监督/绘图，不进入部署。flow 复用观测 memory 做 eval/no-grad 独立部署采样，不重跑 VLM。

Chunk 轨迹图默认将同一帧多相机 RGB/世界点云投影为 XY/XZ/YZ 彩色场景背景，使用浅色网格，背景与轨迹共用坐标与比例。背景最多确定性采样 24,000 点，视窗围绕有效轨迹与当前 TCP 保留局部上下文；远处场景不改变轨迹比例。没有有效点云时显示浅色网格。训练、离线评估及在线 chunk 图均使用该显示，NPZ 同时保存 `scene_points` / `scene_colors`；现有 replay 和 checkpoint 可继续使用，重启训练/评估/服务后新生成的图片生效，旧 PNG 不自动更新。

| 指标 | 解读 |
| --- | --- |
| `chunk_position_error_m` | 完整部署预测的平均位置误差 |
| `chunk_prefix_1_position_error_m` / `chunk_prefix_4_position_error_m` | 前 1/默认 H 项误差 |
| `chunk_rotation_error_degrees` / `chunk_rotation_error_max_degrees` | 完整部署预测的平均/最大 SO(3) 角度误差，单位度 |
| `chunk_prefix_1_rotation_error_degrees` / `chunk_prefix_4_rotation_error_degrees` | 前 1/默认 H 项的姿态误差 |
| `chunk_gripper_accuracy` / `chunk_prefix_4_gripper_accuracy` | 按部署阈值 logit ≥0 为开，统计完整/前 H 项有效 tick 的夹爪准确率 |
| `chunk_stationary_error_m` | 始终保持当前 TCP 的对照 |
| `chunk_displacement_error_m` / `chunk_second_difference_error_m` | GT 位移/位移变化匹配误差 |
| `chunk_position_loss` | 按 position_loss_scale 归一化的 Smooth L1，不是米 |
| `chunk_reconstruction_position_error_m` | flow 带噪重建，不能代替部署采样 |
| `chunk_flow_loss` / `chunk_flow_weighted_loss` | 向量场原始/加权项；tau 三档 count/position loss 需按有效 count 汇总 |
| `chunk_key_condition_gate` | 显式 key gate，不代表整个 refine 已与 key 解耦 |

`chunk_sampled_position_loss`、`chunk_sampled_displacement_loss` 仅作诊断不反传。JSONL 旧 run 可用 `python tools/export_oht_tensorboard.py --run "$RUN"` 导到新 `tensorboard-from-jsonl/`，不能补出未保存图片；用完整日志或快照，截断末行会报错。

训练进度条新增 `rot_deg/grip_acc`，详细日志保留前 H 项角度和准确率；flow 使用独立部署采样的结果。
最大角度指标在 rank 0 的累积 microbatch 间取最大值，其余均值指标沿用现有累积平均，不是跨卡全局统计。
新生成的 `train_visualization/chunk` 图底部显示同一张样本的姿态误差/夹爪准确率，NPZ 另存
`predicted_rotation_matrix`、`target_quaternion_xyzw`、预测/观测夹爪和当前 TCP pose，可核对位置图隐藏的误差。
这些是诊断，不改变模型输出、loss、checkpoint 参数或执行协议；重启训练即可记录，不需要重建数据。

### 离线与闭环评估

```bash
RUN=/path/to/actual/run
python -u -m finetune.OHT.eval open-loop \
  --checkpoint "$RUN/model_last.pth" \
  --replay "$OUT/replay-source-v4" --split val \
  --output "$OUT/eval/regression-state-val.json"
```

先检查少量样本可加 `--limit 8 --visualize-every 1`，正式去掉 limit；默认每 50 样本保存图，0 关闭。报告/图片用新路径，默认图目录为 `<报告名>_visualizations/`。V4 `index.html/sample_*/chunk.png/npz` 展示轨迹及可用 coarse/refine 图，JSON 按未来 offset 统计误差、有效数与 key 指标。

快速比较 checkpoint 时可用 `--samples-per-episode 16` 在每个选中 episode 内等距取可用观测，抽样与动作标签无关，并与 `--limit` 互斥。完整 episode 统计可省略这两个参数；可选 `--task assemble_left --episode-index 7` 筛选，编号必须存在于指定 val/test split，两项均可重复传入。单关键点使用其原 replay；V4 chunk 使用 `replay-source-v4`，入口按 checkpoint 自动区分：

```bash
python -u -m finetune.OHT.eval open-loop \
  --checkpoint "$RUN/model_last.pth" \
  --replay "$OUT/replay-source-v4" --split val \
  --task assemble_left --episode-index 7 \
  --visualize-every 0 --output "$OUT/eval/assemble-left-ep7.json"
```

报告保留原均值，新增 `statistics`：min/max、mean、总体 std、RMSE、median、p05/p25/p75/p90/p95/p99；位置单位 m、旋转为 SO(3) 测地角度 °（不是欧拉角分量误差），joint 模式为 rad，另有 XYZ/各关节 signed bias、MAE 和分布、夹爪准确率及错误数。`within_tolerance` 分别报告位置 ≤5/10/20/50 mm、旋转 ≤1/5/10° 的比例，不代表任务成功率，分位数也不是均值的置信区间。

`per_episode["task/000007"]` 含覆盖率、帧范围及完整性；`episode_coverage` 以选中 split 的 replay 索引为分母，limit 截断和未运行 episode 不会算完整。**完整指该 episode 所有可用 replay 观测，非原始视频全部帧**；被构建时跳过的窗口或 stride 未采样帧不会补出。`episode_macro` 对完整 episode 等权，旧全局均值仍按样本/预测项加权。

V4 每个 episode 分开保存 `all_actions`、`per_offset`、`key_goals` 和 `executed_prefix_forecasts`（offset ≤模型 execute_horizon 的预测，非实际执行）。重叠 chunk 另外提供 `timeline_actions` / `timeline_predictions`：同一目标帧固定选最近一次真实观测发出的预测，不选误差最小者、不平均、不补缺失帧；无效 padding 排除。它是 teacher-forced 时间轴，**不是把预测动作执行一整集后的累计漂移**。详细逐 episode / 逐帧结果保存在 JSON，终端不展开 `per_episode`。

V4 汇总的 `stationary_gripper_accuracy` 表示始终保持当前夹爪开闭状态的准确率。`gripper_change_accuracy` 只按目标开闭状态与当前状态不同的有效预测项计算，同时查看 `gripper_change_samples`；分母为零时准确率为 `null`。

PNG 热图逐视图拉伸，不能用颜色比较置信度。采样计时含诊断载荷采集、图片写盘在外；`--progress-interval 5` 调心跳，`--quiet` 关闭。首次推理可能构建反投影缓存，结合后续速度和 GPU 指标判断，不以心跳证明 GPU 正在计算。

离线误差不等于闭环成功率；闭环、模型/GT 回放、任务评分与自定义工厂见 [runtime.md](runtime.md)。原生客户端默认 `--score none` 不报成功率，实际成绩须有真实初始化、成功条件和完整 cases，失败保留分母。

open-loop 每次从记录的真实当前观测预测，上一预测不会改变下一输入；closed-loop 的下一观测则来自
实际执行后的状态，跟踪偏差、接触和预测误差都可能改变后续条件。以 replay 契约中的 `action_dt` 计算窗口。
仿真 60 Hz replay 的 K=16 覆盖约 0.267 秒，真机装轮的 K=16 覆盖名义 0.320 秒。
判断动作是否持续推进时，应同时查看前 H 项误差、运动方向及
保持当前 TCP 的对照。离线准确而在线来回摆动时，优先核对同状态下的 RGB-D/本体输入、实际跟踪
及相邻 chunk 的同时间预测变化。[可选时间对齐融合](runtime.md#v4-chunk) 用于减轻重规划抖动，不能替代这些核对。

### PaliGemma 图像 token 索引报错

若日志出现 `IndexKernel.cu:93 index out of bounds`，随后堆栈停在 PaliGemma 的
`inputs_embeds[special_image_mask]`，不能由异步 CUDA 堆栈直接判定是 loss、TCP 裁剪或该行导致。
当前高效 forward 改为先校验词表 ID、每个样本的图像 token 数量、padding 和图像有限性，
再按一维 token 位置合并整行图像特征，绕过上游展开为 `[B,L,D]` 的布尔索引计数。
图像编码、projector、缩放、语言模型和最终 hidden state 路径保持一致；不裁剪非法 ID、不跳过样本。
联合训练的 key 分支异常附带本批实际参与监督的 sample ID。

修改不改变参数或数据 schema，原 V4 replay、配置和兼容 checkpoint 可继续使用。
小型真实 PaliGemma 的多视图、padding、梯度及 gradient checkpointing 等价测试在 CPU 通过；
原多卡 CUDA 崩溃是否消除仍需重跑验证。若仍报错，用 `CUDA_LAUNCH_BLOCKING=1` 重跑定位最先失败的操作，
并保留新日志中的 sample ID；不要在已触发 device assert 的进程中继续训练。

### 本地验证

按变化范围选测试；本次文档合并没有重跑 GPU 或历史整组测试。动作、归一化和状态回归入口：

```bash
python -m pytest -q \
  tests/test_oht_spatial_chunk.py \
  tests/test_oht_chunk_temporal.py \
  tests/test_oht_chunk_state.py
```

CPU 合成数据、renderer/VLM 替身和 Gloo DDP 检查坐标、mask、梯度、GT 隔离与保存恢复，不代表完整 CUDA 精度/吞吐或 IsaacLab 成绩。数据与浏览器回归入口见 [data.md](data.md)。

## 单关键点与物体辅助

以下为旧单目标路线，不接受 V4 chunk replay。常用目录 `replay-source-v3` 的实际 schema 是 `oht_bridgevla_v2`；不要改 contract 混用。数据/teacher 构建统一见[单关键点数据](data.md#单关键点数据)。

| 路线 | 配置 | 额外来源 |
| --- | --- | --- |
| 单关键点 EEF | [oht_full_8x40.yaml](../../finetune/OHT/configs/oht_full_8x40.yaml) | 无对象输入 |
| 内部 T/R | [role_queries.yaml](../../finetune/OHT/configs/role_queries.yaml) | role teacher 仅训练监督 |
| 外部预测 | [predicted_external.yaml](../../finetune/OHT/configs/predicted_external.yaml) | 预测缓存及真实 predictor |

### 单关键点训练

```bash
OHT_REPLAY=/common-data-32t/data/robot_data/oht_curobo_pd_v423/replay-source-v3
PRETRAIN=/home/yiwei/project/BridgeVLA/checkpoints/pretrain
GPUS_PER_NODE=2 bash finetune/OHT/train_8x40.sh \
  --config finetune/OHT/configs/oht_full_8x40.yaml \
  --replay "$OHT_REPLAY" --pretrain-path "$PRETRAIN" \
  --log-dir exp/OHT --exp-id oht-keypoint-eef-s0
```

full 同为 60/40000/batch16/83300 上限，实际约 33.32 采样轮；冻结视觉塔、projector、Gemma 前 **6** 层，启用 scene/EEF 联合变换增强。它与 chunk 前 16 层/禁用全局增强不同。`baseline.yaml` 为早期 step 预算，`smoke.yaml` 为两步 GPU 短测，不是 full 配方。

### 内部角色预测

两个固定 T/R queries 从当前渲染特征/任务上下文预测角色，coarse/refine 继承角色与几何。teacher 只监督 maps、Reference NULL 和可见表面 T/R 分离，推理不读 GT 对象；角色顺序不是持久实例 ID。

`role_loss.diversity/refine_diversity` 各 0.01，后者只在 `object.refine_role_masks: true` 的局部 head 生效，该 head 默认关闭。只在可靠、可见、XYZ 一致且 footprint 不重叠的另一角色表面抑制响应，不把 NULL/unknown/遮挡当背景；允许角色共享对象区域时关闭分离或提供互斥标签。[对象条件策略分离边界](../guides/object-conditioning.md#tr-分离监督与日志)

```bash
ROLE_CACHE=/path/to/role-teacher
python -m finetune.OHT.train \
  --config finetune/OHT/configs/role_queries.yaml \
  --replay "$OHT_REPLAY" --role-cache "$ROLE_CACHE" \
  --output /path/to/new/role-queries-s0 \
  --init-checkpoint /path/to/trained/baseline/model_last.pth
```

查看 `object_slot_diversity_loss/object_refine_diversity_loss` 及 `*_diversity_supervised_pixels`，像素数 0 时 loss 为 0 是正常跳过。有效 replay/teacher 不重建；新 loss 需新训练才影响预测。baseline 继续同样优化预算，或两组同预训练各训同预算，3 seeds 配对；不能把继续训练收益全部算作物体辅助。

### 外部物体预测

仓库有严格缓存/在线 wrapper，未附新的检测模型/感知权重。应用提供无参 `module:factory`，返回 callable：

```python
from finetune.OHT.data.role_cache import role_fields

class Predictor:
    def reset(self):
        pass

    def __call__(self, observation, goal):
        # 用当前观测计算 points/valid/present/confidence，不读 GT 或未来。
        return role_fields("predicted", points, valid, present,
                           confidence=confidence)
```

observation 只含当前 low_dim、每相机 RGB/depth/world XYZ/K/外参；points 为有限 float32 `[2,512,3]`，valid/present 为 bool `[2]`，confidence 为 `[2]`、范围 [0,1]，顺序 T/R。缺几何设 valid=false，遮挡不改成语义 absence；必需 Reference 不可用时当前门控关闭整对辅助。

provenance YAML 至少含真实权重 `model_sha256`、`training_split`；工具附加 factory 并写缓存/checkpoint。这是调用方声明，不独立证明无测试集污染；OHT 感知训练只用 train。

```bash
PREDICTIONS=/path/to/new/predictions-v1
PROVENANCE=/path/to/predictor-provenance.yaml
python tools/predict_oht_objects.py \
  --replay "$OHT_REPLAY" --predictor my_oht_predictor:create \
  --provenance "$PROVENANCE" --output "$PREDICTIONS"
python -m finetune.OHT.train \
  --config finetune/OHT/configs/predicted_external.yaml \
  --replay "$OHT_REPLAY" --role-cache "$PREDICTIONS" \
  --output /path/to/new/predicted-external-s0 \
  --init-checkpoint /path/to/trained/baseline/model_last.pth
```

缓存覆盖对应 train/val 集合。离线按 replay 帧序列、在线按控制采样频率调用，episode 开始 reset；有历史 predictor 须对齐采样策略。在线 factory/provenance 必须与训练缓存一致。

单关键点 open-loop 使用上方 eval 命令但 replay 换为 `$OHT_REPLAY`；外部模式还加 `--predictor my_oht_predictor:create --predictor-provenance "$PROVENANCE"`，由真实 predictor 生成当前角色，不将 teacher 输入策略。部署见 [runtime.md](runtime.md)。

## 单动作表示

`action_mode` 与对象辅助 mode 独立，baseline/role_queries/predicted_external 支持下列四种单目标头；**V4 chunk 仅支持绝对 EEF**，不能扩展解释为四种 chunk。

### 坐标与标签

| 模式 | 动作维度/内容 | 标签 |
| --- | --- | --- |
| `eef` | 8：world TCP xyz/xyzw/grip | 未来关键点实测 TCP |
| `relative_eef` | 7：delta_xyz/rotation_vector/grip | 当前实测 TCP 到同一未来目标 |
| `joint` | 7：q1…q6/grip，rad | 未来关键点实测关节 |
| `relative_joint` | 7：delta_q1…q6/grip，rad | 同一未来关节减当前关节 |

夹爪均为 0 关 1 开，raw action 前六维不参与标签。相对量是到未来 key 的目标位移/转角，不是速度或必然一个控制步的增量。relative_eef 默认 world 平移、当前 TCP body 旋转；body 为 `R_currentᵀ(p_target-p_current)`、`log(R_currentᵀR_target)`，world/base 用相应左乘误差，不减四元数分量。base 必须显式给固定 `world_from_base`。

joint 顺序为记录 `observation.state[:6]`，joint_names 声明但不重排训练数据，执行映射见 [runtime.md](runtime.md)。默认关节直接相减；仅确认连续关节才设置 joint_wrap，不能和 joint_limits 同用。限位配置使训练拒绝越界 GT、解码限制预测，不替代仿真独立限位检查。

prepared 输入中 eef 沿用旧行为，relative_eef 加当前 tcp_pose，joint/relative_joint 加当前 joint_positions；未来 joint_action 只作标签。raw HTTP 各模式仍需实测关节/EE/夹爪供公共预处理与执行，不能据 prepared 白名单删 HTTP 字段。缺当前状态/未来关节的旧缓存须重建，不能填零；数据迁移见 [data.md](data.md)。

### 向量动作头与损失

eef 用原 heatmap、旋转分类和夹爪头。其余模式用物体辅助后的 global/local 特征和当前状态回归六维量+夹爪 logit，优化 Smooth L1、夹爪 BCE、正权重辅助 EEF translation 及已启用角色 loss；不叠加旧旋转/夹爪分类训练项。日志看 `vector_action_loss`、`vector_gripper_loss`、`eef_aux_translation_loss`，关节图中的定位热图是辅助 EEF，不是关节预测。

`mvt.final_dim=2048` 对齐 VLM global/local 特征；旧 64 分类头尺寸不能直接套用。向量模式冻结未用分类头，初始化允许其尺寸不同，EEF 仍严格校验。

### 动作配置与初始化

现有 `relative_eef.yaml/joint.yaml/relative_joint.yaml` 对应三种向量模式，EEF 默认 `baseline.yaml`。如组合 full/角色路线，在 configs 目录新建继承配置，例如：

```yaml
extends: oht_full_8x40.yaml
action_mode: relative_joint
agent:
  transform_augmentation: false
```

相对继承路径按该 YAML 目录解析。向量模式必须关闭全局 transform_augmentation；action 的 hidden_dim 默认 256，translation/rotation/joint scale 默认 1，auxiliary_translation_weight 默认 1 且需大于 0。这些 scale 是训练归一化，输出还原物理单位，不是控制器 gain。adapter_only 时新动作头仍训练。

checkpoint 保存完整动作配置，strict load/resume 拒绝不匹配；显式 init 可迁移骨干并在动作设置改变时重建回归头，必须重训。不能只改 EEF 配置就部署为 joint；预算/卡数/RNG 限制沿用[初始化与续训](#初始化与续训)。完整 CUDA/IsaacLab 四模式仍需现场验收。
