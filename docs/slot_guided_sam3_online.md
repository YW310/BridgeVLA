# Slot-guided SAM3 online RLBench inference

## GT 提示引导已训练策略

已训练的 `rlbench_o2_semantic_gt.yaml` 模型可直接运行
[GT → SAM3 → 原 Relation adapter：完整命令、诊断与限制](../finetune/RLBench/configs/README.md#gt-sam3)。
使用 `gt_guided_sam3_online.py`，无需重训：当前 GT 只生成提示，SAM3 mask/RGB-D 生成动作条件，
不预测 Slot。新 `oracle_online_experiment` 与下文预测式 `online_experiment` 分开记录；
训练仍使用原配置，SAM3 不参与训练。

<a id="original-slots-sam3"></a>

## 原版 slots → SAM3 → 原版动作条件

使用 [internal-slots online 配置](../finetune/RLBench/configs/rlbench_slot_guided_sam3_internal_slots_online.yaml)，
直接加载原版 checkpoint，无需重训。
Coarse T/R→原图 box/point→SAM3→融合点云→两个原版 Anchor；保留独立 Refine、角色图与 tokens。
导出时还原 correlation 尺度；旧 Anchor 使用有效点的中心与 max−min 范围，不计 padding。
Reference 不可见、低置信或几何失败则回退原动作；明确 NULL 支持 Target-only。

替换原版 run 路径（含 `exp_cfg.yaml`、`mvt_cfg.yaml`），仓库根目录运行：

```bash
MODEL_FOLDER=/path/to/original_internal_slots_run
python finetune/RLBench/slot_guided_sam3_online.py \
  --config finetune/RLBench/configs/rlbench_slot_guided_sam3_internal_slots_online.yaml \
  --model-folder "$MODEL_FOLDER" --model-name model_last.pth \
  --eval-datafolder "$PWD/LPY/BridgeVLA_RLBench_EVAL_DATA" \
  --tasks open_drawer --eval-episodes 25 --episode-length 25 --device 0 \
  --slot-sam3-task-repo ../SAM31 \
  --slot-sam3-checkpoint "$PWD/checkpoints/sam3/sam3.pt"
```

默认保存 mask、prompt 和融合诊断到 `outputs/slot_sam3_internal_slots/run_*`。
同一命令加 `--slot-sam3-stage segmentation` 可先运行分割而保持原动作；
加 `--slot-sam3-stage prompts` 则不加载 SAM3。当前原版仅使用 Coarse 提示，Refine 独立预测保留。
新配置只用于冻结推理；继续训练仍使用原 YAML。真实分割和闭环性能尚待服务器验证。

The [role-query online profile](../finetune/RLBench/configs/rlbench_slot_guided_sam3_online.yaml)
uses `slot_guided_sam3_online.py`: current Coarse Guidance → original-camera prompts
→ persistent frozen SAM3 → RGB-D geometry → existing Relation Anchor → action.
No frame export is required. The [offline runner](slot_guided_sam3_offline.md)
supports A/B/C and prompt ablations. Global defaults and the legacy draft stay disabled.

<a id="offline-online-training"></a>

## Offline、Online 与训练

这里的 offline / online 是两种**推理方式**，都不会更新模型权重。

| 对比 | Offline | Online |
| --- | --- | --- |
| 输入 | 保存的单帧 RGB-D、相机参数和机器人状态 | RLBench 每一步的当前观测 |
| 动作处理 | 保存动作预测和诊断，不执行环境动作 | 预测动作 → 环境执行 → 获取下一帧 |
| 模型状态 | BridgeVLA 和 SAM3 全部冻结 | BridgeVLA 和 SAM3 全部冻结，模型常驻加载 |
| 用途 | 可复现的 A/B/C、box-only / point+box 消融 | 当前观测驱动的闭环任务评估 |

Online 不要求先运行 offline，也不表示在线学习或保证实时频率。
TASK 中复用的 `configs/slot_guided_offline.json` 只是提示、候选选择和融合参数；
该文件名不改变在线执行方式。

以下训练步骤针对 role-query 配置；原版 slots 使用上方命令和原训练配置。

1. 准备带 T/R 角色监督的 Semantic-GT replay，以及 BridgeVLA 初始化 checkpoint。
   使用 `rlbench_o2_role_queries.yaml` 训练固定双角色 query、现有 Relation Anchor
   和动作策略，保持 `rvt.enable_slot_guided_sam3=False`。
   完整命令、冻结范围和续训方式见[Role Query 训练](guides/object-conditioning.md#role-query-training)。
2. 训练完成后，使用实际输出的 run 目录中的 checkpoint、`exp_cfg.yaml` 和
   `mvt_cfg.yaml`，运行下方的[在线命令](#commands-and-configuration)，或运行
   [offline 快照实验](slot_guided_sam3_offline.md#commands)。已有兼容 checkpoint 可直接推理。
   推理时冻结整个策略；训练时的部分冻结设置不代表在线仍有参数更新。
3. 当前尚未实现“让 SAM3 几何参与动作微调”。启用新功能时，Agent 的
   `build(training=True)`、`update()` 和 `train()` 均拒绝训练。
   如需冻结 Slot/SAM3 后仅训练 Anchor / Action Head，仍需新增几何 replay、
   训练接口和梯度隔离，不能把 online/offline YAML 直接交给 `train.py` 实现。

## Actual online path

### Startup progress / apparent hangs

Online evaluation prints `[Startup] START` / `DONE` with elapsed seconds for
`agent.eval()`, observation-provider setup, environment construction, SAM3
preload, and CoppeliaSim launch. During preload, `[SlotSAM3]` also identifies
TASK API import and the backend checkpoint/device. Backend loading includes
model construction, device transfer and checkpoint SHA256 computation.

If one of these startup phases takes more than 60 seconds, Python prints all
thread stacks to stderr every 60 seconds until the phase finishes. This is a
diagnostic timer, not a failure or process termination; it is cancelled when
the phase exits. The existing `2>&1 | tee "$RUN_DIR/runtime.log"` captures both
progress and stacks. Inspect the last `START` without a matching `DONE` and
the stack to distinguish imports/model loading/file I/O from simulator launch.
The earlier `Evaluation branch: internal Target/Reference slots` line only
identifies the policy branch; it does not mean SAM3 or the simulator is ready.

After simulator startup, `[Rollout]` reports each advance of the episode
generator with task, zero-based episode, attempt and step. The first advance
includes demo loading, live scene reset, observation preparation, policy/SAM3
inference and action execution. Later advances run the following step or finish
the episode. Each advance has its own 60-second repeating stack diagnostic,
cancelled before the transition is handed back to the evaluator. This also
covers `--visualize`. A completed CoppeliaSim launch only confirms simulator
startup.

Progress is flushed immediately through the existing `tee` command. Each active
phase prints `START` / `DONE` and elapsed seconds. While it is still running,
`WAIT ... elapsed=...s` appears every 5 seconds, using the deepest active phase:
demo loading, scene reset, initial observation capture, baseline policy, current
frame validation, individual SAM3 camera segmentation, geometry fusion,
conditioned policy, or action execution / observation capture. Nested phases
share one logging thread; that thread does not call Torch or the simulator.
The 60-second thread-stack diagnostic remains active during generator advances.
Each policy result prints its status and fallback reason if present; each
returned step prints reward, terminal and timeout without waiting for the end
of the episode. Baseline evaluation without online Slot/SAM3 keeps its existing
output behavior. No additional flags are required.

1. The existing evaluator strictly loads a matching slots/Anchor or role-query/Anchor checkpoint
   and freezes BridgeVLA. SAM3's runtime-checked TASK backend loads once, before the
   simulator launches. A package/API/weight loading failure stops startup.
2. An opt-in `extract_obs` hook captures the actual current RGB, world point
   cloud and `Observation.misc` camera matrices alongside the original policy
   tensor dictionary. It never reads an expert action or uses demo frames as
   perception input. The benchmark's existing demo-based **initial scene reset**
   remains unchanged; no recorded future observation enters the SAM3 path.
3. `OnlineSlotSam3Policy.act` retains the current original Slot action (C) and
   exports its detached coarse points with independent renderer coverage and
   the exact scene inverse. The provider checks the current policy RGB/XYZ,
   intrinsics, extrinsics, low-dimensional state and instruction against that
   captured observation.
4. `OnlineTaskPerception` directly reuses the TASK checkout's
   `coarse_slot_guidance`, `run_camera` and `fuse_geometry`. Each RGB is encoded
   once for both roles. Prompts, depth and cross-view consistency select instances;
   masks and geometry are recomputed for each current observation.
5. The in-memory result supplies fixed `[1,2,N,3]` object points, masks,
   confidence and Reference NULL state. An explicit `OnlineGeometry` validates
   the current RGB-D content, calibration, source index and actual loaded model
   identity before the second policy pass. Existing coarse/refine scene and crop
   transforms are applied. Slot semantic priors/tokens are preserved, and no
   new learned parameters, cross-attention or backbone are introduced.
6. The returned action goes through the existing YARR rollout and RLBench
   `env.step`. The provider instance in that environment is the same instance
   used by the policy wrapper. This first integration uses the evaluator's
   direct single-process rollout, not a copied generic YARR worker agent.

The `online_experiment` scope is an explicit runtime selection. It does not
fabricate a reviewed performance report and cannot satisfy the separate
`reviewed` acceptance gate. Online geometry accepts predicted coarse roles and
real SAM3 only; Oracle labels and Mock backends cannot condition its actions.
Enabled training remains rejected.

## Current camera coordinates and source indices

<a id="refine-guidance"></a>

### Refine 修正定位后生成 SAM3 提示（可选）

原 online 配置仍使用 Coarse Guidance。新增 `--slot-sam3-guidance-stage refine`，
逐角色优先使用**实际运行过的 Refine mask head**，失败时保留 Coarse 并记录原因。
`--slot-sam3-stage prompts` 控制是否执行 SAM3；`guidance-stage` 控制提示的几何来源，两者独立。

```text
Coarse 角色/NULL → 当前局部 RGB/XYZ + 已训练 Refine head
  → 从 Refine 角色图重新提取观测点 → 实际裁剪逆变换 → Coarse cube → world 米
  → 原相机 box/point → SAM3 → RGB-D 融合 → 既有动作条件接口
```

实现位于 [refine_slot_export.py](../finetune/bridgevla/models/refine_slot_export.py)。
不使用 Refine 输出里可能继承自 Coarse 的旧 `points` 或 `geometry[17]` 作为新几何。
独立 coverage render 使用当前场景点和同一组虚拟相机；校验 `rev_trans` 确实还原当前场景。
在支持像素、有限非背景 XYZ 和局部 cube 范围内，读取概率 ≥0.25 的点，
要求选中至少 4 个不同 XYZ，再以 top-k 上限输出。该分数是地图支持强度，未校准为实例身份概率。
所有导出均在 no-grad 下完成；新功能关闭或选择 `coarse` 时不增加 Refine coverage render。

`guidance_selection` 同时写入逐步日志、`prompt_report.json`、推理记录和几何 provenance，
包括每个角色的 `selected_stage`、`failure_reason`、点数和裁剪逆矩阵；
`guidance.json` 的 `transform.source` 也保留来源。TASK v1 的 `guidance_source` 仍为
`predicted_coarse_slot`，表示 Coarse 条件下的预测路线，具体点来源以选择记录为准。
诊断时 `slot_arrays.npz` 的 `slot_*` 是实际提示输入，`coarse_original_*` 保留选择前输出。

| 情况 | 行为 |
| --- | --- |
| 没有局部 mask head / 仅继承重投影 | `REFINE_MASK_HEAD_NOT_RUN`，使用 Coarse |
| 局部不可用或响应不足 | `REFINE_LOCAL_INVALID` / `REFINE_NO_SUPPORTED_RESPONSE`，使用 Coarse |
| 支持点不足 / 重复点 | `REFINE_INSUFFICIENT_POINTS`，使用 Coarse |
| 语义无效 / Reference NULL | 保留原语义；不靠局部亮点复活对象 |
| 逆变换错误 / 非法概率 | 显式失败，按既有 baseline/fail 策略处理 |

回退并不意味着 Coarse 正确；它仍须通过原有 Guidance、投影、深度和候选检查。
不会在失败时构造框，也不会读取未来 observation、expert action 或 Oracle 标签。

**先训练，再启用。** `rlbench_o2_role_queries/11_08_09_27` 的旧权重不含已训练局部头，
不能只换 YAML 评估。复用已有训练配置
[rlbench_o2_internal_slots_refine.yaml](../finetune/RLBench/configs/rlbench_o2_internal_slots_refine.yaml)，
从同为 `role_queries` 的 checkpoint 用 `--init_checkpoint` 初始化，保留兼容的角色与动作权重，
新增局部头重新初始化；不用旧 optimizer resume。以下从仓库根目录执行并替换数据路径：

```bash
SEMANTIC_BUFFER=/data/BridgeVLA_RLBench_SEMANTIC_GT_MASK_VERIFIED_Buffer
COARSE_CHECKPOINT="$PWD/finetune/RLBench/train/rlbench_o2_role_queries/11_08_09_27/model_last.pth"
cd finetune/RLBench
GPUS_PER_NODE=2 bash train_8x40.sh \
  --exp_cfg_path configs/rlbench_o2_internal_slots_refine.yaml \
  --train_replay_storage_dir "$SEMANTIC_BUFFER" \
  --init_checkpoint "$COARSE_CHECKPOINT" \
  --exp_note refine_guidance_s0 --log_dir exp/RLBench \
  --save_optimizer_state --log_backend tensorboard \
  --exp_cfg_opts "bs 2 num_workers 2 pin_memory False"
```

此训练不运行 SAM3，沿用已有 Coarse/Refine 角色、NULL、分离和动作损失；
局部监督是点投影热图，尚非密集实例 mask GT。训练配置的 0.05 gate 让较弱的
Coarse token 进入局部头，最终 Guidance 与 TASK 的默认 0.25 gate 保留。
现有裁剪扰动是 waypoint 扰动；本次没有新增角色漏检模拟、多候选搜索或全局重搜训练。

训练后回到仓库根目录，用**实际新 run 目录**先检查提示。启动校验会拒绝与局部头结构不匹配的旧权重：

```bash
REFINE_MODEL_DIR=/absolute/path/to/trained/refine/run
python -u finetune/RLBench/slot_guided_sam3_online.py \
  --config finetune/RLBench/configs/rlbench_slot_guided_sam3_refine_online.yaml \
  --model-folder "$REFINE_MODEL_DIR" --model-name model_last.pth \
  --eval-datafolder "$PWD/LPY/BridgeVLA_RLBench_EVAL_DATA" \
  --tasks open_drawer --eval-episodes 1 --episode-length 3 --device 0 \
  --slot-sam3-task-repo ../SAM31 \
  --slot-sam3-stage prompts --slot-sam3-guidance-stage refine \
  --slot-sam3-diagnostics-dir "$PWD/eval_debug/refine_prompt"
```

查看 `index.html` 的选择记录、Coarse/Refine 各自热图和原相机提示。环境仍执行 baseline 动作。
确认后将 `--slot-sam3-stage prompts` 改为 `action`，并添加
`--slot-sam3-checkpoint "$PWD/checkpoints/sam3/sam3.pt"` 执行真实 SAM3 路径。
对照实验用同一新 checkpoint、同一配置与 episodes，只切换 `--slot-sam3-guidance-stage coarse`。

**上述残差配置的纠错边界：** 目标在局部观测内且角色 token 尚有效时，可由已训练局部头提供修正证据；
Coarse 硬判无效、错误 NULL、目标在裁剪外或语义选错实例，目前没有全局恢复机制。
这条路径只是具备局部修正接口，不能保证恢复 FN。
本机只验证 CPU/Mock 契约与回归，未测真实 mask IoU、定位误差、FN 恢复率或操作成功率。

<a id="local-relocalize"></a>

### 像 fine key-action 一样独立修正角色定位

新增 `object_conditioning.refine_relocalize: True`，默认 False，所有旧配置不变。
它复用当前 action waypoint 的局部渲染与 VLM 特征，独立预测 T/R 图；
Coarse token 保留角色条件，位置仅为有界软偏置，不再以 Coarse 峰值有效性或三倍 spread 限制搜索。
该模式要求 local mask/query head，并关闭 `refine_heatmap_residual`，避免未知位置被固定的极低投影概率锁住。
不新增 VLM 前向、无序 slot 角色匹配或双向 Cross-Attention。

局部图在当前观测表面预测，按局部峰值重新判断可靠性。只有可靠角色更新局部点与软几何，
否则保留原全局角色上下文。Reference NULL posterior 不重选，空深度和 cube 外 XYZ 不产生局部证据。
SAM3 导出允许经过新模式验证的局部结果恢复 Coarse 低峰值门控；仍要求 coverage、至少 4 个不同 XYZ、
0.25 响应与原相机深度/投影检查。选择记录中的 `coarse_gate_recovered` 表示软件门控被恢复，**不是 GT 检测召回率**。

训练使用 [rlbench_o2_role_relocalize.yaml](../finetune/RLBench/configs/rlbench_o2_role_relocalize.yaml)：
原 Coarse/Refine 图、NULL、角色分离和动作损失保留，增加
`rvt.object_refine_center_loss_weight: 1.0`。该项是**局部 cube 内、经 XYZ 可见性核验的 GT 表面点重心 L1**，
不是完整对象中心或密集分割 GT；没有可靠可见 GT、未知角色或语义不存在时跳过。
它不受预测的 valid/NULL 门控，teacher/XYZ 不接收梯度，未标注像素不伪造成背景。
沿用 waypoint 裁剪扰动，并以 `refine_hint_dropout: 0.25` 在训练时按角色随机移除位置提示，
角色 token、实际 RGB-D、NULL 和全局回退上下文保持原样。评估不执行该 dropout。

以下在仓库根目录初始化新实验；替换 Semantic-GT replay 路径：

```bash
SEMANTIC_BUFFER=/data/BridgeVLA_RLBench_SEMANTIC_GT_MASK_VERIFIED_Buffer
COARSE_CHECKPOINT="$PWD/finetune/RLBench/train/rlbench_o2_role_queries/11_08_09_27/model_last.pth"
cd finetune/RLBench
GPUS_PER_NODE=2 bash train_8x40.sh \
  --exp_cfg_path configs/rlbench_o2_role_relocalize.yaml \
  --train_replay_storage_dir "$SEMANTIC_BUFFER" \
  --init_checkpoint "$COARSE_CHECKPOINT" \
  --exp_note role_relocalize_s0 --log_dir exp/RLBench \
  --save_optimizer_state --log_backend tensorboard \
  --exp_cfg_opts "bs 2 num_workers 2 pin_memory False"
```

SAM3 在训练中关闭。现有 role-query checkpoint 可保留兼容权重并初始化新局部头；
不要用旧 optimizer resume，也不要直接加载旧残差头到独立头。
checkpoint 记录新路由，评估会拒绝用未训练该模式的旧权重冒充新模型。

若遇到 `AttributeError: 'GemmaModel' object has no attribute 'model'`，先更新代码再重跑原命令。
[模型结构兼容工具](../finetune/bridgevla/utils/paligemma_compat.py)支持旧 causal-LM wrapper、直接 decoder
及嵌套 PaliGemma core；冻结层、projector 和高效前向 norm hook 使用实际模块，分层学习率兼容两类名称。
`--init_checkpoint` 仅在目标键名与形状匹配时转换权重名称，不恢复 optimizer，不忽略 backbone 缺失。

回到仓库根目录，训练后先做提示诊断：

```bash
MODEL_DIR=/absolute/path/to/trained/role_relocalize/run
python -u finetune/RLBench/slot_guided_sam3_online.py \
  --config finetune/RLBench/configs/rlbench_slot_guided_sam3_relocalize_online.yaml \
  --model-folder "$MODEL_DIR" --model-name model_last.pth \
  --eval-datafolder "$PWD/LPY/BridgeVLA_RLBench_EVAL_DATA" \
  --tasks open_drawer --eval-episodes 1 --episode-length 3 --device 0 \
  --slot-sam3-task-repo ../SAM31 --slot-sam3-stage prompts \
  --slot-sam3-diagnostics-dir "$PWD/eval_debug/role_relocalize"
```

诊断显示 `relocalized`、Coarse/Refine 各自定位图和最终原图提示；环境仍执行当前 baseline 动作。
真实 SAM3 模式改用 `--slot-sam3-stage action` 并添加实际 `--slot-sam3-checkpoint`。
用同一新 checkpoint 将 `--slot-sam3-guidance-stage` 设为 `coarse` 或 `refine` 比较提示来源；
新旧定位架构则需按相同数据、初始化、训练预算重新训练后配对评估。

当前仍共享 action 裁剪，目标完全在裁剪外、语义选错实例或错误 NULL 没有全局恢复保证；
独立角色裁剪/重搜尚未实现。CPU 合成优化测试只验证能从错误先验学习移动局部定位，
Mock 验证软件链路；真实 FN、3D 误差、IoU、延迟及 paired task success 仍需实测。

<a id="slot-prompt-probe"></a>

### 先检查 Slot 图与 Prompt：不运行 SAM3

同一 `slot_guided_sam3_online.py` 入口新增显式 `--slot-sam3-stage prompts`。
默认 `action` 模式不变。`prompts` 模式只执行当前 baseline 前向、导出 Slot 和构造提示，
不加载 SAM3、不分割、不融合、不进行第二遍几何条件动作预测。环境仍按原 baseline
动作推进，并非只读观察或静止场景。checkpoint 的启动完整性校验保留，未加入摘要缓存。

下面使用用户最近的仓库/模型路径；TASK 假设为同级 `SAM31`。按实际部署替换目录。
先用 1 episode、3 steps 检查；只看初始观测可设 `--episode-length 1`，但仍会执行一步 baseline 动作。

```bash
cd /media/34.92/common-data-2t/project/BridgeVLA
RUN_DIR="$PWD/eval_debug/slot_prompt_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
set -o pipefail
python -u finetune/RLBench/slot_guided_sam3_online.py \
  --model-folder "$PWD/finetune/RLBench/train/rlbench_o2_role_queries/11_08_09_27" \
  --model-name model_last.pth \
  --eval-datafolder "$PWD/LPY/BridgeVLA_RLBench_EVAL_DATA" \
  --tasks open_drawer --eval-episodes 1 --episode-length 3 --device 0 \
  --slot-sam3-task-repo ../SAM31 \
  --slot-sam3-stage prompts \
  --slot-sam3-diagnostics-dir "$RUN_DIR/probe" \
  2>&1 | tee "$RUN_DIR/runtime.log"
```

这里无需 `--slot-sam3-checkpoint` 或 `--visualize`，也无需安装 SAM3；BridgeVLA 模型、
TASK、RLBench/PyRep 和 GPU 渲染依赖仍然需要。`--check-config` 可先检查参数而不启动模型：

```bash
python finetune/RLBench/slot_guided_sam3_online.py --check-config \
  --slot-sam3-stage prompts --slot-sam3-diagnostics-dir /tmp/slot_prompt_probe
```

输出目录为 `probe/run_<id>/episode_<counter>/frame_<call>/`，终端打印具体 `index.html` 路径：

| 文件 | 含义 |
| --- | --- |
| `index.html` | 静态汇总页；查看本地副本时复制整个 frame 目录 |
| `coarse/view_*.png` | Coarse 自身虚拟 RGB、Target/Reference 软角色图和动作热图 |
| `refine/view_*.png` | Refine 自身局部虚拟 RGB 和对应角色/动作图；单阶段模型无此目录 |
| `<camera>.diagnostic.png` | 当前原相机 RGB、框和正负提示点；没有 SAM mask 是正常的 |
| `slot_arrays.npz` | 原始概率/logits、角色有效性、点支持及坐标变换，非二值实例 mask |
| `guidance.json` / `prompt_report.json` | 角色状态、NULL、置信度、提示失败原因及当前帧标定身份 |

角色图按固定绝对概率着色；动作热图单独使用相对空间 softmax。不能把动作位置热图当成
对象实例分割。当前 role-query 配置的 Refine 继承 Coarse 角色；报告标明 `roles_inherited`、
`masks_refined`、几何 `valid` 与语义 `role_token_valid`，以及 Refine 裁剪中心、比例和
`matrix_world_from_refine_cube`。Coarse 是全局 cube，Refine 是局部 crop，两个阶段的像素
不能直接叠在同一幅图上比较。若 Target 在 Coarse 已错，先查角色选择；若裁剪中心偏离目标，
先查 Coarse 动作定位；若目标位于 crop 内但 Refine 图仍偏，保留上述数组与图片继续定位。

JSONL 使用 `status=prompts_only`、`sam3_executed=false`、`action_source=baseline_C`，
它不是 `conditioned`，也不提供 GT mask IoU、正确实例命中率或任务效果提升证据。
CPU/Mock 测试覆盖真实 TASK 提示、分阶段保存、原动作保留和禁止 SAM/融合调用；本机未运行真实模型/模拟器。

### Multi-step dtype binding and step-log interpretation

YARR converts float64 observation arrays to float32 when creating reset history,
but appends subsequent raw observations without that conversion. With a
non-float32-exact K, comparing a later float64 tensor directly against the
provider's float32 binding can incorrectly report
`front_camera_intrinsics: policy tensors do not match the cached current frame`.
The provider now compares the latest B,T entry in the same float32 representation
used at capture/reset, retaining the existing equality tolerance. It does not
modify YARR or policy observations. Tests run the actual rollout generator for
three calls with one/two history entries, moving wrist poses and nonexact focal
values; genuine changed K/T, stale frames and RGB-D mismatches still fail.

The supplied `open_drawer` step log contained 30 policy calls across episodes
0–9, with 7 `conditioned` calls, 20 calibration-binding fallbacks and 3 fallbacks
without reliable Target geometry. All 10 captured frames were `simstep-0`;
later calls were rejected before SAM3. This confirms that the route reached
action conditioning on some initial frames, not that the intended instance was
correct or that any episode succeeded. The file has no success/GT mask/action
labels, so role accuracy, mask IoU and paired success rate cannot be inferred.

Observed medians were 0.265 s for baseline_C, 42.425 s total for conditioned
calls, 20.877 s for the fusion phase, and 21.116 s for conditioned_action.
The segmentation-phase median was 0.121 s across ten attempts, including prompt
failures/skips; it is not a per-image SAM3 speed benchmark. The maximum logged
allocator peak was 13.14 GiB during per-camera segmentation measurement,
including resident models, not SAM3-exclusive memory or a full-frame peak.
Fusion and action authorization both currently compute full-file Coarse
checkpoint SHA256 digests. These repeated reads are a suspected major contributor
to the two roughly 21 s phases; their isolated cost has not been profiled here.
Full-file integrity checks remain in place, and no latency improvement is claimed.

Step diagnostics now include an episode-local `policy_call_index`,
`camera_roles.<camera>.<role>` prompt failure/candidate counts/rejection reasons,
and `roles.<role>` fusion failure reason, role confidence and supporting cameras.
Mask arrays are excluded from JSONL. To inspect actual boxes, masks and overlays,
append the existing option to the original online command:

```bash
--slot-sam3-diagnostics-dir /absolute/path/to/slot_sam3_diagnostics
```

Each invocation creates its own run directory. Keep that directory alongside
`slot_sam3_steps.jsonl` and episode result files. An empty/low-confidence Target
still falls back to the current baseline; do not lower gates solely to increase
the number of `conditioned` rows. The recorded Reference `NULL` is the model's
semantic prediction and is not a verified GT role label.

If a frame reports `current online RGB-D unavailable` with the old combined
`rigid zero-skew pinhole camera` error, its SAM3 geometry was not used: the
policy returned the original current-frame action as `baseline_fallback`.
That message alone does not identify which camera/condition failed.

The earlier `1e-7` rigid-rotation bound could reject single-precision simulator
arithmetic. Both the BridgeVLA observation adapter and TASK's slot prompt/snapshot
validator now use `1e-5` absolute bounds for maximum `R.T @ R - I` error and
`abs(det(R) - 1)`. They retain the supplied matrices, without orthogonalizing
or changing the world frame. Zero-skew K/homogeneous T bounds remain `1e-10`;
larger scale/shear, reflections, stale calibration and pixel/world round-trip
failures are still rejected. This is a bounded numerical tolerance, not proof
that arbitrary camera metadata is correct.

Update **both** checkouts and restart evaluation:

```bash
git -C /absolute/path/to/BridgeVLA pull --ff-only origin master
git -C /absolute/path/to/TASK_checkout pull --ff-only origin main
```

The TASK path must be the directory supplied to `--slot-sam3-task-repo`
(named `SAM31` on the user's server). A remaining failure now includes the
camera name and specific residual/determinant/threshold. Successful captures
record these under `current_frame.camera_calibration_checks` in online step
diagnostics. The measured camera parameters were not supplied with the original
error, so the deployed frame still needs rerunning to distinguish numerical
roundoff from a genuinely invalid calibration. No model/config precision flag
or Qwen/SAM3 default-path change is required.

All object geometry stays in RLBench world metres. The hook takes each wrist
camera matrix from the **current** raw observation, independently for every
step. It does not look up calibration using the policy-step/sample counter or
a saved expert frame index.

PyRep camera intrinsics can have paired negative focal signs. The adapter
explicitly changes camera basis with `D=diag(-1,-1,1,1)`:

```text
K_optical = K_raw @ D[:3,:3]
T_world_optical = T_world_raw @ D
```

This is a proper rotation; RGB pixels and world points remain unchanged.
Positive focal lengths then satisfy TASK's optical-camera convention without
flipping the image. A single-axis reflection or missing/invalid calibration is
rejected. Tests include general float32 rotations, dynamic wrist poses and
projection/backprojection consistency.

The ordinary policy observation requests a world point cloud rather than a
depth channel. That cloud came from the same current sensor depth/pose in
RLBench. The provider recovers camera Z-depth through the current inverse
extrinsics and verifies the full registered pixel/world round-trip. This is
measured geometry, not a guessed object range; invalid or behind-camera points
remain unsupported. The receipt records this depth source and camera-basis
conversion. Existing policy inputs and observation element shapes are unchanged.

Source indices are **live simulation microsteps**, obtained from PyRep's actual
simulation clock relative to the first capture of an episode. They include
microsteps inside a planned action; they are distinct from policy-step count
and recorded demonstration frame indices. Capture checks that simulation did
not advance while making the snapshot, and consumption rejects a stale cache.
Reset creates a new episode identity, clears old geometry/frame bookkeeping
and retains the loaded models. No recording step callback is replaced.

## Commands and configuration

From the BridgeVLA root, this CPU check loads no model or simulator:

```powershell
& .\.venv-oht\Scripts\python.exe finetune/RLBench/slot_guided_sam3_online.py --check-config
```

It reports `enabled=true`, `execution_mode=online_experiment` and resolved
runtime settings. `--check-config` also honors `--slot-sam3-*` overrides; it
checks configuration rather than claiming package/weight/GPU readiness.

On a prepared GPU/CoppeliaSim/RLBench model environment, run the following from
the BridgeVLA root, replacing paths with actual data/checkpoints. The matching
saved `mvt_cfg.yaml` must be beside the Bridge checkpoint, or passed through
`--mvt_cfg_path`. Saved architecture/routing metadata and weights must match
the online role-query profile; an unrelated Slot/refine checkpoint is not
converted by changing its config.

```bash
python finetune/RLBench/slot_guided_sam3_online.py \
  --model-folder /models/bridgevla_role_queries \
  --model-name model_60.pth \
  --eval-datafolder /data/RLBench_EVAL_DATA \
  --tasks open_drawer --eval-episodes 25 --episode-length 25 --device 0 \
  --slot-sam3-task-repo /path/TASK_Relevant_object-main \
  --slot-sam3-checkpoint /models/sam3.pt \
  --slot-sam3-prompt-mode point+box
```

The local reference checkout is
`C:\Users\yiwei.chen\TASK_Relevant_object-main`. The GPU runtime must have access
to that checkout (or its equivalent on that machine), its dependencies and the
SAM3 version documented in `docs/sam3_api_contract.md`. Source-only API checks
and the CPU Mock suite do not establish model-weight inference readiness.

Runtime fields are under `slot_guided_sam3`:

| Config key | CLI override | Meaning |
| --- | --- | --- |
| `stage` | `--slot-sam3-stage` | `action` (default) or `prompts` (baseline action; no SAM3) |
| `task_repo` | `--slot-sam3-task-repo` | Existing TASK implementation |
| `sam3_checkpoint` | `--slot-sam3-checkpoint` | Actual local SAM3 weights |
| `perception_config` | `--slot-sam3-perception-config` | TASK prompt/selection/fusion JSON |
| `sam3_device` | `--slot-sam3-device` | Defaults to the policy CUDA device |
| `prompt_mode` | `--slot-sam3-prompt-mode` | `point+box` or `box-only` |
| `fallback` | `--slot-sam3-fallback` | `baseline` or `fail` |
| `diagnostics_dir` | `--slot-sam3-diagnostics-dir` | Optional in action stage; required for prompts stage |

Runtime paths resolve from the BridgeVLA root. A relative `perception_config`
resolves inside the TASK checkout; its existing `slot_guided_offline.json` name
is retained because the same prompt/selection/fusion parameters are reused in
memory. The online command does not invoke its offline file runner. Select the
architecture through `--config`; a second `--exp_cfg_path` is rejected.

The launcher generates a unique online evaluation log name unless `--log-name`
is explicitly supplied. Standard RLBench episode success results remain in the
normal evaluator's CSV/journals. `slot_sam3_steps.jsonl` records current-frame
receipts, role validity/status/confidence, timings, measured memory scope,
fallback and error reasons. It also records fail-mode errors before shutdown.
Optional diagnostic writes get a unique run subdirectory and preserve SAM3
masks, scores, prompts, camera overlays and fused object geometry; these files
are not read back into the action path.

<a id="sam3-source-contract"></a>

## SAM3 运行时兼容检查

2026-11-09 按用户要求移除了 TASK 后端的固定源码哈希/commit 门禁。
安装的 SAM3 不再因为源码与参考 revision 不同而被拒绝；无需切换到指定 commit，
也不需要前一版说明中的独立源码检出或 `PYTHONPATH` 启动包装。

如果仍看到 `unverified SAM3 API: source mismatch ...`，服务器使用的 TASK 后端
尚未更新。下面的 `/data/project/SAM31` 是用户部署的 TASK 工程目录，更新该仓库：

```bash
git -C /data/project/SAM31 pull --ff-only origin main
```

随后直接使用本页的普通 `python finetune/RLBench/slot_guided_sam3_online.py`
命令，传入 `--slot-sam3-task-repo /data/project/SAM31`。

实际交互 predictor、point/box 参数签名、原图尺寸、候选 mask 形状、数值和几何
质量检查仍保留。模型加载或真正不支持的接口仍会明确报错。
签名与尺寸检查不能证明所有版本的坐标语义相同，改变实现后仍需用代表性标定帧
检查 prompt/mask 对齐及实际推理。

源码哈希只记录实际可读文件作为缓存溯源，不再与固定值比较，缺少源码文件也不会
阻止启动。指纹记录实际包路径、版本及 `source_hash_enforced=false`，不再将参考
commit 写成实际安装版本。checkpoint、mask 内容和缓存完整性校验保持不变。
原 Qwen/SAM3 入口保持原样，详细调用约定见 TASK 的 `docs/sam3_api_contract.md`。

若权重加载成功后在启动 CoppeliaSim 时报 `xcb` / `cv2/qt/plugins`，见
[Qt 插件路径与显示排查](guides/evaluation.md#rlbench-qt)。当前入口在 SAM3 加载后、
模拟器启动前修复 OpenCV 的 Qt 环境覆盖；仍需设置真实 `COPPELIASIM_ROOT` 和可用显示。

<a id="runtime-traceback"></a>

## 运行时错误与 Float / BFloat16 排查

若评测只打印 `RuntimeError: expected scalar type Float but found BFloat16`，
不能仅凭这一行判断是 PaliGemma、动作/几何算子还是 SAM3 出错。
PaliGemma 主干按 BF16 加载，Slot/动作特征进入相应模块前有 FP32 转换；
SAM3 在设备支持时另有 BF16 autocast。需用调用栈确定发生冲突的边界。

旧 online 启动器会将评测过程中的异常截成单行。当前启动器只对配置错误保留简短
提示和退出码 2；评测启动后的异常会正常传播，显示完整 traceback 和异常链，
未捕获异常按 Python 默认方式非零退出。启动器的这一改动负责保留诊断信息，
已定位的渲染算子故障及其精度修复见下文。

已确认的一类故障出现在第一遍 BridgeVLA 前向的 `PointRenderer.render_batch`，
调用栈终点为 `render_feature_pointcloud_to_image`。该 CUDA 内核通过
`data_ptr<float>()` 读取深度，但此前的投影 `einsum` 在外层 autocast 下可产生
BF16 深度，即使输入点与相机矩阵原本为 FP32。此阶段还没有执行当前帧的 SAM3 分割。

当前修复将渲染点、特征及投影矩阵转换为 FP32，并在投影区域局部禁用 autocast。
像素索引和深度从 FP32 运算得到，避免先用 BF16 投影再转回 FP32 所造成的位置精度损失。
退出该区域后恢复调用方的 autocast；PaliGemma/SAM3 精度配置与权重保持原样。
FP32 基线与 BF16 autocast 下的投影索引、深度在 CPU 回归测试中逐值一致，
覆盖正交/透视相机以及 Python/TorchScript 投影；CUDA 内核由严格检查输入类型的
Mock 替代，真实 GPU/RLBench 闭环仍需在部署环境验证。

此改动只有 Python 代码，已有可用 CUDA 扩展且按 editable 方式安装时无需重编译。
若 traceback 指向其他安装目录，先确认 point-renderer 使用本仓库源码；按本页安装环境
重新执行 `python -m pip install --no-build-isolation -e finetune/bridgevla/libs/point-renderer`。
评测运行指纹同时纳入 point-renderer 的 Python 源码，避免旧运行错误地复用新渲染路径。

更新 BridgeVLA 后重跑原评测参数，可保存两路日志：

```bash
git -C /data/project/BridgeVLA pull --ff-only origin master
cd /data/project/BridgeVLA
set -o pipefail
python -u finetune/RLBench/slot_guided_sam3_online.py \
  --model-folder /data/project/BridgeVLA/finetune/RLBench/train/rlbench_o2_role_queries/11_08_09_27 \
  --model-name model_last.pth \
  --eval-datafolder /data/project/BridgeVLA/LPY/BridgeVLA_RLBench_EVAL_DATA \
  --tasks open_drawer --eval-episodes 25 --episode-length 25 --device 0 \
  --slot-sam3-task-repo /data/project/SAM31 \
  --slot-sam3-checkpoint /data/project/BridgeVLA/checkpoints/sam3/sam3.pt \
  --slot-sam3-prompt-mode point+box \
  2>&1 | tee slot_sam3_online.log
```

以上为用户部署路径示例；换机器时替换目录。保留从 `Traceback (most recent call last)`
到最终异常的内容，包括 `direct cause` / `During handling` 部分，才能定位实际算子。

<a id="paligemma-checkpoint-loading"></a>

## PaliGemma checkpoint 命名兼容

如果程序已经找到 TASK 工程，但在 `loading state_dict for MVT` 报出成对的
`Missing key(s)` / `Unexpected key(s)`，先比较两组参数路径。已观察到的日志是
604 个缺失键与 604 个多余键，全部符合下面的 PaliGemma 模块布局变化。
下表省略共同的 `mvt1.model.` 前缀：

| 较旧布局 | 较新布局 |
| --- | --- |
| `vision_tower.*` | `model.vision_tower.*` |
| `multi_modal_projector.*` | `model.multi_modal_projector.*` |
| `language_model.model.*` | `model.language_model.*` |
| `language_model.lm_head.*` | `lm_head.*` |

这两种结构可分别在 Transformers 的 [v4.51.3 源码](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/paligemma/modeling_paligemma.py)
和 [v4.53.3 转换规则](https://github.com/huggingface/transformers/blob/v4.53.3/src/transformers/models/paligemma/modeling_paligemma.py)中核对。
仅凭键名不能确定服务器的具体包版本。

共享评测加载器现在会针对实际模型执行上述双向键名转换，覆盖 online 和 offline
入口。每次重命名必须找到同形状目标参数；重复映射或形状不一致会明确报错。
转换保留原 tensor，不改写 `.pth`，角色架构元数据校验及严格加载继续生效。
真正缺少的 Slot、Adapter 或动作头权重仍会失败。

更新 BridgeVLA 后重新运行原命令；TASK 路径使用服务器实际目录即可，例如
`--slot-sam3-task-repo /data/project/SAM31`。服务器根目录示例：

```bash
git -C /data/project/BridgeVLA pull --ff-only origin master
python -m pip show transformers torch
```

若发生命名转换，启动日志会出现
`PaliGemma checkpoint compatibility: remapped ... parameter keys`。
这项修复针对推理权重加载；它不转换训练 optimizer state，也不保证不同 Transformers
版本的前向数值完全相同。复现实验仍应统一并记录训练与评测的依赖和处理器设置。

CPU 验证命令（无需下载模型权重或启动模拟器）：

```bash
python -m pytest tests/test_paligemma_checkpoint_compat.py -q
```

测试包含真实小型 Hugging Face PaliGemma 的键名覆盖、双向严格加载、参数及小模型
预测一致、原 checkpoint 文件不变、元数据拒绝和实际架构错误仍报错。
完整服务器 checkpoint 与 RLBench rollout 需要在模型环境中重试。

## Fallback and practical limits

Missing/invalid calibration, failed prompts, exceptions, low-confidence or empty
Target geometry, a changed checkpoint or an outdated frame cause an explicit
fallback to the **current C action** under `fallback=baseline`. No previous
action/geometry is reused. `fallback=fail` raises and the evaluator shuts down.
Reference NULL, present-but-unseen and uncertain roles retain distinct states.
Privileged alignment, Oracle/expert replay and evaluation resume are rejected
for this initial online route; the original evaluator keeps those features when
the new flag is off.

Online means a current-observation closed loop; it does not assert a hard
real-time frequency. The policy uses two neural passes, segmentation per camera
and geometric fusion. The present TASK and Bridge provenance checks also each
hash the Bridge checkpoint per step, so timing includes that I/O. SAM3's weight
fingerprint is cached. Optimize this only with a validated loaded-checkpoint
identity guard, not by silently discarding source checks.

Per-frame wall time includes current inference and enabled diagnostic writes;
model loading happens before simulator launch. Reported peak memory is explicitly
scoped to the SAM camera allocator measurement; a full policy-device/frame peak
is not claimed by that field. No real frequency, mask/role accuracy, action
error, VRAM result or task-success improvement has been measured on this
workstation, which lacks an installed SAM3/CUDA model environment. See the
[validation record](slot_guided_sam3_online_validation.md).
