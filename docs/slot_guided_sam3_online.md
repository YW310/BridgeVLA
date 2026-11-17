# Slot-guided SAM3 online RLBench inference

Use [rlbench_slot_guided_sam3_online.yaml](../finetune/RLBench/configs/rlbench_slot_guided_sam3_online.yaml)
and `finetune/RLBench/slot_guided_sam3_online.py` for the current-observation
control loop. Models are frozen and persistent. Each policy step runs Coarse
Guidance → original-camera prompts → SAM3 → RGB-D object geometry → the existing
Relation Anchor → action, then the normal RLBench environment executes that
action. There is no prerequisite to export a frame JSON/NPZ or run the offline
command before each step.

The [offline entrypoint](slot_guided_sam3_offline.md) remains available for
reproducible A/B/C and box-only versus point+box experiments. The legacy draft
profile stays disabled for old commands. Global defaults remain disabled.

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

当前支持的训练与使用顺序：

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

1. The existing evaluator strictly loads a matching role-query/Anchor checkpoint
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
   `coarse_slot_guidance`, `run_camera` and `fuse_geometry`. SAM3 encodes each
   original RGB image once, sharing its state across Target/Reference. Candidate
   identity uses prompts and geometry, including cross-view consistency. The
   same frozen backend persists; image features, masks and object geometry are
   freshly computed for each source observation.
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
| `task_repo` | `--slot-sam3-task-repo` | Existing TASK implementation |
| `sam3_checkpoint` | `--slot-sam3-checkpoint` | Actual local SAM3 weights |
| `perception_config` | `--slot-sam3-perception-config` | TASK prompt/selection/fusion JSON |
| `sam3_device` | `--slot-sam3-device` | Defaults to the policy CUDA device |
| `prompt_mode` | `--slot-sam3-prompt-mode` | `point+box` or `box-only` |
| `fallback` | `--slot-sam3-fallback` | `baseline` or `fail` |
| `diagnostics_dir` | `--slot-sam3-diagnostics-dir` | Optional per-frame mask/geometry output |

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
