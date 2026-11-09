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
   and freezes BridgeVLA. SAM3's audited TASK backend loads once, before the
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

## SAM3 源码版本不匹配

若启动报 `unverified SAM3 API: source mismatch in model/sam3_image.py`，说明实际导入的
SAM3 源码与 TASK 的四文件审计哈希不同；这不是 BridgeVLA checkpoint 加载错误，
也不能仅凭哈希差异判断新版本的 API 是否兼容。相关约定见 TASK 工程的
`docs/sam3_api_contract.md`。当前审计 revision 为
`f1647e8d0c1d280991f4d1b7d974c2704537fe66`。

可以单独检出审计源码，只为本次评测选择它，不替换现有环境的 SAM3 安装。
从服务器 BridgeVLA 根目录执行；示例中的 `/data/project/SAM31` 是用户部署的
TASK 工程目录，不是固定目录名。首次创建下面的新源码目录：

```bash
SAM3_SOURCE=/data/project/deps/sam3-f1647e8
mkdir -p /data/project/deps
git clone https://github.com/facebookresearch/sam3.git "$SAM3_SOURCE"
git -C "$SAM3_SOURCE" checkout --detach f1647e8d0c1d280991f4d1b7d974c2704537fe66
```

如果已存在该源码目录，核对其 remote、工作区和 revision 后复用；不要在 TASK
工程目录切换这个 SAM3 commit。先做纯源码校验，无需加载模型权重或模拟器：

```bash
PYTHONPATH="$SAM3_SOURCE${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
import importlib.util
from pathlib import Path
import sys

spec = importlib.util.find_spec("sam3")
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError("SAM3 source is not on the selected Python path")
package_root = Path(next(iter(spec.submodule_search_locations))).resolve()
sys.path.insert(0, "/data/project/SAM31")
from slot_guided_sam3 import AUDITED_SAM3_SHA, verify_sam3_api_contract

hashes = verify_sam3_api_contract(package_root)
print("SAM3 source:", package_root)
print("API source audit:", AUDITED_SAM3_SHA, "files:", len(hashes))
PY
```

启动评测时，将原命令的 `python finetune/RLBench/slot_guided_sam3_online.py`
替换为下面的**命令前缀**，并在其后接回原有 `--model-folder`、`--model-name`、
`--eval-datafolder`、`--tasks` 和所有 `--slot-sam3-*` 参数：

```bash
PYTHONPATH="$SAM3_SOURCE${PYTHONPATH:+:$PYTHONPATH}" python -c 'import runpy, sam3; print("SAM3 source:", sam3.__file__); runpy.run_path("finetune/RLBench/slot_guided_sam3_online.py", run_name="__main__")'
```

这里先绑定选定的 `sam3` 包，随后 TASK 插入自己的模块目录也不会改选另一份包。
`PYTHONPATH` 只作用于这一条命令；原 Qwen/SAM3 命令和已安装的包保持原样。
源码校验、运行时参数签名检查及正常模型加载仍会执行，未关闭任何验证。
这只证明所选四文件与审计源码一致；GPU 依赖、权重和实际分割仍需在模型环境验证。
若需要支持另一份源码，应审查其 image state、原图坐标转换及联合提示实现，再更新
TASK 的审计记录，不能只替换哈希或仅依据函数参数名放行。

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
