# Slot-guided SAM3: runnable offline action experiments

`finetune/RLBench/slot_guided_sam3_offline.py` connects the current observation to
the frozen policy. The enabled profile is
[rlbench_slot_guided_sam3_offline.yaml](../finetune/RLBench/configs/rlbench_slot_guided_sam3_offline.yaml).
The old `*_draft.yaml` stays disabled for compatibility; it is no longer the
experiment entrypoint. No training, simulator rollout or robot controller is
started by this command.

## What runs

1. Load the checkpoint's saved `exp_cfg.yaml` / `mvt_cfg.yaml`. Check the selected
   profile and checkpoint routing metadata, load existing weights strictly, and
   freeze the entire model. The supported first architecture is the existing
   two-role query policy with role-conditioned Relation Anchor. Other coarse/
   refine architectures retain their existing paths and fail an incompatible
   offline selection explicitly.
2. Load one calibrated RGB-D snapshot and its current four-value policy state.
   Run configuration C: the original Slot action path with external geometry off.
   A callback exports detached **coarse** role points and per-point renderer
   coverage, using the actual current scene inverse from `place_pc_in_cube`.
   It never substitutes refine coordinates or assumes cube XYZ is world XYZ.
3. Export B Guidance through TASK's existing producer; project original-camera
   prompts, run frozen SAM3 with one image state per camera, preserve candidates,
   and fuse masks with registered depth. TASK's configuration, caches, diagnostics
   and instance-selection rules remain the source of these operations.
4. Load the resulting world-metre object geometry. Validate the exact observation
   receipt, calibration and real model hashes, transform into the policy's scene
   and crop frames, and predict a second action through the existing Anchor.
   Both actions use the same checkpoint and current observation. There are no
   action gradients through SAM3 and no new learned checkpoint parameters.

`OfflineGeometry` permits this explicit `offline_experiment` prediction without
an acceptance report. It cannot satisfy the separate reviewed activation route.
Mock evidence is rejected by the action path. `train.py` rejects enabled geometry
training, and the ordinary `eval.py` rejects this offline profile before rollout.
Global `enable_slot_guided_sam3=False` remains unchanged.

## Inputs: one actual observation

Use the TASK checkout containing PR1–PR3, for example
`C:\Users\yiwei.chen\TASK_Relevant_object-main`. A prepared BridgeVLA inference
environment, matching role-query checkpoint, PaliGemma model/processor cache,
renderer dependencies, TASK dependencies and the locally installed audited SAM3
package plus its checkpoint are required for real inference. The SAM3 source
contract is documented in that repository's `docs/sam3_api_contract.md`; an
arbitrary installed SAM version is rejected by its backend. A source-only API
audit does not prove that weights can execute on this machine.

`frame.json` uses TASK's strict `slot_rgbd_frame` schema:

```json
{
  "schema_version": 1,
  "artifact_type": "slot_rgbd_frame",
  "episode_id": "episode0",
  "frame_id": "sample-2",
  "source_frame_index": 17,
  "observed_at": "2026-11-05T00:00:00+00:00",
  "cameras": [
    {
      "name": "front",
      "rgb_path": "front.png",
      "depth_path": "front.depth.npy",
      "depth_encoding": "meters_npy",
      "source_frame_index": 17,
      "observed_at": "2026-11-05T00:00:00+00:00",
      "calibration_source": "actual current RLBench observation.misc",
      "intrinsics": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
      "extrinsics": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    }
  ]
}
```

The identity matrices above illustrate structure only. Replace them with the
actual original-resolution intrinsics and `T_world_camera` in metres. Supply all
policy cameras (`front`, `left_shoulder`, `right_shoulder`, `wrist`), each at its
current source episode index and timestamp. RGB/depth must be registered at the
same original resolution. Packed RLBench RGB24 depth is also supported when the
snapshot includes the current near/far. Wrist calibration is never indexed by
the display `frame_id` or sample counter. Invalid depth stays unsupported rather
than becoming a point at the world origin.

`state.json` is separate from TASK's strict snapshot schema:

```json
{
  "schema_version": 1,
  "artifact_type": "slot_policy_state",
  "episode_id": "episode0",
  "frame_id": "sample-2",
  "source_frame_index": 17,
  "observed_at": "2026-11-05T00:00:00+00:00",
  "low_dim_state": [1.0, 0.02, 0.02, 0.4],
  "instruction": "pick the cup",
  "task_name": "pick_cup"
}
```

Use the actual current checkpoint-compatible gripper/proprioception/time
vector, not the example values. State and all camera calibration must match
the same observation exactly. The importer opens only named single-frame files;
it does not load an episode pickle or an expert future action. The coarse and
SAM3 passes are bound to the same RGB-D content digest as well as calibration.

## Commands

Run from the BridgeVLA root. This first command is executable on the prepared
local CPU environment without SAM3, model weights or a simulator:

```powershell
& .\.venv-oht\Scripts\python.exe finetune/RLBench/slot_guided_sam3_offline.py --check-config
```

For real inference, select the Python from your GPU model environment. Paths
below are input locations to replace with actual current-frame exports and
existing model weights. `--check-inputs` validates file/config/frame contracts
without loading weights; it does not certify package/GPU readiness:

```powershell
$ModelPython = 'python'
$SlotArgs = @(
  '--config', 'finetune/RLBench/configs/rlbench_slot_guided_sam3_offline.yaml',
  '--checkpoint', 'checkpoints/role_queries/model.pt',
  '--task-repo', 'C:\Users\yiwei.chen\TASK_Relevant_object-main',
  '--sam3-checkpoint', 'checkpoints/sam3.pt',
  '--frame', 'data/current/frame.json',
  '--state', 'data/current/state.json',
  '--device', 'cuda:0'
)
& $ModelPython finetune/RLBench/slot_guided_sam3_offline.py @SlotArgs --check-inputs
& $ModelPython finetune/RLBench/slot_guided_sam3_offline.py @SlotArgs --mode point+box --output-dir outputs/slot_B_point_box
& $ModelPython finetune/RLBench/slot_guided_sam3_offline.py @SlotArgs --mode box-only --output-dir outputs/slot_B_box_only
```

Saved checkpoint configs are loaded from beside `model.pt`, or from explicit
`--exp-cfg` and `--mvt-cfg`. Their architecture and inference routing must match
the profile. The command does not rewrite them or partially load an incompatible
checkpoint. `--perception-config` selects TASK's existing offline JSON config;
without it, TASK uses its own validated defaults.

For A, add `--guidance-source oracle_gt --oracle-labels data/current/oracle.json`
and use a separate output directory. `oracle.json` contains the same four identity
fields, explicit `reference_is_null`, and `role_masks` mapping **every** camera
to role-specific original-resolution boolean `.npy` paths. Example entry:
`"role_masks": {"front": {"target": "front.target.npy", "reference": "front.reference.npy"}}`.
An optional `instance_ids` maps target/reference to known instance IDs. NULL may
omit Reference masks; an existing unseen Reference needs explicit empty masks.
The producer backprojects only current observed GT surfaces. C's action is saved
alongside each A/B run, using the original Slot path. No GT label reaches the
baseline or determines B roles.

## Outputs and fallback

Each new output directory receives `action_report.json`; B also saves coarse
arrays and its complete world inverse. `perception/` contains Guidance JSON/NPZ,
SAM3 candidate masks/scores/prompts, camera diagnostic PNGs, fused object
geometry JSON/NPZ and `perception_report.json`.

The report distinguishes `completed`, `baseline_fallback` and `failed`. Missing
or low-quality Target geometry and perception exceptions return the C baseline
under the default fallback, record the error, and give CLI exit code 2.
`--fallback fail` raises instead, while retaining the failure report. No synthetic
box, geometry or second action is fabricated. A present but unseen/uncertain
Reference and explicit NULL keep their separate states throughout.

Stage timings and the policy CUDA allocator peak are measured for the actual
run. The CLI prepares both models before per-frame timing and records separate
`bridge_model_load_ms` / `sam3_model_load_ms` in `model_config`; the first frame
still includes cold image inference and allocator setup. GPU values are null on CPU. If
SAM3 runs on a separate device, the policy-device peak does not include that
device's allocation. The report names an action's **change from C**, which is
not an action error or evidence of better behavior. It records no robot success
and leaves `real_offline_acceptance_passed=false`. Mask IoU, role/instance
accuracy, 3D error, GT action error and paired RLBench success still need actual
labelled/model experiments. TASK's existing experiment runner supplies its
offline perception A/B/C metrics; this prediction command does not invent them.

## Software validation

```powershell
$env:PYTHONPATH = 'C:\Users\yiwei.chen\BridgeVLA\tmp\slot-guidance-test-deps'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:TASK_RELEVANT_OBJECT_REPO = 'C:\Users\yiwei.chen\TASK_Relevant_object-main'
& .\.venv-oht\Scripts\python.exe -m pytest tests/test_coarse_slot_export.py tests/test_slot_guided_offline_geometry.py tests/test_slot_sam3_perception.py tests/test_slot_sam3_runner.py tests/test_slot_guided_action_draft.py tests/test_slot_guided_draft_config.py tests/test_eval_runtime_fingerprint.py -q
```

These are CPU contract, geometry, real TASK Mock interop, routing and regression
tests. Model-weight inference and manipulation performance remain a separate
measurement. Existing coarse/refine checkpoints, training defaults and the
original Qwen3-VL → SAM3 pipeline are preserved.

See the [implementation review and measured test results](slot_guided_sam3_offline_validation.md)
for this change's validation and environment limits.
