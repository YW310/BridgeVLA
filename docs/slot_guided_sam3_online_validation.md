# Online Slot-guided SAM3 implementation review

This change connects the existing TASK SAM3 perception implementation to actual
current-observation RLBench policy calls. The offline path remains an experiment
tool; it is not used to replay or feed per-frame files to the online policy.

## Interfaces and changed areas

- `rlbench_slot_guided_sam3_online.yaml` and `slot_guided_sam3_online.py` select
  the enabled, frozen `online_experiment` path. Runtime overrides use the
  existing evaluator's `--slot-sam3-*` arguments.
- `slot_sam3_runtime.py` validates/resolves config, preloads persistent SAM3 and
  wraps the agent before the evaluator launches its environment. The same
  current-observation provider is supplied to both environment and policy.
- `slot_sam3_observation.py` and the opt-in custom-environment hook bind raw
  current RGB/XYZ/K/T to policy RGB/XYZ/K/T, state and instruction. Simulation
  microsteps include action execution, independently of demo/policy indices.
  Paired-negative PyRep focal signs use an explicit proper camera rotation;
  original pixels/world points and recording callbacks are preserved.
- `slot_sam3_perception.py` reuses TASK producers, prompt adapters, SAM3 candidate
  inference and multiview RGB-D fusion directly in memory. The frozen backend
  persists, while frames/image state/geometry are fresh. Model loading is
  separated from steady per-frame perception timing.
- `slot_sam3_policy.py` keeps the current C action, then applies fresh SAM3
  geometry through the existing Anchor. Failures return that current C action
  or raise according to config. Model-file identity is pinned at actual loading.
- `OnlineGeometry` requires predicted roles, real SAM3, current RGB-D/calibration
  receipts and matching loaded policy hashes. The existing reviewed gate is
  retained; no acceptance report or performance claim is manufactured.
- Eval fingerprint coverage, step/failure diagnostics, README, config index and
  the [online run guide](slot_guided_sam3_online.md) are updated. TASK source and
  its original Qwen3-VL/SAM3 defaults are not modified.

## Validation

Final targeted run: **209 passed**, with the actual TASK checkout and existing
PR3 writer artifact configured; no interoperability case was skipped. The
separate handbook suite passes **11 tests**. Online `--help` and `--check-config`
(including a real `box-only` CLI override), handbook generation/check and
`git diff --check` are included in the submission checks.

The combined CPU suite contains the five online test files plus the seven
existing coarse/offline/action/config/fingerprint test files. It exercises
actual TASK Mock/writer interfaces and source-extracted Bridge policy/MVT/Anchor
routes; it is not a model-weight or simulator performance measurement.

Coverage includes:

- persistent model initialization and one encoder call per camera shared by both
  roles; new frames re-encode, while reset keeps weights and clears frame history;
- same-class wrong candidates, empty/low-confidence results, NULL/unseen/uncertain
  states, current-C fallback and fail-mode diagnostic persistence;
- real PyRep signed-intrinsics conversion, dynamic wrist poses, measured depth
  reconstruction, 400 general float32-rounded camera rotations and invalid,
  behind-camera or mismatched calibration cases;
- current RGB/XYZ/K/T/state/instruction binding, no simulator advancement,
  live microstep indexing, stale/duplicate/future source rejection;
- actual CLI forwarding/config overrides, model preloading, direct evaluator
  provider identity, disabled baseline route and training/Oracle/resume guards;
- actual checkpoint pin changes, strict state-dict compatibility, no-grad
  transforms, existing coarse/refine geometry routes and explicit Mock rejection.

Use the commands below from the BridgeVLA root. The optional existing writer
artifact test runs when `SLOT_GUIDED_PR3_ARTIFACT` points to its synthetic
`results/synthetic_two_instances/7/B_point+box/object_geometry.json` output;
without that artifact its existing interoperability case is explicitly skipped.

```powershell
$env:PYTHONPATH = 'C:\Users\yiwei.chen\BridgeVLA\tmp\slot-guidance-test-deps'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:TASK_RELEVANT_OBJECT_REPO = 'C:\Users\yiwei.chen\TASK_Relevant_object-main'
& .\.venv-oht\Scripts\python.exe -m pytest tests/test_online_slot_sam3_geometry.py tests/test_online_slot_sam3_runtime.py tests/test_online_slot_sam3_perception.py tests/test_online_slot_sam3_policy.py tests/test_slot_sam3_observation.py tests/test_coarse_slot_export.py tests/test_slot_guided_offline_geometry.py tests/test_slot_sam3_perception.py tests/test_slot_sam3_runner.py tests/test_slot_guided_action_draft.py tests/test_slot_guided_draft_config.py tests/test_eval_runtime_fingerprint.py -q
& .\.venv-oht\Scripts\python.exe -m pytest tests/test_docs_handbook.py -q
& .\.venv-oht\Scripts\python.exe finetune/RLBench/slot_guided_sam3_online.py --check-config
```

The corresponding CMD form uses `set NAME=value` before the same Python
commands. Checks are run sequentially, with documentation in a separate process.
This avoids loading PowerShell/.NET and Torch together when the workstation's
small paging file prevents a process from starting. An earlier OHT HTTP suite
was interrupted by host memory/page-file exhaustion and is not a passed check.

## Limits requiring real experiments

The current local environment lacks an installed SAM3/CUDA model runtime. No
real Bridge/SAM3 checkpoint inference, simulator closed-loop episode or physical
robot action has been executed by this validation. The command and controller
wiring are implemented; robot/perception metrics, paired task success, actual
latency and full-frame GPU peak remain unmeasured.

Checkpoint SHA256 is currently recomputed by TASK fusion and Bridge action
authorization on each step. That cost is included in online timing. SAM3's
fingerprint is already cached. A future optimization must retain loaded-weight
identity checks. Online resume and generic copied/multiprocess YARR worker
integration are not enabled by this first direct-evaluator path.
