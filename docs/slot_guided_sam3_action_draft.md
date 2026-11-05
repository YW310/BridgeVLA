# Slot-guided SAM3 action conditioning: PR4 software draft

This branch implements an inference-only software contract. Real offline A/B/C
experiments have not passed, so deployment/activation is not authorized by these
tests. No robot success, mask quality, action-error, GPU-memory or latency
improvement is claimed.

The default is `rvt.enable_slot_guided_sam3: False` in
`finetune/bridgevla/config.py`. No learned parameters or persistent buffers are
added. Existing MVT, Slot, refine and Anchor checkpoint keys are preserved.
Training settings and original Qwen/SAM pipelines are untouched.

## Data flow

```text
PR3 current-frame object geometry JSON + linked NPZ
  -> detached ObjectGeometry tensors in RLBench world meters
  -> reviewed REAL acceptance + exact frame/source/calibration checks
  -> actual agent scene world-to-cube transform
  -> actual coarse/refine frame and legacy correlation normalization
  -> existing Relation Anchor geometry/points/valid/NULL arguments
  -> frozen existing Action Head
```

The existing Slot role priors and role tokens remain the semantic path. This
draft supplies separate object geometry arguments to the existing
`OracleRelationAnchorFeatureAdapter.forward_with_anchor`; it does not overwrite
`role_prior`, replace Slot selection, add attention modules, or add a fusion
backbone. The draft requires an existing Slot+Relation Anchor architecture with
`role_conditioning=True` in every enabled stage;
unsupported architectures fail explicitly. Both coarse and refine stages use
the same current-frame external geometry transformed into their own frames.

Geometry tensors are:

| Value | Shape |
| --- | --- |
| `object_points` | `[B,2,N,3]`, role order Target/Reference |
| `object_point_valid` | `[B,2,N]`, bool |
| `object_role_valid` | `[B,2]`, geometry availability |
| `reference_null_probability` | `[B]`, continuous posterior |
| `object_geometry_confidence` | `[B,2]` |

Sampling is deterministic without replacement. Fewer than N observed points
produce masked padding, not repeated/fabricated observations. Centers and
standard deviations use only valid points. Geometry confidence below .25
disables that role's external geometry. The existing Anchor consumes the masked
descriptor, so its point-encoder path cannot confuse zero padding with object
surface support.

Semantic statuses remain `VISIBLE`, `NULL`, `NOT_VISIBLE`, and `LOW_CONFIDENCE`.
Only VISIBLE may carry valid geometry; VISIBLE with empty geometry is allowed
after a segmentation/fusion failure. NULL is distinct from a present but unseen
or uncertain Reference. The continuous NULL posterior is preserved independently
of the explicit status, including probabilities exactly zero or one. In
particular, a posterior is not treated as semantic ground truth.

A separate optional `reference_is_null` bool passed to the existing Anchor
hard-selects its existing semantic NULL token for explicit NULL. It suppresses
Reference semantic-token projection in that result without mutating original
Slot tensors or the raw posterior. Present but unseen/uncertain references keep
the existing unknown-token path. An Anchor lacking the existing
`role_conditioning=True` support is rejected in the enabled draft; its original
disabled behavior remains unchanged. This optional argument adds no checkpoint
parameters.

`load_pr3_geometry(path, count=128)` reads one `slot_object_geometry` schema-1
artifact with `coordinate_frame=rlbench_world`, `coordinate_units=m`, a
frame-local NPZ basename, SHA256 match, finite `[N,3]` arrays and matching point
counts. It does not load a whole episode or read any pickle. It imports no TASK
modules. Unavailable roles must have empty arrays.

## Optional act input and trusted current-frame context

The optional API is `RVTAgent.act(..., slot_guided_sam3=payload)`. The disabled
flag ignores the entire payload and adds no network kwargs, freezing, validation,
I/O or random-number operations. A malformed optional payload therefore cannot
affect a disabled baseline.

For a future approved experiment, the trusted snapshot importer supplies:

```python
from bridgevla.models.slot_guided_geometry import load_pr3_geometry

geometry = load_pr3_geometry("current_frame/objects.json", count=128)
observation["slot_guided_current_frames"] = [{
    "episode_id": "episode0",
    "frame_id": "17",
    "source_frame_index": 17,
    "observed_at": "2026-11-05T00:00:00+00:00",
    "calibration_sha256": "<actual current snapshot digest>",
    "max_source_frame_index_read": 17,
}]
payload = {
    "geometry": geometry,
    "acceptance_report": reviewed_report,
    "bridge_checkpoint_path": actual_loaded_checkpoint_path,
    "bridge_git_sha": actual_bridge_source_sha,
}
# This call remains unavailable until real acceptance is provided.
# action = agent.act(step, observation, slot_guided_sam3=payload)
```

The importer is responsible for binding this receipt to the actual current
observation and its calibration; the existing replay/whole-episode loader is not
a permitted importer. The action boundary compares episode ID, frame ID, source
episode index, timezone-aware observation timestamp and calibration digest
exactly against PR3 provenance, and requires the read receipt's maximum source
index to equal the current index. No expert action field is read by this adapter.
Do not use replay `gripper_pose` as current measured state: in RLBench training
replay it is a future supervision label.

The caller must provide the checkpoint file actually loaded in the network and
the actual Bridge source SHA. The file is hashed and matched against the reviewed
report; this draft does not replace existing checkpoint loading or independently
prove that an arbitrary caller-supplied path was loaded. This trusted importer/
loader binding is an explicit remaining integration responsibility.

## Fail-closed real experiment gate

The gate requires all of:

- `evidence_kind: REAL`, `verified: true`, `backend_kind: sam3`, `mock: false`,
  `accepted: true`, and explicit `reviewed_by` / `reviewed_at`.
- Exact report `sources` matching `bridge_git_sha`, the SHA256 of the supplied
  Bridge checkpoint, `guidance_git_sha`, `guidance_checkpoint_sha256`, and
  `sam3_checkpoint_sha256`; source/checkpoint hashes must be present and valid.
- Current calibration SHA256 included in the report's `calibration_sha256`
  list. Artifact provenance must be `guidance_source=predicted_coarse_slot`,
  `backend_kind=sam3`; mock and Oracle geometry cannot activate this action path.
- Non-null finite nonnegative measurements for `role_accuracy`,
  `prompt_instance_hit_rate`, `mask_iou`, `multiview_consistency`, `center_error_m`,
  `action_translation_error_m`, `paired_task_success_rate`, `latency_ms`, and
  `peak_gpu_memory_bytes`. The report's acceptance is a trusted human review
  decision; the code does not invent performance thresholds or authenticate a
  reviewer identity.

A software-test/mock report cannot authorize activation. Unit tests replace the
authorization verification method only within a pytest monkeypatch fixture so
the actual route can be exercised with deterministic CPU data. There is no
production `allow_mock` switch or accepted software-test report type.

## Freeze, coordinate and compatibility behavior

Enabled agents reject training/build/update and `train()` requests, remain in
eval mode, and freeze all network parameters. The optional path disallows
`context_with_grad`, image augmentation and privileged object alignment/anchor
diagnostic combinations. Network-level external calls require no-grad/eval.
External points, confidence and NULL tensors are detached; JSON/NPZ forms a
separate boundary from SAM3, whose weights are not loaded here.

World-to-cube calls the actual `place_pc_in_cube` with the same bounded scene,
centering option and bounds as the policy, rather than fitting a cube to the
object. Refinement uses the actual predicted coarse center and `st_sca`.
Legacy `norm_corr` divides geometry by the same scene/crop maximum absolute
coordinate when clean role XYZ is disabled. Clean-role configurations retain
cube/crop coordinates. Invalid padding is masked after every transform.

CPU checks cover the actual source-extracted MVT forward with deterministic
stage stubs, both stages' transforms, actual existing Anchor routing with
nonzero residual weights, exact disabled outputs, unchanged prior/tokens,
unchanged state_dict keys, strict checkpoint round-trip, masked sampling,
NULL/unseen/uncertain distinctions, posterior independence, stale/future frame
rejection, real-gate failures and parameter freezing.

The PR3-writer interoperability test is opt-in with
`SLOT_GUIDED_PR3_ARTIFACT` pointing to the synthetic two-instance, source-frame-7
`B_point+box/object_geometry.json` output. It loads both point clouds and verifies
that action authorization rejects the mock producer. The reviewed expanded CPU
run passed 136 tests and 3 subtests, including configuration/training helpers and
that actual writer output; a separate existing OHT action-call regression passed.
The unchanged reduced-mode suite has two failures also reproduced in the
original checkout: its expected Gemma prefix 9 differs from YAML value 6, and
its Oracle replay guard string assertion is stale. Those unrelated baseline
settings were not changed.

Run from this checkout:

```powershell
& C:\Users\yiwei.chen\BridgeVLA\.venv-oht\Scripts\python.exe -m pytest tests/test_slot_guided_action_draft.py tests/test_cross_scale_roles.py tests/test_cross_scale_render.py tests/test_refine_role_masks.py tests/test_object_conditioning.py tests/test_oracle_prior.py -q
```

No actual Bridge checkpoint, PaliGemma inference, GPU renderer, SAM3 prediction,
robot action or RLBench rollout was executed. Those checks, trusted current-frame
importer/checkpoint binding, calibrated real snapshots, real reviewed acceptance
and paired task evaluation remain required before runtime activation.
