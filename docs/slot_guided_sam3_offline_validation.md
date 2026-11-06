# Offline implementation review

This follow-up completes the software connection missing from the PR4 draft.
It does not record a completed model or robot experiment.

## Files and interfaces

- `finetune/RLBench/slot_guided_sam3_offline.py` and the enabled
  `rlbench_slot_guided_sam3_offline.yaml` provide the single-frame command.
- `bridgevla/offline/slot_sam3_checkpoint.py` checks saved architecture/routing,
  strictly loads the existing checkpoint and freezes it. Loading hashes are
  retained and checked against the file used for geometry provenance.
- `bridgevla/models/coarse_slot_export.py` exports coarse fields, independently
  rendered point coverage and the actual world inverse. The callback is opt-in;
  the default `RVTAgent.act` path adds no coverage render or encoder call.
- `bridgevla/offline/slot_sam3_perception.py` reuses TASK's producers, prompt/SAM3
  runner, cache and geometry fusion. Original RGB pixels and dynamic camera
  calibration remain owned by the current snapshot. No TASK source is modified.
- `bridgevla/offline/slot_sam3_runner.py` predicts C, runs A or B perception, then
  predicts the conditioned action. RGB-D digest, frame identity, calibration,
  model hashes, NULL/visibility and failure reasons are retained.
- `OfflineGeometry` is an explicit prediction-only experiment scope. The
  reviewed activation route continues to require the real reviewed report.
  Training, ordinary control rollouts, Mock action activation and future-frame
  inputs remain rejected. No learned parameters or checkpoint keys were added.

## Verification on the local CPU environment

The seven-file command in the [run guide](slot_guided_sam3_offline.md#software-validation)
passed **124 tests**, with the actual TASK checkout and existing PR3 writer
artifact configured. No interoperability test was skipped. It covers the actual
predictor point-extraction contract, cube inverse round-trip, default callback
control, A/B geometry scopes, exact Anchor disabled outputs, strict state dict
round-trip, source/frame/calibration mismatch, same-instance candidate support,
dynamic wrist geometry, original tensor preprocessing, mock rejection, low
confidence/empty-mask fallback, the actual CLI preflight, and rollout guards.

The separate handbook suite passed **11 tests**. `--help`, `--check-config`,
handbook generation/check and `git diff --check` passed. Documentation is run in
a separate process from Torch to avoid the workstation's page-file pressure.

An additional existing OHT HTTP-policy integration run encountered Windows
memory/page-file exhaustion and was interrupted. It is not included in the
passed count. Complete PaliGemma/OHT inference regression should be repeated in
the prepared model environment; software route checks do not replace it.

## Running and remaining measurements

See the [single-frame commands and input schemas](slot_guided_sam3_offline.md).
The new profile actually enables `offline_experiment`; the legacy draft profile
and every global default remain disabled. Both BridgeVLA and SAM3 model loading
are recorded separately from per-frame timing.

The prepared local Python currently reports `sam3_installed=False` and
`cuda_available=False`. No matching real checkpoint/current calibrated snapshot
was supplied to this run. Real GPU rendering, SAM3/PaliGemma inference, mask IoU,
role accuracy, GT center/action errors, paired RLBench success, GPU memory and
latency measurements remain unexecuted. Synthetic/Mock tests never set
`real_offline_acceptance_passed=true` and no performance improvement is claimed.
