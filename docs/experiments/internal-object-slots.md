# Internal object slots

This experiment predicts task-relevant objects inside BridgeVLA instead of
injecting Oracle or externally predicted point sets.

## Data flow

```text
multi-view BridgeVLA features + relation state
    -> unordered object slots
    -> slot masks + objectness
    -> Target / Reference role mixing (+ NULL Reference)
    -> role heatmaps
    -> top-scoring XYZ samples from the same rendered views
    -> existing relation/anchor feature adapter
    -> action heads
```

`oracle_*` points are read during training only to rasterize Target/Reference
supervision heatmaps. `MVT.forward()` explicitly replaces their policy-side
prior, validity, and geometry with `None` before calling `MVTSingle`; only slot
predictions reach the adapter. Evaluation asks for no Oracle object fields.

This is currently a role-centric, single-frame experiment, not a complete
scene-entity model: it predicts T/R heatmaps and then samples role XYZ points,
but does not retain a task-relevant set of object/part/region geometries or
temporal tracks. The shared XYZ geometry contract and the current OBB/fallback
box `site` approximation are defined in
[Semantic-GT: interaction entity geometry](../guides/semantic-gt.md#semantic-gt-entity-geometry).
Future entity slots should preserve that set until phase-conditioned role
binding instead of discarding all non-selected entities early.

## Configuration

Use `configs/rlbench_o2_internal_slots.yaml`. The important controls are:

- `object_slots.num_slots`: number of unordered candidates; default `6`.
- `object_slots.slot_dim`: decoder width; default `128`.
- `object_slots.point_samples`: XYZ samples per predicted role; default `128`.
- `rvt.object_prediction_confidence_threshold`: below this, the role prior is
  marked invalid and the adapter safely falls back to the base feature path.
- `rvt.object_slot_*_loss_weight`: role-mask, presence/NULL, and anti-collapse
  auxiliary objectives.

The predictor is stateless in this version. This keeps the first ablation
identifiable; object-centric temporal memory should be added only after the
single-frame slot quality is measured.

## Training

Initialize from the current BridgeVLA or relation-adapter checkpoint:

```bash
bash train.sh \
  --exp_cfg_path configs/rlbench_o2_internal_slots.yaml \
  --train_replay_storage_dir /path/to/semantic_gt_buffer \
  --init_checkpoint /path/to/model_last.pth \
  --train_object_adapter_only
```

The last flag freezes the original BridgeVLA modules and trains both
`object_slot_predictor*` and `oracle_prior_feature_adapter*`. For joint
fine-tuning, resume the resulting checkpoint without that flag.

Monitor `object_slot_mask_loss`, `object_slot_presence_loss`,
`object_slot_diversity_loss`, predicted role confidence/validity, translation
argmax, and closed-loop success. A falling auxiliary loss alone does not prove
that the predicted object prior improves control.
