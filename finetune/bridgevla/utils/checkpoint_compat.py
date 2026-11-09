"""Lossless PaliGemma checkpoint key compatibility for policy inference.

Transformers moved the vision/projector modules under ``model`` and moved the
causal-LM wrapper's head out of ``language_model``. PyTorch's load_state_dict
does not apply the conversion that Hugging Face's from_pretrained performs.
Only the known names are translated, against the actual target state dict;
tensor values, policy heads and checkpoint architecture metadata are untouched.
"""
from __future__ import annotations


_PALIGEMMA_ROOTS = ('mvt1.model.', 'mvt2.model.', 'model.')
_LEGACY_TO_CURRENT = (
    ('vision_tower.', 'model.vision_tower.'),
    ('multi_modal_projector.', 'model.multi_modal_projector.'),
    ('language_model.model.', 'model.language_model.'),
    ('language_model.lm_head.', 'lm_head.'),
)


def _alternate_key(key):
    for root in _PALIGEMMA_ROOTS:
        if not key.startswith(root):
            continue
        suffix = key[len(root):]
        for legacy, current in _LEGACY_TO_CURRENT:
            if suffix.startswith(legacy):
                return root + current + suffix[len(legacy):]
            if suffix.startswith(current):
                return root + legacy + suffix[len(current):]
        break
    return None


def remap_paligemma_state_dict(model_state, target_state):
    """Return (state, source_to_target_names), without copying any tensors.

    Exact keys win. A rename requires an existing target key and identical
    tensor shape, and must not collide with another checkpoint entry. Unknown
    or genuinely missing parameters are left for the caller's strict loader.
    This adapts inference weights only, not optimizer state or model behavior.
    """
    renamed = {}
    destinations = set()
    for key, value in model_state.items():
        if key in target_state:
            continue
        alternate = _alternate_key(key)
        if alternate is None or alternate not in target_state:
            continue
        if alternate in model_state or alternate in destinations:
            raise RuntimeError(
                f'Ambiguous PaliGemma checkpoint keys: {key!r} maps to '
                f'{alternate!r}, which already has a checkpoint value.')
        source_shape = getattr(value, 'shape', None)
        target_shape = getattr(target_state[alternate], 'shape', None)
        if source_shape is None or target_shape is None or source_shape != target_shape:
            raise RuntimeError(
                f'PaliGemma checkpoint shape mismatch: {key!r} -> {alternate!r}: '
                f'checkpoint={source_shape}, model={target_shape}. '
                'Use the matching model architecture; key renaming cannot fix this.')
        renamed[key] = alternate
        destinations.add(alternate)
    if not renamed:
        return model_state, renamed
    return {renamed.get(key, key): value for key, value in model_state.items()}, renamed
