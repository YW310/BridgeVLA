"""Lossless PaliGemma checkpoint key compatibility for policy inference.

Transformers moved the vision/projector modules under ``model`` and moved the
causal-LM wrapper's head out of ``language_model``. PyTorch's load_state_dict
does not apply the conversion that Hugging Face's from_pretrained performs.
Only the known names are translated, against the actual target state dict;
tensor values, policy heads and checkpoint architecture metadata are untouched.
"""
from __future__ import annotations


_PALIGEMMA_ROOTS = ('mvt1.model.', 'mvt2.model.', 'model.')
_NAMESPACE_VARIANTS = (
    ('vision_tower.', 'model.vision_tower.'),
    ('multi_modal_projector.', 'model.multi_modal_projector.'),
    ('language_model.lm_head.', 'lm_head.'),
    ('language_model.model.', 'model.language_model.', 'language_model.'),
)


def _alternate_keys(key):
    for root in _PALIGEMMA_ROOTS:
        if not key.startswith(root):
            continue
        suffix = key[len(root):]
        for variants in _NAMESPACE_VARIANTS:
            for prefix in variants:
                if suffix.startswith(prefix):
                    return [root + other + suffix[len(prefix):] for other in variants if other != prefix]
        break
    return []


def remap_paligemma_state_dict(model_state, target_state):
    """Return (state, source_to_target_names), without copying any tensors.

    Exact keys win. A rename requires an existing target key and identical
    tensor shape, and must not collide with another checkpoint entry. Unknown
    or genuinely missing parameters are left for the caller's strict loader.
    This adapts model weights only, not optimizer state or model behavior.
    """
    renamed = {}
    destinations = set()
    for key, value in model_state.items():
        if key in target_state:
            continue
        alternatives = [name for name in _alternate_keys(key) if name in target_state]
        if not alternatives:
            continue
        if len(alternatives) > 1:
            raise RuntimeError(f'Ambiguous PaliGemma target modules for {key!r}: {alternatives}')
        alternate = alternatives[0]
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
