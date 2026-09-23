"""Compact inference visualizations for internally predicted object slots."""

import json
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw


_MAX_MONTAGE_WIDTH = 1600
_MAX_MONTAGE_HEIGHT = 1000
_OUTER_PADDING = 8
_CELL_GAP = 4
_LABEL_WIDTH = 92
_HEADER_HEIGHT = 52
_BORDER_COLOR = (70, 70, 70)
_ORIGINAL_IMAGE_WEIGHT = 0.30


def _draw_centered_text(draw, bounds, text, fill=(0, 0, 0)):
    left, top, right, bottom = bounds
    text = str(text)
    if hasattr(draw, 'textbbox'):
        box = draw.textbbox((0, 0), text)
        text_width, text_height = box[2] - box[0], box[3] - box[1]
    else:
        text_width, text_height = draw.textsize(text)
    draw.text(
        (left + max(0, (right - left - text_width) // 2),
         top + max(0, (bottom - top - text_height) // 2)),
        text, fill=fill,
    )


def _fit_cell_size(width: int, height: int, columns: int, rows: int):
    available_width = (
        _MAX_MONTAGE_WIDTH - 2 * _OUTER_PADDING - _LABEL_WIDTH
        - max(columns - 1, 0) * _CELL_GAP
    )
    available_height = (
        _MAX_MONTAGE_HEIGHT - 2 * _OUTER_PADDING - _HEADER_HEIGHT
        - max(rows - 1, 0) * _CELL_GAP
    )
    scale = min(
        1.0,
        available_width / max(columns * width, 1),
        available_height / max(rows * height, 1),
    )
    return max(1, int(width * scale)), max(1, int(height * scale))


def _input_images(value: torch.Tensor) -> np.ndarray:
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError('inference visualization input must have shape [V,3,H,W]')
    array = value.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
    if float(array.min()) < 0:
        array = (array + 1.0) / 2.0
    return np.clip(array, 0.0, 1.0)


def _normalize_heatmaps(value: torch.Tensor) -> np.ndarray:
    array = np.nan_to_num(
        value.detach().float().cpu().numpy(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    minimum = array.min(axis=(-2, -1), keepdims=True)
    maximum = array.max(axis=(-2, -1), keepdims=True)
    scale = maximum - minimum
    return np.where(
        scale > 0,
        (array - minimum) / np.maximum(scale, 1e-12),
        0,
    )


def _heatmap_rgb(value: np.ndarray) -> np.ndarray:
    """Dependency-free blue-to-yellow heatmap."""
    value = np.clip(value, 0.0, 1.0)
    red = np.clip(1.5 * value, 0.0, 1.0)
    green = np.clip(1.5 * value - 0.35, 0.0, 1.0)
    blue = np.clip(1.0 - 1.5 * value, 0.0, 1.0)
    return np.stack((red, green, blue), axis=-1)


def blend_heatmap_with_image(
    input_image: np.ndarray,
    normalized_heatmap: np.ndarray,
    image_weight: float = _ORIGINAL_IMAGE_WEIGHT,
) -> np.ndarray:
    """Blend a colored heatmap with its RGB view using 30% RGB by default."""
    if input_image.ndim != 3 or input_image.shape[-1] != 3:
        raise ValueError('input_image must have shape [H,W,3]')
    if not 0.0 <= image_weight <= 1.0:
        raise ValueError('image_weight must be in [0, 1]')
    if normalized_heatmap.ndim != 2:
        raise ValueError('normalized_heatmap must have shape [H,W]')
    if normalized_heatmap.shape != input_image.shape[:2]:
        resampling = getattr(Image, 'Resampling', Image)
        normalized_heatmap = np.asarray(
            Image.fromarray(
                np.asarray(normalized_heatmap, dtype=np.float32), mode='F',
            ).resize(
                (input_image.shape[1], input_image.shape[0]),
                resampling.BILINEAR,
            ),
            dtype=np.float32,
        )
    image = np.clip(input_image, 0.0, 1.0)
    heatmap = _heatmap_rgb(normalized_heatmap)
    return image_weight * image + (1.0 - image_weight) * heatmap


def _as_image(value: np.ndarray, size) -> Image.Image:
    image = Image.fromarray((np.clip(value, 0, 1) * 255).astype(np.uint8))
    if image.size != size:
        resampling = getattr(Image, 'Resampling', Image)
        image = image.resize(size, resampling.BILINEAR)
    return image


def build_internal_slot_stage_payload(
    stage_output: Mapping[str, torch.Tensor],
    rendered_input: torch.Tensor,
    action_heatmap: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Collect no-GT slot diagnostics for one MVT stage."""
    required = ('object_slot_masks', 'object_slot_prior')
    missing = [key for key in required if key not in stage_output]
    if missing:
        raise KeyError('Internal-slot visualization is missing: ' + ', '.join(missing))
    payload = {
        'input': rendered_input.detach().float().cpu(),
        'action_pred': action_heatmap.detach().float().cpu(),
    }
    slot_masks = stage_output['object_slot_masks'][0]
    for slot_index in range(slot_masks.shape[1]):
        payload[f'slot_{slot_index}'] = slot_masks[:, slot_index].detach().float().cpu()
    role_prior = stage_output['object_slot_prior'][0]
    if role_prior.shape[1] != 2:
        raise ValueError('object_slot_prior must contain Target and Reference maps')
    payload['target_pred'] = role_prior[:, 0].detach().float().cpu()
    payload['reference_pred'] = role_prior[:, 1].detach().float().cpu()
    if 'oracle_relation_anchor' in stage_output:
        payload['relation_anchor'] = (
            stage_output['oracle_relation_anchor'][0].detach().float().cpu()
        )
    return payload


def internal_slot_stage_diagnostics(
    stage_output: Mapping[str, torch.Tensor],
) -> Dict[str, object]:
    """Return JSON-safe confidence, role, and NULL diagnostics."""
    confidence = stage_output['object_slot_confidence'][0].detach().float().cpu()
    valid = stage_output['object_slot_valid'][0].detach().bool().cpu()
    objectness = torch.sigmoid(
        stage_output['object_slot_objectness_logits'][0].detach().float().cpu()
    )
    role_probability = torch.softmax(
        stage_output['object_slot_role_logits'][0].detach().float().cpu(), dim=-1,
    )
    null_probability = stage_output[
        'object_slot_reference_null_probability'
    ][0].detach().float().cpu()
    return {
        'target_confidence': float(confidence[0]),
        'reference_confidence': float(confidence[1]),
        'target_valid': bool(valid[0]),
        'reference_valid': bool(valid[1]),
        'reference_null_probability': float(null_probability),
        'slot_objectness': [float(value) for value in objectness],
        'slot_role_probability': [
            {
                'target': float(probability[0]),
                'reference': float(probability[1]),
            }
            for probability in role_probability
        ],
    }


def _column_order(payloads: Mapping[str, Mapping[str, torch.Tensor]]):
    maximum_slots = max(
        sum(key.startswith('slot_') for key in payload)
        for payload in payloads.values()
    )
    columns = [('input', 'Input')]
    columns.extend(
        (f'slot_{index}', f'Slot {index}') for index in range(maximum_slots)
    )
    columns.extend((
        ('target_pred', 'Target pred'),
        ('reference_pred', 'Reference pred'),
        ('relation_anchor', 'Relation anchor'),
        ('action_pred', 'Action pred'),
    ))
    return tuple(columns)


def internal_slot_montage(
    payloads: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    step: int,
    diagnostics: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> Image.Image:
    """Combine every stage/view for one policy step into a bounded image."""
    if not payloads:
        raise ValueError('At least one stage is required for a montage')
    columns = _column_order(payloads)
    rows = []
    source_width = source_height = None
    for stage_name, payload in payloads.items():
        inputs = _input_images(payload['input'])
        view_count, height, width, _ = inputs.shape
        if source_width is None:
            source_width, source_height = width, height
        heatmaps = {}
        for key, _ in columns:
            if key == 'input' or key not in payload:
                continue
            value = payload[key]
            if value.ndim != 3 or value.shape[0] != view_count:
                raise ValueError(f'{stage_name}/{key} must have shape [V,H,W]')
            heatmaps[key] = _normalize_heatmaps(value)
        for view_index in range(view_count):
            rows.append((stage_name, view_index, inputs[view_index], heatmaps))

    cell_width, cell_height = _fit_cell_size(
        source_width, source_height, len(columns), len(rows),
    )
    montage = Image.new(
        'RGB',
        (
            2 * _OUTER_PADDING + _LABEL_WIDTH + len(columns) * cell_width
            + max(len(columns) - 1, 0) * _CELL_GAP,
            2 * _OUTER_PADDING + _HEADER_HEIGHT + len(rows) * cell_height
            + max(len(rows) - 1, 0) * _CELL_GAP,
        ),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(montage)
    metric_summary = ''
    if diagnostics:
        parts = []
        for stage_name, values in diagnostics.items():
            parts.append(
                '{} T={:.2f} R={:.2f} NULL={:.2f}'.format(
                    stage_name,
                    float(values['target_confidence']),
                    float(values['reference_confidence']),
                    float(values['reference_null_probability']),
                )
            )
        metric_summary = '; ' + '; '.join(parts)
    draw.text(
        (_OUTER_PADDING + 4, _OUTER_PADDING + 2),
        f'policy step={step}; no GT; normalized per view{metric_summary}',
        fill=(0, 0, 0),
    )
    for column_index, (_, title) in enumerate(columns):
        x = _OUTER_PADDING + _LABEL_WIDTH + column_index * (
            cell_width + _CELL_GAP
        )
        bounds = (
            x, _OUTER_PADDING + 20,
            x + cell_width - 1, _OUTER_PADDING + _HEADER_HEIGHT - 1,
        )
        draw.rectangle(bounds, fill=(232, 232, 232),
                       outline=_BORDER_COLOR, width=2)
        _draw_centered_text(draw, bounds, title)

    cell_size = (cell_width, cell_height)
    for row_index, (stage_name, view_index, input_image, heatmaps) in enumerate(rows):
        y = _OUTER_PADDING + _HEADER_HEIGHT + row_index * (
            cell_height + _CELL_GAP
        )
        label_bounds = (
            2, y, _OUTER_PADDING + _LABEL_WIDTH - 1, y + cell_height - 1,
        )
        draw.rectangle(label_bounds, fill=(232, 232, 232),
                       outline=_BORDER_COLOR, width=2)
        _draw_centered_text(draw, label_bounds, f'{stage_name} V{view_index}')
        for column_index, (key, _) in enumerate(columns):
            x = _OUTER_PADDING + _LABEL_WIDTH + column_index * (
                cell_width + _CELL_GAP
            )
            if key == 'input':
                cell = _as_image(input_image, cell_size)
            elif key in heatmaps:
                cell = _as_image(
                    blend_heatmap_with_image(
                        input_image, heatmaps[key][view_index],
                    ),
                    cell_size,
                )
            else:
                cell = Image.new('RGB', cell_size, color=(224, 224, 224))
                ImageDraw.Draw(cell).text(
                    (6, 6), 'unavailable', fill=(80, 80, 80),
                )
            montage.paste(cell, (x, y))
            draw.rectangle(
                (x, y, x + cell_width - 1, y + cell_height - 1),
                outline=_BORDER_COLOR, width=2,
            )
    draw.rectangle(
        (0, 0, montage.width - 1, montage.height - 1),
        outline=_BORDER_COLOR, width=2,
    )
    return montage


def save_internal_slot_step_visualization(
    payloads: Mapping[str, Mapping[str, torch.Tensor]],
    diagnostics: Mapping[str, Mapping[str, object]],
    *,
    step: int,
    output_dir,
) -> Dict[str, Path]:
    """Save the combined PNG and machine-readable values for one step."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f'step_{step:04d}'
    montage_path = output_dir / f'{stem}.png'
    diagnostics_path = output_dir / f'{stem}.json'
    internal_slot_montage(
        payloads, step=step, diagnostics=diagnostics,
    ).save(montage_path)
    diagnostics_path.write_text(
        json.dumps(dict(diagnostics), indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    return {'montage': montage_path, 'diagnostics': diagnostics_path}
