"""Rank-zero visualization for processed RLBench training heatmaps."""

from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw


_COLUMNS = (
    ('slot_target_gt', 'Slot Target GT'),
    ('slot_target_pred', 'Slot Target pred'),
    ('slot_reference_gt', 'Slot Reference GT'),
    ('slot_reference_pred', 'Slot Reference pred'),
    ('target_prior', 'Target prior'),
    ('reference_prior', 'Reference prior'),
    ('relation_anchor', 'Relation anchor'),
    ("input", "Input"),
    ("gt", "GT"),
    ("prior", "Oracle prior"),
    ("pred", "Adapted pred"),
)

_MAX_MONTAGE_WIDTH = 1600
_MAX_MONTAGE_HEIGHT = 1000
_OUTER_PADDING = 8
_CELL_GAP = 4
_LABEL_WIDTH = 72
_HEADER_HEIGHT = 52
_BORDER_COLOR = (70, 70, 70)


def _fit_cell_size(width: int, height: int, columns: int, rows: int):
    '''Fit the complete montage on a typical TensorBoard/browser page.'''
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


def visualization_due(enabled: bool, interval: int, step: int) -> bool:
    if not enabled:
        return False
    if interval <= 0:
        raise ValueError("train_visualization.interval must be > 0")
    return step % interval == 0


def _normalize_heatmap(value: torch.Tensor) -> np.ndarray:
    array = np.nan_to_num(
        value.detach().float().cpu().numpy(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    minimum = array.min(axis=(-2, -1), keepdims=True)
    maximum = array.max(axis=(-2, -1), keepdims=True)
    scale = maximum - minimum
    return np.where(scale > 0, (array - minimum) / np.maximum(scale, 1e-12), 0)


def _input_images(value: torch.Tensor) -> np.ndarray:
    if value.ndim != 4 or value.shape[1] != 3:
        raise ValueError("training visualization input must have shape [V,3,H,W]")
    array = value.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
    if float(array.min()) < 0:
        array = (array + 1.0) / 2.0
    return np.clip(array, 0.0, 1.0)


def _heatmap_rgb(value: np.ndarray) -> np.ndarray:
    """Small dependency-free blue-to-yellow heatmap."""
    value = np.clip(value, 0.0, 1.0)
    red = np.clip(1.5 * value, 0.0, 1.0)
    green = np.clip(1.5 * value - 0.35, 0.0, 1.0)
    blue = np.clip(1.0 - 1.5 * value, 0.0, 1.0)
    return np.stack((red, green, blue), axis=-1)


def _as_uint8_image(value: np.ndarray, size) -> Image.Image:
    image = Image.fromarray((np.clip(value, 0, 1) * 255).astype(np.uint8))
    if image.size != size:
        resampling = getattr(Image, "Resampling", Image)
        image = image.resize(size, resampling.BILINEAR)
    return image


def _stage_montage(
    stage_payload: Mapping[str, torch.Tensor],
    *,
    step: int,
    task: str,
    language_goal: str,
) -> Image.Image:
    if "input" not in stage_payload or "gt" not in stage_payload:
        raise KeyError("training visualization requires input and gt")
    inputs = _input_images(stage_payload["input"])
    view_count, height, width, _ = inputs.shape
    title_by_key = dict(_COLUMNS)
    column_order = (
        'input', 'gt', 'slot_target_gt', 'slot_target_pred',
        'slot_reference_gt', 'slot_reference_pred', 'target_prior',
        'reference_prior', 'relation_anchor', 'prior', 'pred',
    )
    columns = tuple(
        (key, title_by_key[key])
        for key in column_order
        if key == 'input' or key in stage_payload
    )
    heatmaps: Dict[str, np.ndarray] = {}
    for key, _ in columns:
        if key == "input" or key not in stage_payload:
            continue
        value = stage_payload[key]
        if value.shape != (view_count, height, width):
            raise ValueError(
                f"{key} must have shape {(view_count, height, width)}, "
                f"got {tuple(value.shape)}"
            )
        heatmaps[key] = _normalize_heatmap(value)

    label_width = _LABEL_WIDTH
    header_height = _HEADER_HEIGHT
    width, height = _fit_cell_size(
        width, height, len(columns), view_count,
    )
    montage = Image.new(
        "RGB",
        (
            2 * _OUTER_PADDING + label_width + len(columns) * width
            + max(len(columns) - 1, 0) * _CELL_GAP,
            2 * _OUTER_PADDING + header_height + view_count * height
            + max(view_count - 1, 0) * _CELL_GAP,
        ),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(montage)
    sample_text = f"step={step} task={task} goal={language_goal}"[:180]
    draw.text((_OUTER_PADDING + 4, _OUTER_PADDING + 2), sample_text,
              fill=(0, 0, 0))
    for column, (_, title) in enumerate(columns):
        x = (
            _OUTER_PADDING + label_width
            + column * (width + _CELL_GAP)
        )
        bounds = (
            x, _OUTER_PADDING + 20,
            x + width - 1, _OUTER_PADDING + header_height - 1,
        )
        draw.rectangle(bounds, fill=(232, 232, 232),
                       outline=_BORDER_COLOR, width=2)
        _draw_centered_text(draw, bounds, title)

    cell_size = (width, height)
    for view_index in range(view_count):
        y = (
            _OUTER_PADDING + header_height
            + view_index * (height + _CELL_GAP)
        )
        label_bounds = (
            2, y,
            _OUTER_PADDING + label_width - 1, y + height - 1,
        )
        draw.rectangle(label_bounds, fill=(232, 232, 232),
                       outline=_BORDER_COLOR, width=2)
        _draw_centered_text(draw, label_bounds, f'View {view_index}')
        for column, (key, _) in enumerate(columns):
            x = (
                _OUTER_PADDING + label_width
                + column * (width + _CELL_GAP)
            )
            if key == "input":
                cell = _as_uint8_image(inputs[view_index], cell_size)
            elif key in heatmaps:
                cell = _as_uint8_image(
                    _heatmap_rgb(heatmaps[key][view_index]),
                    cell_size,
                )
            else:
                cell = Image.new("RGB", cell_size, color=(224, 224, 224))
                ImageDraw.Draw(cell).text(
                    (6, 6), "unavailable", fill=(80, 80, 80)
                )
            montage.paste(cell, (x, y))
            draw.rectangle(
                (x, y, x + width - 1, y + height - 1),
                outline=_BORDER_COLOR, width=2,
            )
    draw.rectangle(
        (0, 0, montage.width - 1, montage.height - 1),
        outline=_BORDER_COLOR, width=2,
    )
    return montage


def record_training_visualization(
    payload: Mapping[str, Mapping[str, torch.Tensor]],
    *,
    step: int,
    output_dir: Path,
    task: str = "",
    language_goal: str = "",
    save_png: bool = True,
    writer: Optional[object] = None,
    write_tensorboard: bool = True,
) -> Dict[str, Path]:
    """Save stage montages and optionally add the same images to TensorBoard."""
    if not save_png and not write_tensorboard:
        return {}
    if write_tensorboard and writer is None:
        raise ValueError(
            "TensorBoard training visualization is enabled but no writer exists"
        )
    saved: Dict[str, Path] = {}
    step_dir = Path(output_dir) / f"step_{step:08d}"
    for stage_name, stage_payload in payload.items():
        montage = _stage_montage(
            stage_payload,
            step=step,
            task=str(task),
            language_goal=str(language_goal),
        )
        if save_png:
            step_dir.mkdir(parents=True, exist_ok=True)
            output_path = step_dir / f"{stage_name}.png"
            montage.save(output_path)
            saved[stage_name] = output_path
        if write_tensorboard:
            writer.add_image(
                f"train_visualization/{stage_name}",
                np.asarray(montage),
                step,
                dataformats="HWC",
            )
    if write_tensorboard:
        writer.add_text(
            "train_visualization/sample",
            f"task={task}; goal={language_goal}",
            step,
        )
    return saved
