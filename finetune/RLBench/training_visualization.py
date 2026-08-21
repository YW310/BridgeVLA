"""Rank-zero visualization for processed RLBench training heatmaps."""

from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw


_COLUMNS = (
    ("input", "Input"),
    ("gt", "GT"),
    ("prior", "Oracle prior"),
    ("raw_pred", "Raw pred"),
    ("pred", "Fused pred"),
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
    heatmaps: Dict[str, np.ndarray] = {}
    for key, _ in _COLUMNS:
        if key == "input" or key not in stage_payload:
            continue
        value = stage_payload[key]
        if value.shape != (view_count, height, width):
            raise ValueError(
                f"{key} must have shape {(view_count, height, width)}, "
                f"got {tuple(value.shape)}"
            )
        heatmaps[key] = _normalize_heatmap(value)

    label_width = 64
    header_height = 42
    montage = Image.new(
        "RGB",
        (label_width + len(_COLUMNS) * width, header_height + view_count * height),
        color=(245, 245, 245),
    )
    draw = ImageDraw.Draw(montage)
    sample_text = f"step={step} task={task} goal={language_goal}"[:180]
    draw.text((4, 2), sample_text, fill=(0, 0, 0))
    for column, (_, title) in enumerate(_COLUMNS):
        draw.text(
            (label_width + column * width + 4, 22),
            title,
            fill=(0, 0, 0),
        )

    cell_size = (width, height)
    for view_index in range(view_count):
        y = header_height + view_index * height
        draw.text((4, y + 4), f"View {view_index}", fill=(0, 0, 0))
        for column, (key, _) in enumerate(_COLUMNS):
            x = label_width + column * width
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
