"""Compact perception summaries and write-only current-frame visualizations."""
from __future__ import annotations


def _draw_prompt_markers(draw, prompt, color):
    if prompt.get('bbox_xyxy') is not None:
        draw.rectangle(prompt['bbox_xyxy'], outline=color, width=1)
    for x, y in prompt.get('positive_points_xy', []):
        draw.line([x - 2, y, x + 2, y], fill=color, width=1)
        draw.line([x, y - 2, x, y + 2], fill=color, width=1)
    for x, y in prompt.get('negative_points_xy', []):
        draw.line([x - 2, y - 2, x + 2, y + 2], fill='white')


def save_prompt_overlays(root, cameras, snapshot):
    """Use the same small positive cross in prompt-only camera diagnostics."""
    from pathlib import Path
    from PIL import Image, ImageDraw

    for name, camera in cameras.items():
        image = Image.fromarray(snapshot.cameras[name].rgb)
        draw = ImageDraw.Draw(image)
        for role, color in (('target', (255, 80, 40)), ('reference', (40, 150, 255))):
            _draw_prompt_markers(draw, camera['roles'][role]['prompt'], color)
        filename = Path(root) / f'{name}.diagnostic.png'
        with Image.open(filename) as original:
            canvas = original.convert('RGB')
        canvas.paste(image, (0, 0))
        canvas.save(filename)


def save_segmentation_overlays(root, inference, snapshot, *, alpha=.75):
    """Redraw selected masks from current RGB, with stronger tint and box sizes.

    This is write-only visualization. TASK's inference, masks and pixel prompts
    remain unchanged; its saved legend is retained beside the original image.
    """
    import json
    from pathlib import Path
    import numpy as np
    from PIL import Image, ImageDraw

    root = Path(root)
    sizes = {}
    def dimensions(value):
        return 'none' if value is None else f'{value[0]:.1f} x {value[1]:.1f}'

    for name, camera in inference['cameras'].items():
        rgb = snapshot.cameras[name].rgb
        height, width = rgb.shape[:2]
        tint = rgb.astype(np.float64).copy()
        sizes[name] = {}
        for role, color in (('target', (255, 80, 40)), ('reference', (40, 150, 255))):
            row = camera['roles'][role]
            box = row['prompt'].get('bbox_xyxy')
            selected = row.get('selected_index')
            item = dict(prompt_bbox_xyxy=box,
                        prompt_span_px=None if box is None else [box[2] - box[0], box[3] - box[1]],
                        selected_mask_bbox_xyxy=None, selected_mask_extent_px=None)
            if selected is not None:
                mask = np.asarray(row['candidates'][selected]['mask'])
                if mask.dtype != bool or mask.shape != (height, width):
                    raise ValueError('selected mask must match current original-camera RGB')
                tint[mask] = tint[mask] * (1 - alpha) + np.asarray(color) * alpha
                y, x = np.nonzero(mask)
                if len(x):
                    # Mask extents use exclusive upper pixel edges.
                    bounds = [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1]
                    item.update(selected_mask_bbox_xyxy=bounds,
                                selected_mask_extent_px=[bounds[2] - bounds[0], bounds[3] - bounds[1]])
            sizes[name][role] = item
        image = Image.fromarray(np.clip(tint, 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        for role, color in (('target', (255, 80, 40)), ('reference', (40, 150, 255))):
            _draw_prompt_markers(draw, camera['roles'][role]['prompt'], color)
        filename = root / f'{name}.diagnostic.png'
        with Image.open(filename) as original:
            canvas = Image.new('RGB', (max(original.width, width + 360), max(original.height, 172)), (25, 25, 25))
            canvas.paste(original, (0, 0))
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((width + 8, 110), f'Selected mask opacity: {alpha:.0%}', fill='white')
        for index, role in enumerate(('target', 'reference')):
            item = sizes[name][role]
            draw.text((width + 8, 128 + 18 * index),
                      f'{role}: box {dimensions(item["prompt_span_px"])}; '
                      f'mask {dimensions(item["selected_mask_extent_px"])} px', fill='white')
        canvas.save(filename)
    report = root / 'bbox_diagnostics.json'
    report.write_text(json.dumps(dict(mask_overlay_alpha=alpha, cameras=sizes), indent=2), encoding='utf-8')
    return str(report)


def camera_role_diagnostics(cameras):
    return {
        name: {
            role: dict(
                status=row.get('status'),
                prompt_valid=row.get('prompt', {}).get('prompt_valid'),
                prompt_failure_reason=row.get('prompt', {}).get('failure_reason'),
                failure_reason=row.get('failure_reason'),
                candidate_count=len(row.get('candidates', [])),
                accepted_candidate_count=sum(candidate.get('accepted') is True
                                             for candidate in row.get('candidates', [])),
                candidate_failure_reasons=sorted({
                    reason for candidate in row.get('candidates', [])
                    for reason in candidate.get('failure_reasons', [])}),
                selected_index=row.get('selected_index'))
            for role, row in camera.get('roles', {}).items()
        } for name, camera in cameras.items()
    }


def guidance_role_diagnostics(guidance):
    return {
        role: dict(status=guidance.metadata[role]['status'],
                   semantic_valid=guidance.metadata[role]['semantic_valid'],
                   confidence=float(guidance.metadata[role]['confidence']),
                   point_count=len(guidance.points(role)))
        for role in ('target', 'reference')
    }


class UnavailableTargetError(ValueError):
    """An action-only short circuit; no fabricated geometry is returned."""

    def __init__(self, diagnostics):
        super().__init__('SAM3/fusion produced no reliable current Target geometry: '
                         'no accepted Target candidates in any camera; fusion skipped')
        self.perception_diagnostics = diagnostics


def print_target_diagnostics(diagnostics, *, call):
    """Make the already recorded failure reasons visible in the runtime log."""
    guidance = diagnostics.get('guidance_roles', {}).get('target', {})
    target = diagnostics.get('roles', {}).get('target', {})
    print(f'[SlotSAM3] TARGET call={call}; guidance_status={guidance.get("status")}; '
          f'confidence={guidance.get("confidence", target.get("role_confidence"))}; '
          f'guidance_points={guidance.get("point_count")}; '
          f'geometry_confidence={target.get("geometry_confidence")}; '
          f'geometry_points={target.get("point_count")}; '
          f'failure={target.get("failure_reason") or diagnostics.get("fusion_skip_reason")}; '
          f'fusion_skipped={diagnostics.get("fusion_skipped", False)}', flush=True)
    for name, roles in diagnostics.get('camera_roles', {}).items():
        row = roles.get('target', {})
        print(f'[SlotSAM3] TARGET call={call} camera={name}; '
              f'prompt_valid={row.get("prompt_valid")}; '
              f'candidates={row.get("candidate_count")}; accepted={row.get("accepted_candidate_count")}; '
              f'failure={row.get("failure_reason") or row.get("prompt_failure_reason")}; '
              f'rejected_by={row.get("candidate_failure_reasons", [])}', flush=True)
