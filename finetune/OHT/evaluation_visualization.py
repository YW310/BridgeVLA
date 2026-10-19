"""PNG/NPZ evaluation diagnostics; held-out labels are consumed after inference."""
import html
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import bootstrap  # noqa: F401
from bridgevla.models.inference_visualization import blend_heatmap_with_image
from .action_modes import action_options, action_descriptor, apply_eef_delta
from .data.common import write_json


def _project(matrix, point):
    return None if point is None else matrix @ np.r_[point, 1.]


def _inside(pixel, width, height):
    return bool(pixel is not None and np.isfinite(pixel).all() and 0 <= pixel[0] < width and 0 <= pixel[1] < height)


def _normalized(heatmap):
    value = np.asarray(heatmap, np.float32)
    low, high = value.min(), value.max()
    return (value - low) / (high - low) if high > low else np.zeros_like(value)


def _gt_heatmaps(pixels, height, width):
    yy, xx = np.mgrid[:height, :width]
    result = np.zeros((len(pixels), height, width), np.float32)
    for index, pixel in enumerate(pixels):
        if _inside(pixel, width, height):
            result[index] = np.exp(-((xx - pixel[0]) ** 2 + (yy - pixel[1]) ** 2) / 8.)
    return result


def _montage(stage, *, title, auxiliary):
    rgb, maps = stage['rgb'], stage['heatmap']
    views, height, width, _ = rgb.shape
    extra = [key for key in ('target_prior', 'reference_prior', 'relation_anchor') if key in stage]
    columns = ['RGB', 'GT point (sigma=2px)', 'Prediction heatmap', 'GT / policy / heatmap TCP'] + extra
    left, top, label = 72, 82, 24
    canvas = Image.new('RGB', (left + width * len(columns), top + views * (height + label)), (35, 35, 35))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), title, fill='white')
    draw.text((8, 24), 'Green: GT   Red: policy TCP   Cyan: heatmap TCP' +
              (' (auxiliary EEF head)' if auxiliary else ''), fill='white')
    draw.text((8, 42), 'Heatmaps use per-view min/max colors; NPZ retains probabilities. Outside points are not clipped.', fill='white')
    for column, name in enumerate(columns):
        draw.text((left + column * width + 3, top - 18), name, fill='white')
    for view in range(views):
        y = top + view * (height + label)
        draw.text((4, y + 6), f'View {view}', fill='white')
        blended = blend_heatmap_with_image(rgb[view], _normalized(maps[view]))
        images = [rgb[view], blend_heatmap_with_image(rgb[view], stage['gt_heatmap'][view]), blended, blended]
        images.extend(blend_heatmap_with_image(rgb[view], _normalized(stage[key][view])) for key in extra)
        for column, image in enumerate(images):
            tile = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
            if column == 3:
                marker = ImageDraw.Draw(tile)
                for key, color in (('gt_pixels', 'lime'), ('heatmap_waypoint_pixels', 'cyan'),
                                   ('policy_pixels', 'red')):
                    pixels = stage.get(key)
                    if pixels is None or not _inside(pixels[view], width, height):
                        continue
                    x, py = map(float, pixels[view])
                    if key == 'gt_pixels':
                        marker.ellipse((x - 5, py - 5, x + 5, py + 5), outline=color, width=2)
                    else:
                        marker.line((x - 4, py - 4, x + 4, py + 4), fill=color, width=2)
                        marker.line((x - 4, py + 4, x + 4, py - 4), fill=color, width=2)
            canvas.paste(tile, (left + column * width, y))
        visible = _inside(stage['gt_pixels'][view], width, height)
        draw.text((left + 3 * width + 3, y + height + 3), f'GT inside view: {visible}', fill='white')
    return canvas


class EvaluationVisualizer:
    def __init__(self, output_dir, every=1):
        if type(every) is not int or every < 1:
            raise ValueError('visualize_every must be a positive integer')
        self.root, self.every = Path(output_dir).expanduser().resolve(), every
        if self.root.exists():
            raise FileExistsError(f'Use a new visualization directory: {self.root}')
        self.entries = []

    def due(self, index):
        return index % self.every == 0

    def write(self, index, payload, row, meta, predicted, options, metrics):
        if not payload or not payload.get('stages'):
            raise RuntimeError('Evaluation visualization requested but agent returned no image payload')
        if not self.entries:
            self.root.mkdir(parents=True, exist_ok=False)
        directory = self.root / f'sample_{index:06d}'
        directory.mkdir()
        options = action_options(options)
        mode = options['mode']
        gt = np.asarray(row['gripper_pose'][:3], np.float64)
        policy_tcp = (predicted[:3] if mode == 'eef' else
                      apply_eef_delta(row['tcp_pose'], predicted[:6], options)[:3]
                      if mode == 'relative_eef' else None)
        waypoint = np.asarray(payload['heatmap_waypoint_world'])
        record = dict(id=meta['id'], task=meta['task'], episode_index=meta['episode_index'],
                      frame=meta['frame'], instruction=row['goal'], action_spec=action_descriptor(options),
                      predicted_action=predicted.tolist(), target_action=row['action'].tolist(),
                      gt_tcp_world=gt.tolist(), policy_tcp_world=None if policy_tcp is None else policy_tcp.tolist(),
                      heatmap_waypoint_world=waypoint.tolist(), heatmap_is_auxiliary=mode != 'eef',
                      metrics=metrics, stages={})
        for name, source in payload['stages'].items():
            if name not in ('mvt1', 'mvt2'):
                raise ValueError(f'Unexpected visualization stage: {name}')
            stage = dict(source)
            projection = stage['world_to_pixel']
            stage.update(gt_pixels=_project(projection, gt), policy_pixels=_project(projection, policy_tcp),
                         heatmap_waypoint_pixels=_project(projection, waypoint))
            views, height, width, channels = stage['rgb'].shape
            if channels != 3 or stage['heatmap'].shape != (views, height, width):
                raise ValueError('Visualization RGB/heatmap shapes do not match')
            stage['gt_heatmap'] = _gt_heatmaps(stage['gt_pixels'], height, width)
            _montage(stage, title=f"{meta['id']} | {name} | {mode}", auxiliary=mode != 'eef').save(directory / f'{name}.png')
            np.savez_compressed(directory / f'{name}.npz', **{key: value for key, value in stage.items() if value is not None})
            record['stages'][name] = dict(png=f'{name}.png', arrays=f'{name}.npz',
                                         gt_in_view=[_inside(p, width, height) for p in stage['gt_pixels']])
        write_json(directory / 'prediction.json', record)
        self.entries.append(dict(id=meta['id'], task=meta['task'], instruction=row['goal'],
                                 directory=directory.name, stages=record['stages'], metrics=metrics))

    def finish(self):
        if not self.entries:
            return dict(output_dir=str(self.root), every=self.every, samples=0)
        write_json(self.root / 'index.json', dict(every=self.every, samples=self.entries))
        sections = []
        for entry in self.entries:
            folder = entry['directory']
            images = ''.join(f'<a href="{folder}/{stage["png"]}"><img loading="lazy" src="{folder}/{stage["png"]}" alt="{name}"></a>'
                             for name, stage in entry['stages'].items())
            sections.append(f'<section><h2>{html.escape(entry["id"])}</h2><p>{html.escape(entry["instruction"])}</p>'
                            f'<a href="{folder}/prediction.json">Prediction and metrics JSON</a>{images}</section>')
        (self.root / 'index.html').write_text(
            '<!doctype html><meta charset="utf-8"><title>OHT evaluation predictions</title>'
            '<style>body{font-family:sans-serif;margin:24px}img{display:block;max-width:100%;margin:12px 0}'
            'section{border-bottom:1px solid #aaa;margin-bottom:24px}</style>'
            '<h1>OHT evaluation predictions</h1><p>GT is used only for post-inference diagnostics.</p>' + ''.join(sections), encoding='utf-8')
        return dict(output_dir=str(self.root), every=self.every, samples=len(self.entries),
                    index=str(self.root / 'index.html'),
                    latency_note='Sampled latency includes CPU diagnostic capture; PNG/NPZ writing is outside inference timing.')
