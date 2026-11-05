"""Small CPU panels for predicted/observed chunk paths, without GT model inputs."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def scene_background(observation, max_points=24000):
    """Bounded, deterministic RGB/XYZ snapshot from current cameras only (CHW)."""
    if max_points < 1:
        raise ValueError('Scene point budget must be positive')
    names = sorted(key[:-12] for key in observation if key.endswith('_point_cloud')
                   and key[:-12] + '_rgb' in observation)
    points, colors = [], []
    for name in names:
        cloud, rgb = np.asarray(observation[name + '_point_cloud']), np.asarray(observation[name + '_rgb'])
        if cloud.ndim != 3 or cloud.shape[0] != 3 or rgb.shape != cloud.shape:
            raise ValueError('Scene RGB/XYZ must be aligned CHW arrays')
        cloud, rgb = cloud.reshape(3, -1).T, rgb.reshape(3, -1).T
        valid = np.isfinite(cloud).all(axis=1) & np.isfinite(rgb).all(axis=1)
        indices = np.flatnonzero(valid)
        count = min(len(indices), max(1, max_points // max(len(names), 1)))
        if not count:
            continue
        selected = indices[np.linspace(0, len(indices) - 1, count, dtype=np.int64)]
        points.append(cloud[selected].astype(np.float32))
        color = rgb[selected]
        if np.issubdtype(rgb.dtype, np.floating) and color.max() <= 1.:
            color = color * 255
        colors.append(np.clip(color, 0, 255).astype(np.uint8))
    xyz = np.concatenate(points) if points else np.empty((0, 3), np.float32)
    rgb = np.concatenate(colors) if colors else np.empty((0, 3), np.uint8)
    return dict(scene_points=xyz[:max_points].copy(), scene_colors=rgb[:max_points].copy())


def _scene_arrays(points, colors):
    if points is None:
        return np.empty((0, 3)), np.empty((0, 3), np.uint8)
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Scene points must have shape [N,3]')
    colors = np.full(points.shape, 140, np.uint8) if colors is None else np.asarray(colors)
    if colors.shape != points.shape:
        raise ValueError('Scene colors must match scene points')
    valid = np.isfinite(points).all(axis=1) & np.isfinite(colors).all(axis=1)
    indices = np.flatnonzero(valid)
    indices = indices[np.linspace(0, len(indices) - 1, min(len(indices), 24000), dtype=np.int64)]
    return points[indices], np.clip(colors[indices], 0, 255).astype(np.uint8)


def _scene_tile(points, colors, center, span, a, b):
    """Project the measured scene into the exact metric axes used by the paths."""
    tile = np.full((240, 240, 3), (246, 248, 250), np.uint8)
    tile[::60, :] = (222, 228, 234)
    tile[:, ::60] = (222, 228, 234)
    if len(points):
        local = (points[:, [a, b]] - center[[a, b]]) * (240 / span)
        pixels = np.rint(local * [1, -1] + [120, 120])
        inside = ((pixels >= 0) & (pixels < 240)).all(axis=1)
        pixels, color = pixels[inside].astype(int), colors[inside]
        # Fade the scene so it provides context without hiding the trajectories.
        color = (.7 * color + .3 * np.array([246, 248, 250])).astype(np.uint8)
        for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
            tile[np.minimum(pixels[:, 1] + dy, 239), np.minimum(pixels[:, 0] + dx, 239)] = color
    return Image.fromarray(tile)


def path_panel(predicted, target=None, valid=None, *, target_color='#40db70',
               current=None, sample_id='', step=None, action_dt=None,
               scene_points=None, scene_colors=None):
    observed_only = predicted is None
    predicted = np.empty((0, 3)) if observed_only else np.asarray(predicted, dtype=float)[:, :3]
    target = np.empty((0, 3)) if target is None else np.asarray(target, dtype=float)[:, :3]
    horizon = len(target) if observed_only else len(predicted)
    if len(target) and not observed_only and len(target) != len(predicted):
        raise ValueError('Prediction and future labels must have the same horizon')
    if valid is not None:
        valid = np.asarray(valid, bool)
        if valid.shape != (horizon,):
            raise ValueError('Chunk validity must match the plotted horizon')
        # The loss ignores padded futures for BOTH trajectories. Drawing all
        # predicted ticks made unsupervised tail values look like real errors.
        if len(target):
            target = target[valid]
        if len(predicted):
            predicted = predicted[valid]
    origin = np.empty((0, 3)) if current is None else np.asarray(current, dtype=float)[:3].reshape(1, 3)
    points = np.concatenate((predicted, target, origin))
    if not len(points) or not np.isfinite(points).all():
        raise ValueError('Plotted chunk positions must be nonempty and finite')
    scene_points, scene_colors = _scene_arrays(scene_points, scene_colors)
    # Keep a local metric view around valid paths; distant background never
    # changes the scale or compresses the prediction/GT comparison.
    margin = .05 if len(scene_points) else .01
    lower, upper = points.min(0) - margin, points.max(0) + margin
    image = Image.new('RGB', (900, 400), '#f6f8fa')
    draw = ImageDraw.Draw(image)
    title = ('GT future TCP positions (valid labels only)' if observed_only else
             'Chunk: red prediction / green observed future (valid labels only)')
    draw.text((8, 5), title, fill='#202b36')
    identity = f'step={step} ' if step is not None else ''
    identity += f'{sample_id} | world XYZ (m) | valid ticks={len(target) if len(target) else len(predicted)}/{horizon}'
    if action_dt is not None:
        identity += f' | dt={float(action_dt):.6g}s'
    draw.text((8, 22), identity[:145], fill='#202b36')
    if len(target) and len(predicted):
        error = np.linalg.norm(predicted - target, axis=1) * 1000
        text = f'Position error: mean={error.mean():.2f} mm | max={error.max():.2f} mm | last valid={error[-1]:.2f} mm'
        if len(origin):
            hold = np.linalg.norm(target - origin, axis=1).mean() * 1000
            text += f' | hold-current baseline={hold:.2f} mm'
        draw.text((8, 40), text, fill='#202b36')
    for panel, (a, b, label) in enumerate(((0, 1, 'XY'), (0, 2, 'XZ'), (1, 2, 'YZ'))):
        center = (lower + upper) / 2
        span = max(upper[a] - lower[a], upper[b] - lower[b])
        draw.text((panel * 300 + 10, 63), f'{label} | equal axis scale', fill='#202b36')
        draw.text((panel * 300 + 10, 80),
                  f'{"XYZ"[a]} [{center[a]-span/2:.3f},{center[a]+span/2:.3f}] '
                  f'{"XYZ"[b]} [{center[b]-span/2:.3f},{center[b]+span/2:.3f}] m', fill='#202b36')
        image.paste(_scene_tile(scene_points, scene_colors, center, span, a, b), (panel * 300 + 30, 110))
        draw.rectangle((panel * 300 + 30, 110, panel * 300 + 270, 350), outline='#8895a3')
        for values, color in ((target, target_color), (predicted, '#ff6060')):
            if not len(values):
                continue
            pixels = [(round(panel * 300 + 150 + float(p[a]-center[a]) * 240/span),
                       round(230 - float(p[b]-center[b]) * 240/span)) for p in values]
            if len(pixels) > 1:
                draw.line(pixels, fill=color, width=2)
            for x, y in pixels:
                draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
        if len(origin):
            x = round(panel * 300 + 150 + float(origin[0, a]-center[a]) * 240/span)
            y = round(230 - float(origin[0, b]-center[b]) * 240/span)
            for color, width in (('#202b36', 4), ('white', 2)):
                draw.line((x-4, y, x+4, y), fill=color, width=width)
                draw.line((x, y-4, x, y+4), fill=color, width=width)
    draw.text((8, 361), 'White cross: current measured TCP. Red/green use the SAME valid future ticks and scale.', fill='#202b36')
    background = 'Current RGB-D scene projected in world coordinates (local window).' if len(scene_points) else 'No valid scene points; showing a light coordinate grid.'
    draw.text((8, 379), background, fill='#516273')
    return image


def record(payload, step, output, settings, logger):
    image = path_panel(payload['predicted'], payload['target'], payload['valid'],
                       current=payload.get('current'), sample_id=payload.get('sample_id', ''),
                       step=step, action_dt=payload.get('action_dt'),
                       scene_points=payload.get('scene_points'), scene_colors=payload.get('scene_colors'))
    directory = Path(settings['output_dir'])
    if not directory.is_absolute():
        directory = Path(output) / directory
    if settings['save_png']:
        directory.mkdir(parents=True, exist_ok=True)
        image.save(directory / f'chunk_step_{step:08d}.png')
        np.savez_compressed(directory / f'chunk_step_{step:08d}.npz',
                            predicted=payload['predicted'], target=payload['target'], valid=payload['valid'],
                            current=np.asarray(payload.get('current', [])),
                            sample_id=payload.get('sample_id', ''), action_dt=payload.get('action_dt', np.nan),
                            scene_points=payload.get('scene_points', np.empty((0, 3), np.float32)),
                            scene_colors=payload.get('scene_colors', np.empty((0, 3), np.uint8)),
                            step=step)
    if settings['tensorboard']:
        logger.writer.add_image('train_visualization/chunk', np.asarray(image), step, dataformats='HWC')
    # Policy heatmaps come from the exact prediction-only forward, with no synthetic GT panel.
    policy = payload.get('policy')
    if policy:
        from .runtime.online_visualization import _montage
        for name in ('mvt1', 'mvt2'):
            stage = policy.get('stages', {}).get(name)
            if stage is None:
                continue
            stage = {key: np.asarray(value) for key, value in stage.items()}
            # Use the same first chunk action as the online server. These are
            # world coordinates; each stage's affine map already includes its
            # cube normalization and (for MVT2) prediction-centered crop.
            first_tcp = np.asarray(payload['predicted'])[0, :3]
            waypoint = np.asarray(policy['heatmap_waypoint_world'])
            projection = stage['world_to_pixel']
            stage['policy_pixels'] = projection @ np.r_[first_tcp, 1.]
            stage['heatmap_waypoint_pixels'] = projection @ np.r_[waypoint, 1.]
            # Keep full tick indices and one shared validity mask so no line
            # connects across padded labels or an offscreen part of a path.
            stage['chunk_valid'] = np.asarray(payload['valid'], bool)
            for key, source in (('chunk_predicted', 'predicted'), ('chunk_target', 'target')):
                world = np.asarray(payload[source])[:, :3]
                stage[key + '_world'] = world
                stage[key + '_pixels'] = np.einsum('vij,kj->vki', projection, np.c_[world, np.ones(len(world))])
            if payload.get('current') is not None:
                stage['measured_tcp_world'] = np.asarray(payload['current'])[:3]
                stage['measured_tcp_pixels'] = projection @ np.r_[stage['measured_tcp_world'], 1.]
            panel = _montage(stage, name, True, policy_label='chunk[0] TCP')
            if settings['save_png']:
                panel.save(directory / f'{name}_step_{step:08d}.png')
                np.savez_compressed(directory / f'{name}_step_{step:08d}.npz', **stage,
                                    policy_tcp_world=first_tcp, heatmap_waypoint_world=waypoint,
                                    sample_id=payload.get('sample_id', ''), step=step,
                                    action_dt=payload.get('action_dt', np.nan))
            if settings['tensorboard']:
                logger.writer.add_image('train_visualization/' + name, np.asarray(panel), step, dataformats='HWC')
    logger.flush()
