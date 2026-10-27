"""Small CPU panels for predicted/observed chunk paths, without GT model inputs."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def path_panel(predicted, target=None, valid=None, *, target_color='#40db70',
               current=None, sample_id='', step=None, action_dt=None):
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
    lower, upper = points.min(0) - .01, points.max(0) + .01
    image = Image.new('RGB', (900, 380), '#202020')
    draw = ImageDraw.Draw(image)
    title = ('GT future TCP positions (valid labels only)' if observed_only else
             'Chunk: red prediction / green observed future (valid labels only)')
    draw.text((8, 5), title, fill='white')
    identity = f'step={step} ' if step is not None else ''
    identity += f'{sample_id} | world XYZ (m) | valid ticks={len(target) if len(target) else len(predicted)}/{horizon}'
    if action_dt is not None:
        identity += f' | dt={float(action_dt):.6g}s'
    draw.text((8, 22), identity[:145], fill='white')
    if len(target) and len(predicted):
        error = np.linalg.norm(predicted - target, axis=1) * 1000
        text = f'Position error: mean={error.mean():.2f} mm | max={error.max():.2f} mm | last valid={error[-1]:.2f} mm'
        if len(origin):
            hold = np.linalg.norm(target - origin, axis=1).mean() * 1000
            text += f' | hold-current baseline={hold:.2f} mm'
        draw.text((8, 40), text, fill='white')
    for panel, (a, b, label) in enumerate(((0, 1, 'XY'), (0, 2, 'XZ'), (1, 2, 'YZ'))):
        center = (lower + upper) / 2
        span = max(upper[a] - lower[a], upper[b] - lower[b])
        draw.text((panel * 300 + 10, 63), f'{label} | equal axis scale', fill='white')
        draw.text((panel * 300 + 10, 80),
                  f'{"XYZ"[a]} [{center[a]-span/2:.3f},{center[a]+span/2:.3f}] '
                  f'{"XYZ"[b]} [{center[b]-span/2:.3f},{center[b]+span/2:.3f}] m', fill='white')
        draw.rectangle((panel * 300 + 30, 110, panel * 300 + 270, 350), outline='#555555')
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
            draw.line((x-4, y, x+4, y), fill='white', width=2)
            draw.line((x, y-4, x, y+4), fill='white', width=2)
    draw.text((8, 361), 'White cross: current measured TCP. Red/green use the SAME valid future ticks and scale.', fill='white')
    return image


def record(payload, step, output, settings, logger):
    image = path_panel(payload['predicted'], payload['target'], payload['valid'],
                       current=payload.get('current'), sample_id=payload.get('sample_id', ''),
                       step=step, action_dt=payload.get('action_dt'))
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
            panel = _montage(stage, name, False)
            if settings['save_png']:
                panel.save(directory / f'{name}_step_{step:08d}.png')
            if settings['tensorboard']:
                logger.writer.add_image('train_visualization/' + name, np.asarray(panel), step, dataformats='HWC')
    logger.flush()
