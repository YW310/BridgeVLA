"""Small CPU panels for predicted/observed chunk paths, without GT model inputs."""
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw


def path_panel(predicted, target=None, valid=None):
    predicted = np.asarray(predicted, dtype=float)[:, :3]
    target = np.empty((0, 3)) if target is None else np.asarray(target, dtype=float)[:, :3]
    if valid is not None:
        target = target[np.asarray(valid, bool)]
    points = np.concatenate((predicted, target))
    lower, upper = points.min(0) - .01, points.max(0) + .01
    image = Image.new('RGB', (900, 320), '#202020')
    draw = ImageDraw.Draw(image)
    draw.text((8, 5), 'Chunk: red prediction / green observed future (valid labels only)', fill='white')
    for panel, (a, b, label) in enumerate(((0, 1, 'XY'), (0, 2, 'XZ'), (1, 2, 'YZ'))):
        draw.text((panel * 300 + 10, 28), label, fill='white')
        for values, color in ((target, '#40db70'), (predicted, '#ff6060')):
            if not len(values):
                continue
            normalized = (values - lower) / (upper - lower)
            pixels = [(panel * 300 + 15 + float(p[a]) * 270, 300 - float(p[b]) * 245) for p in normalized]
            if len(pixels) > 1:
                draw.line(pixels, fill=color, width=2)
            for x, y in pixels:
                draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
    return image


def record(payload, step, output, settings, logger):
    image = path_panel(payload['predicted'], payload['target'], payload['valid'])
    directory = Path(settings['output_dir'])
    if not directory.is_absolute():
        directory = Path(output) / directory
    if settings['save_png']:
        directory.mkdir(parents=True, exist_ok=True)
        image.save(directory / f'chunk_step_{step:08d}.png')
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
