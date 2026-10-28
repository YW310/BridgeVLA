"""Train-only, replay-bound normalization for current-TCP-relative translations."""
import numpy as np

from .chunk_config import chunk_settings
from .data.common import file_digest, inside


def action_scale_statistics(dataset, settings, progress=None):
    # Read only small label files. Current TCP comes from the checksum-bound
    # sample index; loading RGB-D observations here would duplicate expensive IO.
    rows = dataset.samples
    if not rows or any(row['split'] != 'train' for row in rows):
        raise ValueError('Action normalization requires a nonempty training split')
    indices = np.linspace(0, len(rows) - 1, min(len(rows), settings['normalization_max_samples']), dtype=int)
    distances = []
    for completed, index in enumerate(indices, 1):
        row = rows[index]
        path = inside(dataset.root, row['targets'])
        if file_digest(path) != row['targets_sha256']:
            raise ValueError(f'Normalization targets checksum mismatch: {row["id"]}')
        with np.load(path, allow_pickle=False) as source:
            poses = source['action_chunk'][:settings['horizon'], :3]
            valid = source['action_valid'][:settings['horizon']]
        current = np.asarray(row['current_tcp'][:3], dtype=np.float64)
        if (poses.shape != (settings['horizon'], 3) or valid.shape != poses.shape[:1]
                or valid.dtype != np.bool_ or not np.isfinite(poses[valid]).all()
                or not np.isfinite(current).all()):
            raise ValueError(f'Invalid normalization targets/current TCP: {row["id"]}')
        # An isotropic radial scale preserves rigid-frame invariance and avoids
        # amplifying low-variance axes. Padding never contributes statistics.
        distances.append(np.linalg.norm(poses[valid].astype(np.float64) - current, axis=-1))
        if progress and (completed == 1 or completed % 256 == 0 or completed == len(indices)):
            progress(completed, len(indices))
    distances = np.concatenate(distances)
    if not len(distances):
        raise ValueError('No valid training futures for action normalization')
    quantile = float(np.quantile(distances, settings['normalization_quantile']))
    scale = max(quantile, settings['normalization_min_scale'])
    return dict(method='train_quantile', replay_sha256=dataset.contract['sha256'], split='train',
                horizon=settings['horizon'], samples=len(indices), valid_ticks=len(distances),
                quantile=settings['normalization_quantile'], quantile_m=quantile,
                minimum_m=settings['normalization_min_scale'], action_position_scale=scale)


def resolve_training_normalization(config, dataset, *, distributed=None, rank=0):
    settings = chunk_settings(config, dataset.contract)
    if settings['action_position_scale'] != 'train_quantile':
        return
    result = [None, None]
    if rank == 0:
        try:
            def progress(completed, total):
                print(f'[OHT normalization] training labels {completed}/{total}', flush=True)
            result[0] = action_scale_statistics(dataset, settings, progress)
        except Exception as error:
            # All ranks must receive failures before anyone enters model/DDP
            # construction, otherwise peers can wait forever in collectives.
            result[1] = f'{type(error).__name__}: {error}'
    if distributed is not None:
        distributed.broadcast_object_list(result, src=0)
    if result[1] is not None:
        raise ValueError(f'Cannot resolve chunk normalization: {result[1]}')
    statistics = result[0]
    config['chunk_policy']['action_position_scale'] = statistics['action_position_scale']
    config['chunk_normalization'] = statistics
    if rank == 0:
        print(f'[OHT normalization] action_position_scale={statistics["action_position_scale"]:.6g} m '
              f'from {statistics["valid_ticks"]} valid training ticks', flush=True)
