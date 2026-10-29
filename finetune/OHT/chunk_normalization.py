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


def proprio_statistics(dataset, settings, progress=None):
    """Fit q6 from measured training states; opening already has physical [0,1] units."""
    rows = dataset.samples
    if not rows or any(row['split'] != 'train' for row in rows):
        raise ValueError('Proprio normalization requires a nonempty training split')
    indices = np.linspace(0, len(rows) - 1, min(len(rows), settings['normalization_max_samples']), dtype=int)
    joints = []
    for completed, index in enumerate(indices, 1):
        row = rows[index]
        path = inside(dataset.root, row['observation'])
        if progress and (completed == 1 or completed % 64 == 0 or completed == len(indices)):
            progress(completed, len(indices))
        if file_digest(path) != row['observation_sha256']:
            raise ValueError(f'Normalization observation checksum mismatch: {row["id"]}')
        # NPZ access is lazy: only decompress the two measured-state arrays.
        # Hashing still reads the file, keeping statistics bound to the replay.
        with np.load(path, allow_pickle=False) as source:
            joint = np.asarray(source['joint_positions'], dtype=np.float64)
            opening = np.asarray(source['gripper_open01'], dtype=np.float64)
        if (joint.shape != (6,) or opening.shape != () or not np.isfinite(joint).all()
                or not np.isfinite(opening) or not 0 <= opening <= 1):
            raise ValueError(f'Invalid normalization proprioception: {row["id"]}')
        joints.append(joint)
    quantile = settings['normalization_quantile']
    lower, upper = np.quantile(joints, [1 - quantile, quantile], axis=0)
    center = np.r_[(lower + upper) / 2, .5]
    scale = np.r_[np.maximum((upper - lower) / 2, settings['proprio_min_scale']), .5]
    return dict(method='train_quantile', replay_sha256=dataset.contract['sha256'], split='train',
                samples=len(indices), quantile=quantile, minimum_joint_scale_rad=settings['proprio_min_scale'],
                joint_lower=lower.tolist(), joint_upper=upper.tolist(),
                proprio_center=center.tolist(), proprio_scale=scale.tolist())


def resolve_training_normalization(config, dataset, *, distributed=None, rank=0):
    settings = chunk_settings(config, dataset.contract)
    action_pending = settings['action_position_scale'] == 'train_quantile'
    proprio_pending = settings['proprio_normalization'] == 'train_quantile' and settings['proprio_center'] is None
    if not (action_pending or proprio_pending):
        return
    result = [None, None]
    if rank == 0:
        try:
            def progress(completed, total):
                print(f'[OHT normalization] training labels {completed}/{total}', flush=True)
            result[0] = {}
            if action_pending:
                result[0]['action'] = action_scale_statistics(dataset, settings, progress)
            if proprio_pending:
                def proprio_progress(completed, total):
                    print(f'[OHT normalization] measured training states {completed}/{total}', flush=True)
                result[0]['proprio'] = proprio_statistics(dataset, settings, proprio_progress)
        except Exception as error:
            # All ranks must receive failures before anyone enters model/DDP
            # construction, otherwise peers can wait forever in collectives.
            result[1] = f'{type(error).__name__}: {error}'
    if distributed is not None:
        distributed.broadcast_object_list(result, src=0)
    if result[1] is not None:
        raise ValueError(f'Cannot resolve chunk normalization: {result[1]}')
    if action_pending:
        statistics = result[0]['action']
        config['chunk_policy']['action_position_scale'] = statistics['action_position_scale']
        config['chunk_normalization'] = statistics
        if rank == 0:
            print(f'[OHT normalization] action_position_scale={statistics["action_position_scale"]:.6g} m '
                  f'from {statistics["valid_ticks"]} valid training ticks', flush=True)
    if proprio_pending:
        statistics = result[0]['proprio']
        for name in ('proprio_center', 'proprio_scale'):
            config['chunk_policy'][name] = statistics[name]
        config['chunk_proprio_normalization'] = statistics
        if rank == 0:
            print(f'[OHT normalization] proprio center={statistics["proprio_center"]} '
                  f'scale={statistics["proprio_scale"]}', flush=True)
