"""Check V4 target coverage from Parquet without decoding RGB-D or writing replay."""
import json
from pathlib import Path
import sys

import numpy as np

from .actions import gripper_states, keypoints, world_tcp_poses
from .chunk_actions import chunk_options, make_chunk_targets, resolve_chunk_timing
from .common import digest, file_digest, inside
from .observation import validate_data_config
from .source_config import resolve_dataset_config


def check_workspace(root, manifest_path, config, sample_stride=10, *, chunk_config=None, margin=.02, progress=False):
    """Match the builder's sampled key/future targets, including no-key tails.

    Keep/exclude boxes affect observed surfaces, not legal action coordinates;
    this check deliberately tests scene_bounds only. It does not validate video
    streams, point visibility or collision geometry.
    """
    import pyarrow.parquet as pq
    validate_data_config(config)
    if type(sample_stride) is not int or sample_stride < 1:
        raise ValueError('sample_stride must be a positive integer')
    if not np.isfinite(margin) or margin <= 0:
        raise ValueError('Workspace suggestion margin must be finite and positive')
    options = chunk_options(chunk_config)
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    contents = dict(manifest)
    expected = contents.pop('manifest_sha256', None)
    if expected != digest(contents) or manifest.get('schema') != 'oht_audit_v1' or not manifest.get('episodes'):
        raise ValueError('Invalid audit manifest or checksum mismatch')
    bounds = np.asarray(config['scene_bounds'], dtype=float)
    profiles, records, examples = {}, [], []
    minimum, maximum = np.full(3, np.inf), np.full(3, -np.inf)
    total, outside_count, windows, skipped = 0, 0, 0, 0
    for episode, record in enumerate(manifest['episodes'], 1):
        if progress:
            print(f"[OHT workspace] episode={episode}/{len(manifest['episodes'])} "
                  f"task={record['task']} id={record['episode_index']}", file=sys.stderr, flush=True)
        if file_digest(inside(root, record['path'])) != record['parquet_sha256']:
            raise ValueError(f"Raw Parquet changed after audit: {record['path']}")
        if record['dataset'] not in profiles:
            profiles[record['dataset']] = resolve_dataset_config(inside(root, record['dataset']), config)[0]
        data = profiles[record['dataset']]
        # The audit plus matching file digest binds the raw validation. Read
        # only present robot states/timing, not RGB-D references or object data;
        # avoid repeating per-frame network-file existence checks here.
        columns = pq.read_table(inside(root, record['path']), columns=[
            'timestamp', 'observation.state', 'observation.ee_pos_world',
            'observation.ee_quat_world', 'instruction_id']).to_pydict()
        stamps = np.asarray(columns['timestamp'], dtype=np.float64)
        options = resolve_chunk_timing(stamps, options)
        _, grip = gripper_states(columns['observation.state'], data['gripper'])
        poses = world_tcp_poses(columns, data['link_to_tcp'], data['ee_quaternion_order'])
        joints = np.asarray(columns['observation.state'], dtype=np.float32)[:, :6]
        keys = keypoints(poses, grip, columns['instruction_id'], timestamps=stamps, **data['keypoints'])
        frames = sorted(set(range(0, len(stamps) - 1, sample_stride)) | {key for key in keys if key < len(stamps) - 1})
        target_frames = set()
        for frame in frames:
            targets, _ = make_chunk_targets(poses, joints, grip, stamps, frame, keys, options)
            if targets is None:
                skipped += 1
                continue
            windows += 1
            target_frames.update(targets['action_frames'][targets['action_valid']].tolist())
            if bool(targets['key_valid']):
                target_frames.add(int(targets['key_frame']))
        if not target_frames:
            continue
        selected = np.asarray(sorted(target_frames), dtype=int)
        # The builder stores float32 target positions; compare that same precision.
        points = poses[selected, :3].astype(np.float32)
        lower, upper = points.min(0), points.max(0)
        minimum, maximum = np.minimum(minimum, lower), np.maximum(maximum, upper)
        bad = ~((points >= bounds[:3]) & (points < bounds[3:])).all(-1)
        count = int(bad.sum())
        total, outside_count = total + len(selected), outside_count + count
        if progress:
            print(f'[OHT workspace] target_frames={total} outside={outside_count}', file=sys.stderr, flush=True)
        records.append(dict(task=record['task'], episode_index=record['episode_index'],
                            target_min=lower.tolist(), target_max=upper.tolist(), outside_targets=count))
        for index in np.flatnonzero(bad)[:max(0, 20 - len(examples))]:
            point = points[index]
            examples.append(dict(task=record['task'], episode_index=record['episode_index'],
                frame=int(selected[index]), xyz=point.tolist(),
                outside_axes=[axis for i, axis in enumerate('xyz') if not bounds[i] <= point[i] < bounds[i + 3]]))
    if not total:
        raise ValueError('No valid V4 targets to check; inspect audit, timing and sample stride')
    suggested = bounds if not outside_count else np.r_[np.minimum(bounds[:3], minimum - margin),
                                                       np.maximum(bounds[3:], maximum + margin)]
    return dict(kind='v4_workspace_target_coverage', valid=outside_count == 0,
                audit_manifest_sha256=expected, sample_stride=sample_stride, chunk=options,
                scene_bounds=bounds.tolist(), target_min=minimum.tolist(), target_max=maximum.tolist(),
                unique_target_frames=total, outside_target_frames=outside_count,
                valid_windows=windows, skipped_windows=skipped,
                suggestion_margin_m=margin, suggested_scene_bounds=suggested.tolist(),
                examples=examples, episodes=records,
                scope='selected measured key/future TCP targets only; no video/point-cloud/collision validation')
