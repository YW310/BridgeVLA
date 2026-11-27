"""Build a V4 replay from the OHT real wheel-assembly export."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from functools import partial
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from .actions import low_dim
from .chunk_actions import SCHEMA, chunk_options
from .chunk_replay import _target_bounds
from .common import REAL_LINEAGE_VERSION, REAL_TASK, digest, file_digest, inside, write_json
from .observation import validate_data_config, validate_observation
from .real_episodes import DATASET_PATH, episode_records, read_episode
from .real_rgbd import FIXED_CAMERAS, NativeDepthReader, load_camera
from .video import IndexedVideoReader


IMAGE_SIZE = (120, 160)
GRIPPER = dict(source='config', open=1., close=0.)
GOAL = 'Grasp the wheel and fit it onto the shaft.'


def _data_config(calibrations):
    identity = np.eye(4).tolist()
    return dict(
        # The policy's world coordinates are the recorded UR Base frame.
        scene_bounds=[-.65, .02, -.05, .40, .94, 1.03],
        image_size=list(IMAGE_SIZE), cameras={
            name: dict(intrinsics=calibration.intrinsics, optical_to_sensor=identity)
            for name, calibration in calibrations.items()
        },
        intrinsics_source='config', camera_quaternion_order='xyzw',
        camera_extrinsic_direction='camera_to_world', link_to_tcp=identity,
        ee_quaternion_order='xyzw', video_alignment='frame_index',
        depth=dict(encoding='metric', kind='z', limits=[.001, 3.]),
        rotation_classes=72, gripper=GRIPPER, keypoints=dict(method='gripper'),
    )


def _build_episode(record, *, dataset, output, options, sample_stride):
    index = record['episode_index']
    samples, skipped = read_episode(dataset, record, options, sample_stride, GRIPPER)
    if not samples:
        return record, [], None, skipped
    calibrations = {name: load_camera(dataset, index, name, IMAGE_SIZE) for name in FIXED_CAMERAS}
    data = _data_config(calibrations)
    validate_data_config(data, resolved=True)
    profile = 'real-' + digest(data)
    rows = []
    with ExitStack() as stack:
        readers = {}
        for name in FIXED_CAMERAS:
            media = record['cameras'][name]
            rgb = IndexedVideoReader(inside(dataset, media['rgb_path']), media['frames'])
            stack.callback(rgb.close)
            depth = NativeDepthReader(inside(dataset, media['depth_path']))
            stack.callback(depth.close)
            readers[name] = rgb, depth
        for sample in samples:
            frame = sample['frame']
            sample_id = f'{REAL_TASK}/{index:06d}/{frame:06d}'
            targets = sample['targets']
            _target_bounds(targets, data['scene_bounds'], sample_id)
            opening = np.asarray(sample['opening'], dtype=np.float32)
            observation = dict(tcp_pose=sample['tcp'], joint_positions=sample['joints'],
                               gripper_open01=opening, low_dim_state=low_dim(opening, binary_state=sample['gripper']))
            for name, (rgb, depth) in readers.items():
                capture = sample['captures'][name]
                observation.update(calibrations[name].read_observation(rgb.read(capture), depth.read(capture), data))
            try:
                validate_observation(observation, data)
            except ValueError as exc:
                raise ValueError(f'{sample_id}: {exc}') from exc
            row = dict(id=sample_id, task=REAL_TASK, episode_index=index, episode_length=sample['episode_length'],
                       frame=frame, timestamp=sample['timestamp'], target_frame=int(targets['key_frame']),
                       current_tcp=sample['tcp'].tolist(), data_profile=profile, group=record['group'],
                       split=record['split'], goal=GOAL)
            for name, folder, values in (('observation', 'observations', observation), ('targets', 'targets', targets)):
                relative = f'{folder}/{sample_id}.npz'
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **values)
                row[name] = relative
                row[name + '_sha256'] = file_digest(path)
            rows.append(row)
    return record, rows, (profile, data), skipped


def _source_manifest(root, records, seed, episodes_per_split):
    return dict(seed=seed, lineage_version=REAL_LINEAGE_VERSION,
                conversion_report_sha256=file_digest(root / 'conversion_report.json'),
                checksum_inventory_sha256=file_digest(root / 'SHA256SUMS'),
                cameras=list(FIXED_CAMERAS), frame='UR Base', episodes_per_split=episodes_per_split,
                episodes=[{key: record[key] for key in ('episode_index', 'source_episode', 'source_parent',
                                                       'source_root', 'cohort', 'split', 'group')}
                          for record in records])


def _complete_replay(output, source, profiles, options, sample_stride, split_counts,
                     episode_counts, skipped_counts, skipped_episodes):
    if set(split_counts) != {'train', 'val', 'test'}:
        raise ValueError(f'A training replay needs samples in all three splits, got {dict(split_counts)}')
    index_hash = file_digest(output / 'samples.jsonl')
    contract = dict(schema=SCHEMA, replay_version=4, action_targets_version=2,
                    data_config=next(iter(profiles.values()))['data_config'], source_data_configs=profiles,
                    audit_manifest_sha256=digest(source), index_sha256=index_hash,
                    policy_frame='world', quaternion_order='xyzw', reference_point='tcp',
                    target='key_goal_and_continuous_absolute_tcp', language='task_goal',
                    sample_stride=sample_stride, joint_count=6, chunk=options,
                    gripper_convention='1=open,0=close', label_storage='separate_npz',
                    rebuild='real_oht_rgbd',
                    real_source=dict(world_frame='UR Base', capture_index=True, tcp_timing='recorded',
                                     lineage_version=REAL_LINEAGE_VERSION,
                                     supervision='measured TCP and gripper within T58 source spans'))
    contract['sha256'] = digest(contract)
    write_json(output / 'contract.json', contract)
    complete = dict(contract_sha256=contract['sha256'], index_sha256=index_hash, samples=sum(split_counts.values()),
                    split_counts=dict(split_counts), split_episode_counts=dict(episode_counts),
                    skipped_window_counts=dict(skipped_counts), skipped_episodes=skipped_episodes)
    write_json(output / 'complete.json', complete)
    return complete


def build(root, output, *, sample_stride=10, seed=0, workers=1, episodes_per_split=None):
    if sample_stride < 1 or workers < 1 or (episodes_per_split is not None and episodes_per_split < 1):
        raise ValueError('Sample stride, worker count and episodes per split must be positive')
    root = Path(root).resolve()
    dataset = root / DATASET_PATH
    output = Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError('Use a replay directory outside the source export')
    records = episode_records(root, seed)
    if episodes_per_split is not None:
        records = [record for split in ('train', 'val', 'test')
                   for record in [entry for entry in records if entry['split'] == split][:episodes_per_split]]
    options = chunk_options(dict(action_dt=.02, timestamp_atol=.01, timestamp_rtol=0.))
    output.mkdir(parents=True)
    source = _source_manifest(root, records, seed, episodes_per_split)
    write_json(output / 'source_manifest.json', source)
    run_episode = partial(_build_episode, dataset=dataset, output=output, options=options, sample_stride=sample_stride)
    profiles, skipped_episodes = {}, []
    split_counts, skipped_counts, episode_counts = Counter(), Counter(), Counter()
    started = perf_counter()
    index = output / 'samples.jsonl'
    with ExitStack() as stack:
        stream = stack.enter_context(index.open('w', encoding='utf-8'))
        pool = stack.enter_context(ProcessPoolExecutor(max_workers=workers)) if workers > 1 else None
        results = pool.map(run_episode, records) if pool is not None else map(run_episode, records)
        for finished, (record, rows, profile, skipped) in enumerate(results, 1):
            skipped_counts.update(skipped)
            if rows:
                profile_id, data = profile
                profiles.setdefault(profile_id, dict(data_config=data))
                split_counts[record['split']] += len(rows)
                episode_counts[record['split']] += 1
                for row in rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            else:
                skipped_episodes.append(record['episode_index'])
            if finished == 1 or finished % 5 == 0 or finished == len(records):
                print(f'[OHT real] episodes={finished}/{len(records)} samples={sum(split_counts.values())} '
                      f'elapsed_s={perf_counter()-started:.1f}', file=sys.stderr, flush=True)
    return _complete_replay(output, source, profiles, options, sample_stride, split_counts,
                            episode_counts, skipped_counts, skipped_episodes)
