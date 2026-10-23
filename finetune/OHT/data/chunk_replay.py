"""Rebuild V4 RGB-D observations and timed chunks directly from audited raw data."""
import json
import sys
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from .actions import gripper_states, keypoints, low_dim, world_tcp_poses
from .chunk_actions import SCHEMA, chunk_options, make_chunk_targets, resolve_chunk_timing
from .common import GOALS, digest, file_digest, inside, write_json
from .object_preview import capture_object_poses
from .observation import camera_observation, validate_data_config, validate_observation
from .reader import read_episode, validate_episode
from .source_config import resolve_dataset_config
from .video import EpisodeVideos, metric_depth
from .visualization import PreviewWriter


def _target_bounds(targets, bounds):
    bounds = np.asarray(bounds, dtype=float)
    points = targets['action_chunk'][targets['action_valid'], :3]
    if bool(targets['key_valid']):
        points = np.concatenate((points, targets['key_action'][None, :3]))
    if not ((points >= bounds[:3]) & (points < bounds[3:])).all():
        raise ValueError('V4 measured action target outside scene_bounds; inspect bounds, never clip GT')


def build(root, manifest_path, config, output, sample_stride=10, *, chunk_config=None,
          visualize_every=0, visualize_output_dir=None, object_cloud_config=None, progress=True):
    """Build a fresh V4 directory; existing observations are never copied/reused."""
    validate_data_config(config)
    options = chunk_options(chunk_config)
    if type(sample_stride) is not int or sample_stride < 1:
        raise ValueError('sample_stride must be a positive integer')
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    original = dict(manifest)
    recorded_hash = original.pop('manifest_sha256', None)
    if recorded_hash != digest(original):
        raise ValueError('Audit manifest checksum mismatch')
    if manifest.get('schema') != 'oht_audit_v1' or not manifest.get('episodes'):
        raise ValueError('Audit must contain valid OHT episodes')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Use a new V4 replay directory: {output}')
    profiles = {}
    groups, episodes = {}, set()
    for record in manifest['episodes']:
        split, group = record['split'], record['group']
        identity = record['task'], record['episode_index']
        if split not in ('train', 'val', 'test') or identity in episodes:
            raise ValueError('Invalid audit split or duplicate episode')
        episodes.add(identity)
        if groups.setdefault(group, split) != split:
            raise ValueError('Scene/trajectory group crosses dataset splits')
        profile = record['dataset']
        if profile not in profiles:
            resolved, sources = resolve_dataset_config(inside(root, profile), config)
            profiles[profile] = dict(data_config=resolved, metadata_sha256=sources)
    output.mkdir(parents=True)
    preview = PreviewWriter(visualize_output_dir or output / 'preview', visualize_every)
    object_writer = None
    if object_cloud_config is not None:
        from .object_cloud_build import BuildObjectCloudWriter
        object_writer = BuildObjectCloudWriter(output, object_cloud_config,
                                              preview_output=visualize_output_dir)
    contract = dict(schema=SCHEMA, replay_version=4, action_targets_version=2,
                    data_config=next(iter(profiles.values()))['data_config'],
                    source_data_configs=profiles, audit_manifest_sha256=recorded_hash,
                    policy_frame='world', quaternion_order='xyzw', reference_point='tcp',
                    target='key_goal_and_continuous_absolute_tcp', language='task_goal',
                    sample_stride=sample_stride, joint_count=6, chunk=options,
                    gripper_convention='1=open,0=close',
                    label_storage='separate_npz', rebuild='raw_observations_and_targets')
    if object_writer is not None:
        contract['diagnostic_object_clouds'] = object_writer.options
    contract['sha256'] = digest(contract)
    write_json(output / 'contract.json', contract)
    rows, skipped_windows, skipped_episodes, timing, html_frames = [], [], [], [], []
    started = last_report = perf_counter()
    for episode_number, record in enumerate(manifest['episodes'], 1):
        data = profiles[record['dataset']]['data_config']
        if file_digest(inside(root, record['path'])) != record['parquet_sha256']:
            raise ValueError(f"Raw Parquet changed after audit: {record['path']}")
        columns = read_episode(root, record)
        validate_episode(columns, record)
        stamps = np.asarray(columns['timestamp'], dtype=np.float64)
        resolved = resolve_chunk_timing(stamps, options)
        if options['action_dt'] is None:
            options = resolved
            contract['chunk'] = options
            contract.pop('sha256')
            contract['sha256'] = digest(contract)
            write_json(output / 'contract.json', contract)
        timing.append(dict(task=record['task'], episode_index=record['episode_index'],
                           raw_dt_min=float(np.diff(stamps).min()),
                           raw_dt_max=float(np.diff(stamps).max()),
                           raw_dt_median=float(np.median(np.diff(stamps)))))
        measured, grip = gripper_states(columns['observation.state'], data['gripper'])
        poses = world_tcp_poses(columns, data['link_to_tcp'], data['ee_quaternion_order'])
        joints = np.asarray(columns['observation.state'], dtype=np.float32)[:, :6]
        keys = keypoints(poses, grip, columns['instruction_id'], timestamps=stamps, **data['keypoints'])
        n = len(stamps)
        # Keep complete dense coverage, including no-key episodes and post-key tails.
        frames = sorted(set(range(0, n - 1, sample_stride)) | {key for key in keys if key < n - 1})
        videos = None
        episode_samples = 0
        if progress:
            print(f"[OHT build v4] episode={episode_number}/{len(manifest['episodes'])} "
                  f"task={record['task']} id={record['episode_index']} frames={n} "
                  f"action_dt={options['action_dt']:.9g}", file=sys.stderr, flush=True)
        try:
            for frame in frames:
                targets, reason = make_chunk_targets(poses, joints, grip, stamps, frame, keys, options)
                if targets is None:
                    skipped_windows.append(dict(task=record['task'], episode_index=record['episode_index'],
                                                frame=frame, reason=reason))
                    continue
                _target_bounds(targets, data['scene_bounds'])
                if videos is None:
                    videos = EpisodeVideos(inside(root, record['dataset']),
                                           data.get('video_timestamp_tolerance', 1 / 120 + .0001),
                                           alignment=data['video_alignment'], expected_frames=n)
                observation = dict(low_dim_state=low_dim(measured[frame], binary_state=grip[frame]),
                                   joint_positions=joints[frame].copy(), tcp_pose=poses[frame].astype(np.float32),
                                   gripper_open01=np.asarray(measured[frame], dtype=np.float32))
                for camera in data['cameras']:
                    rgb = videos.read(columns[f'observation.images.{camera}'][frame], frame=frame)
                    depth_config = data['cameras'][camera].get('depth', data['depth'])
                    depth = metric_depth(root, record, columns, camera, frame, videos, depth_config)
                    observation.update(camera_observation(
                        camera, rgb, depth, columns[f'observation.{camera}_extrinsic'][frame], data))
                validate_observation(observation, data)
                sample_id = f"{record['task']}/{record['episode_index']:06d}/{frame:06d}"
                relative_obs, relative_targets = f'observations/{sample_id}.npz', f'targets/{sample_id}.npz'
                for relative, values in ((relative_obs, observation), (relative_targets, targets)):
                    path = inside(output, relative)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(path, **values)
                sample = dict(id=sample_id, task=record['task'], split=record['split'],
                              episode_index=record['episode_index'], episode_length=n, frame=frame,
                              target_frame=int(targets['key_frame']), timestamp=float(stamps[frame]),
                              group=record['group'], data_profile=record['dataset'],
                              current_tcp=poses[frame].tolist(), goal=GOALS[record['task']],
                              observation=relative_obs, observation_sha256=file_digest(output / relative_obs),
                              targets=relative_targets, targets_sha256=file_digest(output / relative_targets),
                              diagnostic_objects=capture_object_poses(columns, frame, record))
                if object_writer is not None:
                    sample['object_clouds'] = object_writer.write(observation, data, sample)
                rows.append(sample)
                episode_samples += 1
                # Existing previews are diagnostics; absent key labels stay absent.
                preview_sample = dict(sample, labels={})
                if bool(targets['key_valid']):
                    preview_sample['labels']['gripper_pose'] = targets['key_action'][:7].tolist()
                preview_kwargs = {} if object_writer is None else dict(object_boxes=object_writer.specification(sample))
                preview_path = preview.write(observation, data, preview_sample,
                                             current_tcp=poses[frame], chunk_targets=targets, **preview_kwargs)
                if object_writer is not None and preview_path is not None:
                    object_writer.write_preview(observation, data, preview_sample, preview_image=preview_path)
                elif preview_path is not None:
                    # Basic V4 previews have a browser entry too; object crops
                    # are optional and never required just to produce HTML.
                    from .html_preview import save_point_cloud_html
                    html_path = preview_path.with_name('point-cloud-roi.html')
                    diagnostic = dict(preview_sample, diagnostic_contract_sha256=None,
                                      diagnostic_note='Build preview; check complete.json for replay completion.')
                    save_point_cloud_html(html_path, observation, data, diagnostic,
                                          source='depth', preview_image=preview_path)
                    html_frames.append(dict(sample_id=sample['id'], frame=sample['frame'],
                                            target_frame=sample['target_frame'], timestamp=sample['timestamp'],
                                            html=html_path, image=preview_path))
                now = perf_counter()
                if progress and now - last_report >= 5:
                    print(f'[OHT build v4] samples={len(rows)} skipped_windows={len(skipped_windows)} '
                          f'elapsed_s={now - started:.1f}', file=sys.stderr, flush=True)
                    last_report = now
            if videos is not None:
                videos.validate_lengths()
        finally:
            if videos is not None:
                videos.close()
        if not episode_samples:
            skipped_episodes.append(dict(task=record['task'], episode_index=record['episode_index'],
                                         reason='no_valid_chunk_windows'))
    if not rows:
        raise ValueError('No valid V4 chunk windows; inspect timing/stride/gripper events; no complete marker written')
    (output / 'samples.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows), encoding='utf-8')
    contract['index_sha256'] = file_digest(output / 'samples.jsonl')
    contract.pop('sha256')
    contract['sha256'] = digest(contract)
    write_json(output / 'contract.json', contract)
    if object_writer is not None:
        object_writer.bind_contract(contract['sha256'])
    if html_frames:
        from .html_preview import bind_point_cloud_html_contract
        from .replay_viewer import save_replay_viewer
        for frame in html_frames:
            bind_point_cloud_html_contract(frame['html'], contract['sha256'])
        save_replay_viewer(preview.output / 'index.html', html_frames, contract['sha256'])
    write_json(output / 'complete.json', dict(
        contract_sha256=contract['sha256'], index_sha256=contract['index_sha256'], samples=len(rows),
        split_counts=dict(Counter(row['split'] for row in rows)), skipped_windows=skipped_windows,
        skipped_window_counts=dict(Counter(row['reason'] for row in skipped_windows)),
        skipped_episodes=skipped_episodes, episode_timing=timing))
    if progress:
        print(f'[OHT build v4] complete samples={len(rows)} skipped_windows={len(skipped_windows)} '
              f'elapsed_s={perf_counter() - started:.1f}', file=sys.stderr, flush=True)
        if html_frames or (object_writer is not None and object_writer.frames):
            print(f"[OHT build v4] preview={preview.output / 'index.html'}", file=sys.stderr, flush=True)
    return len(rows)
