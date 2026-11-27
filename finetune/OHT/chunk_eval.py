"""Held-out continuous errors, with independent key-goal metrics and PNG exports."""
import html
from time import perf_counter
import numpy as np
from .data.chunk_dataset import OHTChunkDataset
from .data.common import write_json
from .data.geometry import rotation_error
from .chunk_visualization import path_panel, scene_background
from .runtime.chunk_protocol import chunk_spec, validate_prediction
from .runtime.predicted_wrapper import observation_fields
from .evaluation_statistics import episode_reports, select_indices, summarize_errors


def summarize_motion(rows):
    """Compare forecast errors to holding the measured pose, in physical units."""
    result = summarize_errors(rows)
    for name in ('translation_m', 'rotation_degrees'):
        result.setdefault(name + '_mean', None)
        if rows:
            baseline = float(np.mean([row['stationary_' + name] for row in rows]))
            result['stationary_' + name + '_mean'] = baseline
            result[name + '_skill_vs_stationary'] = (
                1 - result[name + '_mean'] / baseline if baseline > 1e-8 else None)
    if not rows or 'stationary_gripper_correct' in rows[0]:
        held = [row['stationary_gripper_correct'] for row in rows]
        changed = [row['gripper_correct'] for row, correct in zip(rows, held) if not correct]
        result['stationary_gripper_accuracy'] = float(np.mean(held)) if held else None
        result['gripper_change_samples'] = len(changed)
        result['gripper_change_accuracy'] = float(np.mean(changed)) if changed else None
    return result


def open_loop_chunk(policy, replay, split='val', limit=None, *, progress=None, visualizer=None,
                    tasks=None, episode_indices=None, samples_per_episode=None):
    data = OHTChunkDataset(replay, split)
    if policy.contract['sha256'] != data.contract['sha256']:
        raise ValueError('Chunk checkpoint/evaluation replay mismatch')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    indices, selected = select_indices(data.samples, tasks, episode_indices, limit,
                                       samples_per_episode=samples_per_episode)
    total = len(indices)
    if progress is not None:
        progress.update('samples_ready', completed=0, total=total, split=split)
    spec = chunk_spec(policy.chunk_settings, policy.contract)
    fields = observation_fields(data.contract['data_config']['cameras'])
    fields.update(('tcp_pose', 'joint_positions', 'gripper_open01'))
    rows, pictures, last_episode = [], [], None
    for index, data_index in enumerate(indices):
        sample, meta = data[data_index], data.samples[data_index]
        episode = meta['task'], meta['episode_index']
        if episode != last_episode:
            policy.reset()
            last_episode = episode
        captured = []
        due = visualizer is not None and visualizer.due(index)
        observation = {name: sample[name].copy() for name in fields}
        if progress is not None:
            progress.update('chunk_predict', emit=index == 0, completed=index, total=total, sample=meta['id'])
        started = perf_counter()
        prediction = validate_prediction(policy.act(observation, sample['goal'], meta['frame'],
            **({'visualization_callback': captured.append} if due else {})), spec)
        latency = perf_counter() - started
        actions = prediction['actions']
        target, valid = sample['action_chunk'][:len(actions)], sample['action_valid'][:len(actions)]
        current_gripper = sample['low_dim_state'][0]
        offsets = []
        for j in np.flatnonzero(valid):
            offsets.append(dict(offset=int(j + 1), target_frame=int(sample['action_frames'][j]),
                                timestamp=float(sample['action_timestamps'][j]),
                                translation_error_xyz_m=(actions[j, :3] - target[j, :3]).tolist(),
                                translation_m=float(np.linalg.norm(actions[j, :3] - target[j, :3])),
                                rotation_degrees=float(np.rad2deg(rotation_error(actions[j, 3:7], target[j, 3:7]))),
                                stationary_translation_m=float(np.linalg.norm(sample['tcp_pose'][:3] - target[j, :3])),
                                stationary_rotation_degrees=float(np.rad2deg(rotation_error(sample['tcp_pose'][3:7], target[j, 3:7]))),
                                gripper_correct=bool(actions[j, 7] == target[j, 7]),
                                stationary_gripper_correct=bool(current_gripper == target[j, 7])))
        key_metrics = None
        if sample['key_valid']:
            key, reference = prediction['key_action'], sample['key_action']
            key_metrics = dict(translation_m=float(np.linalg.norm(key[:3] - reference[:3])),
                rotation_degrees=float(np.rad2deg(rotation_error(key[3:7], reference[3:7]))),
                stationary_translation_m=float(np.linalg.norm(sample['tcp_pose'][:3] - reference[:3])),
                stationary_rotation_degrees=float(np.rad2deg(rotation_error(sample['tcp_pose'][3:7], reference[3:7]))),
                gripper_correct=bool(key[7] == reference[7]), target_frame=int(sample['key_frame']),
                stationary_gripper_correct=bool(current_gripper == reference[7]),
                translation_error_xyz_m=(key[:3] - reference[:3]).tolist())
        rows.append(dict(id=meta['id'], task=meta['task'], episode_index=meta['episode_index'],
                         frame=meta['frame'], timestamp=meta['timestamp'], latency_seconds=latency,
                         offsets=offsets, key=key_metrics))
        if due:
            folder = visualizer.root / f'sample_{index:06d}'
            folder.mkdir(parents=True, exist_ok=False)
            scene = scene_background(sample)
            path_panel(actions, target, valid, current=sample['tcp_pose'][:3], sample_id=meta['id'],
                       action_dt=spec['action_dt'], **scene).save(folder / 'chunk.png')
            np.savez_compressed(folder / 'chunk.npz', actions=actions, target=target, valid=valid,
                                key_action=prediction['key_action'], key_valid=sample['key_valid'],
                                current=sample['tcp_pose'][:3], sample_id=meta['id'], action_dt=spec['action_dt'], **scene)
            if captured:
                from .runtime.online_visualization import _montage
                for name, stage in captured[0]['stages'].items():
                    _montage(stage, name, False).save(folder / f'{name}.png')
            pictures.append(dict(id=meta['id'], chunk=f'{folder.name}/chunk.png', arrays=f'{folder.name}/chunk.npz'))
        if progress is not None:
            progress.update('chunk_sample_done', emit=index == 0 or index + 1 == total,
                            completed=index + 1, total=total, sample=meta['id'])

    summarize = summarize_motion
    result = dict(kind='open_loop_chunk_errors', split=split, samples=total, chunk_spec=spec,
        chunk_policy=dict(policy.chunk_settings),
        contract_sha256=data.contract['sha256'], per_offset={str(j): summarize([
            item for row in rows for item in row['offsets'] if item['offset'] == j]) for j in range(1, spec['horizon'] + 1)},
        all_actions=summarize([item for row in rows for item in row['offsets']]),
        prefix_motion={str(length): summarize([item for row in rows for item in row['offsets']
                                               if item['offset'] <= length])
                       for length in sorted({1, spec['execute_horizon'], spec['horizon']})},
        key_goals=summarize([row['key'] for row in rows if row['key'] is not None]),
        per_task={task: summarize([item for row in rows if row['task'] == task for item in row['offsets']])
                  for task in sorted({row['task'] for row in rows})},
        latency_seconds_mean=float(np.mean([row['latency_seconds'] for row in rows])), predictions=rows,
        note='Observed future errors do not measure closed-loop task success.')
    result.update(episode_reports(rows, selected, chunk=True, execute_horizon=spec['execute_horizon']))
    if samples_per_episode is not None:
        result['samples_per_episode'] = samples_per_episode
    if pictures:
        write_json(visualizer.root / 'index.json', pictures)
        (visualizer.root / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Chunk predictions</title>' +
            ''.join(f'<p>{html.escape(row["id"])}</p><img width="900" src="{row["chunk"]}">' for row in pictures), encoding='utf-8')
        result['visualization'] = dict(output_dir=str(visualizer.root), samples=len(pictures), index=str(visualizer.root / 'index.html'))
    return result


def flow_consistency_chunk(policy, replay, split='val', limit=None, *, progress=None, tasks=None,
                           episode_indices=None, steps=(1, 2, 4, 8, 16), seeds=8):
    """Average ``flow_consistency_report`` over held-out samples using the deployed encoder/crop path."""
    import torch
    from .flow_diagnostics import flow_consistency_report
    decoder = policy.agent._net_mod.decoder
    if not getattr(decoder, 'is_flow', False):
        raise ValueError('Flow consistency requires a flow_matching chunk checkpoint')
    data = OHTChunkDataset(replay, split)
    if policy.contract['sha256'] != data.contract['sha256']:
        raise ValueError('Chunk checkpoint/evaluation replay mismatch')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    indices, _ = select_indices(data.samples, tasks, episode_indices, limit)
    reports, device = [], policy.agent._device
    for count, data_index in enumerate(indices):
        sample, meta = data[data_index], data.samples[data_index]
        policy.reset()
        context = policy.chunk_decoder_context(sample, sample['goal'])
        target = torch.as_tensor(sample['action_chunk'], device=device, dtype=torch.float32)[None]
        valid = torch.as_tensor(sample['action_valid'], device=device, dtype=torch.bool)[None]
        if not valid[:, :decoder.settings['horizon']].any():
            continue
        reports.append(flow_consistency_report(decoder, context, target, valid, steps=steps, seeds=seeds,
                                               noise_seed=data_index))
        if progress is not None:
            progress.update('flow_consistency_sample_done', completed=count + 1, total=len(indices), sample=meta['id'])
    if not reports:
        raise ValueError('No selected sample has valid future supervision')
    return dict(kind='flow_consistency', split=split, samples=len(reports), steps=list(steps), seeds=seeds,
                flow_steps=decoder.settings['flow_steps'], flow_time_sampling=decoder.settings['flow_time_sampling'],
                flow_inference_seed=decoder.settings['flow_inference_seed'],
                contract_sha256=data.contract['sha256'], report=_mean_tree(reports),
                note='Open-loop errors on observed futures; they do not measure closed-loop task success.')


def _mean_tree(items):
    first = items[0]
    if isinstance(first, dict):
        return {key: _mean_tree([item[key] for item in items]) for key in first}
    values = [item for item in items if item is not None]
    return float(np.mean(values)) if values else None
