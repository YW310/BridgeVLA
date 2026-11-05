"""Held-out continuous errors, with independent key-goal metrics and PNG exports."""
import html
from time import perf_counter
import numpy as np
from .data.chunk_dataset import OHTChunkDataset
from .data.common import write_json
from .data.geometry import rotation_error
from .chunk_visualization import path_panel, scene_background
from .runtime.chunk_protocol import chunk_spec, validate_prediction
from .evaluation_statistics import episode_reports, select_indices, summarize_errors


def open_loop_chunk(policy, replay, split='val', limit=None, *, progress=None, visualizer=None,
                    tasks=None, episode_indices=None):
    data = OHTChunkDataset(replay, split)
    if policy.contract['sha256'] != data.contract['sha256']:
        raise ValueError('Chunk checkpoint/evaluation replay mismatch')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    indices, selected = select_indices(data.samples, tasks, episode_indices, limit)
    total = len(indices)
    spec = chunk_spec(policy.chunk_settings, policy.contract)
    rows, pictures, last_episode = [], [], None
    for index, data_index in enumerate(indices):
        sample, meta = data[data_index], data.samples[data_index]
        episode = meta['task'], meta['episode_index']
        if episode != last_episode:
            policy.reset()
            last_episode = episode
        captured = []
        due = visualizer is not None and visualizer.due(index)
        started = perf_counter()
        prediction = validate_prediction(policy.act(sample, sample['goal'], meta['frame'],
            **({'visualization_callback': captured.append} if due else {})), spec)
        latency = perf_counter() - started
        actions = prediction['actions']
        target, valid = sample['action_chunk'][:len(actions)], sample['action_valid'][:len(actions)]
        offsets = []
        for j in np.flatnonzero(valid):
            offsets.append(dict(offset=int(j + 1), target_frame=int(sample['action_frames'][j]),
                                timestamp=float(sample['action_timestamps'][j]),
                                translation_error_xyz_m=(actions[j, :3] - target[j, :3]).tolist(),
                                translation_m=float(np.linalg.norm(actions[j, :3] - target[j, :3])),
                                rotation_degrees=float(np.rad2deg(rotation_error(actions[j, 3:7], target[j, 3:7]))),
                                gripper_correct=bool(actions[j, 7] == target[j, 7])))
        key_metrics = None
        if sample['key_valid']:
            key, reference = prediction['key_action'], sample['key_action']
            key_metrics = dict(translation_m=float(np.linalg.norm(key[:3] - reference[:3])),
                rotation_degrees=float(np.rad2deg(rotation_error(key[3:7], reference[3:7]))),
                gripper_correct=bool(key[7] == reference[7]), target_frame=int(sample['key_frame']),
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
            progress.update('chunk_sample_done', completed=index + 1, total=total, sample=meta['id'])

    def summarize(values):
        result = summarize_errors(values)
        for name in ('translation_m', 'rotation_degrees'):
            result.setdefault(name + '_mean', None)
        return result
    result = dict(kind='open_loop_chunk_errors', split=split, samples=total, chunk_spec=spec,
        chunk_policy=dict(policy.chunk_settings),
        contract_sha256=data.contract['sha256'], per_offset={str(j): summarize([
            item for row in rows for item in row['offsets'] if item['offset'] == j]) for j in range(1, spec['horizon'] + 1)},
        all_actions=summarize([item for row in rows for item in row['offsets']]),
        key_goals=summarize([row['key'] for row in rows if row['key'] is not None]),
        per_task={task: summarize([item for row in rows if row['task'] == task for item in row['offsets']])
                  for task in sorted({row['task'] for row in rows})},
        latency_seconds_mean=float(np.mean([row['latency_seconds'] for row in rows])), predictions=rows,
        note='Observed future errors do not measure closed-loop task success.')
    result.update(episode_reports(rows, selected, chunk=True, execute_horizon=spec['execute_horizon']))
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
                contract_sha256=data.contract['sha256'], report=_mean_tree(reports),
                note='Open-loop errors on observed futures; they do not measure closed-loop task success.')


def _mean_tree(items):
    first = items[0]
    if isinstance(first, dict):
        return {key: _mean_tree([item[key] for item in items]) for key in first}
    values = [item for item in items if item is not None]
    return float(np.mean(values)) if values else None
