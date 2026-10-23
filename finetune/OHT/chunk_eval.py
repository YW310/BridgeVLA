"""Held-out continuous errors, with independent key-goal metrics and PNG exports."""
import html
from time import perf_counter
import numpy as np
from .data.chunk_dataset import OHTChunkDataset
from .data.common import write_json
from .data.geometry import rotation_error
from .chunk_visualization import path_panel
from .runtime.chunk_protocol import chunk_spec, validate_prediction


def open_loop_chunk(policy, replay, split='val', limit=None, *, progress=None, visualizer=None):
    data = OHTChunkDataset(replay, split)
    if policy.contract['sha256'] != data.contract['sha256']:
        raise ValueError('Chunk checkpoint/evaluation replay mismatch')
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive')
    total = len(data) if limit is None else min(limit, len(data))
    spec = chunk_spec(policy.chunk_settings, policy.contract)
    rows, pictures, last_episode = [], [], None
    for index in range(total):
        sample, meta = data[index], data.samples[index]
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
            offsets.append(dict(offset=int(j + 1), translation_m=float(np.linalg.norm(actions[j, :3] - target[j, :3])),
                                rotation_degrees=float(np.rad2deg(rotation_error(actions[j, 3:7], target[j, 3:7]))),
                                gripper_correct=bool(actions[j, 7] == target[j, 7])))
        key_metrics = None
        if sample['key_valid']:
            key, reference = prediction['key_action'], sample['key_action']
            key_metrics = dict(translation_m=float(np.linalg.norm(key[:3] - reference[:3])),
                rotation_degrees=float(np.rad2deg(rotation_error(key[3:7], reference[3:7]))),
                gripper_correct=bool(key[7] == reference[7]))
        rows.append(dict(id=meta['id'], task=meta['task'], latency_seconds=latency,
                         offsets=offsets, key=key_metrics))
        if due:
            folder = visualizer.root / f'sample_{index:06d}'
            folder.mkdir(parents=True, exist_ok=False)
            path_panel(actions, target, valid).save(folder / 'chunk.png')
            np.savez_compressed(folder / 'chunk.npz', actions=actions, target=target, valid=valid,
                                key_action=prediction['key_action'], key_valid=sample['key_valid'])
            if captured:
                from .runtime.online_visualization import _montage
                for name, stage in captured[0]['stages'].items():
                    _montage(stage, name, False).save(folder / f'{name}.png')
            pictures.append(dict(id=meta['id'], chunk=f'{folder.name}/chunk.png', arrays=f'{folder.name}/chunk.npz'))
        if progress is not None:
            progress.update('chunk_sample_done', completed=index + 1, total=total, sample=meta['id'])

    def summarize(values):
        return dict(samples=len(values), **{name + '_mean': float(np.mean([row[name] for row in values]))
                    if values else None for name in ('translation_m', 'rotation_degrees', 'gripper_correct')})
    result = dict(kind='open_loop_chunk_errors', split=split, samples=total, chunk_spec=spec,
        contract_sha256=data.contract['sha256'], per_offset={str(j): summarize([
            item for row in rows for item in row['offsets'] if item['offset'] == j]) for j in range(1, spec['horizon'] + 1)},
        all_actions=summarize([item for row in rows for item in row['offsets']]),
        key_goals=summarize([row['key'] for row in rows if row['key'] is not None]),
        per_task={task: summarize([item for row in rows if row['task'] == task for item in row['offsets']])
                  for task in sorted({row['task'] for row in rows})},
        latency_seconds_mean=float(np.mean([row['latency_seconds'] for row in rows])), predictions=rows,
        note='Observed future errors do not measure closed-loop task success.')
    if pictures:
        write_json(visualizer.root / 'index.json', pictures)
        (visualizer.root / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Chunk predictions</title>' +
            ''.join(f'<p>{html.escape(row["id"])}</p><img width="900" src="{row["chunk"]}">' for row in pictures), encoding='utf-8')
        result['visualization'] = dict(output_dir=str(visualizer.root), samples=len(pictures), index=str(visualizer.root / 'index.html'))
    return result
