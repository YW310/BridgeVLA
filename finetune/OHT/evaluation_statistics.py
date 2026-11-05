"""Finite, JSON-safe error distributions and replay-relative episode coverage."""
from collections import defaultdict

import numpy as np


SCALAR_ERRORS = ('translation_m', 'rotation_degrees', 'joint_mae_radians',
                 'joint_rmse_radians', 'latency_seconds')


def distribution(values):
    """Population std and empirical quantiles, not confidence intervals."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('Error statistics require finite one-dimensional values')
    names = ('mean', 'std', 'rmse', 'min', 'max', 'median', 'p05', 'p25',
             'p75', 'p90', 'p95', 'p99')
    if not values.size:
        return dict(count=0, **dict.fromkeys(names))
    quantiles = np.quantile(values, [.05, .25, .5, .75, .9, .95, .99])
    return dict(count=int(values.size), mean=float(values.mean()), std=float(values.std()),
                rmse=float(np.sqrt(np.mean(values ** 2))), min=float(values.min()),
                max=float(values.max()), **dict(zip(
                    ('p05', 'p25', 'median', 'p75', 'p90', 'p95', 'p99'),
                    map(float, quantiles))))


def summarize_errors(rows):
    """Keep historical mean keys; add distributions, tolerances and signed axes."""
    result = dict(samples=len(rows), statistics={})
    for name in SCALAR_ERRORS:
        if rows and name in rows[0]:
            stats = distribution([row[name] for row in rows])
            result[name + '_mean'] = stats['mean']
            result['statistics'][name] = stats
    grip = [row['gripper_correct'] for row in rows if 'gripper_correct' in row]
    if len(grip) != len(rows):
        raise ValueError('Each evaluated action requires gripper correctness')
    result.update(gripper_accuracy=float(np.mean(grip)) if grip else None,
                  gripper_correct_mean=float(np.mean(grip)) if grip else None,
                  gripper_errors=int(len(grip) - sum(grip)))
    # These are independent empirical coverage rates, not task success rates.
    tolerances = {}
    for name, thresholds in (('translation_m', (.005, .01, .02, .05)),
                             ('rotation_degrees', (1., 5., 10.))):
        if rows and name in rows[0]:
            values = np.asarray([row[name] for row in rows])
            tolerances[name] = {str(threshold): float(np.mean(values <= threshold))
                                for threshold in thresholds}
    result['within_tolerance'] = tolerances
    for field, labels in (('translation_error_xyz_m', ('x', 'y', 'z')),
                          ('joint_errors_radians', tuple(f'joint_{j}' for j in range(6)))):
        if rows and field in rows[0]:
            errors = np.asarray([row[field] for row in rows], dtype=np.float64)
            if errors.shape != (len(rows), len(labels)) or not np.isfinite(errors).all():
                raise ValueError(f'Invalid per-axis errors: {field}')
            result[field] = {name: dict(bias=float(errors[:, j].mean()),
                mae=float(np.abs(errors[:, j]).mean()), **distribution(errors[:, j]))
                for j, name in enumerate(labels)}
    return result


def select_indices(samples, tasks=None, episode_indices=None, limit=None):
    """Filter the split before applying limit; retain the full selection for coverage."""
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError('limit must be positive')
    if episode_indices is not None and any(type(index) is not int or index < 0 for index in episode_indices):
        raise ValueError('episode indices must be nonnegative integers')
    tasks = set(tasks) if tasks is not None else None
    episodes = set(episode_indices) if episode_indices is not None else None
    selected = [index for index, sample in enumerate(samples)
                if (tasks is None or sample['task'] in tasks)
                and (episodes is None or sample['episode_index'] in episodes)]
    if not selected:
        raise ValueError('No replay samples match the selected tasks/episodes in this split')
    if tasks is not None and tasks - {samples[index]['task'] for index in selected}:
        raise ValueError('Some selected tasks have no matching episodes in this split')
    if episodes is not None and episodes - {samples[index]['episode_index'] for index in selected}:
        raise ValueError('Some selected episode indices are missing in this split')
    return selected if limit is None else selected[:limit], [samples[index] for index in selected]


def timeline_actions(rows):
    """One forecast per GT target frame: latest observation, never best error.

    Input rows must belong to one episode. This is a teacher-forced forecast
    timeline, not an executed trajectory; missing target frames remain missing.
    """
    chosen = {}
    for row in rows:
        for item in row['offsets']:
            frame = item['target_frame']
            candidate = dict(item, source_id=row['id'], source_frame=row['frame'])
            previous = chosen.get(frame)
            if previous is None or candidate['source_frame'] > previous['source_frame']:
                chosen[frame] = candidate
    return [chosen[frame] for frame in sorted(chosen)]


def episode_reports(rows, selected_metadata, *, chunk=False, execute_horizon=None):
    if chunk and (type(execute_horizon) is not int or execute_horizon < 1):
        raise ValueError('Chunk episode statistics require a positive execute_horizon')
    expected, observed = defaultdict(list), defaultdict(list)
    for meta in selected_metadata:
        expected[meta['task'], meta['episode_index']].append(meta)
    for row in rows:
        observed[row['task'], row['episode_index']].append(row)
    if observed.keys() - expected.keys():
        raise ValueError('Evaluated episodes must belong to the selected replay population')
    reports, unique_actions = {}, []
    for (task, index), metadata in sorted(expected.items()):
        values = observed[task, index]
        expected_ids, actual_ids = {meta['id'] for meta in metadata}, {row['id'] for row in values}
        if len(expected_ids) != len(metadata) or len(actual_ids) != len(values) or not actual_ids <= expected_ids:
            raise ValueError('Episode statistics require unique replay sample identities')
        frames = [row['frame'] for row in values]
        report = dict(task=task, episode_index=index, expected_samples=len(metadata),
                      evaluated_samples=len(values), complete=actual_ids == expected_ids,
                      coverage_fraction=len(values) / len(metadata),
                      evaluated_frame_range=[min(frames), max(frames)] if frames else None,
                      replay_frame_range=[min(meta['frame'] for meta in metadata),
                                          max(meta['frame'] for meta in metadata)],
                      source_episode_frames=metadata[0].get('episode_length'))
        report['latency_seconds'] = distribution([row['latency_seconds'] for row in values
                                                  if 'latency_seconds' in row])
        if chunk:
            actions = [item for row in values for item in row['offsets']]
            timeline = timeline_actions(values)
            unique_actions.extend(timeline)
            report.update(all_actions=summarize_errors(actions), timeline_actions=summarize_errors(timeline),
                          key_goals=summarize_errors([row['key'] for row in values if row['key'] is not None]),
                          per_offset={str(j): summarize_errors([item for item in actions if item['offset'] == j])
                                      for j in sorted({item['offset'] for item in actions})},
                          executed_prefix_forecasts=summarize_errors([
                              item for item in actions if item['offset'] <= execute_horizon]),
                          timeline_predictions=timeline)
        else:
            report['errors'] = summarize_errors(values)
        reports[f'{task}/{index:06d}'] = report
    complete = [row for row in reports.values() if row['complete']]
    macro = dict(episodes=len(complete), weighting='equal weight per complete replay episode', statistics={})
    for name in (*SCALAR_ERRORS, 'gripper_accuracy'):
        summaries = [row['timeline_actions'] if chunk else row['errors'] for row in complete]
        key = name if name == 'gripper_accuracy' else name + '_mean'
        if name == 'latency_seconds':
            means = [row['latency_seconds']['mean'] for row in complete
                     if row['latency_seconds']['mean'] is not None]
        else:
            means = [summary[key] for summary in summaries if summary.get(key) is not None]
        if means:
            macro['statistics'][name] = distribution(means)
    coverage = dict(selected_episodes=len(reports), evaluated_episodes=len(observed),
                    complete_episodes=len(complete), expected_samples=len(selected_metadata),
                    evaluated_samples=len(rows), all_selected_episodes_complete=len(complete) == len(reports),
                    definition='Complete means all indexed replay observations in the selected split; '
                               'not all raw sensor frames or an executed open-loop rollout.')
    result = dict(per_episode=reports, episode_coverage=coverage, episode_macro=macro)
    if chunk:
        result['timeline_actions'] = summarize_errors(unique_actions)
        result['timeline_rule'] = ('Per episode/target_frame, use the prediction from the latest recorded '
            'observation; no best-error selection or averaging. all_actions/per_offset retain '
            'overlapping forecasts. Invalid padded offsets are excluded. '
            'executed_prefix_forecasts are predictions, not executed actions.')
    return result
