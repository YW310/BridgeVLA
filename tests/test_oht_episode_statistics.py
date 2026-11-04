"""Known-error CPU policies verify complete-episode and overlapping-chunk reports."""
import json
import numpy as np
import pytest

from finetune.OHT import eval as evaluation
from finetune.OHT import chunk_eval
from finetune.OHT.evaluation_statistics import (
    distribution, episode_reports, select_indices, summarize_errors, timeline_actions,
)


def test_distribution_reports_range_population_std_and_empirical_quantiles():
    values = np.array([0., 1., 2., 5.])
    stats = distribution(values)
    assert stats['count'] == 4 and stats['min'] == 0 and stats['max'] == 5
    assert stats['mean'] == 2 and stats['median'] == 1.5
    assert stats['std'] == pytest.approx(values.std(ddof=0))
    assert stats['rmse'] == pytest.approx(np.sqrt(np.mean(values ** 2)))
    assert stats['p95'] == pytest.approx(np.quantile(values, .95))
    assert distribution([])['mean'] is None
    assert distribution([2.])['std'] == 0
    json.dumps(distribution([]), allow_nan=False)


@pytest.mark.parametrize('values', [[np.nan], [np.inf], [[1.]]])
def test_nonfinite_or_nested_statistics_rejected(values):
    with pytest.raises(ValueError, match='finite one-dimensional'):
        distribution(values)


def test_signed_axes_tolerance_and_gripper_statistics():
    rows = [dict(translation_m=.01, rotation_degrees=5., gripper_correct=True,
                 translation_error_xyz_m=[.01, 0., 0.]),
            dict(translation_m=.03, rotation_degrees=15., gripper_correct=False,
                 translation_error_xyz_m=[-.03, 0., 0.])]
    result = summarize_errors(rows)
    assert result['translation_m_mean'] == .02
    assert result['statistics']['translation_m']['max'] == .03
    assert result['translation_error_xyz_m']['x']['bias'] == pytest.approx(-.01)
    assert result['translation_error_xyz_m']['x']['mae'] == .02
    assert result['within_tolerance']['translation_m']['0.01'] == .5
    assert result['gripper_accuracy'] == .5 and result['gripper_errors'] == 1


def metadata():
    return [dict(id=f'{task}/{episode:06d}/{frame:06d}', task=task, episode_index=episode,
                 frame=frame, target_frame=frame + 1, timestamp=frame * .1, episode_length=5)
            for task in ('assemble_left', 'assemble_right') for episode in (2, 7) for frame in (0, 1, 2)]


def test_selection_uses_source_episode_and_limit_keeps_full_expected_population():
    indices, selected = select_indices(metadata(), ['assemble_right'], [7], 1)
    assert indices == [9]
    assert len(selected) == 3
    assert {row['episode_index'] for row in selected} == {7}
    with pytest.raises(ValueError, match='No replay samples'):
        select_indices(metadata(), episode_indices=[999])
    with pytest.raises(ValueError, match='missing'):
        select_indices(metadata(), episode_indices=[2, 999])
    with pytest.raises(ValueError, match='nonnegative'):
        select_indices(metadata(), episode_indices=[-1])


def test_episode_complete_flags_and_equal_episode_macro_not_frame_weighted():
    selected = metadata()[:3] + metadata()[3:4]
    rows = [dict(meta, translation_m=.01 if meta['episode_index'] == 2 else .09,
                 gripper_correct=True) for meta in selected]
    result = episode_reports(rows, selected)
    assert result['episode_coverage']['all_selected_episodes_complete']
    assert result['episode_macro']['statistics']['translation_m']['mean'] == pytest.approx(.05)
    assert summarize_errors(rows)['translation_m_mean'] == pytest.approx(.03)
    partial = episode_reports(rows[:1], selected)
    assert partial['episode_coverage']['complete_episodes'] == 0
    assert partial['per_episode']['assemble_left/000002']['coverage_fraction'] == pytest.approx(1 / 3)
    assert partial['per_episode']['assemble_left/000007']['evaluated_samples'] == 0
    assert partial['episode_macro']['episodes'] == 0
    json.dumps(partial, allow_nan=False)


def test_timeline_selects_latest_source_not_smallest_error():
    rows = [dict(id='old', frame=0, offsets=[dict(target_frame=2, offset=2, translation_m=.001)]),
            dict(id='new', frame=1, offsets=[dict(target_frame=2, offset=1, translation_m=.1)])]
    result = timeline_actions(rows)
    assert len(result) == 1 and result[0]['source_id'] == 'new'
    assert result[0]['translation_m'] == .1


class FakeDataset:
    def __init__(self, *args, **kwargs):
        self.samples = metadata()
        self.contract = dict(sha256='fixture', chunk=dict(action_dt=.1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        meta = self.samples[index]
        frame = meta['frame']
        target = np.array([.01 * (frame + 1), 0., .5, 0., 0., 0., 1., 1.], dtype=np.float32)
        actions = np.tile(target, (3, 1))
        actions[:, 0] = .01 * (frame + np.arange(1, 4))
        valid = frame + np.arange(1, 4) < meta['episode_length']
        frames = frame + np.arange(1, 4)
        return dict(goal='assemble wheel', tcp_pose=target.copy(), action=target,
                    action_chunk=actions, action_valid=valid, action_frames=frames,
                    action_timestamps=frames * .1, key_valid=np.asarray(frame != 2),
                    key_action=target.copy(), key_frame=np.asarray(frame + 1))


class FakePolicy:
    contract = dict(sha256='fixture', chunk=dict(action_dt=.1))
    action_config = None

    def __init__(self, chunk=False):
        self.chunk_settings = dict(horizon=3, execute_horizon=2) if chunk else None
        self.resets = 0

    def reset(self):
        self.resets += 1

    def act(self, sample, goal, frame):
        if self.chunk_settings is None:
            prediction = sample['action'].copy()
            prediction[0] += .01
            return prediction
        actions = sample['action_chunk'].copy()
        actions[:, 0] += np.array([.003, .02, .04])
        # Padding deliberately has a large error and must never enter reports.
        actions[~sample['action_valid'], 0] += 100
        return dict(actions=actions, key_action=sample['key_action'].copy())


def test_single_action_full_episode_and_partial_coverage(monkeypatch):
    monkeypatch.setattr(evaluation, 'OHTDataset', FakeDataset)
    policy = FakePolicy()
    result = evaluation.open_loop(policy, 'fixture', tasks=['assemble_left'], episode_indices=[7])
    assert len(result['predictions']) == 3 and policy.resets == 1
    episode = result['per_episode']['assemble_left/000007']
    assert episode['complete'] and episode['evaluated_frame_range'] == [0, 2]
    assert episode['errors']['statistics']['translation_m']['mean'] == pytest.approx(.01)
    assert episode['errors']['statistics']['rotation_degrees']['max'] == 0
    assert result['episode_coverage']['selected_episodes'] == 1
    assert 'statistics' in result['all_samples']
    partial = evaluation.open_loop(policy, 'fixture', limit=1, tasks=['assemble_left'], episode_indices=[7])
    assert not partial['episode_coverage']['all_selected_episodes_complete']
    assert partial['per_episode']['assemble_left/000007']['expected_samples'] == 3


def test_chunk_episode_excludes_padding_and_separates_overlapping_forecasts(monkeypatch):
    monkeypatch.setattr(chunk_eval, 'OHTChunkDataset', FakeDataset)
    result = evaluation.open_loop(FakePolicy(chunk=True), 'fixture', tasks=['assemble_left'], episode_indices=[2])
    episode = result['per_episode']['assemble_left/000002']
    assert episode['complete']
    assert episode['all_actions']['samples'] == 8
    assert episode['timeline_actions']['samples'] == 4
    assert episode['executed_prefix_forecasts']['samples'] == 6
    assert episode['key_goals']['samples'] == 2
    assert result['per_offset']['3']['samples'] == 2
    assert result['all_actions']['statistics']['translation_m']['max'] == pytest.approx(.04)
    # Target frame 4 has no offset-1 forecast in replay; retain its offset-2 forecast.
    assert [row['offset'] for row in episode['timeline_predictions']] == [1, 1, 1, 2]
    assert result['timeline_actions']['translation_m_mean'] == pytest.approx((.003 * 3 + .02) / 4)
    assert len(result['per_episode']) == 1
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('mode', ['joint', 'relative_joint', 'relative_eef'])
def test_vector_modes_keep_physical_units_wrapping_and_world_xyz(monkeypatch, mode):
    class VectorDataset(FakeDataset):
        def __getitem__(self, index):
            sample = super().__getitem__(index)
            sample['action'] = np.array([0., 0., 0., 0., 0., 0., 1.])
            sample['tcp_pose'] = np.array([1., 2., 3., 0., 0., np.sqrt(.5), np.sqrt(.5)])
            return sample

    class VectorPolicy(FakePolicy):
        action_config = dict(mode=mode, joint_wrap=[0], translation_frame='body')

        def act(self, sample, goal, frame):
            prediction = sample['action'].copy()
            prediction[0] += .02 if mode == 'relative_eef' else 2 * np.pi + .02
            prediction[-1] = 0.
            if mode == 'relative_eef':
                prediction[5] = np.deg2rad(5.)
            else:
                prediction[1] = .04
            return prediction

    monkeypatch.setattr(evaluation, 'OHTDataset', VectorDataset)
    result = evaluation.open_loop(VectorPolicy(), 'fixture', tasks=['assemble_left'], episode_indices=[2])
    errors = result['per_episode']['assemble_left/000002']['errors']
    assert errors['gripper_accuracy'] == 0 and errors['gripper_errors'] == 3
    if mode == 'relative_eef':
        assert errors['translation_m_mean'] == pytest.approx(.02)
        assert errors['rotation_degrees_mean'] == pytest.approx(5.)
        assert errors['translation_error_xyz_m']['y']['bias'] == pytest.approx(.02)
        assert errors['translation_error_xyz_m']['x']['mae'] == pytest.approx(0., abs=1e-12)
    else:
        assert errors['joint_errors_radians']['joint_0']['bias'] == pytest.approx(.02)
        assert errors['joint_errors_radians']['joint_1']['mae'] == .04
        assert errors['joint_mae_radians_mean'] == pytest.approx(.01)
        assert 'translation_m_mean' not in errors


def test_cli_filters_episode_and_writes_detailed_json_not_full_stdout(tmp_path, monkeypatch, capsys):
    from finetune.OHT.runtime import loading
    monkeypatch.setattr(evaluation, 'OHTDataset', FakeDataset)
    monkeypatch.setattr(loading, 'load_policy', lambda *args, **kwargs: FakePolicy())
    checkpoint, output = tmp_path / 'fixture.pth', tmp_path / 'report.json'
    checkpoint.write_bytes(b'not a real checkpoint')
    assert evaluation.main(['open-loop', '--checkpoint', str(checkpoint), '--replay', 'fixture',
        '--task', 'assemble_right', '--episode-index', '7', '--visualize-every', '0', '--quiet',
        '--output', str(output)]) == 0
    result = json.loads(output.read_text(encoding='utf-8'))
    assert result['episode_coverage']['all_selected_episodes_complete']
    assert set(result['per_episode']) == {'assemble_right/000007'}
    assert result['checkpoint_sha256']
    stdout = json.loads(capsys.readouterr().out)
    assert 'per_episode' not in stdout and stdout['all_samples']['samples'] == 3


def test_cli_rejects_negative_episode_before_loading(tmp_path):
    with pytest.raises(ValueError, match='--episode-index'):
        evaluation.main(['open-loop', '--checkpoint', 'unused.pth', '--replay', 'unused',
                         '--episode-index', '-1', '--output', str(tmp_path / 'report.json')])
