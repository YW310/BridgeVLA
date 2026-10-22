"""GT training endpoints through the waypoint path, distinct from dense frame replay."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tests.test_oht_trajectory_replay import trajectory, source_args, replay_simulator  # noqa: F401
from finetune.OHT.runtime.keypoint_replay import KeypointReplay
from finetune.OHT.runtime.trajectory_replay import TrajectoryReplay


def _rewrite(fixture, columns=None, keypoint_config=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml
    if columns is not None:
        pq.write_table(pa.table(columns), fixture.data)
    if keypoint_config is not None:
        config = deepcopy(fixture.config)
        config['keypoints'] = keypoint_config
        fixture.config_path.write_text(yaml.safe_dump(config), encoding='utf-8')


def test_gripper_keypoint_replay_uses_training_endpoints_not_next_frame(trajectory):
    source = KeypointReplay(source_args(trajectory))
    dense = TrajectoryReplay(source_args(trajectory))
    assert dense.length == 6 and dense.action(0)[1]['target_frame'] == 1
    assert source.length == 2 and source.keyframe_indices == [3, 5]
    assert source.metadata['source'] == 'observed_future_keypoint_tcp'
    assert source.metadata['server_queries'] is False
    assert source.info['action_spec']['mode'] == 'eef'
    first, first_meta = source.action(0)
    second, second_meta = source.action(1)
    np.testing.assert_allclose(first[:3], trajectory.poses[3], atol=1e-7)
    np.testing.assert_allclose(second[:3], trajectory.poses[5], atol=1e-7)
    assert first[-1] == 0 and second[-1] == 1
    assert (first_meta['source_frame'], first_meta['target_frame']) == (0, 3)
    assert (second_meta['source_frame'], second_meta['target_frame']) == (3, 5)
    assert first_meta['target_timestamp'] == pytest.approx(3/60)
    assert second_meta['source_timestamp'] == pytest.approx(3/60)
    assert first_meta['keypoint_method'] == 'gripper' and first_meta['target_schedule'] == 'keypoints'
    assert source.metadata['source_frame_interval']['min_seconds'] == pytest.approx(1/60)
    first[:] = 123
    np.testing.assert_allclose(source.action(0)[0][:3], trajectory.poses[3], atol=1e-7)
    with pytest.raises(IndexError, match='exhausted'):
        source.action(2)


def test_keypoint_start_on_event_advances_to_strictly_future_event(trajectory):
    source = KeypointReplay(source_args(trajectory, replay_start_frame=3))
    assert source.length == 1 and source.keyframe_indices == [5]
    assert source.metadata['keypoints']['all_indices'] == [3, 5]
    assert source.metadata['keypoints']['indices'] == [5]
    assert source.action(0)[1]['source_frame'] == 3
    assert source.action(0)[1]['target_frame'] == 5


@pytest.mark.parametrize('start', [5, 6])
def test_last_keypoint_or_episode_tail_cannot_invent_a_future_goal(trajectory, start):
    with pytest.raises(ValueError, match='future recorded keypoints|following recorded frame'):
        KeypointReplay(source_args(trajectory, replay_start_frame=start))


def test_gripper_episode_without_events_does_not_get_a_fake_terminal(trajectory):
    columns = deepcopy(trajectory.columns)
    columns['observation.state'] = [list(row[:6]) + [-.03] for row in columns['observation.state']]
    columns['observation.gripper_joints'] = [[-.015, -.015]] * trajectory.n
    _rewrite(trajectory, columns=columns)
    with pytest.raises(ValueError, match='No future recorded keypoints'):
        KeypointReplay(source_args(trajectory))


def test_geometric_mode_retains_real_instruction_boundaries_and_terminal(trajectory):
    columns = deepcopy(trajectory.columns)
    columns['observation.state'] = [list(row[:6]) + [-.03] for row in columns['observation.state']]
    columns['observation.gripper_joints'] = [[-.015, -.015]] * trajectory.n
    columns['instruction_id'] = [0, 0, 0, 0, 1, 1, 1]
    _rewrite(trajectory, columns=columns, keypoint_config=dict(method='geometric', max_translation=10.,
             max_rotation_degrees=180., max_frames=100))
    source = KeypointReplay(source_args(trajectory))
    assert source.keyframe_indices == [3, 4, 6]
    assert source.metadata['keypoints']['method'] == 'geometric'
    np.testing.assert_allclose(source.action(2)[0][:3], trajectory.poses[6], atol=1e-7)


def test_bridgevla_mode_keeps_its_actual_terminal_pruning(trajectory):
    _rewrite(trajectory, keypoint_config=dict(method='bridgevla'))
    source = KeypointReplay(source_args(trajectory))
    assert source.keyframe_indices == [3, 6]
    assert source.metadata['keypoints']['method'] == 'bridgevla'


def test_contract_keypoint_definition_wins_over_local_dataset_default(trajectory, monkeypatch):
    from finetune.OHT.data import replay
    config = deepcopy(trajectory.config)
    config['keypoints'] = dict(method='geometric', max_translation=10., max_rotation_degrees=180., max_frames=100)
    contract = dict(sha256='resolved-training-contract',
                    source_data_configs={f'{trajectory.task}/lerobot_dataset': {'data_config': config}})
    monkeypatch.setattr(replay, 'load_contract', lambda path: contract)
    source = KeypointReplay(source_args(trajectory, replay_contract='existing-buffer', replay_data_config=None))
    assert source.keyframe_indices == [2, 3, 4, 5, 6]
    assert source.info['contract_sha256'] == 'resolved-training-contract'
    assert source.metadata['calibration']['sha256'] == 'resolved-training-contract'


def test_sparse_replay_validates_waypoint_mode_not_collection_frequency(trajectory):
    source = KeypointReplay(source_args(trajectory))
    args = source_args(trajectory, execution_mode='waypoint', max_steps=2, max_substeps=7)
    source.validate_runtime(SimpleNamespace(step_dt=.02), args)
    args.execution_mode = 'dense'
    with pytest.raises(ValueError, match='waypoint execution'):
        source.validate_runtime(SimpleNamespace(step_dt=.02), args)


@pytest.mark.parametrize('max_steps', [0, -1, None, True, float('inf')])
def test_sparse_replay_requires_a_bounded_target_budget(trajectory, max_steps):
    source = KeypointReplay(source_args(trajectory))
    args = source_args(trajectory, execution_mode='waypoint', max_steps=max_steps)
    with pytest.raises(ValueError, match='positive bounded integer'):
        source.validate_runtime(SimpleNamespace(step_dt=1/60), args)


@pytest.mark.parametrize('dt', [0., -.01, float('nan'), float('inf')])
def test_sparse_replay_requires_valid_simulator_dt(trajectory, dt):
    source = KeypointReplay(source_args(trajectory))
    args = source_args(trajectory, execution_mode='waypoint', max_steps=2)
    with pytest.raises(ValueError, match='step_dt'):
        source.validate_runtime(SimpleNamespace(step_dt=dt), args)


def test_keypoint_actions_keep_original_joint_initialization_without_repeated_teleport(
        trajectory, monkeypatch):
    source = KeypointReplay(source_args(trajectory, replay_start_frame=2,
                            replay_joint_names=[f'j{i}' for i in range(6)],
                            replay_gripper_joint_names=['j6', 'j7']))
    env, _ = replay_simulator(monkeypatch, trajectory)
    env.action_manager.get_term('arm_action')._joint_ids = [5, 3, 1, 4, 2, 0]
    bridge = SimpleNamespace(env=env, robot=env.scene['robot'], ref=SimpleNamespace(torch=torch))
    report = source.initialize(bridge)
    assert report['source_frame'] == 2 and report['arm_joint_ids'] == list(range(6))
    assert len(env.writes) == 1
    np.testing.assert_allclose(env.writes[0][0, :6], trajectory.joints[2], atol=1e-7)
    np.testing.assert_allclose(env.writes[0][0, 6:], trajectory.fingers[2], atol=1e-7)
    for step in range(source.length):
        source.action(step)
    assert len(env.writes) == 1 and not env.commands
