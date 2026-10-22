"""Native client executes GT endpoints with waypoint timing and dense GT unchanged."""
import json

import numpy as np
import pytest

from tests.test_oht_trajectory_replay import trajectory, replay_simulator, command as _base_command  # noqa: F401
from finetune.OHT import isaaclab_client as runner


def command(source, output, *extra):
    # These tests exercise control sequencing without an Isaac Sim GUI backend.
    return _base_command(source, output, '--no-visualize', *extra)


def physical_gripper_simulator(monkeypatch, source, *, step_dt=1/60):
    env, state = replay_simulator(monkeypatch, source, position_scale=.01)
    env.step_dt = step_dt
    original_step = env.step

    def step(value):
        result = original_step(value)
        # The fixture's action layout is gripper first. Reflect the physical
        # calibrated endpoints so arrival and aperture settling use measurements.
        finger_position = -.015 if value[0, 0] > 0 else .011
        env.scene['robot'].data.joint_pos[0, 6:] = finger_position
        return result

    env.step = step
    return env, state


def test_native_keypoint_replay_runs_each_future_goal_once_after_arm_and_gripper_settle(
        trajectory, tmp_path, monkeypatch):
    # Sparse endpoint tracking has variable control time, independent of the
    # original 60 Hz frame cadence; this environment intentionally uses 50 Hz.
    env, state = physical_gripper_simulator(monkeypatch, trajectory, step_dt=.02)
    output = tmp_path/'keypoints'
    assert runner.main(command(trajectory, output, '--replay-targets', 'keypoints', '--log-interval', '0')) == 0
    row = json.loads((output/'episode_0000.json').read_text())
    metadata = json.loads((output/'run.json').read_text())
    summary = json.loads((output/'summary.json').read_text())
    assert env.closed and state.app_closed and len(env.writes) == 1
    assert metadata['execution_mode'] == 'waypoint'
    assert metadata['replay']['source'] == 'observed_future_keypoint_tcp'
    assert metadata['replay']['available_actions'] == 2
    assert metadata['replay']['keypoints']['indices'] == [3, 5]
    assert metadata['replay']['server_queries'] is False and metadata['model_id'] is None
    assert metadata['replay']['source_frame_interval']['min_seconds'] == pytest.approx(1/60)
    assert row['stop_reason'] == 'trajectory_end' and row['steps'] == 2 and row['success'] is None
    assert [item['target_frame'] for item in row['actions']] == [3, 5]
    assert [item['source_frame'] for item in row['actions']] == [0, 3]
    assert summary['episodes'][0]['stop_reason'] == 'trajectory_end'
    assert summary['success_rate'] is None
    assert env.common_step_counter == sum(len(item['commands']) for item in row['actions']) > 2
    assert row['actions'][1]['timestamp'] == pytest.approx(len(row['actions'][0]['commands']) * .02)
    np.testing.assert_allclose(env.writes[0][0, :6], trajectory.joints[0], atol=1e-7)
    np.testing.assert_allclose(env.writes[0][0, 6:], trajectory.fingers[0], atol=1e-7)
    for index, item in enumerate(row['actions']):
        execution = item['execution']
        assert item['action_source'] == 'gt_keypoint' and item['target_schedule'] == 'keypoints'
        assert item['timing_ms'] == {}
        assert execution['status'] == 'completed' and execution['completed'] is True
        assert execution['approach_steps'] >= 3 and execution['gripper_steps'] >= 10
        assert execution['steps'] == len(item['commands'])
        assert item['errors_after']['translation_m'] < 1e-6
        expected_hold, expected_new = (1., -1.) if index == 0 else (-1., 1.)
        commands = np.asarray(item['commands'])
        np.testing.assert_array_equal(commands[:execution['approach_steps'], 0], expected_hold)
        np.testing.assert_array_equal(commands[execution['approach_steps']:, 0], expected_new)
        np.testing.assert_allclose(item['target'][:3], trajectory.poses[[3, 5][index]], atol=1e-7)


def test_keypoint_replay_does_not_advance_to_next_goal_when_waypoint_times_out(
        trajectory, tmp_path, monkeypatch):
    env, state = physical_gripper_simulator(monkeypatch, trajectory)
    output = tmp_path/'timeout'
    # Two steps cannot satisfy the default three consecutive arrival samples.
    assert runner.main(command(trajectory, output, '--replay-targets', 'keypoints',
                               '--waypoint-max-steps', '2', '--log-interval', '0')) == 1
    row = json.loads((output/'episode_0000.json').read_text())
    assert row['stop_reason'] == 'waypoint_timeout' and row['steps'] == 1
    assert len(row['actions']) == 1 and row['actions'][0]['target_frame'] == 3
    assert row['actions'][0]['execution']['status'] == 'timeout'
    assert env.common_step_counter == 2 and len(env.writes) == 1
    np.testing.assert_array_equal(np.asarray(env.commands)[:, 0], 1.)
    assert env.closed and state.app_closed


def test_default_dense_replay_still_steps_once_per_recorded_transition(
        trajectory, tmp_path, monkeypatch):
    env, state = physical_gripper_simulator(monkeypatch, trajectory)
    output = tmp_path/'frames'
    assert runner.main(command(trajectory, output, '--log-interval', '0')) == 0
    row = json.loads((output/'episode_0000.json').read_text())
    metadata = json.loads((output/'run.json').read_text())
    assert metadata['execution_mode'] == 'streaming'
    assert metadata['replay']['source'] == 'observed_next_frame_tcp'
    assert row['stop_reason'] == 'trajectory_end' and row['steps'] == trajectory.n - 1
    assert env.common_step_counter == trajectory.n - 1 and len(env.writes) == 1
    assert [item['target_frame'] for item in row['actions']] == list(range(1, trajectory.n))
    for index, item in enumerate(row['actions']):
        assert item['action_source'] == 'gt_trajectory' and 'execution' not in item
        assert len(item['commands']) == 1
        assert item['timestamp'] == pytest.approx(index/60)
        assert item['target_timestamp'] == pytest.approx((index+1)/60)
    assert env.closed and state.app_closed


def test_dense_replay_still_rejects_repeated_source_actions_before_control(
        trajectory, tmp_path, monkeypatch):
    env, state = physical_gripper_simulator(monkeypatch, trajectory)
    with pytest.raises(ValueError, match='max-substeps 1'):
        runner.main(command(trajectory, tmp_path/'repeated-frames', '--max-substeps', '2'))
    assert not env.writes and not env.commands
    assert env.closed and state.app_closed


@pytest.mark.parametrize('options', [
    ['--mode', 'replay', '--execution-mode', 'waypoint'],
    ['--mode', 'replay', '--replay-targets', 'keypoints', '--execution-mode', 'streaming'],
    ['--mode', 'policy', '--replay-targets', 'keypoints'],
])
def test_incompatible_target_and_execution_modes_fail_before_http_or_environment(
        tmp_path, monkeypatch, options):
    monkeypatch.setattr(runner, 'BridgeVLAClient', lambda *a, **k: pytest.fail('Unexpected HTTP client'))
    monkeypatch.setattr(runner, '_factory', lambda *a, **k: pytest.fail('Unexpected simulator factory'))
    with pytest.raises(ValueError, match='requires streaming|requires waypoint|require --mode replay'):
        runner.main(['--output', str(tmp_path/'invalid'), *options])
