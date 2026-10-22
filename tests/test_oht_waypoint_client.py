"""Default policy runner executes complete waypoints across its actual HTTP bridge."""
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import numpy as np
import pytest

from finetune.OHT import isaaclab_client as runner
from finetune.OHT.action_modes import action_options
from finetune.OHT.runtime.http_client import BridgeVLAClient
from tests.test_oht_isaaclab_http import Opener, raw_sample
from tests.test_oht_migration import replay_fixture  # noqa: F401
from tests.test_oht_native_runtime import environment
from tests.test_oht_standalone_client import fake_simulator, sensors_for
from tests.test_oht_vla_server import fixture  # noqa: F401


def configure_robot(monkeypatch, fixture, replay_fixture, mode='eef', jammed=False):
    tensors, _, raw = raw_sample(replay_fixture.replay, fixture.policy)
    config = fixture.policy.contract.get('source_data_configs', {})
    data = config[raw['data_profile']]['data_config'] if config else fixture.policy.contract['data_config']
    calibration = data['gripper']
    env = environment(mode, raw['task'])
    robot = env.scene['robot']
    env.action_manager.get_term('arm_action').cfg = SimpleNamespace(
        body_name='tool', position_scale=1., orientation_scale=1.)
    env.scene.sensors = sensors_for(tensors, fixture.policy.contract['data_config']['cameras'])
    original_reset, original_step = env.reset, env.step
    physical = []

    def reset():
        original_reset()
        robot.data.joint_pos[0, 6:] = float(calibration['open']) / 2

    def step(command):
        before_pose = robot.data.body_pos_w.clone()
        before_joints = robot.data.joint_pos[:, :6].clone()
        result = original_step(command)
        if jammed:
            robot.data.body_pos_w[:] = before_pose
            robot.data.joint_pos[:, :6] = before_joints
        # Use the training fixture's endpoints (opposite sign to another OHT
        # exporter). A closed command settles against an object at open01=.4.
        endpoint = (calibration['open'] if command[0, 0] > 0 else
                    calibration['close'] - (calibration['close'] - calibration['open']) * .4)
        current = float(robot.data.joint_pos[0, 6:].sum())
        raw_position = current + float(np.clip(endpoint - current, -.006, .006))
        robot.data.joint_pos[0, 6:] = raw_position / 2
        physical.append(dict(position=robot.data.body_pos_w[0, 0].numpy().copy(),
                             gripper_command=float(command[0, 0]), raw_gripper=raw_position))
        return result

    env.reset, env.step = reset, step
    state = fake_simulator(monkeypatch, env)
    return env, raw, calibration, physical, state


def actions_for(mode):
    if mode == 'eef':
        return [np.array([.30, 0, .5, 0, 0, 0, 1, 0], np.float32),
                np.array([.45, 0, .5, 0, 0, 0, 1, 1], np.float32)]
    if mode == 'relative_eef':
        return [np.array([.20, 0, 0, 0, 0, 0, 0], np.float32),
                np.array([.15, 0, 0, 0, 0, 0, 1], np.float32)]
    return [np.array([.20, 0, 0, 0, 0, 0, 0], np.float32),
            np.array([.15 if mode == 'relative_joint' else .35, 0, 0, 0, 0, 0, 1], np.float32)]


def install_policy(fixture, env, calibration, mode):
    fixture.policy.action_config = action_options({'mode': mode})
    original = fixture.policy.act
    predictions, actions = [], actions_for(mode)

    def act(observation, instruction, step):
        original(observation, instruction, step)
        raw_gripper = float(env.scene['robot'].data.joint_pos[0, 6:].sum())
        ratio = (calibration['close'] - raw_gripper) / (calibration['close'] - calibration['open'])
        predictions.append(dict(step=step, simulator_steps=env.common_step_counter,
                                measured_ratio=ratio, input_gripper=int(observation['low_dim_state'][0])))
        return actions[step].copy()

    fixture.policy.act = act
    return predictions


def run_policy(monkeypatch, fixture, raw, output, extra=()):
    with TestClient(fixture.app(visualize_every=0)) as server:
        monkeypatch.setattr(runner, 'BridgeVLAClient',
                            lambda *args: BridgeVLAClient('http://testserver', opener=Opener(server)))
        # Deliberately omit --execution-mode: policy must default to waypoint.
        return runner.main(['--env-id', 'OHT-test', '--ee-link', 'tool', '--task', raw['task'],
                            '--episodes', '1', '--max-steps', '2', '--no-visualize',
                            '--output', str(output), '--headless', *extra])


@pytest.mark.parametrize('mode', ['eef', 'relative_eef', 'joint', 'relative_joint'])
def test_default_policy_waits_for_complete_waypoints_before_next_prediction(
        fixture, replay_fixture, tmp_path, monkeypatch, mode):
    env, raw, calibration, physical, native = configure_robot(monkeypatch, fixture, replay_fixture, mode)
    predictions = install_policy(fixture, env, calibration, mode)
    output = tmp_path / f'waypoint-{mode}'
    extra = ['--joint-max-step', '.04'] if 'joint' in mode else []
    assert run_policy(monkeypatch, fixture, raw, output, extra) == 0
    run = json.loads((output / 'run.json').read_text(encoding='utf-8'))
    episode = json.loads((output / 'episode_0000.json').read_text(encoding='utf-8'))
    rows = episode['actions']
    assert run['execution_mode'] == 'waypoint' and run['waypoint_options']['arrival_steps'] == 3
    assert episode['stop_reason'] == 'max_steps' and len(rows) == len(predictions) == 2
    assert [p['step'] for p in predictions] == [0, 1]
    assert predictions[0]['simulator_steps'] == 0
    assert predictions[1]['simulator_steps'] == rows[0]['execution']['steps'] > 1
    assert predictions[1]['measured_ratio'] == pytest.approx(.4, abs=1e-5)
    assert [p['input_gripper'] for p in predictions] == [1, 0]
    for index, row in enumerate(rows):
        result = row['execution']
        assert result['status'] == 'completed' and result['completed']
        assert result['steps'] == len(row['commands'])
        assert result['gripper_steps'] >= 10
        approach = result['approach_steps']
        old, new = ((1., -1.) if index == 0 else (-1., 1.))
        assert [command[0] for command in row['commands'][:approach]] == [old] * approach
        assert [command[0] for command in row['commands'][approach:]] == [new] * result['gripper_steps']
        assert row['client_timing_ms']['execute'] >= 0 and row['timing_ms']['policy'] >= 0
        assert row['target'] == result['target']
        if 'joint' in mode:
            assert row['errors_after']['joint_max_rad'] <= .01
        else:
            assert row['errors_after']['translation_m'] <= .005
    if mode == 'relative_eef':
        expected = rows[0]['measured_after']['tcp_pose_xyzw'][0] + .15
        assert rows[1]['target'][0] == pytest.approx(expected, abs=1e-6)
        assert rows[1]['measured_after']['tcp_pose_xyzw'][0] == pytest.approx(expected, abs=.005)
    elif mode == 'relative_joint':
        expected = rows[0]['measured_after']['joint_positions'][0] + .15
        assert rows[1]['target'][0] == pytest.approx(expected, abs=1e-6)
        assert rows[1]['measured_after']['joint_positions'][0] == pytest.approx(expected, abs=.01)
    assert len(physical) == env.common_step_counter == sum(row['execution']['steps'] for row in rows)
    assert env.closed and native.app_closed


@pytest.mark.parametrize('failure,flags,expected_steps', [
    ('stalled', ['--waypoint-stall-steps', '4'], 4),
    ('timeout', ['--waypoint-max-steps', '2'], 2),
])
def test_failed_waypoint_saves_partial_commands_and_stops_predictions_and_future_episodes(
        fixture, replay_fixture, tmp_path, monkeypatch, failure, flags, expected_steps):
    env, raw, calibration, _, native = configure_robot(monkeypatch, fixture, replay_fixture, jammed=True)
    predictions = install_policy(fixture, env, calibration, 'eef')
    output = tmp_path / f'failure-{failure}'
    assert run_policy(monkeypatch, fixture, raw, output, ['--episodes', '2', *flags]) == 1
    episode = json.loads((output / 'episode_0000.json').read_text(encoding='utf-8'))
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    assert episode['stop_reason'] == 'waypoint_' + failure
    assert len(episode['actions']) == len(predictions) == len(fixture.policy.calls) == 1
    action = episode['actions'][0]
    assert action['execution']['status'] == failure and not action['execution']['completed']
    assert len(action['commands']) == action['execution']['steps'] == env.common_step_counter == expected_steps
    assert all(command[0] == 1 for command in action['commands'])
    assert 'error' in episode and summary['attempted'] == 1 and summary['unattempted'] == 1
    assert not (output / 'episode_0001.json').exists()
    assert env.closed and native.app_closed
