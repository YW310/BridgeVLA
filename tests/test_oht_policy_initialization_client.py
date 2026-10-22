"""CLI integration: recorded starts still predict and execute live policy actions."""
from copy import deepcopy
import json

import numpy as np
import pytest

from finetune.OHT import isaaclab_client as runner
from tests.test_oht_policy_initialization import client_for
from tests.test_oht_trajectory_replay import trajectory, replay_simulator  # noqa: F401


class LivePolicy:
    def __init__(self, f, env, *, mode='eef'):
        source, self.profile_name = client_for(f, mode=mode)
        self.info, self.action_config = source.info, source.action_config
        self.info.update(model_id='live-policy-model', cameras=[])
        self.task, self.env = f.task, env
        self.resets, self.predictions = [], []

    def profile(self, task, name=None):
        assert task == self.task and name in (None, self.profile_name)
        return self.profile_name

    def reset(self, episode=None):
        self.resets.append(dict(episode=episode, writes=len(self.env.writes),
                                simulator_steps=self.env.common_step_counter))
        return {'reset': True, 'next_step': 0}

    def predict(self, measured, instruction, timestamp, *, task, data_profile):
        # The target depends on the actual live snapshot. It deliberately moves
        # in Y; all recorded future TCP targets in this fixture move only in X.
        snapshot = {name: np.asarray(value).copy() for name, value in measured.items()}
        self.predictions.append(dict(measured=snapshot, instruction=instruction, timestamp=timestamp,
                                     task=task, profile=data_profile, writes=len(self.env.writes)))
        if self.action_config['mode'] == 'relative_eef':
            action = np.array([0., .02, 0., 0., 0., 0., 1.], np.float32)
        else:
            pose = snapshot['ee_link_pose']
            action = np.r_[pose[:3] + [0., .02, 0.], pose[3:][[1, 2, 3, 0]], 1.].astype(np.float32)
        return action, {'timing_ms': {'policy': 1.}}


def command(f, output, *extra):
    return ['--mode', 'policy', '--initial-state-root', str(f.root),
            '--initial-state-data-config', str(f.config_path), '--initial-state-episode', '0',
            '--task', f.task, '--env-id', 'OHT-test', '--execution-mode', 'streaming',
            '--episodes', '1', '--max-steps', '1', '--output', str(output), '--headless', '--no-visualize', *extra]


@pytest.mark.parametrize('mode', ['eef', 'relative_eef'])
@pytest.mark.parametrize('frame', [0, 3, 6])
def test_native_cli_restores_state_once_then_uses_live_model_predictions(trajectory, tmp_path, monkeypatch, mode, frame):
    f = trajectory
    env, app_state = replay_simulator(monkeypatch, f)
    env.step_dt = .02  # Intentionally differs from recorded 1/60: this is initialization only.
    policy = LivePolicy(f, env, mode=mode)
    original_config = deepcopy(policy.action_config)
    monkeypatch.setattr(runner, 'BridgeVLAClient', lambda *args: policy)
    output = tmp_path / 'policy'
    assert runner.main(command(f, output, '--initial-state-frame', str(frame))) == 0
    assert app_state.launched and app_state.app_closed and env.closed
    assert len(env.writes) == len(policy.resets) == len(policy.predictions) == len(env.commands) == 1
    assert policy.resets[0]['writes'] == 1 and policy.resets[0]['simulator_steps'] == 0
    prediction = policy.predictions[0]
    np.testing.assert_allclose(prediction['measured']['joint_positions'], f.joints[frame], atol=1e-7)
    np.testing.assert_allclose(prediction['measured']['ee_link_pose'][:3], f.poses[frame], atol=1e-7)
    assert prediction['measured']['gripper_position'][0] == pytest.approx(f.fingers[frame].sum(), abs=1e-7)
    assert prediction['writes'] == 1 and prediction['timestamp'] == 0
    assert prediction['task'] == f.task and prediction['profile'] == policy.profile_name
    assert policy.action_config == original_config
    metadata = json.loads((output / 'run.json').read_text(encoding='utf-8'))
    episode = json.loads((output / 'episode_0000.json').read_text(encoding='utf-8'))
    assert metadata['mode'] == 'policy' and metadata['initialization'] == 'trajectory'
    assert metadata['model_id'] == 'live-policy-model' and metadata['action_spec']['mode'] == mode
    assert metadata['replay'] is None and not metadata['initial_state']['future_actions_used']
    assert metadata['initial_state']['source_frame'] == frame
    assert episode['initialization']['source_frame'] == frame
    assert episode['initial_alignment']['errors']['translation_m'] < 1e-7
    assert episode['actions'][0]['action_source'] == 'policy'
    assert episode['actions'][0]['timing_ms']['policy'] == 1.
    expected = f.poses[frame] + [0., .02, 0.]
    np.testing.assert_allclose(episode['actions'][0]['target'][:3], expected, atol=1e-7)
    assert episode['stop_reason'] == 'max_steps' and env.common_step_counter == 1


def test_initial_calibration_mismatch_stops_before_simulator_start_and_predictions(trajectory, tmp_path, monkeypatch):
    f = trajectory
    env, app_state = replay_simulator(monkeypatch, f)
    policy = LivePolicy(f, env)
    policy.info['raw_observation']['data_profiles'][policy.profile_name]['gripper']['open'] += .001
    monkeypatch.setattr(runner, 'BridgeVLAClient', lambda *args: policy)
    output = tmp_path / 'mismatch'
    with pytest.raises(ValueError, match=r'gripper\.open'):
        runner.main(command(f, output))
    assert not app_state.launched and not policy.predictions and not policy.resets
    assert not env.writes and not env.commands and not output.exists()


def test_failed_initial_alignment_logs_error_and_cleans_up_without_policy_reset(trajectory, tmp_path, monkeypatch):
    f = trajectory
    env, app_state = replay_simulator(monkeypatch, f)
    policy = LivePolicy(f, env)
    monkeypatch.setattr(runner, 'BridgeVLAClient', lambda *args: policy)
    original_forward = env.sim.forward
    def wrong_forward():
        original_forward()
        env.scene['robot'].data.body_pos_w[0, 0, 0] += .1
    env.sim.forward = wrong_forward
    output = tmp_path / 'alignment-error'
    assert runner.main(command(f, output)) == 1
    assert env.closed and app_state.app_closed and len(env.writes) == 1
    assert not policy.predictions and not policy.resets and not env.commands
    episode = json.loads((output / 'episode_0000.json').read_text(encoding='utf-8'))
    assert episode['stop_reason'] == 'error' and episode['steps'] == 0
    assert 'initial TCP mismatch' in episode['error']


def test_initialization_dry_run_does_not_restore_joints_or_predict(trajectory, tmp_path, monkeypatch):
    f = trajectory
    env, app_state = replay_simulator(monkeypatch, f)
    policy = LivePolicy(f, env)
    monkeypatch.setattr(runner, 'BridgeVLAClient', lambda *args: policy)
    output = tmp_path / 'dry'
    assert runner.main(command(f, output, '--dry-run')) == 0
    assert env.closed and app_state.app_closed and not env.writes and not env.commands
    assert not policy.predictions and not policy.resets
    assert (output / 'control_diagnostics.json').is_file()
    assert not (output / 'episode_0000.json').exists()


@pytest.mark.parametrize('extra', [
    ['--initial-state-frame', '0'],
    ['--initial-state-frame=0'],
    ['--initial-state-episode', '0'],
    ['--initial-state-data-config', 'dataset.yaml'],
    ['--initial-state-position-tolerance', '.005'],
    ['--initialization', 'trajectory'],
])
def test_initial_flags_without_root_fail_before_http_or_simulator(tmp_path, monkeypatch, extra):
    def forbidden(*args):
        pytest.fail('Invalid CLI initialization must fail before contacting a model')
    monkeypatch.setattr(runner, 'BridgeVLAClient', forbidden)
    with pytest.raises(ValueError, match='initial-state'):
        runner.main(['--output', str(tmp_path / 'invalid'), *extra])


@pytest.mark.parametrize('extra', [
    ['--mode', 'replay'],
    ['--initialization', 'env-reset'],
    ['--initialization', 'task'],
    ['--initialization', 'collection'],
    ['--reference-client', 'legacy.py'],
])
def test_initial_root_rejects_wrong_mode_or_initialization_before_http(trajectory, tmp_path, monkeypatch, extra):
    def forbidden(*args):
        pytest.fail('Invalid initialization combination must fail before HTTP startup')
    monkeypatch.setattr(runner, 'BridgeVLAClient', forbidden)
    with pytest.raises(ValueError):
        runner.main(command(trajectory, tmp_path / 'invalid', *extra))
