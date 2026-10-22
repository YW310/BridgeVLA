"""Recorded state initialization does not replace live policy observations/actions."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from finetune.OHT.action_modes import action_descriptor, action_options
from finetune.OHT.runtime.isaaclab_bridge import IsaacLabBridge
from finetune.OHT.runtime.isaaclab_runtime import IsaacLabRuntime
from finetune.OHT.runtime import trajectory_initialization as initial
from tests.test_oht_trajectory_replay import trajectory, replay_simulator  # noqa: F401


def arguments(f, **changes):
    values = dict(initial_state_root=str(f.root), initial_state_episode=0, initial_state_frame=0,
                  initial_state_data_config=str(f.config_path), initial_state_contract=None,
                  initial_state_object_map=None, initial_state_joint_names=None,
                  initial_state_gripper_joint_names=None, initial_state_position_tolerance=.005,
                  initial_state_rotation_tolerance=.05)
    return SimpleNamespace(**{**values, **changes})


def client_for(f, *, profile=None, mode='eef', joint_names=None):
    profile = profile or f'{f.task}/lerobot_dataset'
    calibration = {name: deepcopy(f.config[name]) for name in
                   ('link_to_tcp', 'gripper', 'scene_bounds', 'image_size')}
    calibration['task'] = f.task
    options = action_options({'mode': mode, 'joint_names': joint_names})
    def forbidden(*args, **kwargs):
        pytest.fail('State initialization must not contact the model or execute recorded actions')
    client = SimpleNamespace(info={'raw_observation': {'data_profiles': {profile: calibration}},
                                   'contract_sha256': 'a' * 64, 'action_spec': action_descriptor(options)},
                             action_config=options, predict=forbidden, reset=forbidden)
    return client, profile


def bridge_for(f, monkeypatch, client, profile):
    env, _ = replay_simulator(monkeypatch, f)
    runtime = IsaacLabRuntime(env, {}, ee_link='tool', quaternion_order='wxyz', simulation_app=None,
                              owns_app=False, ee_pose_source='controller')
    bridge = IsaacLabBridge(runtime, env, client, f.task, profile, {}, joint_executor=lambda *a, **k: [])
    return bridge


@pytest.mark.parametrize('frame', [0, 3, 6])
def test_initializes_one_recorded_state_including_last_frame_without_playing_actions(trajectory, monkeypatch, frame):
    f = trajectory
    client, profile = client_for(f)
    source_class = initial.TrajectoryReplay
    class StateOnlySource(source_class):
        def action(self, *args):
            pytest.fail('Recorded future actions must not be used for policy initialization')
        def validate_runtime(self, *args):
            pytest.fail('Initialization must not impose full-rate trajectory timing')
    monkeypatch.setattr(initial, 'TrajectoryReplay', StateOnlySource)
    initializer = initial.InitialState(arguments(f, initial_state_frame=frame), client, f.task, profile)
    bridge = bridge_for(f, monkeypatch, client, profile)
    bridge.env.step_dt = .123  # Policy control rate is independent of the stored trajectory.
    report = initializer.initialize(bridge)
    check = initializer.check_initial(bridge)
    assert report['source_frame'] == frame and report['object_state'] == 'environment_reset'
    assert len(bridge.env.writes) == 1 and not bridge.env.commands and bridge.env.common_step_counter == 0
    np.testing.assert_allclose(bridge.robot.data.joint_pos[0, :6], f.joints[frame], atol=1e-7)
    np.testing.assert_allclose(bridge.robot.data.joint_pos[0, 6:], f.fingers[frame], atol=1e-7)
    assert check['errors']['translation_m'] < 1e-7 and check['errors']['rotation_rad'] < 1e-7
    assert initializer.metadata['source'] == 'recorded_initial_state'
    assert initializer.metadata['source_frame'] == frame
    assert not initializer.metadata['server_queries'] and not initializer.metadata['future_actions_used']
    assert 'available_actions' not in initializer.metadata and 'action_config' not in initializer.metadata


@pytest.mark.parametrize('mode', ['eef', 'relative_eef', 'joint', 'relative_joint'])
def test_initialization_preserves_policy_action_configuration_and_session(trajectory, monkeypatch, mode):
    f = trajectory
    client, profile = client_for(f, mode=mode)
    original = deepcopy(client.action_config)
    initializer = initial.InitialState(arguments(f, initial_state_frame=2), client, f.task, profile)
    bridge = bridge_for(f, monkeypatch, client, profile)
    initializer.initialize(bridge)
    initializer.check_initial(bridge)
    assert client.action_config == original and client.info['action_spec']['mode'] == mode
    assert not bridge.env.commands


def test_default_model_profile_is_validated_by_calibration_not_raw_dataset_name(trajectory):
    f = trajectory
    client, profile = client_for(f, profile='default')
    client.info['raw_observation']['data_profiles'][profile]['task'] = None
    initializer = initial.InitialState(arguments(f), client, f.task, profile)
    assert initializer.metadata['data_profile'] == 'default'
    assert initializer.metadata['recorded_data_profile'] == f'{f.task}/lerobot_dataset'


@pytest.mark.parametrize('field', ['link_to_tcp', 'gripper.open', 'gripper.close', 'scene_bounds', 'image_size'])
def test_calibration_mismatch_is_rejected_before_any_simulator_state_write(trajectory, field):
    f = trajectory
    client, profile = client_for(f)
    calibration = client.info['raw_observation']['data_profiles'][profile]
    if field.startswith('gripper.'):
        calibration['gripper'][field.split('.')[1]] += .001
    elif field == 'link_to_tcp':
        calibration[field][0][3] += .001
    else:
        calibration[field][0] += 1
    with pytest.raises(ValueError, match=field.replace('.', r'\.')):
        initial.InitialState(arguments(f), client, f.task, profile)


@pytest.mark.parametrize('failure', ['missing_profile', 'task', 'path_task', 'different_dataset', 'missing_calibration'])
def test_selected_model_task_and_profile_must_match(trajectory, failure):
    f = trajectory
    client, profile = client_for(f)
    profiles = client.info['raw_observation']['data_profiles']
    if failure == 'missing_profile':
        profile = 'missing'
    elif failure == 'task':
        profiles[profile]['task'] = 'assemble_left'
    elif failure == 'path_task':
        profile = 'assemble_left/lerobot_dataset'
        profiles[profile] = {**profiles.pop(next(iter(profiles))), 'task': None}
    elif failure == 'different_dataset':
        profile = f'{f.task}/other_dataset'
        profiles[profile] = profiles.pop(next(iter(profiles)))
    else:
        profiles[profile].pop('link_to_tcp')
    with pytest.raises(ValueError):
        initial.InitialState(arguments(f), client, f.task, profile)


@pytest.mark.parametrize('matching', [True, False])
@pytest.mark.parametrize('default_profile', [True, False])
def test_explicit_initial_contract_requires_checkpoint_sha_match(trajectory, monkeypatch, matching, default_profile):
    from finetune.OHT.data import replay as buffers
    f = trajectory
    client, profile = client_for(f)
    config = deepcopy(f.config)
    contract = dict(sha256='a' * 64 if matching else 'b' * 64, data_config=config)
    if not default_profile:
        contract['source_data_configs'] = {profile: {'data_config': config}}
    monkeypatch.setattr(buffers, 'load_contract', lambda path: contract)
    args = arguments(f, initial_state_contract='existing-replay-buffer', initial_state_data_config=None)
    if matching:
        initializer = initial.InitialState(args, client, f.task, profile)
        assert initializer.metadata['calibration']['sha256'] == client.info['contract_sha256']
    else:
        with pytest.raises(ValueError, match='contract SHA256'):
            initial.InitialState(args, client, f.task, profile)


def test_model_joint_names_are_inherited_and_explicit_conflicts_are_rejected(trajectory, monkeypatch):
    f = trajectory
    names = [f'j{i}' for i in [5, 3, 1, 4, 2, 0]]
    client, profile = client_for(f, mode='joint', joint_names=names)
    initializer = initial.InitialState(arguments(f, initial_state_frame=3), client, f.task, profile)
    bridge = bridge_for(f, monkeypatch, client, profile)
    report = initializer.initialize(bridge)
    assert report['arm_joint_ids'] == [5, 3, 1, 4, 2, 0]
    assert initializer.metadata['joint_names'] == names
    np.testing.assert_allclose(bridge.robot.data.joint_pos[0, report['arm_joint_ids']], f.joints[3], atol=1e-7)
    with pytest.raises(ValueError, match='joint names/order'):
        initial.InitialState(arguments(f, initial_state_joint_names=list(reversed(names))), client, f.task, profile)


def test_only_explicit_mapped_rigid_objects_are_restored(trajectory, monkeypatch, tmp_path):
    f = trajectory
    client, profile = client_for(f)
    path = tmp_path / 'objects.json'
    path.write_text(json.dumps({'quaternion_order': 'wxyz', 'assets': {'wheel': 0}}), encoding='utf-8')
    initializer = initial.InitialState(arguments(f, initial_state_object_map=str(path)), client, f.task, profile)
    bridge = bridge_for(f, monkeypatch, client, profile)
    poses, velocities = [], []
    bridge.env.scene['wheel'] = SimpleNamespace(write_root_pose_to_sim=lambda value: poses.append(value.clone()),
                                               write_root_velocity_to_sim=lambda value: velocities.append(value.clone()))
    report = initializer.initialize(bridge)
    assert report['objects_restored'] == ['wheel'] and report['object_state'] == 'mapped_recorded_pose'
    np.testing.assert_allclose(poses[0][0], [.4, .2, .7, 1, 0, 0, 0], atol=1e-7)
    assert len(poses) == len(velocities) == 1 and not bridge.env.commands


def test_alignment_gate_and_explicit_tolerance_overrides(trajectory, monkeypatch):
    f = trajectory
    client, profile = client_for(f)
    args = arguments(f)
    initializer = initial.InitialState(args, client, f.task, profile)
    bridge = bridge_for(f, monkeypatch, client, profile)
    initializer.initialize(bridge)
    bridge.robot.data.body_pos_w[0, 0, 0] += .01
    with pytest.raises(ValueError, match='initial TCP mismatch'):
        initializer.check_initial(bridge)
    check = initializer.check_initial(bridge, arguments(f, initial_state_position_tolerance=.02))
    assert .009 < check['errors']['translation_m'] < .011
    assert not bridge.env.commands


@pytest.mark.parametrize('field', ['task', 'profile'])
def test_initializer_cannot_be_applied_to_another_bridge_task_or_profile(trajectory, monkeypatch, field):
    f = trajectory
    client, profile = client_for(f)
    initializer = initial.InitialState(arguments(f), client, f.task, profile)
    bridge = bridge_for(f, monkeypatch, client, profile)
    setattr(bridge, field, 'another')
    with pytest.raises(ValueError, match='selected policy task and data profile'):
        initializer.initialize(bridge)
    assert not bridge.env.writes


@pytest.mark.parametrize('changes', [dict(initial_state_root=None), dict(initial_state_episode=-1),
                                    dict(initial_state_frame=-1), dict(initial_state_frame=7),
                                    dict(initial_state_frame=True), dict(initial_state_position_tolerance=0),
                                    dict(initial_state_rotation_tolerance=float('nan')),
                                    dict(initial_state_contract='buffer')])
def test_invalid_initialization_selection_fails_before_state_write(trajectory, changes):
    f = trajectory
    client, profile = client_for(f)
    with pytest.raises(ValueError):
        initial.InitialState(arguments(f, **changes), client, f.task, profile)
