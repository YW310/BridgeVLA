"""Real Parquet GT loading and native replay without HTTP or model weights."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from finetune.OHT import isaaclab_client as runner
from finetune.OHT.runtime.trajectory_replay import TrajectoryReplay
from tests.test_oht_native_runtime import environment
from tests.test_oht_standalone_client import fake_simulator


@pytest.fixture
def trajectory(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml
    task, n = 'disassemble_left', 7
    root = tmp_path / 'raw'
    data = root / task / 'lerobot_dataset/data/chunk-000/episode_000000.parquet'
    data.parent.mkdir(parents=True)
    joints = np.zeros((n, 6))
    joints[:, 0] = np.arange(n) * .01
    fingers = np.array([[-.015, -.015]] * 3 + [[.011, .011]] * 2 + [[-.015, -.015]] * 2)
    states = np.c_[joints, fingers.sum(axis=1)]
    poses = np.c_[.1 + joints[:, 0], np.zeros(n), np.full(n, .6)]
    columns = {
        'frame_index': list(range(n)), 'episode_index': [0] * n, 'timestamp': (np.arange(n)/60).tolist(),
        'instruction_id': [0]*n, 'instruction': ['private phase']*n,
        'next.done': [False]*(n-1)+[True], 'next.success': [False]*(n-1)+[True],
        'action': [[900.]*6 + [-1.]] * n,  # Deliberately unusable raw OSC controls.
        'observation.state': states.tolist(), 'observation.gripper_joints': fingers.tolist(),
        'observation.ee_pos_world': poses.tolist(), 'observation.ee_quat_world': [[1., 0., 0., 0.]]*n,
        'observation.objects_pos': [[.4, .2, .7, .5, .3, .8]]*n,
        'observation.objects_quat': [[1., 0., 0., 0., 1., 0., 0., 0.]]*n,
    }
    pq.write_table(pa.table(columns), data)
    config = dict(scene_bounds=[-2, -2, -2, 2, 2, 2], link_to_tcp=np.eye(4).tolist(),
                  image_size=[2, 3], camera_quaternion_order='wxyz', ee_quaternion_order='wxyz',
                  gripper=dict(source='config', open=-.03, close=.022), keypoints=dict(method='gripper'),
                  depth=dict(encoding='metric', kind='z'),
                  cameras={'wrist': dict(intrinsics=[[3,0,1],[0,3,1],[0,0,1]], optical_to_sensor=np.eye(4).tolist())})
    config_path = tmp_path / 'dataset.yaml'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    return SimpleNamespace(root=root, data=data, columns=columns, config_path=config_path,
                           config=config, task=task, poses=poses, joints=joints, fingers=fingers, n=n)


def source_args(f, **changes):
    return SimpleNamespace(**{**dict(replay_root=str(f.root), task=f.task, replay_episode=0,
                 replay_start_frame=0, replay_contract=None, replay_data_config=str(f.config_path),
                 replay_joint_names=None, replay_gripper_joint_names=None, replay_object_map=None, replay_cameras=False,
                 max_substeps=1, max_steps=f.n-1, replay_initial_position_tolerance=.005,
                 replay_initial_rotation_tolerance=.05), **changes})


def test_loader_uses_next_measured_tcp_and_training_gripper_not_raw_actions(trajectory):
    f = trajectory
    replay = TrajectoryReplay(source_args(f))
    assert replay.length == f.n-1 and replay.info['cameras'] == []
    assert replay.info['action_spec']['mode'] == 'eef'
    for step in range(replay.length):
        target, meta = replay.action(step)
        np.testing.assert_allclose(target[:3], f.poses[step+1], atol=1e-7)
        np.testing.assert_allclose(target[3:7], [0,0,0,1])
        assert meta['source_frame'] == step and meta['target_frame'] == step+1
        assert target[-1] == (0 if step+1 in (3,4) else 1)
    assert replay.metadata['server_queries'] is False
    assert replay.metadata['source'] == 'observed_next_frame_tcp'
    with pytest.raises(IndexError, match='exhausted'):
        replay.action(replay.length)


def test_start_frame_preserves_full_gripper_history(trajectory):
    import pyarrow as pa
    import pyarrow.parquet as pq
    f = trajectory
    # Ambiguous opening at the start retains an earlier close state.
    columns = deepcopy(f.columns)
    columns['observation.state'][3][-1] = -.004
    columns['observation.gripper_joints'][3] = [-.002, -.002]
    pq.write_table(pa.table(columns), f.data)
    replay = TrajectoryReplay(source_args(f, replay_start_frame=2))
    assert replay.action(0)[0][-1] == 0
    assert replay.length == f.n-3


@pytest.mark.parametrize('order', ['wxyz', 'xyzw'])
def test_replay_preserves_recorded_absolute_rotation_order(trajectory, order):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import yaml
    f = trajectory
    rotation = Rotation.from_euler('xyz', [20, -30, 60], degrees=True).as_quat()
    columns = deepcopy(f.columns)
    columns['observation.ee_quat_world'] = [rotation[[3,0,1,2]].tolist() if order == 'wxyz'
                                           else rotation.tolist()] * f.n
    pq.write_table(pa.table(columns), f.data)
    config = deepcopy(f.config)
    config['ee_quaternion_order'] = order
    f.config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    replay = TrajectoryReplay(source_args(f))
    np.testing.assert_allclose(replay.action(0)[0][3:7], rotation, atol=1e-7)


@pytest.mark.parametrize('failure', ['buffer', 'missing_episode', 'last_frame', 'bad_fingers', 'nonfinite', 'bad_time'])
def test_invalid_source_fails_before_simulator(trajectory, failure, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    f = trajectory
    args = source_args(f)
    if failure == 'buffer':
        (f.root/'samples.jsonl').write_text('{}\n')
    elif failure == 'missing_episode':
        args.replay_episode = 999
    elif failure == 'last_frame':
        args.replay_start_frame = f.n-1
    else:
        columns = deepcopy(f.columns)
        if failure == 'bad_fingers':
            columns['observation.gripper_joints'][0] = [1., 1.]
        elif failure == 'nonfinite':
            columns['observation.ee_pos_world'][1][0] = float('nan')
        else:
            columns['timestamp'][2] = columns['timestamp'][1]
        pq.write_table(pa.table(columns), f.data)
    with pytest.raises(ValueError):
        TrajectoryReplay(args)


@pytest.mark.parametrize('dt,substeps', [(.02,1), (1/60,2), (float('nan'),1)])
def test_replay_rejects_changed_control_rate_or_repeated_actions(trajectory, dt, substeps):
    args = source_args(trajectory, max_substeps=substeps)
    source = TrajectoryReplay(args)
    with pytest.raises(ValueError):
        source.validate_runtime(SimpleNamespace(step_dt=dt), args)


def replay_simulator(monkeypatch, f, *, position_scale=1.):
    env = environment('eef', f.task)
    env.step_dt = 1/60
    robot = env.scene['robot']
    robot.data.joint_vel = torch.zeros_like(robot.data.joint_pos)
    cfg = env.action_manager.get_term('arm_action').cfg
    cfg.body_name, cfg.position_scale = 'tool', position_scale
    cfg.body_offset = SimpleNamespace(pos=[0., 0., .1], rot=[1., 0., 0., 0.])
    env.writes = []
    def write(q, qd):
        env.writes.append(q.clone())
        robot.data.joint_pos[:] = q
        robot.data.joint_vel[:] = qd
    robot.write_joint_state_to_sim = write
    robot.set_joint_position_target = lambda q: None
    robot.set_joint_velocity_target = lambda q: None
    def forward():
        robot.data.body_pos_w[0, 0] = torch.tensor([.1, 0., .5]) + robot.data.joint_pos[0, :3]
    env.sim.forward = forward
    original_step = env.step
    def step(command):
        # The actual ActionTerm applies its configured scale.
        physical = command.clone()
        physical[:, 1:4] *= position_scale
        result = original_step(physical)
        env.commands[-1] = command.numpy()[0].copy()
        return result
    env.step = step
    state = fake_simulator(monkeypatch, env)
    monkeypatch.setattr(runner, 'BridgeVLAClient', lambda *a, **k: pytest.fail('Replay contacted an HTTP server'))
    return env, state


def command(f, output, *extra):
    return ['--mode', 'replay', '--replay-root', str(f.root), '--replay-data-config', str(f.config_path),
            '--task', f.task, '--env-id', 'OHT-test', '--output', str(output), '--headless', *extra]


def test_native_replay_runs_complete_trajectory_without_http_or_camera_binding(trajectory, tmp_path, monkeypatch):
    f = trajectory
    env, state = replay_simulator(monkeypatch, f, position_scale=.01)
    output = tmp_path/'run'
    assert runner.main(command(f, output)) == 0
    assert env.closed and state.app_closed
    assert len(env.writes) == 1 and env.common_step_counter == f.n-1
    np.testing.assert_allclose(env.writes[0][0, 6:], f.fingers[0], atol=1e-7)
    episode = json.loads((output/'episode_0000.json').read_text())
    metadata = json.loads((output/'run.json').read_text())
    summary = json.loads((output/'summary.json').read_text())
    assert episode['stop_reason'] == 'trajectory_end' and episode['success'] is None
    assert summary['planned'] == summary['attempted'] == 1 and summary['success_rate'] is None
    assert metadata['mode'] == 'replay' and metadata['model_id'] is None and metadata['camera_source'] == 'none'
    assert metadata['initialization'] == 'trajectory' and metadata['replay']['server_queries'] is False
    assert episode['initialization']['object_state'] == 'environment_reset'
    for i, action in enumerate(episode['actions']):
        assert action['action_source'] == 'gt_trajectory' and action['target_frame'] == i+1
        assert action['timing_ms'] == {}
        np.testing.assert_allclose(action['target'][:3], f.poses[i+1], atol=1e-7)
        assert action['errors_after']['translation_m'] < 1e-6
    np.testing.assert_allclose(np.asarray(env.commands)[:, 1], 1., atol=1e-5)
    np.testing.assert_allclose(np.asarray(env.commands)[:, 0], [1,1,-1,-1,1,1])


@pytest.mark.parametrize('option,value,expected', [('--max-steps','2','max_steps'),
                                                 ('--replay-start-frame','2','trajectory_end')])
def test_replay_limits_and_initial_frame(trajectory, tmp_path, monkeypatch, option, value, expected):
    f = trajectory
    env, _ = replay_simulator(monkeypatch, f)
    out = tmp_path/'run'
    assert runner.main(command(f, out, option, value)) == 0
    row = json.loads((out/'episode_0000.json').read_text())
    assert row['stop_reason'] == expected
    if option == '--replay-start-frame':
        assert row['actions'][0]['source_frame'] == 2
        np.testing.assert_allclose(env.writes[0][0, 0], .02)
    else:
        assert row['steps'] == 2


def test_replay_dry_run_does_not_teleport_or_control(trajectory, tmp_path, monkeypatch):
    env, state = replay_simulator(monkeypatch, trajectory)
    output = tmp_path/'dry'
    assert runner.main(command(trajectory, output, '--dry-run')) == 0
    assert (output/'control_diagnostics.json').is_file()
    assert not env.commands and not env.writes and env.closed and state.app_closed


def test_wrong_tcp_reference_stops_before_first_action_and_saves_error(trajectory, tmp_path, monkeypatch):
    env, state = replay_simulator(monkeypatch, trajectory)
    output = tmp_path/'bad-tcp'
    assert runner.main(command(trajectory, output, '--ee-pose-source', 'body')) == 1
    row = json.loads((output/'episode_0000.json').read_text())
    assert 'initial TCP mismatch' in row['error']
    assert row['stop_reason'] == 'error' and row['steps'] == 0 and not row['actions']
    assert not env.commands and env.closed and state.app_closed


def test_environment_done_stops_replay_without_claiming_success(trajectory, tmp_path, monkeypatch):
    env, _ = replay_simulator(monkeypatch, trajectory)
    env.done_after = 2
    out = tmp_path/'done'
    assert runner.main(command(trajectory, out)) == 0
    row = json.loads((out/'episode_0000.json').read_text())
    assert row['stop_reason'] == 'environment_done' and row['steps'] == 2 and row['success'] is None


def test_replay_cleans_up_if_control_rate_is_wrong(trajectory, tmp_path, monkeypatch):
    env, state = replay_simulator(monkeypatch, trajectory)
    env.step_dt = .02
    with pytest.raises(ValueError, match='control rate'):
        runner.main(command(trajectory, tmp_path/'bad-rate'))
    assert env.closed and state.app_closed and not env.commands and not env.writes


def test_explicit_joint_names_restore_recorded_order_not_controller_term_order(trajectory, monkeypatch):
    f = trajectory
    env, _ = replay_simulator(monkeypatch, f)
    arm = env.action_manager.get_term('arm_action')
    arm._joint_ids = [5, 3, 1, 4, 2, 0]
    gripper = env.action_manager.get_term('gripper_action')
    gripper._joint_ids = [7, 6]
    source = TrajectoryReplay(source_args(f, replay_joint_names=[f'j{i}' for i in range(6)],
                                         replay_gripper_joint_names=['j6', 'j7']))
    source.states[0, :6] = np.arange(6)*.1
    source.fingers[0] = [-.01, -.02]
    bridge = SimpleNamespace(env=env, robot=env.scene['robot'], ref=SimpleNamespace(torch=torch))
    report = source.initialize(bridge)
    np.testing.assert_allclose(env.writes[0][0, :6], np.arange(6)*.1, atol=1e-7)
    np.testing.assert_allclose(env.writes[0][0, 6:], [-.01,-.02], atol=1e-7)
    assert report['arm_joint_ids'] == list(range(6)) and report['gripper_joint_ids'] == [6,7]


def test_repeated_episodes_restore_once_per_episode(trajectory, tmp_path, monkeypatch):
    env, _ = replay_simulator(monkeypatch, trajectory)
    out = tmp_path/'repeated'
    assert runner.main(command(trajectory, out, '--episodes', '2', '--max-steps', '2')) == 0
    assert len(env.writes) == 2 and len(env.commands) == 4
    summary = json.loads((out/'summary.json').read_text())
    assert summary['attempted'] == 2 and summary['success_rate'] is None


def test_calibration_from_existing_buffer_matches_source_profile(trajectory, monkeypatch):
    from finetune.OHT.data import replay as buffers
    f = trajectory
    config = deepcopy(f.config)
    config['link_to_tcp'][2][3] = .03
    contract = dict(sha256='test-contract', source_data_configs={f'{f.task}/lerobot_dataset': {'data_config': config}})
    monkeypatch.setattr(buffers, 'load_contract', lambda path: contract)
    source = TrajectoryReplay(source_args(f, replay_contract='existing-buffer', replay_data_config=None))
    assert source.info['contract_sha256'] == 'test-contract'
    np.testing.assert_allclose(source.poses[:, 2], .63)
    assert source.metadata['calibration']['sha256'] == 'test-contract'


@pytest.mark.parametrize('sdk_order,source_order', [('wxyz','xyzw'), ('xyzw','wxyz')])
def test_explicit_object_mapping_restores_pose_and_origin(trajectory, tmp_path, monkeypatch, sdk_order, source_order):
    f = trajectory
    env, _ = replay_simulator(monkeypatch, f)
    # Use the loader/initializer directly to verify both quaternion conversions.
    mapping = tmp_path/'objects.json'
    mapping.write_text(json.dumps(dict(quaternion_order=source_order, assets={'wheel': 1})))
    args = source_args(f, replay_object_map=str(mapping))
    source = TrajectoryReplay(args)
    rotation = Rotation.from_euler('xyz', [20,30,40], degrees=True).as_quat()
    source.orientations[:, 1] = rotation if source_order == 'xyzw' else rotation[[3,0,1,2]]
    env.scene.env_origins = torch.tensor([[10.,20.,30.]])
    seen = {}
    env.scene['wheel'] = SimpleNamespace(write_root_pose_to_sim=lambda pose: seen.update(pose=pose.numpy()),
                                         write_root_velocity_to_sim=lambda v: seen.update(velocity=v.numpy()))
    bridge = SimpleNamespace(env=env, robot=env.scene['robot'],
                             ref=SimpleNamespace(torch=torch, quaternion_order=sdk_order))
    report = source.initialize(bridge)
    np.testing.assert_allclose(seen['pose'][0,:3], [10.5,20.3,30.8], atol=2e-6)
    expected = rotation if sdk_order == 'xyzw' else rotation[[3,0,1,2]]
    np.testing.assert_allclose(seen['pose'][0,3:], expected, atol=1e-7)
    assert not seen['velocity'].any() and report['objects_restored'] == ['wheel']


@pytest.mark.parametrize('definition', [{'quaternion_order':'guess','assets':{'wheel':0}},
                                       {'quaternion_order':'wxyz','assets':{'wheel':99}},
                                       {'quaternion_order':'wxyz','assets':{'left':0,'right':0}}])
def test_object_mapping_rejects_guesses_and_invalid_indices(trajectory, tmp_path, definition):
    path = tmp_path/'bad-map.json'
    path.write_text(json.dumps(definition))
    with pytest.raises(ValueError):
        TrajectoryReplay(source_args(trajectory, replay_object_map=str(path)))


@pytest.mark.parametrize('extra', [[], ['--mode','replay'],
                                 ['--initialization','trajectory'],
                                 ['--mode','replay','--initialization','env-reset','--replay-object-map','objects.json']])
def test_invalid_mode_options_do_not_contact_server_or_launch_simulator(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(runner, 'BridgeVLAClient', lambda *a, **k: pytest.fail('Unexpected HTTP request'))
    if not extra:
        extra = ['--replay-root', str(tmp_path)]  # policy mode must not ignore a replay source
    with pytest.raises(ValueError):
        runner.main(['--output', str(tmp_path/'bad-run'), *extra])


@pytest.mark.parametrize('extra', [
    ['--replay-cameras'], ['--replay-contract', 'buffer'], ['--replay-data-config', 'data.yaml'],
    ['--replay-object-map', 'objects.json'], ['--replay-episode', '0'], ['--replay-start-frame=0'],
    ['--replay-joint-names', 'a', 'b', 'c', 'd', 'e', 'f'],
    ['--replay-gripper-joint-names', 'left', 'right'],
    ['--replay-initial-position-tolerance', '.005'], ['--replay-initial-rotation-tolerance', '.05'],
])
def test_explicit_replay_flags_require_replay_mode(tmp_path, monkeypatch, extra):
    monkeypatch.setattr(runner, 'BridgeVLAClient', lambda *a, **k: pytest.fail('Policy contacted HTTP with replay-only flags'))
    with pytest.raises(ValueError, match='require --mode replay'):
        runner.main(['--output', str(tmp_path/'bad-options'), *extra])


@pytest.mark.parametrize('name', ['joint_pos_limits', 'joint_limits', 'default_joint_pos_limits', 'default_joint_limits'])
def test_recorded_initial_state_uses_hard_joint_limits_not_soft(trajectory, monkeypatch, name):
    f = trajectory
    env, _ = replay_simulator(monkeypatch, f)
    robot = env.scene['robot']
    robot.data.soft_joint_pos_limits = torch.tensor([[[-2., 2.]] * 8])
    setattr(robot.data, name, torch.tensor([[[-3., 3.]] * 8]))
    source = TrajectoryReplay(source_args(f))
    source.states[0, 0] = 2.9
    bridge = SimpleNamespace(env=env, robot=robot, ref=SimpleNamespace(torch=torch))
    report = source.initialize(bridge)
    assert len(env.writes) == 1 and report['joint_limit_source'] == name
    assert env.writes[0][0, 0] == pytest.approx(2.9)
    env.writes.clear()
    source.states[0, 0] = 3.1
    with pytest.raises(ValueError, match='hard joint limits'):
        source.initialize(bridge)
    assert not env.writes and not env.commands


@pytest.mark.parametrize('selection', ['robot', 'other_articulation', 'aliased_object'])
def test_object_map_rejects_articulations_and_aliases_before_writing(trajectory, tmp_path, monkeypatch, selection):
    env, _ = replay_simulator(monkeypatch, trajectory)
    robot = env.scene['robot']
    written = []
    robot.write_root_pose_to_sim = lambda value: written.append(value)
    robot.write_root_velocity_to_sim = lambda value: written.append(value)
    if selection == 'robot':
        assets = {'robot': 0}
    elif selection == 'other_articulation':
        env.scene['other'] = SimpleNamespace(write_joint_state_to_sim=lambda *a: None,
                         write_root_pose_to_sim=lambda value: written.append(value),
                         write_root_velocity_to_sim=lambda value: written.append(value))
        assets = {'other': 0}
    else:
        obj = SimpleNamespace(write_root_pose_to_sim=lambda value: written.append(value),
                              write_root_velocity_to_sim=lambda value: written.append(value))
        env.scene['left'] = env.scene['right'] = obj
        assets = {'left': 0, 'right': 1}
    path = tmp_path/'object-map.json'
    path.write_text(json.dumps(dict(quaternion_order='wxyz', assets=assets)))
    source = TrajectoryReplay(source_args(trajectory, replay_object_map=str(path)))
    bridge = SimpleNamespace(env=env, robot=robot, ref=SimpleNamespace(torch=torch))
    with pytest.raises(ValueError, match='rigid object|same scene object'):
        source.initialize(bridge)
    assert not env.writes and not written and not env.commands
