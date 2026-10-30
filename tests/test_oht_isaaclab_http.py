"""Raw-sensor HTTP parity and reference OSC math without launching Isaac Sim."""
import copy
import io
import json
from types import SimpleNamespace
import urllib.error
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from fastapi.testclient import TestClient

from tests.test_oht_migration import replay_fixture  # noqa: F401
from tests.test_oht_vla_server import fixture, prediction, envelope  # noqa: F401
from finetune.OHT.action_modes import action_options, action_descriptor
from finetune.OHT.data.dataset import OHTDataset
from finetune.OHT.data.geometry import pose_matrix
from finetune.OHT.runtime.raw_observation import RAW_FORMAT, decode_raw, OPENGL_TO_OPTICAL, profile_task
from finetune.OHT.runtime.transport import pack_observation
from finetune.OHT.runtime.http_client import BridgeVLAClient
from finetune.OHT.runtime.isaaclab_bridge import osc_command, IsaacLabBridge, pose_wxyz_matrix
from finetune.OHT.isaaclab_client import load_reference
from finetune.OHT.runtime.inference_api import API_VERSION


@pytest.mark.parametrize('field,value', [('frame', 'base'), ('quaternion_order', 'wxyz'),
                                       ('size', 7), ('units', 'millimetres'),
                                       ('gripper', 'close01'), ('reference_point', 'flange')])
def test_client_rejects_semantically_incompatible_server_before_control(field, value):
    spec = action_descriptor(action_options())
    spec[field] = value
    info = dict(api_version=API_VERSION, observation_formats=[RAW_FORMAT], action_spec=spec)
    class InfoOnly:
        def open(self, request, timeout):
            assert request.full_url.endswith('/model_info')
            return io.BytesIO(json.dumps(info).encode())
    with pytest.raises(ValueError, match='Unsupported action_spec'):
        BridgeVLAClient('http://testserver', opener=InfoOnly())


def raw_sample(replay, policy):
    data = OHTDataset(replay, 'train')
    row, meta = data[0], data.samples[0]
    profile = meta.get('data_profile', 'default')
    configs = policy.contract.get('source_data_configs', {})
    config = configs[profile]['data_config'] if configs else policy.contract['data_config']
    link = pose_matrix(row['tcp_pose'][:3], row['tcp_pose'][3:]) @ np.linalg.inv(config['link_to_tcp'])
    q = Rotation.from_matrix(link[:3, :3]).as_quat()
    tensors = dict(joint_positions=row['joint_positions'],
                   ee_link_pose=np.r_[link[:3, 3], q[[3, 0, 1, 2]]].astype(np.float32),
                   gripper_position=np.array([config['gripper']['open']], np.float32))
    for name in config['cameras']:
        camera = row[name + '_camera_extrinsics'] @ OPENGL_TO_OPTICAL
        q = Rotation.from_matrix(camera[:3, :3]).as_quat()
        tensors.update({name + '_rgb': row[name + '_rgb'].transpose(1, 2, 0),
                        name + '_depth': row[name + '_depth'][0],
                        name + '_intrinsics': row[name + '_camera_intrinsics'],
                        name + '_pose': np.r_[camera[:3, 3], q[[3, 0, 1, 2]]].astype(np.float32)})
    return tensors, row, dict(task=meta['task'], data_profile=profile, tensors=pack_observation(tensors))


@pytest.mark.parametrize('mode', ['eef', 'relative_eef', 'joint', 'relative_joint'])
def test_raw_sensors_reproduce_replay_geometry_and_state(fixture, replay_fixture, mode):
    fixture.policy.action_config = action_options({'mode': mode})
    _, row, packed = raw_sample(replay_fixture.replay, fixture.policy)
    prepared, state = decode_raw(packed, fixture.policy)
    for name in fixture.policy.contract['data_config']['cameras']:
        for suffix in ('rgb', 'depth', 'point_cloud', 'camera_intrinsics', 'camera_extrinsics'):
            np.testing.assert_allclose(prepared[name + '_' + suffix], row[name + '_' + suffix],
                                       atol=1e-6, equal_nan=True)
    np.testing.assert_allclose(prepared['low_dim_state'], row['low_dim_state'])
    if mode == 'relative_eef':
        np.testing.assert_allclose(prepared['tcp_pose'], row['tcp_pose'], atol=1e-6)
    if 'joint' in mode:
        np.testing.assert_allclose(prepared['joint_positions'], row['joint_positions'])
    assert state[0] == pytest.approx(1.) and state[1] == 1
    assert 'action' not in prepared and 'gripper_pose' not in prepared


@pytest.mark.parametrize('bad', ['missing_camera', 'teacher', 'profile', 'task', 'dtype', 'quaternion'])
def test_raw_bad_inputs_do_not_advance_or_call_model(fixture, replay_fixture, bad):
    _, _, raw = raw_sample(replay_fixture.replay, fixture.policy)
    with TestClient(fixture.app()) as client:
        info = client.get('/model_info').json()
        request = prediction(fixture, info)
        request.update(observation_format=RAW_FORMAT, observation=copy.deepcopy(raw))
        observation = request['observation']
        if bad == 'missing_camera':
            observation['tensors'].pop('wrist_depth')
        elif bad == 'teacher':
            observation['tensors']['teacher_target'] = observation['tensors']['gripper_position']
        elif bad == 'profile':
            observation['data_profile'] = 'missing'
        elif bad == 'task':
            observation['task'] = 'missing'
        elif bad == 'dtype':
            observation['tensors']['wrist_rgb']['dtype'] = 'float32'
        else:
            observation['tensors'].update(pack_observation({'ee_link_pose': np.zeros(7, np.float32)}))
        assert client.post('/predict', json=request).status_code == 422
        assert not fixture.policy.calls
        assert client.get('/health').json()['next_step'] == 0
        request['observation'] = raw
        assert client.post('/predict', json=request).status_code == 200
        # Changing observation formats inside an episode must be explicit.
        next_request = prediction(fixture, info, step=1)
        assert client.post('/predict', json=next_request).status_code == 409


class Opener:
    def __init__(self, client):
        self.client, self.fail = client, False
    def open(self, request, timeout):
        if self.fail:
            raise TimeoutError('uncertain control reply')
        path = request.full_url.split('http://testserver', 1)[-1]
        response = self.client.request(request.get_method(), path, content=request.data,
                                       headers={'Content-Type': 'application/json'})
        if response.status_code != 200:
            raise urllib.error.HTTPError(request.full_url, response.status_code, response.text, {}, None)
        return io.BytesIO(response.content)


@pytest.mark.parametrize('task', ['assemble_left', 'assemble_right', 'disassemble_left', 'disassemble_right'])
@pytest.mark.parametrize('suffix', ['', '-seg'])
def test_profile_task_recognizes_raw_dataset_directory(task, suffix):
    assert profile_task(f'{task}{suffix}/lerobot_dataset') == task


@pytest.mark.parametrize('name', ['default', '.', 'lerobot_dataset',
                                  'not_disassemble_left-seg/lerobot_dataset',
                                  'disassemble_left-seg-extra/lerobot_dataset',
                                  'assemble_left/disassemble_right-seg/lerobot_dataset'])
def test_profile_task_does_not_guess_unknown_or_ambiguous_names(name):
    assert profile_task(name) is None


@pytest.mark.parametrize('profiles,expected', [
    ({'disassemble_left/lerobot_dataset': {'task': 'disassemble_left'}},
     'disassemble_left/lerobot_dataset'),
    ({'disassemble_left-seg/lerobot_dataset': {'task': 'disassemble_left'}},
     'disassemble_left-seg/lerobot_dataset'),
    ({'export-01': {'task': 'disassemble_left'}}, 'export-01'),
    ({'default': {'task': None}}, 'default'),
    ({'.': {'task': None}}, '.'),
    ({'lerobot_dataset': {'task': None}}, 'lerobot_dataset'),
    ({'disassemble_left-seg/lerobot_dataset': {'task': None},
      'disassemble_right-seg/lerobot_dataset': {'task': None}},
     'disassemble_left-seg/lerobot_dataset'),
    ({'left': {'task': 'disassemble_left'}, 'right': {'task': 'disassemble_right'},
      'generic': {'task': None}}, 'left'),
])
def test_client_selects_unique_advertised_task_or_single_generic_profile(profiles, expected):
    client = BridgeVLAClient.__new__(BridgeVLAClient)
    client.info = {'raw_observation': {'data_profiles': profiles}}
    assert client.profile('disassemble_left') == expected


@pytest.mark.parametrize('profiles', [
    {},
    {'right': {'task': 'disassemble_right'}},
    {'left-a': {'task': 'disassemble_left'}, 'left-b': {'task': 'disassemble_left'}},
    {'disassemble_left/lerobot_dataset': {'task': 'disassemble_left'},
     'disassemble_left-seg/lerobot_dataset': {'task': 'disassemble_left'}},
    {'generic-a': {'task': None}, 'generic-b': {'task': None}},
    {'right': {'task': 'disassemble_right'}, 'generic': {'task': None}},
])
def test_client_profile_selection_failure_lists_candidates(profiles):
    client = BridgeVLAClient.__new__(BridgeVLAClient)
    client.info = {'raw_observation': {'data_profiles': profiles}}
    with pytest.raises(ValueError, match='Select --data-profile') as error:
        client.profile('disassemble_left')
    assert 'disassemble_left' in str(error.value)
    for name, profile in profiles.items():
        assert name in str(error.value)
        assert str(profile['task']) in str(error.value)


def test_client_explicit_profile_resolves_ambiguity_but_rejects_wrong_task():
    client = BridgeVLAClient.__new__(BridgeVLAClient)
    client.info = {'raw_observation': {'data_profiles': {
        'left-a': {'task': 'disassemble_left'}, 'left-b': {'task': 'disassemble_left'},
        'right': {'task': 'disassemble_right'}, 'generic': {'task': None},
        'disassemble_right-seg/lerobot_dataset': {'task': None}}}}
    assert client.profile('disassemble_left', 'left-b') == 'left-b'
    assert client.profile('disassemble_left', 'generic') == 'generic'
    for name in ('right', 'missing', 'disassemble_right-seg/lerobot_dataset'):
        with pytest.raises(ValueError, match='Unknown/mismatched'):
            client.profile('disassemble_left', name)


@pytest.mark.parametrize('suffix', ['', '-seg'])
def test_profile_http_auto_selection_and_task_guard(fixture, replay_fixture, suffix):
    tensors, row, raw = raw_sample(replay_fixture.replay, fixture.policy)
    profiles = fixture.policy.contract['source_data_configs']
    name = raw['task'] + suffix + '/lerobot_dataset'
    profiles[name] = profiles.pop(raw['data_profile'])
    raw['data_profile'] = name
    with TestClient(fixture.app()) as server:
        client = BridgeVLAClient('http://testserver', opener=Opener(server))
        assert client.info['raw_observation']['data_profiles'][name]['task'] == raw['task']
        assert client.profile(raw['task']) == name
        info = client.info
        request = prediction(fixture, info)
        request.update(observation_format=RAW_FORMAT, observation=copy.deepcopy(raw))
        request['observation']['task'] = ('disassemble_right' if raw['task'] != 'disassemble_right'
                                          else 'disassemble_left')
        assert server.post('/predict', json=request).status_code == 422
        assert not fixture.policy.calls
        client.reset()
        action, response = client.predict(tensors, row['goal'], 0., task=raw['task'], data_profile=name)
        assert action.shape == (8,) and response['step'] == 0


def test_http_client_raw_session_and_no_retry_after_timeout(fixture, replay_fixture):
    tensors, row, raw = raw_sample(replay_fixture.replay, fixture.policy)
    with TestClient(fixture.app()) as server:
        opener = Opener(server)
        client = BridgeVLAClient('http://testserver', opener=opener)
        assert client.profile(raw['task']) == raw['data_profile']
        client.reset()
        action, response = client.predict(tensors, row['goal'], 0., task=raw['task'], data_profile=raw['data_profile'])
        assert action.shape == (8,) and response['step'] == 0
        assert set(fixture.policy.calls[0][0]) == set(fixture.observation)
        opener.fail = True
        with pytest.raises(TimeoutError):
            client.predict(tensors, row['goal'], .1, task=raw['task'], data_profile=raw['data_profile'])
        opener.fail = False
        with pytest.raises(RuntimeError, match='fresh episode'):
            client.predict(tensors, row['goal'], .1, task=raw['task'], data_profile=raw['data_profile'])
        assert len(fixture.policy.calls) == 1
        client.reset()
        client.predict(tensors, row['goal'], 0., task=raw['task'], data_profile=raw['data_profile'])
        assert len(fixture.policy.calls) == 2


def test_osc_rotates_world_to_base_and_clips_before_scale_division():
    base = pose_matrix([2, 3, 0], Rotation.from_euler('z', 90, degrees=True).as_quat())
    current = base.copy()
    target = base @ pose_matrix([.1, 0, 0], Rotation.from_rotvec([0, 0, .1]).as_quat())
    command = osc_command(current, target, base, .01, .2, .05, .15, 0)
    np.testing.assert_allclose(command, [5, 0, 0, 0, 0, .5, -1], atol=1e-6)


def test_fixed_relative_tcp_target_is_not_accumulated_again_per_substep():
    import torch
    link_to_tcp = np.eye(4)
    link_to_tcp[0, 3] = .2
    client = SimpleNamespace(action_config=action_options({'mode': 'relative_eef'}),
                             info={'raw_observation': {'data_profiles': {'p': {'link_to_tcp': link_to_tcp}}}})
    cfg = SimpleNamespace(position_scale=1., orientation_scale=1.)
    term = type('OperationalSpaceAction', (), {'cfg': cfg})()
    robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=np.zeros((1, 3)),
                                                 root_quat_w=np.array([[1., 0, 0, 0]])))
    class Scene(dict):
        env_origins = np.zeros((1, 3))
    env = SimpleNamespace(scene=Scene(robot=robot), action_manager=SimpleNamespace(get_term=lambda name: term), device='cpu')
    state, commands = np.zeros(3), []
    def step(value):
        command = value.numpy()[0]
        commands.append(command.copy())
        state[:] += command[:3]
    env.step = step
    ref = SimpleNamespace(torch=torch, tensor_to_numpy=lambda value, dtype: np.asarray(value, dtype),
                          get_ee_pose_in_world_frame=lambda env, robot: (state.copy(), np.array([1, 0, 0, 0])))
    bridge = IsaacLabBridge(ref, env, client, 'assemble_left', 'p', {}, max_substeps=2,
                            max_translation=1., max_rotation=1.)
    measured = dict(ee_link_pose=np.array([0, 0, 0, 1, 0, 0, 0]), joint_positions=np.zeros(6))
    bridge.execute(np.array([.1, 0, 0, 0, 0, 0, 1]), measured)
    np.testing.assert_allclose(commands[0][:3], [.1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(commands[1][:3], [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(state, [.1, 0, 0], atol=1e-6)


def test_reference_import_preserves_main_guard_and_argv(tmp_path):
    import sys
    path = tmp_path / 'runtime.py'
    path.write_text('from types import SimpleNamespace\nimport sys\n'
                    'simulation_app=SimpleNamespace(close=lambda:None)\n'
                    'arguments=list(sys.argv)\n'
                    'if __name__ == "__main__":\n    raise RuntimeError("JEPA loop ran")\n', encoding='utf-8')
    previous = list(sys.argv)
    reference = load_reference(path, ['--headless'])
    assert reference.arguments == [str(path), '--headless']
    assert sys.argv == previous


def test_raw_gripper_history_is_transactional_and_reset_clears_it(fixture, replay_fixture):
    _, _, raw = raw_sample(replay_fixture.replay, fixture.policy)
    profiles = fixture.policy.contract.get('source_data_configs', {})
    cfg = profiles[raw['data_profile']]['data_config'] if profiles else fixture.policy.contract['data_config']
    grip = cfg['gripper']
    def at(measured):
        result = copy.deepcopy(raw)
        value = grip['close'] - measured * (grip['close'] - grip['open'])
        result['tensors'].update(pack_observation({'gripper_position': np.array([value], np.float32)}))
        return result
    with TestClient(fixture.app()) as client:
        info = client.get('/model_info').json()
        for step, measured in enumerate([.6, .5]):
            request = prediction(fixture, info, step=step)
            request.update(observation_format=RAW_FORMAT, observation=at(measured))
            assert client.post('/predict', json=request).status_code == 200
        assert fixture.policy.calls[-1][0]['low_dim_state'][0] == 0
        history = client.app.state.service.gripper_history
        bad = prediction(fixture, info, step=2)
        bad.update(observation_format=RAW_FORMAT, observation=at(.8))
        bad['observation']['tensors'].pop('wrist_pose')
        assert client.post('/predict', json=bad).status_code == 422
        assert client.app.state.service.gripper_history == history
        assert client.post('/reset', json=envelope(info, episode='new', request_id='new-reset')).status_code == 200
        assert client.app.state.service.gripper_history is None
        request = prediction(fixture, info, episode='new')
        request.update(observation_format=RAW_FORMAT, observation=at(.5))
        assert client.post('/predict', json=request).status_code == 200
        assert fixture.policy.calls[-1][0]['low_dim_state'][0] == 1


def test_http_runner_capture_predict_execute_and_summary(fixture, replay_fixture, tmp_path, monkeypatch):
    import torch
    from finetune.OHT import isaaclab_client as runner
    tensors, _, raw = raw_sample(replay_fixture.replay, fixture.policy)
    class Scene(dict):
        env_origins = np.zeros((1, 3))
    position = tensors['ee_link_pose'][:3].copy()
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=np.zeros((1, 6)), root_pos_w=np.zeros((1, 3)),
                                                 root_quat_w=np.array([[1, 0, 0, 0]])))
    cfg = SimpleNamespace(position_scale=1., orientation_scale=1.)
    term = type('OperationalSpaceAction', (), {'cfg': cfg, '_joint_ids': list(range(6))})()
    env = SimpleNamespace(scene=Scene(robot=robot), action_manager=SimpleNamespace(get_term=lambda name: term),
                          device='cpu', common_step_counter=0, step_dt=.02, closed=False)
    def reset():
        env.common_step_counter = 0
        position[:] = tensors['ee_link_pose'][:3]
    def step(action):
        env.common_step_counter += 1
        position[:] += action.numpy()[0, :3]
    env.reset, env.step, env.close = reset, step, lambda: setattr(env, 'closed', True)
    class Camera(dict):
        pass
    camera_map = {name: 'TOOL_CAMERA' if name == 'wrist' else name.upper()
                  for name in fixture.policy.contract['data_config']['cameras']}
    sensors = {usd: Camera({suffix: tensors[f'{name}_{suffix}'] for suffix in ('rgb', 'depth', 'intrinsics', 'pose')})
               for name, usd in camera_map.items()}
    for name, sensor in sensors.items():
        sensor.name = name
    class CameraManager(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(sensors)
        def warmup(self, env, num_steps):
            pass
    app = SimpleNamespace(is_running=lambda: True, closed=False)
    app.close = lambda: setattr(app, 'closed', True)
    ref = SimpleNamespace(torch=torch, args_cli=SimpleNamespace(device='cpu'), simulation_app=app,
                          parse_env_cfg=lambda *args, **kwargs: SimpleNamespace(),
                          gym=SimpleNamespace(make=lambda *args, **kwargs: SimpleNamespace(unwrapped=env)),
                          OnlineReplicatorCameraManager=CameraManager,
                          _replicator_render_tick=lambda env: None,
                          tensor_to_numpy=lambda value, dtype: np.asarray(value, dtype),
                          get_ee_pose_in_world_frame=lambda env, robot: (position.copy(), tensors['ee_link_pose'][3:]),
                          get_online_gripper_physical_state=lambda env, robot: tensors['gripper_position'],
                          camera_rgb=lambda camera: camera['rgb'], camera_depth=lambda camera: camera['depth'],
                          camera_intrinsic=lambda camera: camera['intrinsics'],
                          camera_pose_ros=lambda camera, env, robot: camera['pose'])
    monkeypatch.setattr(runner, 'load_reference', lambda *args: ref)
    with TestClient(fixture.app()) as server:
        monkeypatch.setattr(runner, 'BridgeVLAClient',
                            lambda *args: BridgeVLAClient('http://testserver', opener=Opener(server)))
        output = tmp_path / 'rollout'
        map_path = tmp_path / 'cameras.json'
        map_path.write_text(json.dumps(camera_map), encoding='utf-8')
        assert runner.main(['--reference-client', 'unused.py', '--task', raw['task'], '--episodes', '1',
                            '--execution-mode', 'streaming', '--max-steps', '2', '--output', str(output), '--camera-map', str(map_path)]) == 0
        assert len(fixture.policy.calls) == env.common_step_counter == 2
        assert env.closed and app.closed
        assert sensors['TOOL_CAMERA'].name == 'WRIST'
        summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
        assert summary['attempted'] == 1 and summary['success_rate'] is None
        record = json.loads((output / 'episode_0000.json').read_text(encoding='utf-8'))
        assert len(record['actions']) == 2 and record['actions'][0]['timestamp'] == 0.
