"""Standalone launcher and built-in native runtime, without requiring Isaac Sim."""
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from scipy.spatial.transform import Rotation

from tests.test_oht_migration import replay_fixture  # noqa: F401
from tests.test_oht_vla_server import fixture  # noqa: F401
from tests.test_oht_isaaclab_http import raw_sample, Opener
from tests.test_oht_native_runtime import environment
from finetune.OHT import isaaclab_client as runner
from finetune.OHT.action_modes import action_options
from finetune.OHT.runtime import isaaclab_factory as native
from finetune.OHT.runtime import isaaclab_scene as scenes
from finetune.OHT.runtime.http_client import BridgeVLAClient
from finetune.OHT.runtime.isaaclab_bridge import IsaacLabBridge
from finetune.OHT.runtime.isaaclab_runtime import IsaacLabRuntime
from finetune.OHT.runtime.policy_worker import PolicyWorker
from finetune.OHT.runtime.raw_observation import raw_schema
from finetune.OHT.runtime.usd_cameras import (
    ReplicatorCamera, ReplicatorCameras, ReplicatorRuntime, projection_intrinsics, select_camera_paths, usd_rigid_matrix,
)


def native_arguments(**changes):
    values = dict(camera_width=3, camera_height=2, camera_warmup_ticks=4, runtime_args=['--headless'],
                  task_import=['project.tasks'], env_id='OHT-test', env_config=None, disable_fabric=False,
                  scene_preset='registry', expected_robot_usd=None, isaaclab_root=None,
                  seed=42, camera_source='auto', ee_link=None, ee_pose_source='controller', quaternion_order='wxyz', camera_map=None)
    return SimpleNamespace(**{**values, **changes})


def sensors_for(tensors, names):
    class Camera:
        def __init__(self, name):
            self.data = SimpleNamespace(output={'rgb': tensors[name + '_rgb'][None],
                                                'distance_to_image_plane': tensors[name + '_depth'][None, ..., None]},
                                        pos_w=tensors[name + '_pose'][None, :3],
                                        quat_w_opengl=tensors[name + '_pose'][None, 3:],
                                        intrinsic_matrices=tensors[name + '_intrinsics'][None])
            self.updates = 0
        def update(self, dt, force_recompute):
            assert dt == 0 and force_recompute
            self.updates += 1
    return {name: Camera(name) for name in names}


def fake_simulator(monkeypatch, env):
    state = SimpleNamespace(launched=False, app_closed=False, cfg=None, parse_calls=[])
    app = SimpleNamespace(is_running=lambda: True, close=lambda: setattr(state, 'app_closed', True))
    def launch(extra):
        assert len(sys.argv) == 1
        state.launched = True
        return app, SimpleNamespace(device='cpu')
    def parse(name, **kwargs):
        state.parse_calls.append((name, kwargs))
        return SimpleNamespace(scene=SimpleNamespace(num_envs=1), sim=SimpleNamespace(device='cpu', use_fabric=True))
    def make(name, cfg):
        assert state.launched and name == 'OHT-test'
        state.cfg = cfg
        return SimpleNamespace(unwrapped=env)
    def modules(imports):
        assert state.launched and len(sys.argv) == 1
        state.imports = imports
        return SimpleNamespace(make=make), torch, parse
    monkeypatch.setattr(native, 'launch_application', launch)
    monkeypatch.setattr(native, 'simulation_modules', modules)
    return state


@pytest.mark.parametrize('mode', ['eef', 'relative_eef', 'joint', 'relative_joint'])
def test_builtin_default_launches_and_runs_http(fixture, replay_fixture, tmp_path, monkeypatch, mode):
    tensors, _, raw = raw_sample(replay_fixture.replay, fixture.policy)
    fixture.policy.action_config = action_options({'mode': mode})
    original = fixture.policy.act
    def act(*args):
        original(*args)
        return (np.array([.2, 0, .5, 0, 0, 0, 1, 1], np.float32) if mode == 'eef' else
                np.array([.01, 0, 0, 0, 0, 0, 1], np.float32))
    fixture.policy.act = act
    env = environment(mode, raw['task'])
    env.action_manager.get_term('arm_action').cfg = SimpleNamespace(body_name='tool', position_scale=1., orientation_scale=1.)
    env.scene.sensors = sensors_for(tensors, fixture.policy.contract['data_config']['cameras'])
    state = fake_simulator(monkeypatch, env)
    previous_argv = list(sys.argv)
    output = tmp_path / 'standalone-run'
    with TestClient(fixture.app()) as server:
        monkeypatch.setattr(runner, 'BridgeVLAClient',
                            lambda *args: BridgeVLAClient('http://testserver', opener=Opener(server)))
        assert runner.main(['--env-id', 'OHT-test', '--ee-link', 'tool', '--task', raw['task'], '--episodes', '1',
                            '--execution-mode', 'streaming', '--max-steps', '2', '--output', str(output), '--headless']) == 0
    assert len(fixture.policy.calls) == env.common_step_counter == 2
    assert env.closed and state.app_closed and sys.argv == previous_argv
    metadata = json.loads((output / 'run.json').read_text(encoding='utf-8'))
    assert metadata['runtime_factory'] == 'finetune.OHT.runtime.isaaclab_factory:create_runtime'
    assert metadata['reference_client'] is None and metadata['camera_source'] == 'sensors'
    assert metadata['ee_link'] == 'tool'
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    assert summary['episodes'][0]['stop_reason'] == 'max_steps'
    assert summary['success_rate'] is None
    episode = json.loads((output / 'episode_0000.json').read_text(encoding='utf-8'))
    assert 'gripper_position' in episode['actions'][0]['measured_after']
    assert episode['actions'][0]['errors_after']


@pytest.mark.parametrize('outcome', ['success', 'closed', 'late_closed', 'timeout'])
def test_policy_wait_keeps_gui_responsive_without_advancing_physics(
        fixture, replay_fixture, tmp_path, monkeypatch, outcome):
    tensors, _, raw = raw_sample(replay_fixture.replay, fixture.policy)
    env = environment('eef', raw['task'])
    env.action_manager.get_term('arm_action').cfg.body_name = 'tool'
    cameras = sensors_for(tensors, fixture.policy.contract['data_config']['cameras'])
    main_thread = threading.get_ident()
    state = SimpleNamespace(running=True, closed=False, phase=None, resets=0, predictions=0,
                            request_threads=[], rendered_phases=[])
    release = threading.Event()
    finished = threading.Event()
    app = SimpleNamespace(is_running=lambda: state.running,
                          close=lambda: setattr(state, 'closed', True))
    runtime = IsaacLabRuntime(env, cameras, ee_link='tool', quaternion_order='wxyz',
                              simulation_app=app, owns_app=True)

    def render():
        assert threading.get_ident() == main_thread
        if state.phase is not None:
            assert env.common_step_counter == 0 and not env.commands
            state.rendered_phases.append(state.phase)
            if outcome == 'closed' and state.phase == 'predict':
                state.running = False
            release.set()

    env.sim.render = render
    env.sim.has_gui = lambda: True
    monkeypatch.setattr(runner, '_factory', lambda name: lambda **kwargs: runtime)

    class DelayedClient(BridgeVLAClient):
        def wait_for_gui(self, phase):
            state.request_threads.append(threading.get_ident())
            assert threading.get_ident() != main_thread
            release.clear()
            state.phase = phase
            assert release.wait(3), 'Main thread did not refresh the GUI'

        def reset(self, episode=None):
            state.resets += 1
            self.wait_for_gui('reset')
            return super().reset(episode)

        def predict(self, *args, **kwargs):
            state.predictions += 1
            try:
                self.wait_for_gui('predict')
                if outcome == 'timeout':
                    raise TimeoutError('request completion is unknown')
                return super().predict(*args, **kwargs)
            finally:
                state.phase = None
                finished.set()

    class CloseAfterReply(PolicyWorker):
        def result(self, *args, **kwargs):
            result = super().result(*args, **kwargs)
            if isinstance(result, tuple):
                state.running = False
            return result

    if outcome == 'late_closed':
        monkeypatch.setattr(runner, 'PolicyWorker', CloseAfterReply)
    output = tmp_path / 'responsive-run'
    with TestClient(fixture.app()) as server:
        monkeypatch.setattr(runner, 'BridgeVLAClient',
                            lambda *args: DelayedClient('http://testserver', opener=Opener(server)))
        try:
            result = runner.main(['--task', raw['task'], '--episodes', '2' if outcome != 'success' else '1',
                                  '--execution-mode', 'streaming', '--max-steps', '1', '--output', str(output)])
            assert result == (0 if outcome == 'success' else 1)
        finally:
            release.set()
            assert finished.wait(3)
    assert state.closed and env.closed
    assert state.resets == state.predictions == 1
    assert len(set(state.request_threads)) == 1
    assert {'reset', 'predict'} <= set(state.rendered_phases)
    episode = json.loads((output / 'episode_0000.json').read_text(encoding='utf-8'))
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    if outcome == 'success':
        assert env.common_step_counter == len(env.commands) == 1
        action = episode['actions'][0]
        timing = action['client_timing_ms']
        assert set(timing) == {'capture', 'pack', 'serialize', 'http', 'predict', 'request', 'execute', 'total'}
        assert all(np.isfinite(value) and value >= 0 for value in timing.values())
        assert timing['total'] >= timing['capture'] + timing['request'] + timing['execute']
        assert timing['predict'] >= timing['pack'] + timing['serialize'] + timing['http']
        assert 'policy' in action['timing_ms'] and 'capture' not in action['timing_ms']
        metadata = json.loads((output / 'run.json').read_text(encoding='utf-8'))
        assert metadata['policy_wait'] == 'background_http_render_only'
    else:
        assert not env.commands and env.common_step_counter == 0
        assert not episode['actions'] and episode['stop_reason'] == 'error'
        assert summary['attempted'] == 1 and summary['unattempted'] == 1


def test_camera_snapshot_owns_calibration_and_pose_while_waiting(fixture, replay_fixture):
    tensors, _, raw = raw_sample(replay_fixture.replay, fixture.policy)
    env = environment('eef', raw['task'])
    env.action_manager.get_term('arm_action').cfg.body_name = 'tool'
    cameras = sensors_for(tensors, fixture.policy.contract['data_config']['cameras'])
    runtime = IsaacLabRuntime(env, cameras, ee_link='tool', quaternion_order='wxyz', simulation_app=None)
    client = SimpleNamespace(action_config=fixture.policy.action_config,
                             info={'raw_observation': raw_schema(fixture.policy)})
    bridge = IsaacLabBridge(runtime, env, client, raw['task'], raw['data_profile'], cameras)
    observed = bridge.capture()
    expected = {key: value.copy() for key, value in observed.items()}
    for camera in cameras.values():
        camera.data.intrinsic_matrices[...] = 0
        camera.data.pos_w[...] = 99
        camera.data.quat_w_opengl[...] = 0
        camera.data.output['rgb'][...] = 0
        camera.data.output['distance_to_image_plane'][...] = 99
    env.scene['robot'].data.joint_pos[:] = 99
    env.scene['robot'].data.body_pos_w[:] = 99
    for key, value in observed.items():
        np.testing.assert_array_equal(value, expected[key], err_msg=key)


def test_dry_run_saves_control_contract_without_predictions_or_actions(fixture, replay_fixture, tmp_path, monkeypatch):
    tensors, _, raw = raw_sample(replay_fixture.replay, fixture.policy)
    env = environment('eef', raw['task'])
    env.action_manager.get_term('arm_action').cfg.body_name = 'tool'
    env.scene.sensors = sensors_for(tensors, fixture.policy.contract['data_config']['cameras'])
    state = fake_simulator(monkeypatch, env)
    output = tmp_path / 'dry-run'
    with TestClient(fixture.app()) as server:
        monkeypatch.setattr(runner, 'BridgeVLAClient',
                            lambda *args: BridgeVLAClient('http://testserver', opener=Opener(server)))
        assert runner.main(['--env-id', 'OHT-test', '--task', raw['task'], '--dry-run', '--output', str(output), '--headless']) == 0
        assert server.get('/health').json()['episode_id'] is None
    assert not env.commands and not fixture.policy.calls and env.closed and state.app_closed
    report = json.loads((output / 'control_diagnostics.json').read_text(encoding='utf-8'))
    assert report['action_spec']['gripper'] == 'open01'
    assert report['ee_pose_source'] == 'controller' and report['gripper']['open_command'] == 1.
    assert report['gripper']['training_physical_endpoints'] is not None


def test_gripper_calibration_direction_mismatch_stops_before_control(fixture, replay_fixture, tmp_path, monkeypatch):
    tensors, _, raw = raw_sample(replay_fixture.replay, fixture.policy)
    env = environment('eef', raw['task'])
    env.action_manager.get_term('arm_action').cfg.body_name = 'tool'
    env.scene.sensors = sensors_for(tensors, fixture.policy.contract['data_config']['cameras'])
    state = fake_simulator(monkeypatch, env)
    original_report = runner.control_report
    def report(bridge):
        result = original_report(bridge)
        result['gripper'].update(verified=True, calibration_direction_matches=False)
        return result
    monkeypatch.setattr(runner, 'control_report', report)
    output = tmp_path / 'bad-gripper-calibration'
    with TestClient(fixture.app()) as server:
        monkeypatch.setattr(runner, 'BridgeVLAClient',
                            lambda *args: BridgeVLAClient('http://testserver', opener=Opener(server)))
        with pytest.raises(ValueError, match='physical direction disagrees'):
            runner.main(['--env-id', 'OHT-test', '--task', raw['task'], '--output', str(output), '--headless'])
        assert server.get('/health').json()['episode_id'] is None
    assert (output / 'control_diagnostics.json').is_file()
    assert not env.commands and not fixture.policy.calls and env.closed and state.app_closed


def test_explicit_scene_configuration_and_cleanup_on_camera_failure(tmp_path, monkeypatch):
    env = environment('eef')
    env.action_manager.get_term('arm_action').cfg.body_name = 'tool'
    env.scene.sensors = {}
    state = fake_simulator(monkeypatch, env)
    expected = SimpleNamespace(scene=SimpleNamespace(num_envs=12), sim=SimpleNamespace(device='old', use_fabric=True))
    monkeypatch.setattr(scenes, 'factory', lambda value: lambda: expected)
    args = native_arguments(env_config='oht:V4Config', camera_source='sensors', disable_fabric=True)
    before = list(sys.argv)
    with pytest.raises(ValueError, match='Missing registered Camera'):
        native.create_runtime(args=args, model_info={'cameras': ['wrist'], 'action_spec': {'mode': 'eef'}})
    assert state.cfg is expected and not state.parse_calls
    assert expected.scene.num_envs == 1 and expected.sim.device == 'cpu' and not expected.sim.use_fabric
    assert state.cfg.seed == 42 and state.imports == ['project.tasks']
    assert env.closed and state.app_closed and sys.argv == before


def test_wrong_controller_rejected_before_camera_creation(monkeypatch):
    env = environment('eef')
    state = fake_simulator(monkeypatch, env)
    monkeypatch.setattr(native, 'build_cameras', lambda *args: pytest.fail('Unexpected camera creation'))
    with pytest.raises(ValueError, match='JointPositionAction environment'):
        native.create_runtime(args=native_arguments(), model_info={'action_spec': {'mode': 'joint'}})
    assert env.closed and state.app_closed
    with pytest.raises(ValueError, match='--ee-link'):
        native.ee_link(environment('joint'), None)


def test_application_camera_enable_and_strict_argument_parser(monkeypatch):
    app = object()
    seen = []
    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument('--headless', action='store_true')
            parser.add_argument('--device', default='cuda:0')
        def __init__(self, options):
            seen.append(options)
            self.app = app
    monkeypatch.setitem(sys.modules, 'isaaclab', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'isaaclab.app', SimpleNamespace(AppLauncher=AppLauncher))
    result, options = native.launch_application(['--headless', '--device', 'cpu'])
    assert result is app and options.headless and options.enable_cameras and options.device == 'cpu'
    with pytest.raises(SystemExit):
        native.launch_application(['--unknown-simulator-flag'])
    assert len(seen) == 1


def test_camera_binding_is_unambiguous():
    paths = ['/World/Robot/WRIST', '/World/left/GLOBAL_LEFT', '/World/other/GLOBAL_LEFT']
    assert select_camera_paths(paths, {'wrist': 'WRIST'}) == {'wrist': paths[0]}
    with pytest.raises(ValueError, match='exact USD path'):
        select_camera_paths(paths, {'global_left': 'GLOBAL_LEFT'})
    assert select_camera_paths(paths, {'global_left': paths[1]})['global_left'] == paths[1]
    with pytest.raises(ValueError, match='distinct USD'):
        select_camera_paths(paths, {'a': 'WRIST', 'b': paths[0]})


def test_projection_and_usd_units_match_metric_geometry():
    projection = np.eye(4)
    projection[0, 0], projection[1, 1] = 2., 3.
    projection[0, 2], projection[1, 2] = .2, -.1
    params = dict(cameraProjection=projection, renderProductResolution=[640, 480], cameraModel='pinhole')
    np.testing.assert_allclose(projection_intrinsics(params, 640, 480), [[640, 0, 256], [0, 720, 216], [0, 0, 1]])
    with pytest.raises(ValueError, match='resolution'):
        projection_intrinsics(params, 224, 224)
    with pytest.raises(ValueError, match='pinhole'):
        projection_intrinsics({**params, 'cameraModel': 'fisheye'}, 640, 480)
    matrix = np.diag([2., 2., 2., 1.])
    matrix[3, :3] = [100., 200., 300.]
    Usd = SimpleNamespace(TimeCode=SimpleNamespace(Default=lambda: None))
    geom = SimpleNamespace(Xformable=lambda prim: SimpleNamespace(ComputeLocalToWorldTransform=lambda time: matrix))
    value = usd_rigid_matrix(None, Usd, geom, .01)
    np.testing.assert_allclose(value[:3, :3], np.eye(3))
    np.testing.assert_allclose(value[:3, 3], [1., 2., 3.])


def test_wrist_pose_uses_live_body_and_snapshot_has_no_gt():
    env = environment('eef')
    env.scene.env_origins = torch.tensor([[1., 0., 0.]])
    runtime = ReplicatorRuntime(env, SimpleNamespace(cameras={}), ee_link='tool', quaternion_order='wxyz',
                                 simulation_app=None, owns_app=False)
    camera = object.__new__(ReplicatorCamera)
    camera.body_index = 0
    camera.body_from_camera = np.eye(4)
    camera.body_from_camera[:3, 3] = [.1, 0, 0]
    # No USD/view-transform access is permitted for a moving robot-mounted camera.
    camera.usd_matrix = lambda prim: pytest.fail('Read stale USD wrist pose')
    env.scene['robot'].data.body_pos_w[0, 0] = torch.tensor([2., 1., .5])
    q = Rotation.from_euler('z', 90, degrees=True).as_quat()[[3, 0, 1, 2]]
    env.scene['robot'].data.body_quat_w[0, 0] = torch.tensor(q, dtype=torch.float32)
    np.testing.assert_allclose(camera.pose(runtime)[:3], [1., 1.1, .5], atol=1e-6)
    np.testing.assert_allclose(camera.pose(runtime)[3:], q, atol=1e-6)


def test_render_products_are_cleaned_when_annotator_attach_fails():
    # Construction failure must release every partially attached annotator and product.
    calls = []
    prim = SimpleNamespace(GetParent=lambda: SimpleNamespace(IsValid=lambda: False), GetPath=lambda: '/World/GLOBAL_LEFT')
    geom = SimpleNamespace(Camera=lambda prim: SimpleNamespace(GetProjectionAttr=lambda: SimpleNamespace(Get=lambda: 'perspective')))
    def annotator(kind):
        def attach(product):
            if kind == 'distance_to_image_plane':
                raise RuntimeError('attachment failed')
        return SimpleNamespace(attach=attach, detach=lambda product: calls.append(kind))
    product = SimpleNamespace(destroy=lambda: calls.append('destroy'))
    rep = SimpleNamespace(create=SimpleNamespace(render_product=lambda *args: product),
                          AnnotatorRegistry=SimpleNamespace(get_annotator=annotator))
    with pytest.raises(RuntimeError, match='attachment failed'):
        ReplicatorCamera('global_left', prim, environment(), rep, None, geom, 1., 3, 2)
    assert calls == ['rgb', 'distance_to_image_plane', 'destroy']


@pytest.mark.parametrize('failure', [None, 'invalid_node', 'other_type_error', 'destroy'])
def test_multicamera_cleanup_detaches_all_before_destroy_and_is_idempotent(failure, capsys):
    manager = object.__new__(ReplicatorCameras)
    manager.cameras = {}
    events, products = [], []
    for index in range(5):
        product = SimpleNamespace(destroyed=False)
        products.append(product)
        def destroy(index=index, product=product):
            events.append(('destroy', index))
            product.destroyed = True
            if failure == 'destroy' and index == 0:
                raise RuntimeError('render product destroy failed')
        product.destroy = destroy
        camera = object.__new__(ReplicatorCamera)
        camera.name, camera.product, camera.annotators = f'camera{index}', product, {}
        for kind in ('rgb', 'depth', 'params'):
            def detach(value, index=index, kind=kind, product=product):
                events.append(('detach', index, kind))
                assert value is product
                assert all(not item.destroyed for item in products), 'Destroyed a shared graph producer too early'
                if index > 0 and failure == 'invalid_node':
                    raise TypeError('Invalid NodeObj object in Py_Node in getAttributes')
                if index == 0 and kind == 'rgb' and failure == 'other_type_error':
                    raise TypeError('unexpected annotator API mismatch')
            camera.annotators[kind] = SimpleNamespace(detach=detach)
        manager.cameras[camera.name] = camera
    manager.close()
    assert [event[0] for event in events] == ['detach'] * 15 + ['destroy'] * 5
    assert all(product.destroyed for product in products)
    output = capsys.readouterr().out
    if failure == 'invalid_node':
        assert output.count('already-invalid OmniGraph') == 1 and '12' in output
        assert 'camera cleanup warning' not in output
    elif failure == 'other_type_error':
        assert 'camera cleanup warning' in output and 'unexpected annotator API mismatch' in output
        assert 'already-invalid' not in output
    elif failure == 'destroy':
        assert 'camera cleanup warning' in output and 'render product destroy failed' in output
    else:
        assert not output
    manager.close()
    assert len(events) == 20 and capsys.readouterr().out == ''


def test_usd_capture_through_http_without_reference_script(fixture, replay_fixture, tmp_path, monkeypatch):
    tensors, _, raw = raw_sample(replay_fixture.replay, fixture.policy)
    names = list(fixture.policy.contract['data_config']['cameras'])
    env = environment('eef', raw['task'])
    env.action_manager.get_term('arm_action').cfg.body_name = 'tool'
    fake_simulator(monkeypatch, env)
    fixture.policy.act = lambda *args: (fixture.policy.calls.append(args) or
                                       np.array([.2, 0., .5, 0., 0., 0., 1., 1.], np.float32))
    height, width = tensors[names[0] + '_rgb'].shape[:2]
    class Prim:
        def __init__(self, path, matrix=None, parent=None):
            self.path, self.matrix, self.parent = path, matrix, parent
        def IsValid(self):
            return bool(self.path)
        def GetPath(self):
            return self.path
        def GetName(self):
            return self.path.rsplit('/', 1)[-1]
        def IsA(self, kind):
            return self.path in data_by_path
        def GetParent(self):
            return self.parent or Prim('')
    body_matrix = np.eye(4)
    body_matrix[:3, 3] = [.1, 0., .5]
    body = Prim('/World/envs/env_0/Robot/tool', body_matrix)
    prims, data_by_path = {}, {}
    for name in names:
        pose = tensors[name + '_pose']
        matrix = np.eye(4)
        matrix[:3, :3] = Rotation.from_quat(pose[3:][[1, 2, 3, 0]]).as_matrix()
        matrix[:3, 3] = pose[:3]
        path = (body.path if name == 'wrist' else '/World') + '/' + name.upper()
        prims[path] = Prim(path, matrix, body if name == 'wrist' else None)
        k = tensors[name + '_intrinsics']
        projection = np.eye(4)
        projection[0, 0], projection[1, 1] = 2 * k[0, 0] / width, 2 * k[1, 1] / height
        projection[0, 2], projection[1, 2] = 1 - 2 * k[0, 2] / width, 2 * k[1, 2] / height - 1
        data_by_path[path] = dict(rgb=tensors[name + '_rgb'], distance_to_image_plane=tensors[name + '_depth'],
                                  camera_params=dict(cameraProjection=projection, renderProductResolution=[width, height]))
    products, detached = [], []
    def product(path, size):
        assert size == (width, height)
        value = SimpleNamespace(path=path, destroyed=False)
        value.destroy = lambda: setattr(value, 'destroyed', True)
        products.append(value)
        return value
    class Annotator:
        def __init__(self, kind):
            self.kind = kind
        def attach(self, value):
            self.product = value
        def get_data(self):
            return data_by_path[self.product.path][self.kind]
        def detach(self, value):
            detached.append((value.path, self.kind))
    stage = SimpleNamespace(Traverse=lambda: iter(prims.values()), GetPrimAtPath=lambda path: prims.get(path, Prim('')))
    omni, usd, replicator, rep, pxr = [ModuleType(name) for name in ('omni', 'omni.usd', 'omni.replicator',
                                                                   'omni.replicator.core', 'pxr')]
    usd.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
    rep.create = SimpleNamespace(render_product=product)
    rep.AnnotatorRegistry = SimpleNamespace(get_annotator=Annotator)
    omni.usd, omni.replicator, replicator.core = usd, replicator, rep
    class CameraGeometry:
        def __init__(self, prim):
            self.prim = prim
        def GetProjectionAttr(self):
            return SimpleNamespace(Get=lambda: 'perspective')
    pxr.Usd = SimpleNamespace(TimeCode=SimpleNamespace(Default=lambda: None))
    pxr.UsdGeom = SimpleNamespace(Camera=CameraGeometry, GetStageMetersPerUnit=lambda stage: 1.,
                                 Xformable=lambda prim: SimpleNamespace(ComputeLocalToWorldTransform=lambda time: prim.matrix.T))
    for module in (omni, usd, replicator, rep, pxr):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    output = tmp_path / 'usd-run'
    with TestClient(fixture.app()) as server:
        monkeypatch.setattr(runner, 'BridgeVLAClient',
                            lambda *args: BridgeVLAClient('http://testserver', opener=Opener(server)))
        assert runner.main(['--env-id', 'OHT-test', '--task', raw['task'], '--camera-source', 'usd',
                            '--camera-width', str(width), '--camera-height', str(height), '--episodes', '1',
                            '--execution-mode', 'streaming', '--max-steps', '2', '--output', str(output), '--headless']) == 0
    assert len(products) == len(names) and all(item.destroyed for item in products)
    assert len(detached) == 3 * len(names)
    assert env.common_step_counter == 2 and len(fixture.policy.calls) == 2
    observation = fixture.policy.calls[0][0]
    for name in names:
        np.testing.assert_allclose(observation[name + '_camera_intrinsics'], tensors[name + '_intrinsics'], atol=1e-6)
        np.testing.assert_allclose(observation[name + '_depth'][0], tensors[name + '_depth'], atol=1e-6, equal_nan=True)
    metadata = json.loads((output / 'run.json').read_text(encoding='utf-8'))
    assert metadata['camera_source'] == 'usd' and metadata['reference_client'] is None


@pytest.mark.parametrize('copied', [False, True])
def test_launcher_help_from_isaaclab_directory(tmp_path, copied):
    root = Path(__file__).resolve().parents[1]
    launcher = root / 'tools' / 'oht_bridgevla_client.py'
    extra = []
    if copied:
        target = tmp_path / 'oht-bridgevla-client.py'
        target.write_bytes(launcher.read_bytes())
        launcher = target
        extra = ['--bridgevla-root', str(root)]
    result = subprocess.run([sys.executable, '-X', 'utf8', str(launcher), *extra, '--help'], cwd=tmp_path,
                            capture_output=True, text=True, encoding='utf-8', timeout=30)
    assert result.returncode == 0, result.stderr
    assert '--bridgevla-root' in result.stdout and '--isaaclab-root' in result.stdout
    assert '--camera-source' in result.stdout and '--env-config' in result.stdout
