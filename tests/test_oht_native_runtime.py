"""BridgeVLA-owned joint control, task isolation and native-runtime HTTP integration."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from tests.test_oht_migration import replay_fixture  # noqa: F401
from tests.test_oht_vla_server import fixture  # noqa: F401
from tests.test_oht_isaaclab_http import raw_sample, Opener
from finetune.OHT import isaaclab_client as runner
from finetune.OHT.action_modes import action_options
from finetune.OHT.data.common import GOALS
from finetune.OHT.runtime.http_client import BridgeVLAClient
from finetune.OHT.runtime.isaaclab_runtime import IsaacLabRuntime
from finetune.OHT.runtime.isaaclab_bridge import IsaacLabBridge
from finetune.OHT.runtime.joint_executor import JointPositionExecutor
from finetune.OHT.runtime.task_runtime import TaskRuntime, EnvironmentSuccess, task_settings


class JointPositionAction:
    action_dim = 6
    _joint_ids = [5, 3, 1, 4, 2, 0]
    def __init__(self):
        self._scale = torch.tensor([[.2, -.5, 2., .3, 1., 1.]])
        self._offset = torch.tensor([[.1, .2, -.1, 0., .3, -.4]])
        self.raw_actions = torch.zeros(1, 6)
        self.processed_actions = self._offset.clone()
        self.clip = None
    def process_actions(self, value):
        self.raw_actions[:] = value
        self.processed_actions = value * self._scale + self._offset
        if self.clip is not None:
            self.processed_actions.clamp_(-self.clip, self.clip)


class Scene(dict):
    env_origins = torch.zeros(1, 3)
    def write_data_to_sim(self):
        pass
    def update(self, dt):
        pass


def environment(mode='joint', task='assemble_left'):
    data = SimpleNamespace(joint_pos=torch.zeros(1, 8), root_pos_w=torch.zeros(1, 3),
                           root_quat_w=torch.tensor([[1., 0, 0, 0]]),
                           body_pos_w=torch.tensor([[[.1, 0, .5]]]),
                           body_quat_w=torch.tensor([[[1., 0, 0, 0]]]),
                           soft_joint_pos_limits=torch.tensor([[[-3., 3.]] * 8]))
    robot = SimpleNamespace(data=data, joint_names=[f'j{i}' for i in range(8)], body_names=['tool'])
    arm = (JointPositionAction() if 'joint' in mode else
           type('OperationalSpaceAction', (), {'_joint_ids': list(range(6)),
                                               'cfg': SimpleNamespace(position_scale=1., orientation_scale=1.)})())
    gripper = SimpleNamespace(_joint_ids=[6, 7])
    terms = {'arm_action': arm, 'gripper_action': gripper}
    manager = SimpleNamespace(get_term=lambda name: terms[name], active_terms=['gripper_action', 'arm_action'],
                              action_term_dim=[1, 6])
    state = {f'{name}_success': np.array([False]) for name in GOALS}
    termination = SimpleNamespace(active_terms=list(state), get_term=lambda name: state[name],
                                  get_term_cfg=lambda name: SimpleNamespace(time_out=False))
    env = SimpleNamespace(scene=Scene(robot=robot), action_manager=manager, termination_manager=termination,
                          device='cpu', num_envs=1, common_step_counter=0, step_dt=.02, closed=False,
                          factor=1., done_after=None, commands=[])
    def reset():
        env.common_step_counter = 0
        data.joint_pos.zero_()
        data.body_pos_w[:] = torch.tensor([.1, 0, .5])
        for value in state.values():
            value[:] = False
    def step(value):
        env.commands.append(value.numpy()[0].copy())
        env.common_step_counter += 1
        if 'joint' in mode:
            arm.process_actions(value[:, 1:])
            current = data.joint_pos[:, arm._joint_ids]
            data.joint_pos[:, arm._joint_ids] = current + env.factor * (arm.processed_actions - current)
        else:
            data.body_pos_w[0, 0] += value[0, 1:4]
        state[f'{task}_success'][:] = env.common_step_counter >= 2
        done = env.done_after is not None and env.common_step_counter >= env.done_after
        return None, None, np.array([done]), np.array([False]), {}
    env.step, env.reset, env.close = step, reset, lambda: setattr(env, 'closed', True)
    env.sim = SimpleNamespace(render=lambda: None)
    return env


@pytest.mark.parametrize('mode', ['joint', 'relative_joint'])
def test_joint_order_affine_and_relative_target_snapshot(mode):
    env = environment(mode)
    options = action_options({'mode': mode, 'joint_names': [f'j{i}' for i in range(6)]})
    executor = JointPositionExecutor(env, torch, options, hold_steps=3, tolerance=0)
    env.factor = .5
    runtime = SimpleNamespace(torch=torch)
    client = SimpleNamespace(action_config=options, info={'raw_observation': {'data_profiles': {
        'p': {'link_to_tcp': np.eye(4).tolist()}}}})
    bridge = IsaacLabBridge(runtime, env, client, 'assemble_left', 'p', {}, joint_executor=executor)
    action = np.r_[np.arange(1, 7) * .1, 1]
    measured = dict(ee_link_pose=np.array([0, 0, 0, 1, 0, 0, 0]), joint_positions=np.zeros(6))
    bridge.execute(action, measured)
    np.testing.assert_allclose(env.scene['robot'].data.joint_pos[0, :6], action[:6] * .875, atol=1e-6)
    np.testing.assert_allclose(bridge.last_target, action, atol=1e-6)
    # Gripper is first in this environment; arm commands use term order and inverse affine mapping.
    assert env.commands[0][0] == 1 and len(env.commands) == 3
    arm = env.action_manager.get_term('arm_action')
    np.testing.assert_allclose(env.commands[0][1:] * arm._scale.numpy()[0] + arm._offset.numpy()[0],
                               action[:6][arm._joint_ids], atol=1e-6)


@pytest.mark.parametrize('failure', ['limit', 'zero_scale', 'clip'])
def test_joint_rejects_unexecutable_targets_before_physics(failure):
    env = environment()
    executor = JointPositionExecutor(env, torch, action_options())
    target = np.r_[np.full(6, .5), 1]
    arm = env.action_manager.get_term('arm_action')
    if failure == 'limit':
        target[0] = 4
    elif failure == 'zero_scale':
        arm._scale[:] = 0
    else:
        arm.clip = .1
    with pytest.raises(ValueError):
        executor(target)
    assert not env.commands
    assert not arm.raw_actions.any()


def test_joint_stops_on_environment_termination():
    env = environment()
    env.factor, env.done_after = .2, 1
    executor = JointPositionExecutor(env, torch, action_options(), hold_steps=10, tolerance=0)
    executor(np.r_[np.full(6, .5), 1])
    assert executor.last_done and len(env.commands) == 1


def test_osc_tracks_configured_body_offset_in_root_frame():
    env = environment('eef')
    term = env.action_manager.get_term('arm_action')
    term.cfg.body_name = 'tool'
    term.cfg.body_offset = SimpleNamespace(pos=(.1, 0., 0.), rot=(1., 0., 0., 0.))
    runtime = IsaacLabRuntime(env, {}, ee_link='tool', quaternion_order='wxyz', simulation_app=None)
    client = SimpleNamespace(action_config=action_options({'mode': 'eef'}),
                             info={'raw_observation': {'data_profiles': {'p': {'link_to_tcp': np.eye(4).tolist()}}}})
    bridge = IsaacLabBridge(runtime, env, client, 'assemble_left', 'p', {},
                            max_translation=1., max_rotation=4.)
    measured = dict(ee_link_pose=np.array([.1, 0., .5, 1., 0., 0., 0.]), joint_positions=np.zeros(6))
    bridge.execute(np.array([.2, 0., .5, 0., 0., np.sqrt(.5), np.sqrt(.5), 1.]), measured)
    # Body offset rotates with the target link. The controlled point moves along Y.
    np.testing.assert_allclose(env.commands[0][1:4], [0., .1, 0.], atol=1e-6)
    term.cfg.task_frame_rel_path = '/World/custom_frame'
    with pytest.raises(ValueError, match='task frame'):
        IsaacLabBridge(runtime, env, client, 'assemble_left', 'p', {})
    term.cfg.task_frame_rel_path = None
    term.cfg.body_name = 'another_link'
    with pytest.raises(ValueError, match='body_name'):
        IsaacLabBridge(runtime, env, client, 'assemble_left', 'p', {})


@pytest.mark.parametrize('task', list(GOALS))
def test_task_reset_and_success_are_isolated(task, tmp_path):
    env = environment(task=task)
    class Asset:
        def __init__(self, x):
            self.data = SimpleNamespace(root_state_w=torch.tensor([[x, 0., 0., 1., 0., 0., 0., 1., 2., 3., 4., 5., 6.]]))
        def write_root_pose_to_sim(self, value):
            self.data.root_state_w[:, :7] = value
        def write_root_velocity_to_sim(self, value):
            self.data.root_state_w[:, 7:] = value
    dict.update(env.scene, {name: Asset(float(i)) for i, name in enumerate(GOALS)})
    path = tmp_path / 'tasks.yaml'
    # JSON is also a YAML mapping; no geometry thresholds are synthesized here.
    path.write_text(json.dumps(dict(version=1, tasks={name: dict(reset_assets=[name], success_term=f'{name}_success')
                                                    for name in GOALS})), encoding='utf-8')
    settings = task_settings(path, task)
    runtime = TaskRuntime(env, None, task, settings, score=True)
    initial = env.scene[task].data.root_state_w.clone()
    for name in GOALS:
        env.scene[name].data.root_state_w[:, 0] += 10
    runtime.initialize(episode_index=0, seed=42)
    np.testing.assert_allclose(env.scene[task].data.root_state_w[:, :7], initial[:, :7])
    assert not env.scene[task].data.root_state_w[:, 7:].any()
    for name in set(GOALS) - {task}:
        assert env.scene[name].data.root_state_w[0, 0] >= 10
    runtime.reset(0)
    assert runtime.update(step=1, executed_gripper_cmd=1) is False
    env.step(torch.zeros(1, 7))
    env.step(torch.zeros(1, 7))
    assert runtime.update(step=2, executed_gripper_cmd=1) is True
    runtime.reset(1)
    assert runtime.success is False


@pytest.mark.parametrize('bad', ['missing', 'timeout', 'nonboolean'])
def test_scoring_does_not_infer_success_from_done_or_numbers(bad):
    env = environment()
    if bad == 'missing':
        with pytest.raises(ValueError):
            EnvironmentSuccess(env, 'missing')
    elif bad == 'timeout':
        env.termination_manager.get_term_cfg = lambda name: SimpleNamespace(time_out=True)
        with pytest.raises(ValueError):
            EnvironmentSuccess(env, 'assemble_left_success')
    else:
        scorer = EnvironmentSuccess(env, 'assemble_left_success')
        env.termination_manager.get_term = lambda name: np.array([1.])
        with pytest.raises(ValueError):
            scorer.update(step=1, executed_gripper_cmd=1)


@pytest.mark.parametrize('mode', ['eef', 'relative_eef', 'joint', 'relative_joint'])
@pytest.mark.parametrize('task', list(GOALS))
def test_native_factory_all_tasks_and_actions_over_http(fixture, replay_fixture, tmp_path, monkeypatch, mode, task):
    tensors, _, _ = raw_sample(replay_fixture.replay, fixture.policy)
    fixture.policy.action_config = action_options({'mode': mode})
    original_act = fixture.policy.act
    def act(*args):
        original_act(*args)
        return (np.array([.2, 0, .5, 0, 0, 0, 1, 1], np.float32) if mode == 'eef' else
                np.array([.01, 0, 0, 0, 0, 0, 1], np.float32))
    fixture.policy.act = act
    env = environment(mode, task)
    env.done_after = 2
    class Camera:
        def __init__(self, name):
            self.data = SimpleNamespace(output={'rgb': tensors[name + '_rgb'][None],
                                                'distance_to_image_plane': tensors[name + '_depth'][None, ..., None]},
                                        pos_w=tensors[name + '_pose'][None, :3],
                                        quat_w_opengl=tensors[name + '_pose'][None, 3:],
                                        intrinsic_matrices=tensors[name + '_intrinsics'][None])
        def update(self, *args, **kwargs):
            assert kwargs['force_recompute'] is True
    cameras = {name: Camera(name) for name in fixture.policy.contract['data_config']['cameras']}
    app = SimpleNamespace(is_running=lambda: True, close=lambda: None)
    runtime = IsaacLabRuntime(env, cameras, ee_link='tool', quaternion_order='wxyz', simulation_app=app)
    def create(*, args, model_info):
        assert args.task == task and model_info['action_spec']['mode'] == mode
        return runtime
    monkeypatch.setattr(runner, '_factory', lambda name: create)
    with TestClient(fixture.app()) as server:
        monkeypatch.setattr(runner, 'BridgeVLAClient',
                            lambda *args: BridgeVLAClient('http://testserver', opener=Opener(server)))
        output = tmp_path / 'client-run'
        assert runner.main(['--runtime-factory', 'project:create', '--task', task, '--episodes', '1',
                            '--max-steps', '5', '--output', str(output), '--score', 'task',
                            '--success-term', f'{task}_success']) == 0
        summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
        metadata = json.loads((output / 'run.json').read_text(encoding='utf-8'))
        assert summary['success_rate'] == 1. and summary['attempted'] == 1
        assert len(fixture.policy.calls) == env.common_step_counter == 2
        assert all(call[1] == GOALS[task] for call in fixture.policy.calls)
        assert metadata['runtime_factory'] == 'project:create' and metadata['reference_client'] is None
        assert metadata['instruction'] == GOALS[task] and metadata['action_spec']['mode'] == mode
        assert env.closed


@pytest.mark.parametrize('order', ['wxyz', 'xyzw'])
def test_native_pose_conventions_are_explicit(order):
    env = environment()
    camera = SimpleNamespace(data=SimpleNamespace(pos_w=np.array([[1., 2., 3.]]),
                                                  quat_w_opengl=np.array([[1., 0, 0, 0]])))
    if order == 'xyzw':
        env.scene['robot'].data.root_quat_w[:] = torch.tensor([0, 0, 0, 1])
        env.scene['robot'].data.body_quat_w[:] = torch.tensor([0, 0, 0, 1])
        camera.data.quat_w_opengl[:] = [0, 0, 0, 1]
    runtime = IsaacLabRuntime(env, {}, ee_link='tool', quaternion_order=order, simulation_app=None)
    np.testing.assert_allclose(runtime.world_from_base(), np.eye(4))
    np.testing.assert_allclose(runtime.camera_pose_ros(camera, env), [1, 2, 3, 1, 0, 0, 0])
    np.testing.assert_allclose(runtime.get_ee_pose_in_world_frame(env, env.scene['robot'])[1], [1, 0, 0, 0])
