"""Recorded controller TCP, bare body measurements and gripper command semantics."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from tests.test_oht_native_runtime import environment
from finetune.OHT.action_modes import action_options
from finetune.OHT.runtime.control_diagnostics import gripper_report
from finetune.OHT.runtime.isaaclab_runtime import IsaacLabRuntime, controller_offset
from finetune.OHT.runtime.isaaclab_bridge import IsaacLabBridge
from finetune.OHT.data.geometry import pose_matrix


@pytest.mark.parametrize('mode', ['eef', 'relative_eef'])
@pytest.mark.parametrize('order', ['wxyz', 'xyzw'])
def test_body_with_explicit_tcp_and_controller_with_identity_tcp_are_equivalent(mode, order):
    commands = []
    for source in ('body', 'controller'):
        env = environment('eef')
        cfg = env.action_manager.get_term('arm_action').cfg
        cfg.body_name = 'tool'
        rotation = Rotation.from_euler('x', 30, degrees=True).as_quat()
        cfg.body_offset = SimpleNamespace(pos=(.08, -.03, .12), rot=rotation[[3, 0, 1, 2]] if order == 'wxyz' else rotation)
        if order == 'xyzw':
            env.scene['robot'].data.body_quat_w[:] = torch.tensor([0., 0., 0., 1.])
            env.scene['robot'].data.root_quat_w[:] = torch.tensor([0., 0., 0., 1.])
        offset = controller_offset(cfg, order)
        runtime = IsaacLabRuntime(env, {}, ee_link='tool', quaternion_order=order, simulation_app=None, ee_pose_source=source)
        profile_offset = offset if source == 'body' else np.eye(4)
        client = SimpleNamespace(action_config=action_options({'mode': mode}),
                                 info={'raw_observation': {'data_profiles': {'p': {'link_to_tcp': profile_offset.tolist()}}}})
        bridge = IsaacLabBridge(runtime, env, client, 'assemble_left', 'p', {}, max_translation=1., max_rotation=1.)
        measured = dict(ee_link_pose=bridge.link_pose(), joint_positions=np.zeros(6))
        current_tcp = np.asarray(bridge.measured_state({**measured, 'gripper_position': [0.]})['tcp_pose_xyzw'])
        np.testing.assert_allclose(current_tcp[:3], [.18, -.03, .62], atol=1e-6)
        if mode == 'eef':
            action = np.r_[current_tcp[:3] + [.02, -.01, .03], current_tcp[3:], 1.]
        else:
            action = np.array([.02, -.01, .03, 0., 0., 0., 1.])
        bridge.execute(action, measured)
        commands.append(env.commands[0])
        if source == 'controller':
            np.testing.assert_allclose(bridge.control_offset, np.eye(4), atol=1e-7)
    np.testing.assert_allclose(commands[0], commands[1], atol=1e-6)
    # Fake environment puts the gripper first; translation is in arm columns 1:4.
    np.testing.assert_allclose(commands[0][1:4], [.02, -.01, .03], atol=1e-6)


def test_current_recorded_tcp_target_does_not_double_apply_controller_offset():
    env = environment('eef')
    cfg = env.action_manager.get_term('arm_action').cfg
    cfg.body_name = 'tool'
    cfg.body_offset = SimpleNamespace(pos=(0., 0., .15), rot=(1., 0., 0., 0.))
    runtime = IsaacLabRuntime(env, {}, ee_link='tool', quaternion_order='wxyz', simulation_app=None, ee_pose_source='controller')
    client = SimpleNamespace(action_config=action_options(),
                             info={'raw_observation': {'data_profiles': {'p': {'link_to_tcp': np.eye(4).tolist()}}}})
    bridge = IsaacLabBridge(runtime, env, client, 'assemble_left', 'p', {})
    measured = dict(ee_link_pose=bridge.link_pose(), joint_positions=np.zeros(6))
    np.testing.assert_allclose(measured['ee_link_pose'][:3], [.1, 0., .65], atol=1e-6)
    bridge.execute(np.array([.1, 0., .65, 0., 0., 0., 1., 1.]), measured)
    np.testing.assert_allclose(env.commands[0][1:], np.zeros(6), atol=1e-6)


@pytest.mark.parametrize('order', ['wxyz', 'xyzw'])
def test_full_tcp_to_osc_chain_with_nonidentity_origin_base_and_offsets(order):
    env = environment('eef')
    robot = env.scene['robot']
    env.scene.env_origins = torch.tensor([[10., 20., 0.]])
    root_rotation = Rotation.from_euler('xyz', [10, -20, 90], degrees=True)
    body_rotation = Rotation.from_euler('xyz', [30, 15, -40], degrees=True)
    control_rotation = Rotation.from_euler('xyz', [-15, 20, 10], degrees=True)
    def stored(rotation):
        q = rotation.as_quat()
        return q[[3, 0, 1, 2]] if order == 'wxyz' else q
    robot.data.root_pos_w[:] = torch.tensor([[12., 19., .3]])
    robot.data.root_quat_w[:] = torch.tensor(stored(root_rotation))
    robot.data.body_pos_w[:] = torch.tensor([[[12.2, 19.1, .9]]])
    robot.data.body_quat_w[:] = torch.tensor(stored(body_rotation))
    cfg = env.action_manager.get_term('arm_action').cfg
    cfg.body_name = 'tool'
    cfg.position_scale, cfg.orientation_scale = .01, .3
    cfg.body_offset = SimpleNamespace(pos=[.08, -.03, .12], rot=stored(control_rotation))
    link_to_tcp = pose_matrix([-.02, .06, .04], Rotation.from_euler('y', 25, degrees=True).as_quat())
    runtime = IsaacLabRuntime(env, {}, ee_link='tool', quaternion_order=order, simulation_app=None,
                             ee_pose_source='controller')
    client = SimpleNamespace(action_config=action_options(),
                             info={'raw_observation': {'data_profiles': {'p': {'link_to_tcp': link_to_tcp}}}})
    bridge = IsaacLabBridge(runtime, env, client, 'assemble_left', 'p', {})
    current_control = pose_matrix([2.2, -.9, .9], body_rotation.as_quat()) @ controller_offset(cfg, order)
    current_tcp = current_control @ link_to_tcp
    target_rotation = Rotation.from_rotvec([.2, -.1, .3]) * Rotation.from_matrix(current_tcp[:3, :3])
    target_tcp = pose_matrix(current_tcp[:3, 3] + [.03, -.07, .02], target_rotation.as_quat())
    target_control = target_tcp @ np.linalg.inv(link_to_tcp)
    # Independent physical residual expressed in robot-root axes.
    displacement = root_rotation.inv().apply(target_control[:3, 3] - current_control[:3, 3])
    rotvec = (root_rotation.inv() * Rotation.from_matrix(target_control[:3, :3]) *
              Rotation.from_matrix(current_control[:3, :3]).inv() * root_rotation).as_rotvec()
    measured = dict(ee_link_pose=bridge.link_pose(), joint_positions=np.zeros(6))
    bridge.execute(np.r_[target_tcp[:3, 3], target_rotation.as_quat(), 0.], measured)
    command = env.commands[0]
    assert command[0] == -1  # Live action term order puts gripper first.
    np.testing.assert_allclose(command[1:4], np.clip(displacement, -.05, .05) / .01, atol=2e-4)
    np.testing.assert_allclose(command[4:], np.clip(rotvec, -.15, .15) / .3, atol=2e-6)


@pytest.mark.parametrize('reversed_sign', [False, True])
def test_gripper_probe_separates_open01_command_and_physical_sum_without_moving(reversed_sign):
    env = environment()
    class BinaryJointPositionAction:
        _joint_ids = [6, 7]
        _open_command = torch.tensor([-.015, -.015])
        _close_command = torch.tensor([.011, .011])
        raw_actions = torch.ones(1, 1)
        processed_actions = _open_command[None].clone()
        cfg = SimpleNamespace(open_command_expr={'fingers': -.015}, close_command_expr={'fingers': .011})
        def process_actions(self, value):
            self.raw_actions[:] = value
            mask = value < 0 if not reversed_sign else value > 0
            self.processed_actions = torch.where(mask, self._close_command, self._open_command)
    gripper = BinaryJointPositionAction()
    gripper.process_actions(gripper.raw_actions.clone())
    saved_raw, saved_target = gripper.raw_actions.clone(), gripper.processed_actions.clone()
    original = env.action_manager.get_term
    env.action_manager.get_term = lambda name: gripper if name == 'gripper_action' else original(name)
    report = gripper_report(env, torch, {'open': -.03, 'close': .022})
    assert report['verified'] is (not reversed_sign)
    if not reversed_sign:
        assert report['open_physical_sum'] == pytest.approx(-.03)
        assert report['close_physical_sum'] == pytest.approx(.022)
        assert abs(report['endpoint_difference']['open']) < 1e-7
        assert report['calibration_direction_matches'] is True
    opposite = gripper_report(env, torch, {'open': .022, 'close': -.03})
    assert opposite['calibration_direction_matches'] is reversed_sign
    np.testing.assert_allclose(gripper.raw_actions, saved_raw)
    np.testing.assert_allclose(gripper.processed_actions, saved_target)
    assert not env.commands


@pytest.mark.parametrize('order', ['wxyz', 'xyzw'])
def test_registered_moving_camera_refreshes_pose_instead_of_initial_pose(order):
    env = environment('eef')
    if order == 'xyzw':
        env.scene['robot'].data.body_quat_w[:] = torch.tensor([0., 0., 0., 1.])
        env.scene['robot'].data.root_quat_w[:] = torch.tensor([0., 0., 0., 1.])
    class Camera:
        cfg = SimpleNamespace(update_latest_camera_pose=False)
        data = SimpleNamespace(pos_w=np.zeros((1, 3)), quat_w_opengl=np.zeros((1, 4)))
        def update(self, dt, force_recompute):
            assert dt == 0 and force_recompute
            if self.cfg.update_latest_camera_pose:
                self.data.pos_w[:] = env.scene['robot'].data.body_pos_w[0].numpy()
                self.data.quat_w_opengl[:] = env.scene['robot'].data.body_quat_w[0].numpy()
    camera = Camera()
    runtime = IsaacLabRuntime(env, {'wrist': camera}, ee_link='tool', quaternion_order=order, simulation_app=None)
    assert camera.cfg.update_latest_camera_pose is True
    runtime._replicator_render_tick(env)
    before = runtime.camera_pose_ros(camera, env)
    env.scene['robot'].data.body_pos_w[0, 0] += torch.tensor([.2, .1, -.1])
    q = Rotation.from_euler('xyz', [20, 30, 40], degrees=True).as_quat()
    env.scene['robot'].data.body_quat_w[0, 0] = torch.tensor(q[[3, 0, 1, 2]] if order == 'wxyz' else q)
    runtime._replicator_render_tick(env)
    after = runtime.camera_pose_ros(camera, env)
    np.testing.assert_allclose(after[:3] - before[:3], [.2, .1, -.1], atol=1e-6)
    np.testing.assert_allclose(after[3:], q[[3, 0, 1, 2]], atol=1e-6)
    assert not env.commands
