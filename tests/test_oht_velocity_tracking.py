"""Moving-target OSC math, frames, limits and chunk integration."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from finetune.OHT.runtime.velocity_tracking import (
    VelocityTargetOSC, chunk_target_twists, control_twist_in_base,
)
from finetune.OHT.runtime.chunk_executor import ChunkExecutor
from finetune.OHT.runtime.chunk_protocol import chunk_spec
from tests.test_oht_chunk_pipeline import fake_bridge


def actions_for(positions, rotations=None):
    positions = np.asarray(positions, float)
    actions = np.zeros((len(positions), 8))
    actions[:, :3] = positions
    actions[:, 3:7] = ([0, 0, 0, 1] if rotations is None else rotations.as_quat())
    actions[:, 7] = 1
    return actions


def test_twists_use_future_target_motion_and_world_rotation_with_quaternion_sign_invariance():
    r = Rotation.from_euler('z', [0, .1, .2]) * Rotation.from_euler('x', .7)
    actions = actions_for([[0, 0, 0], [.01, 0, 0], [.02, 0, 0]], r)
    twists, clipped = chunk_target_twists(actions, .1)
    np.testing.assert_allclose(twists, np.tile([.1, 0, 0, 0, 0, 1.], (3, 1)), atol=1e-12)
    assert not clipped.any()
    actions[1, 3:7] *= -1
    np.testing.assert_allclose(chunk_target_twists(actions, .1)[0], twists, atol=1e-12)
    np.testing.assert_array_equal(chunk_target_twists(actions[:1], .1)[0], np.zeros((1, 6)))


def test_speed_caps_preserve_direction_and_zero_references():
    actions = actions_for([[0, 0, 0], [3, 4, 0]], Rotation.from_euler('z', [0, 1]))
    twists, clipped = chunk_target_twists(actions, .1, max_linear=.5, max_angular=2.)
    np.testing.assert_allclose(twists, np.tile([.3, .4, 0, 0, 0, 2.], (2, 1)))
    assert clipped.all()
    twists, clipped = chunk_target_twists(np.tile(actions[0], (3, 1)), .1)
    np.testing.assert_array_equal(twists, np.zeros((3, 6)))
    assert not clipped.any()
    for dt in (0, float('nan')):
        with pytest.raises(ValueError): chunk_target_twists(actions, dt)


def test_tcp_to_controller_lever_arm_and_root_rotation():
    base = np.eye(4)
    base[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    offset = np.eye(4)
    offset[:3, 3] = [1, 0, 0]
    pose = [0, 0, 0, 0, 0, 0, 1]
    twist = control_twist_in_base([0, 0, 0, 0, 0, 2], pose, offset, base)
    np.testing.assert_allclose(twist, [2, 0, 0, 0, 0, 2], atol=1e-12)


class StationaryOSC:
    cfg = 'unchanged'

    def compute(self, *, current_ee_vel_b, pose_error=None, **kwargs):
        return 250 * pose_error - 2 * 250 ** .5 * current_ee_vel_b


def test_velocity_adapter_preserves_measurements_pose_and_stationary_behavior():
    controller = StationaryOSC()
    adapter = VelocityTargetOSC(controller)
    velocity = torch.tensor([[1., 2, 3, 4, 5, 6]])
    before = velocity.clone()
    error = torch.ones_like(velocity) * .01
    torch.testing.assert_close(adapter.compute(current_ee_vel_b=velocity, pose_error=error),
                               controller.compute(current_ee_vel_b=velocity, pose_error=error))
    adapter.target_velocity_b = torch.tensor([[.5, 0, 0, 0, 0, .2]])
    torque = adapter.compute(current_ee_vel_b=velocity, pose_error=error)
    torch.testing.assert_close(torque, 250 * error + 2 * 250 ** .5 * (adapter.target_velocity_b - velocity))
    torch.testing.assert_close(velocity, before)
    assert adapter.cfg is controller.cfg


def test_constant_velocity_plant_eliminates_stationary_target_steady_lag():
    # Test the spring-damper mechanism with physics substeps, not a kinematic echo.
    def run(feedforward):
        controller = VelocityTargetOSC(StationaryOSC())
        x, v, target_speed, dt = 0., 0., .5, .001
        for step in range(4000):
            target = target_speed * (step + 1) * dt
            controller.target_velocity_b = torch.tensor([[target_speed]]) if feedforward else None
            acceleration = controller.compute(current_ee_vel_b=torch.tensor([[v]]),
                                              pose_error=torch.tensor([[target - x]])).item()
            v += acceleration * dt
            x += v * dt
        return abs(target - x)
    stationary, moving = run(False), run(True)
    assert .05 < stationary < .07  # Kd/Kp * .5 ~= .0632 m.
    assert moving < .001


def test_executor_passes_only_predicted_motion_to_velocity_adapter_and_keeps_step_guard():
    bridge, steps = fake_bridge(start_pose=[0, 0, 0, 0, 0, 0, 1])
    passed = []
    bridge.enable_velocity_feedforward = lambda: passed.append('enabled')
    original_step = bridge.step_target
    def step(target, opening, after_step=None, *, target_twist_world):
        passed.append(target_twist_world.copy())
        return original_step(target, opening, after_step)
    bridge.step_target = step
    spec = chunk_spec(dict(horizon=4, execute_horizon=2), dict(chunk=dict(action_dt=.01)))
    executor = ChunkExecutor(bridge, spec, velocity_feedforward=True)
    actions = actions_for([[.001, 0, 0], [.002, 0, 0], [.003, 0, 0], [.004, 0, 0]])
    measured = bridge.measured_state()
    executor.execute(actions, measured)
    assert passed[0] == 'enabled' and len(passed) == 3
    np.testing.assert_allclose(passed[1:], np.tile([.1, 0, 0, 0, 0, 0], (2, 1)), atol=1e-8)
    assert executor.last_result['velocity_feedforward'] is True
    assert all('target_twist_world' in t for t in executor.last_result['tracking_samples'])
    count = len(steps)
    actions[:, 0] += 1.
    executor.execute(actions, bridge.measured_state())
    assert executor.last_result['status'] == 'step_limit' and len(steps) == count


@pytest.mark.parametrize('fail_step', [False, True])
def test_bridge_limits_velocity_reference_to_one_step_even_when_simulator_raises(fail_step):
    from finetune.OHT.runtime.isaaclab_bridge import IsaacLabBridge
    native = type('OperationalSpaceController', (StationaryOSC,),
                  {'__module__': 'isaaclab.controllers.operational_space'})()
    cfg = SimpleNamespace(position_scale=1., orientation_scale=1.)
    term = type('OperationalSpaceAction', (), {'cfg': cfg, '_osc': native})()
    robot = SimpleNamespace(is_fixed_base=True, data=SimpleNamespace(
        root_pos_w=np.zeros((1, 3)), root_quat_w=np.array([[1., 0, 0, 0]])))
    env = SimpleNamespace(scene={'robot': robot}, device='cpu',
                          action_manager=SimpleNamespace(get_term=lambda name: term))
    ref = SimpleNamespace(torch=torch, tensor_to_numpy=lambda value, dtype: np.asarray(value, dtype),
        get_ee_pose_in_world_frame=lambda env, robot: (np.zeros(3), np.array([1., 0, 0, 0])))
    client = SimpleNamespace(action_config={'mode': 'eef'}, info={
        'raw_observation': {'data_profiles': {'p': {'link_to_tcp': np.eye(4)}}}})
    bridge = IsaacLabBridge(ref, env, client, 'disassemble_left', 'p', {})
    bridge.enable_velocity_feedforward()
    seen = []
    def step(action):
        seen.append(term._osc.target_velocity_b.clone())
        if fail_step:
            raise RuntimeError('test simulator error')
        return None
    env.step = step
    target = np.array([.01, 0, 0, 0, 0, 0, 1, 1])
    if fail_step:
        with pytest.raises(RuntimeError, match='test simulator error'):
            bridge.step_target(target, 1, target_twist_world=[.1, 0, 0, 0, 0, 0])
    else:
        bridge.step_target(target, 1, target_twist_world=[.1, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(seen[0].numpy(), [[.1, 0, 0, 0, 0, 0]])
    assert term._osc.target_velocity_b is None
