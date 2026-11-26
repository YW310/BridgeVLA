from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from finetune.OHT.runtime.stable_osc import StableOSC, pose_error_wxyz
from finetune.OHT.runtime.velocity_tracking import VelocityTargetOSC
from finetune.OHT.runtime.chunk_executor import ChunkExecutor
from finetune.OHT.runtime.chunk_protocol import chunk_spec
from tests.test_oht_chunk_pipeline import fake_bridge


def controller():
    return SimpleNamespace(
        cfg=SimpleNamespace(target_types=['pose_rel'], inertial_dynamics_decoupling=True,
            partial_inertial_dynamics_decoupling=False, nullspace_control='none', gravity_compensation=True),
        desired_ee_pose_b=torch.tensor([[.01, .02, -.01, 1., 0, 0, 0]], dtype=torch.float64),
        desired_ee_wrench_b=None,
        _motion_p_gains_b=torch.eye(6, dtype=torch.float64)[None]*250,
        _motion_d_gains_b=torch.eye(6, dtype=torch.float64)[None]*32,
        _selection_matrix_motion_b=torch.eye(6, dtype=torch.float64)[None])


def inputs(jac=None):
    return dict(jacobian_b=torch.eye(6, dtype=torch.float64)[None] if jac is None else jac,
        mass_matrix=torch.diag(torch.arange(1., 7., dtype=torch.float64))[None],
        current_ee_pose_b=torch.tensor([[0., 0, 0, 1, 0, 0, 0]], dtype=torch.float64),
        current_ee_vel_b=torch.zeros(1, 6, dtype=torch.float64),
        gravity=torch.tensor([[1., 2, 3, 4, 5, 6]], dtype=torch.float64))


def test_pose_error_matches_spatial_rotation_and_quaternion_sign_invariance():
    a = Rotation.random(32, random_state=3)
    b = Rotation.random(32, random_state=5)
    current = torch.tensor(np.c_[np.zeros((32, 3)), a.as_quat()[:, [3, 0, 1, 2]]])
    desired = torch.tensor(np.c_[np.ones((32, 3)), b.as_quat()[:, [3, 0, 1, 2]]])
    expected = np.c_[np.ones((32, 3)), (b*a.inv()).as_rotvec()]
    np.testing.assert_allclose(pose_error_wxyz(current, desired), expected, atol=1e-12)
    desired[:, 3:] *= -1
    np.testing.assert_allclose(pose_error_wxyz(current, desired), expected, atol=1e-12)


def test_well_conditioned_motion_preserves_original_full_inertia_law_and_inputs():
    native = controller()
    adapter = StableOSC(native, torch.ones(6)*150)
    values = inputs()
    before = {k:v.clone() for k,v in values.items()}
    expected = (values['mass_matrix'] @ (250*pose_error_wxyz(values['current_ee_pose_b'], native.desired_ee_pose_b)).unsqueeze(-1)).squeeze(-1) + values['gravity']
    torch.testing.assert_close(adapter.compute(**values), expected)
    assert adapter.diagnostics()['inertia_regularization'] == 0
    for k,v in before.items():
        torch.testing.assert_close(values[k], v)


@pytest.mark.parametrize('minimum', [0., 1e-8, 1e-4])
def test_singular_dynamics_are_finite_and_torque_respects_physical_limits(minimum):
    native = controller()
    native.desired_ee_pose_b[:, 2] = 1.
    jac = torch.diag(torch.tensor([1., 1, minimum, 1, 1, 1], dtype=torch.float64))[None]
    adapter = StableOSC(native, torch.tensor([150., 150, 150, 28, 28, 28]))
    torque = adapter.compute(**inputs(jac))
    assert torch.isfinite(torque).all()
    assert (torque.abs() <= adapter.effort_limits + 1e-8).all()
    assert adapter.diagnostics()['inertia_regularization'] > 0
    assert adapter.diagnostics()['regularized_inertia_condition'] <= 1e4 * (1+1e-10)


def test_motion_torque_uses_common_headroom_scale_with_gravity_preserved():
    native = controller()
    native.desired_ee_pose_b[:, :3] = 10.
    adapter = StableOSC(native, torch.ones(6)*28)
    values = inputs()
    torque = adapter.compute(**values)
    motion = (values['mass_matrix'] @ (250*pose_error_wxyz(values['current_ee_pose_b'], native.desired_ee_pose_b)).unsqueeze(-1)).squeeze(-1)
    scale = adapter.diagnostics()['motion_torque_scale']
    assert 0 < scale < 1
    torch.testing.assert_close(torque-values['gravity'], motion*scale)
    assert torque.abs().max() <= 28


def test_zero_task_torque_at_a_gravity_limit_does_not_block_other_joints():
    native = controller()
    values = inputs()
    values['gravity'][0, -1] = 28.
    adapter = StableOSC(native, torch.ones(6)*28)
    torque = adapter.compute(**values)
    assert adapter.diagnostics()['motion_torque_scale'] == 1
    assert torque[0, 0] > values['gravity'][0, 0]
    assert torque[0, -1] == 28


def test_velocity_reference_composes_with_regularization_without_modifying_measurement():
    native = controller()
    stable = StableOSC(native, torch.ones(6)*150)
    velocity = VelocityTargetOSC(stable)
    velocity.target_velocity_b = torch.ones(1, 6)*.01
    values = inputs()
    expected = stable.compute(**dict(values, current_ee_vel_b=values['current_ee_vel_b']-.01))
    torch.testing.assert_close(velocity.compute(**values), expected)
    assert not values['current_ee_vel_b'].any()
    values['current_ee_vel_b'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'):
        velocity.compute(**values)


def test_stall_is_latched_across_reanchored_chunks_and_cleared_by_episode_reset():
    bridge, sent = fake_bridge(start_pose=[0, 0, 0, 0, 0, 0, 1])
    def blocked_step(target, opening, callback):
        sent.append(target.copy())
        return [target.tolist()]
    bridge.step_target = blocked_step
    spec = chunk_spec(dict(horizon=4, execute_horizon=4), dict(chunk=dict(action_dt=.01)))
    executor = ChunkExecutor(bridge, spec, stall_timeout=.12)
    targets = np.tile([.004, 0, 0, 0, 0, 0, 1, 1], (4, 1))
    for _ in range(3):
        executor.execute(targets, {})
    assert executor.last_result['status'] == 'blocked'
    assert executor.last_result['stall']['weak_response_s'] == pytest.approx(.12)
    count = len(sent)
    executor.execute(targets, {})
    assert executor.last_result['status'] == 'blocked' and len(sent) == count
    executor.reset()
    executor.execute(targets, {})
    assert executor.last_result['completed'] and len(sent) == count+4


def test_stationary_target_near_actual_does_not_trigger_stall():
    bridge, _ = fake_bridge(start_pose=[0, 0, 0, 0, 0, 0, 1])
    spec = chunk_spec(dict(horizon=4, execute_horizon=4), dict(chunk=dict(action_dt=.01)))
    executor = ChunkExecutor(bridge, spec, stall_timeout=.04)
    executor.execute(np.tile([0, 0, 0, 0, 0, 0, 1, 1], (4, 1)), {})
    assert executor.last_result['completed']
