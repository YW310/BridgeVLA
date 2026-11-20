"""Moving-target velocity feedback for IsaacLab's stationary-target OSC.

The installed OSC computes Kp * pose_error - Kd * measured_velocity. This
adapter supplies the relative velocity (measured - reference) to that same
computation, giving Kp * pose_error + Kd * (reference - measured). It does
not change pose commands, controller gains, measured tensors or torque limits.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def chunk_target_twists(actions, action_dt, max_linear=1.5, max_angular=3.):
    """World-frame TCP twists from adjacent predicted poses, never tracking errors.

    Each future target uses its forward interval; the last uses its backward
    interval. A one-target chunk has zero reference velocity. Quaternion signs
    do not affect spatial angular velocity.
    """
    actions = np.asarray(actions, dtype=float)
    if (actions.ndim != 2 or actions.shape[1] != 8 or not len(actions) or
            not np.isfinite(actions).all()):
        raise ValueError('Velocity tracking requires finite [K,8] actions')
    if not np.isfinite([action_dt, max_linear, max_angular]).all() or min(action_dt, max_linear, max_angular) <= 0:
        raise ValueError('Velocity tracking requires positive finite dt and speed limits')
    twists = np.zeros((len(actions), 6))
    if len(actions) > 1:
        rotations = Rotation.from_quat(actions[:, 3:7])
        twists[:-1, :3] = np.diff(actions[:, :3], axis=0) / action_dt
        twists[:-1, 3:] = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / action_dt
        twists[-1] = twists[-2]
    raw = twists.copy()
    for part, limit in ((slice(0, 3), max_linear), (slice(3, 6), max_angular)):
        norm = np.linalg.norm(twists[:, part], axis=1, keepdims=True)
        twists[:, part] *= np.minimum(1., limit / np.maximum(norm, 1e-12))
    return twists, np.any(np.abs(twists - raw) > 1e-10, axis=1)


def control_twist_in_base(twist_world, tcp_pose, tcp_to_control, world_from_base):
    """Shift TCP twist to the OSC point, then express it in fixed robot-root axes."""
    twist = np.asarray(twist_world, dtype=float)
    if twist.shape != (6,) or not np.isfinite(twist).all():
        raise ValueError('Expected a finite world TCP twist [linear, angular]')
    lever = Rotation.from_quat(tcp_pose[3:7]).apply(np.asarray(tcp_to_control)[:3, 3])
    base_rotation = np.asarray(world_from_base)[:3, :3].T
    return np.r_[base_rotation @ (twist[:3] + np.cross(twist[3:], lever)),
                 base_rotation @ twist[3:]]


class VelocityTargetOSC:
    """Delegate OSC state while adding a velocity reference during one env.step.

    Integration is explicitly restricted by IsaacLabBridge to the native OSC
    whose compute() uses current_ee_vel_b only in its stationary velocity error.
    The original measured velocity is never modified in place.
    """
    def __init__(self, controller):
        self.controller = controller
        self.target_velocity_b = None

    def __getattr__(self, name):
        return getattr(self.controller, name)

    def compute(self, *, current_ee_vel_b=None, **kwargs):
        velocity = current_ee_vel_b
        if self.target_velocity_b is not None:
            if velocity is None:
                raise ValueError('Velocity-target OSC requires measured EE velocity')
            velocity = velocity - self.target_velocity_b.to(velocity)
        return self.controller.compute(current_ee_vel_b=velocity, **kwargs)
