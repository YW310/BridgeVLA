"""CPU mechanism experiment, NOT an IsaacLab/controller calibration.

An oracle predicts the exact next constant-velocity poses on the demonstration
path. Both schedules use the production ChunkExecutor and OSC residual mapping.
The plant is an explicitly chosen first-order position response, not robot
physics. This can test whether reanchoring is sufficient for slow progress;
it cannot establish that an actual robot has this response or diagnose contact.
"""
from types import SimpleNamespace

import numpy as np

from .data.geometry import pose_matrix
from .runtime.chunk_executor import ChunkExecutor
from .runtime.chunk_protocol import SCHEMA
from .runtime.isaaclab_bridge import osc_command


class _LagBridge:
    def __init__(self, gain, dt):
        self.gain, self.state = gain, np.array([0., 0., 0., 0., 0., 0., 1.])
        self.client = SimpleNamespace(action_config={'mode': 'eef'})
        self.env = SimpleNamespace(step_dt=dt, scene={})
        self.last_target = None
        self.last_done = self.last_stopped = False

    def measured_state(self, measured=None):
        return dict(tcp_pose_xyzw=self.state.copy(), joint_positions=np.zeros(6))

    def measured_gripper(self, measured=None):
        return 1., 1

    def tracking_errors(self, state):
        return dict(translation_m=float(np.linalg.norm(np.asarray(self.last_target[:3]) -
                                                      state['tcp_pose_xyzw'][:3])), rotation_rad=0.)

    def step_target(self, target, opening, after_step=None):
        command = osc_command(pose_matrix(self.state[:3], self.state[3:]),
                              pose_matrix(target[:3], target[3:7]), np.eye(4), .1, .3,
                              gripper_open=opening)
        # First-order response fraction per control tick; no hidden feedforward.
        self.state[:3] += self.gain * command[:3].astype(float) * .1
        self.last_stopped = after_step is not None and bool(after_step(2 * opening - 1))
        return [command.tolist()]


def validate_tracking_mechanism(*, gains=(1., .3, .1), steps=600, horizon=16,
                                execute_horizon=4, action_dt=1 / 60, speed_m_s=.01):
    if (type(steps) is not int or steps < 1 or type(horizon) is not int or
            type(execute_horizon) is not int or not 1 <= execute_horizon <= horizon or
            steps % execute_horizon):
        raise ValueError('Require positive steps divisible by H, and 1 <= H <= K')
    if not np.isfinite([action_dt, speed_m_s]).all() or min(action_dt, speed_m_s) <= 0:
        raise ValueError('Require positive finite dt and speed')
    gains = tuple(gains)
    if not gains or any(not np.isfinite(gain) or not 0 < gain <= 1 for gain in gains):
        raise ValueError('Synthetic response fractions must be in (0,1]')
    spec = dict(schema=SCHEMA, horizon=horizon, execute_horizon=execute_horizon, action_dt=action_dt,
                frame='world', quaternion_order='xyzw', reference_point='tcp',
                gripper='1=open,0=close', action_size=8)
    distance = speed_m_s * action_dt
    cases = []
    for gain in gains:
        for mode in ('fixed_time', 'measured_state_reanchor'):
            bridge = _LagBridge(float(gain), action_dt)
            executor = ChunkExecutor(bridge, spec)
            trace = []
            for start in range(0, steps, execute_horizon):
                # No learned bias: both give exactly the reference futures when
                # the measured state lies on the demonstration at this time.
                anchor = start * distance if mode == 'fixed_time' else bridge.state[0]
                actions = np.tile([0., 0., 0., 0., 0., 0., 1., 1.], (horizon, 1))
                actions[:, 0] = anchor + np.arange(1, horizon + 1) * distance
                executor.execute(actions, {})
                if not executor.last_result['completed']:
                    raise RuntimeError(f'Mechanism experiment stopped: {executor.last_result["status"]}')
                trace.extend(executor.last_result['tracking_samples'])
            actual = np.array([sample['measured_after_pose_xyzw'][:3] for sample in trace])
            targets = np.array([sample['target_pose_xyzw'][:3] for sample in trace])
            reference = np.arange(1, steps + 1) * distance
            response = (1 - (1 - gain) * (1 - (1 - gain) ** execute_horizon) /
                        (gain * execute_horizon))
            cases.append(dict(mode=mode, response_fraction_per_tick=float(gain),
                reference_final_x_m=float(reference[-1]), measured_final_x_m=float(actual[-1, 0]),
                progress_ratio=float(actual[-1, 0] / reference[-1]),
                reanchor_progress_ratio_theory=float(response) if mode == 'measured_state_reanchor' else None,
                mean_target_tracking_error_m=float(np.linalg.norm(targets - actual, axis=-1).mean()),
                mean_reference_position_error_m=float(np.abs(actual[:, 0] - reference).mean()),
                reversed_motion_steps=int(np.sum(np.diff(np.r_[0., actual[:, 0]]) < -1e-8)),
                reference_x_m=reference.tolist(), target_x_m=targets[:, 0].tolist(),
                measured_x_m=actual[:, 0].tolist()))
    return dict(kind='synthetic_chunk_tracking_mechanism', actual_robot_validated=False,
                plant='chosen first-order response; not IsaacLab physics',
                predictor='constant-velocity oracle on the reference trajectory; no learned prediction error',
                steps=steps, horizon=horizon, execute_horizon=execute_horizon,
                action_dt=action_dt, speed_m_s=speed_m_s, cases=cases)
