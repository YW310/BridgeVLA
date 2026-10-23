"""Timed absolute EEF prefix tracking; no keypoint arrival/gripper waiting phase."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .chunk_protocol import validate_spec, validate_prediction
from .control_diagnostics import joint_report


class ChunkExecutor:
    def __init__(self, bridge, spec, execute_horizon=None, max_translation_error=.10, max_rotation_error=.5):
        self.bridge, self.spec = bridge, validate_spec(spec)
        self.horizon = spec['execute_horizon'] if execute_horizon is None else execute_horizon
        if type(self.horizon) is not int or not 1 <= self.horizon <= spec['horizon']:
            raise ValueError('Chunk execute horizon must be within [1,K]')
        if bridge.client.action_config['mode'] != 'eef':
            raise ValueError('Timed chunks require absolute EEF execution')
        dt = float(bridge.env.step_dt)
        ratio = spec['action_dt'] / dt if np.isfinite(dt) and dt > 0 else np.nan
        if not np.isfinite(ratio) or ratio < 1 or not np.isclose(ratio, round(ratio), atol=1e-5, rtol=1e-5):
            raise ValueError('Chunk action_dt must be an integer multiple of the simulator control interval')
        self.substeps = int(round(ratio))
        if not np.isfinite([max_translation_error, max_rotation_error]).all() or min(max_translation_error, max_rotation_error) <= 0:
            raise ValueError('Chunk tracking error limits must be finite and positive')
        self.max_translation_error, self.max_rotation_error = max_translation_error, max_rotation_error
        self.last_result = None

    def execute(self, actions, measured, after_step=None):
        bridge = self.bridge
        actions = validate_prediction(dict(actions=actions, key_action=actions[0]), self.spec)['actions']
        start = np.asarray(bridge.measured_state(measured)['tcp_pose_xyzw'], dtype=float)
        _, opening = bridge.measured_gripper(measured)
        commands, consumed = [], 0
        last_gripper = int(opening)
        status = 'completed'
        bridge.last_done = bridge.last_stopped = False
        for tick, action in enumerate(actions[:self.horizon]):
            rotations = Slerp([0., 1.], Rotation.from_quat(np.stack((start[3:7], action[3:7]))))
            for sub in range(1, self.substeps + 1):
                fraction = sub / self.substeps
                target = np.r_[start[:3] + fraction * (action[:3] - start[:3]),
                               rotations(fraction).as_quat(), action[-1]]
                # A switch recorded at the next tick is not applied before that tick.
                grip = int(action[-1]) if sub == self.substeps else int(opening)
                bridge.last_target = target.tolist()
                before = bridge.measured_state()
                hard = joint_report(bridge, before).get('hard')
                if hard and hard['outside']:
                    status = 'joint_limit'
                    break
                commands.extend(bridge.step_target(target, grip, after_step))
                last_gripper = grip
                if sub == self.substeps:
                    consumed = tick + 1
                state = bridge.measured_state()
                errors = bridge.tracking_errors(state)
                if bridge.last_done or bridge.last_stopped:
                    status = 'environment_done' if bridge.last_done else 'task_stopped'
                    break
                if (errors['translation_m'] > self.max_translation_error or errors['rotation_rad'] > self.max_rotation_error):
                    status = 'tracking_error'
                    break
            if status != 'completed':
                break
            consumed = tick + 1
            start, opening = action[:7], int(action[-1])
        self.last_result = dict(status=status, completed=status in ('completed', 'environment_done', 'task_stopped'),
                                steps=len(commands), consumed_actions=consumed, requested_actions=self.horizon,
                                action_dt=self.spec['action_dt'], control_substeps=self.substeps,
                                last_commanded_gripper=last_gripper,
                                discarded_suffix=self.spec['horizon'] - consumed)
        return commands
