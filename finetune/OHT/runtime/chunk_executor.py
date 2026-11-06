"""Timed absolute EEF prefix tracking; no keypoint arrival/gripper waiting phase."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .chunk_protocol import validate_spec, validate_prediction
from .control_diagnostics import joint_report, tcp_motion_report
from ..data.geometry import rotation_error


class ChunkExecutor:
    def __init__(self, bridge, spec, execute_horizon=None, max_translation_error=.10, max_rotation_error=.5,
                 *, max_step_translation=.10, max_step_rotation=.5):
        self.bridge, self.spec = bridge, validate_spec(spec)
        self.horizon = spec['execute_horizon'] if execute_horizon is None else execute_horizon
        if type(self.horizon) is not int or not 1 <= self.horizon <= spec['horizon']:
            raise ValueError('Chunk execute horizon must be within [1,K]')
        if bridge.client.action_config['mode'] != 'eef':
            raise ValueError('Timed chunks require absolute EEF execution')
        dt = float(bridge.env.step_dt)
        ratio = spec['action_dt'] / dt if np.isfinite(dt) and dt > 0 else np.nan
        # Timestamp roundoff can put a valid unit ratio just below 1.
        substeps = int(round(ratio)) if np.isfinite(ratio) else 0
        if substeps < 1 or not np.isclose(ratio, substeps, atol=1e-5, rtol=1e-5):
            raise ValueError('Chunk action_dt must be an integer multiple of the simulator control interval: '
                             f'action_dt={spec["action_dt"]}, env.step_dt={dt}, ratio={ratio}')
        self.substeps = substeps
        if not np.isfinite([max_translation_error, max_rotation_error]).all() or min(max_translation_error, max_rotation_error) <= 0:
            raise ValueError('Chunk tracking error limits must be finite and positive')
        self.max_translation_error, self.max_rotation_error = max_translation_error, max_rotation_error
        if not np.isfinite([max_step_translation, max_step_rotation]).all() or min(max_step_translation, max_step_rotation) <= 0:
            raise ValueError('Chunk per-target step limits must be finite and positive')
        self.max_step_translation, self.max_step_rotation = max_step_translation, max_step_rotation
        self.last_result = None

    def _tracking_failed(self, errors):
        return (errors['translation_m'] > self.max_translation_error or
                errors['rotation_rad'] > self.max_rotation_error)

    def _step_violation(self, start, actions):
        """Largest jump between consecutive executed targets, beginning at the measured TCP."""
        previous, worst = np.asarray(start[:7], dtype=float), dict(translation_m=0., rotation_rad=0.)
        for tick, action in enumerate(actions):
            translation = float(np.linalg.norm(action[:3] - previous[:3]))
            rotation = float(rotation_error(previous[3:7], action[3:7]))
            worst = dict(translation_m=max(worst['translation_m'], translation),
                         rotation_rad=max(worst['rotation_rad'], rotation))
            if translation > self.max_step_translation or rotation > self.max_step_rotation:
                return dict(worst, tick=tick)
            previous = action[:7]
        return None

    def execute(self, actions, measured, after_step=None):
        bridge = self.bridge
        actions = validate_prediction(dict(actions=actions, key_action=actions[0]), self.spec)['actions']
        start = np.asarray(bridge.measured_state(measured)['tcp_pose_xyzw'], dtype=float)
        _, opening = bridge.measured_gripper(measured)
        commands, consumed, tracking_samples = [], 0, []
        last_gripper = int(opening)
        status = 'completed'
        errors, check_phase = None, None
        bridge.last_done = bridge.last_stopped = False
        step_violation = self._step_violation(start, actions[:self.horizon])
        if step_violation is not None:
            # Refuse the whole prefix before any motion: the target jump is implausible.
            status = 'step_limit'
        for tick, action in enumerate(actions[:self.horizon] if step_violation is None else ()):
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
                errors, check_phase = bridge.tracking_errors(before), 'before_step'
                if self._tracking_failed(errors):
                    status = 'tracking_error'
                    break
                errors_before = dict(errors)
                commands.extend(bridge.step_target(target, grip, after_step))
                last_gripper = grip
                if sub == self.substeps:
                    consumed = tick + 1
                state = bridge.measured_state()
                errors, check_phase = bridge.tracking_errors(state), 'after_step'
                tracking_samples.append(dict(chunk_tick=tick, control_substep=sub,
                    target_pose_xyzw=target[:7].tolist(),
                    measured_before_pose_xyzw=np.asarray(before['tcp_pose_xyzw']).tolist(),
                    measured_after_pose_xyzw=np.asarray(state['tcp_pose_xyzw']).tolist(),
                    errors_before=errors_before, errors_after=dict(errors)))
                if bridge.last_done or bridge.last_stopped:
                    status = 'environment_done' if bridge.last_done else 'task_stopped'
                    break
                if self._tracking_failed(errors):
                    status = 'tracking_error'
                    break
            if status != 'completed':
                break
            consumed = tick + 1
            start, opening = action[:7], int(action[-1])
        self.last_result = dict(status=status, completed=status in ('completed', 'environment_done', 'task_stopped'),
                                steps=len(commands), consumed_actions=consumed, requested_actions=self.horizon,
                                action_dt=self.spec['action_dt'], control_substeps=self.substeps,
                                errors=errors, tracking_check_phase=check_phase,
                                last_commanded_gripper=last_gripper,
                                step_violation=step_violation,
                                tracking_samples=tracking_samples,
                                tcp_motion=tcp_motion_report(tracking_samples),
                                discarded_suffix=self.spec['horizon'] - consumed)
        return commands
