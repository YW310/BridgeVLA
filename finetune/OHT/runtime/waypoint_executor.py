"""Execute one future waypoint before advancing the policy's gripper phase."""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class WaypointOptions:
    max_steps: int = 600
    position_tolerance: float = .005
    rotation_tolerance: float = .05
    joint_tolerance: float = .01
    arrival_steps: int = 3
    gripper_min_steps: int = 10
    gripper_stable_steps: int = 5
    gripper_stall_tolerance: float = .002  # range of normalized open01, not metres
    stall_steps: int = 100
    minimum_progress: float = .001  # arm metric units; normalized open01 for aperture progress

    def __post_init__(self):
        integers = ('max_steps', 'arrival_steps', 'gripper_min_steps', 'gripper_stable_steps', 'stall_steps')
        for name in integers:
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in set(self.__dataclass_fields__) - set(integers):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')


class WaypointExecutor:
    """Keep one absolute goal and defer a new gripper state until arm arrival.

    The bridge supplies CPU measurements and exactly one physical control step.
    Closing may settle against an object without reaching the empty-gripper
    endpoint. Stability uses a range of calibrated open01 readings. Settled
    means a stable aperture, not proof of a successful grasp.
    """

    def __init__(self, bridge, **options):
        self.bridge, self.options = bridge, WaypointOptions(**options)
        calibration = getattr(bridge, 'gripper_calibration', {})
        self._open_threshold = float(calibration.get('open_threshold', .9))
        self._close_threshold = float(calibration.get('close_threshold', .1))
        self._close_movement = float(calibration.get('diff_threshold', .05))
        if (not np.isfinite([self._open_threshold, self._close_threshold, self._close_movement]).all()
                or not 0 <= self._close_threshold < self._open_threshold <= 1
                or not 0 < self._close_movement <= 1):
            raise ValueError('Invalid gripper hysteresis calibration')
        self.last_result = None

    def _running(self):
        return self.bridge.ref.simulation_app.is_running()

    def _errors(self, target, state):
        if 'joint' in self.bridge.client.action_config['mode']:
            current = np.asarray(state['joint_positions'], np.float64)
            if current.shape != (6,) or not np.isfinite(current).all():
                raise ValueError('Expected six finite measured joints')
            return dict(joint_max_rad=float(np.max(np.abs(target[:6] - current))))
        current = np.asarray(state['tcp_pose_xyzw'], np.float64)
        if current.shape != (7,) or not np.isfinite(current).all():
            raise ValueError('Expected a finite measured TCP pose')
        return dict(translation_m=float(np.linalg.norm(target[:3] - current[:3])),
                    rotation_rad=float((Rotation.from_quat(target[3:7]) * Rotation.from_quat(current[3:7]).inv()).magnitude()))

    def _arrived(self, errors):
        if 'joint_max_rad' in errors:
            return errors['joint_max_rad'] <= self.options.joint_tolerance
        return (errors['translation_m'] <= self.options.position_tolerance and
                errors['rotation_rad'] <= self.options.rotation_tolerance)

    def _metric(self, errors, ratio, target_gripper, phase):
        if 'joint_max_rad' in errors:
            arm = errors['joint_max_rad']
        else:
            # Put rotation progress on a metre-like scale using the configured
            # arrival tolerances; position alone would miss rotation-only stalls.
            arm = max(errors['translation_m'], errors['rotation_rad'] *
                      self.options.position_tolerance / self.options.rotation_tolerance)
        opening = (max(0., self._open_threshold - ratio) * self.options.position_tolerance
                   if phase == 'gripper' and target_gripper else 0.)
        return max(arm, opening)

    def execute(self, action, measured, after_step=None):
        options, bridge = self.options, self.bridge
        commands, target, errors = [], None, {}
        phase, total, arm_steps, gripper_steps, consecutive = 'approach', 0, 0, 0, 0
        ratio = initial_gripper = desired_gripper = commanded_gripper = None
        callback_success, callback_error = False, None
        phase_start_ratio = best_aperture = None
        self.last_result = None

        def finish(status, error=None):
            self.last_result = dict(status=status, completed=status in ('completed', 'success'), phase=phase,
                steps=total, approach_steps=arm_steps, gripper_steps=gripper_steps,
                target=None if target is None else target.tolist(), initial_gripper=initial_gripper,
                target_gripper=desired_gripper, commanded_gripper=commanded_gripper,
                gripper_open01=ratio, errors=errors, error=error)
            return commands

        def on_step(gripper):
            nonlocal callback_success, callback_error
            if after_step is not None:
                try:
                    callback_success = bool(after_step(gripper))
                except Exception as exc:
                    callback_error = f'{type(exc).__name__}: {exc}'
            return callback_success or callback_error is not None

        try:
            # Relative commands resolve exactly once, against their request's
            # measurement. All subsequent steps track this same absolute goal.
            target = np.asarray(bridge.resolve_target(action, measured), np.float64).copy()
            desired_gripper = int(target[-1])
            ratio, binary = bridge.measured_gripper(measured)
            ratio, initial_gripper = float(ratio), int(binary)
            if not np.isfinite(ratio) or not 0 <= ratio <= 1 or initial_gripper not in (0, 1):
                raise ValueError('Expected measured open01 and binary gripper state')
            state = bridge.measured_state(measured)
            errors = self._errors(target, state)
            commanded_gripper = initial_gripper
            best = self._metric(errors, ratio, desired_gripper, phase)
            stagnant = 0
            grip_positions = deque(maxlen=options.gripper_stable_steps + 1)
            for _ in range(options.max_steps):
                if not self._running():
                    return finish('application_stopped')
                sent = bridge.step_target(target, commanded_gripper, after_step=on_step)
                commands.extend(sent)
                total += len(sent)
                if len(sent) != 1:
                    raise ValueError('Waypoint bridge must execute exactly one control step')
                if phase == 'approach':
                    arm_steps += 1
                else:
                    gripper_steps += 1
                if callback_error:
                    return finish('error', callback_error)
                if callback_success:
                    return finish('success')
                if bridge.last_done:
                    return finish('environment_done')
                if not self._running():
                    return finish('application_stopped')
                state = bridge.measured_state()
                ratio, _ = bridge.measured_gripper()
                ratio = float(ratio)
                if not np.isfinite(ratio) or not 0 <= ratio <= 1:
                    raise ValueError('Expected measured open01 in [0,1]')
                errors = self._errors(target, state)
                arrived = self._arrived(errors)
                consecutive = consecutive + 1 if arrived else 0
                metric = self._metric(errors, ratio, desired_gripper, phase)
                if metric <= best - options.minimum_progress:
                    best, stagnant = metric, 0
                else:
                    stagnant += 1
                if phase == 'approach':
                    if arrived:
                        stagnant = 0
                    if consecutive >= options.arrival_steps:
                        phase, commanded_gripper = 'gripper', desired_gripper
                        phase_start_ratio = ratio
                        best_aperture = ratio
                        best, stagnant = self._metric(errors, ratio, desired_gripper, phase), 0
                        grip_positions.append(ratio)
                        continue
                else:
                    # Arm arrival alone says nothing about motor progress.
                    # Track aperture in its own normalized units so a slowly
                    # closing gripper is not declared stalled merely because
                    # the already-arrived arm has no further error reduction.
                    aperture_progress = (ratio - best_aperture if desired_gripper else best_aperture - ratio)
                    if aperture_progress >= options.minimum_progress:
                        best_aperture, stagnant = ratio, 0
                    grip_positions.append(ratio)
                    stable = (len(grip_positions) == grip_positions.maxlen and
                              max(grip_positions) - min(grip_positions) <= options.gripper_stall_tolerance)
                    # A stable fully open gripper is not a successful close.
                    # Accept contact at a partial aperture after real closure,
                    # or an already closed initial state, without requiring an
                    # impossible empty-gripper endpoint around a held object.
                    aperture_ready = (ratio >= self._open_threshold if desired_gripper else
                                      initial_gripper == 0 or ratio <= self._close_threshold or
                                      phase_start_ratio - ratio >= self._close_movement)
                    if (gripper_steps >= options.gripper_min_steps and stable and aperture_ready and
                            consecutive >= options.arrival_steps):
                        return finish('completed')
                if stagnant >= options.stall_steps:
                    return finish('stalled')
            return finish('timeout')
        except Exception as exc:
            return finish('error', f'{type(exc).__name__}: {exc}')
