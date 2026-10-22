"""Future-waypoint execution with a synthetic robot and no IsaacLab dependency."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from finetune.OHT.action_modes import action_options
from finetune.OHT.runtime.executor import execution_target
from finetune.OHT.runtime.waypoint_executor import WaypointExecutor, WaypointOptions


class RobotBridge:
    def __init__(self, mode='eef', *, ratio=1., close_contact=.4):
        self.client = SimpleNamespace(action_config=action_options({'mode': mode}))
        self.running = True
        self.ref = SimpleNamespace(simulation_app=SimpleNamespace(is_running=lambda: self.running))
        self.pose = np.array([0., 0, .5, 0, 0, 0, 1])
        self.joints = np.zeros(6)
        self.ratio, self.binary, self.close_contact = ratio, int(ratio >= .5), close_contact
        self.steps, self.resolve_calls, self.targets, self.grippers = 0, 0, [], []
        self.last_done = self.last_stopped = self.arm_jammed = self.gripper_jammed = False
        self.after_physics = None

    def measure(self):
        return dict(tcp_pose_xyzw=self.pose.copy(), joint_positions=self.joints.copy(),
                    gripper_position=self.ratio, ratio=self.ratio, binary=self.binary)

    def measured_state(self, measured=None):
        return self.measure() if measured is None else measured

    def measured_gripper(self, measured=None):
        return ((self.ratio, self.binary) if measured is None else (measured['ratio'], measured['binary']))

    def resolve_target(self, action, measured):
        self.resolve_calls += 1
        return execution_target(action, self.client.action_config,
                                current_tcp=measured['tcp_pose_xyzw'], current_joints=measured['joint_positions'])

    def step_target(self, target, gripper_open, after_step=None):
        if not self.running:
            raise RuntimeError('Application stopped')
        self.targets.append(np.asarray(target).copy())
        self.grippers.append(gripper_open)
        self.steps += 1
        if not self.arm_jammed:
            if 'joint' in self.client.action_config['mode']:
                self.joints += np.clip(target[:6] - self.joints, -.03, .03)
            else:
                self.pose[:3] += np.clip(target[:3] - self.pose[:3], -.03, .03)
                current, desired = Rotation.from_quat(self.pose[3:]), Rotation.from_quat(target[3:7])
                rotation = (desired * current.inv()).as_rotvec()
                angle = np.linalg.norm(rotation)
                self.pose[3:] = (Rotation.from_rotvec(rotation * min(1., .2 / max(angle, 1e-12))) * current).as_quat()
        if not self.gripper_jammed:
            endpoint = 1. if gripper_open else self.close_contact
            self.ratio += np.clip(endpoint - self.ratio, -.2, .2)
            self.binary = gripper_open
        if self.after_physics:
            self.after_physics(self)
        self.last_stopped = bool(after_step and after_step(2 * gripper_open - 1))
        return [[0., 0, 0, 0, 0, 0, float(2 * gripper_open - 1)]]


def target(x=.12, gripper=0):
    return np.array([x, 0, .5, 0, 0, 0, 1, gripper])


def executor(bridge, **changes):
    return WaypointExecutor(bridge, **{**dict(max_steps=80, arrival_steps=3, gripper_min_steps=6,
                                             gripper_stable_steps=3, stall_steps=30), **changes})


def test_perfect_future_grasp_keeps_gripper_open_until_arm_arrives_and_closes_against_object():
    bridge = RobotBridge()
    follow = executor(bridge)
    commands = follow.execute(target(), bridge.measure())
    assert follow.last_result['status'] == 'completed'
    assert follow.last_result['approach_steps'] >= 6  # four movement steps, then stable arrival
    approach = follow.last_result['approach_steps']
    assert bridge.grippers[:approach] == [1] * approach
    assert bridge.grippers[approach:] and set(bridge.grippers[approach:]) == {0}
    assert len(commands) == follow.last_result['steps'] == bridge.steps
    assert follow.last_result['gripper_steps'] >= 6
    assert bridge.ratio == pytest.approx(.4)  # object contact, not empty-gripper endpoint zero
    np.testing.assert_allclose(bridge.pose[:3], target()[:3])


@pytest.mark.parametrize('mode', ['relative_eef', 'relative_joint'])
def test_relative_target_is_resolved_once_and_never_reintegrated(mode):
    bridge = RobotBridge(mode)
    follow = executor(bridge)
    follow.execute(np.array([.12, 0, 0, 0, 0, 0, 0]), bridge.measure())
    assert follow.last_result['status'] == 'completed' and bridge.resolve_calls == 1
    assert all(np.array_equal(item, bridge.targets[0]) for item in bridge.targets)
    assert bridge.targets[0][0] == pytest.approx(.12)
    assert (bridge.joints[0] if 'joint' in mode else bridge.pose[0]) == pytest.approx(.12)


def test_absolute_joint_waypoint_uses_arrival_gate_and_held_initial_gripper():
    bridge = RobotBridge('joint')
    follow = executor(bridge)
    follow.execute(np.array([.12, 0, 0, 0, 0, 0, 0]), bridge.measure())
    assert follow.last_result['status'] == 'completed'
    assert bridge.grippers[:follow.last_result['approach_steps']] == [1] * follow.last_result['approach_steps']
    np.testing.assert_allclose(bridge.joints, [.12, 0, 0, 0, 0, 0])


def test_rotation_arrival_is_required_even_when_translation_is_already_correct():
    bridge = RobotBridge()
    desired = target(x=0)
    desired[3:7] = Rotation.from_euler('z', 1.).as_quat()
    follow = executor(bridge)
    follow.execute(desired, bridge.measure())
    assert follow.last_result['status'] == 'completed'
    assert follow.last_result['approach_steps'] >= 7
    assert follow.last_result['errors']['rotation_rad'] <= follow.options.rotation_tolerance


def test_opening_requires_actual_open_aperture_and_stability():
    bridge = RobotBridge(ratio=.4)
    follow = executor(bridge)
    follow.execute(target(x=0, gripper=1), bridge.measure())
    assert follow.last_result['status'] == 'completed' and bridge.ratio >= .9
    approach = follow.last_result['approach_steps']
    assert bridge.grippers[:approach] == [0] * approach
    assert set(bridge.grippers[approach:]) == {1}


def test_stalled_arm_aborts_before_applying_future_gripper():
    bridge = RobotBridge()
    bridge.arm_jammed = True
    follow = executor(bridge, stall_steps=4)
    commands = follow.execute(target(), bridge.measure())
    assert follow.last_result['status'] == 'stalled'
    assert len(commands) == 4 and bridge.grippers == [1] * 4
    assert not follow.last_result['completed']


def test_step_limit_aborts_without_predicting_a_new_target():
    bridge = RobotBridge()
    follow = executor(bridge, max_steps=2)
    commands = follow.execute(target(x=10), bridge.measure())
    assert follow.last_result['status'] == 'timeout' and len(commands) == 2
    assert bridge.resolve_calls == 1 and bridge.grippers == [1, 1]


def test_blocked_opening_cannot_complete_from_a_stable_closed_aperture():
    bridge = RobotBridge(ratio=.4)
    bridge.gripper_jammed = True
    follow = executor(bridge, stall_steps=12)
    follow.execute(target(x=0, gripper=1), bridge.measure())
    assert follow.last_result['status'] == 'stalled' and not follow.last_result['completed']
    assert follow.last_result['phase'] == 'gripper' and bridge.ratio < .9


def test_a_failed_close_that_stays_fully_open_cannot_complete_from_stability_alone():
    bridge = RobotBridge()
    bridge.gripper_jammed = True
    follow = executor(bridge, stall_steps=12)
    follow.execute(target(x=0), bridge.measure())
    assert follow.last_result['status'] == 'stalled' and not follow.last_result['completed']
    assert bridge.ratio == 1 and follow.last_result['phase'] == 'gripper'


def test_slow_closure_progress_prevents_false_stall_after_arm_has_already_arrived():
    bridge = RobotBridge()
    bridge.gripper_jammed = True  # the hook supplies a slow but functioning motor

    def slow_close(robot):
        if robot.grippers[-1] == 0:
            robot.ratio = max(.4, robot.ratio - .01)

    bridge.after_physics = slow_close
    follow = executor(bridge, max_steps=100, arrival_steps=1, stall_steps=4,
                      gripper_stable_steps=2, gripper_min_steps=6)
    follow.execute(target(x=0), bridge.measure())
    assert follow.last_result['status'] == 'completed'
    assert follow.last_result['approach_steps'] == 1
    assert follow.last_result['gripper_steps'] >= 60 > follow.options.stall_steps
    assert bridge.ratio == pytest.approx(.4)


def test_already_closed_gripper_holding_an_object_does_not_require_more_closure_motion():
    bridge = RobotBridge(ratio=.4)
    bridge.gripper_jammed = True
    follow = executor(bridge)
    follow.execute(target(x=0), bridge.measure())
    assert follow.last_result['status'] == 'completed' and bridge.ratio == pytest.approx(.4)


def test_closure_evidence_respects_calibrated_movement_threshold():
    bridge = RobotBridge(close_contact=.9)
    bridge.gripper_calibration = dict(open_threshold=.95, close_threshold=.05, diff_threshold=.2)
    follow = executor(bridge, stall_steps=12)
    follow.execute(target(x=0), bridge.measure())
    assert follow.last_result['status'] == 'stalled'  # .1 movement does not satisfy configured .2


def test_drift_after_closing_begins_never_reopens_gripper_and_fails_if_arm_cannot_reacquire():
    bridge = RobotBridge()
    moved = False

    def drift(robot):
        nonlocal moved
        if robot.grippers[-1] == 0 and not moved:
            robot.pose[0] += .1
            robot.arm_jammed = moved = True

    bridge.after_physics = drift
    follow = executor(bridge, stall_steps=8)
    follow.execute(target(x=0), bridge.measure())
    assert follow.last_result['status'] == 'stalled' and follow.last_result['phase'] == 'gripper'
    first_close = bridge.grippers.index(0)
    assert set(bridge.grippers[first_close:]) == {0}


def test_transient_drift_can_reacquire_same_fixed_target_while_retaining_closed_command():
    bridge = RobotBridge()
    moved = False

    def drift(robot):
        nonlocal moved
        if robot.grippers[-1] == 0 and not moved:
            robot.pose[0] += .1
            moved = True

    bridge.after_physics = drift
    follow = executor(bridge)
    follow.execute(target(x=0), bridge.measure())
    assert follow.last_result['status'] == 'completed'
    assert all(item[0] == 0 for item in bridge.targets)
    assert set(bridge.grippers[bridge.grippers.index(0):]) == {0}


@pytest.mark.parametrize('reason', ['application_stopped', 'environment_done', 'success'])
def test_external_stop_preserves_commands_and_prevents_further_steps(reason):
    bridge = RobotBridge()

    def stop(robot):
        if robot.steps == 2:
            robot.running = reason != 'application_stopped'
            robot.last_done = reason == 'environment_done'

    bridge.after_physics = stop
    follow = executor(bridge)
    callbacks = []

    def after_step(gripper):
        callbacks.append(gripper)
        return reason == 'success' and bridge.steps == 2

    commands = follow.execute(target(), bridge.measure(), after_step)
    assert follow.last_result['status'] == reason
    assert len(commands) == bridge.steps == len(callbacks) == 2


def test_application_closed_before_execution_sends_no_command():
    bridge = RobotBridge()
    bridge.running = False
    follow = executor(bridge)
    assert follow.execute(target(), bridge.measure()) == []
    assert follow.last_result['status'] == 'application_stopped'


def test_callback_failure_keeps_the_already_executed_command_in_failure_result():
    bridge = RobotBridge()
    follow = executor(bridge)

    def failed(_):
        raise RuntimeError('scorer failed')

    commands = follow.execute(target(), bridge.measure(), failed)
    assert len(commands) == bridge.steps == 1
    assert follow.last_result['status'] == 'error' and 'scorer failed' in follow.last_result['error']


def test_controller_failure_preserves_all_previous_commands():
    bridge = RobotBridge()
    original = bridge.step_target

    def broken(*args, **kwargs):
        if bridge.steps == 2:
            raise RuntimeError('controller failed')
        return original(*args, **kwargs)

    bridge.step_target = broken
    follow = executor(bridge)
    commands = follow.execute(target(), bridge.measure())
    assert len(commands) == bridge.steps == 2
    assert follow.last_result['status'] == 'error' and 'controller failed' in follow.last_result['error']


def test_consecutive_arrival_readings_are_required_before_gripper_phase():
    bridge = RobotBridge()

    def jitter(robot):
        if robot.steps in (2, 4):
            robot.pose[0] = .02

    bridge.after_physics = jitter
    follow = executor(bridge)
    follow.execute(target(x=0), bridge.measure())
    assert follow.last_result['status'] == 'completed'
    assert bridge.grippers[:7] == [1] * 7


def test_slow_cumulative_gripper_motion_is_not_mistaken_for_stability():
    bridge = RobotBridge(ratio=.8)
    bridge.gripper_jammed = True

    def creeping(robot):
        if robot.grippers[-1] == 0:
            robot.ratio -= .001

    bridge.after_physics = creeping
    follow = executor(bridge, max_steps=20, gripper_stable_steps=5, stall_steps=100)
    follow.execute(target(x=0), bridge.measure())
    assert follow.last_result['status'] == 'timeout'  # last-six range=.005 > tolerance .002


@pytest.mark.parametrize('mode', ['eef', 'joint'])
def test_real_bridge_cpu_adapter_preserves_arrival_gate_and_single_step_accounting(mode):
    from tests.test_oht_native_runtime import environment
    from finetune.OHT.runtime.isaaclab_bridge import IsaacLabBridge
    from finetune.OHT.runtime.isaaclab_runtime import IsaacLabRuntime
    from finetune.OHT.runtime.joint_executor import JointPositionExecutor

    env = environment(mode)
    robot = env.scene['robot']
    robot.data.joint_pos[0, 6:] = -.015
    original = env.step

    def physical_step(command):
        result = original(command)
        # The fixture's action layout has gripper first; simulate motor closure
        # against an object at open01=.4 rather than an empty-gripper endpoint.
        desired = -.03 if command[0, 0] > 0 else .0012
        raw = float(robot.data.joint_pos[0, 6:].sum())
        robot.data.joint_pos[0, 6:] = (raw + np.clip(desired - raw, -.01, .01)) / 2
        return result

    env.step = physical_step
    runtime = IsaacLabRuntime(env, {}, ee_link='tool', quaternion_order='wxyz',
                             simulation_app=SimpleNamespace(is_running=lambda: True))
    config = action_options({'mode': mode})
    profile = dict(link_to_tcp=np.eye(4), gripper=dict(open=-.03, close=.022))
    client = SimpleNamespace(action_config=config,
                             info={'raw_observation': {'data_profiles': {'p': profile}}})
    joints = JointPositionExecutor(env, runtime.torch, config, hold_steps=50) if mode == 'joint' else None
    bridge = IsaacLabBridge(runtime, env, client, 'assemble_left', 'p', {}, joint_executor=joints)
    follow = WaypointExecutor(bridge, max_steps=80)
    action = target(x=.18) if mode == 'eef' else np.array([.12, 0, 0, 0, 0, 0, 0])
    commands = follow.execute(action, bridge.capture())
    assert follow.last_result['status'] == 'completed'
    assert len(commands) == env.common_step_counter == follow.last_result['steps']
    arrival = follow.last_result['approach_steps']
    assert all(command[0] == 1 for command in commands[:arrival])
    assert all(command[0] == -1 for command in commands[arrival:])
    assert follow.last_result['gripper_open01'] == pytest.approx(.4, abs=1e-5)


@pytest.mark.parametrize('changes', [dict(max_steps=0), dict(arrival_steps=True), dict(stall_steps=1.5),
                                     dict(position_tolerance=0), dict(rotation_tolerance=float('nan')),
                                     dict(gripper_stall_tolerance=-1), dict(minimum_progress=True)])
def test_invalid_options_fail_before_any_execution(changes):
    with pytest.raises(ValueError):
        WaypointOptions(**changes)
