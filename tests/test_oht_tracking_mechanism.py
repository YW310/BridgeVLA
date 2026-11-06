"""Reproduce controller-lag amplification through the production chunk executor."""
import json

import numpy as np
import pytest

from finetune.OHT.control_validation import _LagBridge, validate_tracking_mechanism
from finetune.OHT.runtime.chunk_executor import ChunkExecutor
from finetune.OHT.runtime.chunk_protocol import SCHEMA
from finetune.OHT.runtime.control_diagnostics import tcp_motion_report


@pytest.mark.parametrize('horizon', [1, 4, 16])
def test_perfect_tracking_has_no_reanchor_penalty(horizon):
    result = validate_tracking_mechanism(gains=[1.], steps=64, execute_horizon=horizon)
    fixed, replanned = result['cases']
    assert result['actual_robot_validated'] is False
    np.testing.assert_allclose(fixed['measured_x_m'], replanned['measured_x_m'], atol=1e-8)
    assert replanned['progress_ratio'] == pytest.approx(1., abs=1e-6)
    assert fixed['mean_target_tracking_error_m'] < 1e-9


@pytest.mark.parametrize('gain', [.1, .3])
def test_lag_reduces_progress_with_zero_oracle_error_and_no_direction_flip(gain):
    result = validate_tracking_mechanism(gains=[gain], steps=600)
    fixed, replanned = result['cases']
    count, horizon = result['steps'], result['execute_horizon']
    # Independent closed-form solutions to the explicitly declared lag plant.
    expected_fixed = 1 - (1 - gain) * (1 - (1 - gain) ** count) / (gain * count)
    expected_replan = 1 - (1 - gain) * (1 - (1 - gain) ** horizon) / (gain * horizon)
    assert fixed['progress_ratio'] == pytest.approx(expected_fixed, abs=1e-6)
    assert replanned['progress_ratio'] == pytest.approx(expected_replan, abs=1e-6)
    assert replanned['progress_ratio'] < fixed['progress_ratio']
    assert replanned['mean_reference_position_error_m'] > 10 * fixed['mean_reference_position_error_m']
    assert replanned['mean_target_tracking_error_m'] < .001
    # This mechanism alone does not explain the user's reversed predicted X.
    assert fixed['reversed_motion_steps'] == replanned['reversed_motion_steps'] == 0


def test_longer_prefix_reduces_this_synthetic_reanchor_penalty():
    ratios = [validate_tracking_mechanism(gains=[.1], steps=64, execute_horizon=h)['cases'][1]['progress_ratio']
              for h in (1, 4, 16)]
    assert ratios[0] < ratios[1] < ratios[2] < 1


def executor(bridge):
    spec = dict(schema=SCHEMA, horizon=4, execute_horizon=4, action_dt=1 / 60,
                frame='world', quaternion_order='xyzw', reference_point='tcp',
                gripper='1=open,0=close', action_size=8)
    return ChunkExecutor(bridge, spec)


def actions():
    targets = np.tile([0., 0., 0., 0., 0., 0., 1., 1.], (4, 1))
    targets[:, 0] = .4 + np.arange(1, 5) * (.01 / 60)
    return targets


def test_trace_preserves_targets_before_after_responses_and_physics_step_count():
    bridge = _LagBridge(.1, 1 / 60)
    bridge.state[0] = .4
    control = executor(bridge)
    targets = actions()
    sent = control.execute(targets, {})
    result = control.last_result
    trace = result['tracking_samples']
    assert result['completed'] and result['steps'] == len(trace) == len(sent) == 4
    np.testing.assert_allclose([row['target_pose_xyzw'] for row in trace], targets[:, :7], atol=2e-8)
    np.testing.assert_allclose(trace[0]['measured_before_pose_xyzw'], [.4, 0, 0, 0, 0, 0, 1])
    np.testing.assert_allclose([row['measured_after_pose_xyzw'][0] - .4 for row in trace],
                              np.array([.1, .29, .561, .9049]) * (.01 / 60), atol=1e-8)
    for previous, current in zip(trace, trace[1:]):
        assert previous['measured_after_pose_xyzw'] == current['measured_before_pose_xyzw']
    assert result['tcp_motion']['projected_response_ratio'] == pytest.approx(.226225, abs=5e-5)
    assert result['tcp_motion']['target_vs_measured_angle_degrees'] == pytest.approx(0)
    # Diagnostics own snapshots; later control steps cannot rewrite old traces.
    before = json.dumps(trace)
    control.execute(targets, {})
    assert json.dumps(trace) == before


def test_no_executed_motion_is_not_reported_as_a_direction_agreement():
    bridge = _LagBridge(0., 1 / 60)
    bridge.state[0] = .4
    control = executor(bridge)
    control.execute(actions(), {})
    assert control.last_result['tcp_motion']['projected_response_ratio'] == 0
    assert control.last_result['tcp_motion']['target_vs_measured_angle_degrees'] is None
    assert tcp_motion_report([]) is None
    stationary = control.last_result['tracking_samples'][:1]
    stationary[0]['target_pose_xyzw'] = stationary[0]['measured_before_pose_xyzw']
    report = tcp_motion_report(stationary)
    assert report['projected_response_ratio'] is None
    json.dumps(report, allow_nan=False)


def test_guard_refusal_has_no_response_samples():
    bridge = _LagBridge(.1, 1 / 60)
    control = executor(bridge)
    control.execute(actions(), {})
    assert control.last_result['status'] == 'step_limit'
    assert control.last_result['tracking_samples'] == []
    assert control.last_result['tcp_motion'] is None


@pytest.mark.parametrize('phase', ['before_step', 'after_step'])
def test_tracking_stop_keeps_only_executed_samples(phase):
    bridge = _LagBridge(.1, 1 / 60)
    bridge.state[0] = .4
    control = executor(bridge)
    original = bridge.tracking_errors
    def errors(state):
        failed = (bridge.last_target[0] > .4002 if phase == 'before_step'
                  else state['tcp_pose_xyzw'][0] > .4)
        return dict(translation_m=.2, rotation_rad=0.) if failed else original(state)
    bridge.tracking_errors = errors
    sent = control.execute(actions(), {})
    result = control.last_result
    assert result['status'] == 'tracking_error' and result['tracking_check_phase'] == phase
    assert len(sent) == len(result['tracking_samples']) == 1
    first = result['tracking_samples'][0]
    assert first['chunk_tick'] == 0
    assert result['tcp_motion']['target_displacement_world_m'][0] == pytest.approx(
        first['target_pose_xyzw'][0] - .4)
    if phase == 'before_step':
        # The pending rejected target must not enter the executed-motion summary.
        assert bridge.last_target[0] > first['target_pose_xyzw'][0]


def test_validation_cli_writes_explicitly_synthetic_report(tmp_path, capsys):
    from tools.validate_oht_chunk_tracking import main
    path = tmp_path / 'report.json'
    assert main(['--steps', '16', '--gains', '.1', '--output', str(path)]) == 0
    result = json.loads(path.read_text())
    assert result['actual_robot_validated'] is False
    assert len(result['cases']) == 2
    assert 'reference_x_m' in result['cases'][0]
    assert 'Report:' in capsys.readouterr().out
