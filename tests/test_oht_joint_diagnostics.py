"""Actual articulation limit evidence must survive a failed waypoint."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from finetune.OHT.runtime.control_diagnostics import joint_report
from tests.test_oht_native_runtime import environment


def bridge_for(env):
    return SimpleNamespace(env=env, client=SimpleNamespace(action_config={}))


@pytest.mark.parametrize('source', [
    'joint_pos_limits', 'joint_limits',
])
@pytest.mark.parametrize('batched', [True, False])
def test_actual_hard_limit_margin_uses_live_order_and_remains_distinct_from_soft(source, batched):
    env = environment('joint')
    robot = env.scene['robot']
    bounds = torch.tensor([[[-2 * np.pi, 2 * np.pi]] * 8])
    setattr(robot.data, source, bounds if batched else bounds[0])
    robot.data.joint_pos[0, 3] = -2 * np.pi - 1e-6
    report = joint_report(bridge_for(env))
    assert report['joint_names'] == ['j5', 'j3', 'j1', 'j4', 'j2', 'j0']
    assert report['hard']['source'] == source
    assert report['hard']['near_lower'] == ['j3']
    assert report['hard']['outside'] == []  # PhysX floating point tolerance.
    assert report['hard']['lower_margin_rad'][1] == pytest.approx(-1e-6, abs=5e-7)
    assert report['soft']['outside'] == ['j3']
    json.dumps(report, allow_nan=False)
    assert not env.commands


def test_missing_hard_limits_stay_unavailable_and_unbounded_soft_limits_are_json_safe():
    env = environment()
    env.scene['robot'].data.soft_joint_pos_limits[:] = torch.tensor([-torch.inf, torch.inf])
    report = joint_report(bridge_for(env))
    assert report['hard'] is None
    assert report['soft']['position_bounds_rad'] == [[None, None]] * 6
    assert report['soft']['lower_margin_rad'] == [None] * 6
    assert report['soft']['near_lower'] == report['soft']['near_upper'] == []
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('source', ['default_joint_pos_limits', 'default_joint_limits'])
def test_initial_default_limits_cannot_establish_current_physical_stops(source):
    env = environment()
    setattr(env.scene['robot'].data, source, torch.tensor([[[-2., 2.]] * 8]))
    assert joint_report(bridge_for(env))['hard'] is None


def test_completed_state_snapshot_is_used_instead_of_a_later_live_read():
    env = environment('eef')
    env.scene['robot'].data.joint_pos_limits = torch.tensor([[[-2., 2.]] * 8])
    env.scene['robot'].data.joint_pos[0, 2] = 1.5
    report = joint_report(bridge_for(env), {'joint_positions': [0, 0, 2.1, 0, 0, 0]})
    assert report['positions_rad'][2] == 2.1
    assert report['hard']['outside'] == report['hard']['near_upper'] == ['j2']


@pytest.mark.parametrize('bounds', [np.zeros((2, 2, 2, 2)), [[float('nan'), 1.]] * 8,
                                  [[2., -2.]] * 8])
def test_invalid_sdk_limit_field_is_reported_without_replacing_waypoint_failure(bounds):
    env = environment()
    env.scene['robot'].data.joint_pos_limits = np.asarray(bounds)
    report = joint_report(bridge_for(env))
    assert 'ValueError' in report['error']
    assert not env.commands
