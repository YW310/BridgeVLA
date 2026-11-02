"""Same-future-time fusion must not average different ticks or action phases."""
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from finetune.OHT.runtime.chunk_ensemble import ChunkTemporalEnsemble
from finetune.OHT.runtime.chunk_protocol import chunk_spec


def ensemble(**kwargs):
    spec = chunk_spec({'horizon': 8, 'execute_horizon': 2}, {'chunk': {'action_dt': .01}})
    return ChunkTemporalEnsemble(spec, **kwargs)


def actions(shift=0, bias=0., grip=1):
    value = np.tile([0., 0, .5, 0, 0, 0, 1, grip], (8, 1)).astype(np.float32)
    value[:, 0] = (np.arange(1, 9) + shift) * .001 + bias
    return value


def test_aligns_by_simulator_time_not_same_array_index_and_preserves_raw_inputs():
    worker = ensemble(decay=0.)
    first, second = actions(), actions(shift=2, bias=.002)
    original = second.copy()
    np.testing.assert_array_equal(worker.update(first, 0.), first)
    fused = worker.update(second, .02)
    np.testing.assert_allclose(fused[:6, 0], (first[2:, 0] + second[:6, 0]) / 2, atol=1e-8)
    np.testing.assert_array_equal(fused[6:], second[6:])
    np.testing.assert_array_equal(second, original)
    assert worker.last_result['contributors'] == [2] * 6 + [1] * 2
    assert worker.last_result['max_position_adjustment_m'] == pytest.approx(.001)


def test_raw_history_avoids_recursively_averaging_fused_outputs():
    worker = ensemble(decay=0.)
    worker.update(actions(), 0.)
    worker.update(actions(2, .002), .02)
    fused = worker.update(actions(4, .004), .04)
    assert fused[0, 0] == pytest.approx((.005 + .007 + .009) / 3)
    assert worker.last_result['contributors'][:4] == [3] * 4


def test_recent_prediction_has_more_weight_using_age_in_control_ticks():
    worker = ensemble(decay=.1)
    worker.update(actions(), 0.)
    fused = worker.update(actions(2, .002), .02)
    weight = math.exp(-.1 * 2)
    assert fused[0, 0] == pytest.approx((.003 * weight + .005) / (weight + 1))


@pytest.mark.parametrize('timestamp', [.025, .1])
def test_nonmatching_times_and_expired_chunks_do_not_contribute(timestamp):
    worker = ensemble()
    worker.update(actions(), 0.)
    latest = actions(2, .002)
    np.testing.assert_array_equal(worker.update(latest, timestamp), latest)
    assert worker.last_result['contributors'] == [1] * 8


def test_rotation_mean_handles_quaternion_signs_and_pi_wrap():
    worker = ensemble(decay=0.)
    first, second = actions(), actions(2)
    first[:, 3:7] = Rotation.from_euler('z', -179, degrees=True).as_quat()
    second[:, 3:7] = -Rotation.from_euler('z', 179, degrees=True).as_quat()
    worker.update(first, 0.)
    result = worker.update(second, .02)
    expected = Rotation.from_euler('z', 180, degrees=True)
    assert (Rotation.from_quat(result[0, 3:7]) * expected.inv()).magnitude() < 1e-6
    assert np.dot(result[0, 3:7], second[0, 3:7]) > 0
    np.testing.assert_array_equal(result[:, 7], second[:, 7])


@pytest.mark.parametrize('change', ['position', 'rotation', 'gripper'])
def test_inconsistent_predictions_remain_raw_instead_of_masking_a_new_mode_or_outlier(change):
    worker = ensemble()
    worker.update(actions(), 0.)
    latest = actions(2)
    if change == 'position':
        latest[:, 0] += .15
    elif change == 'rotation':
        latest[:, 3:7] = Rotation.from_euler('z', 30, degrees=True).as_quat()
    else:
        latest[:, 7] = 0
    np.testing.assert_allclose(worker.update(latest, .02), latest, atol=1e-7)
    assert worker.last_result['contributors'] == [1] * 8
    assert worker.last_result['rejected_inconsistent'] == 6


def test_reset_discards_previous_episode_and_failed_updates_do_not_commit_state():
    worker = ensemble()
    worker.update(actions(), 0.)
    for timestamp in [0., -.1, float('nan'), float('inf')]:
        with pytest.raises(ValueError, match='timestamps'):
            worker.update(actions(), timestamp)
    broken = actions()
    broken[0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        worker.update(broken, .02)
    worker.update(actions(2), .02)
    assert worker.last_result['contributors'][0] == 2
    worker.reset()
    assert worker.last_result is None
    np.testing.assert_array_equal(worker.update(actions(), 0.), actions())
    assert worker.last_result['contributors'] == [1] * 8


@pytest.mark.parametrize('patch', [dict(decay=-1), dict(decay=float('nan')),
    dict(max_translation=0), dict(max_rotation=float('inf'))])
def test_invalid_options_are_rejected(patch):
    with pytest.raises(ValueError, match='agreement limits'):
        ensemble(**patch)
