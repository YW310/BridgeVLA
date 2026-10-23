"""V4 absolute, observed-state chunks: timing, masks and event fidelity."""
from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from finetune.OHT.data.chunk_actions import (
    SCHEMA, chunk_options, make_chunk_targets, resolve_chunk_timing,
    validate_chunk_targets,
)


def trajectory(n=7, *, gripper=None, dt=.1, timestamp=2.):
    poses = np.zeros((n, 7), dtype=np.float64)
    poses[:, :3] = np.asarray([.2, -.3, .5]) + np.arange(n)[:, None] * [.01, .02, -.01]
    poses[:, 3:] = Rotation.from_euler("xyz", [20, 30, 40], degrees=True).as_quat()
    joints = np.arange(n * 6, dtype=np.float64).reshape(n, 6) / 10
    gripper = np.ones(n, dtype=np.int64) if gripper is None else np.asarray(gripper)
    return poses, joints, gripper, timestamp + np.arange(n) * dt


def targets(*, frame=0, keys=(3, 5), options=None, states=None):
    states = trajectory() if states is None else states
    config = resolve_chunk_timing(states[3], options)
    value, reason = make_chunk_targets(*states, frame, keys, config)
    assert reason is None
    row = dict(frame=frame, timestamp=float(states[3][frame]),
               target_frame=int(value["key_frame"]), episode_length=len(states[0]))
    validate_chunk_targets(value, config, row)
    return value, config, row


def test_options_defaults_are_independent_and_schema_is_explicit():
    assert SCHEMA == "oht_bridgevla_v4"
    expected = dict(horizon=16, frame_stride=1, action_dt=None,
                    timestamp_atol=1e-5, timestamp_rtol=1e-4)
    assert chunk_options() == expected
    value = chunk_options()
    value["horizon"] = 99
    assert chunk_options() == expected
    assert chunk_options(dict(timestamp_atol=0, timestamp_rtol=0))["timestamp_atol"] == 0


@pytest.mark.parametrize("options", [
    [], True, {"unknown": 1}, {"horizon": True}, {"frame_stride": np.bool_(True)},
    {"action_dt": False}, {"timestamp_atol": True}, {"timestamp_rtol": False},
    {"horizon": 0}, {"frame_stride": -1}, {"horizon": 1.5}, {"frame_stride": 2.5},
    {"action_dt": 0}, {"action_dt": -1}, {"action_dt": np.inf},
    {"timestamp_atol": -1}, {"timestamp_rtol": np.nan}, {"horizon": "16"},
    {"horizon": 10 ** 400}, {"frame_stride": None},
])
def test_options_reject_invalid_values(options):
    with pytest.raises(ValueError, match="chunk"):
        chunk_options(options)


def test_timing_infers_stride_scaled_median_without_hiding_a_gap():
    stamps = [0, .1, .2, .5, .6, .7]
    config = resolve_chunk_timing(stamps, dict(frame_stride=2))
    assert config["action_dt"] == pytest.approx(.2)
    assert chunk_options()["action_dt"] is None
    assert resolve_chunk_timing(stamps, {**config, "action_dt": .200001})["action_dt"] == .200001
    with pytest.raises(ValueError, match="action_dt.*median"):
        resolve_chunk_timing(stamps, {**config, "action_dt": .1})


@pytest.mark.parametrize("stamps", [
    [], [0], [[0, .1]], [0, np.nan], [0, np.inf], [-.1, 0],
    [0, 0], [0, .2, .1], [0, 1j], [False, True],
])
def test_timing_requires_finite_nonnegative_strictly_increasing_timestamps(stamps):
    with pytest.raises(ValueError, match="timestamps"):
        resolve_chunk_timing(stamps)


def test_future_ticks_start_one_stride_after_anchor_and_keep_absolute_units_and_xyzw():
    states = trajectory(gripper=[1, 1, 0, 0, 1, 1, 1])
    value, _, _ = targets(keys=(0, 5), options=dict(horizon=2, frame_stride=2), states=states)
    np.testing.assert_array_equal(value["action_frames"], [2, 4])
    np.testing.assert_allclose(value["action_timestamps"], [2.2, 2.4])
    np.testing.assert_allclose(value["action_chunk"][:, :7], states[0][[2, 4]])
    np.testing.assert_allclose(value["joint_chunk"][:, :6], states[1][[2, 4]])
    np.testing.assert_array_equal(value["action_chunk"][:, 7], [0, 1])
    assert bool(value["key_valid"]) and int(value["key_frame"]) == 5
    np.testing.assert_allclose(value["key_action"][:7], states[0][5])
    np.testing.assert_allclose(value["key_joint_action"][:6], states[1][5])
    assert value["key_action"][-1] == states[2][5]


def test_anchor_at_key_targets_strictly_next_key_even_beyond_chunk():
    value, _, _ = targets(frame=3, keys=(0, 3, 6), options=dict(horizon=1))
    assert int(value["key_frame"]) == 6
    np.testing.assert_array_equal(value["action_frames"], [4])


@pytest.mark.parametrize("frame,keys", [(0, ()), (5, (3, 5))])
def test_no_event_and_post_key_tail_keep_chunks_without_synthetic_terminal_key(frame, keys):
    value, _, _ = targets(frame=frame, keys=keys, options=dict(horizon=4))
    assert not bool(value["key_valid"])
    assert int(value["key_frame"]) == -1
    np.testing.assert_array_equal(value["key_action"], [0, 0, 0, 0, 0, 0, 1, 0])
    np.testing.assert_array_equal(value["key_joint_action"], np.zeros(7))
    assert value["action_valid"][0]


def test_tail_padding_has_prefix_mask_identity_quaternion_and_exact_sentinels():
    value, _, _ = targets(frame=4, keys=(), options=dict(horizon=4))
    np.testing.assert_array_equal(value["action_valid"], [True, True, False, False])
    np.testing.assert_array_equal(value["action_frames"], [5, 6, -1, -1])
    np.testing.assert_array_equal(value["action_timestamps"][2:], 0)
    np.testing.assert_array_equal(value["action_chunk"][2:], [[0, 0, 0, 0, 0, 0, 1, 0]] * 2)
    np.testing.assert_array_equal(value["joint_chunk"][2:], 0)


@pytest.mark.parametrize("frame,stride", [(6, 1), (5, 2)])
def test_no_actual_future_tick_has_explicit_skip_reason(frame, stride):
    states = trajectory()
    value, reason = make_chunk_targets(*states, frame, [],
        resolve_chunk_timing(states[3], dict(frame_stride=stride)))
    assert value is None and reason == "no_future_target"


def test_stationary_segments_are_preserved_without_interpolation_or_deduplication():
    states = trajectory()
    states[0][:] = states[0][0]
    states[1][:] = states[1][0]
    value, _, _ = targets(keys=(), options=dict(horizon=4), states=states)
    assert value["action_valid"].all()
    np.testing.assert_array_equal(value["action_frames"], [1, 2, 3, 4])
    np.testing.assert_allclose(value["action_chunk"][:, :7], np.tile(states[0][0], (4, 1)))
    np.testing.assert_allclose(value["joint_chunk"][:, :6], np.tile(states[1][0], (4, 1)))


def test_source_quaternions_are_normalized_canonicalized_without_mutating_inputs():
    states = trajectory()
    states[0][:, 3:] *= -3
    before = [a.copy() for a in states]
    value, _, _ = targets(states=states)
    np.testing.assert_allclose(np.linalg.norm(value["action_chunk"][:, 3:7], axis=1), 1, atol=1e-6)
    assert (value["action_chunk"][:, 6] >= 0).all()
    for actual, original in zip(states, before):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("gripper", [
    [1, 0, 1, 1, 1, 1, 1], [1, 0, 1, 0, 1, 1, 1],
])
def test_multiple_raw_gripper_transitions_inside_one_tick_are_rejected(gripper):
    states = trajectory(gripper=gripper)
    config = resolve_chunk_timing(states[3], dict(horizon=2, frame_stride=3))
    assert make_chunk_targets(*states, 0, [1, 2], config) == (None, "gripper_event_aliasing")


def test_single_gripper_transition_is_recorded_at_first_following_tick_never_earlier():
    states = trajectory(gripper=[1, 1, 1, 1, 0, 0, 0])
    value, _, _ = targets(options=dict(horizon=3, frame_stride=2), states=states)
    np.testing.assert_array_equal(value["action_frames"], [2, 4, 6])
    np.testing.assert_array_equal(value["action_chunk"][:, 7], [1, 0, 0])
    np.testing.assert_array_equal(value["joint_chunk"][:, 6], [1, 0, 0])


def test_timestamp_gap_rejects_only_windows_that_cross_it_including_skipped_raw_frames():
    states = trajectory(n=9)
    states[3][3:] += .2
    config = resolve_chunk_timing(states[3], dict(horizon=1, frame_stride=2))
    assert make_chunk_targets(*states, 2, [], config) == (None, "timestamp_gap")
    value, reason = make_chunk_targets(*states, 4, [], config)
    assert reason is None
    np.testing.assert_array_equal(value["action_frames"], [6])


def test_accumulated_timestamp_jitter_must_also_meet_tick_tolerance():
    states = trajectory(n=7, dt=.100009, timestamp=0)
    config = chunk_options(dict(action_dt=.1, horizon=6, timestamp_atol=1e-5, timestamp_rtol=0))
    assert np.isclose(np.diff(states[3]), .1, atol=1e-5, rtol=0).all()
    assert make_chunk_targets(*states, 0, [], config) == (None, "timestamp_gap")


@pytest.mark.parametrize("field,index,value", [
    (0, (2, 3), 0), (0, (2, 6), np.nan), (1, (2, 0), np.inf), (2, 2, .5),
])
def test_referenced_source_values_are_validated(field, index, value):
    states = list(trajectory())
    states[2] = states[2].astype(float)
    if field == 0 and value == 0:
        states[0][2, 3:] = 0
    else:
        states[field][index] = value
    with pytest.raises(ValueError):
        targets(states=states)


def test_key_beyond_chunk_is_also_validated():
    states = trajectory()
    states[0][6, 3:] = 0
    with pytest.raises(ValueError, match="quaternion"):
        targets(keys=(6,), options=dict(horizon=1), states=states)


@pytest.mark.parametrize("keys", [[2, 1], [1, 1], [-1], [7], [1.5], [True], [[1]],
                                  np.asarray([2, 1], dtype=np.uint64)])
def test_invalid_key_indexes_are_rejected(keys):
    with pytest.raises(ValueError, match="keys"):
        targets(keys=keys)


def test_target_archive_is_numeric_and_loadable_without_pickle(tmp_path):
    value, config, row = targets()
    path = tmp_path / "target.npz"
    np.savez_compressed(path, **value)
    with np.load(path, allow_pickle=False) as loaded:
        validate_chunk_targets(loaded, config, row)


@pytest.mark.parametrize("change,match", [
    (lambda v, r: v.pop("key_valid"), "keys"),
    (lambda v, r: v.update(extra=np.zeros(1)), "keys"),
    (lambda v, r: v.update(action_chunk=v["action_chunk"].astype(np.float64)), "float32"),
    (lambda v, r: v.update(action_valid=v["action_valid"].astype(np.int64)), "bool"),
    (lambda v, r: v.update(key_frame=np.asarray(5, dtype=np.int32)), "int64"),
    (lambda v, r: v.update(key_valid=np.asarray([True])), "key_valid"),
    (lambda v, r: v["action_chunk"].__setitem__((0, 0), np.inf), "finite"),
    (lambda v, r: v["action_chunk"].__setitem__((0, slice(3, 7)), 0), "unit xyzw"),
    (lambda v, r: v["action_chunk"].__setitem__((0, slice(3, 7)), [0, 0, 0, 2]), "unit xyzw"),
    (lambda v, r: v["action_chunk"].__setitem__((0, 7), .5), "binary gripper"),
    (lambda v, r: v["joint_chunk"].__setitem__((0, 6), 0), "binary gripper"),
    (lambda v, r: v["action_valid"].__setitem__(0, False), "prefix"),
    (lambda v, r: v["action_valid"].__setitem__(1, False), "prefix"),
    (lambda v, r: v["action_frames"].__setitem__(0, r["frame"]), "future frame"),
    (lambda v, r: v["action_timestamps"].__setitem__(0, r["timestamp"]), "relative"),
    (lambda v, r: v["action_timestamps"].__setitem__(0, 99), "relative"),
    (lambda v, r: v["action_frames"].__setitem__(-1, 0), "sentinels"),
    (lambda v, r: v["action_timestamps"].__setitem__(-1, .1), "sentinels"),
    (lambda v, r: v["action_chunk"].__setitem__((-1, 6), 0), "sentinels"),
    (lambda v, r: v["joint_chunk"].__setitem__((-1, 0), .1), "sentinels"),
    (lambda v, r: v.update(key_frame=np.asarray(-1, dtype=np.int64)), "target_frame"),
    (lambda v, r: v.update(key_valid=np.asarray(False)), "target_frame"),
    (lambda v, r: r.update(target_frame=0), "target_frame"),
    (lambda v, r: r.update(episode_length=2), "inside episode_length"),
    (lambda v, r: r.update(timestamp=np.nan), "row.timestamp"),
])
def test_validator_rejects_corrupt_targets_or_row_metadata(change, match):
    value, config, row = targets(frame=3, keys=(5,), options=dict(horizon=5))
    change(value, row)
    with pytest.raises(ValueError, match=match):
        validate_chunk_targets(value, config, row)


@pytest.mark.parametrize("field,index", [("key_action", 0), ("key_action", 6), ("key_joint_action", 0)])
def test_invalid_key_sentinels_are_exact(field, index):
    value, config, row = targets(keys=())
    value[field][index] += .01
    with pytest.raises(ValueError, match="neutral sentinels"):
        validate_chunk_targets(value, config, row)


def test_validator_rejects_key_outside_episode_even_if_metadata_matches():
    value, config, row = targets(keys=(5,))
    value["key_frame"] = np.asarray(row["episode_length"], dtype=np.int64)
    row["target_frame"] = row["episode_length"]
    with pytest.raises(ValueError, match="future frame within"):
        validate_chunk_targets(value, config, row)


def test_unresolved_timing_is_rejected_before_target_construction_or_validation():
    states = trajectory()
    with pytest.raises(ValueError, match="resolved"):
        make_chunk_targets(*states, 0, [], {})
    value, _, row = targets()
    with pytest.raises(ValueError, match="resolved"):
        validate_chunk_targets(value, {}, row)


def test_validator_does_not_mutate_archive_or_metadata():
    value, config, row = targets()
    saved, saved_config, saved_row = deepcopy(value), config.copy(), row.copy()
    validate_chunk_targets(value, config, row)
    for name in saved:
        np.testing.assert_array_equal(value[name], saved[name])
    assert config == saved_config and row == saved_row
