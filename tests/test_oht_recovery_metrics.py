import json

import numpy as np
import pytest

from finetune.OHT.recovery_metrics import paired_recovery


def pose(position=(0.0, 0.0, 0.0), angle=0.0):
    return np.r_[position, 0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)]


def example():
    base = pose()
    perturbed = pose((0.0, 0.005, 0.0))
    actions = np.array([np.r_[pose((0.001 * h, 0.0, 0.0)), 1.0]
                        for h in range(1, 5)])
    path = np.array([pose((-0.01, 0.0, 0.0)), pose((0.01, 0.0, 0.0))])
    return [base, perturbed, actions, actions.copy(), path]


def test_translation_following_has_zero_recovery_gain_and_serializable_output():
    args = example()
    args[3][:, :3] += args[1][:3] - args[0][:3]
    before = [array.copy() for array in args]
    result = paired_recovery(*args)
    assert result["intended_output_only"] is True
    assert result["rotation_gain_is_local"] is True
    assert [item["prefix"] for item in result["prefixes"]] == [1, 2, 3, 4]
    for item in result["prefixes"]:
        assert item["position"]["gain"] == pytest.approx(0.0)
        assert item["position"]["normal_gain"] == pytest.approx(0.0)
        np.testing.assert_allclose(item["position"]["delta_u_world_m"], 0.0)
        assert item["rotation"]["gain"] is None
    json.dumps(result, allow_nan=False)
    for original, copied in zip(args, before):
        np.testing.assert_array_equal(original, copied)


def test_world_fixed_endpoints_have_unit_recovery_gain():
    result = paired_recovery(*example())
    for item in result["prefixes"]:
        assert item["position"]["gain"] == pytest.approx(1.0)
        assert item["position"]["normal_gain"] == pytest.approx(1.0)
        np.testing.assert_allclose(item["position"]["delta_u_world_m"], [0.0, -0.005, 0.0])


@pytest.mark.parametrize("remaining_fraction, expected", [(0.75, 0.25), (1.5, -0.5)])
def test_partial_correction_and_amplification(remaining_fraction, expected):
    args = example()
    args[3][:, 1] += remaining_fraction * 0.005
    result = paired_recovery(*args)
    assert result["prefixes"][-1]["position"]["gain"] == pytest.approx(expected)
    assert result["prefixes"][-1]["position"]["normal_gain"] == pytest.approx(expected)


def test_pure_tangential_offset_has_no_normal_gain():
    args = example()
    args[1] = pose((0.005, 0.0, 0.0))
    result = paired_recovery(*args)
    assert result["prefixes"][-1]["position"]["gain"] == pytest.approx(1.0)
    assert result["prefixes"][-1]["position"]["normal_gain"] is None
    np.testing.assert_allclose(result["reference_tangent_world"], [1.0, 0.0, 0.0])


def test_path_distance_uses_continuous_segments_and_clamps_at_endpoints():
    args = example()
    args[0] = pose((0.0, 0.3, 0.0))
    args[1] = pose((2.0, 0.0, 0.0))
    args[2][:, :3] = [0.0, 0.3, 0.0]
    args[3][:, :3] = [2.0, 0.0, 0.0]
    args[4] = np.array([pose((-1.0, 0.0, 0.0)), pose((1.0, 0.0, 0.0))])
    result = paired_recovery(*args)
    assert result["initial"]["path_distance_m"] == pytest.approx({"base": 0.3, "perturbed": 1.0})
    assert result["prefixes"][-1]["path_distance_m"] == pytest.approx({"base": 0.3, "perturbed": 1.0})


def test_zero_length_segment_and_one_point_path_are_supported():
    args = example()
    args[4] = np.array([pose(), pose(), pose((0.01, 0.0, 0.0))])
    result = paired_recovery(*args)
    assert result["initial"]["path_distance_m"]["perturbed"] == pytest.approx(0.005)
    args[4] = np.array([pose()])
    result = paired_recovery(*args)
    assert result["reference_tangent_world"] is None
    assert result["prefixes"][-1]["position"]["normal_gain"] is None
    assert result["prefixes"][-1]["path_distance_m"]["base"] == pytest.approx(0.004)


def test_rotation_following_and_fixed_endpoint_are_quaternion_sign_invariant():
    args = example()
    args[1] = pose(angle=0.02)
    args[2][:, 3:7] = pose(angle=0.01)[3:7]
    args[3][:, 3:7] = pose(angle=0.03)[3:7]
    following = paired_recovery(*args)
    for item in following["prefixes"]:
        assert item["rotation"]["gain"] == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(item["rotation"]["e_world_rad"], [0.0, 0.0, 0.02])
    args[0][3:7] *= -1
    args[1][3:7] *= -1
    args[2][::2, 3:7] *= -1
    args[3][1::2, 3:7] *= -1
    args[4][:, 3:7] *= -1
    signed = paired_recovery(*args)
    assert signed == following
    args[3][:, 3:7] = args[2][:, 3:7]
    fixed = paired_recovery(*args)
    assert fixed["prefixes"][-1]["rotation"]["gain"] == pytest.approx(1.0)


def test_pi_rotation_has_deterministic_axis_under_quaternion_sign_change():
    args = example()
    args[1][3:7] = [0.0, 0.0, 1.0, 0.0]
    result = paired_recovery(*args)
    args[1][3:7] *= -1
    assert paired_recovery(*args) == result
    assert result["initial"]["rotation_error_rad"] == pytest.approx(np.pi)


def test_rotation_error_is_expressed_in_world_axes_for_rotated_base():
    args = example()
    sine = np.sqrt(0.5)
    args[0][3:7] = [sine, 0.0, 0.0, sine]
    # R_perturbed = Rz(0.02) Rx(pi/2), so the spatial error is around world z.
    c, s = np.cos(0.01), np.sin(0.01)
    args[1][3:7] = sine * np.array([c, s, s, c])
    result = paired_recovery(*args)
    np.testing.assert_allclose(result["initial"]["rotation_error_world_rad"],
                               [0.0, 0.0, 0.02], atol=1e-14)
    assert result["prefixes"][-1]["rotation"]["gain"] == pytest.approx(1.0)


def test_small_errors_and_small_reference_tangent_return_null_gains():
    args = example()
    args[1] = pose((0.0005, 0.0, 0.0), angle=np.deg2rad(0.05))
    args[4] = np.array([pose(), pose((0.0005, 0.0, 0.0))])
    result = paired_recovery(*args)
    assert result["reference_tangent_world"] is None
    for item in result["prefixes"]:
        assert item["position"]["gain"] is None
        assert item["position"]["normal_gain"] is None
        assert item["rotation"]["gain"] is None
    args[1] = args[0].copy()
    zero = paired_recovery(*args, position_floor_m=0.0, rotation_floor_rad=0.0)
    assert zero["prefixes"][-1]["position"]["gain"] is None
    assert zero["prefixes"][-1]["rotation"]["gain"] is None


def test_requested_horizon_selects_prefixes_without_mutating_chunk():
    result = paired_recovery(*example(), execute_horizon=np.int64(2))
    assert result["execute_horizon"] == 2
    assert len(result["prefixes"]) == 2


@pytest.mark.parametrize("horizon", [0, -1, 5, 1.0, True, None])
def test_invalid_horizon_is_rejected(horizon):
    with pytest.raises(ValueError, match="execute_horizon"):
        paired_recovery(*example(), execute_horizon=horizon)


@pytest.mark.parametrize("name,value", [
    ("position_floor_m", -1.0), ("position_floor_m", np.nan),
    ("rotation_floor_rad", np.inf), ("rotation_floor_rad", True),
])
def test_invalid_floors_are_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        paired_recovery(*example(), **{name: value})


@pytest.mark.parametrize("index,bad", [
    (0, np.zeros(6)), (1, np.zeros((1, 7))), (2, np.zeros((4, 7))),
    (3, np.empty((0, 8))), (4, np.empty((0, 7))), (4, np.zeros((2, 8))),
])
def test_invalid_shapes_are_rejected(index, bad):
    args = example()
    args[index] = bad
    with pytest.raises(ValueError, match="shape"):
        paired_recovery(*args)


def test_mismatched_chunk_lengths_are_rejected():
    args = example()
    args[3] = args[3][:3]
    with pytest.raises(ValueError, match="matching shapes"):
        paired_recovery(*args)


@pytest.mark.parametrize("index", range(5))
@pytest.mark.parametrize("problem", ["nonfinite", "nonunit"])
def test_all_pose_sources_require_finite_values_and_unit_quaternions(index, problem):
    args = example()
    row = args[index] if args[index].ndim == 1 else args[index][0]
    if problem == "nonfinite":
        row[0] = np.nan
    else:
        row[3:7] *= 2.0
    with pytest.raises(ValueError, match="finite" if problem == "nonfinite" else "unit xyzw"):
        paired_recovery(*args)


def test_gripper_scalar_must_also_be_finite():
    args = example()
    args[2][0, 7] = np.inf
    with pytest.raises(ValueError, match="finite"):
        paired_recovery(*args)
