"""Cross-check shared pose primitives against SciPy and legacy Torch values."""

import math

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from torch.nn import functional as F

from finetune.bridgevla.geometry.torch_ops import (
    local_to_world_points,
    quaternion_matrix,
    rotation_6d,
    rotation_matrix_error_rad,
    rotation_matrix_squared_error,
    so3_exp,
    validate_floating_tensor,
    validate_rigid_transform,
    world_to_local_points,
)


def test_oht_conversion_exports_point_to_shared_functions():
    from finetune.OHT import chunk_model
    assert chunk_model.quaternion_matrix is quaternion_matrix
    assert chunk_model.rotation_6d is rotation_6d


def test_quaternion_and_six_dimensional_columns_match_scipy_and_keep_sign_invariance():
    rotations = Rotation.from_euler('xyz', [[30., 70., -20.], [180., 0., 0.], [0., 0., 90.]], degrees=True)
    quaternion = torch.tensor(rotations.as_quat(), dtype=torch.float64).reshape(1, 3, 4)
    matrix = quaternion_matrix(quaternion * 3)
    np.testing.assert_allclose(matrix.numpy()[0], rotations.as_matrix(), atol=1e-12)
    torch.testing.assert_close(quaternion_matrix(-quaternion), quaternion_matrix(quaternion), rtol=0, atol=0)
    columns = torch.cat((matrix[..., :, 0], matrix[..., :, 1]), dim=-1)
    torch.testing.assert_close(rotation_6d(columns), matrix, rtol=1e-12, atol=1e-12)


def test_oht_normalization_epsilons_and_degenerate_behavior_are_preserved():
    value = torch.tensor([[0., 0., 0., 0., 0., 0.], [1e-10, 0., 0., 0., 1e-10, 0.],
                          [2., 0., 0., 3., 0., 0.], [2., 3., 4., -1., 2., 3.]])
    first = F.normalize(value[..., :3], dim=-1, eps=1e-8)
    second = F.normalize(value[..., 3:] - (first * value[..., 3:]).sum(-1, keepdim=True) * first,
                         dim=-1, eps=1e-8)
    legacy = torch.stack((first, second, torch.linalg.cross(first, second)), dim=-1)
    torch.testing.assert_close(rotation_6d(value), legacy, rtol=0, atol=0)
    assert torch.linalg.det(rotation_6d(value)[0]) == 0
    # Observation boundaries reject zero measured quaternions. The shared
    # low-level converter preserves the old F.normalize zero-input result.
    torch.testing.assert_close(quaternion_matrix(torch.zeros(4)), torch.eye(3), rtol=0, atol=0)
    tiny = torch.tensor([1e-14, 0., 0., 0.])
    expected = torch.diag(torch.tensor([1., 1. - 2e-4, 1. - 2e-4]))
    torch.testing.assert_close(quaternion_matrix(tiny), expected)


@pytest.mark.parametrize('sample_shape', [(), (5,), (2, 3, 4)])
def test_point_transform_shapes_match_scipy_and_roundtrip(sample_shape):
    generator = torch.Generator().manual_seed(8)
    points = torch.randn((2, *sample_shape, 3), dtype=torch.float64, generator=generator)
    origin = torch.randn(2, 3, dtype=torch.float64, generator=generator)
    rotation = Rotation.from_euler('xyz', [[35., -20., 80.], [-60., 40., -30.]], degrees=True)
    matrix = torch.tensor(rotation.as_matrix(), dtype=torch.float64)
    world = local_to_world_points(points, origin, matrix)
    assert world.shape == points.shape
    for batch in range(2):
        expected = rotation[batch].apply(points[batch].numpy().reshape(-1, 3)) + origin[batch].numpy()
        np.testing.assert_allclose(world[batch].numpy(), expected.reshape(points[batch].shape), atol=1e-12)
    torch.testing.assert_close(world_to_local_points(world, origin, matrix), points, rtol=1e-12, atol=1e-12)


def test_point_transforms_keep_gradients_for_points_origin_and_orientation():
    points = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
    origin = torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    omega = torch.tensor([[.1, .2, -.1], [-.2, .05, .1]], dtype=torch.float64, requires_grad=True)
    transform = lambda value, center, vector: world_to_local_points(value, center, so3_exp(vector))
    assert torch.autograd.gradcheck(transform, (points, origin, omega), atol=1e-6)
    local_to_world_points(points, origin, so3_exp(omega)).square().sum().backward()
    for value in (points, origin, omega):
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() > 0


@pytest.mark.parametrize('angle', [0., 1e-10, .3, math.pi - 1e-7, math.pi])
def test_exponential_angular_error_and_chordal_distance_match_known_rotations(angle):
    axis = np.array([1., 2., -3.])
    axis /= np.linalg.norm(axis)
    omega = torch.tensor(axis * angle, dtype=torch.float64, requires_grad=True)
    matrix = so3_exp(omega)
    np.testing.assert_allclose(matrix.detach().numpy(), Rotation.from_rotvec(axis * angle).as_matrix(), atol=1e-12)
    identity = torch.eye(3, dtype=torch.float64)
    angular = rotation_matrix_error_rad(matrix, identity)
    chordal = rotation_matrix_squared_error(matrix, identity)
    torch.testing.assert_close(angular, torch.tensor(angle, dtype=torch.float64), rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(chordal, torch.tensor(8 * math.sin(angle / 2) ** 2, dtype=torch.float64),
                               rtol=1e-9, atol=1e-12)
    (angular + chordal).backward()
    assert torch.isfinite(omega.grad).all()


def test_rotation_errors_preserve_leading_dimensions_and_mse_normalization():
    omega = torch.linspace(-.3, .3, 18).reshape(2, 3, 3)
    predicted, target = so3_exp(omega), torch.eye(3).expand(2, 3, 3, 3)
    squared = rotation_matrix_squared_error(predicted, target)
    assert squared.shape == (2, 3)
    assert rotation_matrix_error_rad(predicted, target).shape == (2, 3)
    torch.testing.assert_close(squared.mean() / 9, F.mse_loss(predicted, target))


def test_rigid_validation_supports_multiple_batch_axes_and_returns_original_tensor():
    value = torch.eye(4).repeat(2, 3, 1, 1).requires_grad_()
    checked = validate_rigid_transform(value, 'camera_to_world')
    assert checked is value
    checked[..., :3, 3].sum().backward()
    assert value.grad is not None
    assert validate_floating_tensor(value, 'pose') is value
    validate_rigid_transform(torch.eye(4, dtype=torch.float16), 'half_pose')


def test_rigid_validation_preserves_explicit_tolerance_choices():
    value = torch.eye(4)
    value[3, 0] = 5e-5
    validate_rigid_transform(value, 'pose')
    with pytest.raises(ValueError, match='rigid'):
        validate_rigid_transform(value, 'pose', homogeneous_atol=1e-5)
    value = torch.eye(4)
    value[0, 0] += 6e-5
    with pytest.raises(ValueError, match='rigid'):
        validate_rigid_transform(value, 'pose')
    validate_rigid_transform(value, 'pose', rotation_rtol=1e-4)


@pytest.mark.parametrize('kind', ['reflection', 'scale', 'row', 'nan', 'integer', 'shape', 'empty'])
def test_invalid_rigid_transforms_are_rejected(kind):
    value = torch.eye(4)
    if kind == 'reflection':
        value[0, 0] = -1
    elif kind == 'scale':
        value[1, 1] = 2
    elif kind == 'row':
        value[3, 2] = .1
    elif kind == 'nan':
        value[1, 3] = float('nan')
    elif kind == 'integer':
        value = value.long()
    elif kind == 'shape':
        value = torch.eye(3)
    elif kind == 'empty':
        value = torch.empty(0, 4, 4)
    with pytest.raises(ValueError):
        validate_rigid_transform(value, 'pose')


@pytest.mark.parametrize('kind', ['point_shape', 'origin_batch', 'rotation_shape', 'device'])
def test_point_transform_contract_is_checked(kind):
    points, origin, rotation = torch.zeros(2, 4, 3), torch.zeros(2, 3), torch.eye(3).repeat(2, 1, 1)
    if kind == 'point_shape':
        points = torch.zeros(2, 4, 2)
    elif kind == 'origin_batch':
        origin = torch.zeros(1, 3)
    elif kind == 'rotation_shape':
        rotation = torch.eye(3)
    elif kind == 'device':
        origin = origin.to('meta')
    for transform in (local_to_world_points, world_to_local_points):
        with pytest.raises(ValueError):
            transform(points, origin, rotation)


def test_point_transforms_preserve_legacy_autocast_with_float32_measured_state():
    torch.manual_seed(54)
    points = torch.randn(2, 4, 3, dtype=torch.bfloat16)
    origin = torch.randn(2, 3)
    rotation = quaternion_matrix(torch.randn(2, 4))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        expected_world = torch.einsum('bij,bkj->bki', rotation, points) + origin[:, None]
        expected_local = torch.einsum('bij,bnj->bni', rotation.transpose(-1, -2),
                                      points - origin[:, None])
        actual_world = local_to_world_points(points, origin, rotation)
        actual_local = world_to_local_points(points, origin, rotation)
    assert actual_world.dtype == expected_world.dtype == torch.float32
    assert actual_local.dtype == expected_local.dtype == torch.bfloat16
    torch.testing.assert_close(actual_world, expected_world, atol=0, rtol=0)
    torch.testing.assert_close(actual_local, expected_local, atol=0, rtol=0)


def test_explicit_validator_rejects_nonfinite_while_math_ops_propagate_it():
    points, origin, rotation = torch.zeros(1, 2, 3), torch.zeros(1, 3), torch.eye(3)[None]
    points[0, 0, 0] = float('inf')
    with pytest.raises(ValueError, match='finite'):
        validate_floating_tensor(points, 'points')
    for transform in (local_to_world_points, world_to_local_points):
        assert not torch.isfinite(transform(points, origin, rotation)).all()
    assert not torch.isfinite(quaternion_matrix(torch.full((4,), float('nan')))).all()
    assert not torch.isfinite(rotation_6d(torch.full((6,), float('nan')))).all()


def test_representation_and_rotation_error_shapes_are_checked():
    for transform, value in ((so3_exp, torch.zeros(2, 4)), (quaternion_matrix, torch.zeros(2, 3)),
                             (rotation_6d, torch.zeros(2, 7))):
        with pytest.raises(ValueError, match='shape'):
            transform(value)
    for error in (rotation_matrix_squared_error, rotation_matrix_error_rad):
        with pytest.raises(ValueError, match='matching'):
            error(torch.eye(3), torch.eye(3).repeat(2, 1, 1))
