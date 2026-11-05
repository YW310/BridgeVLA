"""Shared differentiable Torch pose operations with explicit frame conventions.

Quaternions use xyzw. Point transforms take a local-to-world rotation and a
world-frame origin. These primitives do not define a policy's action units,
normalization, temporal accumulation, or loss reduction.
"""

import math

import torch
from torch.nn import functional as F


def _floating_metadata(value, name):
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise ValueError(f'{name} must be a floating point tensor')


def validate_floating_tensor(value, name):
    """Check a finite floating tensor and return the original tensor.

    Call at data/model boundaries. Pure mathematical operations below check
    metadata only, avoiding repeated device synchronizations in policy paths.
    """
    _floating_metadata(value, name)
    if not torch.isfinite(value).all():
        raise ValueError(f'{name} must contain only finite values')
    return value


def validate_rigid_transform(value, name, *, rotation_atol=1e-4,
                             rotation_rtol=0., homogeneous_atol=1e-4):
    """Check nonempty [...,4,4] SE(3) transforms, returning ``value`` itself.

    Validation does not detach the returned value or change its dtype/device.
    Half precision is checked in float32 so CPU determinant is supported.
    """
    validate_floating_tensor(value, name)
    if value.ndim < 2 or value.shape[-2:] != (4, 4) or value.numel() == 0:
        raise ValueError(f'{name} must have nonempty shape [...,4,4]')
    checked = value.detach()
    if checked.dtype in (torch.float16, torch.bfloat16):
        checked = checked.float()
    bottom = checked.new_tensor([0., 0., 0., 1.]).expand_as(checked[..., 3, :])
    rotation = checked[..., :3, :3]
    identity = torch.eye(3, dtype=checked.dtype, device=checked.device).expand_as(rotation)
    if (not torch.allclose(checked[..., 3, :], bottom, atol=homogeneous_atol, rtol=0)
            or not torch.allclose(rotation.transpose(-1, -2) @ rotation, identity,
                                  atol=rotation_atol, rtol=rotation_rtol)
            or not torch.allclose(torch.linalg.det(rotation), torch.ones_like(checked[..., 0, 0]),
                                  atol=rotation_atol, rtol=rotation_rtol)):
        raise ValueError(f'{name} must contain rigid SE(3) poses (proper rotations and homogeneous row)')
    return value


def so3_exp(rotation_vector):
    """Rodrigues exponential [...,3] -> [...,3,3], stable at zero and pi."""
    _floating_metadata(rotation_vector, 'rotation_vector')
    if rotation_vector.shape[-1:] != (3,):
        raise ValueError('rotation_vector must have shape [...,3]')
    x, y, z = rotation_vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1)
    skew = skew.reshape(*rotation_vector.shape[:-1], 3, 3)
    theta = torch.linalg.vector_norm(rotation_vector, dim=-1)
    a = torch.sinc(theta / math.pi)[..., None, None]
    b = (.5 * torch.sinc(theta / (2 * math.pi)).square())[..., None, None]
    identity = torch.eye(3, dtype=rotation_vector.dtype, device=rotation_vector.device)
    return identity + a * skew + b * (skew @ skew)


def quaternion_matrix(quaternion):
    """xyzw [...,4] -> [...,3,3], preserving OHT's F.normalize convention.

    Measured quaternion validity is checked at the observation boundary. This
    low-level conversion intentionally retains the legacy zero-norm behavior.
    """
    _floating_metadata(quaternion, 'quaternion')
    if quaternion.shape[-1:] != (4,):
        raise ValueError('quaternion must have shape [...,4] in xyzw order')
    q = F.normalize(quaternion, dim=-1)
    x, y, z, w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def rotation_6d(value):
    """Two column vectors [...,6] -> [...,3,3], with legacy OHT epsilon.

    Collinear/zero columns remain degenerate; this function does not silently
    substitute another rotation or alter the decoder's training behavior.
    """
    _floating_metadata(value, 'rotation_6d')
    if value.shape[-1:] != (6,):
        raise ValueError('rotation_6d must have shape [...,6]')
    first = F.normalize(value[..., :3], dim=-1, eps=1e-8)
    second = F.normalize(value[..., 3:] - (first * value[..., 3:]).sum(-1, keepdim=True) * first,
                         dim=-1, eps=1e-8)
    return torch.stack((first, second, torch.linalg.cross(first, second)), dim=-1)


def _point_transform_inputs(points, origin, rotation):
    for value, name in ((points, 'points'), (origin, 'origin'), (rotation, 'rotation')):
        _floating_metadata(value, name)
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError('points must have shape [B,...,3]')
    batch = points.shape[0]
    if origin.shape != (batch, 3) or rotation.shape != (batch, 3, 3):
        raise ValueError('origin [B,3] and rotation [B,3,3] must match points batch')
    if any(value.device != points.device for value in (origin, rotation)):
        raise ValueError('points, origin, and rotation must have matching device')
    sample_axes = (1,) * (points.ndim - 2)
    return origin.reshape(batch, *sample_axes, 3), rotation.reshape(batch, *sample_axes, 3, 3)


def world_to_local_points(points, origin, rotation):
    """Apply R^T(points-origin), retaining native Torch promotion/autocast."""
    origin, rotation = _point_transform_inputs(points, origin, rotation)
    return (rotation.transpose(-1, -2) @ (points - origin)[..., None]).squeeze(-1)


def local_to_world_points(points, origin, rotation):
    """Apply origin+R@points, retaining native Torch promotion/autocast."""
    origin, rotation = _point_transform_inputs(points, origin, rotation)
    return origin + (rotation @ points[..., None]).squeeze(-1)


def _rotation_pair(predicted, target):
    _floating_metadata(predicted, 'predicted rotation')
    _floating_metadata(target, 'target rotation')
    if predicted.ndim < 2 or predicted.shape[-2:] != (3, 3) or predicted.shape != target.shape:
        raise ValueError('rotations must have matching shape [...,3,3]')
    if predicted.dtype != target.dtype or predicted.device != target.device:
        raise ValueError('rotations must have matching dtype and device')


def rotation_matrix_squared_error(predicted, target):
    """Return per-matrix squared Frobenius distance, with no reduction."""
    _rotation_pair(predicted, target)
    return (predicted - target).square().sum(dim=(-2, -1))


def rotation_matrix_error_rad(predicted, target):
    """Return geodesic SO(3) error in [0, pi] using stable atan2 values."""
    _rotation_pair(predicted, target)
    relative = predicted.transpose(-1, -2) @ target
    skew = .5 * torch.stack((relative[..., 2, 1] - relative[..., 1, 2],
                            relative[..., 0, 2] - relative[..., 2, 0],
                            relative[..., 1, 0] - relative[..., 0, 1]), dim=-1)
    sine = torch.linalg.vector_norm(skew, dim=-1)
    cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * .5).clamp(-1., 1.)
    return torch.atan2(sine, cosine)
