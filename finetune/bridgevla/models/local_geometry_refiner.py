"""A small, offline pose residual head over an observed local geometry field.

The five field channels are signed projective distance, surface support, free
support, known, and conflict. This module never backpropagates into that field;
it does not complete unobserved geometry or change the upstream perception.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from ..geometry.torch_ops import (
    local_to_world_points,
    rotation_matrix_error_rad,
    rotation_matrix_squared_error,
    so3_exp as _so3_exp,
    validate_floating_tensor as _validate_tensor,
    validate_rigid_transform,
)


def _positive_scale(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite positive number')
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a finite positive number')
    return value


def _fraction(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a finite number in [0, 1]')
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{name} must be a finite number in [0, 1]')
    return value


def _validate_pose(pose, name):
    _validate_tensor(pose, name)
    if pose.ndim != 3 or pose.shape[1:] != (4, 4) or pose.shape[0] == 0:
        raise ValueError(f'{name} must have shape [B,4,4] with B > 0')
    validate_rigid_transform(pose, name)


def _validate_pose_pair(predicted_pose, target_pose):
    _validate_pose(predicted_pose, 'predicted_pose')
    _validate_pose(target_pose, 'target_pose')
    if (predicted_pose.shape != target_pose.shape
            or predicted_pose.device != target_pose.device
            or predicted_pose.dtype != target_pose.dtype):
        raise ValueError('predicted_pose and target_pose must have matching shape, device, and dtype')


def _bounded_vector(vector, limit):
    """Stable limit * v/sqrt(1+||v||^2), with a rounding safety margin.

    Rescaling before squaring avoids overflow even for very large finite head
    outputs. The small margin keeps the result strictly below the configured
    limit after floating point rounding; the derivative at zero remains nonzero.
    """
    scale = vector.abs().amax(dim=-1, keepdim=True).clamp_min(1.)
    scaled = vector / scale
    denominator = ((1 / scale).square() + scaled.square().sum(-1, keepdim=True)).sqrt()
    safe_limit = limit * (1 - 8 * torch.finfo(vector.dtype).eps)
    return safe_limit * scaled / denominator


class LocalGeometryRefiner(nn.Module):
    """Predict bounded pose corrections in the candidate gripper frame.

    ``p = p_candidate + R_candidate @ translation_local`` and
    ``R = R_candidate @ Exp(rotation_vector)``. Translation is an ordinary
    local displacement, not the translational coordinate of an SE(3) twist.
    A zero initialized output head exactly preserves the candidate pose. A
    sample is corrected only when its field has enough evidence: the fraction
    of voxels with a known indicator must reach ``min_known_coverage`` and the
    fraction with near-surface support must reach ``min_surface_support``
    (and at least one known voxel is always required). Otherwise the residuals
    are exactly zero. Free-space-only views therefore never move the pose.
    """

    def __init__(self, context_dim: int, hidden_dim=32,
                 max_translation_m=.02, max_rotation_rad=.15,
                 min_known_coverage=.02, min_surface_support=.002):
        super().__init__()
        if isinstance(context_dim, bool) or not isinstance(context_dim, int) or context_dim < 0:
            raise ValueError('context_dim must be a nonnegative integer')
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim < 1:
            raise ValueError('hidden_dim must be a positive integer')
        self.context_dim = context_dim
        self.max_translation_m = _positive_scale(max_translation_m, 'max_translation_m')
        self.max_rotation_rad = _positive_scale(max_rotation_rad, 'max_rotation_rad')
        self.min_known_coverage = _fraction(min_known_coverage, 'min_known_coverage')
        self.min_surface_support = _fraction(min_surface_support, 'min_surface_support')
        self.geometry_encoder = nn.Sequential(
            nn.Conv3d(5, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)),
            nn.Flatten(start_dim=1),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 8 + context_dim, hidden_dim * 2),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden_dim * 2, 6)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, field, candidate_pose, context):
        _validate_tensor(field, 'field')
        if (field.ndim != 5 or field.shape[1] != 5 or field.shape[0] == 0
                or min(field.shape[2:]) < 1):
            raise ValueError('field must have shape [B,5,D,H,W] with nonempty dimensions')
        _validate_pose(candidate_pose, 'candidate_pose')
        _validate_tensor(context, 'context')
        batch = field.shape[0]
        if candidate_pose.shape[0] != batch:
            raise ValueError('candidate_pose batch must match field batch')
        if context.shape != (batch, self.context_dim):
            raise ValueError(f'context must have shape [B,{self.context_dim}] matching field batch')
        parameter = self.output.weight
        if any(value.device != parameter.device or value.dtype != parameter.dtype
               for value in (field, candidate_pose, context)):
            raise ValueError('field, candidate_pose, and context must match the model device and dtype')

        observed = field.detach()
        known_voxels = observed[:, 3] > 0
        known_coverage = known_voxels.flatten(1).to(observed.dtype).mean(dim=1, keepdim=True)
        surface_support = (observed[:, 1] > 0).flatten(1).to(observed.dtype).mean(dim=1, keepdim=True)
        allowed = (known_voxels.flatten(1).any(dim=1, keepdim=True)
                   & (known_coverage >= self.min_known_coverage)
                   & (surface_support >= self.min_surface_support))
        geometry = self.geometry_encoder(observed)
        raw = self.output(self.fusion(torch.cat((geometry, context), dim=-1)))
        translation = _bounded_vector(raw[:, :3], self.max_translation_m)
        rotation_vector = _bounded_vector(raw[:, 3:], self.max_rotation_rad)
        translation = torch.where(allowed, translation, torch.zeros_like(translation))
        rotation_vector = torch.where(allowed, rotation_vector, torch.zeros_like(rotation_vector))

        candidate_rotation = candidate_pose[:, :3, :3]
        rotation = candidate_rotation @ _so3_exp(rotation_vector)
        position = local_to_world_points(translation, candidate_pose[:, :3, 3], candidate_rotation)
        pose = torch.cat((torch.cat((rotation, position[..., None]), dim=-1), candidate_pose[:, 3:4]), dim=1)
        return {'pose': pose, 'translation_local': translation, 'rotation_vector': rotation_vector,
                'refine_allowed': allowed[:, 0], 'known_coverage': known_coverage[:, 0],
                'surface_support': surface_support[:, 0]}


def pose_refinement_loss(predicted_pose, target_pose,
                         translation_scale_m=.01, rotation_scale_rad=.1):
    """Mean SmoothL1 translation plus a smooth chordal rotation surrogate.

    Translation is normalized by its metric scale. The rotation term is
    ``||R_pred-R_target||_F^2/(4*rotation_scale_rad^2)``, which approaches
    ``0.5*(angle/rotation_scale_rad)^2`` near zero. This is not an exact angular
    loss; its finite derivatives avoid the acos singularities at zero and pi.
    """
    translation_scale_m = _positive_scale(translation_scale_m, 'translation_scale_m')
    rotation_scale_rad = _positive_scale(rotation_scale_rad, 'rotation_scale_rad')
    _validate_pose_pair(predicted_pose, target_pose)
    offset = (predicted_pose[:, :3, 3] - target_pose[:, :3, 3]) / translation_scale_m
    translation = F.smooth_l1_loss(offset, torch.zeros_like(offset), reduction='none').sum(-1)
    rotation = .25 * rotation_matrix_squared_error(
        predicted_pose[:, :3, :3], target_pose[:, :3, :3]) / rotation_scale_rad ** 2
    return (translation + rotation).mean()


def pose_error_metrics(predicted_pose, target_pose):
    """Per-sample metric translation and geodesic SO(3) error in [0, pi]."""
    _validate_pose_pair(predicted_pose, target_pose)
    translation = torch.linalg.vector_norm(predicted_pose[:, :3, 3] - target_pose[:, :3, 3], dim=-1)
    rotation = rotation_matrix_error_rad(predicted_pose[:, :3, :3], target_pose[:, :3, :3])
    return {'translation_m': translation, 'rotation_rad': rotation}
