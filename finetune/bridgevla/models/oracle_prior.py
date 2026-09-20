"""Utilities for the optional O2 GT-instance translation prior.

This module is deliberately independent of PaliGemma and the MVT feature
extractor. It supports both the legacy single active instance and the fixed
Target/Reference pair used by the relation-aware feature-adapter path.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .object_conditioning import soft_role_geometry


ORACLE_ROLE_UNKNOWN = 0
ORACLE_ROLE_TARGET = 1
ORACLE_ROLE_REFERENCE = 2

ORACLE_PRIOR_MODES = ("none", "o2_gt_instance")
ORACLE_ACTIVE_ROLES = ("auto", "target", "reference")


INTERNAL_OBJECT_SLOT_MODE = 'o2_internal_slots'
ORACLE_PRIOR_MODES = (
    *ORACLE_PRIOR_MODES,
    'o2_predicted_relation',
    INTERNAL_OBJECT_SLOT_MODE,
)
OBJECT_PRIOR_MODES = ORACLE_PRIOR_MODES


def validate_oracle_prior_config(
    mode: str,
    sigma: float,
    active_role: str,
) -> None:
    if mode not in ORACLE_PRIOR_MODES:
        raise ValueError(
            f"Unknown oracle_prior_mode={mode!r}; expected one of "
            f"{ORACLE_PRIOR_MODES}."
        )
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("oracle_prior_sigma must be finite and >= 0")
    if active_role not in ORACLE_ACTIVE_ROLES:
        raise ValueError(
            f"Unknown oracle_prior_active_role={active_role!r}; expected one "
            f"of {ORACLE_ACTIVE_ROLES}."
        )


def resolve_object_prior_mode(
    object_prior_mode: str,
    oracle_prior_mode: str,
) -> str:
    '''Resolve the generic mode while preserving legacy Oracle configs.'''
    if object_prior_mode not in OBJECT_PRIOR_MODES:
        raise ValueError(
            f'Unknown object_prior_mode={object_prior_mode!r}; expected one of '
            f'{OBJECT_PRIOR_MODES}.'
        )
    if oracle_prior_mode not in OBJECT_PRIOR_MODES:
        raise ValueError(
            f'Unknown oracle_prior_mode={oracle_prior_mode!r}; expected one of '
            f'{OBJECT_PRIOR_MODES}.'
        )
    if (
        object_prior_mode != 'none'
        and oracle_prior_mode != 'none'
        and object_prior_mode != oracle_prior_mode
    ):
        raise ValueError(
            'object_prior_mode and legacy oracle_prior_mode select different '
            'object sources'
        )
    if object_prior_mode != 'none':
        return object_prior_mode
    return oracle_prior_mode


def latest_replay_value(value: torch.Tensor, expected_ndim: int) -> torch.Tensor:
    """Remove a replay time dimension, if present, by taking its last value."""
    if value.ndim == expected_ndim:
        return value
    if value.ndim == expected_ndim + 1:
        return value[:, -1]
    raise ValueError(
        f"Expected a tensor with {expected_ndim} dimensions (or one replay "
        f"time dimension), got shape {tuple(value.shape)}"
    )


def select_active_instance_points(
    object_points: torch.Tensor,
    object_valid: torch.Tensor,
    object_roles: torch.Tensor,
    *,
    gripper_open: Optional[torch.Tensor] = None,
    active_role: str = "auto",
    strict: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the current O2 instance from padded Oracle replay slots.

    ``auto`` uses target while the current gripper is open and reference while
    it is closed.  This changes only which GT instance supplies the spatial
    prior; it does not use the next action position.

    Returns:
        selected_points: ``[B, P, 3]``
        selected_valid: ``[B]``
        selected_slots: ``[B]`` (``-1`` when no instance is available)
    """
    if active_role not in ORACLE_ACTIVE_ROLES:
        raise ValueError(
            f"active_role must be one of {ORACLE_ACTIVE_ROLES}, got "
            f"{active_role!r}"
        )
    if object_points.ndim != 4 or object_points.shape[-1] != 3:
        raise ValueError(
            "oracle_object_points must have shape [B, O, P, 3], got "
            f"{tuple(object_points.shape)}"
        )
    batch_size, num_objects, num_points, _ = object_points.shape
    if object_valid.shape != (batch_size, num_objects):
        raise ValueError("oracle_object_valid shape does not match object points")
    if object_roles.shape != (batch_size, num_objects):
        raise ValueError("oracle_object_roles shape does not match object points")

    if active_role == "auto":
        if gripper_open is None:
            raise ValueError(
                "gripper_open is required when oracle_prior_active_role=auto"
            )
        gripper_open = gripper_open.reshape(batch_size)
        desired_roles = torch.where(
            gripper_open >= 0.5,
            torch.full_like(gripper_open, ORACLE_ROLE_TARGET, dtype=torch.long),
            torch.full_like(
                gripper_open, ORACLE_ROLE_REFERENCE, dtype=torch.long
            ),
        )
    else:
        role_code = (
            ORACLE_ROLE_TARGET
            if active_role == "target"
            else ORACLE_ROLE_REFERENCE
        )
        desired_roles = torch.full(
            (batch_size,), role_code, device=object_roles.device, dtype=torch.long
        )

    candidates = object_valid.bool() & (
        object_roles.long() == desired_roles.unsqueeze(1)
    )
    candidate_counts = candidates.sum(dim=1)
    if strict and torch.any(candidate_counts != 1):
        bad = torch.nonzero(candidate_counts != 1, as_tuple=False).flatten()
        details = ", ".join(
            f"batch {int(index)}: {int(candidate_counts[index])} candidates"
            for index in bad.detach().cpu()
        )
        raise ValueError(
            "O2 requires exactly one valid instance for the active role; " + details
        )

    # Non-strict mode skips missing or ambiguous labels instead of choosing an
    # arbitrary slot. The adapter residual is disabled for that sample.
    selected_valid = candidate_counts == 1
    selected_slots = torch.where(
        selected_valid,
        candidates.long().argmax(dim=1),
        torch.full(
            (batch_size,), -1, device=object_points.device, dtype=torch.long
        ),
    )
    safe_slots = selected_slots.clamp_min(0)
    batch_indices = torch.arange(batch_size, device=object_points.device)
    selected_points = object_points[batch_indices, safe_slots]
    selected_points = torch.where(
        selected_valid[:, None, None],
        selected_points,
        torch.zeros(
            (batch_size, num_points, 3),
            device=object_points.device,
            dtype=object_points.dtype,
        ),
    )
    return selected_points, selected_valid, selected_slots


def select_relation_instance_points(
    object_points: torch.Tensor,
    object_valid: torch.Tensor,
    object_roles: torch.Tensor,
    *,
    strict: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if object_points.ndim != 4 or object_points.shape[-1] != 3:
        raise ValueError('oracle_object_points must have shape [B,O,P,3]')
    batch_size, num_objects, num_points, _ = object_points.shape
    if object_valid.shape != (batch_size, num_objects):
        raise ValueError('oracle_object_valid shape does not match points')
    if object_roles.shape != (batch_size, num_objects):
        raise ValueError('oracle_object_roles shape does not match points')

    role_codes = torch.tensor(
        [ORACLE_ROLE_TARGET, ORACLE_ROLE_REFERENCE],
        device=object_roles.device,
        dtype=object_roles.dtype,
    )
    candidates = object_valid.bool().unsqueeze(-1) & (
        object_roles.long().unsqueeze(-1) == role_codes.long()
    )
    candidate_counts = candidates.sum(dim=1)
    if strict and torch.any(candidate_counts != 1):
        bad = torch.nonzero(candidate_counts != 1, as_tuple=False)
        details = ', '.join(
            f'batch {int(batch)}, role {int(role)}: '
            f'{int(candidate_counts[batch, role])} candidates'
            for batch, role in bad.detach().cpu()
        )
        raise ValueError(
            'O2 relation requires exactly one Target and one Reference; '
            + details
        )

    selected_valid = candidate_counts == 1
    selected_slots = torch.where(
        selected_valid,
        candidates.long().argmax(dim=1),
        torch.full_like(candidate_counts, -1, dtype=torch.long),
    )
    safe_slots = selected_slots.clamp_min(0)
    batch_indices = torch.arange(
        batch_size, device=object_points.device,
    ).unsqueeze(1)
    selected_points = object_points[batch_indices, safe_slots]
    selected_points = torch.where(
        selected_valid[:, :, None, None],
        selected_points,
        torch.zeros(
            (batch_size, 2, num_points, 3),
            device=object_points.device,
            dtype=object_points.dtype,
        ),
    )
    return selected_points, selected_valid, selected_slots


def rasterize_instance_points(
    projected_points: torch.Tensor,
    instance_valid: torch.Tensor,
    image_size: Tuple[int, int],
    sigma: float,
) -> torch.Tensor:
    """Rasterize projected instance points to peak-normalized view heatmaps."""
    if projected_points.ndim != 4 or projected_points.shape[-1] != 2:
        raise ValueError(
            "projected_points must have shape [B, P, V, 2], got "
            f"{tuple(projected_points.shape)}"
        )
    batch_size, _, num_views, _ = projected_points.shape
    if instance_valid.shape != (batch_size,):
        raise ValueError("instance_valid must have shape [B]")
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image_size must contain positive values")
    if sigma < 0:
        raise ValueError("sigma must be >= 0")

    xy = projected_points.permute(0, 2, 1, 3)
    finite = torch.isfinite(xy).all(dim=-1)
    x = torch.round(torch.nan_to_num(xy[..., 0])).long()
    y = torch.round(torch.nan_to_num(xy[..., 1])).long()
    inside = finite & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    inside &= instance_valid[:, None, None]

    flat_indices = y.clamp(0, height - 1) * width + x.clamp(0, width - 1)
    heatmap = projected_points.new_zeros(
        (batch_size, num_views, height * width)
    )
    heatmap.scatter_add_(2, flat_indices, inside.to(heatmap.dtype))
    heatmap = heatmap.view(batch_size * num_views, 1, height, width)

    if sigma > 0:
        radius = max(1, int(math.ceil(3 * sigma)))
        coords = torch.arange(
            -radius, radius + 1, device=heatmap.device, dtype=heatmap.dtype
        )
        kernel_1d = torch.exp(-(coords**2) / (2 * sigma**2))
        kernel = torch.outer(kernel_1d, kernel_1d).view(
            1, 1, 2 * radius + 1, 2 * radius + 1
        )
        heatmap = F.conv2d(heatmap, kernel, padding=radius)

    peak = heatmap.amax(dim=(-2, -1), keepdim=True)
    heatmap = torch.where(peak > 0, heatmap / peak.clamp_min(1e-12), heatmap)
    return heatmap.view(batch_size, num_views, height, width)


def _relation_valid_mask(instance_valid, batch_size):
    if instance_valid.shape == (batch_size,):
        return instance_valid.bool()
    if instance_valid.ndim == 2 and instance_valid.shape[0] == batch_size:
        # Target gates the residual. Legacy callers may close an uncertain
        # pair; soft-role conditioning instead masks Reference geometry and
        # keeps an available Target, distinguishing unknown from semantic NULL.
        return instance_valid[:, 0].bool()
    raise ValueError('instance_valid must have shape [B] or [B,R]')


def valid_oracle_translation_loss(
    loss_values, oracle_valid, distributed=False,
):
    '''Average over complete Oracle samples, with optional DDP normalization.'''
    if oracle_valid is None:
        raise ValueError('oracle_valid_only_loss requires Oracle validity')
    if loss_values.ndim < 1:
        raise ValueError('translation loss must retain a batch dimension')
    if oracle_valid.ndim == 2:
        sample_valid = oracle_valid.bool().all(dim=1)
    else:
        sample_valid = _relation_valid_mask(
            oracle_valid, loss_values.shape[0],
        )
    per_sample = loss_values.reshape(loss_values.shape[0], -1).mean(dim=1)
    valid_weight = sample_valid.to(
        device=per_sample.device, dtype=per_sample.dtype,
    )
    numerator = (per_sample * valid_weight).sum()
    denominator = valid_weight.sum()
    if distributed and dist.is_available() and dist.is_initialized():
        denominator = denominator.detach().clone()
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
        # DDP averages gradients across ranks. Multiplying the local numerator
        # by world size recovers a global sum / global valid-count gradient.
        numerator = numerator * dist.get_world_size()
    return numerator / denominator.clamp_min(1.0)


def route_oracle_adapter_features(
    base_features, adapted_features, translation_only,
):
    '''Return translation features and the feature tensor used by action heads.'''
    if base_features.shape != adapted_features.shape:
        raise ValueError('base and adapted feature shapes must match')
    action_features = (
        base_features if translation_only else adapted_features
    )
    return adapted_features, action_features


def choose_oracle_translation_loss(
    all_sample_loss, valid_sample_loss, valid_only,
):
    '''Select the configured translation objective without changing metrics.'''
    if not valid_only:
        return all_sample_loss
    if valid_sample_loss is None:
        raise ValueError('valid-only translation loss is unavailable')
    return valid_sample_loss


class OraclePriorFeatureAdapter(nn.Module):
    def __init__(
        self, feature_channels: int, rank: int = 16, prior_channels: int = 1,
    ):
        super().__init__()
        if feature_channels <= 0 or rank <= 0 or prior_channels <= 0:
            raise ValueError('feature_channels, rank and prior_channels must be positive')
        self.feature_channels = feature_channels
        self.prior_channels = prior_channels
        self.feature_reduce = nn.Conv2d(feature_channels, rank, 1)
        self.prior_project = nn.Conv2d(prior_channels, rank, 3, padding=1)
        self.feature_expand = nn.Conv2d(rank, feature_channels, 1)
        nn.init.zeros_(self.feature_expand.weight)
        nn.init.zeros_(self.feature_expand.bias)

    def forward(
        self, features, prior, instance_valid, relation_points=None,
    ):
        if features.ndim != 4:
            raise ValueError('features must have shape [B*V,C,H,W]')
        if features.shape[1] != self.feature_channels:
            raise ValueError('unexpected feature channel count')
        if prior.ndim == 4:
            prior = prior.unsqueeze(2)
        if prior.ndim != 5:
            raise ValueError('prior must have shape [B,V,R,H,W]')
        batch_size, num_views, prior_channels = prior.shape[:3]
        if prior_channels != self.prior_channels:
            raise ValueError('unexpected prior channel count')
        if features.shape[0] != batch_size * num_views:
            raise ValueError('feature batch does not match prior batch and views')
        relation_valid = _relation_valid_mask(instance_valid, batch_size)

        prior_features = prior.reshape(
            batch_size * num_views, prior_channels, *prior.shape[-2:]
        ).to(device=features.device, dtype=features.dtype)
        prior_features = F.interpolate(
            prior_features, size=features.shape[-2:], mode='bilinear',
            align_corners=False,
        )
        hidden = self.feature_reduce(features) + self.prior_project(
            prior_features
        )
        residual = self.feature_expand(F.gelu(hidden))
        valid = relation_valid.to(
            device=features.device, dtype=features.dtype,
        ).repeat_interleave(num_views).view(-1, 1, 1, 1)
        return features + residual * valid


class OracleRelationGatedFeatureAdapter(OraclePriorFeatureAdapter):
    '''Low-rank adapter conditioned on explicit Target/Reference geometry.

    It combines the two projected prior channels with a shared PointNet-style
    encoder of the two 3-D point sets. The final feature expansion remains zero
    initialized, preserving the original BridgeVLA output at initialization.
    '''

    def __init__(
        self, feature_channels: int, rank: int = 16, prior_channels: int = 2,
    ):
        if prior_channels != 2:
            raise ValueError(
                'Relation-gated adapter requires Target/Reference prior channels'
            )
        super().__init__(feature_channels, rank, prior_channels)
        self.rank = rank
        self.point_encoder = nn.Sequential(
            nn.Linear(3, rank),
            nn.GELU(),
            nn.Linear(rank, rank),
            nn.GELU(),
        )
        relation_input_channels = 2 * rank + 15
        self.relation_encoder = nn.Sequential(
            nn.Linear(relation_input_channels, rank),
            nn.GELU(),
            nn.Linear(rank, 2 * rank + 1),
        )
        nn.init.zeros_(self.relation_encoder[-1].weight[-1:])
        nn.init.zeros_(self.relation_encoder[-1].bias[-1:])

    def _encode_relation(self, points, instance_valid, dtype, device, geometry=None):
        if points is None:
            raise ValueError(
                'Relation-gated adapter requires oracle relation points'
            )
        if points.ndim != 4 or points.shape[1] != 2 or points.shape[-1] != 3:
            raise ValueError('relation points must have shape [B,2,P,3]')
        batch_size = points.shape[0]
        if instance_valid.shape != (batch_size, 2):
            raise ValueError(
                'Relation-gated adapter requires instance_valid shape [B,2]'
            )

        points = points.to(device=device, dtype=dtype)
        role_valid = instance_valid.to(device=device).bool()
        valid_float = role_valid.to(dtype=dtype).view(batch_size, 2, 1)
        pooled = self.point_encoder(points).mean(dim=2) * valid_float
        centers = points.mean(dim=2) * valid_float
        extents = (points.amax(dim=2) - points.amin(dim=2)) * valid_float
        displacement = centers[:, 1] - centers[:, 0]
        if geometry is not None:
            if geometry.shape != (batch_size, 17):
                raise ValueError('soft geometry must have shape [B,17]')
            geometry = geometry.to(device=device, dtype=dtype)
            centers = geometry[:, :6].reshape(batch_size, 2, 3)
            extents = geometry[:, 6:12].reshape(batch_size, 2, 3)
            displacement = geometry[:, 12:15]
            # The new path never relies on non-differentiable top-k points.
            pooled = self.point_encoder(centers) * valid_float
        descriptor = torch.cat(
            (
                pooled.reshape(batch_size, -1),
                centers.reshape(batch_size, -1),
                extents.reshape(batch_size, -1),
                displacement,
            ),
            dim=1,
        )
        modulation = self.relation_encoder(descriptor)
        gamma, beta, gate_logit = torch.split(
            modulation, (self.rank, self.rank, 1), dim=1,
        )
        return torch.tanh(gamma), beta, torch.sigmoid(gate_logit)

    def _relation_components(
        self, features, prior, instance_valid, relation_points=None,
        geometry=None,
    ):
        if features.ndim != 4:
            raise ValueError('features must have shape [B*V,C,H,W]')
        if features.shape[1] != self.feature_channels:
            raise ValueError('unexpected feature channel count')
        if prior.ndim != 5:
            raise ValueError('relation prior must have shape [B,V,2,H,W]')
        batch_size, num_views, prior_channels = prior.shape[:3]
        if prior_channels != self.prior_channels:
            raise ValueError('unexpected prior channel count')
        if features.shape[0] != batch_size * num_views:
            raise ValueError('feature batch does not match prior batch and views')
        relation_valid = _relation_valid_mask(instance_valid, batch_size)

        prior_features = prior.reshape(
            batch_size * num_views, prior_channels, *prior.shape[-2:]
        ).to(device=features.device, dtype=features.dtype)
        prior_features = F.interpolate(
            prior_features, size=features.shape[-2:], mode='bilinear',
            align_corners=False,
        )
        if geometry is not None:
            prior_features = prior_features * instance_valid.to(
                device=features.device, dtype=features.dtype,
            ).repeat_interleave(num_views, dim=0)[:, :, None, None]
        gamma, beta, gate = self._encode_relation(
            relation_points, instance_valid, features.dtype, features.device,
            geometry=geometry,
        )
        gamma = gamma.repeat_interleave(num_views, dim=0).view(
            -1, self.rank, 1, 1,
        )
        beta = beta.repeat_interleave(num_views, dim=0).view(
            -1, self.rank, 1, 1,
        )
        gate = gate.repeat_interleave(num_views, dim=0).view(-1, 1, 1, 1)

        feature_hidden = self.feature_reduce(features)
        hidden = (
            feature_hidden * (1.0 + gamma)
            + self.prior_project(prior_features)
            + beta
        )
        residual = self.feature_expand(F.gelu(hidden))
        valid = relation_valid.to(
            device=features.device, dtype=features.dtype,
        ).repeat_interleave(num_views).view(-1, 1, 1, 1)
        shared_features = features + residual * gate * valid
        return (
            shared_features,
            hidden,
            batch_size,
            num_views,
        )

    def forward(self, features, prior, instance_valid, relation_points=None):
        shared_features, _, _, _ = self._relation_components(
            features, prior, instance_valid, relation_points,
        )
        return shared_features


class OracleRelationAnchorFeatureAdapter(OracleRelationGatedFeatureAdapter):
    '''Relation-gated adapter with a translation-anchor enhancement.

    This is an enhancement of :class:`OracleRelationGatedFeatureAdapter`, not a
    second adapter. The original relation residual remains the shared feature
    path for rotation/gripper/collision. The same relation-conditioned hidden
    feature is masked-pooled inside Target/Reference priors to predict a soft
    spatial anchor and an extra residual. Legacy routing applies that residual
    to translation only; opt-in shared-action routing also uses it for R/G/C.

    No phase label or hand-authored action anchor is consumed. A learned NULL
    token supports target-only samples. The anchor output projection is zero
    initialized, so enabling this subclass preserves the original adapter's
    output at initialization.
    '''

    def __init__(
        self,
        feature_channels: int,
        rank: int = 16,
        prior_channels: int = 2,
        anchor_rank: int = 16,
        state_channels: int = 3,
        use_context: bool = False,
        role_conditioning: bool = False,
        role_token_dim: int = 128,
    ):
        super().__init__(feature_channels, rank, prior_channels)
        if anchor_rank <= 0 or state_channels < 0:
            raise ValueError(
                'anchor_rank must be positive and state_channels non-negative'
            )
        self.anchor_rank = anchor_rank
        self.state_channels = state_channels
        self.use_context = bool(use_context)
        self.role_conditioning = bool(role_conditioning)
        if self.use_context:
            self.context_projection = nn.Sequential(
                nn.LayerNorm(feature_channels),
                nn.Linear(feature_channels, anchor_rank),
                nn.GELU(),
                nn.Linear(anchor_rank, anchor_rank),
            )
            # Preserve a trained anchor query when initializing from old O2.
            nn.init.zeros_(self.context_projection[-1].weight)
            nn.init.zeros_(self.context_projection[-1].bias)
        if self.role_conditioning:
            self.unknown_reference = nn.Parameter(torch.zeros(rank))
            self.role_token_projection = nn.Linear(role_token_dim, rank)
        # Two visual tokens plus centers, extents, displacement, and validity.
        self.anchor_query_encoder = nn.Sequential(
            nn.Linear(2 * rank + 17 + state_channels, anchor_rank),
            nn.GELU(),
            nn.Linear(anchor_rank, anchor_rank),
        )
        self.null_reference = nn.Parameter(torch.zeros(rank))
        self.anchor_key = nn.Conv2d(rank, anchor_rank, 1)
        self.anchor_expand = nn.Conv2d(rank, feature_channels, 1)
        nn.init.zeros_(self.anchor_expand.weight)
        nn.init.zeros_(self.anchor_expand.bias)

    def _anchor_geometry(self, points, instance_valid, dtype, device):
        if points is None:
            raise ValueError(
                'Relation-anchor adapter requires oracle relation points'
            )
        if points.ndim != 4 or points.shape[1] != 2 or points.shape[-1] != 3:
            raise ValueError('relation points must have shape [B,2,P,3]')
        batch_size = points.shape[0]
        if instance_valid.shape != (batch_size, 2):
            raise ValueError(
                'Relation-anchor adapter requires instance_valid shape [B,2]'
            )
        points = points.to(device=device, dtype=dtype)
        role_valid = instance_valid.to(device=device).bool()
        valid_float = role_valid.to(dtype=dtype).unsqueeze(-1)
        centers = points.mean(dim=2) * valid_float
        extents = (points.amax(dim=2) - points.amin(dim=2)) * valid_float
        displacement = (centers[:, 1] - centers[:, 0]) * valid_float[:, 1]
        return torch.cat(
            (
                centers.reshape(batch_size, -1),
                extents.reshape(batch_size, -1),
                displacement,
                valid_float.squeeze(-1),
            ),
            dim=1,
        )

    def forward_with_anchor(
        self, features, prior, instance_valid, relation_points=None,
        relation_state=None,
        *, current_state=None, context=None, role_tokens=None,
        reference_null_probability=None, geometry=None,
    ):
        if current_state is not None:
            if relation_state is not None:
                raise ValueError('pass current_state or legacy relation_state, not both')
            relation_state = current_state
        slot_role_tokens = role_tokens
        (
            shared_features,
            relation_hidden,
            batch_size,
            num_views,
        ) = self._relation_components(
            features, prior, instance_valid, relation_points, geometry=geometry,
        )
        height, width = relation_hidden.shape[-2:]
        hidden_views = relation_hidden.view(
            batch_size, num_views, self.rank, height, width,
        )
        # Area pooling retains more support for small/thin object masks than
        # bilinear-sampling a 224px prior directly onto a 16px feature grid.
        flat_prior = prior.reshape(
            batch_size * num_views, self.prior_channels, *prior.shape[-2:]
        ).to(device=features.device, dtype=features.dtype)
        if flat_prior.shape[-2] >= height and flat_prior.shape[-1] >= width:
            pooled_prior = F.interpolate(
                flat_prior, size=(height, width), mode='area',
            )
        else:
            pooled_prior = F.interpolate(
                flat_prior, size=(height, width), mode='bilinear',
                align_corners=False,
            )
        prior_views = pooled_prior.view(
            batch_size, num_views, self.prior_channels, height, width,
        )

        weights = prior_views / prior_views.sum(
            dim=(-2, -1), keepdim=True,
        ).clamp_min(1e-6)
        role_tokens = torch.einsum(
            'bvrhw,bvchw->bvrc', weights, hidden_views,
        )
        role_valid = instance_valid.to(device=features.device).bool()
        target_token = (
            role_tokens[:, :, 0] * role_valid[:, None, 0, None]
        )
        reference_valid = role_valid[:, None, 1, None]
        null_reference = self.null_reference.to(
            device=features.device, dtype=features.dtype,
        ).view(1, 1, self.rank)
        reference_token = torch.where(
            reference_valid, role_tokens[:, :, 1], null_reference,
        )
        if self.role_conditioning:
            unknown_reference = self.unknown_reference.view(1, 1, self.rank)
            reference_token = torch.where(
                reference_valid, role_tokens[:, :, 1], unknown_reference,
            )
            if slot_role_tokens is not None:
                projected = self.role_token_projection(slot_role_tokens)
                target_token = target_token + projected[:, None, 0] * role_valid[:, None, 0, None]
                reference_token = reference_token + projected[:, None, 1] * reference_valid
            if reference_null_probability is not None:
                probability = reference_null_probability[:, None, None].to(features.dtype)
                reference_token = (1 - probability) * reference_token + probability * null_reference
        role_tokens = torch.stack((target_token, reference_token), dim=2)

        if geometry is None:
            geometry = self._anchor_geometry(
                relation_points, instance_valid, features.dtype, features.device,
            )
        else:
            geometry = geometry.to(device=features.device, dtype=features.dtype)
        geometry = geometry[:, None].expand(-1, num_views, -1)
        if relation_state is None:
            relation_state = features.new_zeros(
                batch_size, self.state_channels,
            )
        if relation_state.shape != (batch_size, self.state_channels):
            raise ValueError(
                'relation_state must have shape '
                f'[B,{self.state_channels}]'
            )
        relation_state = relation_state.to(
            device=features.device, dtype=features.dtype,
        )
        relation_state = relation_state[:, None].expand(-1, num_views, -1)
        query = self.anchor_query_encoder(torch.cat(
            (role_tokens.flatten(2), geometry, relation_state), dim=-1,
        ))
        if self.use_context:
            if context is None or context.shape != (batch_size, self.feature_channels):
                raise ValueError('instruction context must have shape [B,C]')
            query = query + self.context_projection(context.to(features.dtype))[:, None]
        query = query.reshape(
            batch_size * num_views, self.anchor_rank, 1, 1,
        )

        anchor_logits = (
            self.anchor_key(relation_hidden) * query
        ).sum(dim=1, keepdim=True) / math.sqrt(
            self.anchor_rank
        )
        anchor = torch.sigmoid(anchor_logits)
        residual = self.anchor_expand(F.gelu(relation_hidden) * anchor)
        # Target is required; Reference=False represents a valid NULL.
        target_valid = role_valid[:, 0].to(
            device=features.device, dtype=features.dtype,
        ).repeat_interleave(num_views).view(-1, 1, 1, 1)
        translation_features = shared_features + residual * target_valid
        return (
            translation_features,
            shared_features,
            anchor.view(batch_size, num_views, height, width),
        )

    def forward(
        self, features, prior, instance_valid, relation_points=None,
        relation_state=None,
        *, current_state=None, context=None, role_tokens=None,
        reference_null_probability=None, geometry=None,
    ):
        translation_features, _, _ = self.forward_with_anchor(
            features,
            prior,
            instance_valid,
            relation_points,
            relation_state,
            current_state=current_state, context=context, role_tokens=role_tokens,
            reference_null_probability=reference_null_probability, geometry=geometry,
        )
        return translation_features


class InternalObjectSlotPredictor(nn.Module):
    '''Predict Target/Reference masks and point sets from BridgeVLA features.

    Learned slots attend jointly to all virtual-view tokens. Role heads then
    mix unordered slot masks into Target and Reference priors, with an explicit
    NULL alternative for Reference. GT object tensors are not consumed here.
    '''

    def __init__(
        self,
        feature_channels: int,
        num_views: int,
        num_slots: int = 6,
        slot_dim: int = 128,
        decoder_layers: int = 2,
        num_heads: int = 4,
        point_samples: int = 128,
        confidence_threshold: float = 0.25,
        state_channels: int = 3,
        use_context: bool = False,
        soft_conditioning: bool = False,
    ):
        super().__init__()
        if feature_channels <= 0 or num_views <= 0 or num_slots <= 0:
            raise ValueError('feature_channels, num_views and num_slots must be positive')
        if slot_dim <= 0 or slot_dim % num_heads:
            raise ValueError('slot_dim must be positive and divisible by num_heads')
        if decoder_layers <= 0 or point_samples <= 0:
            raise ValueError('decoder_layers and point_samples must be positive')
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError('confidence_threshold must be in [0, 1]')

        self.num_views = num_views
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.point_samples = point_samples
        self.confidence_threshold = confidence_threshold
        self.state_channels = state_channels
        self.use_context = bool(use_context)
        self.soft_conditioning = bool(soft_conditioning)
        if self.use_context:
            self.context_projection = nn.Sequential(
                nn.LayerNorm(feature_channels), nn.Linear(feature_channels, slot_dim),
            )
            nn.init.zeros_(self.context_projection[-1].weight)
            nn.init.zeros_(self.context_projection[-1].bias)

        self.feature_reduce = nn.Conv2d(feature_channels, slot_dim, 1)
        self.mask_key = nn.Conv2d(slot_dim, slot_dim, 1)
        self.mask_query = nn.Linear(slot_dim, slot_dim)
        self.slot_queries = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.02)
        self.view_embedding = nn.Parameter(torch.randn(num_views, slot_dim) * 0.02)
        self.xy_embedding = nn.Linear(2, slot_dim)
        self.state_embedding = nn.Linear(state_channels, slot_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=slot_dim,
            nhead=num_heads,
            dim_feedforward=4 * slot_dim,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, decoder_layers)
        self.objectness_head = nn.Linear(slot_dim, 1)
        self.role_head = nn.Linear(slot_dim, 2)
        self.reference_null_head = nn.Linear(slot_dim + state_channels, 1)
        # Start below the validity threshold so a new predictor cannot perturb
        # a pretrained policy before the auxiliary object loss has learned.
        nn.init.constant_(self.objectness_head.bias, -2.0)

    @staticmethod
    def _xy_grid(height, width, device, dtype):
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype),
            indexing='ij',
        )
        return torch.stack((xx, yy), dim=-1)

    def _extract_points(self, role_prior, rendered_xyz, role_valid):
        if rendered_xyz.ndim != 5 or rendered_xyz.shape[2] != 3:
            raise ValueError('rendered_xyz must have shape [B,V,3,H,W]')
        batch_size, num_views, _, height, width = rendered_xyz.shape
        if num_views != self.num_views:
            raise ValueError('rendered_xyz view count does not match predictor')
        scores = F.interpolate(
            role_prior.reshape(batch_size * num_views, 2, *role_prior.shape[-2:]),
            size=(height, width),
            mode='bilinear',
            align_corners=False,
        ).view(batch_size, num_views, 2, height, width)
        xyz = rendered_xyz.permute(0, 1, 3, 4, 2).reshape(
            batch_size, num_views * height * width, 3,
        )
        finite = torch.isfinite(xyz).all(dim=-1)
        # The CUDA RVT renderer uses zero-valued feature pixels as background.
        # Keep -1 support for alternate renderers used by older checkpoints.
        background = (xyz == 0).all(dim=-1) | (xyz == -1).all(dim=-1)
        xyz_valid = finite & ~background
        xyz = torch.where(xyz_valid[..., None], xyz, torch.zeros_like(xyz))
        scores = scores.permute(0, 2, 1, 3, 4).reshape(
            batch_size, 2, num_views * height * width,
        )
        scores = scores.masked_fill(~xyz_valid[:, None], -1.0)
        sample_count = min(self.point_samples, scores.shape[-1])
        indices = scores.topk(sample_count, dim=-1).indices
        xyz_roles = xyz[:, None].expand(-1, 2, -1, -1)
        points = torch.gather(
            xyz_roles, 2, indices.unsqueeze(-1).expand(-1, -1, -1, 3),
        )
        points = points * role_valid[:, :, None, None].to(points.dtype)
        return points

    def forward(self, features, rendered_xyz, relation_state=None, *, current_state=None,
                context=None):
        if current_state is not None:
            if relation_state is not None:
                raise ValueError('pass current_state or legacy relation_state, not both')
            relation_state = current_state
        if features.ndim != 4:
            raise ValueError('features must have shape [B*V,C,H,W]')
        if features.shape[0] % self.num_views:
            raise ValueError('feature batch is not divisible by num_views')
        batch_size = features.shape[0] // self.num_views
        height, width = features.shape[-2:]
        if rendered_xyz.shape[:2] != (batch_size, self.num_views):
            raise ValueError('feature and rendered_xyz batches do not match')

        reduced = self.feature_reduce(features).view(
            batch_size, self.num_views, self.slot_dim, height, width,
        )
        grid = self._xy_grid(height, width, reduced.device, reduced.dtype)
        position = self.xy_embedding(grid).permute(2, 0, 1)
        memory = reduced + position[None, None]
        memory = memory + self.view_embedding[None, :, :, None, None]
        memory = memory.permute(0, 1, 3, 4, 2).reshape(
            batch_size, self.num_views * height * width, self.slot_dim,
        )

        if relation_state is None:
            relation_state = features.new_zeros(batch_size, self.state_channels)
        if relation_state.shape != (batch_size, self.state_channels):
            raise ValueError(
                f'relation_state must have shape [B,{self.state_channels}]'
            )
        relation_state = relation_state.to(device=features.device, dtype=features.dtype)
        state_token = self.state_embedding(relation_state)[:, None]
        queries = self.slot_queries[None].expand(batch_size, -1, -1) + state_token
        if self.use_context:
            if context is None or context.shape != (batch_size, features.shape[1]):
                raise ValueError('instruction context must have shape [B,C]')
            queries = queries + self.context_projection(context.to(features.dtype))[:, None]
        slots = self.decoder(queries, memory)

        keys = self.mask_key(reduced.reshape(-1, self.slot_dim, height, width))
        keys = keys.view(
            batch_size, self.num_views, self.slot_dim, height, width,
        )
        mask_queries = self.mask_query(slots)
        mask_logits = torch.einsum(
            'bkd,bvdhw->bvkhw', mask_queries, keys,
        ) / math.sqrt(self.slot_dim)
        slot_masks = torch.sigmoid(mask_logits)
        objectness_logits = self.objectness_head(slots).squeeze(-1)
        role_logits = self.role_head(slots)
        pooled_slot = slots.mean(dim=1)
        reference_null_logit = self.reference_null_head(torch.cat(
            (pooled_slot, relation_state), dim=-1,
        )).squeeze(-1)

        object_log_prob = F.logsigmoid(objectness_logits)
        role_log_prob = F.log_softmax(role_logits, dim=-1)
        target_weights = torch.softmax(
            object_log_prob + role_log_prob[:, :, 0], dim=1,
        )
        reference_scores = object_log_prob + role_log_prob[:, :, 1]
        reference_all = torch.softmax(torch.cat(
            (reference_scores, reference_null_logit[:, None]), dim=1,
        ), dim=1)
        reference_weights = reference_all[:, :-1]
        reference_null_probability = reference_all[:, -1]
        target_prior = torch.einsum('bk,bvkhw->bvhw', target_weights, slot_masks)
        reference_prior = torch.einsum(
            'bk,bvkhw->bvhw', reference_weights, slot_masks,
        )
        role_prior = torch.stack((target_prior, reference_prior), dim=2)

        object_probability = torch.sigmoid(objectness_logits)
        target_confidence = (target_weights * object_probability).sum(dim=1)
        reference_mass = 1.0 - reference_null_probability
        normalized_reference = reference_weights / reference_mass[:, None].clamp_min(1e-6)
        reference_confidence = reference_mass * (
            normalized_reference * object_probability
        ).sum(dim=1)
        role_confidence = torch.stack(
            (target_confidence, reference_confidence), dim=1,
        )
        target_valid = target_confidence >= self.confidence_threshold
        reference_valid = reference_confidence >= self.confidence_threshold
        reference_is_null = (
            reference_null_probability >= 1.0 - self.confidence_threshold
        )
        # A low-confidence Reference is not automatically a semantic NULL.
        # Use the pair only when Reference is found or NULL is itself confident.
        if not self.soft_conditioning:
            target_valid = target_valid & (reference_valid | reference_is_null)
        role_valid = torch.stack((target_valid, reference_valid), dim=1)
        geometry = None
        if self.soft_conditioning:
            geometry, role_valid = soft_role_geometry(role_prior, rendered_xyz, role_valid)
        points = self._extract_points(role_prior, rendered_xyz, role_valid)
        role_tokens = torch.stack((
            torch.einsum('bk,bkd->bd', target_weights, slots),
            torch.einsum('bk,bkd->bd', normalized_reference, slots),
        ), dim=1)

        return {
            'prior': role_prior,
            'prior_logits': torch.logit(role_prior.clamp(1e-5, 1.0 - 1e-5)),
            'points': points,
            'valid': role_valid,
            'confidence': role_confidence,
            'slot_masks': slot_masks,
            'mask_logits': mask_logits,
            'objectness_logits': objectness_logits,
            'role_logits': role_logits,
            'reference_null_logit': reference_null_logit,
            'reference_null_probability': reference_null_probability,
            'reference_is_null': reference_is_null,
            'role_tokens': role_tokens,
            'geometry': geometry,
        }


def _translation_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert one sample of [V, H, W] logits to per-view probabilities."""
    if logits.ndim != 3:
        raise ValueError("translation logits must have shape [V, H, W]")
    views, height, width = logits.shape
    return torch.softmax(
        logits.float().reshape(views, height * width), dim=-1
    ).reshape(views, height, width)


def build_training_visualization_payload(
    output: Mapping[str, torch.Tensor],
    action_translation: torch.Tensor,
    *,
    num_views: int,
    height: int,
    width: int,
    stage_two: bool,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Collect the first processed training sample without retaining its graph.

    The GT tensor is the final translation heatmap after the same augmentation
    and projection used for the loss. Predictions are per-view probabilities.
    """
    expected_stages = 2 if stage_two else 1
    expected_channels = num_views * expected_stages
    if action_translation.ndim != 3:
        raise ValueError(
            "action_translation must have shape [B, H*W, V*stages]"
        )
    if action_translation.shape[1:] != (
        height * width,
        expected_channels,
    ):
        raise ValueError(
            "action_translation shape does not match visualization dimensions: "
            f"{tuple(action_translation.shape)}"
        )

    gt_views = action_translation[0].transpose(0, 1).reshape(
        expected_channels, height, width
    )
    stages = [("mvt1", output, output.get("mvt1_ori_img"))]
    if stage_two:
        if "mvt2" not in output:
            raise KeyError("stage-two output is missing mvt2")
        stages.append(("mvt2", output["mvt2"], output.get("mvt2_ori_img")))

    payload: Dict[str, Dict[str, torch.Tensor]] = {}
    for stage_index, (stage_name, stage_output, rendered) in enumerate(stages):
        if rendered is None:
            raise KeyError(f"{stage_name} rendered input is unavailable")
        if rendered.ndim != 5 or rendered.shape[2] < 6:
            raise ValueError(
                f"{stage_name} rendered input must have shape [B,V,C,H,W] "
                "with at least six channels"
            )
        start = stage_index * num_views
        end = start + num_views
        stage_payload = {
            "input": rendered[0, :, 3:6],
            "gt": gt_views[start:end],
            "pred": _translation_probabilities(stage_output["trans"][0]),
        }
        if "oracle_instance_prior" in stage_output:
            stage_payload["prior"] = stage_output["oracle_instance_prior"][0]
        if 'oracle_target_prior' in stage_output:
            stage_payload['target_prior'] = stage_output[
                'oracle_target_prior'
            ][0]
        if 'oracle_reference_prior' in stage_output:
            stage_payload['reference_prior'] = stage_output[
                'oracle_reference_prior'
            ][0]
        if 'object_slot_prior' in stage_output:
            slot_prior = stage_output['object_slot_prior'][0]
            if slot_prior.shape[-2:] != (height, width):
                slot_prior = F.interpolate(
                    slot_prior.permute(1, 0, 2, 3),
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False,
                ).permute(1, 0, 2, 3)
            stage_payload['slot_target_pred'] = slot_prior[:, 0]
            stage_payload['slot_reference_pred'] = slot_prior[:, 1]
        if 'object_slot_target_prior' in stage_output:
            slot_gt = stage_output['object_slot_target_prior'][0]
            stage_payload['slot_target_gt'] = slot_gt[:, 0]
            stage_payload['slot_reference_gt'] = slot_gt[:, 1]
        if 'oracle_relation_anchor' in stage_output:
            relation_anchor = stage_output[
                'oracle_relation_anchor'
            ][0]
            if relation_anchor.shape[-2:] != (height, width):
                relation_anchor = F.interpolate(
                    relation_anchor[:, None].float(),
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False,
                )[:, 0]
            stage_payload['relation_anchor'] = relation_anchor
        payload[stage_name] = {
            key: value.detach().float().cpu()
            for key, value in stage_payload.items()
        }
    return payload
