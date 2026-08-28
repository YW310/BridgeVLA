"""Utilities for the optional O2 GT-instance translation prior.

This module is deliberately independent of PaliGemma and the MVT feature
extractor. It supports both the legacy single active instance and the fixed
Target/Reference pair used by the relation-aware trainable fusion path.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


ORACLE_ROLE_UNKNOWN = 0
ORACLE_ROLE_TARGET = 1
ORACLE_ROLE_REFERENCE = 2

ORACLE_PRIOR_MODES = ("none", "o2_gt_instance")
ORACLE_ACTIVE_ROLES = ("auto", "target", "reference")


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
    # arbitrary slot. The fusion head then falls back to raw BridgeVLA logits.
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
        return instance_valid.bool().all(dim=1)
    raise ValueError('instance_valid must have shape [B] or [B,R]')


def valid_oracle_translation_loss(
    loss_values, oracle_valid, distributed=False,
):
    '''Average over complete Oracle samples, with optional DDP normalization.'''
    if oracle_valid is None:
        raise ValueError('oracle_valid_only_loss requires Oracle validity')
    if loss_values.ndim < 1:
        raise ValueError('translation loss must retain a batch dimension')
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

    def _encode_relation(self, points, instance_valid, dtype, device):
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

    def forward(self, features, prior, instance_valid, relation_points=None):
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
        gamma, beta, gate = self._encode_relation(
            relation_points, instance_valid, features.dtype, features.device,
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
        return features + residual * gate * valid


class OraclePriorFusion(nn.Module):
    """Learn a residual correction from raw and GT-instance heatmaps."""

    def __init__(
        self, hidden_channels: int = 16, multiscale: bool = False,
        prior_channels: int = 1,
    ):
        super().__init__()
        self.multiscale = bool(multiscale)
        self.prior_channels = prior_channels
        if hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive")
        if prior_channels <= 0:
            raise ValueError('prior_channels must be positive')
        self.net = nn.Sequential(
            nn.Conv2d(
                1 + prior_channels, hidden_channels, kernel_size=3, padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        if self.multiscale:
            context_channels = 1 + 2 * prior_channels
            if prior_channels == 2:
                context_channels += 1
            self.context = nn.Sequential(
                nn.Conv2d(context_channels, hidden_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(
                    hidden_channels, hidden_channels, 3,
                    padding=2, dilation=2,
                ),
                nn.GELU(),
                nn.Conv2d(hidden_channels, 1, 1),
            )
        else:
            self.context = None
        if self.context is not None:
            nn.init.zeros_(self.context[-1].weight)
            nn.init.zeros_(self.context[-1].bias)

    def _forward_relation(self, raw_logits, prior, instance_valid):
        batch_size, num_views, prior_channels, height, width = prior.shape
        if raw_logits.shape != (batch_size, num_views, height, width):
            raise ValueError('raw logits and relation prior shapes must match')
        if prior_channels != self.prior_channels:
            raise ValueError('unexpected relation prior channel count')
        features = torch.cat((raw_logits.unsqueeze(2), prior), dim=2).reshape(
            batch_size * num_views, 1 + prior_channels, height, width
        )
        residual = self.net(features).reshape(
            batch_size, num_views, height, width
        )
        if self.context is not None:
            raw_prior = (raw_logits.unsqueeze(2) * prior).reshape(
                batch_size * num_views, prior_channels, height, width
            )
            context_parts = [features, raw_prior]
            if prior_channels == 2:
                target_reference = (prior[:, :, 0] * prior[:, :, 1]).reshape(
                    batch_size * num_views, 1, height, width
                )
                context_parts.append(target_reference)
            context_features = torch.cat(context_parts, dim=1)
            residual = residual + self.context(context_features).reshape(
                batch_size, num_views, height, width
            )
        relation_valid = _relation_valid_mask(instance_valid, batch_size)
        residual = residual * relation_valid[:, None, None, None].to(
            residual.dtype
        )
        return raw_logits + residual

    def forward(self, raw_logits, prior, instance_valid):
        if prior.ndim == 5:
            return self._forward_relation(raw_logits, prior, instance_valid)
        if raw_logits.shape != prior.shape:
            raise ValueError("raw logits and prior shapes must match")
        if instance_valid.shape != (raw_logits.shape[0],):
            raise ValueError("instance_valid must have shape [B]")
        batch_size, num_views, height, width = raw_logits.shape
        features = torch.stack((raw_logits, prior), dim=2).reshape(
            batch_size * num_views, 2, height, width
        )
        residual = self.net(features).reshape(
            batch_size, num_views, height, width
        )
        if self.context is not None:
            interaction = (raw_logits * prior).reshape(
                batch_size * num_views, 1, height, width
            )
            context_features = torch.cat((features, interaction), dim=1)
            residual = residual + self.context(context_features).reshape(
                batch_size, num_views, height, width
            )
        residual = residual * instance_valid[:, None, None, None].to(
            residual.dtype
        )
        return raw_logits + residual


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
        if "trans_raw" in stage_output:
            stage_payload["raw_pred"] = _translation_probabilities(
                stage_output["trans_raw"][0]
            )
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
        payload[stage_name] = {
            key: value.detach().float().cpu()
            for key, value in stage_payload.items()
        }
    return payload
