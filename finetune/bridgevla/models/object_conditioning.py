"""Small, backbone-independent helpers for opt-in object conditioning."""

import torch
import torch.nn.functional as F


def pool_instruction_context(hidden, attention_mask, image_tokens, input_ids=None,
                             excluded_token_ids=()):
    """Pool text after the image prefix, excluding padding and special tokens.

    Count *valid* prefix tokens so both left and right padding are supported.
    This uses the already computed VLM hidden states, not another text encoder.
    """
    valid = attention_mask.bool()
    text = valid & (valid.long().cumsum(dim=1) > image_tokens)
    if input_ids is not None:
        for token_id in excluded_token_ids:
            if token_id is not None:
                text = text & (input_ids != token_id)
    weights = text.to(hidden.dtype).unsqueeze(-1)
    selected = torch.where(text[..., None], hidden, torch.zeros_like(hidden))
    return selected.sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def action_feature_routes(base, translation, legacy_action, shared=False):
    """Return translation, local-action and global-action feature sources."""
    if shared:
        return translation, translation, translation
    return translation, legacy_action, base


def select_object_candidate_from_waypoint(
    waypoint, candidate_points, candidate_valid, temperature=0.05,
    max_distance=0.20,
):
    """Attribute a decoded translation waypoint to a visible object candidate.

    This helper is evaluation-only. ``candidate_points`` are simulator object
    point clouds in the same world frame as ``waypoint``. The returned
    confidence is a softmax over each candidate's nearest-surface distance; it
    is a diagnostic score, not a calibrated object posterior.
    """
    if waypoint.ndim != 2 or waypoint.shape[-1] != 3:
        raise ValueError('waypoint must have shape [B,3]')
    if candidate_points.ndim != 4 or candidate_points.shape[-1] != 3:
        raise ValueError('candidate_points must have shape [B,K,P,3]')
    if candidate_valid.shape != candidate_points.shape[:2]:
        raise ValueError('candidate_valid must have shape [B,K]')
    if waypoint.shape[0] != candidate_points.shape[0]:
        raise ValueError('waypoint and candidate batch dimensions must match')
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    if max_distance <= 0:
        raise ValueError('max_distance must be positive')

    finite_points = torch.isfinite(candidate_points).all(dim=-1)
    distances = torch.linalg.vector_norm(
        candidate_points - waypoint[:, None, None], dim=-1,
    )
    distances = distances.masked_fill(~finite_points, torch.inf)
    distances = distances.amin(dim=-1)
    available = candidate_valid.bool() & finite_points.any(dim=-1)
    distances = distances.masked_fill(~available, torch.inf)
    any_available = available.any(dim=-1)

    nearest = distances.argmin(dim=-1)
    safe_logits = torch.where(
        available,
        -distances / float(temperature),
        torch.full_like(distances, -torch.finfo(distances.dtype).max),
    )
    probabilities = torch.softmax(safe_logits, dim=-1)
    probabilities = torch.where(
        any_available[:, None], probabilities, torch.zeros_like(probabilities),
    )
    gather_index = nearest.unsqueeze(-1)
    selected_distance = distances.gather(1, gather_index).squeeze(1)
    confidence = probabilities.gather(1, gather_index).squeeze(1)
    selected_distance = torch.where(
        any_available, selected_distance,
        torch.full_like(selected_distance, torch.inf),
    )
    accepted = any_available & (selected_distance <= float(max_distance))
    confidence = confidence * (
        1.0 - selected_distance / float(max_distance)
    ).clamp(0.0, 1.0)
    confidence = torch.where(
        accepted, confidence, torch.zeros_like(confidence),
    )
    selected = torch.where(
        accepted, nearest, torch.full_like(nearest, -1),
    )
    return selected, selected_distance, confidence


def reference_null_loss(probability, present=None, known=None):
    """Supervise the actual NULL posterior, never geometric invalidity."""
    if present is None or known is None:
        return probability.sum() * 0.0
    present = present.to(device=probability.device).bool()
    known = known.to(device=probability.device).bool()
    if present.shape != (probability.shape[0], 2) or known.shape != present.shape:
        raise ValueError('role presence and known masks must have shape [B,2]')
    values = F.binary_cross_entropy(
        probability.clamp(1e-6, 1 - 1e-6),
        (~present[:, 1]).to(probability.dtype), reduction='none',
    )
    weights = known[:, 1].to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1)


def soft_role_geometry(prior, rendered_xyz, role_valid):
    """Differentiable visible-point centers/spreads in the current render frame.

    Spread is a weighted standard deviation, not the object's full extent.
    Background and non-finite XYZ have zero weight, including empty scenes.
    """
    if rendered_xyz.ndim != 5 or rendered_xyz.shape[2] != 3:
        raise ValueError('rendered_xyz must have shape [B,V,3,H,W]')
    batch, views, _, height, width = rendered_xyz.shape
    if prior.shape[:3] != (batch, views, 2) or role_valid.shape != (batch, 2):
        raise ValueError('role maps/validity do not match rendered_xyz')
    weights = F.interpolate(
        prior.reshape(batch * views, 2, *prior.shape[-2:]),
        size=(height, width), mode='bilinear', align_corners=False,
    ).reshape(batch, views, 2, height, width)
    xyz = rendered_xyz.permute(0, 1, 3, 4, 2).reshape(batch, -1, 3)
    finite = torch.isfinite(xyz).all(dim=-1)
    background = (xyz == 0).all(dim=-1) | (xyz == -1).all(dim=-1)
    support = finite & ~background
    xyz = torch.where(support[..., None], xyz, torch.zeros_like(xyz))
    weights = weights.permute(0, 2, 1, 3, 4).reshape(batch, 2, -1)
    weights = weights * support[:, None].to(weights.dtype)
    mass = weights.sum(dim=-1, keepdim=True)
    normalized = weights / mass.clamp_min(1e-6)
    centers = torch.einsum('brn,bnc->brc', normalized, xyz)
    delta = xyz[:, None] - centers[:, :, None]
    variance = (normalized[..., None] * delta.square()).sum(dim=2)
    spreads = (variance + 1e-8).sqrt()
    available = role_valid.bool() & (mass.squeeze(-1) > 1e-6)
    mask = available[..., None].to(centers.dtype)
    centers, spreads = centers * mask, spreads * mask
    displacement = (centers[:, 1] - centers[:, 0]) * (
        available.all(dim=1, keepdim=True).to(centers.dtype)
    )
    descriptor = torch.cat((centers.flatten(1), spreads.flatten(1),
                            displacement, available.to(centers.dtype)), dim=1)
    return descriptor, available
