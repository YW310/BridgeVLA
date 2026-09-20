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
