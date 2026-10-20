"""Local mask readout conditioned on inherited roles; no slot/NULL selection."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .object_conditioning import role_supervision_mask


class RoleConditionedRefineMaskHead(nn.Module):
    """Fuse local VLM semantics with half/quarter-resolution RGB/XYZ details.

    The inherited packet supplies identity conditions and conservative local
    support. New masks may grow beyond its point hints, but cannot activate a
    role with no inherited local support. Global tokens/geometry/NULL are kept.
    """

    def __init__(self, feature_channels, role_token_dim, hidden_dim=32):
        super().__init__()
        if min(feature_channels, role_token_dim) <= 0 or hidden_dim < 2:
            raise ValueError('refine mask dimensions must be positive, hidden_dim >= 2')
        self.semantic = nn.Conv2d(feature_channels, hidden_dim, 1)
        self.detail_half = nn.Sequential(nn.Conv2d(7, hidden_dim // 2, 3, 2, 1), nn.GELU())
        self.detail_quarter = nn.Sequential(
            nn.Conv2d(hidden_dim // 2, hidden_dim, 3, 2, 1), nn.GELU())
        self.fuse = nn.Sequential(nn.Conv2d(2 * hidden_dim, hidden_dim, 3, padding=1), nn.GELU())
        self.keys = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.edge_keys = nn.Conv2d(hidden_dim // 2, hidden_dim, 1)
        self.query = nn.Linear(role_token_dim, hidden_dim)
        self.role_embedding = nn.Parameter(torch.randn(2, hidden_dim) * .02)
        self.bias = nn.Parameter(torch.full((2,), -2.0))
        # The extra edge residual starts at zero; the dense head is opt-in.
        nn.init.zeros_(self.edge_keys.weight)
        nn.init.zeros_(self.edge_keys.bias)

    def forward(self, features, rgb, xyz, packet, *, detach_action=False):
        if rgb.ndim != 5 or rgb.shape != xyz.shape or rgb.shape[2] != 3:
            raise ValueError('refine RGB/XYZ must have matching [B,V,3,H,W] shapes')
        batch, views, _, height, width = rgb.shape
        if features.ndim != 4 or features.shape[0] != batch * views:
            raise ValueError('refine feature batch must equal B*V')
        tokens = packet['role_tokens']
        if tokens.shape[:2] != (batch, 2):
            raise ValueError('inherited role tokens must have shape [B,2,D]')
        support = torch.isfinite(xyz).all(dim=2)
        support &= ~(xyz.eq(0).all(dim=2) | xyz.eq(-1).all(dim=2))
        clean_xyz = torch.where(support[:, :, None], xyz, 0.)
        clean_rgb = torch.where(support[:, :, None],
                                torch.nan_to_num(rgb, nan=0., posinf=0., neginf=0.), 0.)
        details = torch.cat((clean_rgb, clean_xyz, support[:, :, None].to(rgb.dtype)), dim=2)
        half = self.detail_half(details.reshape(batch * views, 7, height, width).to(features.dtype))
        quarter = self.detail_quarter(half)
        semantic = F.interpolate(self.semantic(features), size=quarter.shape[-2:],
                                 mode='bilinear', align_corners=False)
        dense = self.fuse(torch.cat((semantic, quarter), dim=1))
        keys = F.interpolate(self.keys(dense), size=half.shape[-2:],
                             mode='bilinear', align_corners=False) + self.edge_keys(half)
        keys = keys.reshape(batch, views, -1, *half.shape[-2:])
        query = self.query(tokens.to(features.dtype)) + self.role_embedding[None]
        logits = torch.einsum('brd,bvdhw->bvrhw', query, keys) / math.sqrt(query.shape[-1])
        logits = F.interpolate(logits.reshape(batch * views, 2, *half.shape[-2:]),
                               size=(height, width), mode='bilinear', align_corners=False)
        logits = logits.reshape(batch, views, 2, height, width).float()
        logits = logits + self.bias[None, None, :, None, None]
        probabilities = logits.sigmoid() * support[:, :, None]
        valid = packet['valid'].bool() & packet['role_token_valid'].bool()
        valid = valid & support.any(dim=(1, 2, 3))[:, None]
        non_null = 1 - packet['reference_null_probability']
        valid = torch.stack((valid[:, 0], valid[:, 1] & non_null.gt(1e-6)), dim=1)
        prior = probabilities * valid[:, None, :, None, None]
        prior = torch.stack((prior[:, :, 0], prior[:, :, 1] * non_null[:, None, None, None]), dim=2)
        # Detach only the local map path; the coarse/global conditioning path
        # remains unchanged, and auxiliary supervision still sees live logits.
        if detach_action:
            prior = prior.detach()
        result = dict(packet)
        result.update(prior=prior, prior_logits=torch.logit(prior.clamp(1e-5, 1 - 1e-5)),
                      valid=valid, masks_refined=True, refine_mask_logits=logits,
                      refine_pixel_valid=support)
        return result


def refine_role_mask_losses(logits, teacher, role_valid, pixel_valid,
                            role_present=None, role_present_known=None):
    """BCE/Dice on sparse teacher footprints, not fabricated dense background.

    Current replay teachers are Gaussian point projections. Pixels outside
    their nonzero support are unknown, and empty views/invalid roles are
    ignored. This is weak heatmap supervision, not dense instance-mask GT.
    """
    if logits.ndim != 5 or logits.shape[2] != 2:
        raise ValueError('refine logits must have shape [B,V,2,H,W]')
    batch, views, _, height, width = logits.shape
    if teacher.shape[:3] != (batch, views, 2) or teacher.ndim != 5:
        raise ValueError('refine teacher must have shape [B,V,2,H,W]')
    teacher = teacher.detach().to(device=logits.device, dtype=logits.dtype)
    if teacher.shape[-2:] != (height, width):
        teacher = F.interpolate(teacher.reshape(batch * views, 2, *teacher.shape[-2:]),
                                size=(height, width), mode='area').reshape_as(logits)
    supervised = role_supervision_mask(teacher, role_valid, role_present, role_present_known)
    if pixel_valid.shape != (batch, views, height, width):
        raise ValueError('refine pixel validity must have shape [B,V,H,W]')
    known = supervised[..., None, None] & pixel_valid[:, :, None].bool()
    known &= torch.isfinite(teacher) & teacher.gt(0)
    zero = logits.sum() * 0.
    if not known.any():
        return dict(mask=zero, bce=zero, dice=zero)
    targets = torch.nan_to_num(teacher).clamp(0, 1)
    bce = F.binary_cross_entropy_with_logits(logits[known], targets[known])
    weights = known.to(logits.dtype)
    prediction = logits.sigmoid() * weights
    targets = targets * weights
    dice = 1 - (2 * (prediction * targets).sum(dim=(-2, -1)) + 1) / (
        prediction.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1)) + 1)
    dice = dice[known.any(dim=(-2, -1))].mean()
    return dict(mask=bce + dice, bce=bce, dice=dice)
