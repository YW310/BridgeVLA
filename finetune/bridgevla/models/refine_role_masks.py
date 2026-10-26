"""Local mask readout conditioned on inherited roles; no slot/NULL selection."""

import math

import torch
from torch import nn
import torch.nn.functional as F

from .object_conditioning import role_supervision_mask


HEATMAP_FUSION_VERSION = 'spatial_tokens_v1'


class RefineHeatmapTokenFusion(nn.Module):
    """Fuse aligned heatmap patches into post-VLM image tokens, per view.

    Patch coordinates/XYZ belong to the rendered refine crop, never the
    original coarse image. T/R remain separate channels; known zero scores
    differ from unknown projection. An empty view contributes no update.
    """

    def __init__(self, channels):
        super().__init__()
        # T/R probabilities, T/R coverage, XYZ, UV and view coordinate.
        self.encode = nn.Sequential(nn.Linear(10, channels), nn.GELU(),
                                    nn.Linear(channels, channels))
        self.position = nn.Linear(6, channels)
        self.image_norm = nn.LayerNorm(channels)
        self.guide_norm = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.output = nn.Linear(channels, channels)

    @staticmethod
    def patch_inputs(probabilities, valid, xyz, support, size):
        """Pool onto the exact image-token grid without treating unknown as 0."""
        batch, views, _, height, width = xyz.shape

        def pool(value):
            return F.adaptive_avg_pool2d(value.reshape(batch * views, -1, height, width), size)

        coverage = pool(valid.float())
        scores = pool(probabilities * valid) / coverage.clamp_min(1e-6)
        depth = pool(support[:, :, None].float())
        positions = pool(xyz) / depth.clamp_min(1e-6)
        ph, pw = size
        y, x = torch.meshgrid(
            (torch.arange(ph, device=xyz.device, dtype=xyz.dtype) + .5) * (2. / ph) - 1,
            (torch.arange(pw, device=xyz.device, dtype=xyz.dtype) + .5) * (2. / pw) - 1,
            indexing='ij')
        uv = torch.stack((x, y))[None].expand(batch * views, -1, -1, -1)
        view = (torch.arange(views, device=xyz.device, dtype=xyz.dtype) + .5) * (2. / views) - 1
        view = view[None, :, None, None, None].expand(batch, views, 1, ph, pw)
        position = torch.cat((positions, uv, view.reshape(batch * views, 1, ph, pw)), dim=1)
        inputs = torch.cat((scores, coverage, position), dim=1).flatten(2).transpose(1, 2)
        return inputs, coverage.any(dim=1).flatten(1), depth.gt(0).flatten(1)

    def forward(self, images, probabilities, valid, xyz, support):
        # Spatial guides are predictions/observations, not a second GT path.
        inputs, known, depth = self.patch_inputs(
            probabilities.detach(), valid.detach(), xyz.detach(), support, images.shape[-2:])
        inputs = inputs.to(images.dtype)
        spatial = self.position(inputs[..., 4:])
        guide = self.guide_norm(self.encode(inputs) + spatial)
        image_tokens = images.flatten(2).transpose(1, 2)
        queries = self.query(self.image_norm(image_tokens) + spatial)
        scores = torch.bmm(queries, self.key(guide).transpose(1, 2)) / math.sqrt(images.shape[1])
        # Avoid NaNs for all-unknown crops, including mixed-precision runs.
        weights = scores.float().masked_fill(~known[:, None], -torch.finfo(torch.float32).max).softmax(-1)
        weights = weights * known[:, None]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-6)
        attended = torch.bmm(weights.to(images.dtype), self.value(guide))
        # Aligned skip preserves each patch location; attention supplies context.
        update = self.output(attended + guide * known[..., None])
        update = update * (known.any(-1, keepdim=True) & depth)[..., None]
        return images + update.transpose(1, 2).reshape_as(images)


class CoarseConditionedLocalQuery(nn.Module):
    """Two independent role queries read current, spatially supported features.

    Geometry supplies a soft spatial cue, not an instance/visibility label.
    A three-spread neighbourhood of observed XYZ can recover missing point
    hints; neither empty space nor a crop-external role fabricates an object.
    """

    def __init__(self, token_dim, channels):
        super().__init__()
        self.query_norm = nn.LayerNorm(token_dim)
        self.memory_norm = nn.LayerNorm(channels)
        self.query = nn.Linear(token_dim, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.position = nn.Linear(3, channels)
        self.token_expand = nn.Linear(channels, token_dim)
        self.spatial_gain = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.token_expand.weight)
        nn.init.zeros_(self.token_expand.bias)

    @staticmethod
    def _pool(value, size, *, maximum=False):
        batch, views, channels, height, width = value.shape
        flat = value.reshape(batch * views, channels, height, width)
        pool = F.adaptive_max_pool2d if maximum else F.adaptive_avg_pool2d
        return pool(flat, size).reshape(batch, views, channels, *size)

    def forward(self, tokens, memory, xyz, support, packet):
        batch, views, channels, mh, mw = memory.shape
        height, width = xyz.shape[-2:]
        expected = (batch, views, 2, height, width)
        hints = packet['prior'].detach().to(device=memory.device, dtype=memory.dtype)
        if hints.shape != expected:
            raise ValueError('inherited hints must match refine [B,V,2,H,W]')
        hints = torch.nan_to_num(hints, nan=0., posinf=0., neginf=0.).clamp_min(0)
        hints = hints * support[:, :, None]
        # Remove amplitude (including Reference non-NULL mass) from spatial
        # hints. The actual coarse NULL posterior is applied once to output maps.
        hints = hints / hints.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        has_hint = hints.gt(0).any(dim=(-2, -1), keepdim=True)
        bias = torch.where(has_hint, hints.clamp_min(1e-4).log(), 0.)
        evidence = hints.gt(0)
        geometry = packet.get('geometry')
        if geometry is not None:
            if geometry.shape != (batch, 17):
                raise ValueError('inherited geometry must have shape [B,17]')
            geometry = geometry.to(device=memory.device, dtype=memory.dtype)
            centers = geometry[:, :6].reshape(batch, 2, 3)
            spread = geometry[:, 6:12].reshape(batch, 2, 3)
            known = geometry[:, 15:17].gt(.5)
            known = known & torch.isfinite(centers).all(dim=-1)
            known = known & torch.isfinite(spread).all(dim=-1) & spread.ge(0).all(dim=-1)
            centers = torch.where(known[..., None], centers, 0.)
            # Spread is visible-point standard deviation, not full object size.
            spread = torch.where(known[..., None], spread, 1.).clamp_min(6. / min(height, width))
            distance = ((xyz[:, :, None] - centers[:, None, :, :, None, None])
                        / spread[:, None, :, :, None, None]).square().sum(dim=3)
            geometric = known[:, None, :, None, None]
            bias = bias + torch.where(geometric, -.5 * distance.clamp_max(50.), 0.)
            evidence = evidence | (geometric & distance.le(9.))
        semantic = packet['role_token_valid'].bool()
        non_null = 1 - packet['reference_null_probability']
        semantic = torch.stack((semantic[:, 0], semantic[:, 1] & non_null.gt(1e-6)), dim=1)
        evidence = evidence & support[:, :, None] & semantic[:, None, :, None, None]
        active = evidence.any(dim=(1, 3, 4))
        size = (mh, mw)
        pixel_weight = self._pool(support[:, :, None].to(memory.dtype), size)
        pooled_xyz = self._pool(xyz, size) / pixel_weight.clamp_min(1e-6)
        pooled_bias = self._pool(bias * support[:, :, None], size) / pixel_weight.clamp_min(1e-6)
        permitted = self._pool(evidence.to(memory.dtype), size, maximum=True).bool()
        permitted = permitted.permute(0, 2, 1, 3, 4).reshape(batch, 2, -1)
        spatial = pooled_bias.permute(0, 2, 1, 3, 4).reshape(batch, 2, -1)
        values = memory.permute(0, 1, 3, 4, 2).reshape(batch, -1, channels)
        position = pooled_xyz.permute(0, 1, 3, 4, 2).reshape(batch, -1, 3)
        values = self.memory_norm(values)
        query = self.query(self.query_norm(tokens.to(memory.dtype)))
        keys = self.key(values) + self.position(position)
        scores = torch.einsum('brd,bnd->brn', query, keys) / math.sqrt(channels)
        scores = scores + spatial
        # An all-masked role must produce exact zero, not softmax(-inf) NaNs.
        attention = scores.masked_fill(~permitted, -torch.finfo(scores.dtype).max).softmax(dim=-1)
        attention = attention * permitted
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        local = torch.einsum('brn,bnd->brd', attention, self.value(values))
        delta = self.token_expand(local) * active[..., None]
        return tokens + delta, bias, active, delta, evidence


class RoleConditionedRefineMaskHead(nn.Module):
    """Fuse local VLM semantics with half/quarter-resolution RGB/XYZ details.

    The inherited packet supplies identity conditions and conservative local
    support. Optional local queries also consume geometric/spatial evidence,
    update the inherited tokens, and leave unsupported roles unchanged.
    Global geometry/NULL are never reselected.
    """

    def __init__(self, feature_channels, role_token_dim, hidden_dim=32, *, local_query=False,
                 heatmap_residual=False):
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
        self.local_query = (CoarseConditionedLocalQuery(role_token_dim, hidden_dim)
                            if local_query else None)
        self.heatmap_residual = bool(heatmap_residual)
        if self.heatmap_residual:
            self.heatmap_tokens = RefineHeatmapTokenFusion(hidden_dim)
            self.residual_output = nn.Conv2d(hidden_dim + 2, 2, 1)
            nn.init.zeros_(self.residual_output.weight)
            nn.init.zeros_(self.residual_output.bias)

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
        projected = projected_valid = None
        if self.heatmap_residual:
            projected = packet['projected_role_probabilities'].detach().to(device=features.device, dtype=torch.float32)
            projected_valid = packet['projected_role_valid'].detach().to(device=features.device).bool()
            expected = (batch, views, 2, height, width)
            if projected.shape != expected or projected_valid.shape != expected:
                raise ValueError('projected role maps/validity must match [B,V,2,H,W]')
            projected_valid = projected_valid & torch.isfinite(projected) & support[:, :, None]
            projected = torch.where(projected_valid, projected.clamp(0, 1), 0.)
        quarter = self.detail_quarter(half)
        semantic = self.semantic(features)
        if self.heatmap_residual:
            semantic = self.heatmap_tokens(semantic, projected, projected_valid, clean_xyz, support)
        semantic = F.interpolate(semantic, size=quarter.shape[-2:], mode='bilinear', align_corners=False)
        dense = self.fuse(torch.cat((semantic, quarter), dim=1))
        query_state = None
        if self.local_query is not None:
            memory = dense.reshape(batch, views, -1, *dense.shape[-2:])
            query_state = self.local_query(tokens, memory, clean_xyz, support, packet)
            tokens = query_state[0]
        keys = F.interpolate(self.keys(dense), size=half.shape[-2:],
                             mode='bilinear', align_corners=False) + self.edge_keys(half)
        keys = keys.reshape(batch, views, -1, *half.shape[-2:])
        query = self.query(tokens.to(features.dtype)) + self.role_embedding[None]
        logits = torch.einsum('brd,bvdhw->bvrhw', query, keys) / math.sqrt(query.shape[-1])
        if self.heatmap_residual:
            residual_input = torch.cat((keys, logits.to(keys.dtype)), dim=2)
            logits = self.residual_output(residual_input.reshape(batch * views, -1, *half.shape[-2:]))
        logits = F.interpolate(logits.reshape(batch * views, 2, *half.shape[-2:]),
                               size=(height, width), mode='bilinear', align_corners=False)
        logits = logits.reshape(batch, views, 2, height, width).float()
        residual = None
        if self.heatmap_residual:
            residual = logits
            # Unknown projection starts near-dark but remains trainable from
            # actual GT/action losses. It is not a fabricated background label.
            base = torch.where(projected_valid, projected, 1e-4)
            logits = torch.logit(base.clamp(1e-4, 1 - 1e-4)) + residual
        else:
            logits = logits + self.bias[None, None, :, None, None]
            if query_state is not None:
                logits = logits + self.local_query.spatial_gain * query_state[1]
        role_support = support[:, :, None].expand(-1, -1, 2, -1, -1)
        if query_state is not None and not self.heatmap_residual:
            # Use the same role/view-specific support as query attention.
            # A role active in one view must not occupy unrelated surfaces in
            # every other view merely because those pixels have valid depth.
            role_support = role_support & query_state[4]
        probabilities = logits.sigmoid() * role_support
        if self.heatmap_residual:
            valid = packet['role_token_valid'].bool()
        elif query_state is not None:
            valid = query_state[2]
        else:
            valid = packet['valid'].bool() & packet['role_token_valid'].bool()
        valid = valid & support.any(dim=(1, 2, 3))[:, None]
        non_null = 1 - packet['reference_null_probability']
        valid = torch.stack((valid[:, 0], valid[:, 1] & non_null.gt(1e-6)), dim=1)
        prior = probabilities * valid[:, None, :, None, None]
        prior = torch.stack((prior[:, :, 0], prior[:, :, 1] * non_null[:, None, None, None]), dim=2)
        # Detach only the local map path; tokens (including optional local
        # updates) still reach the anchor, and auxiliary logits stay trainable.
        if detach_action:
            prior = prior.detach()
        result = dict(packet)
        if self.heatmap_residual:
            projected_prior = torch.stack((projected[:, :, 0],
                                           projected[:, :, 1] * non_null[:, None, None, None]), dim=2)
            result.update(heatmap_residual=True, refine_projected_prior=projected_prior,
                          refine_projection_valid=projected_valid,
                          refine_residual_logits=residual)
        if query_state is not None:
            result.update(role_tokens=tokens, queries_refined=True,
                          refine_query_active=query_state[2],
                          refine_point_hint_valid=packet['valid'].bool(),
                          refine_query_update_norm=query_state[3].detach().norm(dim=-1))
        result.update(prior=prior, prior_logits=torch.logit(prior.clamp(1e-5, 1 - 1e-5)),
                      valid=valid, masks_refined=True, refine_mask_logits=logits,
                      refine_pixel_valid=support, refine_role_pixel_valid=role_support)
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
