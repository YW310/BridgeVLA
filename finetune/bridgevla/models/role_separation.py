"""Visibility-checked T/R separation labels, used only by auxiliary losses."""
import torch
import torch.nn.functional as F

from .object_conditioning import role_supervision_mask


@torch.no_grad()
def visible_role_support(points, role_valid, rendered_xyz, project):
    """Mark exact GT point pixels whose rendered surface agrees in stage XYZ.

    Inputs must share the current normalized cube (including the refine crop).
    One pixel pitch of 3D tolerance allows raster quantization; this is a
    conservative sampled-surface check, not completion of invisible geometry.
    Neither predictions nor learned NULL probabilities choose these labels.
    """
    if points.ndim != 4 or points.shape[1] != 2 or points.shape[-1] != 3:
        raise ValueError('role points must have shape [B,2,P,3]')
    batch, _, count, _ = points.shape
    if role_valid.shape != (batch, 2):
        raise ValueError('role_valid must have shape [B,2]')
    if rendered_xyz.ndim != 5 or rendered_xyz.shape[0] != batch or rendered_xyz.shape[2] != 3:
        raise ValueError('rendered_xyz must have shape [B,V,3,H,W]')
    _, views, _, height, width = rendered_xyz.shape
    valid = torch.isfinite(points).all(dim=-1) & role_valid.bool()[..., None]
    valid &= points.ge(-1).all(dim=-1) & points.le(1).all(dim=-1)
    safe = torch.where(valid[..., None], points, 0.)
    xy = project(safe.reshape(batch, 2 * count, 3)).reshape(batch, 2, count, views, 2)
    inside = torch.isfinite(xy).all(dim=-1)
    inside &= xy[..., 0].ge(0) & xy[..., 0].lt(width) & xy[..., 1].ge(0) & xy[..., 1].lt(height)
    pixel = torch.nan_to_num(xy, nan=0., posinf=0., neginf=0.).floor().long()
    indices = pixel[..., 1].clamp(0, height - 1) * width + pixel[..., 0].clamp(0, width - 1)
    indices = indices.permute(0, 3, 1, 2)
    observed = rendered_xyz.flatten(-2).transpose(2, 3).gather(
        2, indices.reshape(batch, views, -1)[..., None].expand(-1, -1, -1, 3))
    observed = observed.reshape(batch, views, 2, count, 3).permute(0, 2, 3, 1, 4)
    supported = torch.isfinite(observed).all(dim=-1)
    supported &= ~(observed.eq(0).all(dim=-1) | observed.eq(-1).all(dim=-1))
    distance = torch.linalg.vector_norm(observed.float() - safe.float()[..., None, :], dim=-1)
    supported &= distance.le(2. / min(height, width)) & valid[..., None] & inside
    result = torch.zeros(batch, views, 2, height * width, dtype=torch.int32, device=points.device)
    result.scatter_add_(3, indices, supported.permute(0, 3, 1, 2).to(torch.int32))
    return result.gt(0).reshape(batch, views, 2, height, width)


def role_separation_loss(probabilities, teacher, role_valid, visible_support=None,
                         role_present=None, role_present_known=None, pixel_valid=None,
                         logits=None):
    """Suppress a role on the other role's verified, nonoverlapping anchors.

    Missing surface checks yield zero (not Gaussian-as-visibility guesses).
    Both roles must have valid visible anchors in the view. Overlap in either
    projected teacher support is unknown for exclusivity, so is ignored. The
    background, occluded points, crop-external roles and NULL are not negatives.
    Raw masks are used so learned NULL cannot suppress this loss. Pass logits
    when available for stable BCE gradients even for saturated predictions.
    """
    zero = torch.nan_to_num(probabilities).sum() * 0.
    empty = dict(loss=zero, supervised_pixels=zero.detach())
    if visible_support is None:
        return empty
    if probabilities.ndim != 5 or probabilities.shape[2] != 2:
        raise ValueError('role probabilities must have shape [B,V,2,H,W]')
    batch, views, _, height, width = probabilities.shape
    if teacher.ndim != 5 or teacher.shape[:3] != (batch, views, 2):
        raise ValueError('role teacher must have shape [B,V,2,H,W]')
    if visible_support.dtype != torch.bool or visible_support.shape != teacher.shape:
        raise ValueError('visible_support must be bool and match the original role teacher')
    target = teacher.detach().to(device=probabilities.device, dtype=probabilities.dtype)
    visible = visible_support.detach().to(probabilities.device)
    if target.shape[-2:] != (height, width):
        # Any projected footprint in a pooled cell excludes it as a negative;
        # do not average away overlap, sparse anchors, NaNs or Gaussian tails.
        def pool(value):
            return F.adaptive_max_pool2d(value.reshape(batch * views, 2, *value.shape[-2:]),
                                        (height, width)).reshape_as(probabilities)
        invalid = pool((~torch.isfinite(target)).float()).bool()
        target = pool(torch.nan_to_num(target, nan=0., posinf=0., neginf=0.)).masked_fill(invalid, float('nan'))
        visible = pool(visible.float()).bool()
    supervised = role_supervision_mask(target, role_valid, role_present, role_present_known)
    if pixel_valid is not None:
        if pixel_valid.shape != (batch, views, height, width):
            raise ValueError('pixel_valid must have shape [B,V,H,W] at prediction resolution')
        visible = visible & pixel_valid[:, :, None].to(visible.device).bool()
    visible = visible & supervised[..., None, None] & torch.isfinite(target) & target.gt(0)
    paired = visible.any(dim=(-2, -1)).all(dim=2)
    other = visible.flip(2)
    negative = other & torch.isfinite(target) & target.eq(0) & paired[:, :, None, None, None]
    count = negative.sum().to(probabilities.dtype)
    if not negative.any():
        return empty
    if logits is not None and logits.shape != probabilities.shape:
        raise ValueError('logits must match role probabilities')
    selected = probabilities[negative] if logits is None else logits.float()[negative]
    if not torch.isfinite(selected).all():
        raise ValueError('Nonfinite predictions at supervised role separation anchors')
    # BCE against zero supplies a gradient even when both maps collapse high.
    loss = (F.softplus(selected).mean() if logits is not None else
            -torch.log1p(-selected.clamp(0, 1 - 1e-6)).mean())
    return dict(loss=loss, supervised_pixels=count.detach())
