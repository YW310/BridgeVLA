"""Observation-only transport of dense role scores across cube/crop frames."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def reproject_role_heatmaps(probabilities, coarse_xyz, refine_xyz, crop_center, scale,
                            project, role_valid, *, chunk_size=8192):
    """Sample coarse scores at each refine surface point with XYZ agreement.

    x_coarse = x_refine / scale + crop_center. Only views observing that same
    surface contribute; their probabilities are averaged, without peak
    normalization. The per-role validity map distinguishes unknown projection
    from a verified low response. Spatial transport is detached like the old
    point hints; tokens/global geometry retain their existing gradient paths.
    Chunking bounds temporary memory without discarding low-scoring pixels.
    """
    if probabilities.ndim != 5 or probabilities.shape[2] != 2:
        raise ValueError('probabilities must have shape [B,V,2,H,W]')
    batch, views = probabilities.shape[:2]
    if coarse_xyz.ndim != 5 or coarse_xyz.shape[:3] != (batch, views, 3):
        raise ValueError('coarse XYZ must have shape [B,V,3,H,W]')
    if refine_xyz.ndim != 5 or refine_xyz.shape[0] != batch or refine_xyz.shape[2] != 3:
        raise ValueError('refine XYZ must have shape [B,Vf,3,Hf,Wf]')
    if crop_center.shape != (batch, 3) or not torch.isfinite(crop_center).all():
        raise ValueError('crop center must be finite [B,3]')
    if role_valid.shape != (batch, 2):
        raise ValueError('role validity must have shape [B,2]')
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    scale = float(scale)
    if not 0 < scale < float('inf'):
        raise ValueError('crop scale must be finite and positive')
    height, width = coarse_xyz.shape[-2:]
    out_views, out_height, out_width = refine_xyz.shape[1], *refine_xyz.shape[-2:]
    scores = probabilities.float().reshape(batch * views, 2, *probabilities.shape[-2:])
    if scores.shape[-2:] != (height, width):
        scores = F.interpolate(scores, size=(height, width), mode='bilinear', align_corners=False)
    scores = scores.reshape(batch, views, 2, height * width).transpose(2, 3)
    source_xyz = coarse_xyz.float().flatten(-2).transpose(2, 3)
    points = refine_xyz.float().permute(0, 1, 3, 4, 2).reshape(batch, -1, 3)
    total = points.shape[1]
    result = points.new_zeros(batch, total, 2)
    known = torch.zeros_like(result, dtype=torch.bool)
    role_valid = role_valid.to(device=points.device).bool()
    center = crop_center.to(points)[:, None]
    for start in range(0, total, chunk_size):
        local = points[:, start:start + chunk_size]
        valid = torch.isfinite(local).all(-1)
        valid &= ~(local.eq(0).all(-1) | local.eq(-1).all(-1))
        valid &= local.ge(-1).all(-1) & local.le(1).all(-1)
        world = torch.where(valid[..., None], local / scale + center, 0.)
        valid &= world.ge(-1).all(-1) & world.le(1).all(-1)
        xy = project(world)
        if xy.shape != (batch, local.shape[1], views, 2):
            raise ValueError('project must return [B,N,V,2]')
        inside = torch.isfinite(xy).all(-1)
        inside &= xy[..., 0].ge(0) & xy[..., 0].lt(width)
        inside &= xy[..., 1].ge(0) & xy[..., 1].lt(height)
        pixel = torch.nan_to_num(xy, nan=0., posinf=0., neginf=0.).floor().long()
        index = (pixel[..., 1].clamp(0, height - 1) * width
                 + pixel[..., 0].clamp(0, width - 1)).transpose(1, 2)
        observed = source_xyz.gather(2, index[..., None].expand(-1, -1, -1, 3)).transpose(1, 2)
        surface = torch.isfinite(observed).all(-1)
        surface &= ~(observed.eq(0).all(-1) | observed.eq(-1).all(-1))
        surface &= torch.linalg.vector_norm(observed - world[:, :, None], dim=-1).le(2. / min(height, width))
        surface &= inside & valid[..., None]
        values = scores.gather(2, index[..., None].expand(-1, -1, -1, 2)).transpose(1, 2)
        accepted = surface[..., None] & torch.isfinite(values) & role_valid[:, None, None]
        count = accepted.sum(dim=2)
        summed = torch.where(accepted, values.clamp(0, 1), 0.).sum(dim=2)
        result[:, start:start + local.shape[1]] = summed / count.clamp_min(1)
        known[:, start:start + local.shape[1]] = count.gt(0)
    shape = (batch, out_views, out_height, out_width, 2)
    return dict(probabilities=result.reshape(shape).permute(0, 1, 4, 2, 3),
                valid=known.reshape(shape).permute(0, 1, 4, 2, 3))
