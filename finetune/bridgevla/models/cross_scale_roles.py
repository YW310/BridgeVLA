"""Stateless coarse-to-refine role transport; no teacher or temporal state."""

import torch

from .oracle_prior import rasterize_instance_points


def transform_role_geometry(geometry, crop_center, scale):
    """Transform supported centers/spreads with the actual x2=s*(x1-c)."""
    if geometry.ndim != 2 or geometry.shape[1] != 17:
        raise ValueError('role geometry must have shape [B,17]')
    if crop_center.shape != (geometry.shape[0], 3):
        raise ValueError('crop_center must have shape [B,3]')
    scale = torch.as_tensor(scale, device=geometry.device, dtype=geometry.dtype)
    if scale.numel() != 1 or not torch.isfinite(scale).all() or scale.item() <= 0:
        raise ValueError('crop scale must be a finite positive scalar')
    valid = geometry[:, 15:17].bool()
    mask = valid[..., None]
    centers = (geometry[:, :6].reshape(-1, 2, 3) - crop_center[:, None]) * scale
    spreads = geometry[:, 6:12].reshape(-1, 2, 3) * scale.abs()
    displacement = geometry[:, 12:15] * scale
    displacement = torch.where(valid.all(dim=1, keepdim=True), displacement, 0.)
    return torch.cat((torch.where(mask, centers, 0.).flatten(1), torch.where(mask, spreads, 0.).flatten(1),
                      displacement, geometry[:, 15:17]), dim=1)


def inherit_coarse_roles(stage_output, crop_center, scale, rendered_xyz, project,
                         sigma=2.0):
    """Reproject predicted point hints, retaining differentiable global context.

    A current rendered XYZ match (within three pixel pitches of the unit cube)
    is required per point/view. Merely projecting onto a nonempty/occluding
    pixel is not evidence for this role. No local support means unknown, not
    NULL; neither role tokens nor coarse geometry is replaced by another object.
    Point hints are discrete and are not a trainable refine role-map decoder.
    """
    required = ('object_slot_points', 'object_slot_valid', 'object_slot_geometry',
                'object_slot_role_tokens', 'object_slot_role_token_valid',
                'object_slot_reference_null_probability',
                'object_slot_reference_is_null', 'object_slot_confidence')
    if any(key not in stage_output for key in required):
        raise ValueError('coarse role inheritance requires predicted role outputs')
    coarse_points = stage_output['object_slot_points']
    if coarse_points.ndim != 4 or coarse_points.shape[1] != 2 or coarse_points.shape[-1] != 3:
        raise ValueError('coarse points must have shape [B,2,P,3]')
    batch, _, count, _ = coarse_points.shape
    if rendered_xyz.ndim != 5 or rendered_xyz.shape[0] != batch or rendered_xyz.shape[2] != 3:
        raise ValueError('rendered_xyz must have shape [B,V,3,H,W]')
    views, height, width = rendered_xyz.shape[1], rendered_xyz.shape[-2], rendered_xyz.shape[-1]
    geometry = transform_role_geometry(stage_output['object_slot_geometry'], crop_center, scale)
    # Preserve actual predicted points; invalid/padded ones must never be clamped
    # into the crop or treated as a new object at its origin.
    point_valid = torch.isfinite(coarse_points).all(dim=-1)
    point_valid &= ~(coarse_points.eq(0).all(dim=-1) | coarse_points.eq(-1).all(dim=-1))
    point_valid &= stage_output['object_slot_valid'][:, :, None].bool()
    safe_points = torch.where(point_valid[..., None], coarse_points, 0.)
    points = float(scale) * (safe_points - crop_center[:, None, None])
    with torch.no_grad():
        xy = project(points.reshape(batch, 2 * count, 3)).reshape(batch, 2, count, views, 2)
        finite = torch.isfinite(xy).all(dim=-1)
        # Match the CUDA renderer: raw frustum bounds, then floor/truncation
        # for nonnegative pixel coordinates (not nearest-pixel rounding).
        pixel = torch.floor(torch.nan_to_num(xy)).long()
        x, y = pixel[..., 0], pixel[..., 1]
        inside = finite & xy[..., 0].ge(0) & xy[..., 0].lt(width)
        inside &= xy[..., 1].ge(0) & xy[..., 1].lt(height)
        indices = (y.clamp(0, height - 1) * width + x.clamp(0, width - 1))
        indices = indices.permute(0, 3, 1, 2).reshape(batch, views, 2 * count)
        image_xyz = rendered_xyz.flatten(-2).transpose(2, 3)
        observed = image_xyz.gather(2, indices[..., None].expand(-1, -1, -1, 3))
        observed = observed.reshape(batch, views, 2, count, 3).permute(0, 2, 3, 1, 4)
        observed_valid = torch.isfinite(observed).all(dim=-1)
        observed_valid &= ~(observed.eq(0).all(dim=-1) | observed.eq(-1).all(dim=-1))
        distance = torch.linalg.vector_norm(observed - points[..., None, :], dim=-1)
        tolerance = 6.0 / min(height, width)
        visible = inside & observed_valid & point_valid[..., None] & distance.le(tolerance)
        # Integer pixel centers also keep hint rasterization aligned with the
        # verified XYZ pixel, including the last row/column.
        xy = pixel.to(xy.dtype).masked_fill(~visible[..., None], torch.nan)
        priors = torch.stack([
            rasterize_instance_points(xy[:, role], torch.ones(batch, device=points.device, dtype=torch.bool),
                                      (height, width), sigma)
            for role in range(2)
        ], dim=2)
        # Reference spatial support carries the same coarse non-NULL mass.
        priors[:, :, 1] *= (1 - stage_output['object_slot_reference_null_probability'])[:, None, None, None]
        local_valid = visible.any(dim=(2, 3))
        local_valid[:, 1] &= (1 - stage_output['object_slot_reference_null_probability']).gt(1e-6)
    return {
        'prior': priors,
        'prior_logits': torch.logit(priors.clamp(1e-5, 1 - 1e-5)),
        'points': torch.where(local_valid[:, :, None, None] & point_valid[..., None], points, 0.),
        'valid': local_valid,
        'geometry': geometry,
        'role_tokens': stage_output['object_slot_role_tokens'],
        'role_token_valid': stage_output['object_slot_role_token_valid'],
        'confidence': stage_output['object_slot_confidence'],
        'reference_null_probability': stage_output['object_slot_reference_null_probability'],
        'reference_is_null': stage_output['object_slot_reference_is_null'],
        'roles_inherited': True,
        'predictor_type': stage_output.get('object_slot_predictor_type', 'slots'),
    }
