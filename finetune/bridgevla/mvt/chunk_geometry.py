"""Observed surface geometry aligned with rendered VLM patch tokens."""
import torch
from torch.nn import functional as F


def pool_chunk_geometry(xyz, coverage, grid_shape):
    """Pool visible pixels, including legitimate points at the frame origin.

    xyz/coverage are premultiplied render features from the SAME z-buffer as
    RGB. Dividing their pooled sums handles antialiasing and background pixels.
    """
    views, height, width = grid_shape
    if xyz.ndim != 5 or xyz.shape[1:3] != (views, 3) or coverage.shape != (*xyz.shape[:2], 1, *xyz.shape[-2:]):
        raise ValueError('Chunk rendered geometry requires [B,V,3,H,W] and coverage [B,V,1,H,W]')
    batch = xyz.shape[0]
    finite = torch.isfinite(xyz).all(2, keepdim=True) & torch.isfinite(coverage)
    coverage = torch.where(finite, coverage, 0.).clamp(0, 1)
    xyz = torch.where(finite & (coverage > 0), xyz, 0.)
    mass = F.adaptive_avg_pool2d(coverage.flatten(0, 1), (height, width))
    pooled = F.adaptive_avg_pool2d(xyz.flatten(0, 1), (height, width)) / mass.clamp_min(1e-8)
    valid = mass > 1e-8
    pooled = torch.where(valid, pooled, 0.)
    return (pooled.reshape(batch, views, 3, height, width).permute(0, 1, 3, 4, 2).reshape(batch, -1, 3),
            valid.reshape(batch, -1))


def chunk_geometry_to_world(output, cube_to_world):
    """Undo the exact predicted crop and per-sample world-to-cube transform."""
    for name in (None, 'mvt2'):
        if name is not None and name not in output:
            continue
        stage = output if name is None else output[name]
        if 'oht_chunk_xyz' not in stage:
            continue
        points = stage['oht_chunk_xyz']
        if name is not None:
            # trans_pc's inverse is defined on [B,3]. Recover its diagonal
            # affine map rather than broadcasting a [B,3] origin over [B,N,3].
            origin = output['rev_trans'](torch.zeros_like(points[:, 0]))
            unit = output['rev_trans'](torch.ones_like(points[:, 0]))
            points = points * (unit - origin)[:, None] + origin[:, None]
        points = torch.stack([reverse(value) for reverse, value in zip(cube_to_world, points)])
        stage['oht_chunk_xyz_world'] = torch.where(stage['oht_chunk_valid'][..., None], points, 0.).detach()
