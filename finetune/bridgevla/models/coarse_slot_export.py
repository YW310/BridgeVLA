"""Detached coarse Slot export using actual renderer support and cube inverse.

This optional offline hook performs one extra coverage render, never an image
encoder pass. It reads no episode, checkpoint, robot action or future state.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _numpy(value):
    value = value.detach().cpu()
    if value.dtype == torch.bfloat16:
        value = value.float()
    return value.numpy().copy()


def cube_inverse_metadata(reverse_cube, scene_cube, scene_world):
    """Recover the affine from the exact agent inverse and verify the scene."""
    if (scene_cube.ndim != 2 or scene_cube.shape[-1] != 3 or not len(scene_cube)
            or scene_world.shape != scene_cube.shape
            or not torch.isfinite(scene_cube).all() or not torch.isfinite(scene_world).all()):
        raise ValueError('coarse export needs matching finite nonempty cube/world scene points')
    origin = scene_cube.new_zeros(3)
    center = reverse_cube(origin)
    basis = reverse_cube(torch.eye(3, device=origin.device, dtype=origin.dtype)) - center
    scale = basis.diag().mean()
    if (not torch.isfinite(center).all() or not torch.isfinite(basis).all()
            or not torch.isfinite(scale) or scale <= 0
            or not torch.allclose(basis, torch.eye(3, device=basis.device, dtype=basis.dtype) * scale,
                                  rtol=2e-5, atol=2e-6)):
        raise ValueError('coarse cube inverse must be a finite positive isotropic affine')
    restored = reverse_cube(scene_cube)
    if (not torch.allclose(restored, scene_world, rtol=2e-5, atol=2e-6)
            or not torch.allclose(scene_cube * scale + center, scene_world, rtol=2e-5, atol=2e-6)):
        raise ValueError('coarse inverse does not restore the actual world scene')
    matrix = torch.eye(4, device=origin.device, dtype=origin.dtype)
    matrix[:3, :3] *= scale
    matrix[:3, 3] = center
    return dict(cube_center_world=_numpy(center), cube_scale_to_world=float(scale),
                matrix_world_from_source=_numpy(matrix))


def coarse_point_support(prior, rendered_xyz, role_points, role_valid, coverage, point_samples):
    """Reproduce predictor indices, then gather independently rendered coverage.

    Matching the current predictor extraction is checked against its actual
    returned points. A mismatch fails rather than silently guessing indices.
    XYZ sentinels reproduce predictor behavior; coverage is the independent
    observation support that prevents finite unsupported pixels from passing.
    """
    if rendered_xyz.ndim != 5 or rendered_xyz.shape[0] != 1 or rendered_xyz.shape[2] != 3:
        raise ValueError('coarse XYZ must be one sample [1,V,3,H,W]')
    _, views, _, height, width = rendered_xyz.shape
    if prior.ndim != 5 or prior.shape[:3] != (1, views, 2):
        raise ValueError('coarse prior must be [1,V,2,h,w]')
    count = min(point_samples, views * height * width)
    if role_points.shape != (1, 2, count, 3) or role_valid.shape != (1, 2):
        raise ValueError('coarse point/valid shape does not match predictor point_samples')
    if role_valid.dtype != torch.bool or coverage.shape != (1, views, height, width):
        raise ValueError('invalid coarse role validity or renderer coverage shape')
    if not torch.isfinite(prior).all() or not torch.isfinite(coverage).all():
        raise ValueError('coarse prior and coverage must be finite')
    if (coverage < -1e-6).any() or (coverage > 1. + 1e-6).any():
        raise ValueError('ones coverage render must be in [0,1]')
    scores = F.interpolate(prior.reshape(views, 2, *prior.shape[-2:]),
                           size=(height, width), mode='bilinear', align_corners=False)
    scores = scores.view(1, views, 2, height, width).permute(0, 2, 1, 3, 4).reshape(1, 2, -1)
    xyz = rendered_xyz.permute(0, 1, 3, 4, 2).reshape(1, -1, 3)
    finite = torch.isfinite(xyz).all(-1)
    background = xyz.eq(0).all(-1) | xyz.eq(-1).all(-1)
    xyz_valid = finite & ~background
    clean_xyz = torch.where(xyz_valid[..., None], xyz, 0.)
    indices = scores.masked_fill(~xyz_valid[:, None], -1.).topk(count, dim=-1).indices
    predicted = torch.gather(clean_xyz[:, None].expand(-1, 2, -1, -1), 2,
                             indices[..., None].expand(-1, -1, -1, 3))
    predicted *= role_valid[:, :, None, None]
    if not torch.allclose(predicted, role_points, rtol=2e-5, atol=2e-6):
        raise ValueError('coarse points do not match current predictor top-k extraction')
    supported_pixels = xyz_valid & coverage.flatten(1).gt(1e-6)
    point_valid = supported_pixels[:, None].expand(-1, 2, -1).gather(2, indices)
    point_valid &= role_valid[:, :, None]
    return point_valid, indices


@torch.no_grad()
def build_coarse_slot_packet(network, output, scene_cube, reverse_cube, scene_world):
    """Export the first/coarse stage only, for supported role-query checkpoints."""
    if getattr(network, 'training', False):
        raise ValueError('coarse Slot export requires eval mode')
    if getattr(network, 'object_slot_predictor_type', None) != 'role_queries':
        raise ValueError('coarse Slot export currently requires role_queries architecture')
    if len(scene_cube) != 1 or len(reverse_cube) != 1 or len(scene_world) != 1:
        raise ValueError('coarse Slot export currently supports a single current frame (B=1)')
    keys = ('points', 'confidence', 'valid', 'role_token_valid',
            'reference_null_probability', 'reference_is_null', 'geometry')
    if any('object_slot_' + key not in output for key in keys) or 'mvt1_ori_img' not in output:
        raise ValueError('coarse network output is missing required Slot/render fields')
    image = output['mvt1_ori_img']
    if image.ndim != 5 or image.shape[0] != 1 or image.shape[2] < 3:
        raise ValueError('coarse rendered image must have shape [1,V,C,H,W]')
    renderer = network.renderer
    if float(getattr(renderer, 'default_color', float('nan'))) != 0.:
        raise ValueError('coarse support export requires a zero-background renderer')
    # Same geometry/cameras/render settings as the policy, but a unit feature
    # carries visibility independently of the numerical value of rendered XYZ.
    points = scene_cube[0]
    support_image = renderer(points, torch.ones_like(points[:, :1]), fix_cam=True, dyn_cam_info=None)
    if (support_image.ndim != 4 or support_image.shape[:3]
            != (image.shape[1], image.shape[-2], image.shape[-1]) or support_image.shape[-1] < 1):
        raise ValueError('coarse support renderer returned incompatible image dimensions')
    point_valid, indices = coarse_point_support(
        output['object_slot_prior'], image[:, :, :3], output['object_slot_points'],
        output['object_slot_valid'], support_image[..., 0][None],
        network.object_slot_predictor1.point_samples)
    slot_output = {}
    for key in keys:
        value = output['object_slot_' + key]
        if value is None:
            slot_output[key] = None
        else:
            if value.shape[0] != 1:
                raise ValueError('coarse output batch is not B=1')
            slot_output[key] = _numpy(value[0])
    return dict(schema_version=1, artifact_type='bridgevla_coarse_slot_export',
                source_frame='bridgevla_coarse', slot_output=slot_output,
                point_valid=_numpy(point_valid[0]), support_pixel_indices=_numpy(indices[0]),
                point_support_source='renderer_coverage_and_predictor_topk',
                **cube_inverse_metadata(reverse_cube[0], points, scene_world[0]))
