"""Optional local role-map readout for SAM prompts; no model/action changes."""

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .coarse_slot_export import _numpy


@torch.no_grad()
def attach_refine_guidance(network, output, scene_cube, coarse_packet, *, min_probability=.25, min_points=4):
    """Select observed Refine XYZ per role, retaining Coarse semantics and inverse.

    Refine ``points``/``geometry`` can be inherited and are deliberately ignored.
    Local points are mapped back to the coarse cube, so the proven coarse/world
    inverse remains the single TASK transform. No role/NULL is reclassified.
    Unsupported local crops fall back per role, never fabricate a new object.
    """
    if network.training:
        raise ValueError('Refine Guidance requires frozen eval mode')
    if not 0 < min_probability <= 1 or type(min_points) is not int or min_points < 1:
        raise ValueError('invalid Refine support thresholds')
    packet = copy.deepcopy(coarse_packet)
    if 'visualization' in packet:
        packet['coarse_original_slot_output'] = copy.deepcopy(packet['slot_output'])
        packet['coarse_original_point_valid'] = packet['point_valid'].copy()
    slots = packet['slot_output']
    selection = dict(requested_stage='refine', min_probability=min_probability, min_points=min_points, roles={})
    packet['guidance_selection'] = selection

    def fallback_all(reason):
        for role in ('target', 'reference'):
            selection['roles'][role] = dict(selected_stage='coarse', failure_reason=reason)
        return packet

    refine = output.get('mvt2')
    if refine is None or 'mvt2_ori_img' not in output:
        return fallback_all('REFINE_UNAVAILABLE')
    if not bool(refine.get('object_slot_masks_refined', False)):
        return fallback_all('REFINE_MASK_HEAD_NOT_RUN')
    if not bool(refine.get('object_slot_roles_inherited', False)):
        return fallback_all('REFINE_ROLES_NOT_INHERITED')
    image = output['mvt2_ori_img']
    prior = refine['object_slot_prior']
    if (image.ndim != 5 or image.shape[0] != 1 or image.shape[2] < 3
            or prior.ndim != 5 or prior.shape[:3] != (1, image.shape[1], 2)):
        raise ValueError('Refine images/role maps must be [1,V,C,H,W] / [1,V,2,h,w]')
    if not torch.isfinite(prior).all() or (prior < 0).any() or (prior > 1).any():
        raise ValueError('Refine role probabilities must be finite in [0,1]')
    center, scale = output['wpt_local1'], float(network.st_sca)
    if center.shape != (1, 3) or not torch.isfinite(center).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError('invalid actual Refine crop transform')
    if len(scene_cube) != 1:
        raise ValueError('Refine Guidance supports B=1')
    local_scene = scale * (scene_cube[0] - center[0])
    # Check the actual closure returned by trans_pc, including nonzero crops.
    restored = output['rev_trans'](local_scene)
    if not torch.allclose(restored, scene_cube[0], rtol=2e-5, atol=2e-6):
        raise ValueError('actual Refine inverse does not restore the coarse scene')
    if float(getattr(network.renderer, 'default_color', float('nan'))) != 0.:
        raise ValueError('Refine support requires a zero-background renderer')
    coverage = network.renderer(local_scene, torch.ones_like(local_scene[:, :1]), fix_cam=True, dyn_cam_info=None)
    views, height, width = image.shape[1], image.shape[-2], image.shape[-1]
    if coverage.ndim != 4 or coverage.shape[:3] != (views, height, width) or coverage.shape[-1] < 1:
        raise ValueError('Refine coverage shape differs from its actual RGB/XYZ image')
    coverage = coverage[..., 0]
    if not torch.isfinite(coverage).all() or (coverage < -1e-6).any() or (coverage > 1 + 1e-6).any():
        raise ValueError('Refine coverage must be finite in [0,1]')
    xyz = image[0, :, :3].permute(0, 2, 3, 1).reshape(-1, 3)
    support = torch.isfinite(xyz).all(-1) & coverage.flatten().gt(1e-6)
    support &= ~(xyz.eq(0).all(-1) | xyz.eq(-1).all(-1))
    support &= xyz.abs().le(1. + 1e-5).all(-1)
    scores = F.interpolate(prior[0], size=(height, width), mode='bilinear', align_corners=False)
    scores = scores.permute(1, 0, 2, 3).flatten(1)
    # The readout may improve geometry, but cannot invent a semantic role or
    # undo the global NULL posterior. Both stages must retain the same identity.
    semantic = slots['role_token_valid'] & _numpy(refine['object_slot_role_token_valid'][0])
    local_valid = _numpy(refine['object_slot_valid'][0])
    count = slots['points'].shape[1]
    reverse_crop = np.eye(4)
    reverse_crop[:3, :3] /= scale
    reverse_crop[:3, 3] = _numpy(center[0])
    selection.update(crop_center_coarse=_numpy(center[0]).tolist(), crop_scale=scale,
                     matrix_world_from_refine_cube=(np.asarray(packet['matrix_world_from_source']) @ reverse_crop).tolist())
    for index, role in enumerate(('target', 'reference')):
        row = dict(selected_stage='coarse', failure_reason=None, supported_point_count=0)
        selection['roles'][role] = row
        if role == 'reference' and (bool(slots['reference_is_null']) or float(slots['reference_null_probability']) >= .75):
            row['failure_reason'] = 'REFERENCE_NULL'
            continue
        if not bool(semantic[index]):
            row['failure_reason'] = 'COARSE_SEMANTIC_INVALID'
            continue
        if not bool(local_valid[index]):
            row['failure_reason'] = 'REFINE_LOCAL_INVALID'
            continue
        eligible = support & scores[index].ge(min_probability)
        indices = eligible.nonzero().flatten()
        if not len(indices):
            row['failure_reason'] = 'REFINE_NO_SUPPORTED_RESPONSE'
            continue
        indices = indices[scores[index, indices].argsort(descending=True)]
        # Multiple views of a single observed XYZ do not establish spatial support.
        unique_count = len(torch.unique(xyz[indices], dim=0))
        row['supported_point_count'] = unique_count
        if unique_count < min_points or count < min_points:
            row['failure_reason'] = 'REFINE_INSUFFICIENT_POINTS'
            continue
        indices = indices[:count]
        if len(torch.unique(xyz[indices], dim=0)) < min_points:
            row['failure_reason'] = 'REFINE_INSUFFICIENT_SELECTED_POINTS'
            continue
        coarse_points = output['rev_trans'](xyz[indices])
        expected = xyz[indices] / scale + center[0]
        if not torch.allclose(coarse_points, expected, rtol=2e-5, atol=2e-6):
            raise ValueError('Refine point inverse differs from recorded crop transform')
        slots['points'][index] = 0.
        packet['point_valid'][index] = False
        slots['points'][index, :len(indices)] = _numpy(coarse_points)
        packet['point_valid'][index, :len(indices)] = True
        slots['valid'][index] = True
        slots['confidence'][index] = float(scores[index, indices].max())
        row.update(selected_stage='refine', selected_point_count=len(indices),
                   confidence=float(slots['confidence'][index]))
    # Geometry[17] is inherited global context; do not mislabel it as the new
    # local centers/spreads. TASK does not use it to generate the actual points.
    slots['geometry'] = None
    packet['point_support_source'] += '; refine_role_map_with_current_renderer_coverage'
    return packet
