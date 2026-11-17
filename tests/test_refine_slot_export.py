"""Refine Guidance selects current observed XYZ, never stale inherited points."""

import copy

import numpy as np
import pytest
import torch

from test_coarse_slot_export import sample, callback_block
from bridgevla.models.coarse_slot_export import build_coarse_slot_packet


def refine_sample(with_mean=False):
    network, output, cube, reverse, world = sample(with_mean)
    center = torch.tensor([[.2, -.1, .3]])
    scale = 4.
    local = torch.tensor([[-.4, -.3, .1], [-.3, -.3, .1], [-.4, -.2, .1], [-.3, -.2, .1],
                          [.3, .2, .1], [.4, .2, .1], [.3, .3, .1], [.4, .3, .1]])
    cube = [torch.cat((cube[0], local / scale + center))]
    world = [reverse[0](cube[0])]
    network.st_sca = scale
    network.renderer.coverage[:] = 0.
    network.renderer.coverage[:, :2] = 1.
    xyz = torch.zeros(1, 1, 3, 3, 4)
    xyz[0, 0, :, :2] = local.reshape(2, 4, 3).permute(2, 0, 1)
    prior = torch.zeros(1, 1, 2, 3, 4)
    prior[:, :, 0, 0] = .8
    prior[:, :, 1, 1] = .7
    output['mvt2_ori_img'] = torch.cat((xyz, torch.ones_like(xyz) * .4), dim=2)
    output['mvt2'] = dict(object_slot_prior=prior.requires_grad_(),
                          object_slot_masks_refined=True, object_slot_roles_inherited=True,
                          object_slot_confidence=torch.tensor([[.8, .7]]),
                          object_slot_reference_null_probability=torch.tensor([.1]),
                          object_slot_valid=torch.ones(1, 2, dtype=torch.bool),
                          object_slot_role_token_valid=torch.ones(1, 2, dtype=torch.bool),
                          # Poison the inherited geometry; local readout must ignore it.
                          object_slot_points=torch.full((1, 2, 5, 3), 999.))
    output['wpt_local1'] = center
    output['rev_trans'] = lambda p: p / scale + center
    return network, output, cube, reverse, world, local


@pytest.mark.parametrize('with_mean', [False, True])
def test_refine_corrects_points_with_proven_nonzero_crop_world_inverse(with_mean):
    network, output, cube, reverse, world, local = refine_sample(with_mean)
    coarse = build_coarse_slot_packet(network, output, cube, reverse, world)
    result = build_coarse_slot_packet(network, output, cube, reverse, world, guidance_stage='refine')
    assert network.renderer.calls == 3  # Coarse control, then Coarse+Refine coverage.
    assert result['slot_output']['geometry'] is None
    for index, role in enumerate(('target', 'reference')):
        row = result['guidance_selection']['roles'][role]
        assert row['selected_stage'] == 'refine' and row['failure_reason'] is None
        mask = result['point_valid'][index]
        assert mask.sum() == 4 and row['supported_point_count'] == 4
        observed = result['slot_output']['points'][index, mask]
        expected = local[index * 4:(index + 1) * 4].numpy() / 4 + np.array([.2, -.1, .3])
        np.testing.assert_allclose(np.sort(observed, axis=0), np.sort(expected, axis=0), atol=1e-7)
        assert not np.array_equal(observed, coarse['slot_output']['points'][index, mask])
        world_actual = np.c_[observed, np.ones(4)] @ result['matrix_world_from_source'].T
        matrix = np.asarray(result['guidance_selection']['matrix_world_from_refine_cube'])
        world_expected = np.c_[local[index * 4:(index + 1) * 4].numpy(), np.ones(4)] @ matrix.T
        np.testing.assert_allclose(np.sort(world_actual, axis=0), np.sort(world_expected, axis=0), atol=1e-7)
    result['slot_output']['points'][:] = 123.
    assert output['mvt2']['object_slot_points'].max() == 999.
    assert output['mvt2']['object_slot_prior'].requires_grad
    assert isinstance(result['point_valid'], np.ndarray)


@pytest.mark.parametrize('missing,reason', [('mvt2', 'REFINE_UNAVAILABLE'),
    ('head', 'REFINE_MASK_HEAD_NOT_RUN'), ('identity', 'REFINE_ROLES_NOT_INHERITED')])
def test_no_refine_head_is_an_explicit_coarse_fallback_without_extra_render(missing, reason):
    network, output, cube, reverse, world, _ = refine_sample()
    if missing == 'mvt2':
        del output['mvt2']
    else:
        key = 'object_slot_masks_refined' if missing == 'head' else 'object_slot_roles_inherited'
        output['mvt2'][key] = False
    result = build_coarse_slot_packet(network, output, cube, reverse, world, guidance_stage='refine')
    assert network.renderer.calls == 1
    assert all(row['selected_stage'] == 'coarse' and row['failure_reason'] == reason
               for row in result['guidance_selection']['roles'].values())


@pytest.mark.parametrize('failure,reason', [
    ('weak', 'REFINE_NO_SUPPORTED_RESPONSE'), ('few', 'REFINE_INSUFFICIENT_POINTS'),
    ('duplicate', 'REFINE_INSUFFICIENT_POINTS'), ('coverage', 'REFINE_NO_SUPPORTED_RESPONSE'),
    ('outside', 'REFINE_NO_SUPPORTED_RESPONSE'), ('depth', 'REFINE_NO_SUPPORTED_RESPONSE'),
    ('local_invalid', 'REFINE_LOCAL_INVALID'), ('semantic', 'COARSE_SEMANTIC_INVALID')])
def test_target_failure_falls_back_per_role_without_suppressing_good_reference(failure, reason):
    network, output, cube, reverse, world, _ = refine_sample()
    xyz, refine = output['mvt2_ori_img'], output['mvt2']
    if failure == 'weak':
        refine['object_slot_prior'] = refine['object_slot_prior'].detach().clone()
        refine['object_slot_prior'][:, :, 0] = .1
    elif failure == 'few':
        xyz[:, :, :3, 0, 2:] = 0.
    elif failure == 'duplicate':
        xyz[:, :, :3, 0] = xyz[:, :, :3, 0, :1].clone()
    elif failure == 'coverage':
        network.renderer.coverage[:, 0] = 0.
    elif failure == 'outside':
        xyz[:, :, :3, 0] = 2.
    elif failure == 'depth':
        xyz[:, :, :3, 0] = float('nan')
    elif failure == 'local_invalid':
        refine['object_slot_valid'][0, 0] = False
    else:
        refine['object_slot_role_token_valid'][0, 0] = False
    coarse = build_coarse_slot_packet(network, output, cube, reverse, world)
    result = build_coarse_slot_packet(network, output, cube, reverse, world, guidance_stage='refine')
    assert result['guidance_selection']['roles']['target']['failure_reason'] == reason
    assert result['guidance_selection']['roles']['reference']['selected_stage'] == 'refine'
    np.testing.assert_array_equal(result['slot_output']['points'][0], coarse['slot_output']['points'][0])
    np.testing.assert_array_equal(result['point_valid'][0], coarse['point_valid'][0])


def test_reference_null_is_not_overridden_by_bright_local_map():
    network, output, cube, reverse, world, _ = refine_sample()
    output['object_slot_reference_null_probability'] = torch.tensor([.99])
    output['object_slot_reference_is_null'] = torch.tensor([True])
    result = build_coarse_slot_packet(network, output, cube, reverse, world, guidance_stage='refine')
    assert result['guidance_selection']['roles']['reference']['failure_reason'] == 'REFERENCE_NULL'
    assert result['slot_output']['reference_null_probability'] > .98
    assert result['guidance_selection']['roles']['target']['selected_stage'] == 'refine'


def test_bad_actual_inverse_and_nonfinite_probabilities_fail_instead_of_faking_geometry():
    network, output, cube, reverse, world, _ = refine_sample()
    output['rev_trans'] = lambda p: p / 2.
    with pytest.raises(ValueError, match='restore'):
        build_coarse_slot_packet(network, output, cube, reverse, world, guidance_stage='refine')
    network, output, cube, reverse, world, _ = refine_sample()
    output['mvt2']['object_slot_prior'] = torch.full((1, 1, 2, 3, 4), float('nan'))
    with pytest.raises(ValueError, match='probabilities'):
        build_coarse_slot_packet(network, output, cube, reverse, world, guidance_stage='refine')


def test_actual_agent_callback_uses_refine_only_when_requested():
    from types import SimpleNamespace
    network, output, cube, reverse, world, _ = refine_sample()
    captured = []
    def callback(packet):
        captured.append(packet)
    callback.guidance_stage = 'refine'
    scope = dict(self=SimpleNamespace(_net_mod=network), out=output, pc=cube,
                 rev_trans=reverse, coarse_scene_world=world, coarse_slot_callback=callback)
    exec(callback_block(), scope)
    assert captured[0]['guidance_selection']['roles']['target']['selected_stage'] == 'refine'


def test_refine_diagnostics_keep_original_coarse_support_and_points_for_comparison():
    network, output, cube, reverse, world, _ = refine_sample()
    original = build_coarse_slot_packet(network, output, cube, reverse, world)
    refined = build_coarse_slot_packet(network, output, cube, reverse, world,
                                       guidance_stage='refine', include_visualization=True)
    np.testing.assert_array_equal(refined['coarse_original_point_valid'], original['point_valid'])
    np.testing.assert_array_equal(refined['coarse_original_slot_output']['points'], original['slot_output']['points'])
    assert not np.array_equal(refined['slot_output']['points'], original['slot_output']['points'])
    assert refined['visualization']['refine']['masks_refined']
