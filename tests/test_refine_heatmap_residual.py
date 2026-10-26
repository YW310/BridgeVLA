"""Dense XYZ transport, zero-init residuals, gradients and legacy isolation."""
import pytest
import torch
import torch.nn.functional as F

from finetune.bridgevla.models.heatmap_transport import reproject_role_heatmaps
from finetune.bridgevla.models.refine_role_masks import (
    HEATMAP_FUSION_VERSION, RefineHeatmapTokenFusion,
    RoleConditionedRefineMaskHead, refine_role_mask_losses,
)


def test_opt_in_requires_direct_role_head_and_old_config_default_is_disabled():
    from finetune.bridgevla.config import get_cfg_defaults
    from test_role_feature_config import _model_guard_namespace
    assert not get_cfg_defaults().object_conditioning.refine_heatmap_residual
    with pytest.raises(ValueError, match='refine_heatmap_residual requires'):
        _model_guard_namespace(object_conditioning_refine_heatmap_residual=True)
    _model_guard_namespace(object_conditioning_refine_heatmap_residual=True,
                           object_conditioning_refine_role_masks=True,
                           object_conditioning_inherit_coarse_roles=True,
                           object_conditioning_preserve_role_tokens=True,
                           object_slot_predictor_type='role_queries', object_slot_num_slots=2)


def test_old_head_weights_initialize_new_branches_with_zero_output():
    old = RoleConditionedRefineMaskHead(4, 8, hidden_dim=8, local_query=True)
    new = RoleConditionedRefineMaskHead(4, 8, hidden_dim=8, local_query=True, heatmap_residual=True)
    incompatible = new.load_state_dict(old.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert set(incompatible.missing_keys) == {
        key for key in new.state_dict() if key.startswith(('heatmap_tokens.', 'residual_output.'))}
    assert not hasattr(new, 'prior_half')
    assert not new.residual_output.weight.any() and not new.residual_output.bias.any()
    torch.testing.assert_close(new.detail_half[0].weight, old.detail_half[0].weight)


def test_new_routing_is_checkpointed_and_old_resume_and_eval_are_rejected():
    from types import SimpleNamespace
    from test_role_feature_config import _model, _train_namespace, _isolated_functions
    model = _model(refine_role_masks=True, refine_local_query=True, refine_heatmap_residual=True)
    metadata = dict(refine_role_masks=True, refine_local_query=True)
    namespace, saved = _train_namespace(dict(epoch=0, model_state={}, object_conditioning=metadata))
    with pytest.raises(RuntimeError, match='--init_checkpoint'):
        namespace['load_training_checkpoint'](SimpleNamespace(_network=model), 'old.pth')
    namespace['save_agent'](SimpleNamespace(_network=model), 'new.pth', 0)
    assert saved[0][0]['object_conditioning']['refine_heatmap_residual'] is True
    assert saved[0][0]['object_conditioning']['refine_heatmap_fusion'] == HEATMAP_FUSION_VERSION
    validate = _isolated_functions('finetune/RLBench/eval.py',
                                   {'_validate_role_feature_checkpoint'}, {})['_validate_role_feature_checkpoint']
    cfg = SimpleNamespace(refine_role_masks=True, refine_local_query=True, refine_heatmap_residual=True)
    with pytest.raises(RuntimeError, match='refine_heatmap_residual'):
        validate({'object_conditioning': metadata}, cfg, 'old.pth')
    validate(saved[0][0], cfg, 'new.pth')
    # The earlier CNN prototype has the boolean but a different architecture.
    cnn = dict(metadata, refine_heatmap_residual=True)
    with pytest.raises(RuntimeError, match='fusion differs'):
        validate({'object_conditioning': cnn}, cfg, 'cnn.pth')
    namespace, _ = _train_namespace(dict(epoch=0, model_state={}, object_conditioning=cnn))
    with pytest.raises(RuntimeError, match='--init_checkpoint'):
        namespace['load_training_checkpoint'](SimpleNamespace(_network=model), 'cnn.pth')


def scene(views=1):
    axis = torch.linspace(-.875, .875, 8)
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    xyz = torch.stack((x, y, torch.full_like(x, .3)))[None, None].repeat(1, views, 1, 1, 1)
    maps = torch.full((1, views, 2, 8, 8), .02)
    maps[:, :, 0, 1, 2] = .8
    maps[:, :, 1, 5, 6] = .15
    return maps, xyz


def project(views=1):
    return lambda points: ((points[..., :2] + 1) * 4)[:, :, None].expand(-1, -1, views, -1)


def warp(maps, source, local=None, center=None, scale=1, **kwargs):
    return reproject_role_heatmaps(
        maps, source, source if local is None else local,
        torch.zeros(1, 3) if center is None else center, scale,
        project(source.shape[1]), torch.ones(1, 2).bool(), **kwargs)


def test_identity_preserves_full_amplitude_low_scores_and_chunk_equivalence():
    maps, xyz = scene()
    maps.requires_grad_()
    result = warp(maps, xyz, chunk_size=7)
    torch.testing.assert_close(result['probabilities'], maps.detach())
    assert result['valid'].all()
    assert not result['probabilities'].requires_grad
    torch.testing.assert_close(result['probabilities'], warp(maps, xyz, chunk_size=256)['probabilities'])
    assert result['probabilities'][0, 0, 1].max() == .15  # No per-role peak normalization.


def test_inverse_crop_uses_xyz_instead_of_resizing_the_heatmap():
    maps, xyz = scene()
    center = torch.tensor([[0., 0., .25]])
    local = (xyz[:, :, :, 2:6, 2:6] - center[:, None, :, None, None]) * 2
    result = warp(maps, xyz, local, center, scale=2)
    torch.testing.assert_close(result['probabilities'], maps[:, :, :, 2:6, 2:6])
    assert result['valid'].all()


def test_only_matching_visible_views_contribute_to_probability_fusion():
    maps, xyz = scene(2)
    maps[:, 0] = .2
    maps[:, 1] = .6
    local = xyz[:, :1].clone()
    fused = warp(maps, xyz, local)
    torch.testing.assert_close(fused['probabilities'], torch.full((1, 1, 2, 8, 8), .4))
    xyz[:, 1, 2] = .9  # Other surface at the same coarse pixel.
    visible = warp(maps, xyz, local)
    torch.testing.assert_close(visible['probabilities'], torch.full((1, 1, 2, 8, 8), .2))


def test_known_zero_differs_from_unknown_and_bad_surface_never_becomes_background_gt():
    maps, xyz = scene()
    zero = warp(torch.zeros_like(maps), xyz)
    assert zero['valid'].all() and not zero['probabilities'].any()
    unknown = warp(maps, torch.zeros_like(xyz), xyz)
    assert not unknown['valid'].any() and not unknown['probabilities'].any()
    missing = xyz.clone()
    missing[:, :, :, :4] = float('nan')
    result = warp(maps, xyz, missing)
    assert not result['valid'][:, :, :, :4].any()
    assert torch.isfinite(result['probabilities']).all()


def test_outside_crop_and_fractional_out_of_frame_samples_stay_unknown():
    maps, xyz = scene()
    outside = xyz.clone()
    outside[:, :, 2] = 1.01
    assert not warp(maps, xyz, outside)['valid'].any()
    result = reproject_role_heatmaps(maps, xyz, xyz, torch.zeros(1, 3), 1.,
                                    lambda p: p.new_full((1, p.shape[1], 1, 2), -.01),
                                    torch.ones(1, 2).bool())
    assert not result['valid'].any()


def test_role_validity_is_separate_from_projected_surface_validity():
    maps, xyz = scene()
    result = reproject_role_heatmaps(maps, xyz, xyz, torch.zeros(1, 3), 1., project(),
                                    torch.tensor([[True, False]]))
    assert result['valid'][:, :, 0].all()
    assert not result['valid'][:, :, 1].any()
    assert not result['probabilities'][:, :, 1].any()


def test_low_resolution_coarse_maps_are_sampled_at_the_rendered_xyz_pixels():
    maps, xyz = scene()
    low = maps[..., ::4, ::4]
    low[..., 0, 0] = .8
    expected = F.interpolate(low.reshape(1, 2, 2, 2), size=(8, 8), mode='bilinear', align_corners=False)
    torch.testing.assert_close(warp(low, xyz)['probabilities'], expected[:, None])


def test_tokens_keep_patch_coordinates_role_channels_and_known_zero_separate():
    maps, xyz = scene(2)
    maps[:, :, 0] = .02
    maps[:, :, 1] = 0.
    known = torch.ones_like(maps).bool()
    known[:, :, 0, :4, :4] = False
    # Unknown values must not dilute the average over the known half-patch.
    known[:, :, 0, :2, 4:] = False
    inputs, available, depth = RefineHeatmapTokenFusion.patch_inputs(
        maps, known, xyz, torch.ones(1, 2, 8, 8).bool(), (2, 2))
    assert inputs.shape == (2, 4, 10)
    torch.testing.assert_close(inputs[0, :, 0], torch.tensor([0., .02, .02, .02]))
    torch.testing.assert_close(inputs[0, :, 1], torch.zeros(4))
    torch.testing.assert_close(inputs[0, :, 2], torch.tensor([0., .5, 1., 1.]))
    torch.testing.assert_close(inputs[..., 3], torch.ones(2, 4))  # Known Reference zero.
    torch.testing.assert_close(inputs[0, :, 4:7],
                               xyz[0, 0].reshape(3, 2, 4, 2, 4).mean((2, 4)).flatten(1).T)
    torch.testing.assert_close(inputs[0, :, 7:9],
                               torch.tensor([[-.5, -.5], [.5, -.5], [-.5, .5], [.5, .5]]))
    torch.testing.assert_close(inputs[:, :, 9], torch.tensor([[-.5] * 4, [.5] * 4]))
    assert available.all() and depth.all()


def test_spatial_tokens_do_not_mix_views_or_backpropagate_into_guides():
    torch.manual_seed(12)
    fusion = RefineHeatmapTokenFusion(8)
    maps, xyz = scene(2)
    maps.requires_grad_()
    xyz.requires_grad_()
    images = torch.randn(2, 8, 2, 2, requires_grad=True)
    valid = torch.ones_like(maps).bool()
    support = torch.ones(1, 2, 8, 8).bool()
    output = fusion(images, maps, valid, xyz, support)
    changed = maps.detach().clone()
    changed[:, 0, 0] = .9
    other = fusion(images, changed, valid, xyz, support)
    torch.testing.assert_close(output[1], other[1], rtol=0, atol=0)
    assert not torch.allclose(output[0], other[0])
    output.square().sum().backward()
    assert images.grad.abs().sum() > 0
    assert fusion.encode[0].weight.grad.abs().sum() > 0
    assert maps.grad is None and xyz.grad is None


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16, torch.float16])
def test_all_unknown_view_is_exact_identity_and_invalid_depth_gets_no_update(dtype):
    fusion = RefineHeatmapTokenFusion(8).to(dtype)
    maps, xyz = scene(2)
    valid = torch.ones_like(maps).bool()
    valid[:, 0] = False
    support = torch.ones(1, 2, 8, 8).bool()
    support[:, 1, :4, :4] = False
    xyz = torch.where(support[:, :, None], xyz, 0.)
    images = torch.randn(2, 8, 2, 2).to(dtype)
    result = fusion(images, maps, valid & support[:, :, None], xyz, support)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[0], images[0], rtol=0, atol=0)
    torch.testing.assert_close(result[1, :, 0, 0], images[1, :, 0, 0], rtol=0, atol=0)


def inputs():
    from test_refine_local_query import RefineLocalQueryTest
    _, features, rgb, xyz, packet = RefineLocalQueryTest()._inputs()
    head = RoleConditionedRefineMaskHead(4, 8, hidden_dim=8, local_query=True, heatmap_residual=True)
    prior = torch.full((1, 3, 2, 16, 16), .2)
    prior[:, :, 1] = .7
    packet.update(projected_role_probabilities=prior.requires_grad_(),
                  projected_role_valid=torch.ones_like(prior).bool())
    return head, features, rgb, xyz, packet


def test_zero_residual_inherits_projected_probabilities_and_null_mass_exactly_once():
    head, features, rgb, xyz, packet = inputs()
    result = head(features, rgb, xyz, packet)
    expected = packet['projected_role_probabilities'].detach().clone()
    expected[:, :, 1] *= .8
    torch.testing.assert_close(result['prior'], expected)
    torch.testing.assert_close(result['refine_projected_prior'], expected)
    assert result['refine_residual_logits'].count_nonzero() == 0
    assert head.detail_half[0].in_channels == 7
    assert isinstance(head.heatmap_tokens, RefineHeatmapTokenFusion)
    packet['reference_null_probability'] = torch.tensor([.6], requires_grad=True)
    changed = head(features, rgb, xyz, packet)
    torch.testing.assert_close(changed['prior'][:, :, 1], result['prior'][:, :, 1] * .5)
    changed['prior'].sum().backward()
    assert packet['reference_null_probability'].grad < 0
    assert packet['projected_role_probabilities'].grad is None


def test_learned_residual_consumes_aligned_heatmap_tokens_not_only_prior_skip():
    head, features, rgb, xyz, packet = inputs()
    with torch.no_grad():
        head.residual_output.weight.normal_(std=.1)
    first = head(features, rgb, xyz, packet)
    changed = dict(packet)
    changed['projected_role_probabilities'] = packet['projected_role_probabilities'].detach().flip(2)
    second = head(features, rgb, xyz, changed)
    assert not torch.allclose(first['refine_residual_logits'], second['refine_residual_logits'])


def test_unknown_projection_is_near_dark_but_gt_can_train_local_recovery():
    head, features, rgb, xyz, packet = inputs()
    packet['projected_role_valid'].zero_()
    result = head(features, rgb, xyz, packet, detach_action=True)
    assert result['prior'].max() < .001
    assert not result['prior'].requires_grad
    assert not result['refine_projection_valid'].any()
    teacher = torch.zeros_like(result['refine_mask_logits'])
    teacher[:, :, 0, 5, 5] = .9
    loss = refine_role_mask_losses(result['refine_mask_logits'], teacher,
                                   torch.ones(1, 2).bool(), result['refine_pixel_valid'])['mask']
    loss.backward()
    assert head.residual_output.weight.grad.abs().sum() > 0
    assert head.residual_output.bias.grad[0] < 0


def test_empty_depth_cannot_acquire_roles_and_nonfinite_prior_remains_unknown():
    head, features, rgb, xyz, packet = inputs()
    packet['projected_role_probabilities'] = torch.full_like(packet['prior'], float('nan'))
    result = head(features, rgb, xyz, packet)
    assert not result['refine_projection_valid'].any()
    assert torch.isfinite(result['prior']).all() and result['prior'].max() < .001
    result = head(features, rgb, torch.zeros_like(xyz), packet)
    assert not result['prior'].any() and not result['valid'].any()


def test_residual_output_then_prior_fusion_learn_without_updating_detached_guide():
    head, features, rgb, xyz, packet = inputs()
    optimizer = torch.optim.SGD(head.parameters(), lr=.01)
    for _ in range(2):
        optimizer.zero_grad()
        result = head(features, rgb, xyz, packet)
        loss = refine_role_mask_losses(result['refine_mask_logits'],
                                       torch.full_like(result['refine_mask_logits'], .9),
                                       torch.ones(1, 2).bool(), result['refine_pixel_valid'])['mask']
        loss.backward()
        optimizer.step()
    assert head.heatmap_tokens.encode[0].weight.grad.abs().sum() > 0
    assert head.heatmap_tokens.query.weight.grad.abs().sum() > 0
    assert packet['projected_role_probabilities'].grad is None
    assert torch.isfinite(loss)


def test_real_policy_uses_projected_prior_and_gt_is_only_a_loss_target():
    from unittest import mock
    from test_refine_role_masks import RefineRoleMaskTest
    from bridgevla.mvt import mvt_single
    from bridgevla.models.oracle_prior import build_training_visualization_payload
    helper, module, _, options, _ = RefineRoleMaskTest()._policy_inputs()
    options['refine_role_mask_head'] = RoleConditionedRefineMaskHead(
        4, 8, hidden_dim=8, local_query=True, heatmap_residual=True)
    packet = options['inherited_object_roles']
    packet['prior'] = torch.zeros(1, 3, 2, 16, 16)
    packet['projected_role_probabilities'] = torch.full((1, 3, 2, 16, 16), .3)
    packet['projected_role_valid'] = torch.ones(1, 3, 2, 16, 16).bool()
    module.eval()
    with mock.patch.object(mvt_single, 'select_feat_from_hm', helper._sample, create=True):
        out = module(**options)
        changed = module(**options, object_slot_target_heatmap=torch.ones(1, 3, 2, 16, 16))
    for key in ('trans', 'feat_ex_rot', 'object_slot_prior'):
        torch.testing.assert_close(out[key], changed[key])
    assert out['object_refine_residual_logits'].count_nonzero() == 0
    torch.testing.assert_close(out['object_slot_prior'], out['object_refine_projected_prior'])
    payload = build_training_visualization_payload(dict(out, mvt1_ori_img=options['img']),
                                                  torch.zeros(1, 256, 3), num_views=3,
                                                  height=16, width=16, stage_two=False)['mvt1']
    torch.testing.assert_close(payload['slot_reference_pred'], payload['reference_prior'])
    assert payload['reference_projection_valid'].all()
    from bridgevla.models.inference_visualization import internal_slot_stage_diagnostics
    diagnostics = internal_slot_stage_diagnostics(out)
    assert diagnostics['mask_source'] == 'coarse_heatmap_residual'
    assert diagnostics['projection_coverage'] == 1.
    assert diagnostics['residual_abs_mean'] == 0.
