"""T/R collapse regression with trusted surfaces, visibility and crop boundaries."""
from types import MethodType, SimpleNamespace

import pytest
import torch

from finetune.bridgevla.models.role_separation import role_separation_loss, visible_role_support


def maps():
    teacher = torch.zeros(1, 1, 2, 8, 8)
    teacher[0, 0, 0, 1, 1] = 1
    teacher[0, 0, 1, 6, 6] = 1
    visible = teacher.bool()
    valid = torch.ones(1, 2, dtype=torch.bool)
    return teacher, visible, valid


def test_union_collapse_has_larger_loss_and_correct_cross_role_gradients():
    teacher, visible, valid = maps()
    clean = torch.full_like(teacher, -5.)
    clean[teacher.bool()] = 5.
    collapsed = clean.clone()
    collapsed[0, 0, 0, 6, 6] = 5
    collapsed[0, 0, 1, 1, 1] = 5
    collapsed.requires_grad_()
    good = role_separation_loss(clean.sigmoid(), teacher, valid, visible)
    bad = role_separation_loss(collapsed.sigmoid(), teacher, valid, visible)
    assert bad['loss'] > good['loss'] + 4
    assert bad['supervised_pixels'] == 2
    assert bad['visibility_available'] == 1
    assert bad['teacher_role_views'] == bad['visible_role_views'] == 2
    assert bad['paired_views'] == 1
    assert bad['overlap_pixels'] == 0
    bad['loss'].backward()
    assert collapsed.grad[0, 0, 0, 6, 6] > 0
    assert collapsed.grad[0, 0, 1, 1, 1] > 0
    assert collapsed.grad.count_nonzero() == 2
    assert teacher.grad is None


@pytest.mark.parametrize('case', ['missing_visibility', 'invalid_role',
                                 'unknown', 'absent', 'overlap', 'no_depth', 'empty'])
def test_untrusted_or_unavailable_pairs_create_no_negative_gradient(case):
    teacher, visible, valid = maps()
    present = known = torch.ones(1, 2, dtype=torch.bool)
    pixels = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    if case == 'missing_visibility': visible = None
    if case == 'invalid_role': valid[:, 1] = False
    if case == 'unknown': known = torch.tensor([[True, False]])
    if case == 'absent': present = torch.tensor([[True, False]])
    if case == 'overlap': teacher[:] = teacher.amax(dim=2, keepdim=True)
    if case == 'no_depth': pixels.zero_()
    if case == 'empty': teacher.zero_()
    logits = torch.zeros_like(teacher, requires_grad=True)
    result = role_separation_loss(logits.sigmoid(), teacher, valid, visible, present, known, pixels)
    assert result['loss'] == result['supervised_pixels'] == 0
    result['loss'].backward()
    assert logits.grad.count_nonzero() == 0


def test_diagnostics_distinguish_missing_labels_visibility_crop_and_overlap():
    teacher, visible, valid = maps()
    probabilities = torch.full_like(teacher, .9, requires_grad=True)
    missing = role_separation_loss(probabilities, teacher, valid)
    assert missing['visibility_available'] == 0
    no_label = role_separation_loss(probabilities, torch.zeros_like(teacher), valid, visible)
    assert no_label['visibility_available'] == 1 and no_label['teacher_role_views'] == 0
    occluded = role_separation_loss(probabilities, teacher, valid, torch.zeros_like(visible))
    assert occluded['teacher_role_views'] == 2 and occluded['visible_role_views'] == 0
    single = visible.clone()
    single[:, :, 1] = False
    cropped = role_separation_loss(probabilities, teacher, valid, single)
    assert cropped['visible_role_views'] == 1 and cropped['paired_views'] == 0
    assert cropped['supervised_pixels'] == 1 and cropped['loss'] > 0
    overlap = teacher.amax(2, keepdim=True).expand_as(teacher).clone()
    ambiguous = role_separation_loss(probabilities, overlap, valid, visible)
    assert ambiguous['paired_views'] == 1 and ambiguous['overlap_pixels'] == 2
    assert ambiguous['loss'] == ambiguous['supervised_pixels'] == 0
    for result in (missing, no_label, occluded, cropped, ambiguous):
        assert not any(value.requires_grad for key, value in result.items() if key != 'loss')


def test_pooling_preserves_overlap_instead_of_inventing_exclusive_coarse_cells():
    teacher = torch.zeros(1, 1, 2, 8, 8)
    teacher[0, 0, 0, 0, 0] = teacher[0, 0, 1, 1, 1] = 1
    logits = torch.zeros(1, 1, 2, 2, 2, requires_grad=True)
    result = role_separation_loss(logits.sigmoid(), teacher, torch.ones(1, 2).bool(), teacher.bool())
    assert result['loss'] == result['supervised_pixels'] == 0


def test_nan_teacher_does_not_invent_cross_role_negatives():
    teacher, visible, valid = maps()
    teacher[0, 0, 0, 6, 6] = float('nan')
    teacher[0, 0, 1, 1, 1] = float('nan')
    result = role_separation_loss(torch.full_like(teacher, .9), teacher, valid, visible)
    assert result['loss'] == 0


def test_saturated_wrong_logits_keep_nonzero_corrective_gradients():
    teacher, visible, valid = maps()
    logits = torch.full_like(teacher, 100., requires_grad=True)
    result = role_separation_loss(logits.sigmoid(), teacher, valid, visible, logits=logits)
    assert result['loss'] == 100
    result['loss'].backward()
    assert logits.grad[0, 0, 0, 6, 6] == .5
    assert logits.grad[0, 0, 1, 1, 1] == .5


def test_one_visible_role_still_supervises_the_other_known_role():
    teacher, visible, valid = maps()
    teacher, visible = teacher.repeat(1, 2, 1, 1, 1), visible.repeat(1, 2, 1, 1, 1)
    visible[:, 1, 1] = False
    logits = torch.zeros_like(teacher, requires_grad=True)
    result = role_separation_loss(logits.sigmoid(), teacher, valid, visible, logits=logits)
    assert result['supervised_pixels'] == 3
    result['loss'].backward()
    assert logits.grad[0, 1, 1, 1, 1] > 0  # Reference must not copy visible Target.
    assert logits.grad[0, 1, 0].count_nonzero() == 0  # No visible Reference source.


def test_crop_external_reference_gets_negative_on_target_with_global_assignment():
    teacher, visible, assigned = maps()
    teacher[:, :, 1] = 0  # No Reference projection in this crop.
    visible[:, :, 1] = False
    stage_valid = torch.tensor([[True, False]])
    logits = torch.full_like(teacher, 5., requires_grad=True)
    result = role_separation_loss(
        logits.sigmoid(), teacher, stage_valid, visible,
        role_present=assigned, role_present_known=assigned,
        logits=logits, assignment_valid=assigned)
    assert result['paired_views'] == 0
    assert result['supervised_pixels'] == 1 and result['loss'] > 4
    result['loss'].backward()
    assert logits.grad[0, 0, 1, 1, 1] > 0
    assert logits.grad.count_nonzero() == 1  # No fabricated full-crop absence mask.


def test_actual_agent_does_not_erase_reference_assignment_when_stage_teacher_is_empty():
    from test_direct_role_losses import _actual_agent_loss
    from finetune.bridgevla.models.refine_role_masks import refine_role_mask_losses
    teacher, visible, assigned = maps()
    raw, local = torch.zeros_like(teacher, requires_grad=True), torch.zeros_like(teacher, requires_grad=True)
    output = dict(object_slot_predictor_type='role_queries', object_slot_mask_logits=raw,
                  object_slot_prior=raw.sigmoid(), object_slot_target_prior=teacher.clone(),
                  object_slot_reference_null_probability=torch.tensor([.1]),
                  object_role_visible_support=visible.clone(), object_slot_target_valid=assigned)
    teacher[:, :, 1] = 0
    visible[:, :, 1] = False
    output['mvt2'] = dict(object_slot_roles_inherited=True, object_slot_masks_refined=True,
                         object_slot_prior=local.sigmoid(), object_refine_mask_logits=local,
                         object_slot_target_prior=teacher, object_role_visible_support=visible,
                         object_slot_target_valid=torch.tensor([[True, False]]),
                         object_refine_pixel_valid=torch.ones(1, 1, 8, 8).bool())
    method = _actual_agent_loss()
    method.__globals__['refine_role_mask_losses'] = refine_role_mask_losses
    agent = SimpleNamespace(internal_object_slots_enabled=True, stage_two=True,
                            _net_mod=SimpleNamespace(object_conditioning_supervise_mixed_role_maps=False))
    result = MethodType(method, agent)(output, assigned, assigned, assigned)
    assert result['refine_diversity_paired_views'] == 0
    assert result['refine_diversity_pixels'] == 1 and result['refine_diversity'] > 0
    result['refine_diversity'].backward()
    assert local.grad[0, 0, 1, 1, 1] > 0
    assert local.grad.count_nonzero() == 1


def projection(points):
    return ((points[..., :2] + 1) * 4).unsqueeze(2)


def test_surface_visibility_rejects_occluders_empty_pixels_outside_crop_and_fractional_bounds():
    points = torch.tensor([[[[.25, .25, .5]], [[-.5, -.5, .25]]]], requires_grad=True)
    xyz = torch.zeros(1, 1, 3, 8, 8)
    xyz[0, 0, :, 5, 5] = points.detach()[0, 0, 0]
    xyz[0, 0, :, 2, 2] = torch.tensor([-.5, -.5, -.5])  # Different front surface.
    support = visible_role_support(points, torch.ones(1, 2).bool(), xyz, projection)
    assert support[0, 0, 0, 5, 5] and support.sum() == 1
    assert not support.requires_grad
    xyz[0, 0, :, 2, 2] = points.detach()[0, 1, 0]
    assert visible_role_support(points, torch.ones(1, 2).bool(), xyz, projection).sum() == 2
    outside = points.detach().clone()
    outside[0, 1, 0, 2] = 1.1
    assert visible_role_support(outside, torch.ones(1, 2).bool(), xyz, projection).sum() == 1
    bad_project = lambda p: p.new_full((1, 2, 1, 2), -.01)
    assert not visible_role_support(points, torch.ones(1, 2).bool(), xyz, bad_project).any()
    assert not visible_role_support(points, torch.ones(1, 2).bool(), torch.zeros_like(xyz), projection).any()


def test_agent_computes_coarse_and_refine_separation_with_live_raw_mask_gradients():
    from test_direct_role_losses import _actual_agent_loss
    from finetune.bridgevla.models.refine_role_masks import refine_role_mask_losses
    teacher, visible, valid = maps()
    raw = torch.zeros_like(teacher, requires_grad=True)
    local = torch.zeros_like(teacher, requires_grad=True)
    null = torch.tensor([.9], requires_grad=True)
    output = dict(object_slot_predictor_type='role_queries', object_slot_mask_logits=raw,
                  object_slot_prior=raw.sigmoid(), object_slot_target_prior=teacher,
                  object_slot_reference_null_probability=null, object_role_visible_support=visible)
    output['mvt2'] = dict(object_slot_roles_inherited=True, object_slot_masks_refined=True,
                         object_slot_prior=local.sigmoid(), object_refine_mask_logits=local,
                         object_slot_target_prior=teacher, object_role_visible_support=visible,
                         object_refine_pixel_valid=torch.ones(1, 1, 8, 8).bool())
    method = _actual_agent_loss()
    method.__globals__['refine_role_mask_losses'] = refine_role_mask_losses
    agent = SimpleNamespace(internal_object_slots_enabled=True, stage_two=True,
                            _net_mod=SimpleNamespace(object_conditioning_supervise_mixed_role_maps=False))
    result = MethodType(method, agent)(output, valid, valid, valid)
    assert result['diversity'] > 0 and result['refine_diversity'] > 0
    assert result['diversity_pixels'] == result['refine_diversity_pixels'] == 2
    assert result['diversity_paired_views'] == result['refine_diversity_paired_views'] == 1
    (result['diversity'] + result['refine_diversity']).backward()
    assert raw.grad[0, 0, 0, 6, 6] > 0 and local.grad[0, 0, 1, 1, 1] > 0
    assert null.grad is None  # This penalty cannot be evaded via predicted NULL.


@pytest.mark.parametrize('dense_residual', [False, True])
def test_outer_mvt_forward_checks_each_stage_frame_without_gt_action_conditioning(dense_residual):
    """Run the real outer forward with CPU render/VLM substitutes."""
    import ast
    from pathlib import Path

    source = Path(__file__).resolve().parents[1] / 'finetune/bridgevla/mvt/mvt.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MVT')
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')

    def transform(points, loc, sca):
        if isinstance(points, list):
            return [(p - loc[i]) * sca for i, p in enumerate(points)], None
        return (points - loc[:, None]) * sca, None

    namespace = dict(torch=torch, visible_role_support=visible_role_support,
                     mvt_utils=SimpleNamespace(trans_pc=transform, add_uni_noi=lambda x, _: x))
    from finetune.bridgevla.models.heatmap_transport import reproject_role_heatmaps
    namespace['reproject_role_heatmaps'] = reproject_role_heatmaps
    namespace['inherit_coarse_roles'] = lambda *args, **kwargs: {}
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(source), 'exec'), namespace)

    calls = []

    def stage(**kwargs):
        calls.append(kwargs)
        assert kwargs['oracle_prior_heatmap'] is None
        assert kwargs['oracle_prior_valid'] is None
        assert kwargs['oracle_relation_points'] is None
        assert 'object_role_visible_support' not in kwargs
        # The stage can receive GT only as a detached loss target, never as
        # role/action conditioning. Its policy output ignores that target.
        return dict(trans=kwargs['img'][:, :, 3].clone(),
                    object_slot_mask_logits=torch.full((1, 1, 2, 8, 8), .3),
                    object_slot_role_token_valid=torch.ones(1, 2).bool())

    stage.get_pt_loc_on_img = lambda p, **_: projection(p)

    def render(pc, **kwargs):
        image = torch.zeros(1, 1, 6, 8, 8)
        for point in pc[0]:
            x, y = projection(point[None, None])[0, 0, 0].floor().long()
            if 0 <= x < 8 and 0 <= y < 8:
                image[0, 0, :3, y, x] = point
        return image

    model = SimpleNamespace(training=True, img_aug_2=0, stage_two=True,
                            verify_inp=lambda **_: None, render=render, mvt1=stage,
                            _build_oracle_instance_prior=lambda *args: None,
                            object_slots_enabled=True, object_slot_predictor_type='role_queries',
                            object_slot_predictor1=None, object_slot_predictor2=None,
                            object_conditioning_shared_action_features=True,
                            object_conditioning_use_context=True,
                            oracle_adapter_translation_only=False,
                            oracle_prior_feature_adapter1=None, oracle_prior_feature_adapter2=None,
                            st_wpt_loc_aug=0, st_sca=1.5, st_wpt_loc_inp_no_noise=True,
                            object_conditioning_inherit_coarse_roles=dense_residual,
                            object_conditioning_refine_heatmap_residual=dense_residual,
                            oracle_relation_gated_adapter=True, refine_role_mask_head=None,
                            object_conditioning_refine_mask_detach=False)
    points = torch.tensor([[[[.25, .25, .5]], [[-.5, -.5, .25]]]])
    args = dict(pc=[points.reshape(-1, 3)], img_feat=[torch.zeros(2, 3)],
                wpt_local=torch.tensor([[.1, .1, .1]]),
                oracle_prior_valid=torch.ones(1, 2).bool())
    result = namespace['forward'](model, oracle_prior_points=points, **args)
    assert result['object_role_visible_support'].sum() == 2
    assert result['mvt2']['object_role_visible_support'].sum() == 2
    # Refine GT must be translated and scaled, not checked in coarse XYZ.
    torch.testing.assert_close(result['mvt2']['object_role_visible_support'],
                               visible_role_support((points - .1) * 1.5, args['oracle_prior_valid'],
                                                    result['mvt2_ori_img'][:, :, :3], projection))
    without = namespace['forward'](model, oracle_prior_points=None, **args)
    assert 'object_role_visible_support' not in without
    torch.testing.assert_close(result['trans'], without['trans'])
    torch.testing.assert_close(result['mvt2']['trans'], without['mvt2']['trans'])
    if dense_residual:
        projected = calls[1]['inherited_object_roles']
        assert projected['projected_role_valid'].sum() == 4  # Two roles at two surfaces.
        assert projected['projected_role_probabilities'].max() > .5
        for key in ('projected_role_valid', 'projected_role_probabilities'):
            torch.testing.assert_close(projected[key], calls[3]['inherited_object_roles'][key])
