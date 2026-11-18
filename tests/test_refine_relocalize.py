"""Independent fine localization can correct weak/wrong coarse spatial hints."""

from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'finetune'))
from finetune.bridgevla.models.refine_role_masks import RoleConditionedRefineMaskHead, refine_role_mask_losses
import test_refine_local_query as query_tests


def inputs():
    _, features, rgb, xyz, packet = query_tests.RefineLocalQueryTest()._inputs(point_support=False)
    torch.manual_seed(29)
    head = RoleConditionedRefineMaskHead(4, 8, hidden_dim=8, local_query=True, relocalize=True)
    packet['valid'].zero_()
    packet['role_token_valid'].zero_()
    packet['geometry'] = torch.zeros(1, 17)
    packet['points'].zero_()
    packet['confidence'].zero_()
    return head, features, rgb, xyz, packet


def test_local_evidence_can_recover_both_coarse_gates_without_changing_role_identity():
    head, features, rgb, xyz, packet = inputs()
    with torch.no_grad():
        head.bias.fill_(2.)  # Controlled reliable local response, not model performance evidence.
    result = head(features, rgb, xyz, packet)
    assert result['valid'].all() and result['role_token_valid'].all()
    assert not packet['valid'].any() and not packet['role_token_valid'].any()
    assert result['relocalized'] and result['roles_inherited']
    torch.testing.assert_close(result['geometry'][:, :6], torch.full((1, 6), .25))
    assert result['geometry'][:, 15:].bool().all()
    assert result['points'].shape == packet['points'].shape
    assert result['confidence'].min() >= .25
    assert result['reference_null_probability'] is packet['reference_null_probability']
    torch.testing.assert_close(result['role_tokens'], packet['role_tokens'])  # Zero-init local token delta.


def test_weak_fine_response_is_not_forced_into_a_detection():
    head, features, rgb, xyz, packet = inputs()
    with torch.no_grad():
        head.bias.fill_(-20.)
    result = head(features, rgb, xyz, packet)
    assert not result['valid'].any() and not result['role_token_valid'].any()
    assert result['prior'].count_nonzero() == 0
    assert result['points'].count_nonzero() == 0


@pytest.mark.parametrize('value', [0., -1., float('nan'), 2.])
def test_empty_invalid_or_crop_external_surfaces_cannot_create_local_geometry(value):
    head, features, rgb, xyz, packet = inputs()
    xyz[:] = value
    with torch.no_grad():
        head.bias.fill_(20.)
    result = head(features, rgb, xyz, packet)
    assert not result['valid'].any()
    assert result['prior'].count_nonzero() == 0
    assert torch.isfinite(result['refine_mask_logits']).all()
    assert torch.isfinite(result['geometry']).all()


def test_bright_refine_response_does_not_reverse_reference_null():
    head, features, rgb, xyz, packet = inputs()
    packet['reference_null_probability'] = torch.tensor([.99])
    packet['reference_is_null'] = torch.tensor([True])
    with torch.no_grad():
        head.bias.fill_(20.)
    result = head(features, rgb, xyz, packet)
    assert result['valid'][0, 0] and not result['valid'][0, 1]
    assert not result['role_token_valid'][0, 1]
    assert result['prior'][:, :, 1].count_nonzero() == 0
    assert result['reference_is_null'] is packet['reference_is_null']


def test_wrong_coarse_position_is_soft_bias_and_never_a_local_search_mask():
    head, features, rgb, xyz, packet = inputs()
    xyz[:, :, 0, :, :8] = -.6
    xyz[:, :, 0, :, 8:] = .6
    packet['prior'][:, :, :, :, :3] = 1.
    packet['geometry'] = torch.tensor([[-.9] * 6 + [.01] * 6 + [0.] * 3 + [1.] * 2])
    with torch.no_grad():
        head.local_query.spatial_gain.fill_(100.)
    result = head(features, rgb, xyz, packet)
    assert result['refine_role_pixel_valid'].all()
    assert result['refine_query_active'].all()
    assert torch.isfinite(result['refine_mask_logits']).all()
    # A GT point far from the coarse hint still trains the local decoder.
    teacher = torch.zeros_like(result['refine_mask_logits'])
    teacher[:, :, :, 8, 12] = 1.
    losses = refine_role_mask_losses(result['refine_mask_logits'], teacher,
        torch.ones(1, 2, dtype=torch.bool), result['refine_pixel_valid'])
    losses['mask'].backward()
    assert head.query.weight.grad.abs().sum() > 0


def test_low_local_reference_keeps_global_reference_context():
    head, features, rgb, xyz, packet = inputs()
    packet['role_token_valid'][0, 1] = True
    packet['geometry'][0, 3:6] = 10.
    packet['geometry'][0, 16] = 1.
    with torch.no_grad():
        head.bias[0] = 2.
        head.bias[1] = -20.
    result = head(features, rgb, xyz, packet)
    assert not result['valid'][0, 1] and result['role_token_valid'][0, 1]
    torch.testing.assert_close(result['geometry'][0, 3:6], packet['geometry'][0, 3:6])
    torch.testing.assert_close(result['points'][0, 1], packet['points'][0, 1])


def test_visible_center_loss_moves_fine_localization_away_from_wrong_coarse_hint():
    head, features, rgb, xyz, packet = inputs()
    xyz[:, :, 0, :, :8] = -.6
    xyz[:, :, 0, :, 8:] = .6
    rgb = rgb.detach()
    rgb[:, :, :, :, :8] = .1
    rgb[:, :, :, :, 8:] = .9
    packet['prior'][:, :, 0, :, :4] = 1.  # Wrong coarse peak on the left.
    packet['reference_is_null'] = torch.tensor([True])
    packet['reference_null_probability'] = torch.tensor([.99])
    teacher = torch.zeros(1, 3, 2, 16, 16)
    teacher[:, :, 0, 8, 12] = 1.
    visible = teacher.bool()
    optimizer = torch.optim.Adam(head.parameters(), lr=.01)
    errors = []
    # Small deterministic training experiment tests a learning/gradient contract.
    # It is not a pretrained-model accuracy or RLBench success measurement.
    for _ in range(30):
        optimizer.zero_grad()
        result = head(features.detach(), rgb, xyz, packet)
        loss = refine_role_mask_losses(result['refine_mask_logits'], teacher,
            torch.ones(1, 2, dtype=torch.bool), result['refine_pixel_valid'],
            torch.tensor([[True, False]]), torch.ones(1, 2, dtype=torch.bool),
            rendered_xyz=result['refine_rendered_xyz'], visible_support=visible)
        errors.append(float(loss['center'].detach()))
        (loss['mask'] + 3 * loss['center']).backward()
        optimizer.step()
    assert errors[-1] < errors[0] * .3
    assert errors[-1] < .05
    assert not packet['role_token_valid'].any()


@pytest.mark.parametrize('kind', ['missing_visibility', 'unknown', 'absent', 'occluded'])
def test_center_teacher_requires_known_present_current_visible_geometry(kind):
    logits = torch.zeros(1, 1, 2, 4, 4, requires_grad=True)
    teacher = torch.zeros_like(logits)
    teacher[:, :, :, 1, 1] = 1.
    xyz = torch.rand(1, 1, 3, 4, 4)
    visible = teacher.bool()
    present = torch.ones(1, 2, dtype=torch.bool)
    known = present.clone()
    if kind == 'missing_visibility':
        visible = None
    elif kind == 'unknown':
        known.zero_()
    elif kind == 'absent':
        present.zero_()
    else:
        visible.zero_()
    loss = refine_role_mask_losses(logits, teacher, torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 1, 4, 4, dtype=torch.bool), present, known,
        rendered_xyz=xyz, visible_support=visible)['center']
    loss.backward()
    assert loss == 0 and logits.grad.count_nonzero() == 0


def test_center_supervision_ignores_predicted_gates_and_detaches_teacher_geometry():
    logits = torch.zeros(1, 1, 2, 4, 4, requires_grad=True)
    teacher = torch.zeros_like(logits)
    teacher[:, :, :, 1, 1] = 1.
    teacher.requires_grad_()
    xyz = torch.rand(1, 1, 3, 4, 4, requires_grad=True)
    loss = refine_role_mask_losses(logits, teacher, torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 1, 4, 4, dtype=torch.bool), rendered_xyz=xyz,
        visible_support=teacher.detach().bool())['center']
    loss.backward()
    assert logits.grad.abs().sum() > 0
    assert teacher.grad is None and xyz.grad is None


def test_actual_policy_relocalizes_without_teacher_inputs_and_records_local_geometry():
    from test_refine_role_masks import RefineRoleMaskTest
    from bridgevla.mvt import mvt_single
    from unittest import mock
    helper, module, _, options, _ = RefineRoleMaskTest()._policy_inputs()
    head = RoleConditionedRefineMaskHead(4, 8, hidden_dim=8, local_query=True, relocalize=True)
    packet = options['inherited_object_roles']
    packet['role_token_valid'].zero_()
    packet['valid'].zero_()
    packet['prior'] = torch.zeros(1, 3, 2, 16, 16)
    with torch.no_grad():
        head.bias.fill_(2.)
    options['refine_role_mask_head'] = head
    module.eval()
    with mock.patch.object(mvt_single, 'select_feat_from_hm', helper._sample, create=True):
        first = module(**options)
        labelled = module(**options, object_slot_target_heatmap=torch.rand(1, 3, 2, 16, 16))
    assert first['object_slot_relocalized']
    assert first['object_slot_valid'].all()
    assert first['object_slot_role_token_valid'].all()
    assert 'object_refine_rendered_xyz' in first
    for key in ('trans', 'feat_ex_rot', 'object_slot_prior', 'object_slot_geometry'):
        torch.testing.assert_close(first[key], labelled[key], rtol=0, atol=0)


def test_legacy_head_state_and_outputs_are_exact_when_new_mode_is_false():
    old, features, rgb, xyz, packet = query_tests.RefineLocalQueryTest()._inputs()
    compatible = RoleConditionedRefineMaskHead(4, 8, hidden_dim=8, local_query=True, relocalize=False)
    compatible.load_state_dict(old.state_dict(), strict=True)
    a, b = old(features, rgb, xyz, packet), compatible(features, rgb, xyz, packet)
    for key in ('prior', 'role_tokens', 'geometry', 'points', 'role_token_valid'):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    assert 'relocalized' not in b


def test_training_hint_dropout_removes_spatial_cues_without_mutating_global_packet():
    head, features, rgb, xyz, packet = inputs()
    packet['prior'].fill_(.8)
    packet['geometry'][:, 15:] = 1.
    head.hint_dropout = 1.
    head.train()
    result = head(features, rgb, xyz, packet)
    assert result['refine_hint_dropped'].all()
    assert packet['prior'].min() == .8 and packet['geometry'][:, 15:].min() == 1.
    assert result['refine_query_active'].all()
    head.eval()
    assert 'refine_hint_dropped' not in head(features, rgb, xyz, packet)


def test_actual_agent_adds_local_center_supervision_without_predicted_visibility_gates():
    from test_direct_role_losses import _actual_agent_loss, _direct_stage
    from types import MethodType, SimpleNamespace
    method = _actual_agent_loss()
    method.__globals__['refine_role_mask_losses'] = refine_role_mask_losses
    coarse = _direct_stage()
    logits = torch.zeros_like(coarse['object_slot_prior'], requires_grad=True)
    refined = dict(coarse, object_slot_roles_inherited=True, object_slot_masks_refined=True,
        object_slot_valid=torch.zeros(1, 2, dtype=torch.bool),
        object_slot_role_token_valid=torch.zeros(1, 2, dtype=torch.bool),
        object_refine_mask_logits=logits, object_refine_pixel_valid=torch.ones(1, 2, 2, 2, dtype=torch.bool),
        object_refine_rendered_xyz=torch.rand(1, 2, 3, 2, 2),
        object_role_visible_support=coarse['object_slot_target_prior'].bool())
    agent = SimpleNamespace(internal_object_slots_enabled=True, stage_two=True,
                            _net_mod=SimpleNamespace(object_conditioning_supervise_mixed_role_maps=False))
    result = MethodType(method, agent)(dict(coarse, mvt2=refined), torch.ones(1, 2, dtype=torch.bool))
    assert result['refine_center'] > 0
    result['refine_center'].backward()
    assert logits.grad.abs().sum() > 0
