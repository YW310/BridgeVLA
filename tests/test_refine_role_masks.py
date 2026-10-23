"""CPU regressions for role-conditioned local readout and weak supervision."""

from types import MethodType, SimpleNamespace
import unittest

import torch
from torch import nn

from finetune.bridgevla.models.refine_role_masks import (
    RoleConditionedRefineMaskHead, refine_role_mask_losses,
)


class RefineRoleMaskTest(unittest.TestCase):
    def inputs(self):
        torch.manual_seed(4)
        head = RoleConditionedRefineMaskHead(4, 8, hidden_dim=8)
        features = torch.randn(3, 4, 4, 4, requires_grad=True)
        rgb = torch.rand(1, 3, 3, 20, 24, requires_grad=True)
        xyz = torch.rand(1, 3, 3, 20, 24) + .1
        packet = dict(
            role_tokens=torch.randn(1, 2, 8, requires_grad=True),
            role_token_valid=torch.ones(1, 2, dtype=torch.bool),
            valid=torch.ones(1, 2, dtype=torch.bool),
            confidence=torch.tensor([[.8, .7]]),
            geometry=torch.randn(1, 17), points=torch.randn(1, 2, 4, 3),
            prior=torch.zeros(1, 3, 2, 20, 24),
            reference_null_probability=torch.tensor([.2], requires_grad=True),
            reference_is_null=torch.tensor([False]), roles_inherited=True,
        )
        return head, features, rgb, xyz, packet

    def test_dense_readout_retains_global_packet_and_grows_beyond_point_hints(self):
        head, features, rgb, xyz, packet = self.inputs()
        result = head(features, rgb, xyz, packet)
        self.assertEqual(result['prior'].shape, (1, 3, 2, 20, 24))
        self.assertGreater(result['prior'].sum().item(), 0)
        for key in ('role_tokens', 'role_token_valid', 'geometry', 'points',
                    'reference_null_probability', 'reference_is_null', 'confidence'):
            self.assertIs(result[key], packet[key])
        self.assertEqual(packet['prior'].count_nonzero().item(), 0)
        self.assertTrue(result['roles_inherited'])
        self.assertTrue(result['masks_refined'])
        self.assertNotIn('role_logits', result)
        self.assertNotIn('slot_masks', result)

    def test_fixed_role_queries_transport_into_refine_with_live_gradients(self):
        from finetune.bridgevla.models.oracle_prior import InternalRoleQueryPredictor
        from finetune.bridgevla.models.cross_scale_roles import inherit_coarse_roles

        head, local_features, rgb, xyz, _ = self.inputs()
        xyz = torch.full_like(xyz, .25)
        coarse_features = torch.randn_like(local_features, requires_grad=True)
        context = torch.randn(1, 4, requires_grad=True)
        predictor = InternalRoleQueryPredictor(
            4, 3, num_slots=2, slot_dim=8, decoder_layers=1, num_heads=2,
            point_samples=4, confidence_threshold=0., use_context=True,
            soft_conditioning=True,
        )
        coarse = predictor(coarse_features, xyz, context=context)
        stage = {'object_slot_' + key: value for key, value in coarse.items()}
        packet = inherit_coarse_roles(
            stage, torch.zeros(1, 3), 1., xyz,
            lambda points: points.new_full((1, points.shape[1], 3, 2), 6.25),
        )
        self.assertTrue(packet['valid'].all())
        result = head(local_features, rgb, xyz, packet)
        self.assertEqual(result['predictor_type'], 'role_queries')
        self.assertIs(result['role_tokens'], coarse['role_tokens'])
        self.assertIs(result['reference_null_probability'],
                      coarse['reference_null_probability'])
        self.assertTrue(result['roles_inherited'])
        self.assertTrue(result['masks_refined'])
        self.assertNotIn('slot_masks', result)
        self.assertNotIn('role_logits', result)
        result['prior'].square().mean().backward()
        for value in (predictor.slot_queries, coarse_features,
                      predictor.context_projection[-1].weight,
                      predictor.reference_null_head.weight, head.query.weight):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0.)
        # The context projection intentionally starts at zero: its weights
        # learn on step one, while upstream text gradients start afterwards.
        torch.testing.assert_close(context.grad, torch.zeros_like(context))

    def test_missing_local_reference_is_not_replaced_or_made_null(self):
        head, features, rgb, xyz, packet = self.inputs()
        packet['valid'][:, 1] = False
        result = head(features, rgb, xyz, packet)
        self.assertEqual(result['prior'][:, :, 1].count_nonzero().item(), 0)
        self.assertGreater(result['prior'][:, :, 0].sum().item(), 0)
        self.assertFalse(result['reference_is_null'].item())
        self.assertTrue(result['role_token_valid'][:, 1].item())

    def test_null_posterior_is_applied_once_to_reference_spatial_mass(self):
        head, features, rgb, xyz, packet = self.inputs()
        first = head(features, rgb, xyz, packet)
        packet['reference_null_probability'] = torch.tensor([.6])
        second = head(features, rgb, xyz, packet)
        torch.testing.assert_close(second['prior'][:, :, 1], first['prior'][:, :, 1] * .5)
        torch.testing.assert_close(second['prior'][:, :, 0], first['prior'][:, :, 0])

    def test_invalid_xyz_and_empty_scene_never_get_spatial_roles(self):
        head, features, rgb, xyz, packet = self.inputs()
        xyz[:, :, :, :6] = 0
        xyz[:, :, :, 6:9] = -1
        xyz[:, :, :, 9:12] = float('nan')
        result = head(features, rgb, xyz, packet)
        self.assertEqual(result['prior'][..., :12, :].count_nonzero().item(), 0)
        self.assertTrue(torch.isfinite(result['prior']).all())
        empty = head(features, rgb, torch.zeros_like(xyz), packet)
        self.assertFalse(empty['valid'].any())
        self.assertEqual(empty['prior'].count_nonzero().item(), 0)
        self.assertFalse(empty['reference_is_null'].item())

    def test_local_map_path_has_action_gradients_into_head_rgb_and_coarse_tokens(self):
        head, features, rgb, xyz, packet = self.inputs()
        head(features, rgb, xyz, packet)['prior'].square().mean().backward()
        for value in (features, rgb, packet['role_tokens'], head.fuse[0].weight,
                      head.detail_half[0].weight, head.query.weight):
            self.assertIsNotNone(value.grad)
            self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_detach_ablation_keeps_segmentation_gradients(self):
        head, features, rgb, xyz, packet = self.inputs()
        result = head(features, rgb, xyz, packet, detach_action=True)
        self.assertFalse(result['prior'].requires_grad)
        result['refine_mask_logits'].square().mean().backward()
        self.assertGreater(head.query.weight.grad.abs().sum().item(), 0)

    def test_weak_teacher_unknown_pixels_empty_views_and_invalid_roles_have_no_gradient(self):
        logits = torch.zeros(1, 2, 2, 4, 4, requires_grad=True)
        teacher = torch.zeros_like(logits)
        teacher[0, 0, 0, 1, 1] = .9
        teacher[0, 0, 1, 2, 2] = .9
        losses = refine_role_mask_losses(logits, teacher, torch.tensor([[True, False]]),
                                        torch.ones(1, 2, 4, 4, dtype=torch.bool))
        losses['mask'].backward()
        self.assertNotEqual(logits.grad[0, 0, 0, 1, 1].item(), 0)
        self.assertEqual(logits.grad.count_nonzero().item(), 1)
        self.assertIsNone(teacher.grad)

    def test_unknown_presence_and_absence_do_not_create_local_or_null_targets(self):
        for present, known in (([True, True], [False, False]), ([True, False], [True, True])):
            logits = torch.zeros(1, 1, 2, 2, 2, requires_grad=True)
            losses = refine_role_mask_losses(
                logits, torch.ones_like(logits), torch.ones(1, 2, dtype=torch.bool),
                torch.ones(1, 1, 2, 2, dtype=torch.bool),
                torch.tensor([present]), torch.tensor([known]))
            losses['mask'].backward()
            self.assertEqual(logits.grad[:, :, 1].count_nonzero().item(), 0)
            if not any(known):
                self.assertEqual(logits.grad.count_nonzero().item(), 0)

    def test_segmentation_loss_does_not_depend_on_predicted_support(self):
        head, features, rgb, xyz, packet = self.inputs()
        packet['valid'].zero_()
        result = head(features, rgb, xyz, packet)
        teacher = torch.full_like(result['refine_mask_logits'], .8)
        loss = refine_role_mask_losses(result['refine_mask_logits'], teacher,
                                       torch.ones(1, 2, dtype=torch.bool),
                                       result['refine_pixel_valid'])['mask']
        loss.backward()
        self.assertGreater(head.query.weight.grad.abs().sum().item(), 0)
        self.assertEqual(result['prior'].count_nonzero().item(), 0)

    def test_agent_adds_refine_loss_without_repeating_role_or_null_losses(self):
        from test_direct_role_losses import _actual_agent_loss, _direct_stage
        method = _actual_agent_loss()
        method.__globals__['refine_role_mask_losses'] = refine_role_mask_losses
        coarse = _direct_stage()
        agent = SimpleNamespace(internal_object_slots_enabled=True, stage_two=True,
                                _net_mod=SimpleNamespace(object_conditioning_supervise_mixed_role_maps=False))
        agent.losses = MethodType(method, agent)
        baseline = dict(coarse, mvt2=dict(coarse, object_slot_roles_inherited=True))
        old = agent.losses(baseline, torch.ones(1, 2, dtype=torch.bool))
        local_logits = torch.zeros_like(coarse['object_slot_prior'], requires_grad=True)
        refined = dict(baseline['mvt2'], object_slot_masks_refined=True,
                       object_refine_mask_logits=local_logits,
                       object_refine_pixel_valid=torch.ones(1, 2, 2, 2, dtype=torch.bool))
        new = agent.losses(dict(coarse, mvt2=refined), torch.ones(1, 2, dtype=torch.bool))
        for key in ('mask', 'presence', 'role', 'objectness', 'diversity'):
            torch.testing.assert_close(old[key], new[key])
        new['refine_mask'].backward()
        self.assertGreater(local_logits.grad.abs().sum().item(), 0)

    def _policy_inputs(self):
        from test_object_conditioning_forward import ObjectConditioningForwardTest
        helper = ObjectConditioningForwardTest()
        module, _, adapter, options, seen = helper._small_policy()
        head, _, _, _, packet = self.inputs()
        options.update(object_slot_predictor=None, inherited_object_roles=packet,
                       refine_role_mask_head=head)
        adapter.preserve_role_tokens = True
        return helper, module, head, options, seen

    def test_actual_prediction_only_forward_ignores_teacher_and_refines_masks(self):
        from unittest import mock
        from bridgevla.mvt import mvt_single
        helper, module, head, options, seen = self._policy_inputs()
        module.eval()
        teacher = torch.zeros(1, 3, 2, 16, 16)
        with mock.patch.object(mvt_single, 'select_feat_from_hm', helper._sample, create=True):
            first = module(**options, object_slot_target_heatmap=teacher)
            second = module(**options, object_slot_target_heatmap=1 - teacher)
            inferred = module(**options)
        for key in ('trans', 'feat_ex_rot', 'object_slot_prior'):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
            torch.testing.assert_close(first[key], inferred[key], rtol=0, atol=0)
        self.assertTrue(inferred['object_slot_roles_inherited'])
        self.assertTrue(inferred['object_slot_masks_refined'])
        self.assertNotIn('object_slot_role_logits', inferred)
        self.assertNotIn('object_slot_target_prior', inferred)
        torch.testing.assert_close(seen['decoded_trans'], inferred['trans'])

    def test_actual_action_loss_and_detach_ablation_route_gradients(self):
        from unittest import mock
        from bridgevla.mvt import mvt_single
        for detached in (False, True):
            helper, module, head, options, _ = self._policy_inputs()
            module.train()
            options.update(wpt_local=torch.tensor([[2., 2., 0.]]), refine_mask_detach=detached)
            with mock.patch.object(mvt_single, 'select_feat_from_hm', helper._sample, create=True):
                out = module(**options)
            loss = out['trans'].square().mean() + out['feat_ex_rot'].square().mean()
            loss.backward()
            if detached:
                self.assertIsNone(head.query.weight.grad)
            else:
                self.assertGreater(head.query.weight.grad.abs().sum().item(), 0)

    def test_one_optimizer_step_updates_head_with_finite_loss(self):
        head, features, rgb, xyz, packet = self.inputs()
        optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
        before = head.query.weight.detach().clone()
        result = head(features, rgb, xyz, packet)
        loss = refine_role_mask_losses(
            result['refine_mask_logits'], torch.full_like(result['refine_mask_logits'], .8),
            torch.ones(1, 2, dtype=torch.bool), result['refine_pixel_valid'])['mask']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(before, head.query.weight))


if __name__ == '__main__':
    unittest.main()
