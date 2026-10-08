"""Numerical tiny-policy integration checks for fixed Target/Reference queries."""

import unittest
from unittest import mock

import torch
from torch import nn

from tests import test_object_conditioning_forward as _fixture
from bridgevla.mvt import mvt_single
from bridgevla.models.cross_scale_roles import inherit_coarse_roles
from bridgevla.models.oracle_prior import InternalRoleQueryPredictor


class RoleQueryForwardTest(unittest.TestCase):
    def _policy(self):
        torch.manual_seed(19)
        module, _, adapter, options, decoded = (
            _fixture.ObjectConditioningForwardTest()._small_policy()
        )
        predictor = InternalRoleQueryPredictor(
            4, 3, num_slots=2, slot_dim=8, decoder_layers=1, num_heads=2,
            point_samples=4, confidence_threshold=0., use_context=True,
            soft_conditioning=True,
        )
        adapter.preserve_role_tokens = True
        options['object_slot_predictor'] = predictor
        return module, predictor, adapter, options, decoded

    def _forward(self, module, options, **extra):
        with mock.patch.object(
            mvt_single, 'select_feat_from_hm',
            _fixture.ObjectConditioningForwardTest._sample, create=True,
        ):
            return module(**options, **extra)

    def test_prediction_only_teacher_isolation_and_final_waypoint_sampling(self):
        module, predictor, adapter, options, decoded = self._policy()
        module.eval()
        predictor.eval()
        teacher = torch.zeros(1, 3, 2, 16, 16)
        with torch.no_grad():
            first = self._forward(module, options, object_slot_target_heatmap=teacher)
            second = self._forward(module, options, object_slot_target_heatmap=1 - teacher)
            predicted_only = self._forward(module, options)
        self.assertEqual(predicted_only['object_slot_predictor_type'], 'role_queries')
        self.assertNotIn('object_slot_target_prior', predicted_only)
        self.assertNotIn('object_slot_role_logits', predicted_only)
        self.assertNotIn('object_slot_objectness_logits', predicted_only)
        for key in ('trans', 'feat_ex_rot'):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
            torch.testing.assert_close(first[key], predicted_only[key], rtol=0, atol=0)
        # get_wpt is invoked inside the R/G/C local sampler with final heatmap.
        torch.testing.assert_close(decoded['decoded_trans'], predicted_only['trans'])

    def test_zero_initialized_adapter_is_identity_for_full_action_features(self):
        module, predictor, adapter, options, _ = self._policy()
        module.eval()
        predictor.eval()
        nn.init.zeros_(adapter.feature_expand.weight)
        nn.init.zeros_(adapter.feature_expand.bias)
        nn.init.zeros_(adapter.anchor_expand.weight)
        nn.init.zeros_(adapter.anchor_expand.bias)
        with torch.no_grad():
            conditioned = self._forward(module, options)
            no_adapter = self._forward(
                module, {**options, 'oracle_feature_adapter': None},
            )
        for key in ('trans', 'feat_ex_rot'):
            torch.testing.assert_close(conditioned[key], no_adapter[key], rtol=0, atol=0)

    def test_one_optimizer_step_backpropagates_action_to_roles_masks_and_context(self):
        module, predictor, adapter, options, _ = self._policy()
        module.train()
        predictor.train()
        options['wpt_local'] = torch.tensor([[2., 2., 0.]])
        # Both residual output layers must be nonzero for action gradients to
        # reach the predictor on the first step; actual training warms them up.
        with torch.no_grad():
            adapter.feature_expand.weight.normal_(std=.1)
            adapter.anchor_expand.weight.normal_(std=.1)
            adapter.role_token_projection.weight.normal_(std=.2)
        parameters = (list(module.parameters()) + list(predictor.parameters())
                      + list(adapter.parameters()))
        optimizer = torch.optim.SGD(parameters, lr=1e-2)
        before = predictor.slot_queries.detach().clone()
        output = self._forward(module, options)
        loss = output['trans'].square().mean() + output['feat_ex_rot'].square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name, parameter in (
            ('role queries', predictor.slot_queries),
            ('role mask head', predictor.mask_query.weight),
            ('instruction context', predictor.context_projection[-1].weight),
            ('anchor output', adapter.anchor_expand.weight),
        ):
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertGreater((predictor.slot_queries.detach() - before).abs().sum().item(), 0)

    def test_inherited_reference_unknown_keeps_token_and_null_posterior(self):
        module, predictor, adapter, options, _ = self._policy()
        module.eval()
        predictor.eval()
        with torch.no_grad():
            adapter.feature_expand.weight.normal_(std=.1)
            adapter.anchor_expand.weight.normal_(std=.1)
            adapter.role_token_projection.weight.normal_(std=.2)
            coarse = self._forward(module, options)
        source = dict(coarse)
        source['object_slot_points'] = torch.tensor([[[[.2, .3, .4]] * 4,
                                                       [[20., 20., 20.]] * 4]])
        source['object_slot_valid'] = torch.tensor([[True, True]])
        source['object_slot_role_token_valid'] = torch.tensor([[True, True]])
        local_xyz = torch.zeros(1, 3, 3, 16, 16)
        local_xyz[:, :, :, 5, 5] = torch.tensor([.2, .3, .4])

        def project(points):
            return torch.full((1, points.shape[1], 3, 2), 5.25)

        inherited = inherit_coarse_roles(
            source, torch.zeros(1, 3), 1., local_xyz, project,
        )
        self.assertEqual(inherited['predictor_type'], 'role_queries')
        torch.testing.assert_close(inherited['valid'], torch.tensor([[True, False]]))
        torch.testing.assert_close(inherited['role_token_valid'],
                                   torch.tensor([[True, True]]))
        torch.testing.assert_close(inherited['reference_null_probability'],
                                   coarse['object_slot_reference_null_probability'])
        self.assertTrue(inherited['prior'][:, :, 1].eq(0).all())
        self.assertFalse(bool(inherited['reference_is_null'].item()))
        inherited_options = {**options, 'object_slot_predictor': None,
                             'inherited_object_roles': inherited}
        with torch.no_grad():
            original = self._forward(module, inherited_options)
            changed = dict(inherited)
            changed['role_tokens'] = inherited['role_tokens'].clone()
            changed['role_tokens'][:, 1] += 2.0
            changed_output = self._forward(
                module, {**inherited_options, 'inherited_object_roles': changed},
            )
        self.assertTrue(original['object_slot_roles_inherited'])
        self.assertEqual(original['object_slot_predictor_type'], 'role_queries')
        self.assertNotIn('object_slot_role_logits', original)
        self.assertGreater((original['trans'] - changed_output['trans']).abs().max().item(),
                           0)


if __name__ == '__main__':
    unittest.main()
