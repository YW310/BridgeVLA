import unittest

import torch

from finetune.bridgevla.models.object_conditioning import (
    action_feature_routes, pool_instruction_context, reference_null_loss,
    soft_role_geometry,
)
from finetune.bridgevla.models.oracle_prior import (
    InternalObjectSlotPredictor, OracleRelationAnchorFeatureAdapter,
)


class ObjectConditioningTest(unittest.TestCase):
    def test_text_pooling_excludes_left_right_padding_image_and_special_tokens(self):
        hidden = torch.tensor([[[99.], [10.], [20.], [2.], [4.], [88.]],
                               [[10.], [20.], [2.], [4.], [88.], [99.]]])
        mask = torch.tensor([[0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 0]])
        ids = torch.tensor([[0, 7, 7, 3, 4, 9], [7, 7, 3, 4, 9, 0]])
        pooled = pool_instruction_context(hidden, mask, 2, ids, (7, 9, 0))
        torch.testing.assert_close(pooled, torch.tensor([[3.], [3.]]))
        empty = pool_instruction_context(hidden, torch.zeros_like(mask), 2)
        self.assertEqual(empty.count_nonzero().item(), 0)

    def test_routes_preserve_legacy_and_share_both_action_sources(self):
        base = torch.randn(2, 3, 4, 4, requires_grad=True)
        trans = torch.randn_like(base, requires_grad=True)
        legacy = torch.randn_like(base, requires_grad=True)
        t, local, global_ = action_feature_routes(base, trans, legacy)
        self.assertIs(t, trans)
        self.assertIs(local, legacy)
        self.assertIs(global_, base)
        t, local, global_ = action_feature_routes(base, trans, legacy, True)
        self.assertIs(local, t)
        self.assertIs(global_, t)
        (local.sum() + global_.sum()).backward()
        self.assertIsNone(base.grad)
        self.assertIsNone(legacy.grad)
        self.assertGreater(trans.grad.abs().sum().item(), 0)

    def test_null_labels_distinguish_absence_occlusion_and_unknown(self):
        probability = torch.tensor([0.8, 0.8, 0.8], requires_grad=True)
        present = torch.tensor([[True, False], [True, True], [False, False]])
        known = torch.tensor([[True, True], [True, True], [False, False]])
        loss = reference_null_loss(probability, present, known)
        loss.backward()
        self.assertLess(probability.grad[0].item(), 0)  # genuine NULL
        self.assertGreater(probability.grad[1].item(), 0)  # present, even if hidden
        self.assertEqual(probability.grad[2].item(), 0)  # unknown/placeholder
        self.assertEqual(reference_null_loss(probability).item(), 0)

    def test_soft_geometry_is_finite_differentiable_and_masks_empty_support(self):
        prior = torch.rand(1, 1, 2, 2, 2, requires_grad=True)
        xyz = torch.tensor([[[[[0.2, 0.4], [0., float('nan')]],
                              [[0.3, 0.5], [0., float('nan')]],
                              [[0.4, 0.6], [0., float('nan')]]]]])
        geometry, available = soft_role_geometry(prior, xyz, torch.ones(1, 2).bool())
        self.assertEqual(tuple(geometry.shape), (1, 17))
        self.assertTrue(available.all())
        self.assertTrue(torch.isfinite(geometry).all())
        geometry.sum().backward()
        self.assertGreater(prior.grad.abs().sum().item(), 0)
        empty, available = soft_role_geometry(prior, torch.zeros_like(xyz), available)
        self.assertFalse(available.any())
        self.assertEqual(empty.count_nonzero().item(), 0)

    def test_anchor_zero_initialization_and_context_role_geometry_gradients(self):
        adapter = OracleRelationAnchorFeatureAdapter(
            4, rank=2, anchor_rank=2, use_context=True,
            role_conditioning=True, role_token_dim=8,
        )
        features = torch.randn(1, 4, 2, 2)
        prior = torch.rand(1, 1, 2, 2, 2, requires_grad=True)
        xyz = torch.rand(1, 1, 3, 2, 2) + 0.1
        geometry, valid = soft_role_geometry(prior, xyz, torch.ones(1, 2).bool())
        context = torch.randn(1, 4, requires_grad=True)
        tokens = torch.randn(1, 2, 8, requires_grad=True)
        arguments = dict(current_state=torch.zeros(1, 3), context=context,
                         role_tokens=tokens, geometry=geometry,
                         reference_null_probability=torch.tensor([0.2]))
        final, shared, _ = adapter.forward_with_anchor(
            features, prior, valid, torch.zeros(1, 2, 4, 3), **arguments,
        )
        torch.testing.assert_close(final, features)
        torch.testing.assert_close(shared, features)
        # After the zero-initialized output projection starts learning, all
        # conditioning paths must receive the full-action gradient.
        torch.nn.init.normal_(adapter.anchor_expand.weight, std=0.1)
        torch.nn.init.normal_(adapter.context_projection[-1].weight, std=0.1)
        final, _, _ = adapter.forward_with_anchor(
            features, prior, valid, torch.zeros(1, 2, 4, 3), **arguments,
        )
        final.square().sum().backward()
        for value in (prior, context, tokens):
            self.assertGreater(value.grad.abs().sum().item(), 0)

    def test_low_reference_confidence_does_not_disable_available_target(self):
        predictor = InternalObjectSlotPredictor(
            4, num_views=1, num_slots=2, slot_dim=8, num_heads=2,
            decoder_layers=1, point_samples=4, soft_conditioning=True,
        )
        with torch.no_grad():
            predictor.objectness_head.weight.zero_()
            predictor.objectness_head.bias.fill_(2.)
            predictor.role_head.weight.zero_()
            predictor.role_head.bias.zero_()
            predictor.reference_null_head.weight.zero_()
            # R confidence < .25, while NULL posterior remains below .75.
            predictor.reference_null_head.bias.fill_(0.85)
        out = predictor(torch.randn(1, 4, 2, 2), torch.rand(1, 1, 3, 2, 2) + 0.1)
        self.assertTrue(out['valid'][0, 0])
        self.assertFalse(out['valid'][0, 1])
        self.assertFalse(out['reference_is_null'][0])
        self.assertEqual(tuple(out['role_tokens'].shape), (1, 2, 8))

    def test_old_trained_anchor_initializes_context_without_changing_translation(self):
        old = OracleRelationAnchorFeatureAdapter(4, rank=2, anchor_rank=2)
        torch.nn.init.normal_(old.anchor_expand.weight, std=.1)
        new = OracleRelationAnchorFeatureAdapter(4, rank=2, anchor_rank=2, use_context=True)
        incompatible = new.load_state_dict(old.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(all(key.startswith('context_projection.')
                            for key in incompatible.missing_keys))
        feature, prior = torch.randn(1, 4, 2, 2), torch.rand(1, 1, 2, 2, 2)
        valid, points = torch.ones(1, 2).bool(), torch.rand(1, 2, 4, 3)
        expected = old.forward_with_anchor(feature, prior, valid, points)[0]
        actual = new.forward_with_anchor(feature, prior, valid, points,
                                         context=torch.randn(1, 4))[0]
        torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
