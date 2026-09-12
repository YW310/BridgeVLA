import unittest
from contextlib import ExitStack
from unittest import mock

import torch
from finetune.bridgevla.models import oracle_prior as oracle_prior_module

from finetune.bridgevla.models.oracle_prior import (
    InternalObjectSlotPredictor,
    OraclePriorFeatureAdapter,
    OracleRelationAnchorFeatureAdapter,
    OracleRelationGatedFeatureAdapter,
    build_training_visualization_payload,
    choose_oracle_translation_loss,
    rasterize_instance_points,
    route_oracle_adapter_features,
    select_active_instance_points,
    select_relation_instance_points,
    valid_oracle_translation_loss,
)


class OraclePriorTest(unittest.TestCase):
    def test_feature_adapter_is_identity_and_receives_gradients(self):
        adapter = OraclePriorFeatureAdapter(8, rank=3)
        features = torch.randn(6, 8, 4, 4)
        prior = torch.rand(2, 3, 8, 8)
        adapted = adapter(features, prior, torch.tensor([True, True]))
        torch.testing.assert_close(adapted, features)
        adapted.square().mean().backward()
        self.assertGreater(
            adapter.feature_expand.weight.grad.abs().sum().item(), 0
        )

    def test_recommended_oracle_adapters_are_lightweight(self):
        adapter = OraclePriorFeatureAdapter(
            2048, rank=16, prior_channels=2,
        )
        per_stage = sum(p.numel() for p in adapter.parameters())
        self.assertEqual(per_stage * 2, 135808)

    def test_relation_gated_oracle_adapters_are_lightweight(self):
        adapter = OracleRelationGatedFeatureAdapter(
            2048, rank=16, prior_channels=2,
        )
        per_stage = sum(p.numel() for p in adapter.parameters())
        self.assertEqual(per_stage * 2, 139138)

    def test_relation_anchor_adapters_are_lightweight(self):
        adapter = OracleRelationAnchorFeatureAdapter(
            2048, rank=16, prior_channels=2, anchor_rank=16,
        )
        per_stage = sum(p.numel() for p in adapter.parameters())
        self.assertEqual(per_stage * 2, 211586)

    def test_feature_adapter_keeps_invalid_sample_unchanged(self):
        adapter = OraclePriorFeatureAdapter(4, rank=2)
        torch.nn.init.ones_(adapter.feature_expand.weight)
        features = torch.randn(2, 4, 3, 3)
        prior = torch.rand(2, 1, 6, 6)
        adapted = adapter(features, prior, torch.tensor([True, False]))
        torch.testing.assert_close(adapted[1], features[1])

    def test_auto_role_uses_target_open_reference_closed(self):
        points = torch.zeros(2, 3, 4, 3)
        points[:, 0] = 10
        points[:, 1] = 20
        points[:, 2] = 30
        valid = torch.ones(2, 3, dtype=torch.bool)
        roles = torch.tensor([[1, 2, 0], [1, 2, 0]])
        selected, selected_valid, slots = select_active_instance_points(
            points, valid, roles, gripper_open=torch.tensor([1.0, 0.0]),
            active_role='auto', strict=True,
        )
        self.assertEqual(slots.tolist(), [0, 1])
        self.assertTrue(selected_valid.all())
        self.assertEqual(selected[:, 0, 0].tolist(), [10.0, 20.0])

    def test_relation_selection_keeps_target_then_reference(self):
        points = torch.zeros(1, 4, 3, 3)
        points[:, 1] = 20
        points[:, 3] = 40
        selected, selected_valid, slots = select_relation_instance_points(
            points,
            torch.ones(1, 4, dtype=torch.bool),
            torch.tensor([[0, 2, 0, 1]]),
            strict=True,
        )
        self.assertEqual(slots.tolist(), [[3, 1]])
        self.assertEqual(selected_valid.tolist(), [[True, True]])
        self.assertEqual(
            selected[:, :, 0, 0].tolist(), [[40.0, 20.0]]
        )

    def test_relation_non_strict_missing_role_disables_pair(self):
        selected, selected_valid, slots = select_relation_instance_points(
            torch.ones(1, 2, 3, 3),
            torch.ones(1, 2, dtype=torch.bool),
            torch.tensor([[1, 0]]),
            strict=False,
        )
        self.assertEqual(selected_valid.tolist(), [[True, False]])
        self.assertEqual(slots.tolist(), [[0, -1]])
        self.assertEqual(selected[:, 1].count_nonzero().item(), 0)

    def test_relation_adapter_is_identity_at_initialization(self):
        adapter = OraclePriorFeatureAdapter(
            8, rank=3, prior_channels=2,
        )
        features = torch.randn(6, 8, 4, 4)
        prior = torch.rand(2, 3, 2, 8, 8)
        adapted = adapter(
            features, prior,
            torch.tensor([[True, True], [True, True]]),
        )
        torch.testing.assert_close(adapted, features)

    def test_relation_gated_adapter_is_identity_and_receives_gradients(self):
        adapter = OracleRelationGatedFeatureAdapter(
            8, rank=3, prior_channels=2,
        )
        features = torch.randn(6, 8, 4, 4)
        prior = torch.rand(2, 3, 2, 8, 8)
        points = torch.randn(2, 2, 5, 3)
        valid = torch.tensor([[True, True], [True, True]])
        adapted = adapter(features, prior, valid, points)
        torch.testing.assert_close(adapted, features)
        adapted.square().mean().backward()
        self.assertGreater(
            adapter.feature_expand.weight.grad.abs().sum().item(), 0
        )

    def test_relation_gated_adapter_supports_null_reference(self):
        adapter = OracleRelationGatedFeatureAdapter(
            4, rank=2, prior_channels=2,
        )
        torch.nn.init.ones_(adapter.feature_expand.weight)
        features = torch.randn(2, 4, 3, 3)
        prior = torch.rand(2, 1, 2, 6, 6)
        points = torch.randn(2, 2, 5, 3)
        valid = torch.tensor([[True, True], [True, False]])
        adapted = adapter(features, prior, valid, points)
        self.assertFalse(torch.allclose(adapted[1], features[1]))

    def test_internal_slots_return_role_priors_points_and_gradients(self):
        predictor = InternalObjectSlotPredictor(
            feature_channels=8,
            num_views=3,
            num_slots=4,
            slot_dim=8,
            decoder_layers=1,
            num_heads=2,
            point_samples=5,
        )
        features = torch.randn(6, 8, 4, 4, requires_grad=True)
        rendered_xyz = torch.randn(2, 3, 3, 8, 8)
        output = predictor(features, rendered_xyz, torch.randn(2, 3))
        self.assertEqual(tuple(output['prior'].shape), (2, 3, 2, 4, 4))
        self.assertEqual(tuple(output['points'].shape), (2, 2, 5, 3))
        self.assertEqual(tuple(output['valid'].shape), (2, 2))
        loss = output['prior'].mean() + output['objectness_logits'].mean()
        loss.backward()
        self.assertGreater(predictor.slot_queries.grad.abs().sum().item(), 0)

    def test_relation_anchor_is_identity_and_returns_view_anchor(self):
        adapter = OracleRelationAnchorFeatureAdapter(
            8, rank=3, prior_channels=2, anchor_rank=3,
        )
        features = torch.randn(6, 8, 4, 4)
        prior = torch.rand(2, 3, 2, 8, 8)
        points = torch.randn(2, 2, 5, 3)
        valid = torch.tensor([[True, True], [True, False]])
        adapted, shared, anchor = adapter.forward_with_anchor(
            features, prior, valid, points,
            torch.randn(2, 3),
        )
        torch.testing.assert_close(adapted, features)
        torch.testing.assert_close(shared, features)
        self.assertEqual(tuple(anchor.shape), (2, 3, 4, 4))
        adapted.square().mean().backward()
        self.assertGreater(
            adapter.anchor_expand.weight.grad.abs().sum().item(), 0
        )

    def test_relation_anchor_supports_null_reference(self):
        adapter = OracleRelationAnchorFeatureAdapter(
            4, rank=2, prior_channels=2, anchor_rank=2,
        )
        torch.nn.init.ones_(adapter.anchor_expand.weight)
        torch.nn.init.ones_(adapter.anchor_expand.bias)
        features = torch.ones(1, 4, 3, 3)
        prior = torch.zeros(1, 1, 2, 6, 6)
        prior[:, :, 0, 1:5, 1:5] = 1
        points = torch.zeros(1, 2, 5, 3)
        points[:, 0] = 0.25
        adapted = adapter(
            features, prior, torch.tensor([[True, False]]), points,
        )
        self.assertFalse(torch.allclose(adapted, features))

    def test_relation_anchor_preserves_original_shared_feature_path(self):
        adapter = OracleRelationAnchorFeatureAdapter(
            4, rank=2, prior_channels=2, anchor_rank=2,
        )
        torch.nn.init.ones_(adapter.anchor_expand.weight)
        torch.nn.init.ones_(adapter.anchor_expand.bias)
        features = torch.ones(1, 4, 3, 3)
        prior = torch.rand(1, 1, 2, 6, 6)
        points = torch.randn(1, 2, 5, 3)
        valid = torch.tensor([[True, True]])
        translation, shared, _ = adapter.forward_with_anchor(
            features, prior, valid, points, torch.zeros(1, 3),
        )
        original_shared = OracleRelationGatedFeatureAdapter.forward(
            adapter, features, prior, valid, points,
        )
        torch.testing.assert_close(shared, original_shared)
        self.assertFalse(torch.allclose(translation, shared))

    def test_relation_anchor_keeps_missing_target_unchanged(self):
        adapter = OracleRelationAnchorFeatureAdapter(
            4, rank=2, prior_channels=2, anchor_rank=2,
        )
        torch.nn.init.ones_(adapter.anchor_expand.weight)
        torch.nn.init.ones_(adapter.anchor_expand.bias)
        features = torch.randn(1, 4, 3, 3)
        adapted = adapter(
            features,
            torch.rand(1, 1, 2, 6, 6),
            torch.tensor([[False, True]]),
            torch.randn(1, 2, 5, 3),
        )
        torch.testing.assert_close(adapted, features)

    def test_valid_oracle_translation_loss_ignores_incomplete_pairs(self):
        values = torch.tensor(
            [[1.0, 3.0], [100.0, 100.0]], requires_grad=True,
        )
        valid = torch.tensor([[True, True], [True, False]])
        loss = valid_oracle_translation_loss(values, valid)
        self.assertEqual(loss.item(), 2.0)
        loss.backward()
        torch.testing.assert_close(values.grad[1], torch.zeros(2))

    def test_valid_oracle_translation_loss_all_invalid_is_zero(self):
        values = torch.tensor([[2.0, 4.0]], requires_grad=True)
        loss = valid_oracle_translation_loss(
            values, torch.tensor([[True, False]]),
        )
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        torch.testing.assert_close(values.grad, torch.zeros_like(values))

    def test_valid_oracle_translation_loss_uses_global_ddp_count(self):
        values = torch.tensor([[2.0]], requires_grad=True)

        def fake_all_reduce(tensor, op=None):
            tensor.mul_(3.0)

        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(
                oracle_prior_module.dist, 'is_available', return_value=True,
            ))
            stack.enter_context(mock.patch.object(
                oracle_prior_module.dist, 'is_initialized', return_value=True,
            ))
            stack.enter_context(mock.patch.object(
                oracle_prior_module.dist, 'get_world_size', return_value=2,
            ))
            stack.enter_context(mock.patch.object(
                oracle_prior_module.dist,
                'all_reduce',
                side_effect=fake_all_reduce,
            ))
            loss = valid_oracle_translation_loss(
                values, torch.tensor([[True, True]]), distributed=True,
            )
        torch.testing.assert_close(loss, torch.tensor(4.0 / 3.0))

    def test_translation_only_route_isolates_action_feature_gradient(self):
        base = torch.randn(1, 2, requires_grad=True)
        adapted = torch.randn(1, 2, requires_grad=True)
        translation, action = route_oracle_adapter_features(
            base, adapted, translation_only=True,
        )
        self.assertIs(translation, adapted)
        self.assertIs(action, base)
        action.sum().backward()
        self.assertIsNone(adapted.grad)
        torch.testing.assert_close(base.grad, torch.ones_like(base))

    def test_joint_route_uses_adapted_action_features(self):
        base = torch.randn(1, 2)
        adapted = torch.randn(1, 2)
        _, action = route_oracle_adapter_features(
            base, adapted, translation_only=False,
        )
        self.assertIs(action, adapted)

    def test_translation_objective_respects_valid_only_switch(self):
        all_loss = torch.tensor(1.0)
        valid_loss = torch.tensor(2.0)
        self.assertIs(
            choose_oracle_translation_loss(
                all_loss, valid_loss, valid_only=False,
            ),
            all_loss,
        )
        self.assertIs(
            choose_oracle_translation_loss(
                all_loss, valid_loss, valid_only=True,
            ),
            valid_loss,
        )

    def test_strict_selection_rejects_ambiguous_gt(self):
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            select_active_instance_points(
                torch.zeros(1, 2, 4, 3),
                torch.ones(1, 2, dtype=torch.bool),
                torch.tensor([[1, 1]]),
                active_role='target',
                strict=True,
            )

    def test_non_strict_ambiguous_gt_disables_adapter_residual(self):
        _, selected_valid, slots = select_active_instance_points(
            torch.zeros(1, 2, 4, 3),
            torch.ones(1, 2, dtype=torch.bool),
            torch.tensor([[1, 1]]),
            active_role='target',
            strict=False,
        )
        self.assertEqual(selected_valid.tolist(), [False])
        self.assertEqual(slots.tolist(), [-1])

    def test_rasterization_uses_full_instance_not_only_center(self):
        projected = torch.tensor(
            [[[[1.0, 1.0]], [[3.0, 3.0]]]]
        )
        prior = rasterize_instance_points(
            projected, torch.tensor([True]), (5, 5), sigma=0.0,
        )
        self.assertEqual(tuple(prior.shape), (1, 1, 5, 5))
        self.assertEqual(prior[0, 0, 1, 1].item(), 1.0)
        self.assertEqual(prior[0, 0, 3, 3].item(), 1.0)
        self.assertEqual(prior[0, 0, 2, 2].item(), 0.0)

    def test_training_visualization_payload_splits_processed_stage_gt(self):
        batch_size, views, height, width = 1, 3, 4, 4
        stage_one = {
            'trans': torch.randn(batch_size, views, height, width),
            'oracle_instance_prior': torch.rand(
                batch_size, views, height, width
            ),
            'oracle_target_prior': torch.rand(
                batch_size, views, height, width
            ),
            'oracle_reference_prior': torch.rand(
                batch_size, views, height, width
            ),
            'oracle_relation_anchor': torch.rand(
                batch_size, views, 2, 2
            ),
        }
        stage_two = {
            'trans': torch.randn(batch_size, views, height, width),
            'oracle_instance_prior': torch.rand(
                batch_size, views, height, width
            ),
        }
        output = {
            **stage_one,
            'mvt1_ori_img': torch.rand(
                batch_size, views, 7, height, width
            ),
            'mvt2': stage_two,
            'mvt2_ori_img': torch.rand(
                batch_size, views, 7, height, width
            ),
        }
        processed_gt = torch.arange(
            batch_size * height * width * views * 2,
            dtype=torch.float32,
        ).reshape(batch_size, height * width, views * 2)
        payload = build_training_visualization_payload(
            output,
            processed_gt,
            num_views=views,
            height=height,
            width=width,
            stage_two=True,
        )
        self.assertEqual(tuple(payload), ('mvt1', 'mvt2'))
        self.assertEqual(payload['mvt1']['gt'].shape, (views, height, width))
        self.assertEqual(payload['mvt2']['gt'].shape, (views, height, width))
        self.assertEqual(
            payload['mvt1']['target_prior'].shape,
            (views, height, width),
        )
        self.assertEqual(
            payload['mvt1']['reference_prior'].shape,
            (views, height, width),
        )
        self.assertEqual(
            payload['mvt1']['relation_anchor'].shape,
            (views, height, width),
        )
        self.assertTrue(
            torch.allclose(
                payload['mvt1']['pred'].sum(dim=(-2, -1)),
                torch.ones(views),
            )
        )


if __name__ == '__main__':
    unittest.main()
