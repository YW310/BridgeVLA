import unittest

import torch

from finetune.bridgevla.models.oracle_prior import (
    OraclePriorFusion,
    build_training_visualization_payload,
    rasterize_instance_points,
    select_active_instance_points,
)


class OraclePriorTest(unittest.TestCase):
    def test_fusion_is_identity_at_initialization(self):
        logits = torch.randn(2, 3, 8, 8)
        prior = torch.rand_like(logits)
        fused = OraclePriorFusion(4)(
            logits, prior, torch.tensor([True, True]),
        )
        torch.testing.assert_close(fused, logits)

    def test_zero_initialized_fusion_still_receives_gradients(self):
        fusion = OraclePriorFusion(4)
        logits = torch.randn(1, 3, 4, 4)
        prior = torch.rand_like(logits)
        loss = fusion(logits, prior, torch.tensor([True])).square().mean()
        loss.backward()
        self.assertGreater(fusion.net[-1].weight.grad.abs().sum().item(), 0)

    def test_invalid_sample_remains_raw_after_fusion_learns(self):
        logits = torch.randn(1, 1, 2, 2)
        prior = torch.rand_like(logits)
        fusion = OraclePriorFusion(4)
        torch.nn.init.ones_(fusion.net[-1].weight)
        fused = fusion(logits, prior, torch.tensor([False]))
        torch.testing.assert_close(fused, logits)

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

    def test_strict_selection_rejects_ambiguous_gt(self):
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            select_active_instance_points(
                torch.zeros(1, 2, 4, 3),
                torch.ones(1, 2, dtype=torch.bool),
                torch.tensor([[1, 1]]),
                active_role='target',
                strict=True,
            )

    def test_non_strict_ambiguous_gt_falls_back_to_raw(self):
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
            'trans_raw': torch.randn(batch_size, views, height, width),
            'oracle_instance_prior': torch.rand(
                batch_size, views, height, width
            ),
        }
        stage_two = {
            'trans': torch.randn(batch_size, views, height, width),
            'trans_raw': torch.randn(batch_size, views, height, width),
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
        self.assertTrue(
            torch.allclose(
                payload['mvt1']['pred'].sum(dim=(-2, -1)),
                torch.ones(views),
            )
        )


if __name__ == '__main__':
    unittest.main()
