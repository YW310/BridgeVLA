"""CPU forward smoke tests with a tiny stand-in for the pretrained VLM."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn


FINETUNE = Path(__file__).resolve().parents[1] / 'finetune'
if str(FINETUNE) not in sys.path:
    sys.path.insert(0, str(FINETUNE))
from bridgevla.mvt import mvt_single  # noqa: E402
from bridgevla.models.oracle_prior import (  # noqa: E402
    InternalObjectSlotPredictor, OracleRelationAnchorFeatureAdapter,
)
from bridgevla.models.object_conditioning import (  # noqa: E402
    active_semantic_target_mask,
    pending_target_candidate_mask,
    select_object_candidate_from_waypoint,
)


class ObjectConditioningForwardTest(unittest.TestCase):
    def test_waypoint_attributes_to_nearest_valid_object(self):
        waypoint = torch.tensor([[0.9, 0.0, 0.0]])
        candidates = torch.tensor([[
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.1, 0.0, 0.0]],
            [[0.91, 0.0, 0.0], [0.92, 0.0, 0.0]],
        ]])
        valid = torch.tensor([[True, True, False]])
        selected, distance, confidence = select_object_candidate_from_waypoint(
            waypoint, candidates, valid)
        self.assertEqual(selected.item(), 1)
        torch.testing.assert_close(distance, torch.tensor([0.1]))
        self.assertGreater(confidence.item(), 0.45)

    def test_waypoint_attribution_rejects_far_free_space(self):
        selected, distance, confidence = select_object_candidate_from_waypoint(
            torch.tensor([[0.5, 0.0, 0.0]]),
            torch.zeros(1, 1, 4, 3),
            torch.ones(1, 1, dtype=torch.bool),
        )
        self.assertEqual(selected.item(), -1)
        torch.testing.assert_close(distance, torch.tensor([0.5]))
        torch.testing.assert_close(confidence, torch.zeros(1))

    def test_waypoint_attribution_returns_unknown_without_valid_candidate(self):
        selected, distance, confidence = select_object_candidate_from_waypoint(
            torch.zeros(1, 3), torch.zeros(1, 2, 4, 3),
            torch.zeros(1, 2, dtype=torch.bool),
        )
        self.assertEqual(selected.item(), -1)
        self.assertTrue(torch.isinf(distance).all())
        torch.testing.assert_close(confidence, torch.zeros(1))

    def test_inactive_configured_candidate_is_not_a_semantic_target(self):
        valid = torch.tensor([[True, True, False]])
        phases = torch.tensor([[0, -1, 1]])
        eligible = active_semantic_target_mask(valid, phases)
        torch.testing.assert_close(
            eligible, torch.tensor([[True, False, False]]))

        waypoint = torch.tensor([[0.9, 0.0, 0.0]])
        candidates = torch.tensor([[
            [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]],
            [[0.9, 0.0, 0.0], [1.0, 0.0, 0.0]],
            [[0.8, 0.0, 0.0], [0.9, 0.0, 0.0]],
        ]])
        anchor, _, _ = select_object_candidate_from_waypoint(
            waypoint, candidates, valid)
        target, _, _ = select_object_candidate_from_waypoint(
            waypoint, candidates, eligible)
        self.assertEqual(anchor.item(), 1)
        self.assertEqual(target.item(), -1)

    def test_completed_ordered_target_is_suppressed_only_after_release(self):
        valid = torch.tensor([[True, True, True, True]])
        phases = torch.tensor([[0, 1, 2, -1]])
        current = torch.tensor([1])

        held = pending_target_candidate_mask(
            valid, phases, current, torch.tensor([False]))
        released = pending_target_candidate_mask(
            valid, phases, current, torch.tensor([True]))

        torch.testing.assert_close(held, valid)
        torch.testing.assert_close(
            released, torch.tensor([[False, True, True, True]]))

    def test_pending_target_mask_keeps_all_candidates_without_current_phase(self):
        valid = torch.tensor([[True, False, True]])
        pending = pending_target_candidate_mask(
            valid,
            torch.tensor([[0, 1, -1]]),
            torch.tensor([-1]),
            torch.tensor([True]),
        )
        torch.testing.assert_close(pending, valid)

    def _small_policy(self):
        module = mvt_single.MVT.__new__(mvt_single.MVT)
        nn.Module.__init__(module)
        module.num_img, module.img_size = 3, 16
        module.num_pat_img, module.vlm_dim = 16, 4
        module.img_patch_size, module.wpt_img_aug = 1, 0.
        module.use_gpu_paligemma_preprocessing = True
        module.use_efficient_paligemma_forward = True
        module.processor = SimpleNamespace(tokenizer=SimpleNamespace(all_special_ids=[0, 9]))
        module.model = SimpleNamespace(config=SimpleNamespace(image_token_index=7))
        module.hidden = nn.Parameter(torch.randn(1, 770, 4))
        module._prepare_paligemma_inputs_gpu = lambda prompts, images: {
            'attention_mask': torch.ones(1, 770),
            'input_ids': torch.tensor([[7] * 768 + [3, 9]]),
        }
        module._forward_efficient_paligemma = lambda inputs: module.hidden
        module.up0 = nn.Conv2d(4, 1, 1)
        seen = {}

        def decode(out, dyn_cam_info):
            seen['decoded_trans'] = out['trans'].clone()
            index = out['trans'].mean(dim=1).flatten(1).argmax(dim=1)
            return torch.stack((index % 16, index // 16, torch.zeros_like(index)), dim=1).float()

        def project(points, dyn_cam_info):
            return points[:, :, None, :2].expand(-1, -1, 3, -1)

        module.get_wpt = decode
        module.get_pt_loc_on_img = project
        module._forward_action_heads = lambda feature, rotation, batch: {'feat_ex_rot': feature}
        predictor = InternalObjectSlotPredictor(
            4, 3, num_slots=2, slot_dim=8, decoder_layers=1, num_heads=2,
            point_samples=4, confidence_threshold=0., use_context=True,
            soft_conditioning=True,
        )
        adapter = OracleRelationAnchorFeatureAdapter(
            4, rank=2, anchor_rank=2, use_context=True,
            role_conditioning=True, role_token_dim=8,
        )
        nn.init.normal_(adapter.anchor_expand.weight, std=.1)
        image = torch.rand(1, 3, 6, 16, 16) + .1
        options = dict(img=image, language_goal=[[['move target']]],
                       oracle_relation_state=torch.zeros(1, 3),
                       oracle_feature_adapter=adapter, object_slot_predictor=predictor,
                       object_conditioning_shared_action_features=True,
                       object_conditioning_use_context=True)
        return module, predictor, adapter, options, seen

    @staticmethod
    def _sample(points, feature):
        # The test controls the sampler, not CUDA rendering or camera geometry.
        return (feature.mean(dim=(-2, -1)),)

    def test_prediction_only_and_teacher_changes_leave_policy_outputs_identical(self):
        module, predictor, adapter, options, seen = self._small_policy()
        module.eval()
        predictor.eval()
        teacher = torch.zeros(1, 3, 2, 16, 16)
        with mock.patch.object(mvt_single, 'select_feat_from_hm', self._sample, create=True):
            first = module(**options, object_slot_target_heatmap=teacher)
            second = module(**options, object_slot_target_heatmap=1 - teacher)
            predicted_only = module(**options)
        for key in ('trans', 'feat_ex_rot'):
            torch.testing.assert_close(first[key], second[key])
            torch.testing.assert_close(first[key], predicted_only[key])
        torch.testing.assert_close(seen['decoded_trans'], predicted_only['trans'])
        self.assertNotIn('object_slot_target_prior', predicted_only)

    def test_single_training_forward_backpropagates_action_to_slots_and_context(self):
        module, predictor, adapter, options, seen = self._small_policy()
        module.train()
        options['wpt_local'] = torch.tensor([[2., 2., 0.]])
        with mock.patch.object(mvt_single, 'select_feat_from_hm', self._sample, create=True):
            output = module(**options)
        loss = output['trans'].square().mean() + output['feat_ex_rot'].square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for parameter in (module.hidden, predictor.feature_reduce.weight,
                          adapter.context_projection[-1].weight, adapter.anchor_expand.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_identity_residual_preserves_legacy_global_local_layout(self):
        module, predictor, adapter, options, seen = self._small_policy()
        module.eval()
        predictor.eval()
        nn.init.zeros_(adapter.anchor_expand.weight)
        with mock.patch.object(mvt_single, 'select_feat_from_hm', self._sample, create=True):
            shared = module(**options)
            options['object_conditioning_shared_action_features'] = False
            legacy = module(**options)
        torch.testing.assert_close(shared['trans'], legacy['trans'])
        torch.testing.assert_close(shared['feat_ex_rot'], legacy['feat_ex_rot'])


if __name__ == '__main__':
    unittest.main()
