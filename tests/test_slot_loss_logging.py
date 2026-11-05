"""Exercise unordered-slot auxiliary losses through the production metric readout."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from finetune.bridgevla.models.oracle_prior import InternalObjectSlotPredictor
from test_direct_role_losses import _agent


ROOT = Path(__file__).resolve().parents[1]


def _production_metrics(losses):
    """Execute the actual update() dictionary that previously called float.item()."""
    tree = ast.parse((ROOT / 'finetune/bridgevla/models/bridgevla_agent.py').read_text(encoding='utf-8'))
    metric = next(node for node in ast.walk(tree) if isinstance(node, ast.Dict)
                  and any(isinstance(key, ast.Constant)
                          and key.value == 'object_slot_diversity_supervised_pixels' for key in node.keys))
    scope = dict(object_slot_losses=losses, self=SimpleNamespace(object_slot_diversity_loss_weight=0.))
    return eval(compile(ast.Expression(body=metric), '<production-slot-metrics>', 'eval'), scope)


class SlotLossLoggingTest(unittest.TestCase):
    def _run(self, soft, stage_two=True, labels='both'):
        torch.manual_seed(12)
        options = dict(feature_channels=4, num_views=3, num_slots=2, slot_dim=8,
                       decoder_layers=1, num_heads=2, point_samples=4)
        coarse = InternalObjectSlotPredictor(**options)
        local = InternalObjectSlotPredictor(**options, soft_coarse_queries=soft)
        modules = [coarse, local] if stage_two else [coarse]
        optimizer = torch.optim.SGD([parameter for module in modules for parameter in module.parameters()], lr=.01)
        features = torch.randn(3, 4, 4, 4)
        xyz = torch.rand(1, 3, 3, 8, 8) + .1
        valid = torch.tensor([[True, labels == 'both']])
        present = valid.clone()
        known = torch.ones_like(valid)
        if labels == 'unknown':
            valid.zero_()
            present.zero_()
            known.zero_()
        elif labels == 'geometry_unavailable':
            valid.zero_()
            present.fill_(True)
        teacher = torch.zeros(1, 3, 2, 4, 4)
        teacher[:, :, 0, :2, :2] = 1
        if labels == 'both':
            teacher[:, :, 1, 2:, 2:] = 1
        if not valid.any():
            teacher.zero_()

        def stage(prediction):
            result = {'object_slot_prior': prediction['prior'], 'object_slot_target_prior': teacher}
            for source, target in (('slot_masks', 'masks'), ('mask_logits', 'mask_logits'),
                                   ('role_logits', 'role_logits'), ('objectness_logits', 'objectness_logits'),
                                   ('reference_null_probability', 'reference_null_probability'),
                                   ('reference_null_logit', 'reference_null_logit')):
                result['object_slot_' + target] = prediction[source]
            return result

        prediction = coarse(features, xyz)
        output = stage(prediction)
        if stage_two:
            hints = (dict(coarse_role_tokens=prediction['role_tokens'],
                          coarse_reference_null_probability=prediction['reference_null_probability'])
                     if soft else {})
            output['mvt2'] = stage(local(features, xyz, **hints))
        losses = _agent(stage_two=stage_two).losses(output, valid, present, known)
        # Match the configured zero diversity weight while retaining mask/NULL gradients.
        total = losses['mask'] + .25 * losses['presence'] + 0. * losses['diversity']
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        metrics = _production_metrics(losses)
        self.assertEqual(metrics['object_slot_diversity_supervised_pixels'], 0.)
        for name, value in losses.items():
            self.assertIsInstance(value, torch.Tensor, name)
            self.assertEqual(value.ndim, 0, name)
            self.assertEqual(value.device, output['object_slot_prior'].device, name)
            self.assertTrue(torch.isfinite(value).item(), name)
        return metrics

    def test_legacy_and_soft_coarse_single_optimizer_step_then_logging(self):
        for soft, stages in ((False, False), (False, True), (True, True)):
            with self.subTest(soft_coarse=soft, stage_two=stages):
                self._run(soft, stages)

    def test_null_and_unknown_geometry_keep_scalar_metric_contract(self):
        for labels in ('target_only', 'unknown', 'geometry_unavailable'):
            with self.subTest(labels=labels):
                self._run(True, labels=labels)


if __name__ == '__main__':
    unittest.main()
