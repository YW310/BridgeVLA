"""CPU regressions for direct T/R query supervision and inherited refine."""

import ast
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

from finetune.bridgevla.models.object_conditioning import (
    hungarian_role_slot_losses, mixed_role_map_losses, reference_null_loss,
    role_supervision_mask,
)
from finetune.bridgevla.models.role_separation import role_separation_loss


ROOT = Path(__file__).resolve().parents[1]


def _actual_agent_loss():
    """Exercise the production method without importing RLBench/CUDA modules."""
    path = ROOT / 'finetune/bridgevla/models/bridgevla_agent.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    agent = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == 'RVTAgent')
    method = next(node for node in agent.body
                  if isinstance(node, ast.FunctionDef)
                  and node.name == '_object_slot_auxiliary_losses')
    scope = dict(torch=torch, F=F,
                 hungarian_role_slot_losses=hungarian_role_slot_losses,
                 mixed_role_map_losses=mixed_role_map_losses,
                 reference_null_loss=reference_null_loss,
                 role_supervision_mask=role_supervision_mask)
    scope['role_separation_loss'] = role_separation_loss
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
    return scope[method.name]


def _agent(*, stage_two=False, mixed=False):
    agent = SimpleNamespace(
        internal_object_slots_enabled=True, stage_two=stage_two,
        _net_mod=SimpleNamespace(object_conditioning_supervise_mixed_role_maps=mixed),
    )
    agent.losses = MethodType(_actual_agent_loss(), agent)
    return agent


def _direct_stage():
    prior = torch.full((1, 2, 2, 2, 2), 0.5, requires_grad=True)
    teacher = torch.zeros_like(prior)
    teacher[0, 0, 0, 0, 0] = 1.0
    teacher[0, 0, 1, 1, 1] = 1.0
    teacher[0, 1, 0, 0, 1] = 1.0
    return {
        'object_slot_predictor_type': 'role_queries',
        'object_slot_prior': prior,
        'object_slot_target_prior': teacher.detach().clone(),
        'object_slot_target_valid': torch.tensor([[True, True]]),
        'object_slot_reference_null_probability': torch.tensor([0.5], requires_grad=True),
    }


class DirectRoleLossTest(unittest.TestCase):
    @staticmethod
    def _logit_prior(logits, null):
        masks = logits.sigmoid()
        return torch.stack((masks[:, :, 0],
                            masks[:, :, 1] * (-null).sigmoid()[:, None, None, None]), dim=2)

    def test_logit_loss_matches_emitted_map_bce_dice_and_gradients(self):
        torch.manual_seed(14)
        logits = torch.randn(2, 3, 2, 4, 4, requires_grad=True)
        null = torch.randn(2, requires_grad=True)
        prior = self._logit_prior(logits, null)
        teacher = torch.rand_like(prior)
        valid = torch.ones(2, 2).bool()
        stable = mixed_role_map_losses(prior, teacher, valid,
                                       conditional_logits=logits, reference_null_logit=null)
        original = mixed_role_map_losses(prior, teacher, valid)
        for key in stable:
            torch.testing.assert_close(stable[key], original[key])
        first = torch.autograd.grad(stable['mask'], (logits, null), retain_graph=True)
        second = torch.autograd.grad(original['mask'], (logits, null))
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)

    def test_wrong_saturated_masks_retain_positive_and_negative_correction_gradients(self):
        for value in (-30., 30.):
            with self.subTest(logit=value):
                logits = torch.full((1, 1, 2, 2, 2), value, requires_grad=True)
                null = torch.tensor([-30.], requires_grad=True)
                prior = self._logit_prior(logits, null)
                teacher = torch.zeros_like(prior)
                teacher[0, 0, 0, 0, 0] = 1.
                teacher[0, 0, 1, 1, 1] = 1.
                losses = mixed_role_map_losses(prior, teacher, torch.ones(1, 2).bool(),
                                               conditional_logits=logits, reference_null_logit=null)
                losses['mask'].backward()
                self.assertTrue(torch.isfinite(losses['mask']))
                if value < 0:
                    self.assertLess(logits.grad[0, 0, 0, 0, 0].item(), -.01)
                    self.assertLess(logits.grad[0, 0, 1, 1, 1].item(), -.01)
                else:
                    self.assertGreater(logits.grad[0, 0, 0, 0, 1].item(), .01)
                    self.assertGreater(logits.grad[0, 0, 1, 0, 1].item(), .01)

    def test_actual_agent_routes_raw_logits_and_recovers_wrong_null_at_saturation(self):
        stage = _direct_stage()
        logits = torch.full_like(stage['object_slot_prior'], -30., requires_grad=True)
        null = torch.tensor([30.], requires_grad=True)
        stage.update(object_slot_mask_logits=logits,
                     object_slot_reference_null_logit=null,
                     object_slot_reference_null_probability=null.sigmoid(),
                     object_slot_prior=self._logit_prior(logits, null))
        losses = _agent().losses(stage, torch.ones(1, 2).bool(),
                                  torch.ones(1, 2).bool(), torch.ones(1, 2).bool())
        (losses['mask'] + losses['presence']).backward()
        self.assertLess(logits.grad[0, 0, 0, 0, 0].item(), -.01)
        self.assertLess(logits.grad[0, 0, 1, 1, 1].item(), -.01)
        self.assertGreater(null.grad.item(), .1)

    def test_saturated_null_absence_label_has_gradient_without_spatial_teacher(self):
        null = torch.tensor([-30.], requires_grad=True)
        loss = reference_null_loss(null.sigmoid(), torch.tensor([[True, False]]),
                                   torch.ones(1, 2).bool(), logits=null)
        loss.backward()
        self.assertLess(null.grad.item(), -.9)

    def test_stable_map_loss_keeps_empty_views_and_unknown_roles_unsupervised(self):
        logits = torch.full((1, 2, 2, 2, 2), -30., requires_grad=True)
        null = torch.tensor([30.], requires_grad=True)
        prior = self._logit_prior(logits, null)
        teacher = torch.zeros_like(prior)
        teacher[:, 0, 0, 0, 0] = 1.
        loss = mixed_role_map_losses(prior, teacher, torch.tensor([[True, False]]),
                                     conditional_logits=logits, reference_null_logit=null)['mask']
        loss.backward()
        self.assertEqual(logits.grad[:, 1].count_nonzero().item(), 0)
        self.assertEqual(logits.grad[:, :, 1].count_nonzero().item(), 0)
        self.assertEqual(null.grad.count_nonzero().item(), 0)

    def test_direct_map_and_actual_null_posterior_have_gradients_without_slots(self):
        stage = _direct_stage()
        present = torch.tensor([[True, False]])
        known = torch.ones(1, 2).bool()
        losses = _agent().losses(stage, torch.ones(1, 2).bool(), present, known)
        self.assertEqual(set(losses), {'mask', 'mask_bce', 'mask_dice',
                                       'role', 'objectness', 'presence', 'diversity', 'diversity_pixels',
                                       'diversity_visibility_available', 'diversity_teacher_role_views',
                                       'diversity_visible_role_views', 'diversity_paired_views',
                                       'diversity_overlap_pixels'})
        for name in ('role', 'objectness', 'diversity'):
            self.assertEqual(losses[name].item(), 0)
            self.assertTrue(losses[name].requires_grad)
        self.assertGreater(losses['mask_bce'].item(), 0)
        (losses['mask'] + losses['presence']).backward()
        self.assertGreater(stage['object_slot_prior'].grad[0, 0, 0].abs().sum().item(), 0)
        self.assertEqual(stage['object_slot_prior'].grad[0, :, 1].count_nonzero().item(), 0)
        self.assertGreater(stage['object_slot_prior'].grad[0, 1, 0].abs().sum().item(), 0)
        self.assertLess(stage['object_slot_reference_null_probability'].grad.item(), 0)

    def test_present_reference_supervised_without_null_pseudo_label(self):
        stage = _direct_stage()
        losses = _agent().losses(
            stage, torch.ones(1, 2).bool(),
            torch.tensor([[True, True]]), torch.ones(1, 2).bool(),
        )
        (losses['mask'] + losses['presence']).backward()
        self.assertGreater(stage['object_slot_prior'].grad[0, 0, 1].abs().sum().item(), 0)
        self.assertGreater(stage['object_slot_reference_null_probability'].grad.item(), 0)

    def test_unknown_and_invalid_geometry_do_not_train_empty_maps_or_null(self):
        stage = _direct_stage()
        stage['object_slot_target_valid'] = torch.tensor([[False, False]])
        losses = _agent().losses(
            stage, torch.ones(1, 2).bool(),
            torch.zeros(1, 2).bool(), torch.zeros(1, 2).bool(),
        )
        for name in losses:
            self.assertEqual(losses[name].item(), 0)
        (losses['mask'] + losses['presence']).backward()
        self.assertEqual(stage['object_slot_prior'].grad.count_nonzero().item(), 0)
        self.assertEqual(stage['object_slot_reference_null_probability'].grad.count_nonzero().item(), 0)

    def test_replay_invalidity_intersects_stage_support(self):
        stage = _direct_stage()
        losses = _agent().losses(stage, torch.tensor([[False, True]]))
        losses['mask'].backward()
        self.assertEqual(stage['object_slot_prior'].grad[:, :, 0].count_nonzero().item(), 0)
        self.assertGreater(stage['object_slot_prior'].grad[0, 0, 1].abs().sum().item(), 0)

    def test_independent_stages_average_direct_maps(self):
        coarse, refine = _direct_stage(), _direct_stage()
        coarse['mvt2'] = refine
        valid = torch.ones(1, 2).bool()
        losses = _agent(stage_two=True).losses(coarse, valid)
        expected = (
            mixed_role_map_losses(coarse['object_slot_prior'],
                                  coarse['object_slot_target_prior'], valid)['mask']
            + mixed_role_map_losses(refine['object_slot_prior'],
                                    refine['object_slot_target_prior'], valid)['mask']
        ) / 2
        torch.testing.assert_close(losses['mask'], expected)
        losses['mask'].backward()
        self.assertGreater(coarse['object_slot_prior'].grad.abs().sum().item(), 0)
        self.assertGreater(refine['object_slot_prior'].grad.abs().sum().item(), 0)

    def test_inherited_refine_does_not_double_count_coarse_direct_loss(self):
        coarse = _direct_stage()
        valid = torch.ones(1, 2).bool()
        expected = _agent().losses(coarse, valid)
        coarse['mvt2'] = {'object_slot_roles_inherited': True,
                          'object_slot_predictor_type': 'role_queries'}
        actual = _agent(stage_two=True).losses(coarse, valid)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])

    def test_mixed_slot_ablation_conflict_and_missing_direct_outputs_fail_closed(self):
        stage = _direct_stage()
        with self.assertRaisesRegex(ValueError, 'disable supervise_mixed_role_maps'):
            _agent(mixed=True).losses(stage, torch.ones(1, 2).bool())
        del stage['object_slot_reference_null_probability']
        with self.assertRaisesRegex(KeyError, 'reference_null_probability'):
            _agent().losses(stage, torch.ones(1, 2).bool())


if __name__ == '__main__':
    unittest.main()
