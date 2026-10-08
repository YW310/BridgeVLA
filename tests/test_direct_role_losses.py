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
    def test_direct_map_and_actual_null_posterior_have_gradients_without_slots(self):
        stage = _direct_stage()
        present = torch.tensor([[True, False]])
        known = torch.ones(1, 2).bool()
        losses = _agent().losses(stage, torch.ones(1, 2).bool(), present, known)
        self.assertEqual(set(losses), {'mask', 'mask_bce', 'mask_dice',
                                       'role', 'objectness', 'presence', 'diversity'})
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
