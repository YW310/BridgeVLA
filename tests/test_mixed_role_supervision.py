"""CPU tests for opt-in policy-map supervision, without simulator imports."""

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


def _agent_loss_method():
    """Load the actual loss method without importing RLBench/CUDA dependencies."""
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
                 role_supervision_mask=role_supervision_mask,
                 reference_null_loss=reference_null_loss)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
    return scope[method.name]


class MixedRoleSupervisionTest(unittest.TestCase):
    def test_actual_soft_mixture_has_mask_and_role_posterior_gradients(self):
        slot_logits = torch.tensor([[[
            [[3., -3.], [-3., -3.]],
            [[-3., -3.], [-3., 3.]],
        ]]], requires_grad=True)
        role_logits = torch.tensor([[[0.1, 0.4], [0.3, 0.2]]], requires_grad=True)
        weights = torch.softmax(role_logits, dim=1)
        prior = torch.einsum('bkr,bvkhw->bvrhw', weights, slot_logits.sigmoid())
        teacher = torch.tensor([[[
            [[1., 0.], [0., 0.]],
            [[0., 0.], [0., 1.]],
        ]]], requires_grad=True)

        losses = mixed_role_map_losses(prior, teacher, torch.ones(1, 2).bool())
        losses['mask'].backward()

        for value in (slot_logits, role_logits):
            self.assertIsNotNone(value.grad)
            self.assertGreater(value.grad.abs().sum().item(), 0)
        self.assertIsNone(teacher.grad)
        self.assertGreater(losses['bce'].item(), 0)
        self.assertGreater(losses['dice'].item(), 0)

    def test_invalid_geometry_and_empty_teacher_views_are_not_negative_labels(self):
        prior = torch.full((1, 2, 2, 2, 2), 0.7, requires_grad=True)
        teacher = torch.zeros_like(prior)
        teacher[0, 0, :, 0, 0] = 1
        teacher[0, 1, 1, 0, 0] = 1
        loss = mixed_role_map_losses(
            prior, teacher, torch.tensor([[True, False]]))['mask']
        loss.backward()

        self.assertGreater(prior.grad[0, 0, 0].abs().sum().item(), 0)
        self.assertEqual(prior.grad[0, 1, 0].count_nonzero().item(), 0)
        self.assertEqual(prior.grad[:, :, 1].count_nonzero().item(), 0)

    def test_null_and_terminal_placeholders_only_use_presence_supervision(self):
        prior = torch.full((2, 1, 2, 2, 2), 0.7, requires_grad=True)
        teacher = torch.ones_like(prior)
        present = torch.tensor([[True, False], [False, False]])
        known = torch.tensor([[True, True], [False, False]])
        losses = mixed_role_map_losses(
            prior, teacher, torch.ones(2, 2).bool(), present, known)
        losses['mask'].backward()

        self.assertGreater(prior.grad[0, :, 0].abs().sum().item(), 0)
        self.assertEqual(prior.grad[0, :, 1].count_nonzero().item(), 0)
        self.assertEqual(prior.grad[1].count_nonzero().item(), 0)
        posterior = torch.tensor([0.5, 0.5], requires_grad=True)
        reference_null_loss(posterior, present, known).backward()
        self.assertLess(posterior.grad[0].item(), 0)
        self.assertEqual(posterior.grad[1].item(), 0)

    def test_missing_presence_metadata_keeps_valid_geometry_supervision(self):
        prior = torch.full((1, 1, 2, 2, 2), 0.5, requires_grad=True)
        teacher = torch.ones_like(prior)
        loss = mixed_role_map_losses(
            prior, teacher, torch.tensor([[True, False]]))['mask']
        loss.backward()
        self.assertGreater(prior.grad[:, :, 0].abs().sum().item(), 0)
        self.assertEqual(prior.grad[:, :, 1].count_nonzero().item(), 0)

    def test_all_invalid_is_finite_differentiable_zero(self):
        for teacher, valid in (
            (torch.ones(1, 1, 2, 2, 2), torch.zeros(1, 2).bool()),
            (torch.zeros(1, 1, 2, 2, 2), torch.ones(1, 2).bool()),
        ):
            prior = torch.rand(1, 1, 2, 2, 2, requires_grad=True)
            losses = mixed_role_map_losses(prior, teacher, valid)
            self.assertTrue(all(torch.isfinite(value).item() for value in losses.values()))
            self.assertEqual(losses['mask'].item(), 0)
            losses['mask'].backward()
            self.assertEqual(prior.grad.count_nonzero().item(), 0)

    def test_teacher_resize_preserves_probability_space_supervision(self):
        prior = torch.full((1, 1, 2, 2, 2), 0.5, requires_grad=True)
        teacher = torch.zeros(1, 1, 2, 4, 4)
        teacher[..., :2, :2] = 1
        losses = mixed_role_map_losses(prior, teacher, torch.ones(1, 2).bool())
        resized = F.interpolate(teacher.flatten(0, 1), (2, 2), mode='area').unsqueeze(0)
        expected = mixed_role_map_losses(prior, resized, torch.ones(1, 2).bool())
        torch.testing.assert_close(losses['mask'], expected['mask'])

    def test_view_mask_preserves_legacy_matching_when_all_views_are_supported(self):
        stage = self._stage()
        args = (stage['object_slot_mask_logits'], stage['object_slot_role_logits'],
                stage['object_slot_objectness_logits'], stage['object_slot_target_prior'],
                torch.ones(1, 2).bool())
        legacy = hungarian_role_slot_losses(*args)
        supported = hungarian_role_slot_losses(
            *args, role_view_valid=torch.ones(1, 1, 2).bool())
        for key in legacy:
            torch.testing.assert_close(supported[key], legacy[key])

    def test_unsupported_views_do_not_affect_assignment_or_mask_gradients(self):
        stage = self._stage()
        teacher = stage['object_slot_target_prior'].repeat(1, 2, 1, 1, 1)
        teacher[:, 1] = torch.nan
        teacher[:, 0, 1] = 0
        logits = stage['object_slot_mask_logits'].detach().repeat(1, 2, 1, 1, 1)
        logits[:, 1] = 100
        logits.requires_grad_()
        valid = torch.ones(1, 2).bool()
        support = role_supervision_mask(teacher, valid)
        losses = hungarian_role_slot_losses(
            logits, stage['object_slot_role_logits'], stage['object_slot_objectness_logits'],
            teacher, valid, role_view_valid=support)
        torch.testing.assert_close(losses['assignments'], torch.tensor([[0, -1]]))
        self.assertTrue(torch.isfinite(losses['mask']))
        (losses['mask'] + losses['objectness']).backward()
        self.assertGreater(logits.grad[:, 0, 0].abs().sum().item(), 0)
        self.assertEqual(logits.grad[:, 1].count_nonzero().item(), 0)
        self.assertEqual(stage['object_slot_objectness_logits'].grad[0, 1].item(), 0)

    def test_no_supported_teacher_returns_no_assignment_and_zero_gradients(self):
        stage = self._stage()
        teacher = torch.zeros_like(stage['object_slot_target_prior'])
        valid = torch.ones(1, 2).bool()
        result = hungarian_role_slot_losses(
            stage['object_slot_mask_logits'], stage['object_slot_role_logits'],
            stage['object_slot_objectness_logits'], teacher, valid,
            role_view_valid=role_supervision_mask(teacher, valid))
        torch.testing.assert_close(result['assignments'], torch.tensor([[-1, -1]]))
        result['mask'].backward()
        for key in ('object_slot_mask_logits', 'object_slot_role_logits',
                    'object_slot_objectness_logits'):
            self.assertEqual(stage[key].grad.count_nonzero().item(), 0)

    @staticmethod
    def _stage():
        logits = torch.tensor([[[
            [[3., -3.], [-3., -3.]],
            [[-3., -3.], [-3., 3.]],
        ]]], requires_grad=True)
        return {
            'object_slot_prior': torch.full((1, 1, 2, 2, 2), 0.5, requires_grad=True),
            'object_slot_target_prior': torch.tensor([[[
                [[1., 0.], [0., 0.]], [[0., 0.], [0., 1.]],
            ]]]),
            'object_slot_target_valid': torch.ones(1, 2).bool(),
            'object_slot_masks': logits.sigmoid(),
            'object_slot_mask_logits': logits,
            'object_slot_objectness_logits': torch.ones(1, 2, requires_grad=True),
            'object_slot_role_logits': torch.zeros(1, 2, 2, requires_grad=True),
            'object_slot_reference_null_probability': torch.tensor([0.5], requires_grad=True),
        }

    @staticmethod
    def _loss_agent(enabled, stage_two=False):
        agent = SimpleNamespace(
            internal_object_slots_enabled=True, stage_two=stage_two,
            _net_mod=SimpleNamespace(object_conditioning_supervise_mixed_role_maps=enabled))
        agent.losses = MethodType(_agent_loss_method(), agent)
        return agent

    def test_switch_off_preserves_legacy_loss_keys_and_values(self):
        stage = self._stage()
        valid = torch.ones(1, 2).bool()
        off = self._loss_agent(False).losses(stage, valid)
        on = self._loss_agent(True).losses(stage, valid)
        self.assertEqual(set(off), {'mask', 'mask_bce', 'mask_dice', 'role',
                                   'objectness', 'presence', 'diversity'})
        for name, value in off.items():
            torch.testing.assert_close(on[name], value)
        self.assertIn('mixed_role', on)
        self.assertGreater(on['mixed_role'].item(), on['mask'].item())

    def test_inherited_refine_skips_raw_heads_and_static_map_supervision(self):
        coarse = self._stage()
        refine = {
            'object_slot_roles_inherited': True,
            'object_slot_prior': torch.full((1, 1, 2, 2, 2), 0.5, requires_grad=True),
            'object_slot_target_prior': coarse['object_slot_target_prior'].clone(),
            'object_slot_target_valid': torch.tensor([[True, False]]),
        }
        valid = torch.ones(1, 2).bool()
        expected_coarse = self._loss_agent(False).losses(coarse, valid)
        coarse['mvt2'] = refine
        losses = self._loss_agent(True, stage_two=True).losses(coarse, valid)
        for name, value in expected_coarse.items():
            torch.testing.assert_close(losses[name], value)
        expected_mixed = mixed_role_map_losses(
            coarse['object_slot_prior'], coarse['object_slot_target_prior'], valid)['mask']
        torch.testing.assert_close(losses['mixed_role'], expected_mixed)
        losses['mixed_role'].backward()
        self.assertGreater(coarse['object_slot_prior'].grad.abs().sum().item(), 0)
        self.assertIsNone(refine['object_slot_prior'].grad)
        # The inherited posterior is not counted as another NULL prediction.
        self.assertIsNone(coarse['object_slot_reference_null_probability'].grad)

    def test_inherited_refine_without_mixed_supervision_keeps_coarse_losses(self):
        coarse = self._stage()
        valid = torch.ones(1, 2).bool()
        expected = self._loss_agent(False).losses(coarse, valid)
        coarse['mvt2'] = {'object_slot_roles_inherited': True}
        actual = self._loss_agent(False, stage_two=True).losses(coarse, valid)
        self.assertEqual(set(actual), set(expected))
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name])

    def test_independent_refine_uses_stage_teacher_validity_for_mixed_loss(self):
        coarse, refine = self._stage(), self._stage()
        refine['object_slot_target_valid'] = torch.tensor([[True, False]])
        coarse['mvt2'] = refine
        losses = self._loss_agent(True, stage_two=True).losses(
            coarse, torch.ones(1, 2).bool())
        losses['mixed_role'].backward()
        self.assertGreater(coarse['object_slot_prior'].grad[:, :, 1].abs().sum().item(), 0)
        self.assertGreater(refine['object_slot_prior'].grad[:, :, 0].abs().sum().item(), 0)
        self.assertEqual(refine['object_slot_prior'].grad[:, :, 1].count_nonzero().item(), 0)

    def test_crop_external_reference_is_skipped_by_both_supervision_paths(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled):
                coarse, refine = self._stage(), self._stage()
                refine['object_slot_target_prior'][:, :, 1] = 0
                refine['object_slot_target_valid'] = torch.tensor([[True, False]])
                coarse['mvt2'] = refine
                losses = self._loss_agent(enabled, stage_two=True).losses(
                    coarse, torch.ones(1, 2).bool())
                loss = losses['mask'] + losses['presence']
                if enabled:
                    loss = loss + losses['mixed_role']
                loss.backward()
                reference_grad = refine['object_slot_objectness_logits'].grad[0, 1].item()
                if enabled:
                    self.assertEqual(reference_grad, 0)
                    self.assertEqual(refine['object_slot_mask_logits'].grad[:, :, 1].count_nonzero().item(), 0)
                    self.assertEqual(refine['object_slot_prior'].grad[:, :, 1].count_nonzero().item(), 0)
                else:
                    self.assertLess(reference_grad, 0)  # historical behavior

    def test_mixed_path_intersects_replay_geometry_validity_with_stage_support(self):
        stage = self._stage()
        losses = self._loss_agent(True).losses(stage, torch.tensor([[True, False]]))
        (losses['mask'] + losses['presence'] + losses['mixed_role']).backward()
        self.assertEqual(stage['object_slot_objectness_logits'].grad[0, 1].item(), 0)
        self.assertEqual(stage['object_slot_prior'].grad[:, :, 1].count_nonzero().item(), 0)

    def test_unknown_presence_skips_both_role_losses_but_keeps_known_null_label(self):
        stage = self._stage()
        losses = self._loss_agent(True).losses(
            stage, torch.ones(1, 2).bool(),
            torch.tensor([[False, False]]), torch.tensor([[False, True]]))
        (losses['mask'] + losses['presence'] + losses['mixed_role']).backward()
        self.assertEqual(stage['object_slot_mask_logits'].grad.count_nonzero().item(), 0)
        self.assertEqual(stage['object_slot_objectness_logits'].grad.count_nonzero().item(), 0)
        self.assertEqual(stage['object_slot_prior'].grad.count_nonzero().item(), 0)
        self.assertLess(stage['object_slot_reference_null_probability'].grad.item(), 0)

    def test_missing_raw_heads_without_inheritance_still_fails_closed(self):
        stage = self._stage()
        del stage['object_slot_mask_logits']
        with self.assertRaisesRegex(KeyError, 'object_slot_mask_logits'):
            self._loss_agent(False).losses(stage, torch.ones(1, 2).bool())


if __name__ == '__main__':
    unittest.main()
