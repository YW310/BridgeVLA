"""Soft coarse hints must not disable independent local object prediction."""

import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

from tests import test_object_conditioning_forward as fixture
from bridgevla.models.oracle_prior import InternalObjectSlotPredictor
from bridgevla.models.soft_coarse_queries import SoftCoarseQueryConditioner
from bridgevla.mvt import mvt_single
from finetune.bridgevla.config import get_cfg_defaults
from test_role_feature_config import _isolated_functions, _model, _model_guard_namespace, _train_namespace


ROOT = Path(__file__).resolve().parents[1]


def predictor(enabled=False):
    return InternalObjectSlotPredictor(
        4, 3, num_slots=2, slot_dim=8, decoder_layers=1, num_heads=2,
        point_samples=4, confidence_threshold=.25, soft_coarse_queries=enabled)


class SoftCoarseQueryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(14)
        self.features = torch.randn(3, 4, 4, 4)
        self.xyz = torch.rand(1, 3, 3, 16, 16) + .1
        self.hints = dict(coarse_role_tokens=torch.randn(1, 2, 8),
                          coarse_reference_null_probability=torch.tensor([.2]))

    def test_zero_init_exactly_preserves_trained_legacy_outputs(self):
        old, new = predictor().eval(), predictor(True).eval()
        incompatible = new.load_state_dict(old.state_dict(), strict=False)
        self.assertFalse(incompatible.unexpected_keys)
        self.assertTrue(all('soft_coarse_conditioner' in key for key in incompatible.missing_keys))
        first = old(self.features, self.xyz)
        second = new(self.features, self.xyz, **self.hints)
        for key in ('mask_logits', 'objectness_logits', 'role_logits', 'prior',
                    'valid', 'points', 'reference_null_probability'):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
        self.assertFalse(any('soft_coarse_conditioner' in key for key in old.state_dict()))

    def test_local_evidence_and_decisions_survive_zero_coarse_tokens_and_null(self):
        local = predictor(True).eval()
        with torch.no_grad():
            local.objectness_head.weight.zero_()
            local.objectness_head.bias.fill_(8.)
            local.reference_null_head.weight.zero_()
            local.reference_null_head.bias.fill_(-8.)
        # Coarse NULL cannot become a local hard decision; no validity field is accepted.
        empty = dict(coarse_role_tokens=torch.zeros(1, 2, 8),
                     coarse_reference_null_probability=torch.ones(1))
        out = local(self.features, self.xyz, **empty)
        self.assertTrue(out['valid'].all())
        self.assertFalse(out['reference_is_null'].any())
        self.assertGreater(out['prior'].max().item(), .1)
        other = local(-self.features, self.xyz, **empty)
        self.assertGreater((out['mask_logits'] - other['mask_logits']).abs().sum().item(), 0)

    def test_hint_changes_are_soft_and_learnable_after_zero_init(self):
        local = predictor(True)
        tokens = self.hints['coarse_role_tokens'].clone().requires_grad_()
        null = self.hints['coarse_reference_null_probability'].clone().requires_grad_()
        optimizer = torch.optim.SGD(local.parameters(), lr=.03)
        first = local(self.features, self.xyz, coarse_role_tokens=tokens,
                      coarse_reference_null_probability=null)
        first['mask_logits'].square().mean().backward()
        self.assertGreater(local.soft_coarse_conditioner.output.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        tokens.grad = null.grad = None
        output = local(self.features, self.xyz, coarse_role_tokens=tokens,
                       coarse_reference_null_probability=null)
        output['mask_logits'].square().mean().backward()
        for value in (tokens.grad, null.grad, local.feature_reduce.weight.grad,
                      local.soft_coarse_conditioner.key.weight.grad):
            self.assertTrue(torch.isfinite(value).all())
            self.assertGreater(value.abs().sum().item(), 0)

    def test_coarse_reference_value_is_softly_suppressed_at_null(self):
        module = SoftCoarseQueryConditioner(8)
        nn.init.eye_(module.output.weight)
        queries = torch.randn(1, 2, 8)
        tokens = torch.randn(1, 2, 8, requires_grad=True)
        result = module(queries, tokens, torch.ones(1))
        result.square().mean().backward()
        # NULL is a value weight, not a hard attention mask or a local absence label.
        self.assertTrue(torch.isfinite(tokens.grad).all())
        self.assertGreater(tokens.grad[:, 0].abs().sum().item(), 0)

    def test_missing_context_fails_instead_of_silently_disabling_conditioning(self):
        with self.assertRaisesRegex(ValueError, 'require predicted coarse tokens'):
            predictor(True)(self.features, self.xyz)
        with self.assertRaisesRegex(ValueError, 'soft_coarse_queries=True'):
            predictor()(self.features, self.xyz, **self.hints)

    def test_policy_forward_teacher_isolation_and_action_gradient(self):
        module, _, adapter, options, decoded = fixture.ObjectConditioningForwardTest()._small_policy()
        local = InternalObjectSlotPredictor(
            4, 3, num_slots=2, slot_dim=8, decoder_layers=1, num_heads=2,
            point_samples=4, confidence_threshold=0., use_context=True,
            soft_conditioning=True, soft_coarse_queries=True)
        coarse = predictor()
        coarse_output = coarse(self.features, self.xyz)
        coarse_output['role_tokens'].retain_grad()
        with torch.no_grad():
            local.soft_coarse_conditioner.output.weight.normal_(std=.1)
            adapter.feature_expand.weight.normal_(std=.1)
            adapter.anchor_expand.weight.normal_(std=.1)
            adapter.role_token_projection.weight.normal_(std=.1)
        options.update(object_slot_predictor=local, coarse_query_context={
            'coarse_role_tokens': coarse_output['role_tokens'],
            'coarse_reference_null_probability': coarse_output['reference_null_probability']})
        teacher = torch.zeros(1, 3, 2, 16, 16)
        module.eval()
        with mock.patch.object(mvt_single, 'select_feat_from_hm',
                               fixture.ObjectConditioningForwardTest._sample, create=True):
            first = module(**options, object_slot_target_heatmap=teacher)
            second = module(**options, object_slot_target_heatmap=1 - teacher)
            only = module(**options)
        for key in ('trans', 'feat_ex_rot'):
            torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
            torch.testing.assert_close(first[key], only[key], rtol=0, atol=0)
        self.assertNotIn('object_slot_target_prior', only)
        self.assertNotIn('object_slot_roles_inherited', only)
        torch.testing.assert_close(decoded['decoded_trans'], only['trans'])
        (only['trans'].square().mean() + only['feat_ex_rot'].square().mean()).backward()
        for parameter in (local.slot_queries, local.soft_coarse_conditioner.output.weight,
                          coarse.slot_queries):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)


class SoftCoarseConfigTest(unittest.TestCase):
    def test_real_constructor_installs_conditioner_only_on_local_predictor(self):
        tree = ast.parse((ROOT / 'finetune/bridgevla/mvt/mvt.py').read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MVT')
        init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
        targets = {'slot_kwargs', 'predictor_class', 'self.object_slot_predictor1',
                   'self.object_slot_predictor2'}
        assignments = [node for node in init.body if isinstance(node, ast.Assign)
                       and ast.unparse(node.targets[0]) in targets]
        self.assertEqual(len(assignments), 4)
        for enabled in (False, True):
            model = SimpleNamespace(mvt1=SimpleNamespace(vlm_dim=4), num_img=3,
                object_slots_enabled=True, object_conditioning_use_context=False,
                object_conditioning_shared_action_features=False,
                object_conditioning_soft_coarse_queries=enabled,
                object_conditioning_inherit_coarse_roles=False)
            scope = dict(self=model, stage_two=True, object_slot_predictor_type='slots',
                object_slot_num_slots=2, object_slot_dim=8, object_slot_decoder_layers=1,
                object_slot_num_heads=2, object_slot_point_samples=4,
                object_slot_confidence_threshold=.25,
                InternalObjectSlotPredictor=InternalObjectSlotPredictor)
            exec(compile(ast.Module(body=assignments, type_ignores=[]), '<predictor-init>', 'exec'), scope)
            self.assertFalse(hasattr(model.object_slot_predictor1, 'soft_coarse_conditioner'))
            self.assertEqual(hasattr(model.object_slot_predictor2, 'soft_coarse_conditioner'), enabled)

    def test_opt_in_profile_retains_legacy_prediction_and_losses(self):
        cfg = get_cfg_defaults()
        self.assertFalse(cfg.object_conditioning.soft_coarse_queries)
        cfg.merge_from_file(str(ROOT / 'finetune/RLBench/configs/rlbench_o2_internal_slots_soft_coarse.yaml'))
        self.assertTrue(cfg.object_conditioning.soft_coarse_queries)
        for flag in ('inherit_coarse_roles', 'refine_role_masks', 'refine_heatmap_residual',
                     'refine_local_query', 'shared_action_features', 'use_context'):
            self.assertFalse(getattr(cfg.object_conditioning, flag))
        self.assertEqual(cfg.object_slots.predictor_type, 'slots')
        self.assertEqual(cfg.object_slots.num_slots, 2)
        self.assertEqual(cfg.rvt.object_prediction_confidence_threshold, .25)
        self.assertEqual(cfg.rvt.object_slot_mask_loss_weight, 1.)
        _model_guard_namespace(object_conditioning_soft_coarse_queries=True)
        for kwargs in ({'object_slots_enabled': False}, {'stage_two': False},
                       {'object_slot_predictor_type': 'role_queries'}):
            with self.assertRaisesRegex(ValueError, 'requires independent unordered slots'):
                _model_guard_namespace(object_conditioning_soft_coarse_queries=True, **kwargs)
        with self.assertRaisesRegex(ValueError, 'conflicts with hard role inheritance'):
            _model_guard_namespace(object_conditioning_soft_coarse_queries=True,
                                   object_conditioning_inherit_coarse_roles=True)

    def test_train_eval_and_saved_metadata_track_soft_route(self):
        for entry in ('train', 'eval'):
            tree = ast.parse((ROOT / f'finetune/RLBench/{entry}.py').read_text(encoding='utf-8'))
            call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name) and node.func.id == 'MVT')
            args = {item.arg: ast.unparse(item.value) for item in call.keywords}
            self.assertEqual(args['object_conditioning_soft_coarse_queries'],
                             'exp_cfg.object_conditioning.soft_coarse_queries')
        model = _model(soft_coarse_queries=True)
        scope, saved = _train_namespace({})
        scope['save_agent'](SimpleNamespace(_network=model), 'new.pth', 0)
        self.assertTrue(saved[0][0]['object_conditioning']['soft_coarse_queries'])
        scope, _ = _train_namespace({'epoch': 0, 'model_state': {}})
        with self.assertRaisesRegex(RuntimeError, '--init_checkpoint'):
            scope['load_training_checkpoint'](SimpleNamespace(_network=model), 'old.pth')
        validate = _isolated_functions('finetune/RLBench/eval.py',
            {'_validate_role_feature_checkpoint'}, {})['_validate_role_feature_checkpoint']
        with self.assertRaisesRegex(RuntimeError, 'soft_coarse_queries'):
            validate({}, SimpleNamespace(soft_coarse_queries=True), 'old.pth')
        validate({'object_conditioning': {'soft_coarse_queries': True}},
                 SimpleNamespace(soft_coarse_queries=True), 'new.pth')
        validate({}, SimpleNamespace(), 'legacy.pth')

    def test_old_unordered_checkpoint_initializes_both_predictors_without_reset(self):
        class TinyPolicy(nn.Module):
            def __init__(self, enabled):
                super().__init__()
                self.backbone = nn.Linear(2, 2)
                self.object_slot_predictor1 = predictor()
                self.object_slot_predictor2 = predictor(enabled)
                self.object_slot_predictor_type = 'slots'
        old, new = TinyPolicy(False), TinyPolicy(True)
        scope, _ = _train_namespace({'model_state': old.state_dict(), 'object_predictor_type': 'slots'})
        with redirect_stdout(io.StringIO()):
            scope['load_initial_model_checkpoint'](SimpleNamespace(_network=new), 'old.pth')
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(new.object_slot_predictor2.soft_coarse_conditioner.output.weight.count_nonzero(), 0)

    def test_real_two_stage_forward_routes_only_soft_predictions_in_train_and_eval(self):
        tree = ast.parse((ROOT / 'finetune/bridgevla/mvt/mvt.py').read_text(encoding='utf-8'))
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MVT')
        forward = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')

        def transform(points, loc, sca):
            result = ([sca * (value - loc[i]) for i, value in enumerate(points)]
                      if isinstance(points, list) else sca * (points - loc))
            return result, lambda value: value / sca + loc

        scope = {'torch': torch, 'mvt_utils': SimpleNamespace(
            trans_pc=transform, add_uni_noi=lambda value, amount: value)}
        exec(compile(ast.Module(body=[forward], type_ignores=[]), '<two-stage-forward>', 'exec'), scope)
        for training in (True, False):
            with self.subTest(training=training):
                coarse, local = predictor(), predictor(True)
                with torch.no_grad():
                    coarse.objectness_head.weight.zero_()
                    coarse.objectness_head.bias.fill_(-12.)
                    local.objectness_head.weight.zero_()
                    local.objectness_head.bias.fill_(8.)
                    local.reference_null_head.weight.zero_()
                    local.reference_null_head.bias.fill_(-8.)
                calls = []

                def stage(**kwargs):
                    calls.append(kwargs)
                    features = torch.randn(3, 4, 4, 4)
                    result = kwargs['object_slot_predictor'](
                        features, kwargs['img'][:, :, :3],
                        **(kwargs.get('coarse_query_context') or {}))
                    return {'object_slot_role_tokens': result['role_tokens'],
                            'object_slot_reference_null_probability': result['reference_null_probability'],
                            'object_slot_valid': result['valid'], 'trans': result['mask_logits']}

                model = SimpleNamespace(
                    training=training, img_aug_2=0., stage_two=True, st_sca=2.,
                    st_wpt_loc_aug=0., st_wpt_loc_inp_no_noise=True,
                    object_slots_enabled=True, object_slot_predictor_type='slots',
                    object_conditioning_shared_action_features=False,
                    object_conditioning_use_context=False, object_conditioning_inherit_coarse_roles=False,
                    object_conditioning_soft_coarse_queries=True, object_conditioning_refine_role_masks=False,
                    object_conditioning_refine_mask_detach=False, oracle_adapter_translation_only=False,
                    oracle_relation_gated_adapter=False, oracle_prior_feature_adapter1=None,
                    oracle_prior_feature_adapter2=None, refine_role_mask_head=None,
                    object_slot_predictor1=coarse, object_slot_predictor2=local,
                    mvt1=stage, verify_inp=lambda **kwargs: None,
                    render=lambda **kwargs: torch.full((1, 3, 6, 16, 16), .2),
                    _build_oracle_instance_prior=lambda *args: None,
                    get_wpt=lambda *args, **kwargs: torch.zeros(1, 3))
                output = scope['forward'](model, [torch.rand(5, 3)], [torch.rand(5, 3)],
                                          wpt_local=torch.zeros(1, 3) if training else None)
                self.assertEqual(len(calls), 2)
                self.assertFalse(output['object_slot_valid'].any())
                self.assertTrue(output['mvt2']['object_slot_valid'].all())
                self.assertIs(calls[1]['object_slot_predictor'], local)
                self.assertIsNone(calls[1]['inherited_object_roles'])
                context = calls[1]['coarse_query_context']
                self.assertEqual(set(context), {'coarse_role_tokens', 'coarse_reference_null_probability'})
                self.assertIs(context['coarse_role_tokens'], output['object_slot_role_tokens'])
                self.assertIsNone(calls[1]['oracle_prior_heatmap'])
                self.assertIsNone(calls[1]['object_slot_target_heatmap'])
                self.assertTrue(torch.isfinite(output['mvt2']['trans']).all())


if __name__ == '__main__':
    unittest.main()
