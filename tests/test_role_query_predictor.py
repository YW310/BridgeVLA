'''Numerical contracts for the opt-in fixed Target/Reference queries.'''

import ast
import copy
import inspect
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'finetune'))
from bridgevla.models.oracle_prior import InternalRoleQueryPredictor, InternalObjectSlotPredictor
from bridgevla.models import oracle_prior
from bridgevla.models.cross_scale_roles import inherit_coarse_roles
from tests.test_role_feature_config import _model_guard_namespace


class RoleQueryPredictorTest(unittest.TestCase):
    def _predictor(self, **kwargs):
        options = dict(feature_channels=4, num_views=3, num_slots=2, slot_dim=8,
                       decoder_layers=1, num_heads=2, point_samples=4,
                       confidence_threshold=0., use_context=True, soft_conditioning=True)
        options.update(kwargs)
        return InternalRoleQueryPredictor(**options)

    def _inputs(self):
        return (torch.randn(3, 4, 4, 4, requires_grad=True),
                torch.rand(1, 3, 3, 8, 8) + .1,
                torch.randn(1, 4, requires_grad=True))

    def test_fixed_roles_have_no_instance_assignment_heads(self):
        predictor = self._predictor()
        features, xyz, context = self._inputs()
        output = predictor(features, xyz, context=context)
        self.assertEqual(predictor.slot_queries.shape, (2, 8))
        self.assertEqual(output['prior'].shape, (1, 3, 2, 4, 4))
        self.assertEqual(output['role_tokens'].shape, (1, 2, 8))
        self.assertEqual(output['geometry'].shape, (1, 17))
        self.assertEqual(output['predictor_type'], 'role_queries')
        for key in ('slot_masks', 'objectness_logits', 'role_logits'):
            self.assertNotIn(key, output)
        for name in predictor.state_dict():
            self.assertNotIn('objectness_head', name)
            self.assertNotIn('role_head', name)
        legacy = InternalObjectSlotPredictor(4, 3, num_slots=6, slot_dim=8,
                                            decoder_layers=1, num_heads=2,
                                            use_context=True, soft_conditioning=True)
        self.assertLess(sum(p.numel() for p in predictor.parameters()),
                        sum(p.numel() for p in legacy.parameters()))

    def test_null_posterior_is_exact_sigmoid_and_only_scales_reference(self):
        predictor = self._predictor()
        features, xyz, context = self._inputs()
        output = predictor(features, xyz, context=context)
        probability = output['reference_null_logit'].sigmoid()
        torch.testing.assert_close(output['reference_null_probability'], probability)
        raw_masks = output['mask_logits'].sigmoid()
        torch.testing.assert_close(output['prior'][:, :, 0], raw_masks[:, :, 0])
        torch.testing.assert_close(output['prior'][:, :, 1],
                                   raw_masks[:, :, 1] * (1 - probability)[:, None, None, None])

    def test_empty_geometry_does_not_change_null_or_semantic_tokens(self):
        predictor = self._predictor()
        features, xyz, context = self._inputs()
        observed = predictor(features, xyz, context=context)
        missing = predictor(features, torch.zeros_like(xyz), context=context)
        self.assertFalse(missing['valid'].any())
        self.assertTrue(missing['role_token_valid'].all())
        for key in ('role_tokens', 'reference_null_probability', 'prior'):
            torch.testing.assert_close(missing[key], observed[key])
        self.assertEqual(missing['points'].count_nonzero().item(), 0)
        self.assertTrue(torch.isfinite(missing['geometry']).all())

    def test_map_geometry_and_null_losses_reach_queries_features_and_context(self):
        predictor = self._predictor()
        features, xyz, context = self._inputs()
        output = predictor(features, xyz, context=context)
        # A nonuniform spatial target also exercises differentiable XYZ weighting.
        loss = (output['prior'].square().mean() + output['geometry'][:, :15].square().mean()
                + output['reference_null_probability'].square().mean())
        loss.backward()
        for parameter in (features, predictor.slot_queries, predictor.mask_query.weight,
                          predictor.context_projection[-1].weight,
                          predictor.reference_null_head.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0.)

    def test_checkpoint_roundtrip_and_prediction_only(self):
        predictor = self._predictor().eval()
        features, xyz, context = self._inputs()
        saved = io.BytesIO()
        torch.save(predictor.state_dict(), saved)
        saved.seek(0)
        restored = self._predictor().eval()
        restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
        first, second = predictor(features, xyz, context=context), restored(features, xyz, context=context)
        for key in ('prior', 'points', 'geometry', 'role_tokens', 'reference_null_probability'):
            torch.testing.assert_close(first[key], second[key], rtol=0., atol=0.)

    def test_inheritance_retains_query_type_and_null_outside_crop(self):
        predictor = self._predictor()
        features, xyz, context = self._inputs()
        output = predictor(features, xyz, context=context)
        stage = {'object_slot_' + key: value for key, value in output.items()}
        packet = inherit_coarse_roles(stage, torch.full((1, 3), 10.), 4., xyz,
                                      lambda points: points[:, :, None, :2].expand(-1, -1, 3, -1))
        self.assertEqual(packet['predictor_type'], 'role_queries')
        self.assertTrue(packet['roles_inherited'])
        self.assertFalse(packet['valid'].any())
        self.assertIs(packet['role_tokens'], output['role_tokens'])
        self.assertIs(packet['reference_null_probability'], output['reference_null_probability'])

    def test_invalid_predictor_options_fail_early(self):
        with self.assertRaisesRegex(ValueError, 'exactly two queries'):
            self._predictor(num_slots=6)
        with self.assertRaisesRegex(ValueError, 'soft_conditioning'):
            self._predictor(soft_conditioning=False)
        with self.assertRaisesRegex(ValueError, 'instruction context'):
            self._predictor()(torch.rand(3, 4, 4, 4), torch.rand(1, 3, 3, 4, 4))

    def test_model_guards_reject_double_supervision_and_unsupported_routes(self):
        direct = dict(object_slot_predictor_type='role_queries', object_slot_num_slots=2)
        _model_guard_namespace(**direct)
        for override, message in (
            ({'object_slots_enabled': False}, 'enabled object slots'),
            ({'object_slot_num_slots': 6}, 'num_slots=2'),
            ({'object_conditioning_shared_action_features': False}, 'shared action features'),
            ({'object_conditioning_supervise_mixed_role_maps': True}, 'already supervises'),
            ({'add_corr': False}, 'rendered XYZ correlation channels'),
        ):
            with self.assertRaisesRegex(ValueError, message):
                _model_guard_namespace(**dict(direct, **override))
        with self.assertRaisesRegex(ValueError, 'must be slots or role_queries'):
            _model_guard_namespace(object_slot_predictor_type='typo')

    def test_constructor_uses_direct_decoder_without_unused_refine_predictor(self):
        source = (Path(__file__).resolve().parents[1] / 'finetune/bridgevla/mvt/mvt.py').read_text(
            encoding='utf-8')
        model = next(node for node in ast.parse(source).body
                     if isinstance(node, ast.ClassDef) and node.name == 'MVT')
        constructor = next(node for node in model.body
                           if isinstance(node, ast.FunctionDef) and node.name == '__init__')
        model.body = [constructor]
        captured = {}

        def backbone(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(vlm_dim=4)

        namespace = {'nn': torch.nn, 'copy': copy, 'MVTSingle': backbone}
        for name in ('InternalRoleQueryPredictor', 'InternalObjectSlotPredictor',
                     'OraclePriorFeatureAdapter', 'OracleRelationAnchorFeatureAdapter',
                     'OracleRelationGatedFeatureAdapter'):
            namespace[name] = getattr(oracle_prior, name)
        exec(compile(ast.Module(body=[model], type_ignores=[]), '<mvt-constructor>', 'exec'), namespace)
        model_class = namespace['MVT']
        arguments = {name: None for name, parameter in inspect.signature(model_class).parameters.items()
                     if parameter.default is inspect.Parameter.empty}
        arguments.update(
            img_size=16, stage_two=True, add_corr=True,
            oracle_prior_adapter_rank=2, oracle_prior_relation=True,
            oracle_relation_gated_adapter=True, oracle_relation_anchor_rank=2,
            object_slots_enabled=True, object_slot_predictor_type='role_queries',
            object_slot_num_slots=2, object_slot_dim=8, object_slot_num_heads=2,
            object_slot_decoder_layers=1,
            object_conditioning_shared_action_features=True,
            object_conditioning_use_context=True, object_conditioning_inherit_coarse_roles=True,
            object_conditioning_preserve_role_tokens=True,
        )
        renderer_module = ModuleType('point_renderer.rvt_renderer')
        renderer_module.RVTBoxRenderer = lambda **kwargs: SimpleNamespace(num_img=3)
        with mock.patch.dict(sys.modules, {'point_renderer': ModuleType('point_renderer'),
                                          'point_renderer.rvt_renderer': renderer_module}):
            direct = model_class(**arguments)
            self.assertIsInstance(direct.object_slot_predictor1, InternalRoleQueryPredictor)
            self.assertIsNone(direct.object_slot_predictor2)
            self.assertNotIn('object_slot_predictor_type', captured)
            ablated = model_class(**dict(arguments, object_conditioning_inherit_coarse_roles=False))
            self.assertIsInstance(ablated.object_slot_predictor2, InternalRoleQueryPredictor)
            legacy = model_class(**dict(arguments, object_slot_predictor_type='slots', object_slot_num_slots=6))
            self.assertIsInstance(legacy.object_slot_predictor2, InternalObjectSlotPredictor)


if __name__ == '__main__':
    unittest.main()
