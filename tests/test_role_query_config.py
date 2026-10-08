"""Dependency-free wiring and checkpoint transition tests for role queries."""

import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


def _isolated_functions(relative, names, namespace):
    source = (ROOT / relative).read_text(encoding='utf-8')
    functions = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(functions) != len(names):
        raise AssertionError(f'Missing {names} from {relative}')
    exec(compile(ast.Module(body=functions, type_ignores=[]), relative, 'exec'), namespace)
    return namespace


class _Network:
    def __init__(self, predictor_type):
        self.object_slot_predictor_type = predictor_type
        self.loaded = None

    def state_dict(self):
        return {'bridgevla.keep': 'backbone'}

    def load_state_dict(self, state, strict=True):
        self.loaded = state
        return SimpleNamespace(
            unexpected_keys=[],
            missing_keys=[
                'object_slot_predictor1.slot_queries',
                'oracle_prior_feature_adapter1.expand.weight',
            ],
        )


def _train_functions(checkpoint):
    saved = []
    namespace = {
        'torch': SimpleNamespace(load=lambda *args, **kwargs: checkpoint,
                                 save=lambda data, path: saved.append(data)),
        'os': SimpleNamespace(replace=lambda *args: None),
        'DDP': type('DDP', (), {}),
        'strip_deprecated_oracle_fusion_state': lambda state: (state, []),
        'validate_semantic_contract': lambda *args, **kwargs: None,
    }
    return _isolated_functions(
        'finetune/RLBench/train.py',
        {'save_agent', 'load_training_checkpoint', 'load_initial_model_checkpoint'},
        namespace,
    ), saved


class RoleQueryConfigTest(unittest.TestCase):
    def test_default_is_legacy_slots_and_opt_in_profile_is_direct(self):
        defaults = (ROOT / 'finetune/bridgevla/config.py').read_text(encoding='utf-8')
        self.assertIn("_C.object_slots.predictor_type = 'slots'", defaults)
        profile = (ROOT / 'finetune/RLBench/configs/rlbench_o2_role_queries.yaml').read_text(
            encoding='utf-8')
        for setting in ('predictor_type: role_queries', 'num_slots: 2',
                        'object_prior_mode: o2_internal_slots',
                        'supervise_mixed_role_maps: False',
                        'inherit_coarse_roles: True', 'preserve_role_tokens: True',
                        'object_slot_diversity_loss_weight: 0.0'):
            self.assertIn(setting, profile)
        reference = (ROOT / 'finetune/RLBench/configs/rlbench_o2_internal_slots_cross_scale.yaml').read_text(
            encoding='utf-8')
        for setting in ('bs: 48', 'train_iter: 38400', 'global_batch_size: 192',
                        'freeze_vision_tower: True', 'freeze_gemma_prefix_layers: 18',
                        'freeze_multimodal_projector: False', 'gemma_lr: 1e-5',
                        'point_samples: 128'):
            self.assertIn(setting, reference)
            self.assertIn(setting, profile)

    def test_train_eval_pass_mode_to_same_model_constructor(self):
        for entry in ('train', 'eval'):
            source = (ROOT / f'finetune/RLBench/{entry}.py').read_text(encoding='utf-8')
            calls = [node for node in ast.walk(ast.parse(source))
                     if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == 'MVT']
            self.assertEqual(len(calls), 1)
            arguments = {keyword.arg: ast.unparse(keyword.value)
                         for keyword in calls[0].keywords}
            self.assertEqual(arguments['object_slot_predictor_type'],
                             'exp_cfg.object_slots.predictor_type')

    def test_checkpoint_persists_type_and_legacy_resume_is_slots(self):
        namespace, saved = _train_functions({'epoch': 0,
                                            'model_state': {'bridgevla.keep': 'backbone'}})
        role_query_model = _Network('role_queries')
        namespace['save_agent'](SimpleNamespace(_network=role_query_model),
                                'unused.pth', 0)
        self.assertEqual(saved[0]['object_predictor_type'], 'role_queries')
        old_model = _Network('slots')
        with redirect_stdout(io.StringIO()):
            namespace['load_training_checkpoint'](
                SimpleNamespace(_network=old_model), 'legacy.pth')
        self.assertIsNotNone(old_model.loaded)
        with self.assertRaisesRegex(RuntimeError, '--init_checkpoint'):
            namespace['load_training_checkpoint'](
                SimpleNamespace(_network=role_query_model), 'legacy.pth')
        self.assertIsNone(role_query_model.loaded)

    def test_resume_rejects_mode_change_before_optimizer_or_weight_load(self):
        checkpoint = {'epoch': 0, 'model_state': {'bridgevla.keep': 'backbone'},
                      'object_predictor_type': 'role_queries'}
        namespace, _ = _train_functions(checkpoint)
        model = _Network('slots')
        with self.assertRaisesRegex(RuntimeError, 'predictor type changed'):
            namespace['load_training_checkpoint'](
                SimpleNamespace(_network=model), 'different-architecture.pth')
        self.assertIsNone(model.loaded)

    def test_init_mode_change_discards_only_predictor_and_adapter(self):
        checkpoint = {'epoch': 1, 'object_predictor_type': 'slots',
                      'model_state': {
                          'object_slot_predictor1.slot_queries': 'six slots',
                          'oracle_prior_feature_adapter1.expand.weight': 'old adapter',
                          'bridgevla.keep': 'backbone',
                          'up0.weight': 'action head',
                      }}
        namespace, _ = _train_functions(checkpoint)
        model = _Network('role_queries')
        messages = io.StringIO()
        with redirect_stdout(messages):
            namespace['load_initial_model_checkpoint'](
                SimpleNamespace(_network=model), 'old_slots.pth')
        self.assertEqual(model.loaded, {'bridgevla.keep': 'backbone',
                                        'up0.weight': 'action head'})
        self.assertIn('reinitialized 1 predictor and 1 adapter tensors',
                      messages.getvalue())
        self.assertNotIn('optimizer', messages.getvalue())

    def test_init_same_mode_preserves_predictor_and_adapter(self):
        checkpoint = {'epoch': 1, 'object_predictor_type': 'role_queries',
                      'model_state': {
                          'object_slot_predictor1.slot_queries': 'two role queries',
                          'oracle_prior_feature_adapter1.expand.weight': 'new adapter',
                          'bridgevla.keep': 'backbone',
                      }}
        namespace, _ = _train_functions(checkpoint)
        model = _Network('role_queries')
        with redirect_stdout(io.StringIO()):
            namespace['load_initial_model_checkpoint'](
                SimpleNamespace(_network=model), 'same_mode.pth')
        self.assertEqual(model.loaded, checkpoint['model_state'])

    def test_eval_rejects_wrong_architecture_even_with_legacy_checkpoint(self):
        namespace = _isolated_functions(
            'finetune/RLBench/eval.py', {'_validate_role_feature_checkpoint'}, {})
        validate = namespace['_validate_role_feature_checkpoint']
        config = SimpleNamespace(shared_action_features=False, use_context=False,
                                 inherit_coarse_roles=False, preserve_role_tokens=False)
        validate({}, config, 'legacy.pth', 'slots')
        with self.assertRaisesRegex(RuntimeError, 'Object predictor type differs'):
            validate({}, config, 'legacy.pth', 'role_queries')
        validate({'object_predictor_type': 'role_queries'}, config,
                 'direct.pth', 'role_queries')


if __name__ == '__main__':
    unittest.main()
