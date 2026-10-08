"""Dependency-free opt-in defaults, routing and checkpoint regressions."""

import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
FLAGS = (
    'supervise_mixed_role_maps', 'inherit_coarse_roles', 'preserve_role_tokens',
)


def _isolated_functions(relative, names, namespace):
    source = (ROOT / relative).read_text(encoding='utf-8')
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(nodes) != len(names):
        raise AssertionError(f'Missing isolated functions in {relative}')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), relative, 'exec'), namespace)
    return namespace


def _model(**flags):
    attributes = {
        f'object_conditioning_{name}': False
        for name in ('shared_action_features', 'use_context') + FLAGS
    }
    attributes.update({f'object_conditioning_{name}': value
                       for name, value in flags.items()})
    model = SimpleNamespace(**attributes)
    model.loaded = None

    def load_state_dict(state, strict=True):
        model.loaded = state
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    model.load_state_dict = load_state_dict
    model.state_dict = lambda: {'unchanged': 1}
    return model


def _train_namespace(checkpoint):
    saved = []
    namespace = {
        'torch': SimpleNamespace(
            load=lambda *args, **kwargs: checkpoint,
            save=lambda value, path: saved.append((value, path)),
        ),
        'os': SimpleNamespace(replace=lambda *args: None),
        'DDP': type('DDP', (), {}),
        'strip_deprecated_oracle_fusion_state': lambda state: (state, []),
        'validate_semantic_contract': lambda *args, **kwargs: None,
    }
    _isolated_functions(
        'finetune/RLBench/train.py',
        {'save_agent', 'load_training_checkpoint', 'load_initial_model_checkpoint'},
        namespace,
    )
    return namespace, saved


def _model_guard_namespace(**overrides):
    """Execute constructor guards without importing Torch or the renderer."""
    source = (ROOT / 'finetune/bridgevla/mvt/mvt.py').read_text(encoding='utf-8')
    model = next(node for node in ast.parse(source).body
                 if isinstance(node, ast.ClassDef) and node.name == 'MVT')
    constructor = next(node for node in model.body
                       if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    guards = []
    for node in constructor.body:
        if isinstance(node, ast.ImportFrom) and node.module == 'point_renderer.rvt_renderer':
            break
        if isinstance(node, ast.If):
            guards.append(node)
    namespace = {
        'oracle_prior_adapter_rank': 16,
        'oracle_prior_relation': True,
        'oracle_relation_gated_adapter': True,
        'oracle_adapter_translation_only': False,
        'oracle_relation_anchor_rank': 16,
        'object_slots_enabled': True,
        'object_slot_predictor_type': 'slots',
        'object_slot_num_slots': 6,
        'object_conditioning_use_context': True,
        'object_conditioning_shared_action_features': True,
        'object_conditioning_supervise_mixed_role_maps': False,
        'object_conditioning_inherit_coarse_roles': False,
        'object_conditioning_preserve_role_tokens': False,
        'stage_two': True,
        'add_corr': True,
        'norm_corr': False,
    }
    namespace.update(overrides)
    exec(compile(ast.Module(body=guards, type_ignores=[]), '<model-guards>', 'exec'), namespace)
    return namespace


class RoleFeatureConfigTest(unittest.TestCase):
    def test_shared_defaults_disable_all_new_features(self):
        source = (ROOT / 'finetune/bridgevla/config.py').read_text(encoding='utf-8')
        assignments = {
            ast.unparse(node.targets[0]): ast.literal_eval(node.value)
            for node in ast.parse(source).body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Attribute)
            and isinstance(node.value, ast.Constant)
        }
        for flag in FLAGS:
            self.assertIs(assignments[f'_C.object_conditioning.{flag}'], False)
        self.assertEqual(assignments['_C.rvt.object_slot_mixed_role_loss_weight'], 1.0)

    def test_train_and_eval_forward_every_flag(self):
        for entry in ('train', 'eval'):
            source = (ROOT / f'finetune/RLBench/{entry}.py').read_text(encoding='utf-8')
            calls = [node for node in ast.walk(ast.parse(source))
                     if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == 'MVT']
            self.assertEqual(len(calls), 1)
            arguments = {keyword.arg: ast.unparse(keyword.value)
                         for keyword in calls[0].keywords}
            for flag in FLAGS:
                self.assertEqual(arguments[f'object_conditioning_{flag}'],
                                 f'exp_cfg.object_conditioning.{flag}')

    def test_cross_scale_config_is_separate_and_retains_joint_budget(self):
        directory = ROOT / 'finetune/RLBench/configs'
        joint = (directory / 'rlbench_o2_internal_slots_joint.yaml').read_text(encoding='utf-8')
        opt_in = (directory / 'rlbench_o2_internal_slots_cross_scale.yaml').read_text(encoding='utf-8')
        for line in joint.splitlines():
            if line and not line.startswith(('#', 'exp_id:')):
                self.assertIn(line, opt_in)
        for flag in FLAGS:
            self.assertIn(f'  {flag}: True', opt_in)
            self.assertNotIn(f'  {flag}: True', joint)
        self.assertIn('  object_slot_mixed_role_loss_weight: 1.0', opt_in)

    def test_model_fails_closed_for_unsupported_flag_combinations(self):
        for flag in FLAGS:
            keyword = f'object_conditioning_{flag}'
            with self.assertRaisesRegex(ValueError, 'internal object slots'):
                _model_guard_namespace(object_slots_enabled=False, **{keyword: True})
        with self.assertRaisesRegex(ValueError, 'stage_two=True'):
            _model_guard_namespace(object_conditioning_inherit_coarse_roles=True,
                                   stage_two=False)
        for flag in ('inherit_coarse_roles', 'preserve_role_tokens'):
            with self.assertRaisesRegex(ValueError, 'shared action features and an anchor'):
                _model_guard_namespace(object_conditioning_shared_action_features=False,
                                       **{f'object_conditioning_{flag}': True})
            with self.assertRaisesRegex(ValueError, 'shared action features and an anchor'):
                _model_guard_namespace(oracle_relation_anchor_rank=0,
                                       object_conditioning_use_context=False,
                                       **{f'object_conditioning_{flag}': True})
        with self.assertRaisesRegex(ValueError, 'rendered XYZ correlation channels'):
            _model_guard_namespace(object_conditioning_inherit_coarse_roles=True,
                                   add_corr=False)
        # The stock rvt2 profile normalizes correlation features; inheritance
        # overrides only XYZ transport, not RGB or the configured VLM path.
        _model_guard_namespace(object_conditioning_inherit_coarse_roles=True,
                               norm_corr=True)
        # Loss supervision is independent of full-action semantic routing.
        _model_guard_namespace(object_conditioning_supervise_mixed_role_maps=True,
                               object_conditioning_shared_action_features=False)
        _model_guard_namespace(object_conditioning_inherit_coarse_roles=True,
                               object_conditioning_preserve_role_tokens=True)

    def test_refine_inherits_predicted_packet_and_does_not_select_again(self):
        source = (ROOT / 'finetune/bridgevla/mvt/mvt.py').read_text(encoding='utf-8')
        model = next(node for node in ast.parse(source).body
                     if isinstance(node, ast.ClassDef) and node.name == 'MVT')
        forward = next(node for node in model.body
                       if isinstance(node, ast.FunctionDef) and node.name == 'forward')
        stage_calls = [node for node in ast.walk(forward)
                       if isinstance(node, ast.Call)
                       and ast.unparse(node.func) == 'self.mvt1']
        self.assertEqual(len(stage_calls), 2)
        stage_calls.sort(key=lambda node: node.lineno)
        coarse = {keyword.arg: keyword.value for keyword in stage_calls[0].keywords}
        refine = {keyword.arg: keyword.value for keyword in stage_calls[1].keywords}
        self.assertEqual(ast.unparse(coarse['object_slot_predictor']), 'self.object_slot_predictor1')
        self.assertNotIn('inherited_object_roles', coarse)
        self.assertEqual(ast.unparse(refine['object_slot_predictor']),
                         'None if inherited_roles is not None else self.object_slot_predictor2')
        self.assertEqual(ast.unparse(refine['inherited_object_roles']), 'inherited_roles')
        for key in ('oracle_prior_heatmap', 'oracle_prior_valid', 'oracle_relation_points'):
            value = refine[key]
            self.assertIsInstance(value, ast.IfExp)
            self.assertEqual(ast.unparse(value.test), 'self.object_slots_enabled')
            self.assertIsNone(ast.literal_eval(value.body))
        packet_calls = [node for node in ast.walk(forward)
                        if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == 'inherit_coarse_roles']
        self.assertEqual(len(packet_calls), 1)
        packet = packet_calls[0]
        self.assertEqual(ast.unparse(packet.args[0]), 'out')
        names = {node.id for node in ast.walk(packet) if isinstance(node, ast.Name)}
        self.assertTrue(names.isdisjoint({'oracle_prior1', 'oracle_prior2',
                                         'oracle_prior_points', 'oracle_prior_valid'}))
        transport = (ROOT / 'finetune/bridgevla/models/cross_scale_roles.py').read_text(
            encoding='utf-8')
        self.assertNotIn('object_slot_target_prior', transport)
        self.assertNotIn('object_slot_target_valid', transport)

    def test_clean_xyz_override_is_opt_in_and_survives_rgb_augmentation(self):
        source = (ROOT / 'finetune/bridgevla/mvt/mvt.py').read_text(encoding='utf-8')
        model = next(node for node in ast.parse(source).body
                     if isinstance(node, ast.ClassDef) and node.name == 'MVT')
        render = next(node for node in model.body
                      if isinstance(node, ast.FunctionDef) and node.name == 'render')
        normalized_guard = ast.parse(
            'mvt.norm_corr and not use_clean_role_xyz',
            mode='eval',
        ).body
        self.assertTrue(any(
            isinstance(node, ast.If)
            and ast.dump(node.test) == ast.dump(normalized_guard)
            for node in ast.walk(render)
        ))
        clean = next(node for node in ast.walk(render)
                     if isinstance(node, ast.Assign)
                     and ast.unparse(node.targets[0]) == 'clean_role_xyz')
        self.assertIsInstance(clean.value, ast.IfExp)
        self.assertIn('use_clean_role_xyz',
                      ast.unparse(clean.value.test))
        self.assertIsNone(ast.literal_eval(clean.value.orelse))
        self.assertEqual(ast.unparse(clean.value.body), 'img[:, :, :3].clone()')
        augmentation = next(node for node in ast.walk(render)
                            if isinstance(node, ast.If)
                            and ast.unparse(node.test) == 'img_aug != 0')
        augmentation_source = ast.unparse(augmentation)
        self.assertIn('img = torch.clamp(img + noise, -1, 1)', augmentation_source)
        self.assertIn('if clean_role_xyz is not None:', augmentation_source)
        self.assertIn('img[:, :, :3] = clean_role_xyz', augmentation_source)
        self.assertNotIn('img[:, :, 3:6] =', augmentation_source)

    def test_inherited_visualization_does_not_require_raw_slot_heads(self):
        source = (ROOT / 'finetune/bridgevla/models/inference_visualization.py').read_text(
            encoding='utf-8')
        functions = {node.name: node for node in ast.parse(source).body
                     if isinstance(node, ast.FunctionDef)}
        payload = ast.unparse(functions['build_internal_slot_stage_payload'])
        self.assertIn("('object_slot_prior',) if inherited or direct else", payload)
        self.assertIn('if not inherited and (not direct):', payload)
        diagnostic = ast.unparse(functions['internal_slot_stage_diagnostics'])
        self.assertIn('[] if inherited or direct else torch.sigmoid', diagnostic)
        self.assertIn('[] if inherited or direct else torch.softmax', diagnostic)
        self.assertIn("result['role_source'] = 'coarse'", diagnostic)

    def test_save_records_all_flags(self):
        namespace, saved = _train_namespace({})
        model = _model(inherit_coarse_roles=True, preserve_role_tokens=True,
                       supervise_mixed_role_maps=True)
        namespace['save_agent'](SimpleNamespace(_network=model), 'unused.pth', 2)
        settings = saved[0][0]['object_conditioning']
        self.assertEqual(set(settings), set(FLAGS) | {'shared_action_features', 'use_context'})
        for flag in FLAGS:
            self.assertIs(settings[flag], True)

    def test_legacy_resume_missing_new_keys_means_false(self):
        for metadata in (None, {'shared_action_features': False, 'use_context': False}):
            checkpoint = {'epoch': 2, 'model_state': {'unchanged': 1}}
            if metadata is not None:
                checkpoint['object_conditioning'] = metadata
            namespace, _ = _train_namespace(checkpoint)
            model = _model()
            with redirect_stdout(io.StringIO()):
                result = namespace['load_training_checkpoint'](
                    SimpleNamespace(_network=model), 'unused.pth',
                )
            self.assertEqual(result, (3, None))
            self.assertEqual(model.loaded, checkpoint['model_state'])

    def test_resume_rejects_changed_routing_but_accepts_loss_only_change(self):
        checkpoint = {'epoch': 0, 'model_state': {'unchanged': 1}}
        namespace, _ = _train_namespace(checkpoint)
        for flag in ('shared_action_features', 'use_context',
                     'inherit_coarse_roles', 'preserve_role_tokens'):
            model = _model(**{flag: True})
            with self.assertRaisesRegex(RuntimeError, '--init_checkpoint'):
                namespace['load_training_checkpoint'](
                    SimpleNamespace(_network=model), 'unused.pth',
                )
            self.assertIsNone(model.loaded)
        model = _model(supervise_mixed_role_maps=True)
        with redirect_stdout(io.StringIO()):
            namespace['load_training_checkpoint'](
                SimpleNamespace(_network=model), 'unused.pth',
            )
        self.assertIsNotNone(model.loaded)

    def test_initialization_ignores_previous_routing_metadata(self):
        checkpoint = {'epoch': 0, 'model_state': {'unchanged': 1},
                      'object_conditioning': {}}
        namespace, _ = _train_namespace(checkpoint)
        model = _model(inherit_coarse_roles=True, preserve_role_tokens=True)
        with redirect_stdout(io.StringIO()):
            namespace['load_initial_model_checkpoint'](
                SimpleNamespace(_network=model), 'unused.pth',
            )
        self.assertEqual(model.loaded, checkpoint['model_state'])

    def test_eval_requires_matching_routing_not_matching_loss(self):
        namespace = _isolated_functions(
            'finetune/RLBench/eval.py', {'_validate_role_feature_checkpoint'}, {},
        )
        validate = namespace['_validate_role_feature_checkpoint']
        config = SimpleNamespace(shared_action_features=False, use_context=False,
                                 inherit_coarse_roles=False, preserve_role_tokens=False,
                                 supervise_mixed_role_maps=True)
        validate({}, config, 'legacy.pth')
        for flag in ('shared_action_features', 'use_context',
                     'inherit_coarse_roles', 'preserve_role_tokens'):
            setattr(config, flag, True)
            with self.assertRaisesRegex(RuntimeError, 'saved exp_cfg.yaml'):
                validate({}, config, 'legacy.pth')
            validate({'object_conditioning': {flag: True}}, config, 'new.pth')
            setattr(config, flag, False)
        with self.assertRaisesRegex(RuntimeError, 'Invalid object_conditioning'):
            validate({'object_conditioning': None}, config, 'invalid.pth')


if __name__ == '__main__':
    unittest.main()
