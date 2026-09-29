import argparse
import ast
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'finetune'
    / 'RLBench'
    / 'training_utils.py'
)
SPEC = importlib.util.spec_from_file_location('rlbench_training_utils', MODULE_PATH)
training_utils = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = training_utils
SPEC.loader.exec_module(training_utils)


class RLBenchTrainingUtilsTest(unittest.TestCase):
    class _Parameter:
        def __init__(self, size):
            self.requires_grad = True
            self._size = size

        def numel(self):
            return self._size

    class _Backbone:
        def __init__(self):
            self.named = [
                ('mvt1.model.weight', RLBenchTrainingUtilsTest._Parameter(100)),
                (
                    'oracle_prior_feature_adapter1.feature_expand.weight',
                    RLBenchTrainingUtilsTest._Parameter(11),
                ),
                (
                    'oracle_prior_feature_adapter1.relation_encoder.weight',
                    RLBenchTrainingUtilsTest._Parameter(5),
                ),
                (
                    'oracle_prior_feature_adapter1.anchor_expand.weight',
                    RLBenchTrainingUtilsTest._Parameter(7),
                ),
                (
                    'object_slot_predictor1.slot_queries',
                    RLBenchTrainingUtilsTest._Parameter(13),
                ),
            ]

        def parameters(self):
            return [parameter for _, parameter in self.named]

        def named_parameters(self):
            return iter(self.named)

    def test_8x40_batch_plan_uses_twelve_micro_batches(self):
        plan = training_utils.build_batch_plan(2, 8, 192)
        self.assertEqual(plan.micro_global_batch_size, 16)
        self.assertEqual(plan.target_global_batch_size, 192)
        self.assertEqual(plan.gradient_accumulation_steps, 12)

    def test_zero_target_preserves_one_update_per_ddp_batch(self):
        plan = training_utils.build_batch_plan(4, 2, 0)
        self.assertEqual(plan.target_global_batch_size, 8)
        self.assertEqual(plan.gradient_accumulation_steps, 1)

    def test_target_batch_must_be_exactly_divisible(self):
        with self.assertRaisesRegex(ValueError, 'must be divisible'):
            training_utils.build_batch_plan(2, 8, 190)

    def test_epoch_count_uses_config_unless_cli_override_is_explicit(self):
        self.assertEqual(training_utils.resolve_training_epochs(50), 50)
        self.assertEqual(training_utils.resolve_training_epochs(50, 80), 80)
        for configured, override in ((0, None), (-1, None), (50, 0), (50, -1)):
            with self.assertRaisesRegex(ValueError, 'epochs must be > 0'):
                training_utils.resolve_training_epochs(configured, override)

    def test_real_train_config_branch_records_effective_epoch_budget(self):
        source = (MODULE_PATH.parent / 'train.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        parser = argparse.ArgumentParser()
        epoch_argument = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
            and ast.unparse(node.value.func) == 'parser.add_argument'
            and node.value.args and isinstance(node.value.args[0], ast.Constant)
            and node.value.args[0].value == '--epochs')
        exec(compile(ast.Module(body=[epoch_argument], type_ignores=[]), '<epoch-cli>', 'exec'),
             {'parser': parser})
        self.assertIsNone(parser.parse_args([]).epochs)

        experiment = next(node for node in tree.body
                          if isinstance(node, ast.FunctionDef) and node.name == 'experiment')
        begin = next(i for i, node in enumerate(experiment.body)
                     if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'exp_cfg')
        end = next(i for i, node in enumerate(experiment.body)
                   if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'cmd_args.epochs')
        fragment = compile(ast.Module(body=experiment.body[begin:end + 1], type_ignores=[]),
                           '<train-config>', 'exec')
        for cli, options, expected in (([], '', 50), ([], 'epochs 60', 60),
                                        (['--epochs', '80'], 'epochs 60', 80)):
            with self.subTest(cli=cli, options=options):
                config = SimpleNamespace(epochs=100)
                config.merge_from_file = lambda path: setattr(config, 'epochs', 50)
                config.merge_from_list = lambda pairs: setattr(config, 'epochs', int(pairs[1]))
                arguments = SimpleNamespace(exp_cfg_path='profile.yaml', exp_cfg_opts=options,
                                            epochs=parser.parse_args(cli).epochs)
                scope = {'cmd_args': arguments,
                         'exp_cfg_mod': SimpleNamespace(get_cfg_defaults=lambda: config),
                         'resolve_training_epochs': training_utils.resolve_training_epochs}
                exec(fragment, scope)
                self.assertEqual(config.epochs, expected)
                self.assertEqual(arguments.epochs, expected)

    def test_reproduction_step_budgets(self):
        self.assertEqual(
            training_utils.optimizer_steps_per_epoch(38400, 192), 200
        )
        self.assertEqual(
            training_utils.optimizer_steps_per_epoch(160000, 192), 833
        )

    def test_epoch_must_contain_a_complete_global_batch(self):
        with self.assertRaisesRegex(ValueError, 'at least one complete'):
            training_utils.optimizer_steps_per_epoch(191, 192)

    def test_translation_only_oracle_training_disables_rgc_loss(self):
        self.assertTrue(
            training_utils.should_disable_rgc_loss(True, True)
        )
        self.assertFalse(
            training_utils.should_disable_rgc_loss(True, False)
        )
        self.assertFalse(
            training_utils.should_disable_rgc_loss(False, True)
        )

    def test_oracle_adaptation_freezes_original_backbone(self):
        backbone = self._Backbone()
        trainable = training_utils.freeze_for_oracle_adaptation(backbone)
        self.assertEqual(trainable, 36)
        self.assertFalse(backbone.named[0][1].requires_grad)
        self.assertTrue(backbone.named[1][1].requires_grad)
        self.assertTrue(backbone.named[2][1].requires_grad)
        self.assertTrue(backbone.named[4][1].requires_grad)


if __name__ == '__main__':
    unittest.main()
