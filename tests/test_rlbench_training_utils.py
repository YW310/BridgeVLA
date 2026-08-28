import importlib.util
import sys
import unittest
from pathlib import Path


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
                    'oracle_prior_fusion1.net.weight',
                    RLBenchTrainingUtilsTest._Parameter(7),
                ),
                (
                    'oracle_prior_feature_adapter1.feature_expand.weight',
                    RLBenchTrainingUtilsTest._Parameter(11),
                ),
                (
                    'oracle_prior_feature_adapter1.relation_encoder.weight',
                    RLBenchTrainingUtilsTest._Parameter(5),
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
            training_utils.should_disable_rgc_loss(False, True, True)
        )
        self.assertTrue(
            training_utils.should_disable_rgc_loss(True, False, False)
        )
        self.assertFalse(
            training_utils.should_disable_rgc_loss(False, True, False)
        )
        self.assertFalse(
            training_utils.should_disable_rgc_loss(False, False, True)
        )

    def test_fusion_only_freezes_original_backbone(self):
        backbone = self._Backbone()
        trainable = training_utils.freeze_for_oracle_fusion(backbone)
        self.assertEqual(trainable, 7)
        self.assertFalse(backbone.named[0][1].requires_grad)
        self.assertTrue(backbone.named[1][1].requires_grad)
        self.assertFalse(backbone.named[2][1].requires_grad)
        self.assertFalse(backbone.named[3][1].requires_grad)

    def test_oracle_adaptation_freezes_original_backbone(self):
        backbone = self._Backbone()
        trainable = training_utils.freeze_for_oracle_adaptation(backbone)
        self.assertEqual(trainable, 23)
        self.assertFalse(backbone.named[0][1].requires_grad)
        self.assertTrue(backbone.named[1][1].requires_grad)
        self.assertTrue(backbone.named[2][1].requires_grad)
        self.assertTrue(backbone.named[3][1].requires_grad)


if __name__ == '__main__':
    unittest.main()
