import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / 'finetune/bridgevla/models/optimizer_utils.py'
)
SPEC = importlib.util.spec_from_file_location('optimizer_utils', MODULE_PATH)
optimizer_utils = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(optimizer_utils)


class OptimizerUtilsTest(unittest.TestCase):
    def test_gemma_layerwise_learning_rates(self):
        kwargs = {
            'base_lr': 8e-5,
            'gemma_lr': 2e-5,
            'gemma_layer_lr_decay': 0.9,
            'num_gemma_layers': 18,
        }
        lr_for = optimizer_utils.parameter_learning_rate
        self.assertEqual(lr_for('module.mvt1.up0.weight', **kwargs), 8e-5)
        self.assertEqual(
            lr_for(
                'module.mvt1.model.language_model.model.layers.17.weight',
                **kwargs,
            ),
            2e-5,
        )
        self.assertAlmostEqual(
            lr_for(
                'module.mvt1.model.language_model.model.layers.9.weight',
                **kwargs,
            ),
            2e-5 * (0.9 ** 8),
        )
        self.assertEqual(
            lr_for(
                'module.mvt1.model.language_model.model.norm.weight',
                **kwargs,
            ),
            2e-5,
        )

    def test_zero_gemma_lr_preserves_base_lr(self):
        self.assertEqual(
            optimizer_utils.parameter_learning_rate(
                'language_model.model.layers.17.weight',
                base_lr=8e-5,
                gemma_lr=0.0,
                gemma_layer_lr_decay=1.0,
                num_gemma_layers=18,
            ),
            8e-5,
        )


if __name__ == '__main__':
    unittest.main()
