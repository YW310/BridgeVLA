import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'finetune'
    / 'RLBench'
    / 'training_visualization.py'
)
SPEC = importlib.util.spec_from_file_location(
    'rlbench_training_visualization', MODULE_PATH
)
training_visualization = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = training_visualization
SPEC.loader.exec_module(training_visualization)


class _Writer:
    def __init__(self):
        self.images = []
        self.text = []

    def add_image(self, *args, **kwargs):
        self.images.append((args, kwargs))

    def add_text(self, *args, **kwargs):
        self.text.append((args, kwargs))


class RLBenchTrainingVisualizationTest(unittest.TestCase):
    def test_interval_is_optimizer_step_based(self):
        self.assertFalse(
            training_visualization.visualization_due(False, 10, 0)
        )
        self.assertTrue(
            training_visualization.visualization_due(True, 10, 20)
        )
        self.assertFalse(
            training_visualization.visualization_due(True, 10, 21)
        )

    def test_saves_png_and_writes_tensorboard_montage(self):
        views, height, width = 3, 4, 5
        payload = {
            'mvt1': {
                'input': torch.rand(views, 3, height, width),
                'gt': torch.rand(views, height, width),
                'prior': torch.rand(views, height, width),
                'raw_pred': torch.rand(views, height, width),
                'pred': torch.rand(views, height, width),
            }
        }
        writer = _Writer()
        with tempfile.TemporaryDirectory() as temporary:
            saved = training_visualization.record_training_visualization(
                payload,
                step=500,
                output_dir=Path(temporary),
                task='stack_blocks',
                language_goal='stack the blocks',
                writer=writer,
            )
            self.assertTrue(saved['mvt1'].is_file())
        self.assertEqual(
            writer.images[0][0][0], 'train_visualization/mvt1'
        )
        self.assertEqual(writer.images[0][1]['dataformats'], 'HWC')
        self.assertEqual(len(writer.text), 1)


if __name__ == '__main__':
    unittest.main()
