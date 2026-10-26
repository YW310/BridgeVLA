import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
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
        self.scalars = []

    def add_image(self, *args, **kwargs):
        self.images.append((args, kwargs))

    def add_text(self, *args, **kwargs):
        self.text.append((args, kwargs))

    def add_scalar(self, *args, **kwargs):
        self.scalars.append((args, kwargs))


class RLBenchTrainingVisualizationTest(unittest.TestCase):
    def test_role_heatmaps_keep_absolute_strength_and_constant_response(self):
        target = torch.tensor([[[.1, .9], [.2, .8]]])
        reference = target * .01
        target_display = training_visualization._display_heatmap('slot_target_pred', target)
        reference_display = training_visualization._display_heatmap('slot_reference_pred', reference)
        np.testing.assert_allclose(reference_display, target_display * .01)
        constant = training_visualization._display_heatmap('slot_reference_pred', torch.full_like(target, .8))
        np.testing.assert_allclose(constant, .8)
        validity = training_visualization._display_heatmap('reference_projection_valid',
                                                          torch.ones_like(target, dtype=torch.bool))
        np.testing.assert_allclose(validity, 1.)
        # Action heatmap contrast is retained independently of role scores.
        action = training_visualization._display_heatmap('pred', reference)
        self.assertEqual(float(action.max()), 1.)
        self.assertEqual(float(action.min()), 0.)

    def test_role_diagnostics_distinguish_weak_shared_shape_and_identical_high_maps(self):
        target = torch.tensor([[[.1, .9], [.2, .8]]])
        weak = training_visualization.role_map_diagnostics(dict(
            slot_target_pred=target, slot_reference_pred=target * .01))[0]
        collapsed = training_visualization.role_map_diagnostics(dict(
            slot_target_pred=target, slot_reference_pred=target))[0]
        self.assertAlmostEqual(weak['cosine'], 1., places=5)
        self.assertAlmostEqual(weak['reference_peak'], .009, places=5)
        self.assertAlmostEqual(collapsed['reference_peak'], .9, places=5)
        self.assertEqual(collapsed['mean_absolute_gap'], 0.)
        empty = training_visualization.role_map_diagnostics(dict(
            slot_target_pred=target, slot_reference_pred=torch.zeros_like(target)))[0]
        self.assertIsNone(empty['cosine'])

    def test_real_montage_and_writer_preserve_role_magnitude_diagnostics(self):
        payload = dict(input=torch.full((1, 3, 8, 8), .5), gt=torch.ones(1, 8, 8),
                       slot_target_pred=torch.full((1, 8, 8), .8),
                       slot_reference_pred=torch.full((1, 8, 8), .02))
        image = training_visualization._stage_montage(payload, step=1, task='test', language_goal='goal')
        x = training_visualization._OUTER_PADDING + training_visualization._LABEL_WIDTH
        y = training_visualization._OUTER_PADDING + training_visualization._HEADER_HEIGHT
        pitch = 8 + training_visualization._CELL_GAP
        # Columns: input, GT, Target pred, Reference pred.
        self.assertNotEqual(image.getpixel((x + pitch * 2 + 4, y + 4)),
                            image.getpixel((x + pitch * 3 + 4, y + 4)))
        writer = _Writer()
        with tempfile.TemporaryDirectory() as temporary:
            training_visualization.record_training_visualization(
                {'mvt2': payload}, step=10, output_dir=Path(temporary), writer=writer)
        values = {args[0]: args[1] for args, _ in writer.scalars}
        self.assertAlmostEqual(values['train_visualization/mvt2/roles/view_0/reference_peak'], .02)
        self.assertAlmostEqual(values['train_visualization/mvt2/roles/view_0/cosine'], 1., places=5)

    def test_heatmap_cells_keep_thirty_percent_of_rgb(self):
        input_image = np.full((2, 3, 3), 0.5, dtype=np.float32)
        heatmap = np.ones((2, 3), dtype=np.float32)
        blended = training_visualization._blend_heatmap_with_image(
            input_image, heatmap,
        )
        np.testing.assert_allclose(
            blended,
            np.broadcast_to(
                np.array([0.85, 0.85, 0.15], dtype=np.float32),
                blended.shape,
            ),
            atol=1e-6,
        )

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
                'target_prior': torch.rand(views, height, width),
                'reference_prior': torch.rand(views, height, width),
                'relation_anchor': torch.rand(views, height, width),
                'prior': torch.rand(views, height, width),
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
                saved['mvt1'].name, 'step_00000500_mvt1.png'
            )
            self.assertEqual(saved['mvt1'].parent, Path(temporary))
        self.assertEqual(
            writer.images[0][0][0], 'train_visualization/mvt1'
        )
        self.assertEqual(writer.images[0][1]['dataformats'], 'HWC')
        self.assertEqual(len(writer.text), 1)

    def test_full_slot_montage_fits_page_and_has_cell_borders(self):
        views, height, width = 3, 224, 224
        payload = {
            'input': torch.rand(views, 3, height, width),
            'gt': torch.rand(views, height, width),
        }
        for key in (
            'slot_target_gt', 'slot_target_pred',
            'slot_reference_gt', 'slot_reference_pred',
            'target_prior', 'reference_prior', 'relation_anchor',
            'prior', 'pred', 'target_projection_valid', 'reference_projection_valid',
        ):
            payload[key] = torch.rand(views, height, width)

        montage = training_visualization._stage_montage(
            payload, step=500, task='stack_blocks',
            language_goal='stack the blocks',
        )

        self.assertLessEqual(
            montage.width, training_visualization._MAX_MONTAGE_WIDTH)
        self.assertLessEqual(
            montage.height, training_visualization._MAX_MONTAGE_HEIGHT)
        self.assertEqual(
            montage.getpixel((0, 0)), training_visualization._BORDER_COLOR)
        first_cell = (
            training_visualization._OUTER_PADDING
            + training_visualization._LABEL_WIDTH,
            training_visualization._OUTER_PADDING
            + training_visualization._HEADER_HEIGHT,
        )
        self.assertEqual(
            montage.getpixel(first_cell), training_visualization._BORDER_COLOR)


if __name__ == '__main__':
    unittest.main()
