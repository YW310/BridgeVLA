import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / 'finetune'
    / 'bridgevla'
    / 'models'
    / 'inference_visualization.py'
)
SPEC = importlib.util.spec_from_file_location(
    'bridgevla_inference_visualization', MODULE_PATH,
)
visualization = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = visualization
SPEC.loader.exec_module(visualization)


def _stage_output(views=3, slots=2, height=12, width=10):
    return {
        'object_slot_masks': torch.rand(1, views, slots, height, width),
        'object_slot_prior': torch.rand(1, views, 2, height, width),
        'object_slot_confidence': torch.tensor([[0.8, 0.6]]),
        'object_slot_valid': torch.tensor([[True, True]]),
        'object_slot_objectness_logits': torch.tensor([[1.0, 0.5]]),
        'object_slot_role_logits': torch.tensor(
            [[[2.0, -1.0], [-0.5, 1.5]]]
        ),
        'object_slot_reference_null_probability': torch.tensor([0.1]),
        'oracle_relation_anchor': torch.rand(1, views, height // 2, width // 2),
    }


class InferenceVisualizationTest(unittest.TestCase):
    def test_heatmap_overlay_keeps_thirty_percent_of_rgb(self):
        input_image = np.full((4, 5, 3), 0.5, dtype=np.float32)
        # Also exercise resizing of a low-resolution relation-anchor map.
        heatmap = np.ones((2, 3), dtype=np.float32)
        blended = visualization.blend_heatmap_with_image(
            input_image, heatmap,
        )
        self.assertEqual(blended.shape, input_image.shape)
        np.testing.assert_allclose(
            blended,
            np.broadcast_to(
                np.array([0.85, 0.85, 0.15], dtype=np.float32),
                blended.shape,
            ),
            atol=1e-6,
        )

    def test_combines_all_stages_and_views_into_one_bounded_image(self):
        payloads = {}
        diagnostics = {}
        for stage_name in ('mvt1', 'mvt2'):
            stage_output = _stage_output()
            payloads[stage_name] = visualization.build_internal_slot_stage_payload(
                stage_output,
                torch.rand(3, 3, 12, 10),
                torch.rand(3, 12, 10),
            )
            diagnostics[stage_name] = (
                visualization.internal_slot_stage_diagnostics(stage_output)
            )

        montage = visualization.internal_slot_montage(
            payloads, step=7, diagnostics=diagnostics,
        )
        self.assertLessEqual(montage.width, visualization._MAX_MONTAGE_WIDTH)
        self.assertLessEqual(montage.height, visualization._MAX_MONTAGE_HEIGHT)
        self.assertEqual(montage.getpixel((0, 0)), visualization._BORDER_COLOR)

        with tempfile.TemporaryDirectory() as temporary:
            saved = visualization.save_internal_slot_step_visualization(
                payloads, diagnostics, step=7, output_dir=temporary,
            )
            self.assertTrue(saved['montage'].is_file())
            self.assertTrue(saved['diagnostics'].is_file())
            values = json.loads(saved['diagnostics'].read_text(encoding='utf-8'))
        self.assertAlmostEqual(values['mvt1']['target_confidence'], 0.8, places=5)
        self.assertEqual(len(values['mvt2']['slot_role_probability']), 2)

    def test_payload_uses_predicted_roles_without_gt(self):
        payload = visualization.build_internal_slot_stage_payload(
            _stage_output(),
            torch.rand(3, 3, 12, 10),
            torch.rand(3, 12, 10),
        )
        self.assertEqual(payload['slot_0'].shape, (3, 12, 10))
        self.assertEqual(payload['slot_1'].shape, (3, 12, 10))
        self.assertEqual(payload['target_pred'].shape, (3, 12, 10))
        self.assertEqual(payload['reference_pred'].shape, (3, 12, 10))
        self.assertNotIn('gt', payload)


if __name__ == '__main__':
    unittest.main()
