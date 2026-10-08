"""Isolated CPU tests for clean XYZ without importing VLM/render extensions."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


class CrossScaleRenderTest(unittest.TestCase):
    @staticmethod
    def _render_method():
        source = (Path(__file__).resolve().parents[1] / 'finetune/bridgevla/mvt/mvt.py').read_text(encoding='utf-8')
        model = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == 'MVT')
        render = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == 'render')
        namespace = {'torch': torch}
        exec(compile(ast.Module(body=[render], type_ignores=[]), '<render>', 'exec'), namespace)
        return namespace['render']

    def _run_render(self, inherited, augmentation=0., predictor_type='slots'):
        capture = {}
        stage = SimpleNamespace(add_corr=True, norm_corr=True, add_pixel_loc=False)

        def renderer(points, feature, **kwargs):
            capture['feature'] = feature.clone()
            return feature.reshape(1, 2, 2, 6)

        model = SimpleNamespace(mvt1=stage, renderer=renderer,
                                object_conditioning_inherit_coarse_roles=inherited,
                                object_slot_predictor_type=predictor_type)
        points = torch.tensor([[1.5, .2, .3], [.2, .3, .4], [.4, .5, .6], [0., 0., 0.]])
        colors = torch.full((4, 3), .3)
        result = self._render_method()(model, [points], [colors], augmentation, True, None)
        return result, points, capture

    def test_disabled_keeps_legacy_max_normalization(self):
        result, points, capture = self._run_render(False)
        torch.testing.assert_close(capture['feature'][:, :3], points / 1.5, rtol=0, atol=0)

    def test_enabled_uses_actual_crop_xyz_despite_norm_corr_true(self):
        result, points, capture = self._run_render(True)
        torch.testing.assert_close(capture['feature'][:, :3], points, rtol=0, atol=0)

    def test_rgb_noise_unchanged_but_xyz_and_background_not_augmented(self):
        torch.manual_seed(13)
        inherited, points, capture = self._run_render(True, .5)
        torch.manual_seed(13)
        legacy, _, _ = self._run_render(False, .5)
        expected = points.reshape(1, 1, 2, 2, 3).permute(0, 1, 4, 2, 3)
        torch.testing.assert_close(inherited[:, :, :3], expected, rtol=0, atol=0)
        torch.testing.assert_close(inherited[:, :, 3:6], legacy[:, :, 3:6], rtol=0, atol=0)
        self.assertEqual(inherited[0, 0, :3, 1, 1].count_nonzero().item(), 0)

    def test_direct_roles_keep_clean_xyz_when_inheritance_is_ablated(self):
        torch.manual_seed(13)
        inherited, _, _ = self._run_render(True, .5)
        torch.manual_seed(13)
        independent, points, capture = self._run_render(False, .5, 'role_queries')
        torch.testing.assert_close(capture['feature'][:, :3], points, rtol=0, atol=0)
        torch.testing.assert_close(independent, inherited, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
