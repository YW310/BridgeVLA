"""Isolated CPU tests for stage-one point-feature augmentation (img_aug_2)."""

import ast
from pathlib import Path
import unittest

import torch

SOURCE = (Path(__file__).resolve().parents[1] / 'finetune/bridgevla/mvt/mvt.py').read_text(encoding='utf-8')


def _mvt_method(name):
    model = next(node for node in ast.parse(SOURCE).body if isinstance(node, ast.ClassDef) and node.name == 'MVT')
    method = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == name)
    method.decorator_list = []
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), f'<{name}>', 'exec'), namespace)
    return namespace[name]


class ImgFeatAugmentationTest(unittest.TestCase):
    def setUp(self):
        self.augment = _mvt_method('augment_img_feat')
        torch.manual_seed(0)
        self.features = [torch.rand(2, 64, 3) * 2 - 1, torch.rand(2, 32, 3) * 2 - 1]

    def test_positive_magnitude_changes_features_within_bounds(self):
        result = self.augment(self.features, .1)
        self.assertEqual(len(result), len(self.features))
        for original, augmented in zip(self.features, result):
            self.assertEqual(augmented.shape, original.shape)
            self.assertFalse(torch.equal(original, augmented))
            self.assertLessEqual((augmented - original).abs().max().item(), .1 + 1e-6)
            self.assertGreaterEqual(augmented.min().item(), -1)
            self.assertLessEqual(augmented.max().item(), 1)

    def test_inputs_are_not_modified_in_place(self):
        snapshot = [feature.clone() for feature in self.features]
        self.augment(self.features, .5)
        for original, saved in zip(self.features, snapshot):
            torch.testing.assert_close(original, saved, rtol=0, atol=0)

    def test_zero_magnitude_is_identity(self):
        for original, augmented in zip(self.features, self.augment(self.features, 0.)):
            torch.testing.assert_close(original, augmented, rtol=0, atol=0)

    def test_forward_uses_the_augmented_list(self):
        forward = next(node for node in ast.walk(ast.parse(SOURCE))
                       if isinstance(node, ast.FunctionDef) and node.name == 'forward'
                       and 'img_aug_2' in ast.unparse(node))
        assignments = [node for node in ast.walk(forward)
                       if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id == 'img_feat' for target in node.targets)
                       and 'augment_img_feat' in ast.unparse(node.value)]
        self.assertTrue(assignments, 'forward must rebind img_feat to the augmented features')


if __name__ == '__main__':
    unittest.main()
