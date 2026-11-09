"""The resumable-eval fingerprint must hash only files that exist and cover the inference chain."""

import ast
from pathlib import Path
import unittest

FINETUNE = Path(__file__).resolve().parents[1] / 'finetune'
EVAL = FINETUNE / 'RLBench' / 'eval.py'


def _constant(name):
    tree = ast.parse(EVAL.read_text(encoding='utf-8'))
    node = next(item for item in tree.body if isinstance(item, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == name for target in item.targets))
    return ast.literal_eval(node.value)


class EvalRuntimeFingerprintTest(unittest.TestCase):
    def test_listed_files_exist(self):
        for name in _constant('_RLBENCH_RUNTIME_FILES'):
            self.assertTrue((FINETUNE / 'RLBench' / name).is_file(), name)
        for name in _constant('_FINETUNE_RUNTIME_FILES'):
            self.assertTrue((FINETUNE / name).is_file(), name)

    def test_object_conditioning_and_policy_modules_are_covered(self):
        self.assertIn('utils/simulator_qt.py', _constant('_RLBENCH_RUNTIME_FILES'))
        listed = set(_constant('_FINETUNE_RUNTIME_FILES'))
        required = {'bridgevla/models/oracle_prior.py', 'bridgevla/models/slot_guided_geometry.py',
                    'bridgevla/models/coarse_slot_export.py',
                    'bridgevla/online/slot_sam3_runtime.py',
                    'bridgevla/online/slot_sam3_policy.py',
                    'bridgevla/online/slot_sam3_perception.py',
                    'bridgevla/offline/slot_sam3_perception.py',
                    'bridgevla/data/observations.py',
                    'bridgevla/models/object_conditioning.py',
                    'bridgevla/models/cross_scale_roles.py', 'bridgevla/models/role_separation.py',
                    'bridgevla/mvt/mvt.py', 'bridgevla/mvt/mvt_single.py'}
        self.assertLessEqual(required, listed)

    def test_signature_builder_uses_the_shared_helper(self):
        source = EVAL.read_text(encoding='utf-8')
        self.assertIn('runtime_files=_inference_runtime_files()', source)


if __name__ == '__main__':
    unittest.main()
