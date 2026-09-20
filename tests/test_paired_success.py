import json
from pathlib import Path
import tempfile
import unittest

from tools.compare_paired_success import compare


class PairedSuccessTest(unittest.TestCase):
    def _journals(self, root, reward):
        roots = {}
        for seed in range(3):
            directory = root / str(seed)
            directory.mkdir(parents=True)
            item = dict(schema_version='rlbench_eval_episode_v1', task='stack_blocks',
                        episode_idx=0, reward=reward,
                        run_signature=dict(eval_datafolder='same_data', episode_length=50))
            (directory / 'episode_0.json').write_text(json.dumps(item), encoding='utf-8')
            roots[str(seed)] = directory
        return roots

    def test_gate_requires_positive_paired_improvement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self._journals(root / 'base', 0)
            candidate = self._journals(root / 'new', 100)
            result = compare(baseline, candidate, resamples=100)
            self.assertEqual(result['ci95'], [1., 1.])
            self.assertTrue(result['gt_gate_passed'])
            self.assertFalse(compare(baseline, baseline, resamples=100)['gt_gate_passed'])

    def test_unmatched_episodes_are_not_silently_filtered(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = self._journals(root / 'base', 0)
            candidate = self._journals(root / 'new', 100)
            (candidate['0'] / 'episode_0.json').unlink()
            with self.assertRaises(ValueError):
                compare(baseline, candidate, resamples=100)


if __name__ == '__main__':
    unittest.main()
