import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RLBenchReducedModeIsolationTest(unittest.TestCase):
    def test_original_and_8x40_launchers_are_separate(self):
        original = (ROOT / 'finetune/RLBench/train.sh').read_text()
        reduced = (ROOT / 'finetune/RLBench/train_8x40.sh').read_text()
        self.assertIn('GPUS_PER_NODE=2', original)
        self.assertIn('GPUS_PER_NODE="${GPUS_PER_NODE:-8}"', reduced)

    def test_efficient_forward_defaults_off_and_is_enabled_in_8x40_profiles(self):
        defaults = (ROOT / 'finetune/bridgevla/config.py').read_text()
        self.assertIn('_C.efficient_paligemma_forward = False', defaults)
        self.assertIn('_C.flash_attention_2 = False', defaults)
        for name in ('rlbench_trend_8x40.yaml', 'rlbench_full_8x40.yaml'):
            profile = (
                ROOT / 'finetune/RLBench/configs' / name
            ).read_text()
            self.assertIn('efficient_paligemma_forward: True', profile)
            self.assertIn('flash_attention_2: True', profile)

    def test_flash_attention_is_only_enabled_in_8x40_profiles(self):
        config_dir = ROOT / 'finetune/RLBench/configs'
        enabled = []
        for path in config_dir.glob('*.yaml'):
            if 'flash_attention_2: True' in path.read_text():
                enabled.append(path.name)
        self.assertEqual(
            sorted(enabled),
            ['rlbench_full_8x40.yaml', 'rlbench_trend_8x40.yaml'],
        )

    def test_new_checkpoint_policy_is_guarded_by_global_batch_mode(self):
        source = (ROOT / 'finetune/RLBench/train.py').read_text()
        self.assertIn(
            'reduced_hardware_mode = exp_cfg.global_batch_size > 0', source
        )
        self.assertIn('if reduced_hardware_mode:', source)
        self.assertIn('should_save = i % 10 == 0', source)


if __name__ == '__main__':
    unittest.main()
