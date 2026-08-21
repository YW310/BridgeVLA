import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RLBenchReducedModeIsolationTest(unittest.TestCase):
    def test_original_and_8x40_launchers_are_separate(self):
        original = (ROOT / 'finetune/RLBench/train.sh').read_text()
        reduced = (ROOT / 'finetune/RLBench/train_8x40.sh').read_text()
        self.assertIn('GPUS_PER_NODE=2', original)
        self.assertIn('GPUS_PER_NODE="${GPUS_PER_NODE:-8}"', reduced)

    def test_safe_paligemma_optimization_profile_for_8x40(self):
        defaults = (ROOT / 'finetune/bridgevla/config.py').read_text()
        self.assertIn('_C.efficient_paligemma_forward = False', defaults)
        self.assertIn('_C.gpu_paligemma_preprocessing = False', defaults)
        self.assertIn('_C.flash_attention_2 = False', defaults)
        for name in ('rlbench_trend_8x40.yaml', 'rlbench_full_8x40.yaml'):
            profile = (
                ROOT / 'finetune/RLBench/configs' / name
            ).read_text()
            self.assertIn('efficient_paligemma_forward: True', profile)
            self.assertIn('gpu_paligemma_preprocessing: False', profile)

    def test_gpu_preprocessing_is_only_enabled_in_8x40_profiles(self):
        config_dir = ROOT / 'finetune/RLBench/configs'
        enabled = []
        for path in config_dir.glob('*.yaml'):
            if 'gpu_paligemma_preprocessing: True' in path.read_text():
                enabled.append(path.name)
        self.assertEqual(
            sorted(enabled),
            [],
        )

    def test_gpu_preprocessing_keeps_images_on_device(self):
        source = (
            ROOT / 'finetune/bridgevla/mvt/mvt_single.py'
        ).read_text()
        self.assertIn('def _prepare_paligemma_inputs_gpu', source)
        self.assertIn('pixel_values = F.interpolate(', source)
        self.assertIn('text_inputs = tokenizer(', source)
        self.assertIn('if self.use_gpu_paligemma_preprocessing:', source)

    def test_flash_attention_is_disabled_in_all_profiles(self):
        config_dir = ROOT / 'finetune/RLBench/configs'
        enabled = []
        for path in config_dir.glob('*.yaml'):
            if 'flash_attention_2: True' in path.read_text():
                enabled.append(path.name)
        self.assertEqual(
            sorted(enabled),
            [],
        )
        for name in ('rlbench_trend_8x40.yaml', 'rlbench_full_8x40.yaml'):
            profile = (config_dir / name).read_text()
            self.assertIn('flash_attention_2: False', profile)

    def test_flash_attention_loads_paligemma_on_the_local_cuda_device(self):
        source = (
            ROOT / 'finetune/bridgevla/mvt/mvt_single.py'
        ).read_text()
        self.assertIn('flash_device = torch.device(renderer_device)', source)
        self.assertIn(
            'model_kwargs["device_map"] = {"": str(flash_device)}', source
        )

    def test_8x40_profiles_use_gemma_prefix_freezing_and_layerwise_lr(self):
        defaults = (ROOT / 'finetune/bridgevla/config.py').read_text()
        self.assertIn('_C.freeze_gemma_prefix_layers = 0', defaults)
        self.assertIn('_C.freeze_multimodal_projector = False', defaults)
        self.assertIn('_C.peract.gemma_lr = 0.0', defaults)
        self.assertIn('_C.peract.gemma_layer_lr_decay = 1.0', defaults)
        for name in ('rlbench_trend_8x40.yaml', 'rlbench_full_8x40.yaml'):
            profile = (ROOT / 'finetune/RLBench/configs' / name).read_text()
            self.assertIn('gradient_checkpointing: False', profile)
            self.assertIn('freeze_gemma_prefix_layers: 9', profile)
            self.assertIn('freeze_multimodal_projector: False', profile)
            self.assertIn('gemma_lr: 2e-5', profile)
            self.assertIn('gemma_layer_lr_decay: 0.9', profile)

    def test_new_checkpoint_policy_is_guarded_by_global_batch_mode(self):
        source = (ROOT / 'finetune/RLBench/train.py').read_text()
        self.assertIn(
            'reduced_hardware_mode = exp_cfg.global_batch_size > 0', source
        )
        self.assertIn('if reduced_hardware_mode:', source)
        self.assertIn('should_save = i % 10 == 0', source)


if __name__ == '__main__':
    unittest.main()
