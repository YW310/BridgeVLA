"""Replay loader resource controls without simulator or CUDA dependencies."""

import ast
from contextlib import nullcontext
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from finetune.bridgevla.config import get_cfg_defaults


ROOT = Path(__file__).resolve().parents[1]
YARR = ROOT / 'finetune/bridgevla/libs/YARR'


class _Replay:
    _batch_size = 2

    def prepare_enumeration(self):
        return 4

    def sample_transition_batch(self, **kwargs):
        return {'points': torch.arange(6).reshape(2, 3), 'lang_goal': ['test']}

    def enumerate_next_transition_batch(self, **kwargs):
        return self.sample_transition_batch(**kwargs)


class ReplayLoaderConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with mock.patch.object(sys, 'path', [str(YARR)] + sys.path):
            module = importlib.import_module(
                'yarr.replay_buffer.wrappers.pytorch_replay_buffer')
        cls.wrapper = module.PyTorchReplayBuffer

    def _get_dataset(self, only_train=False, **options):
        path = ROOT / 'finetune/RLBench/utils/get_dataset.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'get_dataset')
        namespace = {
            'PyTorchReplayBuffer': self.wrapper,
            'create_replay': lambda **kwargs: _Replay(),
            'CAMERAS': [], 'VOXEL_SIZES': [],
            'clip': SimpleNamespace(load=lambda *args, **kwargs: (mock.Mock(), None)),
            'torch': SimpleNamespace(cuda=SimpleNamespace(
                device=lambda device: nullcontext(), empty_cache=lambda: None)),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        return namespace['get_dataset'](
            [], 2, 2, 'train', 'test', 'raw', 1, 1, False, 'cpu',
            num_workers=options.pop('num_workers', 0), only_train=only_train, **options)

    def test_defaults_preserve_historical_pinning(self):
        self.assertTrue(get_cfg_defaults().pin_memory)
        self.assertTrue(self.wrapper(_Replay(), num_workers=0).dataset().pin_memory)
        for loader in self._get_dataset():
            self.assertTrue(loader.pin_memory)

    def test_resource_limited_profiles_opt_out_without_changing_batch_budget(self):
        for profile in ('cross_scale', 'refine'):
            with self.subTest(profile=profile):
                cfg = get_cfg_defaults()
                cfg.merge_from_file(str(ROOT / 'finetune/RLBench/configs' /
                                        f'rlbench_o2_internal_slots_{profile}.yaml'))
                self.assertFalse(cfg.pin_memory)
                self.assertEqual(cfg.num_workers, 2)
                self.assertEqual(cfg.bs, 16)
                self.assertEqual(cfg.global_batch_size, 192)

    def test_existing_joint_profile_keeps_its_loader_settings(self):
        cfg = get_cfg_defaults()
        cfg.merge_from_file(str(ROOT / 'finetune/RLBench/configs' /
                                'rlbench_o2_internal_slots_joint.yaml'))
        self.assertTrue(cfg.pin_memory)
        self.assertEqual(cfg.num_workers, 4)

    def test_cli_style_overrides_can_disable_and_restore_pinning(self):
        cfg = get_cfg_defaults()
        cfg.merge_from_list(['pin_memory', 'False', 'num_workers', '0'])
        self.assertFalse(cfg.pin_memory)
        self.assertEqual(cfg.num_workers, 0)
        cfg.merge_from_list(['pin_memory', 'True', 'num_workers', '2'])
        self.assertTrue(cfg.pin_memory)
        self.assertEqual(cfg.num_workers, 2)

    def test_unpinned_loader_reads_a_batch_without_calling_pin_memory(self):
        loader = self.wrapper(_Replay(), num_workers=0, pin_memory=False).dataset()
        with mock.patch.object(torch.Tensor, 'pin_memory',
                               side_effect=AssertionError('unexpected host pinning')):
            batch = next(iter(loader))
        self.assertFalse(loader.pin_memory)
        self.assertIsNone(loader.batch_size)  # Replay already yields full batches.
        self.assertEqual(tuple(batch['points'].shape), (2, 3))
        self.assertEqual(batch['lang_goal'], ['test'])

    def test_train_and_test_loaders_receive_both_controls(self):
        for pin_memory in (False, True):
            for workers in (0, 2):
                with self.subTest(pin_memory=pin_memory, workers=workers):
                    train, test = self._get_dataset(
                        pin_memory=pin_memory, num_workers=workers)
                    for loader in (train, test):
                        self.assertIs(loader.pin_memory, pin_memory)
                        self.assertEqual(loader.num_workers, workers)

    def test_training_only_does_not_construct_a_test_loader(self):
        train, test = self._get_dataset(only_train=True, pin_memory=False)
        self.assertFalse(train.pin_memory)
        self.assertIsNone(test)
        self.assertEqual(tuple(next(iter(train))['points'].shape), (2, 3))

    def test_train_entry_forwards_the_effective_config(self):
        path = ROOT / 'finetune/RLBench/train.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'get_dataset']
        self.assertEqual(len(calls), 1)
        arguments = {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}
        for option in ('pin_memory', 'num_workers'):
            self.assertEqual(arguments[option], f'exp_cfg.{option}')


if __name__ == '__main__':
    unittest.main()
