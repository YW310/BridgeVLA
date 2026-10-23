"""Config, checkpoint, and optimizer compatibility of the optional local head."""

import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import types
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

from finetune.bridgevla.config import get_cfg_defaults
from finetune.RLBench.training_utils import freeze_for_oracle_adaptation
from test_role_feature_config import (
    _isolated_functions, _model, _model_guard_namespace, _train_namespace,
)


ROOT = Path(__file__).resolve().parents[1]


class RefineMaskConfigTest(unittest.TestCase):
    def test_opt_in_and_legacy_configs(self):
        for name, enabled in (('cross_scale', False), ('joint', False), ('refine', True)):
            cfg = get_cfg_defaults()
            cfg.merge_from_file(str(ROOT / 'finetune/RLBench/configs' /
                                    f'rlbench_o2_internal_slots_{name}.yaml'))
            self.assertEqual(cfg.object_conditioning.refine_role_masks, enabled)
            self.assertEqual(cfg.object_conditioning.refine_local_query, enabled)
            self.assertFalse(cfg.object_conditioning.refine_mask_detach)
        self.assertTrue(cfg.object_conditioning.inherit_coarse_roles)
        self.assertTrue(cfg.object_conditioning.preserve_role_tokens)
        self.assertEqual(cfg.global_batch_size, 192)
        self.assertEqual(cfg.rvt.object_refine_mask_loss_weight, 1.)

    def test_refine_profile_uses_fixed_roles_without_duplicate_slot_losses(self):
        cfg = get_cfg_defaults()
        cfg.merge_from_file(str(ROOT / 'finetune/RLBench/configs' /
                                'rlbench_o2_internal_slots_refine.yaml'))
        self.assertEqual(cfg.object_slots.predictor_type, 'role_queries')
        self.assertEqual(cfg.object_slots.num_slots, 2)
        self.assertEqual(cfg.object_slots.slot_dim, 128)
        self.assertFalse(cfg.object_conditioning.supervise_mixed_role_maps)
        self.assertEqual(cfg.rvt.object_slot_mixed_role_loss_weight, 0.)
        self.assertEqual(cfg.rvt.object_slot_diversity_loss_weight, 0.)
        self.assertEqual(cfg.rvt.object_slot_mask_loss_weight, 1.)
        self.assertEqual(cfg.rvt.object_slot_null_loss_weight, .25)
        self.assertTrue(cfg.object_conditioning.refine_local_query)
        self.assertEqual((cfg.bs, cfg.global_batch_size, cfg.epochs, cfg.train_iter),
                         (24, 192, 50, 38400))
        self.assertTrue(cfg.freeze_vision_tower)
        self.assertTrue(cfg.freeze_multimodal_projector)
        self.assertEqual(cfg.freeze_gemma_prefix_layers, 18)
        _model_guard_namespace(
            object_slot_predictor_type=cfg.object_slots.predictor_type,
            object_slot_num_slots=cfg.object_slots.num_slots,
            object_conditioning_supervise_mixed_role_maps=False,
            object_conditioning_inherit_coarse_roles=True,
            object_conditioning_preserve_role_tokens=True,
            object_conditioning_refine_role_masks=True,
            object_conditioning_refine_local_query=True,
        )
        ablation = cfg.clone()
        ablation.merge_from_list(['object_conditioning.refine_local_query', 'False',
                                  'object_conditioning.refine_role_masks', 'False'])
        self.assertFalse(ablation.object_conditioning.refine_role_masks)
        self.assertEqual(ablation.object_slots, cfg.object_slots)

    def test_train_eval_and_metadata_wire_new_flags(self):
        for entry in ('train', 'eval'):
            tree = ast.parse((ROOT / f'finetune/RLBench/{entry}.py').read_text(encoding='utf-8'))
            call = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name) and node.func.id == 'MVT')
            args = {item.arg: ast.unparse(item.value) for item in call.keywords}
            for flag in ('refine_role_masks', 'refine_local_query', 'refine_mask_detach'):
                self.assertEqual(args['object_conditioning_' + flag],
                                 'exp_cfg.object_conditioning.' + flag)
        model = _model(refine_role_masks=True, refine_local_query=True, refine_mask_detach=True)
        namespace, saved = _train_namespace({})
        namespace['save_agent'](SimpleNamespace(_network=model), 'new.pth', 0)
        self.assertTrue(saved[0][0]['object_conditioning']['refine_role_masks'])
        self.assertTrue(saved[0][0]['object_conditioning']['refine_local_query'])
        self.assertTrue(saved[0][0]['object_conditioning']['refine_mask_detach'])

    def test_constructor_requires_role_inheritance_and_preservation(self):
        with self.assertRaisesRegex(ValueError, 'refine_local_query requires'):
            _model_guard_namespace(object_conditioning_refine_local_query=True)
        with self.assertRaisesRegex(ValueError, 'inherited roles and preserved tokens'):
            _model_guard_namespace(object_conditioning_refine_role_masks=True)
        _model_guard_namespace(object_conditioning_refine_role_masks=True,
                               object_conditioning_inherit_coarse_roles=True,
                               object_conditioning_preserve_role_tokens=True)
        with self.assertRaisesRegex(ValueError, 'refine_mask_detach requires'):
            _model_guard_namespace(object_conditioning_refine_mask_detach=True)

    def test_new_architecture_rejects_old_optimizer_resume(self):
        namespace, _ = _train_namespace({'epoch': 0, 'model_state': {}})
        with self.assertRaisesRegex(RuntimeError, '--init_checkpoint'):
            namespace['load_training_checkpoint'](
                SimpleNamespace(_network=_model(refine_role_masks=True)), 'old.pth')

    def test_local_query_rejects_static_head_optimizer_and_eval_routing(self):
        metadata = {'refine_role_masks': True}
        namespace, _ = _train_namespace({'epoch': 0, 'model_state': {},
                                         'object_conditioning': metadata})
        with self.assertRaisesRegex(RuntimeError, '--init_checkpoint'):
            namespace['load_training_checkpoint'](SimpleNamespace(_network=_model(
                refine_role_masks=True, refine_local_query=True)), 'static.pth')
        scope = _isolated_functions('finetune/RLBench/eval.py',
                                    {'_validate_role_feature_checkpoint'}, {})
        validate = scope['_validate_role_feature_checkpoint']
        cfg = SimpleNamespace(refine_role_masks=True, refine_local_query=True)
        with self.assertRaisesRegex(RuntimeError, 'refine_local_query'):
            validate({'object_conditioning': metadata}, cfg, 'static.pth')
        validate({'object_conditioning': dict(metadata, refine_local_query=True)}, cfg, 'new.pth')
        # Missing new metadata means False, so static-head checkpoints still work.
        validate({'object_conditioning': metadata},
                 SimpleNamespace(refine_role_masks=True), 'static.pth')

    def test_existing_checkpoint_can_initialize_new_head_without_resetting_backbone(self):
        checkpoint = {'model_state': {'unchanged': 1}}
        namespace, _ = _train_namespace(checkpoint)
        model = _model(refine_role_masks=True)
        def load(state, strict):
            self.assertFalse(strict)
            model.loaded = state
            return SimpleNamespace(missing_keys=['refine_role_mask_head.query.weight'],
                                   unexpected_keys=[])
        model.load_state_dict = load
        with redirect_stdout(io.StringIO()):
            namespace['load_initial_model_checkpoint'](SimpleNamespace(_network=model), 'old.pth')
        self.assertEqual(model.loaded, checkpoint['model_state'])

    def test_eval_requires_trained_head_but_not_training_only_detach_flag(self):
        scope = _isolated_functions('finetune/RLBench/eval.py',
                                    {'_validate_role_feature_checkpoint'}, {})
        validate = scope['_validate_role_feature_checkpoint']
        cfg = SimpleNamespace(refine_role_masks=True, refine_mask_detach=True)
        with self.assertRaisesRegex(RuntimeError, 'refine_role_masks'):
            validate({}, cfg, 'old.pth')
        validate({'object_conditioning': {'refine_role_masks': True}}, cfg, 'new.pth')

    def test_adapter_only_mode_does_not_accidentally_freeze_new_head(self):
        model = nn.Module()
        model.backbone = nn.Linear(2, 2)
        model.refine_role_mask_head = nn.Linear(2, 2)
        count = freeze_for_oracle_adaptation(model)
        self.assertEqual(count, 6)
        self.assertTrue(all(p.requires_grad for p in model.refine_role_mask_head.parameters()))
        self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))

    def test_oht_model_builder_and_baseline_initialization_wire_optional_head(self):
        from finetune.OHT.model import build_agent, load_weights
        captured = {}
        class Network(nn.Module):
            def __init__(self, **kwargs):
                super().__init__()
                captured.update(kwargs)
                self.backbone = nn.Linear(2, 2)
                self.refine_role_mask_head = nn.Linear(2, 2)
                self.mvt1 = nn.Module()
        class Agent:
            def __init__(self, network, **kwargs):
                self._net_mod = network
                self.options = kwargs
            def build(self, **kwargs):
                pass
        model_stub = types.ModuleType('bridgevla.mvt.mvt')
        model_stub.MVT = Network
        agent_stub = types.ModuleType('bridgevla.models.bridgevla_agent')
        agent_stub.RVTAgent = Agent
        config = dict(mode='role_queries', learning_rate=4e-5, agent={}, adapter_only=True,
                      object=dict(refine_role_masks=True, refine_local_query=True,
                                  refine_mask_detach=True),
                      role_loss=dict(refine_map=.5))
        contract = dict(data_config=dict(rotation_classes=72, scene_bounds=[-1] * 3 + [1] * 3,
                                        cameras={'front': {}}, image_size=[128, 128]))
        with mock.patch.dict(sys.modules, {'bridgevla.mvt.mvt': model_stub,
                                          'bridgevla.models.bridgevla_agent': agent_stub}):
            agent = build_agent(config, contract, torch.device('cpu'))
        self.assertTrue(captured['object_conditioning_refine_role_masks'])
        self.assertTrue(captured['object_conditioning_refine_local_query'])
        self.assertTrue(captured['object_conditioning_refine_mask_detach'])
        self.assertEqual(agent.options['object_refine_mask_loss_weight'], .5)
        self.assertTrue(agent._net_mod.refine_role_mask_head.weight.requires_grad)
        baseline = {key: value.clone() for key, value in agent._net_mod.state_dict().items()
                    if not key.startswith('refine_role_mask_head.')}
        load_weights(agent, dict(model_state=baseline), initialize=True)


if __name__ == '__main__':
    unittest.main()
