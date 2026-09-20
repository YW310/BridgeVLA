"""Dependency-free configuration, checkpoint and wiring regression checks."""

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ObjectConditioningConfigTest(unittest.TestCase):
    def test_new_configs_are_opt_in_and_joint_not_adapter_only(self):
        for experiment in ('semantic_gt', 'internal_slots'):
            source = (ROOT / 'finetune/RLBench/configs' /
                      f'rlbench_o2_{experiment}_joint.yaml').read_text(encoding='utf-8')
            for setting in ('shared_action_features: True', 'use_context: True',
                            'freeze_vision_tower: True', 'freeze_gemma_prefix_layers: 18',
                            'freeze_multimodal_projector: False', 'gemma_lr: 1e-5',
                            'gemma_layer_lr_decay: 1.0'):
                self.assertIn(setting, source)
        defaults = (ROOT / 'finetune/bridgevla/config.py').read_text(encoding='utf-8')
        self.assertIn('shared_action_features = False', defaults)
        self.assertIn('use_context = False', defaults)

    def test_semantic_configs_require_online_phase_contract(self):
        names = (
            'rlbench_o2_semantic_gt.yaml',
            'rlbench_o2_semantic_gt_relation_anchor.yaml',
            'rlbench_o2_internal_slots.yaml',
            'rlbench_o2_semantic_gt_joint.yaml',
            'rlbench_o2_internal_slots_joint.yaml',
        )
        for name in names:
            source = (ROOT / 'finetune/RLBench/configs' / name).read_text(
                encoding='utf-8')
            self.assertIn('oracle_semantic_contract:', source)
            self.assertIn('required_phase_source: sim_replay', source)
            self.assertIn(
                'role_config: configs/rlbench_o2_semantic_roles.yaml', source)

    def test_eval_validates_checkpoint_semantic_contract(self):
        eval_source = (ROOT / 'finetune/RLBench/eval.py').read_text(
            encoding='utf-8')
        checkpoint_source = (
            ROOT / 'finetune/bridgevla/utils/rvt_utils.py').read_text(
                encoding='utf-8')
        self.assertIn('checkpoint_validator=checkpoint_validator', eval_source)
        self.assertIn("checkpoint.get('semantic_contract')", eval_source)
        self.assertIn('checkpoint_validator(checkpoint)', checkpoint_source)
        self.assertIn("required_phase_source != 'sim_replay'", eval_source)
        self.assertIn(
            'enforce_oracle_contract and exp_cfg.oracle_semantic_audit',
            eval_source)

    def test_training_requires_full_semantic_validation_report(self):
        train_source = (ROOT / 'finetune/RLBench/train.py').read_text(
            encoding='utf-8')
        self.assertIn("directory / 'semantic_role_validation.json'", train_source)
        self.assertIn('validate_semantic_validation_report(', train_source)
        self.assertIn(
            'if not exp_cfg.oracle_semantic_contract.enforce:', train_source)

    def test_sim_replay_manifest_is_stored_namespace_only(self):
        provider_source = (
            ROOT / 'finetune/RLBench/utils/o2_oracle_provider.py'
        ).read_text(encoding='utf-8')
        rollout_source = (
            ROOT / 'finetune/bridgevla/libs/YARR/yarr/utils/rollout_generator.py'
        ).read_text(encoding='utf-8')
        rewrite_source = (
            ROOT / 'tools/rewrite_replay_with_semantic_roles.py'
        ).read_text(encoding='utf-8')
        eval_source = (ROOT / 'finetune/RLBench/eval.py').read_text(
            encoding='utf-8')

        self.assertIn('env.prepare_sim_replay_manifest()', rollout_source)
        self.assertIn("'handle_namespace': 'stored'", provider_source)
        self.assertIn('_stored_manifest_entry(entry)', provider_source)
        self.assertIn("manifest.get('handle_namespace') != 'stored'", rewrite_source)
        self.assertIn("'manifest_handle_namespace': 'stored'", rewrite_source)
        self.assertIn('include_manifests=replay_ground_truth', eval_source)
        self.assertIn('temporary.replace(path)', provider_source)

    def test_both_stages_and_eval_receive_identical_conditioning_flags(self):
        source = (ROOT / 'finetune/bridgevla/mvt/mvt.py').read_text(encoding='utf-8')
        for flag in ('shared_action_features', 'use_context'):
            keyword = f'object_conditioning_{flag}=self.object_conditioning_{flag}'
            self.assertEqual(source.count(keyword), 2)
            for entry in ('train', 'eval'):
                source_entry = (ROOT / f'finetune/RLBench/{entry}.py').read_text(encoding='utf-8')
                self.assertIn(f'object_conditioning_{flag}=exp_cfg.object_conditioning.{flag}',
                              source_entry)

    def test_shared_global_pooling_is_recomputed_and_base_diagnostic_kept(self):
        source = (ROOT / 'finetune/bridgevla/mvt/mvt_single.py').read_text(encoding='utf-8')
        self.assertIn('feat[0] = global_features.view(', source)
        self.assertIn('(base_global_feat, base_local_feat)', source)
        self.assertIn('out={"trans": trans.clone().detach()}', source)

    def test_resume_of_old_routing_checkpoint_requires_initialization(self):
        source = (ROOT / 'finetune/RLBench/train.py').read_text(encoding='utf-8')
        function = next(node for node in ast.parse(source).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'load_training_checkpoint')
        namespace = {'torch': SimpleNamespace(load=lambda *a, **kw: {}),
                     'DDP': type('DDP', (), {})}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<checkpoint>', 'exec'), namespace)
        model = SimpleNamespace(object_conditioning_shared_action_features=True,
                                object_conditioning_use_context=True)
        with self.assertRaisesRegex(RuntimeError, '--init_checkpoint'):
            namespace['load_training_checkpoint'](SimpleNamespace(_network=model), 'unused')

    def test_all_changed_python_sources_parse(self):
        for relative in ('finetune/bridgevla/models/object_conditioning.py',
                         'finetune/bridgevla/models/oracle_prior.py',
                         'finetune/bridgevla/models/bridgevla_agent.py',
                         'finetune/bridgevla/mvt/mvt.py', 'finetune/bridgevla/mvt/mvt_single.py',
                         'finetune/RLBench/train.py', 'finetune/RLBench/eval.py'):
            ast.parse((ROOT / relative).read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
