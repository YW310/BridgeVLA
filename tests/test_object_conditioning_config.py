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
                            'freeze_vision_tower: True', 'gemma_lr: 1e-5',
                            'gemma_layer_lr_decay: 1.0'):
                self.assertIn(setting, source)
        semantic_gt = (
            ROOT / 'finetune/RLBench/configs/rlbench_o2_semantic_gt_joint.yaml'
        ).read_text(encoding='utf-8')
        self.assertIn('freeze_gemma_prefix_layers: 6', semantic_gt)
        self.assertIn('freeze_multimodal_projector: True', semantic_gt)
        internal_slots = (
            ROOT / 'finetune/RLBench/configs/rlbench_o2_internal_slots_joint.yaml'
        ).read_text(encoding='utf-8')
        self.assertIn('freeze_gemma_prefix_layers: 18', internal_slots)
        self.assertIn('freeze_multimodal_projector: False', internal_slots)
        defaults = (ROOT / 'finetune/bridgevla/config.py').read_text(encoding='utf-8')
        self.assertIn('shared_action_features = False', defaults)
        self.assertIn('use_context = False', defaults)

    def test_semantic_configs_require_demo_training_contract(self):
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
            self.assertIn('required_phase_source: demo_events', source)
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
        self.assertIn(
            "required_phase_source not in ('sim_replay', 'demo_events')",
            eval_source)
        self.assertIn(
            'online_phase_source=live_success_conditions', eval_source)
        self.assertIn(
            "allow_missing=(required_phase_source == 'demo_events')",
            eval_source)
        self.assertIn('legacy_demo_checkpoint', eval_source)
        provider_source = (
            ROOT / 'finetune/RLBench/utils/o2_oracle_provider.py'
        ).read_text(encoding='utf-8')
        self.assertIn(
            '"demo_events" if phase_event is not None else "sim_replay"',
            provider_source)
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

    def test_oracle_debug_interval_and_rgb_overlay_are_wired(self):
        eval_source = (ROOT / 'finetune/RLBench/eval.py').read_text(
            encoding='utf-8')
        parser_source = (
            ROOT / 'finetune/bridgevla/utils/rvt_utils.py'
        ).read_text(encoding='utf-8')
        provider_source = (
            ROOT / 'finetune/RLBench/utils/o2_oracle_provider.py'
        ).read_text(encoding='utf-8')
        shell_source = (ROOT / 'finetune/RLBench/eval.sh').read_text(
            encoding='utf-8')
        self.assertIn('"--oracle-debug-interval"', parser_source)
        self.assertIn('oracle_debug_interval=args.oracle_debug_interval',
                      eval_source)
        self.assertIn('debug_interval=oracle_debug_interval', eval_source)
        self.assertIn('ORACLE_DEBUG_INTERVAL="${ORACLE_DEBUG_INTERVAL:-1}"',
                      shell_source)
        self.assertIn('--oracle-debug-interval "${ORACLE_DEBUG_INTERVAL}"',
                      shell_source)
        self.assertIn('self._step_index % self.debug_interval == 0',
                      provider_source)
        self.assertIn('overlay = 0.30 * image', provider_source)
        self.assertIn('0.30 * image[selected]', provider_source)
        self.assertIn('0.70 * np.asarray(color', provider_source)
        self.assertIn(
            'role_audit_step_{self._step_index:03d}.png', provider_source)

    def test_heatmap_action_anchor_attribution_is_eval_only_and_opt_in(self):
        eval_source = (ROOT / 'finetune/RLBench/eval.py').read_text(
            encoding='utf-8')
        parser_source = (
            ROOT / 'finetune/bridgevla/utils/rvt_utils.py').read_text(
            encoding='utf-8')
        provider_source = (
            ROOT / 'finetune/RLBench/utils/o2_oracle_provider.py'
        ).read_text(encoding='utf-8')
        agent_source = (
            ROOT / 'finetune/bridgevla/models/bridgevla_agent.py'
        ).read_text(encoding='utf-8')
        environment_source = (
            ROOT / 'finetune/RLBench/utils/custom_rlbench_env.py'
        ).read_text(encoding='utf-8')
        shell_source = (ROOT / 'finetune/RLBench/eval.sh').read_text(
            encoding='utf-8')
        self.assertIn('"--heatmap-action-anchor"', parser_source)
        self.assertIn('"--heatmap-target-object"', parser_source)
        self.assertIn('"--bridgevla-aligned-objects"', parser_source)
        self.assertIn('"--bridgevla-aligned-reference"', parser_source)
        self.assertIn('dest="heatmap_action_anchor"', parser_source)
        self.assertIn(
            'HEATMAP_ACTION_ANCHOR="${HEATMAP_ACTION_ANCHOR:-${HEATMAP_TARGET_OBJECT:-0}}"',
                       shell_source)
        self.assertIn('BRIDGEVLA_ALIGNED_OBJECTS="${BRIDGEVLA_ALIGNED_OBJECTS:-0}"',
                      shell_source)
        self.assertIn(
            'BRIDGEVLA_ALIGNED_REFERENCE="${BRIDGEVLA_ALIGNED_REFERENCE:-0}"',
            shell_source)
        self.assertIn(
            'heatmap_action_anchor or bridgevla_aligned_objects', eval_source)
        self.assertIn('oracle_target_candidate_points', provider_source)
        self.assertIn("'trans_base' not in base_stage", agent_source)
        self.assertIn("'trans_base', output['trans']", agent_source)
        self.assertIn(
            'oracle_compute_base=(\n                self.heatmap_action_anchor',
            agent_source)
        self.assertIn('phase=-1 objects are diagnostic-only', eval_source)
        self.assertIn("use_base=True", agent_source)
        self.assertIn("final_waypoint=pred_wpt", agent_source)
        self.assertIn('active_semantic_target_mask(\n            pending_targets, phase_indices)',
                      agent_source)
        self.assertIn('pending_target_candidate_mask(', agent_source)
        self.assertIn('completed_policy_target_released', provider_source)
        self.assertIn('and not bool(pending_valid[0, self._bridgevla_target_lock].item())',
                      agent_source)
        self.assertIn('not gripper_open\n                    and grasped_candidate_known',
                      provider_source)
        self.assertIn("f'{prefix}_reference_distance_m'", agent_source)
        rollout_source = (
            ROOT / 'finetune/bridgevla/libs/YARR/yarr/utils/rollout_generator.py'
        ).read_text(encoding='utf-8')
        self.assertNotIn('torch.tensor([v], device=self._env_device)',
                         rollout_source)
        self.assertGreaterEqual(
            rollout_source.count('torch.tensor(np.array([v])'), 2)
        self.assertGreaterEqual(
            rollout_source.count(
                'prepped_data["language_goal"] = [[[env._lang_goal]]]'), 2)
        self.assertIn('replay_elements=heatmap_action_anchor_elements', agent_source)
        self.assertIn('self._bridgevla_target_lock', agent_source)
        self.assertIn('bridgevla_aligned_target_used', agent_source)
        self.assertGreaterEqual(agent_source.count('out = self._network('), 2)
        self.assertIn(
            'heatmap_action_anchor or bridgevla_aligned_objects', eval_source)
        self.assertIn('oracle_target_candidate_reference_points', provider_source)
        self.assertIn(
            "observation['oracle_target_candidate_reference_valid']",
            agent_source,
        )
        self.assertIn('oracle_reference_candidate_occupied', provider_source)
        self.assertIn('self._bridgevla_reference_lock', agent_source)
        self.assertIn('bridgevla_aligned_reference_used', agent_source)
        self.assertIn(
            'bridgevla_aligned_reference requires bridgevla_aligned_objects',
            eval_source)
        self.assertIn('get_grasped_objects', provider_source)
        self.assertIn('oracle_grasped_target_candidate_index', provider_source)
        self.assertIn('bridgevla_aligned_grasp_overrode_heatmap', agent_source)
        self.assertIn('self._bridgevla_failed_candidate', agent_source)
        self.assertIn('blocked_failed_candidate', agent_source)
        self.assertIn('aligned_valid = torch.zeros_like(oracle_valid)', agent_source)
        self.assertIn(
            'if self.bridgevla_aligned_objects else oracle_valid', agent_source)
        self.assertIn('policy_target_prior', agent_source)
        self.assertIn('and aligned_target_used', agent_source)
        self.assertIn('oracle_task_target_object_points', provider_source)
        self.assertIn('oracle_effective_target_candidate_index', provider_source)
        self.assertIn('follow_policy_target', provider_source)
        self.assertIn(
            'follow_policy_target=bridgevla_aligned_objects', eval_source)
        self.assertIn('set_policy_target_candidate', environment_source)
        self.assertIn(
            'bridgevla_aligned_target_locked_index', environment_source)

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
                         'finetune/RLBench/train.py', 'finetune/RLBench/eval.py',
                         'finetune/RLBench/utils/o2_oracle_provider.py',
                         'finetune/bridgevla/utils/rvt_utils.py',
                         'finetune/bridgevla/libs/YARR/yarr/utils/rollout_generator.py'):
            ast.parse((ROOT / relative).read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
