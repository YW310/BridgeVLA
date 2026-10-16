import ast
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from finetune.bridgevla.config import get_cfg_defaults

from finetune.RLBench.utils.semantic_contract import (
    build_semantic_contract,
    validate_semantic_contract,
    validate_semantic_validation_report,
)


ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_CONFIGS = (
    'semantic_gt', 'semantic_gt_relation_anchor', 'semantic_gt_joint',
    'internal_slots', 'internal_slots_joint', 'internal_slots_cross_scale',
    'role_queries',
)


class SemanticContractDefaultsTest(unittest.TestCase):
    @staticmethod
    def _config(name):
        config = get_cfg_defaults()
        config.merge_from_file(str(ROOT / 'finetune/RLBench/configs' /
                                   f'rlbench_o2_{name}.yaml'))
        return config

    @staticmethod
    def _startup(config, validate):
        # Run the actual provenance guard without loading a VLM or simulator.
        tree = ast.parse((ROOT / 'finetune/RLBench/train.py').read_text(encoding='utf-8'))
        experiment = next(node for node in tree.body
                          if isinstance(node, ast.FunctionDef) and node.name == 'experiment')
        guard = next(node for node in experiment.body
                     if isinstance(node, ast.If) and
                     ast.unparse(node.test) == 'exp_cfg.oracle_semantic_audit')
        namespace = dict(exp_cfg=config, semantic_contract=None,
                         train_replay_storage_dir='legacy-buffer-without-report',
                         _validate_semantic_replay_schema=validate, print=Mock())
        exec(compile(ast.Module(body=[guard], type_ignores=[]), '<semantic-startup>', 'exec'), namespace)
        return namespace

    def test_base_and_all_semantic_profiles_default_to_non_strict(self):
        base = get_cfg_defaults()
        self.assertFalse(base.oracle_semantic_audit)
        self.assertFalse(base.oracle_semantic_contract.enforce)
        for name in SEMANTIC_CONFIGS:
            with self.subTest(config=name):
                config = self._config(name)
                self.assertFalse(config.oracle_semantic_audit)
                self.assertFalse(config.oracle_semantic_contract.enforce)
                self.assertTrue(config.use_oracle_objects)  # teacher/input is not disabled
                self.assertEqual(config.oracle_num_points, 512)
                validate = Mock(side_effect=AssertionError('legacy data must not be audited'))
                result = self._startup(config, validate)
                validate.assert_not_called()
                result['print'].assert_not_called()  # no false verified-contract claim
                self.assertIsNone(result['semantic_contract'])

    def test_strict_override_still_validates_and_propagates_failures(self):
        for name in SEMANTIC_CONFIGS:
            with self.subTest(config=name):
                config = self._config(name)
                config.merge_from_list(['oracle_semantic_audit', 'True',
                                        'oracle_semantic_contract.enforce', 'True'])
                self.assertEqual(config.oracle_semantic_contract.required_phase_source, 'demo_events')
                self.assertEqual(config.oracle_semantic_contract.role_config,
                                 'configs/rlbench_o2_semantic_roles.yaml')
                validate = Mock(side_effect=RuntimeError('sampled, not full'))
                with self.assertRaisesRegex(RuntimeError, 'sampled, not full'):
                    self._startup(config, validate)
                validate.assert_called_once_with('legacy-buffer-without-report',
                                                 config.oracle_semantic_contract, 512)

    def test_strict_success_records_contract_only_after_validation(self):
        config = self._config('internal_slots_joint')
        config.merge_from_list(['oracle_semantic_audit', 'True',
                                'oracle_semantic_contract.enforce', 'True'])
        contract = dict(phase_source='demo_events', num_points=512, role_config_sha256='a' * 64)
        result = self._startup(config, Mock(return_value=contract))
        self.assertEqual(result['semantic_contract'], contract)
        self.assertEqual(config.oracle_semantic_contract.role_config_sha256, 'a' * 64)
        self.assertIn('Verified semantic train/eval contract', result['print'].call_args.args[0])

    def test_audit_only_override_is_not_mistaken_for_strict_contract(self):
        config = self._config('internal_slots_joint')
        config.oracle_semantic_audit = True
        validate = Mock()
        with self.assertRaisesRegex(ValueError, 'enforce=True'):
            self._startup(config, validate)
        validate.assert_not_called()


class SemanticContractTest(unittest.TestCase):
    def test_exact_contract_matches(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / 'roles.yaml'
            config.write_text('schema_version: v2\n', encoding='utf-8')
            contract = build_semantic_contract(config, 'sim_replay', 512)
            validate_semantic_contract(contract, dict(contract))
            self.assertEqual(contract['reference_geometry'],
                             'role_typed_point_set_v2')

    def test_phase_points_and_yaml_mismatches_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            first = Path(folder) / 'first.yaml'
            second = Path(folder) / 'second.yaml'
            first.write_text('value: one\n', encoding='utf-8')
            second.write_text('value: two\n', encoding='utf-8')
            expected = build_semantic_contract(first, 'sim_replay', 512)
            variants = (
                build_semantic_contract(first, 'demo_events', 512),
                build_semantic_contract(first, 'sim_replay', 256),
                build_semantic_contract(second, 'sim_replay', 512),
            )
            for stored in variants:
                with self.assertRaisesRegex(RuntimeError, 'contract mismatch'):
                    validate_semantic_contract(stored, expected)

    def test_missing_legacy_contract_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, 'no verified'):
            validate_semantic_contract(None, {'phase_source': 'sim_replay'})

    def test_missing_legacy_contract_can_be_allowed_explicitly(self):
        self.assertFalse(validate_semantic_contract(
            None, {'phase_source': 'demo_events'}, allow_missing=True))
        with self.assertRaisesRegex(RuntimeError, 'contract mismatch'):
            validate_semantic_contract(
                {'phase_source': 'sim_replay'},
                {'phase_source': 'demo_events'},
                allow_missing=True,
            )

    def test_full_validation_report_matches_contract(self):
        expected = {
            'schema_version': 'rlbench_o2_semantic_roles_v2',
            'phase_source': 'demo_events',
            'role_config_sha256': 'a' * 64,
            'num_points': 512,
            'manifest_handle_namespace': 'stored',
        }
        report = {
            'valid': True,
            'validation_complete': True,
            'raw_fallback_files': 0,
            'phase_sources': {'demo_events': 10},
            'schema_version': expected['schema_version'],
            'role_config_sha256': expected['role_config_sha256'],
            'num_points': 512,
            'manifest_handle_namespace': 'stored',
        }
        validate_semantic_validation_report(report, expected, 'report.json')
        for key, value in (
            ('validation_complete', False),
            ('raw_fallback_files', 1),
            ('phase_sources', {'sim_replay': 10}),
            ('role_config_sha256', 'b' * 64),
            ('num_points', 256),
            ('manifest_handle_namespace', 'live'),
        ):
            invalid = dict(report)
            invalid[key] = value
            with self.assertRaises(RuntimeError):
                validate_semantic_validation_report(
                    invalid, expected, 'report.json')


if __name__ == '__main__':
    unittest.main()
