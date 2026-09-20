import tempfile
import unittest
from pathlib import Path

from finetune.RLBench.utils.semantic_contract import (
    build_semantic_contract,
    validate_semantic_contract,
    validate_semantic_validation_report,
)


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
