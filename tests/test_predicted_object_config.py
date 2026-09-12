import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PredictedObjectConfigTest(unittest.TestCase):
    def test_config_uses_predictions_and_disables_oracle_input(self):
        config = (
            ROOT
            / 'finetune'
            / 'RLBench'
            / 'configs'
            / 'rlbench_o2_predicted_objects.yaml'
        ).read_text(encoding='utf-8')
        self.assertIn('use_oracle_objects: False', config)
        self.assertIn('use_predicted_objects: True', config)
        self.assertIn('object_prior_mode: o2_predicted_relation', config)
        self.assertIn('oracle_prior_mode: none', config)

    def test_replay_schema_contains_role_prediction_fields(self):
        source = (
            ROOT / 'finetune' / 'RLBench' / 'utils' / 'dataset.py'
        ).read_text(encoding='utf-8')
        for suffix in (
            'object_points', 'object_valid', 'present', 'confidence',
        ):
            self.assertIn('predicted_{role}_' + suffix, source)


if __name__ == '__main__':
    unittest.main()
