import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class InternalObjectSlotsConfigTest(unittest.TestCase):
    def test_config_uses_oracle_only_as_slot_supervision(self):
        config = (
            ROOT
            / 'finetune'
            / 'RLBench'
            / 'configs'
            / 'rlbench_o2_internal_slots.yaml'
        ).read_text(encoding='utf-8')
        self.assertIn('use_oracle_objects: True', config)
        self.assertIn('use_predicted_objects: False', config)
        self.assertIn('object_prior_mode: o2_internal_slots', config)
        self.assertIn('oracle_prior_mode: none', config)

    def test_internal_forward_withholds_gt_from_adapter(self):
        source = (
            ROOT / 'finetune' / 'bridgevla' / 'mvt' / 'mvt.py'
        ).read_text(encoding='utf-8')
        self.assertIn(
            'policy_prior1 = None if self.object_slots_enabled else oracle_prior1',
            source,
        )
        self.assertIn(
            'policy_points1 = None if self.object_slots_enabled else oracle_prior_points',
            source,
        )
        self.assertIn('object_slot_target_heatmap=(', source)

    def test_inference_does_not_request_oracle_fields(self):
        source = (
            ROOT
            / 'finetune'
            / 'bridgevla'
            / 'models'
            / 'bridgevla_agent.py'
        ).read_text(encoding='utf-8')
        self.assertIn(
            'if self.internal_object_slots_enabled and allow_missing:', source,
        )
        self.assertIn("return {'current_state': current_state}", source)

    def test_inference_visualizations_are_flat_step_files(self):
        agent_source = (
            ROOT
            / 'finetune'
            / 'bridgevla'
            / 'models'
            / 'bridgevla_agent.py'
        ).read_text(encoding='utf-8')
        visualization_source = (
            ROOT
            / 'finetune'
            / 'bridgevla'
            / 'models'
            / 'inference_visualization.py'
        ).read_text(encoding='utf-8')
        self.assertIn(
            'if not self.internal_object_slots_enabled:', agent_source,
        )
        self.assertIn("stem = f'step_{step:04d}'", visualization_source)
        self.assertIn("output_dir / f'{stem}.png'", visualization_source)


if __name__ == '__main__':
    unittest.main()
