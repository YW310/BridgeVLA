"""Keep the single-factor soft-query experiment matched to legacy slots."""

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "finetune/RLBench/configs"
BASELINE = CONFIGS / "rlbench_o2_internal_slots.yaml"
CANDIDATE = CONFIGS / "rlbench_o2_internal_slots_soft_coarse_matched.yaml"
HAS_YAML = importlib.util.find_spec("yaml") is not None
HAS_YACS = importlib.util.find_spec("yacs") is not None


@unittest.skipUnless(HAS_YAML, "PyYAML is required for configuration checks")
class SoftCoarseMatchedConfigTest(unittest.TestCase):
    def read(self, path):
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_only_exp_id_and_soft_query_switch_differ_in_yaml(self):
        base, candidate = self.read(BASELINE), self.read(CANDIDATE)
        self.assertNotEqual(base.pop("exp_id"), candidate.pop("exp_id"))
        self.assertEqual(candidate.pop("object_conditioning"), {"soft_coarse_queries": True})
        self.assertEqual(base, candidate)

    @unittest.skipUnless(HAS_YACS, "YACS is required to check resolved defaults")
    def test_only_soft_query_switch_differs_after_real_default_merge(self):
        import yaml
        spec = importlib.util.spec_from_file_location("matched_config_defaults", ROOT / "finetune/bridgevla/config.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        base, candidate = module.get_cfg_defaults(), module.get_cfg_defaults()
        base.merge_from_file(str(BASELINE))
        candidate.merge_from_file(str(CANDIDATE))
        self.assertFalse(base.object_conditioning.soft_coarse_queries)
        self.assertTrue(candidate.object_conditioning.soft_coarse_queries)
        candidate.exp_id = base.exp_id
        candidate.object_conditioning.soft_coarse_queries = False
        self.assertEqual(yaml.safe_load(base.dump()), yaml.safe_load(candidate.dump()))

    def test_preserves_old_predictor_budget_losses_and_runtime_settings(self):
        candidate = self.read(CANDIDATE)
        self.assertEqual(candidate["object_slots"]["num_slots"], 2)
        self.assertEqual(candidate["object_slots"].get("predictor_type", "slots"), "slots")
        self.assertEqual(candidate["rvt"]["object_prediction_confidence_threshold"], .25)
        self.assertEqual(candidate["rvt"]["object_slot_diversity_loss_weight"], 0.)
        self.assertEqual(candidate["global_batch_size"], 192)
        self.assertEqual(candidate["num_workers"], 4)
        self.assertEqual(candidate["train_visualization"]["interval"], 500)
        self.assertNotIn("freeze_vision_tower", candidate)
        self.assertNotIn("pin_memory", candidate)


if __name__ == "__main__":
    unittest.main()
