import sys
import unittest
from pathlib import Path

import torch
from torch import nn


FINETUNE_ROOT = Path(__file__).resolve().parents[1] / 'finetune'
if str(FINETUNE_ROOT) not in sys.path:
    sys.path.insert(0, str(FINETUNE_ROOT))

from bridgevla.mvt.mvt_single import MVT  # noqa: E402


class _ZeroPositionalEncoding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width

    def forward(self, value):
        return torch.zeros(
            value.shape[0], self.width,
            device=value.device, dtype=torch.float32,
        )


class O2JointActionLossTest(unittest.TestCase):
    def _comparison_head(self):
        module = MVT.__new__(MVT)
        nn.Module.__init__(module)
        module.rot_ver = 1
        module.feat_fc_ex_rot = nn.Linear(4, 4)
        module.feat_fc_init_bn = nn.BatchNorm1d(4)
        module.feat_fc_x = nn.Linear(4, 3)
        module.feat_fc_y = nn.Linear(4, 3)
        module.feat_fc_z = nn.Linear(4, 3)
        module.feat_fc_pe = _ZeroPositionalEncoding(4)
        module.train()
        return module

    def test_base_action_branch_is_detached_and_preserves_batch_norm(self):
        module = self._comparison_head()
        feature = torch.randn(4, 4, requires_grad=True)
        rot_x_y = torch.zeros(4, 2, dtype=torch.long)
        before = {
            name: value.clone()
            for name, value in module.feat_fc_init_bn.named_buffers()
        }

        output = module._forward_base_action_heads(
            feature, rot_x_y, batch_size=4,
        )

        self.assertEqual(
            set(output), {'feat_ex_rot', 'feat_x', 'feat_y', 'feat_z'},
        )
        self.assertTrue(all(not value.requires_grad for value in output.values()))
        for name, value in module.feat_fc_init_bn.named_buffers():
            torch.testing.assert_close(value, before[name])

    def test_frozen_action_batch_norm_keeps_state_and_passes_gradient(self):
        module = self._comparison_head()
        for parameter in module.feat_fc_init_bn.parameters():
            parameter.requires_grad = False
        feature = torch.randn(4, 4, requires_grad=True)
        rot_x_y = torch.zeros(4, 2, dtype=torch.long)
        before = {
            name: value.clone()
            for name, value in module.feat_fc_init_bn.named_buffers()
        }

        output = module._forward_action_heads(
            feature, rot_x_y, batch_size=4,
        )
        sum(value.sum() for value in output.values()).backward()

        self.assertIsNotNone(feature.grad)
        self.assertGreater(feature.grad.abs().sum().item(), 0)
        for name, value in module.feat_fc_init_bn.named_buffers():
            torch.testing.assert_close(value, before[name])

    def test_o2_config_enables_joint_action_losses(self):
        config = (
            FINETUNE_ROOT
            / 'RLBench'
            / 'configs'
            / 'rlbench_o2_gt_instance.yaml'
        ).read_text(encoding='utf-8')
        self.assertIn('exp_id: rlbench_o2_gt_instance_joint_action', config)
        self.assertIn('oracle_adapter_translation_only: False', config)
        self.assertIn('add_rgc_loss: True', config)


if __name__ == '__main__':
    unittest.main()
