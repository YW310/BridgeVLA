"""Direct OHT vector-action heads sharing the existing renderer and role losses."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import bootstrap  # noqa: F401
from bridgevla.models.bridgevla_agent import RVTAgent
from bridgevla.models.oracle_prior import latest_replay_value
from .action_modes import action_options, scales, state_fields


def action_network_class(base):
    """Subclass the real MVT without changing existing checkpoint key prefixes."""
    class ActionNetwork(base):
        def __init__(self, *args, oht_action_config, **kwargs):
            super().__init__(*args, **kwargs)
            self.oht_action_config = action_options(oht_action_config)
            self.mvt1.oht_return_action_features = True
            feature_dim = self.num_img * self.mvt1.vlm_dim * 2
            state_dim = 7 if self.oht_action_config["mode"] == "relative_eef" else 6
            self.oht_action_head = nn.Sequential(
                nn.Linear(feature_dim + state_dim, self.oht_action_config["hidden_dim"]),
                nn.SiLU(), nn.Linear(self.oht_action_config["hidden_dim"], 7))

        def forward(self, *args, oht_current_state=None, **kwargs):
            if oht_current_state is None:
                raise ValueError("Vector policy needs measured current robot state")
            output = super().forward(*args, **kwargs)
            stage = output["mvt2"] if self.stage_two else output
            features = stage["oht_action_features"]
            state = oht_current_state.to(device=features.device, dtype=features.dtype)
            inputs = torch.cat((features, state), dim=-1)
            output["oht_action_raw"] = self.oht_action_head(
                inputs.to(dtype=self.oht_action_head[0].weight.dtype))
            return output
    return ActionNetwork


class VectorActionAgent(RVTAgent):
    def __init__(self, *args, oht_action_config, **kwargs):
        super().__init__(*args, **kwargs)
        self.oht_action_config = action_options(oht_action_config)

    def _network_extra_inputs(self, sample):
        name, = state_fields(self.oht_action_config)
        if name not in sample:
            raise ValueError(f"{self.oht_action_config['mode']} requires measured {name}")
        state = latest_replay_value(sample[name], 2).float()
        if state.ndim != 2 or state.shape[1] != (7 if name == "tcp_pose" else 6) or not torch.isfinite(state).all():
            raise ValueError(f"Invalid measured {name}")
        if name == "tcp_pose":
            state = state.clone()
            norm = state[:, 3:].norm(dim=-1, keepdim=True)
            if (norm < 1e-8).any():
                raise ValueError("Zero current TCP quaternion")
            state[:, 3:] /= norm
            state[:, 3:] *= torch.where(state[:, 6:7] < 0, -1., 1.)
        return dict(oht_current_state=state)

    def _compute_policy_action_loss(self, output, sample, default_loss, translation_loss):
        raw = output["oht_action_raw"]
        target = latest_replay_value(sample["action"], 2).to(device=raw.device, dtype=raw.dtype)
        if raw.shape != target.shape or raw.ndim != 2 or raw.shape[1] != 7 or not torch.isfinite(target).all():
            raise ValueError("Vector action targets must be finite [B,7]")
        scale = torch.as_tensor(scales(self.oht_action_config), device=raw.device, dtype=raw.dtype)
        regression = F.smooth_l1_loss(raw[:, :6], target[:, :6] / scale)
        grip = F.binary_cross_entropy_with_logits(raw[:, 6], target[:, 6])
        # Both rendering scales need supervised waypoint heatmaps to locate the
        # refine crop. The vector action is still learned directly, not IK or
        # a subtraction from a separately predicted absolute EEF.
        auxiliary = translation_loss * self.oht_action_config["auxiliary_translation_weight"]
        # Keep returned diagnostic classification tensors attached with zero
        # weight so DDP can complete their reducer hooks on every iteration.
        loss = regression + grip + auxiliary + default_loss * 0.
        return loss, dict(vector_action_loss=regression.item(), vector_gripper_loss=grip.item(),
                          eef_aux_translation_loss=auxiliary.item(), action_total_loss=loss.item())

    def _format_policy_action(self, output, observation, default_action):
        raw = output["oht_action_raw"]
        value = raw[0, :6].detach().float().cpu().numpy() * scales(self.oht_action_config)
        options = self.oht_action_config
        if options["mode"] == "relative_joint":
            for joint in options["joint_wrap"]:
                value[joint] = (value[joint] + np.pi) % (2 * np.pi) - np.pi
        if options["mode"] in ("joint", "relative_joint") and options["joint_limits"] is not None:
            limits = np.asarray(options["joint_limits"])
            if options["mode"] == "joint":
                value = np.clip(value, limits[:, 0], limits[:, 1])
            else:
                current = latest_replay_value(observation["joint_positions"], 2)[0].detach().cpu().numpy()
                value = np.clip(current + value, limits[:, 0], limits[:, 1]) - current
        return np.r_[value, float(raw[0, 6].item() >= 0)].astype(np.float32)
