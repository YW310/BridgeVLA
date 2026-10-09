"""Identical policy observations for offline validation and network serving."""
import numpy as np
from ..data.dataset import to_device
from .predicted_wrapper import policy_observation
from ..data.observation import validate_observation
from ..data.point_filter import filter_observation_points


class Policy:
    def __init__(self, agent, contract, device, mode, wrapper=None):
        if mode == "predicted_external" and wrapper is None:
            raise ValueError("External prediction requires an online predictor")
        if mode != "predicted_external" and wrapper is not None:
            raise ValueError("Only external mode accepts a predictor wrapper")
        self.agent, self.contract, self.device = agent, contract, device
        self.mode, self.wrapper = mode, wrapper

    def reset(self):
        self.agent.reset()
        if self.wrapper:
            self.wrapper.reset()

    def act(self, observation, goal, step):
        import torch
        current = policy_observation(observation, self.contract["data_config"]["cameras"])
        current = filter_observation_points(current, self.contract["data_config"])
        validate_observation(current, self.contract["data_config"])
        if self.wrapper:
            current.update(self.wrapper.predict(current, goal))
        batch = {key: torch.from_numpy(np.asarray(value).copy())[None, None]
                 for key, value in current.items()}
        batch = to_device(batch, self.device)
        batch["language_goal"] = [[[goal]]]
        with torch.no_grad():
            # Returns the existing eight-value pose+gripper output; collision bit excluded.
            action = self.agent.act(step, batch, deterministic=True, return_gembench_action=True)
        return np.asarray(action, dtype=np.float32)
