"""Identical policy observations for offline validation and network serving."""
import numpy as np
from ..data.dataset import to_device
from .predicted_wrapper import policy_observation
from ..data.observation import validate_observation
from ..data.point_filter import filter_observation_points
from ..action_modes import action_options, validate_action


class Policy:
    def __init__(self, agent, contract, device, mode, wrapper=None, action_config=None):
        if mode == "predicted_external" and wrapper is None:
            raise ValueError("External prediction requires an online predictor")
        if mode != "predicted_external" and wrapper is not None:
            raise ValueError("Only external mode accepts a predictor wrapper")
        self.agent, self.contract, self.device = agent, contract, device
        self.mode, self.wrapper = mode, wrapper
        self.chunk_settings = agent.settings if getattr(agent, 'is_chunk_agent', False) else None
        self.action_config = action_options(action_config or getattr(agent, "oht_action_config", None))
        if hasattr(agent, "oht_action_config") and self.action_config != action_options(agent.oht_action_config):
            raise ValueError("Policy action configuration differs from the trained agent")

    def reset(self):
        self.agent.reset()
        if self.wrapper:
            self.wrapper.reset()

    def act(self, observation, goal, step, *, visualization_callback=None):
        """Optionally capture label-free CPU diagnostics from this same forward."""
        import torch
        if self.chunk_settings is not None:
            # Copy only present-time observation fields; labels never enter the encoder/decoder.
            current = policy_observation(observation, self.contract['data_config']['cameras'], self.action_config)
            for name in ('tcp_pose', 'joint_positions', 'gripper_open01'):
                current[name] = np.asarray(observation[name]).copy()
            current = filter_observation_points(current, self.contract['data_config'])
            validate_observation(current, self.contract['data_config'])
            from .chunk_protocol import chunk_spec, validate_prediction
            captured = []
            def capture(payload):
                captured.append(payload)
            error_handler = getattr(visualization_callback, 'on_error', None)
            if callable(error_handler):
                capture.on_error = error_handler
            spec = chunk_spec(self.chunk_settings, self.contract)
            prediction = validate_prediction(self.agent.predict(current, goal,
                capture if visualization_callback is not None else None), spec)
            if visualization_callback is not None:
                if len(captured) != 1:
                    error = RuntimeError('Chunk visualization requires one encoder snapshot')
                    if callable(error_handler):
                        error_handler(error)
                        return prediction
                    raise error
                captured[0]['chunk'] = dict(spec=spec, actions=prediction['actions'], key_action=prediction['key_action'])
                visualization_callback(captured[0])
            return prediction
        current = policy_observation(observation, self.contract["data_config"]["cameras"], self.action_config)
        current = filter_observation_points(current, self.contract["data_config"])
        validate_observation(current, self.contract["data_config"])
        if self.wrapper:
            current.update(self.wrapper.predict(current, goal))
        batch = {key: torch.from_numpy(np.asarray(value).copy())[None, None]
                 for key, value in current.items()}
        batch = to_device(batch, self.device)
        batch["language_goal"] = [[[goal]]]
        with torch.no_grad():
            # EEF has eight values; vector heads return six controls + gripper.
            extra = {} if visualization_callback is None else {"visualization_callback": visualization_callback}
            action = self.agent.act(step, batch, deterministic=True, return_gembench_action=True, **extra)
        return validate_action(action, self.action_config)
