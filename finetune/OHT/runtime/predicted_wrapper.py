"""Predictors consume current observations only; simulator GT is never forwarded."""
import importlib
import numpy as np
from ..data.role_cache import validate_fields


def load_predictor(spec):
    module, separator, factory = spec.partition(":")
    if not separator or not module or not factory:
        raise ValueError("Predictor must be module:factory")
    predictor = getattr(importlib.import_module(module), factory)()
    if not callable(predictor):
        raise TypeError("Predictor factory must return a callable")
    return predictor


def policy_observation(observation, cameras):
    allowed = {"low_dim_state"}
    for camera in cameras:
        allowed.update(f"{camera}_{suffix}" for suffix in (
            "rgb", "depth", "point_cloud", "camera_intrinsics", "camera_extrinsics"))
    missing = allowed - set(observation)
    if missing:
        raise ValueError(f"Missing current observation fields: {sorted(missing)}")
    # Copies prevent a third-party predictor from mutating action inputs.
    return {key: np.asarray(observation[key]).copy() for key in allowed}


class PredictedObjectWrapper:
    def __init__(self, predictor, cameras, point_count=512):
        self.predictor, self.cameras, self.point_count = predictor, tuple(cameras), point_count

    def predict(self, observation, goal):
        current = policy_observation(observation, self.cameras)
        fields = self.predictor(current, goal)
        return validate_fields(fields, "predicted", self.point_count)

    def reset(self):
        if hasattr(self.predictor, "reset"):
            self.predictor.reset()
