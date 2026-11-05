"""Optional, simulation-time-aligned fusion of overlapping absolute TCP chunks."""
from collections import deque
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .chunk_protocol import validate_prediction, validate_spec
from ..data.geometry import rotation_error


class ChunkTemporalEnsemble:
    def __init__(self, spec, *, decay=.1, max_translation=.02, max_rotation=.15):
        self.spec = validate_spec(spec)
        if (not np.isfinite([decay, max_translation, max_rotation]).all() or
                decay < 0 or min(max_translation, max_rotation) <= 0):
            raise ValueError('Chunk ensemble requires finite decay >=0 and positive agreement limits')
        self.decay, self.max_translation, self.max_rotation = decay, max_translation, max_rotation
        self._history = deque(maxlen=spec['horizon'])
        self.reset()

    @property
    def info(self):
        return dict(enabled=True, alignment='simulation_timestamp', decay_per_tick=self.decay,
                    max_translation_m=self.max_translation, max_rotation_rad=self.max_rotation,
                    gripper='latest_prediction', rotation='weighted_SO3_mean')

    def reset(self):
        self._history.clear()
        self._timestamp, self.last_result = None, None

    def update(self, actions, timestamp):
        if (not np.isfinite(timestamp) or timestamp < 0 or
                (self._timestamp is not None and timestamp <= self._timestamp)):
            raise ValueError('Chunk ensemble requires increasing finite simulation timestamps')
        latest = validate_prediction(dict(actions=actions, key_action=actions[0]), self.spec)['actions'].copy()
        horizon, dt = self.spec['horizon'], self.spec['action_dt']
        aligned = []
        for stamp, previous in self._history:
            age = (timestamp - stamp) / dt
            shift = int(round(age))
            if 0 < shift < horizon and np.isclose(age, shift, atol=1e-5, rtol=1e-5):
                aligned.append((shift, previous, math.exp(-self.decay * age)))
        result, contributors, rejected = latest.copy(), [], 0
        for index, current in enumerate(latest):
            poses, weights = [current], [1.]
            for shift, previous, weight in aligned:
                if index + shift >= horizon or weight == 0:
                    continue
                candidate = previous[index + shift]
                # Do not blend different gripper phases or substantially
                # different motions, including a grossly wrong latest chunk.
                if (candidate[7] != current[7] or
                        np.linalg.norm(candidate[:3] - current[:3]) > self.max_translation or
                        rotation_error(candidate[3:7], current[3:7]) > self.max_rotation):
                    rejected += 1
                    continue
                poses.append(candidate)
                weights.append(weight)
            contributors.append(len(poses))
            if len(poses) == 1:
                continue
            poses = np.asarray(poses)
            result[index, :3] = np.average(poses[:, :3], axis=0, weights=weights)
            quaternion = Rotation.from_quat(poses[:, 3:7]).mean(weights=weights).as_quat()
            if np.dot(quaternion, current[3:7]) < 0:
                quaternion *= -1
            result[index, 3:7] = quaternion
        # Cache raw predictions, never previously fused outputs. Session state
        # changes only after the complete input has passed validation.
        self._history = deque(((stamp, chunk) for stamp, chunk in self._history
                               if stamp + horizon * dt > timestamp + 1e-8 * dt), maxlen=horizon)
        self._history.append((float(timestamp), latest))
        self._timestamp = float(timestamp)
        angles = rotation_error(result[:, 3:7], latest[:, 3:7])
        accepted = sum(count - 1 for count in contributors)
        self.last_result = dict(contributors=contributors, rejected_inconsistent=rejected,
            # Every overlapping historical prediction disagrees with the latest chunk: a
            # sudden jump away from a previously consistent trajectory, or a bad latest chunk.
            latest_outlier=bool(rejected and accepted == 0),
            rejected_fraction=float(rejected / (rejected + accepted)) if rejected + accepted else 0.,
            max_position_adjustment_m=float(np.linalg.norm(result[:, :3] - latest[:, :3], axis=-1).max()),
            max_rotation_adjustment_rad=float(angles.max()))
        return result
