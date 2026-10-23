"""Pure observed-state targets for the OHT BridgeVLA V4 replay format.

Chunk ticks are real future frames, with absolute world TCP poses in xyzw
order. An absent next key is independent of a still-valid future chunk.
"""
from numbers import Integral, Real

import numpy as np


SCHEMA = "oht_bridgevla_v4"
_DEFAULTS = dict(horizon=16, frame_stride=1, action_dt=None,
                 timestamp_atol=1e-5, timestamp_rtol=1e-4)
_NEUTRAL_ACTION = np.asarray([0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32)
_FIELDS = {
    "key_action", "key_joint_action", "key_valid", "key_frame",
    "action_chunk", "joint_chunk", "action_valid", "action_timestamps",
    "action_frames",
}


def chunk_options(config=None):
    """Return a new, strictly validated chunk configuration mapping."""
    if config is not None and not isinstance(config, dict):
        raise ValueError("chunk must be a configuration mapping")
    config = {} if config is None else config
    unknown = set(config) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"Unsupported chunk options: {sorted(unknown, key=str)}")
    options = {**_DEFAULTS, **config}
    for name, value in options.items():
        if name == "action_dt" and value is None:
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"chunk.{name} must be a finite number")
        try:
            number = float(value)
        except (ValueError, OverflowError) as error:
            raise ValueError(f"chunk.{name} must be a finite number") from error
        allow_zero = name in ("timestamp_atol", "timestamp_rtol")
        if not np.isfinite(number) or number < 0 or (not allow_zero and number == 0):
            raise ValueError(f"chunk.{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
        if name in ("horizon", "frame_stride"):
            if number != int(number):
                raise ValueError(f"chunk.{name} must be a positive integer")
            options[name] = int(number)
        else:
            options[name] = number
    return options


def _timestamps(value, *, minimum=1):
    try:
        result = np.asarray(value)
        if result.dtype.kind not in "iuf":
            raise ValueError("Expected numeric timestamps")
        result = result.astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise ValueError("timestamps must be finite nonnegative numbers") from error
    if (result.ndim != 1 or len(result) < minimum or not np.isfinite(result).all()
            or np.any(result < 0) or np.any(np.diff(result) <= 0)):
        raise ValueError("timestamps must be finite, nonnegative and strictly increasing")
    return result


def resolve_chunk_timing(timestamps, config=None):
    """Infer nominal tick spacing while leaving isolated gaps to window checks."""
    options = chunk_options(config)
    timestamps = _timestamps(timestamps, minimum=2)
    inferred = float(np.median(np.diff(timestamps))) * options["frame_stride"]
    if not np.isfinite(inferred) or inferred <= 0:
        raise ValueError("Cannot resolve positive finite chunk.action_dt from timestamps")
    if options["action_dt"] is None:
        options["action_dt"] = inferred
    elif not np.isclose(options["action_dt"], inferred,
                        atol=options["timestamp_atol"], rtol=options["timestamp_rtol"]):
        raise ValueError("chunk.action_dt disagrees with median timestamp spacing * frame_stride")
    return options


def _resolved(config):
    options = chunk_options(config)
    if options["action_dt"] is None:
        raise ValueError("chunk.action_dt must be resolved before making or validating targets")
    return options


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _source(value, shape, name, *, binary=False):
    result = np.asarray(value)
    if result.shape != shape or result.dtype.kind not in ("biuf" if binary else "iuf"):
        raise ValueError(f"Expected numeric {name}{shape}, got {result.shape}")
    return result


def _finite(value, name):
    if not np.isfinite(value).all():
        raise ValueError(f"Expected finite {name}")


def _actions(poses, joints, gripper, indices):
    """Copy only referenced states; full-episode geometry is validated upstream."""
    pose = np.array(poses[indices], dtype=np.float64, copy=True)
    joint = np.asarray(joints[indices], dtype=np.float64)
    grip = np.asarray(gripper[indices], dtype=np.float64)
    _finite(pose, "TCP poses")
    _finite(joint, "joint positions")
    _finite(grip, "binary gripper")
    if not np.isin(grip, (0, 1)).all():
        raise ValueError("Expected binary gripper in {0,1}")
    norms = np.linalg.norm(pose[..., 3:], axis=-1, keepdims=True)
    if np.any(norms < 1e-8) or not np.isfinite(norms).all():
        raise ValueError("Expected nonzero finite xyzw quaternion")
    pose[..., 3:] /= norms
    pose[..., 3:] *= np.where(pose[..., 6:7] < 0, -1, 1)
    with np.errstate(over="ignore"):
        action = np.concatenate((pose, grip[..., None]), axis=-1).astype(np.float32)
        joint_action = np.concatenate((joint, grip[..., None]), axis=-1).astype(np.float32)
    _finite(action, "float32 action")
    _finite(joint_action, "float32 joint action")
    return action, joint_action


def make_chunk_targets(poses, joints, gripper, timestamps, frame, keys, config):
    """Return observed next-key and chunk targets, or an explicit skip reason.

    Timing must be resolved once per episode. To keep repeated sampling cheap,
    this function validates values in the referenced window/key, not every
    pose in the episode. It never mutates inputs or interpolates source states.
    """
    options = _resolved(config)
    poses = np.asarray(poses)
    if poses.ndim != 2 or poses.shape[1] != 7 or not len(poses):
        raise ValueError("Expected nonempty TCP poses[N,7]")
    n = len(poses)
    poses = _source(poses, (n, 7), "TCP poses")
    joints = _source(joints, (n, 6), "joint positions")
    gripper = _source(gripper, (n,), "binary gripper", binary=True)
    timestamps = _source(timestamps, (n,), "timestamps")
    frame = _integer(frame, "frame")
    if frame >= n:
        raise ValueError("frame must be inside the episode")
    keys = np.asarray(keys)
    if (keys.ndim != 1 or (keys.size and (keys.dtype.kind not in "iu" or
            np.any(keys < 0) or np.any(keys >= n) or np.any(keys[1:] <= keys[:-1])))):
        raise ValueError("keys must be strictly increasing integer episode frames")
    horizon, stride = options["horizon"], options["frame_stride"]
    count = min(horizon, (n - 1 - frame) // stride)
    if not count:
        return None, "no_future_target"
    frames = frame + np.arange(1, count + 1, dtype=np.int64) * stride
    last = int(frames[-1])
    window_times = _timestamps(timestamps[frame:last + 1])
    close = dict(atol=options["timestamp_atol"], rtol=options["timestamp_rtol"])
    if (not np.isclose(np.diff(window_times), options["action_dt"] / stride, **close).all()
            or not np.isclose(np.asarray(timestamps[frames], dtype=np.float64) - window_times[0],
                              np.arange(1, count + 1) * options["action_dt"], **close).all()):
        return None, "timestamp_gap"
    window_grip = gripper[frame:last + 1]
    _finite(window_grip, "binary gripper")
    if not np.isin(window_grip, (0, 1)).all():
        raise ValueError("Expected binary gripper in {0,1}")
    transitions = (window_grip[1:] != window_grip[:-1]).reshape(count, stride)
    if np.any(np.count_nonzero(transitions, axis=1) > 1):
        return None, "gripper_event_aliasing"
    action, joint_action = _actions(poses, joints, gripper, frames)
    targets = dict(
        key_action=_NEUTRAL_ACTION.copy(),
        key_joint_action=np.zeros(7, dtype=np.float32),
        key_valid=np.asarray(False, dtype=np.bool_),
        key_frame=np.asarray(-1, dtype=np.int64),
        action_chunk=np.tile(_NEUTRAL_ACTION, (horizon, 1)),
        joint_chunk=np.zeros((horizon, 7), dtype=np.float32),
        action_valid=np.arange(horizon) < count,
        action_timestamps=np.zeros(horizon, dtype=np.float64),
        action_frames=np.full(horizon, -1, dtype=np.int64),
    )
    targets["action_chunk"][:count] = action
    targets["joint_chunk"][:count] = joint_action
    targets["action_timestamps"][:count] = timestamps[frames]
    targets["action_frames"][:count] = frames
    future = keys[keys > frame]
    if len(future):
        key = int(future[0])
        action, joint_action = _actions(poses, joints, gripper, np.asarray([key]))
        targets.update(key_action=action[0], key_joint_action=joint_action[0],
                       key_valid=np.asarray(True, dtype=np.bool_),
                       key_frame=np.asarray(key, dtype=np.int64))
    return targets, None


def validate_chunk_targets(targets, config, row):
    """Validate serialized target values without pickle or scene-bound checks."""
    options = _resolved(config)
    if not hasattr(targets, "keys") or set(targets.keys()) != _FIELDS:
        raise ValueError("V4 targets must contain exactly the chunk target keys")
    horizon, stride = options["horizon"], options["frame_stride"]
    specs = dict(key_action=((8,), "float32"), key_joint_action=((7,), "float32"),
                 key_valid=((), "bool"), key_frame=((), "int64"),
                 action_chunk=((horizon, 8), "float32"),
                 joint_chunk=((horizon, 7), "float32"),
                 action_valid=((horizon,), "bool"),
                 action_timestamps=((horizon,), "float64"),
                 action_frames=((horizon,), "int64"))
    arrays = {}
    for name, (shape, dtype) in specs.items():
        value = np.asarray(targets[name])
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(f"{name}: expected {dtype}{shape}, got {value.dtype}{value.shape}")
        _finite(value, name)
        arrays[name] = value
    try:
        frame = _integer(row["frame"], "row.frame")
        length = _integer(row["episode_length"], "row.episode_length", minimum=1)
        target_frame = _integer(row["target_frame"], "row.target_frame", minimum=-1)
        timestamp = row["timestamp"]
    except (KeyError, TypeError) as error:
        raise ValueError("row requires frame, timestamp, target_frame and episode_length") from error
    if (isinstance(timestamp, (bool, np.bool_)) or not isinstance(timestamp, Real)
            or not np.isfinite(timestamp) or timestamp < 0):
        raise ValueError("row.timestamp must be finite and nonnegative")
    if frame >= length:
        raise ValueError("row.frame must be inside episode_length")
    valid = arrays["action_valid"]
    count = int(valid.sum())
    expected_count = min(horizon, (length - 1 - frame) // stride)
    if (not count or count != expected_count
            or not np.array_equal(valid, np.arange(horizon) < count)):
        raise ValueError("action_valid must be the nonempty available future prefix")
    expected_frames = frame + np.arange(1, count + 1, dtype=np.int64) * stride
    if not np.array_equal(arrays["action_frames"][:count], expected_frames):
        raise ValueError("action_frames must be future frame + stride tick indexes")
    offsets = arrays["action_timestamps"][:count] - timestamp
    if (np.any(offsets <= 0) or not np.isclose(offsets,
            np.arange(1, count + 1) * options["action_dt"],
            atol=options["timestamp_atol"], rtol=options["timestamp_rtol"]).all()):
        raise ValueError("action_timestamps must match future action_dt ticks relative to row.timestamp")
    if (not np.array_equal(arrays["action_chunk"][count:],
                           np.tile(_NEUTRAL_ACTION, (horizon - count, 1)))
            or np.any(arrays["joint_chunk"][count:] != 0)
            or np.any(arrays["action_timestamps"][count:] != 0)
            or np.any(arrays["action_frames"][count:] != -1)):
        raise ValueError("Padded chunk targets must use exact neutral sentinels")
    key_valid = bool(arrays["key_valid"])
    if int(arrays["key_frame"]) != target_frame or key_valid != (target_frame != -1):
        raise ValueError("key_valid/key_frame must agree with row.target_frame")
    if key_valid:
        if not frame < target_frame < length:
            raise ValueError("key_frame must be a future frame within episode_length")
        matching = np.flatnonzero(arrays["action_frames"][:count] == target_frame)
        if len(matching):
            tick = int(matching[0])
            if (not np.array_equal(arrays["key_action"], arrays["action_chunk"][tick]) or
                    not np.array_equal(arrays["key_joint_action"], arrays["joint_chunk"][tick])):
                raise ValueError("Key and chunk labels for the same future frame must agree")
    elif (not np.array_equal(arrays["key_action"], _NEUTRAL_ACTION)
          or np.any(arrays["key_joint_action"] != 0)):
        raise ValueError("An invalid key must use exact neutral sentinels")
    actions = arrays["action_chunk"][:count]
    joint_actions = arrays["joint_chunk"][:count]
    if key_valid:
        actions = np.concatenate((actions, arrays["key_action"][None]), axis=0)
        joint_actions = np.concatenate((joint_actions, arrays["key_joint_action"][None]), axis=0)
    if not np.allclose(np.linalg.norm(actions[:, 3:7], axis=1), 1, atol=1e-5, rtol=1e-5):
        raise ValueError("Action TCP quaternions must be unit xyzw quaternions")
    if not np.isin(actions[:, 7], (0, 1)).all() or not np.array_equal(actions[:, 7], joint_actions[:, 6]):
        raise ValueError("Action and joint action must agree on binary gripper")
