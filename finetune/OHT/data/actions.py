"""Labels use observed future TCP/gripper states, never raw action commands."""
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import array, quaternion, check_bounds, rotation_error, tcp_pose


def gripper_step(measured, previous_measured=None, previous_state=None, config=None):
    """Causal NumPy equivalent of the reference hysteresis-with-diff rule."""
    config = config or {}
    open_th, close_th, diff_th = (float(config.get(key, default)) for key, default in
                                (("open_threshold", .9), ("close_threshold", .1), ("diff_threshold", .05)))
    if not np.isfinite([open_th, close_th, diff_th]).all() or not 0 <= close_th < open_th <= 1 or not 0 < diff_th <= 1:
        raise ValueError("Invalid gripper hysteresis thresholds")
    measured = float(array(measured, (), "measured gripper open01"))
    previous = measured if previous_measured is None else float(array(previous_measured, (), "previous gripper open01"))
    state = config.get("initial_state", 1) if previous_state is None else previous_state
    if state not in (0, 1) or not 0 <= measured <= 1 or not 0 <= previous <= 1:
        raise ValueError("Expected gripper open01 in [0,1] and binary previous state")
    if measured > open_th:
        return 1
    if measured < close_th:
        return 0
    if measured - previous > diff_th:
        return 1
    if measured - previous < -diff_th:
        return 0
    return int(state)


def gripper_states(states, config):
    states = array(states, name="states")
    if states.ndim != 2 or states.shape[1] != 7 or not len(states):
        raise ValueError("Expected nonempty measured state[N,7]")
    open_raw, close_raw = float(config["open"]), float(config["close"])
    if not np.isfinite([open_raw, close_raw]).all() or abs(close_raw - open_raw) < 1e-8:
        raise ValueError("Gripper open/close must be distinct finite endpoints")
    measured = np.clip((close_raw - states[:, 6]) / (close_raw - open_raw), 0, 1)
    binary = np.empty(len(states), dtype=np.int64)
    previous, state = None, None
    for i, value in enumerate(measured):
        state = gripper_step(value, previous, state, config)
        binary[i], previous = state, value
    return measured.astype(np.float32), binary


def low_dim(measured, finger_joints=None, *, binary_state=None):
    # Canonical motor-state open01, not unverified per-finger physical endpoints.
    # The two compatibility opening features are synthetic, not metre readings.
    measured = float(np.clip(array(measured, (), "measured gripper open01"), 0, 1))
    binary_state = gripper_step(measured) if binary_state is None else binary_state
    if binary_state not in (0, 1):
        raise ValueError("Expected measured binary gripper state")
    return np.asarray([binary_state, measured * .04, measured * .04, 0], dtype=np.float32)


def world_tcp_poses(columns, link_to_tcp, quaternion_order):
    positions = array(columns["observation.ee_pos_world"], name="world EE positions")
    orientations = quaternion(columns["observation.ee_quat_world"])
    if quaternion_order == "wxyz":
        orientations = orientations[..., [1, 2, 3, 0]]
    elif quaternion_order != "xyzw":
        raise ValueError("Set ee_quaternion_order explicitly to wxyz or xyzw")
    return np.asarray([tcp_pose(p, q, link_to_tcp) for p, q in zip(positions, orientations)])


def keypoint_options(config=None):
    """Validate the extraction mode; missing method preserves legacy configs."""
    if config is not None and not isinstance(config, dict):
        raise ValueError("keypoints must be a configuration mapping")
    config = config or {}
    method = config.get("method", "geometric")
    if method == "gripper":
        defaults = dict(method=method)
    elif method == "bridgevla":
        defaults = dict(method=method, stopping_translation_speed=.01,
                        stopping_rotation_speed_degrees=5.)
    elif method == "geometric":
        defaults = dict(method=method, max_translation=.04,
                        max_rotation_degrees=8., max_frames=30)
    else:
        raise ValueError("keypoints.method must be gripper, bridgevla or geometric")
    unknown = set(config) - set(defaults)
    if unknown:
        raise ValueError(f"Unsupported keypoints options for {method}: {sorted(unknown)}")
    options = {**defaults, **config}
    for key in set(defaults) - {"method"}:
        try:
            value = float(options[key])
        except (TypeError, ValueError) as error:
            raise ValueError(f"keypoints.{key} must be finite and positive") from error
        if not np.isfinite(value) or value <= 0 or isinstance(options[key], bool):
            raise ValueError(f"keypoints.{key} must be finite and positive")
        if key == "max_frames" and value != int(value):
            raise ValueError("keypoints.max_frames must be a positive integer")
        options[key] = int(value) if key == "max_frames" else value
    return options


def keypoints(poses, observed_gripper, instruction_ids=None, *, timestamps=None, **config):
    """Future action endpoints, extracted before observation subsampling.

    gripper: only observed binary gripper changes. No synthetic terminal or
    stopped endpoints; an episode without changes has no action targets.
    bridgevla: gripper changes / stops / terminal, with the original four-frame
    stopped buffer and adjacent-terminal pruning. Stops use both neighbouring
    TCP motion intervals (real dt), not unverified OHT joint velocities. The
    gripper stability window is local; never wrap around the episode start.
    geometric: original OHT instruction / displacement / rotation / frame gaps.
    """
    options = keypoint_options(config)
    poses = array(poses, name="TCP poses")
    if poses.ndim != 2 or poses.shape[1] != 7:
        raise ValueError("Expected TCP poses[N,7]")
    n = len(poses)
    if n < 2:
        raise ValueError("An episode requires at least two aligned frames")
    observed_gripper = array(observed_gripper, (n,), "observed binary gripper")
    if not np.isin(observed_gripper, [0, 1]).all():
        raise ValueError("Expected observed binary gripper in {0,1}")
    if timestamps is not None:
        timestamps = array(timestamps, (n,), "keypoint timestamps")
        if np.any(np.diff(timestamps) <= 0):
            raise ValueError("Keypoint timestamps must be strictly increasing")
    if options["method"] == "gripper":
        return (np.flatnonzero(observed_gripper[1:] != observed_gripper[:-1]) + 1).tolist()
    if options["method"] == "bridgevla":
        if timestamps is None:
            raise ValueError("BridgeVLA keypoints require actual timestamps")
        dt = np.diff(timestamps)
        translation_speed = np.linalg.norm(np.diff(poses[:, :3], axis=0), axis=1) / dt
        rotations = Rotation.from_quat(quaternion(poses[:, 3:]))
        rotation_speed = np.rad2deg((rotations[:-1].inv() * rotations[1:]).magnitude()) / dt
        still = ((translation_speed <= options["stopping_translation_speed"]) &
                 (rotation_speed <= options["stopping_rotation_speed_degrees"]))
        result, stopped_buffer = [], 0
        for i in range(n):
            stopped = (2 <= i < n - 2 and stopped_buffer <= 0 and
                       np.all(observed_gripper[i - 2:i + 2] == observed_gripper[i]) and
                       still[i - 1] and still[i])
            stopped_buffer = 4 if stopped else stopped_buffer - 1
            if i and (observed_gripper[i] != observed_gripper[i - 1] or i == n - 1 or stopped):
                result.append(i)
        if len(result) > 1 and result[-2] == result[-1] - 1:
            result.pop(-2)
        return result

    instruction_ids = np.zeros(n) if instruction_ids is None else np.asarray(instruction_ids)
    if instruction_ids.shape != (n,):
        raise ValueError("Expected aligned instruction IDs[N]")
    boundaries = {n - 1}
    for i in range(1, n):
        if observed_gripper[i] != observed_gripper[i - 1] or instruction_ids[i] != instruction_ids[i - 1]:
            boundaries.update((max(1, i - 1), i))
    result, last = [], 0
    for i in range(1, n):
        moved = np.linalg.norm(poses[i, :3] - poses[last, :3]) >= options["max_translation"]
        rotated = rotation_error(poses[last, 3:], poses[i, 3:]) >= np.deg2rad(options["max_rotation_degrees"])
        if i in boundaries or moved or rotated or i - last >= options["max_frames"]:
            result.append(i)
            last = i
    return result


def target_labels(pose, gripper_open, bounds, rotation_classes=72, voxel_size=100):
    pose = array(pose, (7,), "TCP target").copy()
    pose[3:] = quaternion(pose[3:])
    if pose[6] < 0:
        pose[3:] *= -1
    bounds = check_bounds(pose[:3], bounds)
    if rotation_classes < 1 or voxel_size < 1 or gripper_open not in (0, 1):
        raise ValueError("Invalid action discretization")
    degrees = Rotation.from_quat(pose[3:]).as_euler("xyz", degrees=True)
    bins = np.round((degrees + 180) / (360 / rotation_classes)).astype(np.int32) % rotation_classes
    indices = np.floor((pose[:3] - bounds[:3]) / (bounds[3:] - bounds[:3]) * voxel_size).astype(np.int32)
    return {
        "action": np.r_[pose, gripper_open].astype(np.float32),
        "gripper_pose": pose.astype(np.float32),
        "rot_grip_action_indicies": np.r_[bins, gripper_open].astype(np.int32),
        "trans_action_indicies": indices,
        "ignore_collisions": np.zeros(1, dtype=np.int32),
    }
