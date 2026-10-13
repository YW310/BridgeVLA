"""Action configuration and physical-unit targets, independent of CUDA/models."""
import numpy as np
from scipy.spatial.transform import Rotation
from .data.geometry import array, quaternion, pose_matrix, transform_matrix

MODES = ("eef", "relative_eef", "joint", "relative_joint")
DEFAULTS = dict(mode="eef", translation_frame="world", rotation_frame="body",
                world_from_base=None, joint_names=None, joint_wrap=[],
                joint_limits=None, hidden_dim=256, translation_scale=1.,
                rotation_scale=1., joint_scale=1., auxiliary_translation_weight=1.)


def action_options(options=None, mode=None):
    if options is not None and not isinstance(options, dict):
        raise ValueError("action must be a configuration mapping")
    options = options or {}
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown action options: {sorted(unknown)}")
    if mode is not None and "mode" in options and mode != options["mode"]:
        raise ValueError("action_mode and action.mode disagree")
    result = {**DEFAULTS, **options}
    if mode is not None:
        result["mode"] = mode
    if result["mode"] not in MODES:
        raise ValueError(f"action_mode must be one of {MODES}")
    for key in ("translation_frame", "rotation_frame"):
        if result[key] not in ("world", "base", "body"):
            raise ValueError(f"action.{key} must be world, base or body")
    if result["world_from_base"] is not None:
        result["world_from_base"] = transform_matrix(result["world_from_base"], "world_from_base").tolist()
    if result["mode"] == "relative_eef" and "base" in (result["translation_frame"], result["rotation_frame"]):
        if result["world_from_base"] is None:
            raise ValueError("Base-relative EEF requires explicit action.world_from_base")
    for key in ("translation_scale", "rotation_scale", "joint_scale", "auxiliary_translation_weight"):
        value = result[key]
        if isinstance(value, bool):
            raise ValueError(f"action.{key} must be finite and positive")
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"action.{key} must be finite and positive") from error
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"action.{key} must be finite and positive")
        result[key] = float(value)
    if type(result["hidden_dim"]) is not int or result["hidden_dim"] < 1:
        raise ValueError("action.hidden_dim must be a positive integer")
    names = result["joint_names"]
    if names is not None and (not isinstance(names, list) or len(names) != 6 or
                              any(not isinstance(v, str) or not v for v in names) or len(set(names)) != 6):
        raise ValueError("action.joint_names must list six unique names in observation.state order")
    wrap = result["joint_wrap"]
    if not isinstance(wrap, list) or any(type(v) is not int or not 0 <= v < 6 for v in wrap) or len(set(wrap)) != len(wrap):
        raise ValueError("action.joint_wrap must list unique continuous-joint indices in [0,5]")
    result["joint_wrap"] = list(wrap)
    result["joint_names"] = None if names is None else list(names)
    if result["joint_limits"] is not None:
        limits = array(result["joint_limits"], (6, 2), "joint_limits")
        if np.any(limits[:, 0] >= limits[:, 1]) or wrap:
            raise ValueError("Joint limits need positive extents; do not combine limits with continuous joint_wrap")
        result["joint_limits"] = limits.tolist()
    return result


def resolve_action(config):
    return action_options(config.get("action"), config.get("action_mode"))


def state_fields(options):
    mode = action_options(options)["mode"]
    return ("joint_positions",) if mode in ("joint", "relative_joint") else ("tcp_pose",) if mode == "relative_eef" else ()


def action_size(options):
    return 8 if action_options(options)["mode"] == "eef" else 7


def scales(options):
    options = action_options(options)
    if options["mode"] in ("joint", "relative_joint"):
        return np.full(6, options["joint_scale"], dtype=np.float32)
    return np.asarray([options["translation_scale"]] * 3 + [options["rotation_scale"]] * 3, dtype=np.float32)


def delta_eef(current_pose, target_pose, options):
    options = action_options(options)
    current_pose = array(current_pose, (7,), "current TCP")
    target_pose = array(target_pose, (7,), "target TCP")
    current = pose_matrix(current_pose[:3], current_pose[3:])
    target = pose_matrix(target_pose[:3], target_pose[3:])
    delta = target[:3, 3] - current[:3, 3]
    frame = options["translation_frame"]
    if frame == "body":
        delta = current[:3, :3].T @ delta
    elif frame == "base":
        delta = transform_matrix(options["world_from_base"])[:3, :3].T @ delta
    if options["rotation_frame"] == "body":
        error = current[:3, :3].T @ target[:3, :3]
    else:
        error = target[:3, :3] @ current[:3, :3].T
        if options["rotation_frame"] == "base":
            rotation = transform_matrix(options["world_from_base"])[:3, :3]
            error = rotation.T @ error @ rotation
    return np.r_[delta, Rotation.from_matrix(error).as_rotvec()].astype(np.float32)


def apply_eef_delta(current_pose, delta, options):
    """Decode an increment once, relative to the timestamped current TCP."""
    options = action_options(options)
    pose = array(current_pose, (7,), "current TCP")
    delta = array(delta, (6,), "EEF delta")
    current = pose_matrix(pose[:3], pose[3:])
    translation = delta[:3]
    if options["translation_frame"] == "body":
        translation = current[:3, :3] @ translation
    elif options["translation_frame"] == "base":
        translation = transform_matrix(options["world_from_base"])[:3, :3] @ translation
    change = Rotation.from_rotvec(delta[3:]).as_matrix()
    if options["rotation_frame"] == "body":
        rotation = current[:3, :3] @ change
    else:
        if options["rotation_frame"] == "base":
            base = transform_matrix(options["world_from_base"])[:3, :3]
            change = base @ change @ base.T
        rotation = change @ current[:3, :3]
    return np.r_[pose[:3] + translation, Rotation.from_matrix(rotation).as_quat()].astype(np.float32)


def selected_target(labels, observation, options):
    options = action_options(options)
    mode = options["mode"]
    eef = array(labels["action"], (8,), "future absolute TCP action")
    if mode == "eef":
        return eef.astype(np.float32)
    if mode == "relative_eef":
        current = observation.get("tcp_pose")
        if current is None:
            raise ValueError("relative_eef requires cached current tcp_pose; rebuild replay")
        return np.r_[delta_eef(current, eef[:7], options), eef[7]].astype(np.float32)
    if "joint_action" not in labels or "joint_positions" not in observation:
        raise ValueError(f"{mode} requires measured current/future joints; rebuild replay")
    future = array(labels["joint_action"], (7,), "future joint action")
    current = array(observation["joint_positions"], (6,), "current joint positions")
    if options["joint_limits"] is not None:
        limits = np.asarray(options["joint_limits"])
        if np.any(future[:6] < limits[:, 0]) or np.any(future[:6] > limits[:, 1]):
            raise ValueError("Joint target outside declared limits")
    if mode == "joint":
        return future.astype(np.float32)
    delta = future[:6] - current
    for index in options["joint_wrap"]:
        delta[index] = (delta[index] + np.pi) % (2 * np.pi) - np.pi
    return np.r_[delta, future[6]].astype(np.float32)


def action_descriptor(options):
    """Wire semantics; scales are training normalization, never controller gain."""
    options = action_options(options)
    mode = options["mode"]
    result = dict(mode=mode, size=action_size(options), gripper="open01",
                  reference_point="tcp" if "eef" in mode else "joint",
                  units="metres_radians" if "eef" in mode else "radians")
    if mode == "eef":
        result.update(frame="world", quaternion_order="xyzw")
    elif mode == "relative_eef":
        result.update(translation_frame=options["translation_frame"], rotation_frame=options["rotation_frame"],
                      rotation_representation="rotvec", world_from_base=options["world_from_base"])
    else:
        result.update(joint_order="observation.state[:6]", joint_names=options["joint_names"],
                      joint_wrap=options["joint_wrap"])
    return result


def validate_action(action, options):
    options = action_options(options)
    value = array(action, (action_size(options),), "policy action").copy()
    if value[-1] not in (0, 1):
        raise ValueError("gripper_open must be binary")
    if options["mode"] == "eef":
        value[3:7] = quaternion(value[3:7])
    return value.astype(np.float32)
