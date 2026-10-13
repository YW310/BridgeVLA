"""Explicit absolute TCP -> relative pose bridge; no simulator assumptions."""
import numpy as np
from scipy.spatial.transform import Rotation
from ..data.geometry import array, pose_matrix, transform_matrix
from ..action_modes import action_options, validate_action, apply_eef_delta


def relative_eef(current_world_pose, target_world_pose, world_from_base,
                 translation_frame, rotation_frame, max_translation, max_rotation):
    current_world_pose = array(current_world_pose, (7,), "measured world TCP")
    target_world_pose = array(target_world_pose, (7,), "target world TCP")
    base_from_world = np.linalg.inv(transform_matrix(world_from_base, "world_from_base"))
    current = base_from_world @ pose_matrix(current_world_pose[:3], current_world_pose[3:])
    target = base_from_world @ pose_matrix(target_world_pose[:3], target_world_pose[3:])
    delta = target[:3, 3] - current[:3, 3]
    if translation_frame == "body":
        delta = current[:3, :3].T @ delta
    elif translation_frame != "base":
        raise ValueError("translation_frame must be base or body")
    if rotation_frame == "body":
        error = current[:3, :3].T @ target[:3, :3]
    elif rotation_frame == "base":
        error = target[:3, :3] @ current[:3, :3].T
    else:
        raise ValueError("rotation_frame must be base or body")
    rotvec = Rotation.from_matrix(error).as_rotvec()
    for value, maximum in ((delta, max_translation), (rotvec, max_rotation)):
        if not np.isfinite(maximum) or maximum <= 0:
            raise ValueError("Explicit positive displacement/rotation limits are required")
        norm = np.linalg.norm(value)
        if norm > maximum:
            value *= maximum / norm
    return np.r_[delta, rotvec].astype(np.float32)


def execution_target(action, options=None, *, current_tcp=None, current_joints=None):
    """Resolve a configured command once to an absolute EEF or joint target.

    A simulator may consume native commands directly instead. Relative values
    refer to the current measurement used for that request, not a velocity.
    """
    options = action_options(options)
    value = validate_action(action, options)
    mode = options["mode"]
    if mode in ("eef", "joint"):
        return value
    if mode == "relative_eef":
        if current_tcp is None:
            raise ValueError("Relative EEF execution requires measured current_tcp")
        return np.r_[apply_eef_delta(current_tcp, value[:6], options), value[-1]].astype(np.float32)
    if current_joints is None:
        raise ValueError("Relative joint execution requires measured current_joints")
    return np.r_[array(current_joints, (6,), "current joints") + value[:6], value[-1]].astype(np.float32)
