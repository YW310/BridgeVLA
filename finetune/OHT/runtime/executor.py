"""Explicit absolute TCP -> relative pose bridge; no simulator assumptions."""
import numpy as np
from scipy.spatial.transform import Rotation
from ..data.geometry import array, pose_matrix, transform_matrix


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
