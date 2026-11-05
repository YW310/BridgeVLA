"""World/optical/TCP geometry. Quaternions here are always xyzw."""
import numpy as np
from scipy.spatial.transform import Rotation
from ...bridgevla.geometry.numpy_ops import (
    array, backproject, pose_matrix, quaternion, rotation_error, transform_matrix,
    validate_intrinsics,
)


def object_pose_arrays(positions, orientations):
    """Validate flattened object poses for one frame or an entire episode."""
    positions = array(positions, name="objects_pos")
    orientations = array(orientations, name="objects_quat")
    if positions.ndim not in (1, 2) or positions.shape[-1] % 3:
        raise ValueError("objects_pos must have 3 values per object at each frame")
    count = positions.shape[-1] // 3
    leading = positions.shape[:-1]
    if orientations.shape != (*leading, count * 4):
        raise ValueError("objects_pos/objects_quat must describe the same objects and frames")
    positions = positions.reshape(*leading, count, 3)
    orientations = orientations.reshape(*leading, count, 4)
    quaternion(orientations)  # Norm validation is independent of component order.
    return positions, orientations


def camera_pose_matrix(sensor_pose, quaternion_order, direction, optical_to_sensor):
    """Canonical optical->world transform, matching the reference converter.

    Invert a world->sensor source BEFORE converting the sensor's optical axes.
    OpenGL conversion is a right multiplication; never transpose R alone or
    pre-multiply the axis flip in world coordinates.
    """
    sensor_pose = array(sensor_pose, (7,), "camera sensor pose")
    if quaternion_order not in ("wxyz", "xyzw"):
        raise ValueError("camera quaternion order must be wxyz or xyzw")
    orientation = sensor_pose[3:]
    if quaternion_order == "wxyz":
        orientation = orientation[[1, 2, 3, 0]]
    source = pose_matrix(sensor_pose[:3], orientation)
    if direction == "world_to_camera":
        source = np.linalg.inv(source)
    elif direction != "camera_to_world":
        raise ValueError("camera_extrinsic_direction must be camera_to_world or world_to_camera")
    return source @ transform_matrix(optical_to_sensor, "optical_to_sensor")


def tcp_pose(position, orientation, link_to_tcp):
    matrix = pose_matrix(position, orientation) @ transform_matrix(link_to_tcp)
    return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()]


def check_bounds(position, bounds):
    bounds = array(bounds, (6,), "scene_bounds")
    if np.any(bounds[3:] <= bounds[:3]):
        raise ValueError("scene_bounds must have positive extents")
    position = array(position, name="target position")
    if np.any(position < bounds[:3]) or np.any(position >= bounds[3:]):
        raise ValueError(f"Target outside scene_bounds: {position.tolist()}")
    return bounds
