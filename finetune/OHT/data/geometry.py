"""World/optical/TCP geometry. Quaternions here are always xyzw."""
import numpy as np
from scipy.spatial.transform import Rotation


def array(value, shape=None, name="value"):
    value = np.asarray(value, dtype=np.float64)
    if (shape is not None and value.shape != shape) or not np.isfinite(value).all():
        raise ValueError(f"{name}: expected finite array {shape}, got {value.shape}")
    return value


def quaternion(value):
    value = array(value, name="quaternion")
    if value.shape[-1:] != (4,):
        raise ValueError("Expected xyzw quaternion")
    norms = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Zero quaternion")
    return value / norms


def pose_matrix(position, orientation):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quaternion(orientation)).as_matrix()
    result[:3, 3] = array(position, (3,), "position")
    return result


def transform_matrix(value, name="transform"):
    value = array(value, (4, 4), name)
    if not np.allclose(value[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError(f"{name}: invalid homogeneous row")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError(f"{name}: expected a rigid rotation")
    return value


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


def backproject(depth, intrinsics, world_from_optical, kind="z", limits=(0.001, 10.0)):
    """Unproject metric depth; Z-depth keeps K^-1[u,v,1] unnormalized.

    The OHT reference pointcloud transform uses Z=depth. Normalize the ray
    only for an explicitly configured Euclidean camera-to-point distance.
    """
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("Metric depth must be HxW")
    intrinsics = array(intrinsics, (3, 3), "intrinsics")
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0 or not np.allclose(intrinsics[2], [0, 0, 1]):
        raise ValueError("Invalid camera intrinsics")
    transform = transform_matrix(world_from_optical)
    v, u = np.indices(depth.shape)
    rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(intrinsics).T
    if kind == "ray":
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    elif kind != "z":
        raise ValueError("depth kind must be z or ray")
    valid = np.isfinite(depth) & (depth >= limits[0]) & (depth <= limits[1])
    points = (rays * depth[..., None]) @ transform[:3, :3].T + transform[:3, 3]
    points[~valid] = np.nan
    return points.astype(np.float32)


def rotation_error(first, second):
    return (Rotation.from_quat(quaternion(first)).inv() * Rotation.from_quat(quaternion(second))).magnitude()


def check_bounds(position, bounds):
    bounds = array(bounds, (6,), "scene_bounds")
    if np.any(bounds[3:] <= bounds[:3]):
        raise ValueError("scene_bounds must have positive extents")
    position = array(position, name="target position")
    if np.any(position < bounds[:3]) or np.any(position >= bounds[3:]):
        raise ValueError(f"Target outside scene_bounds: {position.tolist()}")
    return bounds
