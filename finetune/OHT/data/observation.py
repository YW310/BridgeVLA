"""Create the same camera tensor layout for training and online inference."""
import numpy as np
from .geometry import array, pose_matrix, transform_matrix, backproject


def validate_data_config(config):
    bounds = config.get("scene_bounds")
    from .geometry import check_bounds
    if bounds is None:
        raise ValueError("Set scene_bounds to the world-space OHT policy workspace")
    check_bounds(np.asarray(bounds[:3]) + (np.asarray(bounds[3:]) - bounds[:3]) / 2, bounds)
    transform_matrix(config.get("link_to_tcp"), "link_to_tcp")
    cameras = config.get("cameras", {})
    if not cameras:
        raise ValueError("Camera calibration is required")
    for name, camera in cameras.items():
        array(camera.get("intrinsics"), (3, 3), f"{name} intrinsics")
        if camera.get("optical_to_sensor") is None:
            raise ValueError(
                f"{name} optical_to_sensor is missing: set an explicit 4x4 transform. "
                "Use identity only when the recorded camera pose already uses OpenCV "
                "optical axes (+X right, +Y down, +Z forward). "
                "Update the YAML passed to --config; repository defaults do not "
                "automatically update an existing local config."
            )
        transform_matrix(camera.get("optical_to_sensor"), f"{name} optical_to_sensor")
    if config.get("depth", {}).get("encoding") not in ("metric", "scaled_integer", "linear_channel"):
        raise ValueError("Configure metric depth encoding explicitly")
    if config.get("depth", {}).get("kind") not in ("z", "ray"):
        raise ValueError("Specify depth.kind as z or ray")
    size = config.get("image_size", [128, 128])
    if len(size) != 2 or any(int(v) < 2 for v in size):
        raise ValueError("image_size is [height,width], each >=2")


def camera_observation(camera, rgb, depth, sensor_pose, config):
    rgb = np.asarray(rgb)
    depth = np.asarray(depth)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or rgb.shape[:2] != depth.shape:
        raise ValueError(f"{camera}: expected aligned uint8 HxWx3 RGB and HxW metric depth")
    calibration = config["cameras"][camera]
    K = array(calibration["intrinsics"], (3, 3), "intrinsics")
    sensor_pose = array(sensor_pose, (7,), "camera sensor pose")
    world_from_optical = pose_matrix(sensor_pose[:3], sensor_pose[3:]) @ transform_matrix(calibration["optical_to_sensor"])
    points = backproject(depth, K, world_from_optical, config["depth"]["kind"],
                         tuple(config["depth"].get("limits", [0.001, 10])))
    height, width = map(int, config["image_size"])
    if height > depth.shape[0] or width > depth.shape[1]:
        raise ValueError("Upsampling RGB-D is not supported")
    # Exact integer-stride sampling keeps the stored K consistent with pixels.
    if depth.shape[0] % height or depth.shape[1] % width:
        raise ValueError("image_size must divide source H,W; use [120,160] for 480x640")
    sy, sx = depth.shape[0] // height, depth.shape[1] // width
    rgb, depth, points = rgb[::sy, ::sx], depth[::sy, ::sx], points[::sy, ::sx]
    K = K.copy()
    K[0] /= sx
    K[1] /= sy
    if not np.isfinite(points).all(axis=-1).any():
        raise ValueError(f"{camera}: no valid metric depth")
    return {
        f"{camera}_rgb": rgb.transpose(2, 0, 1),
        f"{camera}_depth": depth[None].astype(np.float32),
        f"{camera}_point_cloud": points.transpose(2, 0, 1),
        f"{camera}_camera_intrinsics": K.astype(np.float32),
        f"{camera}_camera_extrinsics": world_from_optical.astype(np.float32),
    }


def validate_observation(observation, config):
    """Validate cached/online tensors before they reach the existing renderer."""
    height, width = config["image_size"]
    array(observation["low_dim_state"], (4,), "low_dim_state")
    bounds = np.asarray(config["scene_bounds"])
    supported = False
    expected = {"low_dim_state"}
    for camera in config["cameras"]:
        expected.update(f"{camera}_{suffix}" for suffix in (
            "rgb", "depth", "point_cloud", "camera_intrinsics", "camera_extrinsics"))
        rgb = np.asarray(observation[f"{camera}_rgb"])
        depth = np.asarray(observation[f"{camera}_depth"])
        points = np.asarray(observation[f"{camera}_point_cloud"])
        if rgb.shape != (3, height, width) or rgb.dtype != np.uint8:
            raise ValueError(f"{camera}: RGB must be uint8 CHW at configured resolution")
        if depth.shape != (1, height, width) or points.shape != (3, height, width):
            raise ValueError(f"{camera}: depth/XYZ shape mismatch")
        if np.isinf(depth).any() or np.isinf(points).any():
            raise ValueError(f"{camera}: use NaN for invalid depth/XYZ, never infinity")
        array(observation[f"{camera}_camera_intrinsics"], (3, 3), "intrinsics")
        transform_matrix(observation[f"{camera}_camera_extrinsics"])
        cloud = points.reshape(3, -1).T
        valid = np.isfinite(cloud).all(axis=1)
        supported |= bool((valid & (cloud >= bounds[:3]).all(axis=1) &
                           (cloud < bounds[3:]).all(axis=1)).any())
    if set(observation) != expected:
        raise ValueError("Unexpected cached observation fields; keep teachers and labels separate")
    if not supported:
        raise ValueError("No finite scene points inside configured scene_bounds")
    return observation
