"""Create the same camera tensor layout for training and online inference."""
import numpy as np
from .geometry import array, camera_pose_matrix, transform_matrix, backproject
from .point_filter import point_filter_options, point_cloud_mask, filter_world_points


def _camera_quaternion_order(config):
    order = config.get("camera_quaternion_order")
    if order not in ("wxyz", "xyzw"):
        raise ValueError(
            "Set camera_quaternion_order explicitly to wxyz (raw OHT v423 camera poses) "
            "or xyzw. Update the YAML passed to --config. Legacy v423 caches decoded "
            "as xyzw must be rebuilt; editing their contract cannot repair stored XYZ."
        )
    return order


def validate_data_config(config, *, resolved=False):
    bounds = config.get("scene_bounds")
    from .geometry import check_bounds
    if bounds is None:
        raise ValueError("Set scene_bounds to the world-space OHT policy workspace")
    check_bounds(np.asarray(bounds[:3]) + (np.asarray(bounds[3:]) - bounds[:3]) / 2, bounds)
    point_filter_options(config.get("point_cloud_filter"))
    transform_matrix(config.get("link_to_tcp"), "link_to_tcp")
    _camera_quaternion_order(config)
    if config.get("camera_extrinsic_direction", "camera_to_world") not in ("camera_to_world", "world_to_camera"):
        raise ValueError("camera_extrinsic_direction must be camera_to_world or world_to_camera")
    from .video import video_alignment
    video_alignment(config)
    if config.get("ee_quaternion_order") not in ("wxyz", "xyzw"):
        raise ValueError("Set ee_quaternion_order explicitly to wxyz or xyzw; rebuild legacy caches")
    if config.get("intrinsics_source", "config") not in ("config", "metadata"):
        raise ValueError("intrinsics_source must be config or metadata")
    gripper = config.get("gripper", {})
    if gripper.get("source") not in ("config", "metadata"):
        raise ValueError("Set gripper.source to metadata or config with explicit open/close")
    if resolved or gripper["source"] == "config":
        endpoints = array([gripper.get("open"), gripper.get("close")], (2,), "gripper open/close")
        if abs(endpoints[1] - endpoints[0]) < 1e-8:
            raise ValueError("Gripper open/close endpoints must differ")
    from .actions import gripper_step, keypoint_options
    gripper_step(.5, config=gripper)
    keypoint_options(config.get("keypoints"))
    cameras = config.get("cameras", {})
    if not cameras:
        raise ValueError("Camera calibration is required")
    for name, camera in cameras.items():
        if resolved or config.get("intrinsics_source", "config") == "config":
            matrix = array(camera.get("intrinsics"), (3, 3), f"{name} intrinsics")
            if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1]):
                raise ValueError(f"{name}: invalid camera intrinsics")
        if camera.get("optical_to_sensor") is None:
            raise ValueError(
                f"{name} optical_to_sensor is missing: set an explicit 4x4 transform. "
                "Use identity only when the recorded camera pose already uses OpenCV "
                "optical axes (+X right, +Y down, +Z forward). "
                "Update the YAML passed to --config; repository defaults do not "
                "automatically update an existing local config."
            )
        transform_matrix(camera.get("optical_to_sensor"), f"{name} optical_to_sensor")
    if config.get("depth", {}).get("encoding") not in ("metric", "scaled_integer", "linear_channel", "quantized"):
        raise ValueError("Configure metric depth encoding explicitly")
    metadata = config["depth"].get("metadata", False)
    if not isinstance(metadata, bool) and metadata != "reference":
        raise ValueError("depth.metadata must be true, false or reference")
    if metadata and config["depth"]["encoding"] != "quantized":
        raise ValueError("Depth metadata resolution requires encoding=quantized")
    if not isinstance(config["depth"].get("round_to_mm", False), bool):
        raise ValueError("depth.round_to_mm must be boolean")
    if config["depth"]["encoding"] == "quantized":
        from .video import validate_quantization
        validate_quantization(config["depth"])
    if config.get("depth", {}).get("kind") not in ("z", "ray"):
        raise ValueError("Specify depth.kind as z or ray")
    size = config.get("image_size", [128, 128])
    if len(size) != 2 or any(int(v) < 2 for v in size):
        raise ValueError("image_size is [height,width], each >=2")


def camera_observation(camera, rgb, depth, sensor_pose, config):
    """Canonicalize the raw camera pose once; cached extrinsics are optical->world."""
    rgb = np.asarray(rgb)
    depth = np.asarray(depth)
    if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or rgb.shape[:2] != depth.shape:
        raise ValueError(f"{camera}: expected aligned uint8 HxWx3 RGB and HxW metric depth")
    order = _camera_quaternion_order(config)
    calibration = config["cameras"][camera]
    if calibration.get("intrinsics") is None:
        raise ValueError(f"{camera}: unresolved intrinsics; use resolve_dataset_config() or the checkpoint's resolved data profile")
    K = array(calibration["intrinsics"], (3, 3), "intrinsics")
    # Same order as _canonicalize_extrinsic_pose7() followed by the reference
    # transform's OpenGL->optical conversion. EE order is independent.
    world_from_optical = camera_pose_matrix(
        sensor_pose, order, config.get("camera_extrinsic_direction", "camera_to_world"),
        calibration["optical_to_sensor"])
    depth_config = calibration.get("depth", config["depth"])
    points = backproject(depth, K, world_from_optical, depth_config["kind"],
                         tuple(depth_config.get("limits", [0.001, 10])))
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
    # Exclude background before either rendering scale, without blackening RGB
    # or changing raw depth/action labels. An individual camera may become empty.
    points = filter_world_points(points, config.get("point_cloud_filter"))
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
        valid = point_cloud_mask(cloud, config.get("point_cloud_filter"))
        supported |= bool((valid & (cloud >= bounds[:3]).all(axis=1) &
                           (cloud < bounds[3:]).all(axis=1)).any())
    if set(observation) != expected:
        raise ValueError("Unexpected cached observation fields; keep teachers and labels separate")
    if not supported:
        raise ValueError("No finite scene points inside configured scene_bounds after point-cloud filtering")
    return observation
