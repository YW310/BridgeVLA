"""Optional manual world-XYZ filtering, independent of the action workspace."""
import numpy as np
from .geometry import array


def point_filter_options(config=None):
    """Canonical JSON/YAML options; an absent block preserves old behaviour."""
    if config is not None and not isinstance(config, dict):
        raise ValueError("point_cloud_filter must be a configuration mapping")
    config = config or {}
    unknown = set(config) - {"enabled", "keep_bounds", "exclude_boxes"}
    if unknown:
        raise ValueError(f"Unsupported point_cloud_filter options: {sorted(unknown)}")
    enabled = config.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("point_cloud_filter.enabled must be boolean")

    def box(value, name):
        try:
            bounds = array(value, (6,), f"point_cloud_filter.{name}")
        except (TypeError, ValueError) as error:
            raise ValueError(f"point_cloud_filter.{name} must contain six finite numbers") from error
        if np.any(bounds[3:] <= bounds[:3]):
            raise ValueError(f"point_cloud_filter.{name} must have positive extents")
        return bounds.tolist()

    keep = config.get("keep_bounds")
    keep = None if keep is None else box(keep, "keep_bounds")
    exclude = config.get("exclude_boxes", [])
    if not isinstance(exclude, (list, tuple)):
        raise ValueError("point_cloud_filter.exclude_boxes must be a list of six-value boxes")
    return dict(enabled=enabled, keep_bounds=keep,
                exclude_boxes=[box(value, f"exclude_boxes[{i}]") for i, value in enumerate(exclude)])


def point_cloud_mask(points, config=None):
    """Keep finite XYZ inside the optional ROI and outside all excluded boxes.

    Boxes are world metres [xmin,ymin,zmin,xmax,ymax,zmax], lower-inclusive,
    upper-exclusive. They neither alter scene_bounds nor use future labels.
    """
    options = point_filter_options(config)
    points = np.asarray(points)
    if points.shape[-1:] != (3,):
        raise ValueError("Expected world points[...,3]")
    valid = np.isfinite(points).all(axis=-1)
    if not options["enabled"]:
        return valid

    def inside(bounds):
        return (points >= bounds[:3]).all(axis=-1) & (points < bounds[3:]).all(axis=-1)

    if options["keep_bounds"] is not None:
        valid &= inside(options["keep_bounds"])
    for bounds in options["exclude_boxes"]:
        valid &= ~inside(bounds)
    return valid


def filter_world_points(points, config=None):
    """Mark excluded XYZ as NaN, preserving organized pixel layout and inputs."""
    options = point_filter_options(config)
    if not options["enabled"]:
        return points
    points = np.asarray(points)
    if not np.issubdtype(points.dtype, np.floating):
        raise ValueError("point_cloud_filter requires floating-point world XYZ")
    if np.isinf(points).any():
        raise ValueError("Use NaN for invalid XYZ, never infinity")
    keep = point_cloud_mask(points, options)
    if keep.all():
        return points
    result = points.copy()
    result[~keep] = np.nan
    return result


def filter_observation_points(observation, config):
    """Online counterpart of camera_observation; leave RGB/depth/state alone."""
    options = point_filter_options(config.get("point_cloud_filter"))
    if not options["enabled"]:
        return observation
    result = dict(observation)
    for camera in config["cameras"]:
        name = f"{camera}_point_cloud"
        points = np.asarray(observation[name])
        if points.ndim != 3 or points.shape[0] != 3:
            raise ValueError(f"{camera}: expected point_cloud[3,H,W]")
        result[name] = filter_world_points(points.transpose(1, 2, 0), options).transpose(2, 0, 1)
    return result
