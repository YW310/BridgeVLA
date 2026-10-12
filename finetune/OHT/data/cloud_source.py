"""Aligned cached XYZ/RGB or metric-depth reconstruction for diagnostics."""
import numpy as np

from .geometry import backproject


def camera_cloud(observation, config, camera, source="xyz"):
    if source not in ("xyz", "depth"):
        raise ValueError("Cloud source must be xyz or depth")
    cloud = np.asarray(observation[f"{camera}_point_cloud"])
    rgb = np.asarray(observation[f"{camera}_rgb"])
    if cloud.ndim != 3 or cloud.shape[0] != 3 or rgb.shape != cloud.shape or rgb.dtype != np.uint8:
        raise ValueError(f"{camera}: expected aligned point_cloud/RGB[3,H,W] with uint8 RGB")
    if source == "depth":
        names = [f"{camera}_{suffix}" for suffix in ("depth", "camera_intrinsics", "camera_extrinsics")]
        if any(name not in observation for name in names):
            raise ValueError(f"{camera}: depth requires cached metric depth/K/optical extrinsics; use source=xyz otherwise")
        depth = np.asarray(observation[names[0]])
        if depth.shape != (1, *rgb.shape[1:]) or not np.issubdtype(depth.dtype, np.floating):
            raise ValueError(f"{camera}: expects already-metric depth[1,H,W], never raw integer codes")
        spec = config["cameras"][camera].get("depth", config.get("depth"))
        if not spec or "kind" not in spec:
            raise ValueError(f"{camera}: depth requires the cached depth kind contract")
        cloud = backproject(depth[0], observation[names[1]], observation[names[2]],
                            spec["kind"], tuple(spec.get("limits", [.001, 10.]))).transpose(2, 0, 1)
    return cloud.reshape(3, -1).T, rgb.reshape(3, -1).T
