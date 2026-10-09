"""Offline, read-only world-XYZ ROI inspection; no browser/plotting dependency."""
import base64
import json
from pathlib import Path

import numpy as np

from .geometry import array, check_bounds, backproject
from .point_filter import point_filter_options
from .visualization import CAMERA_COLORS


MAX_HTML_POINTS = 60_000
_PAYLOAD_MARKER = "__OHT_POINT_CLOUD_DATA__"


def html_point_limit(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 1 <= value <= MAX_HTML_POINTS:
        raise ValueError(f"HTML max_points must be an integer in [1,{MAX_HTML_POINTS}]")
    return int(value)


def point_cloud_payload(observation, config, sample, max_points=40_000, *, source="xyz"):
    """Include finite cached points BEFORE any additional ROI/workspace masking.

    XYZ mode cannot recover cached NaNs. Depth mode can reconstruct ROI-masked
    points wherever cached metric depth/K/extrinsics are still valid. Display
    sampling never changes the replay, contract, point filter or action labels.
    """
    max_points = html_point_limit(max_points)
    if source not in ("xyz", "depth"):
        raise ValueError("HTML source must be xyz or depth")
    bounds = array(config["scene_bounds"], (6,), "scene_bounds")
    check_bounds((bounds[:3] + bounds[3:]) / 2, bounds)
    options = point_filter_options(config.get("point_cloud_filter"))
    chunks, color_chunks, camera_chunks, cameras = [], [], [], []
    if not 1 <= len(config["cameras"]) <= 255:
        raise ValueError("HTML preview requires 1-255 cameras")
    for index, camera in enumerate(config["cameras"]):
        cloud = np.asarray(observation[f"{camera}_point_cloud"])
        rgb = np.asarray(observation[f"{camera}_rgb"])
        if cloud.ndim != 3 or cloud.shape[0] != 3 or rgb.shape != cloud.shape or rgb.dtype != np.uint8:
            raise ValueError(f"{camera}: expected aligned point_cloud/RGB[3,H,W] with uint8 RGB")
        if source == "depth":
            names = [f"{camera}_{suffix}" for suffix in ("depth", "camera_intrinsics", "camera_extrinsics")]
            if any(name not in observation for name in names):
                raise ValueError(f"{camera}: depth HTML requires cached metric depth/K/optical extrinsics; use source=xyz otherwise")
            depth = np.asarray(observation[names[0]])
            if depth.shape != (1, *rgb.shape[1:]) or not np.issubdtype(depth.dtype, np.floating):
                raise ValueError(f"{camera}: HTML expects already-metric depth[1,H,W], never raw integer codes")
            spec = config["cameras"][camera].get("depth", config.get("depth"))
            if not spec or "kind" not in spec:
                raise ValueError(f"{camera}: depth HTML requires the cached depth kind contract")
            cloud = backproject(depth[0], observation[names[1]], observation[names[2]],
                                spec["kind"], tuple(spec.get("limits", [.001, 10.]))).transpose(2, 0, 1)
        xyz = cloud.reshape(3, -1).T
        finite = np.isfinite(xyz).all(axis=1)
        chunks.append(xyz[finite])
        color_chunks.append(rgb.reshape(3, -1).T[finite])
        camera_chunks.append(np.full(int(finite.sum()), index, np.uint8))
        cameras.append(dict(name=camera, color=list(CAMERA_COLORS.get(camera, (160, 160, 160))),
                            valid_points=int(finite.sum()), invalid_points=int((~finite).sum())))
    points, colors, camera_ids = np.concatenate(chunks), np.concatenate(color_chunks), np.concatenate(camera_chunks)
    total = len(points)
    if not total:
        if source == "xyz":
            raise ValueError("No finite cached XYZ to inspect; try source=depth if cached metric depth is valid")
        raise ValueError("No valid cached metric depth to backproject; HTML cannot invent geometry")
    if total > max_points:
        selected = np.linspace(0, total - 1, max_points, dtype=int)
        points, colors, camera_ids = points[selected], colors[selected], camera_ids[selected]

    def encoded(values, dtype):
        return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode("ascii")

    def marker(value):
        return None if value is None else array(value, name="TCP/goal").reshape(-1)[:3].tolist()

    return dict(sample_id=str(sample["id"]), frame=int(sample["frame"]),
                target_frame=int(sample["target_frame"]), scene_bounds=bounds.tolist(),
                point_cloud_filter=options, cameras=cameras, total_points=total, displayed_points=len(points),
                xyz=encoded(points, "<f4"), rgb=encoded(colors, "u1"), camera_ids=encoded(camera_ids, "u1"),
                geometry_source="cached_xyz" if source == "xyz" else "cached_metric_depth_backprojection",
                current_tcp=marker(sample.get("current_tcp")),
                goal=marker(sample["labels"]["gripper_pose"]),
                source_note=("Cached world XYZ only. Previously filtered/invalid points cannot be recovered. " if source == "xyz" else
                             "Display-only backprojection of cached metric depth/K/optical extrinsics, without ROI masking. "
                             "Can restore ROI-masked XYZ where cached depth is still valid, not unobserved surfaces. "
                             "Does not fix historical depth encoding, calibration or synchronization errors. ") +
                            "Counts describe the display sample, not full-dataset coverage. "
                            "Goal is future GT for diagnostics only; no policy conditioning or cache edits.")


def save_point_cloud_html(path, observation, config, sample, max_points=40_000, *, source="xyz"):
    """Write one self-contained, offline HTML; refuse to overwrite any file."""
    payload = point_cloud_payload(observation, config, sample, max_points, source=source)
    template = Path(__file__).with_name("assets").joinpath("point-cloud-roi.html").read_text(encoding="utf-8")
    if template.count(_PAYLOAD_MARKER) != 1:
        raise ValueError("Invalid HTML preview template")
    # The data block is JSON, not executable JS. Escape '<' so even an untrusted
    # sample/camera name cannot close the script tag; UI uses textContent only.
    serialized = json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    document = template.replace(_PAYLOAD_MARKER, serialized)
    if len(document.encode("utf-8")) >= 2_000_000:
        raise ValueError("HTML preview exceeds 2 MB; reduce --max-points")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(document)
    return path
