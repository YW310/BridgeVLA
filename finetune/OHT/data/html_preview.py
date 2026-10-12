"""Offline, read-only world-XYZ ROI inspection; no browser/plotting dependency."""
import base64
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .geometry import array, check_bounds
from .cloud_source import camera_cloud
from .point_filter import point_filter_options
from .visualization import CAMERA_COLORS, OBJECT_COLORS, OBJECT_OVERLAP, current_position_label
from .object_preview import object_payload
from .object_cloud import box_mask, resolve_object_boxes


MAX_HTML_POINTS = 60_000
_PAYLOAD_MARKER = "__OHT_POINT_CLOUD_DATA__"


def html_point_limit(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 1 <= value <= MAX_HTML_POINTS:
        raise ValueError(f"HTML max_points must be an integer in [1,{MAX_HTML_POINTS}]")
    return int(value)


def point_cloud_payload(observation, config, sample, max_points=40_000, *, source="xyz", object_boxes=None):
    """Include finite cached points BEFORE any additional ROI/workspace masking.

    XYZ mode cannot recover cached NaNs. Depth mode can reconstruct ROI-masked
    points wherever cached metric depth/K/extrinsics are still valid. Display
    sampling never changes the replay, contract, point filter or action labels.
    """
    max_points = html_point_limit(max_points)
    if source not in ("xyz", "depth"):
        raise ValueError("HTML source must be xyz or depth")
    preset = None
    if object_boxes is not None:
        preset, box_source = resolve_object_boxes(object_boxes, sample, sample.get("diagnostic_contract_sha256"),
                                                  allow_empty=True)
        if box_source != source:
            raise ValueError("HTML object boxes and point cloud must use the same source")
    bounds = array(config["scene_bounds"], (6,), "scene_bounds")
    check_bounds((bounds[:3] + bounds[3:]) / 2, bounds)
    options = point_filter_options(config.get("point_cloud_filter"))
    chunks, color_chunks, camera_chunks, cameras = [], [], [], []
    if not 1 <= len(config["cameras"]) <= 255:
        raise ValueError("HTML preview requires 1-255 cameras")
    for index, camera in enumerate(config["cameras"]):
        xyz, rgb = camera_cloud(observation, config, camera, source)
        finite = np.isfinite(xyz).all(axis=1)
        chunks.append(xyz[finite])
        color_chunks.append(rgb[finite])
        camera_chunks.append(np.full(int(finite.sum()), index, np.uint8))
        cameras.append(dict(name=camera, color=list(CAMERA_COLORS.get(camera, (160, 160, 160))),
                            valid_points=int(finite.sum()), invalid_points=int((~finite).sum())))
    points, colors, camera_ids = np.concatenate(chunks), np.concatenate(color_chunks), np.concatenate(camera_chunks)
    total = len(points)
    if not total:
        if source == "xyz":
            raise ValueError("No finite cached XYZ to inspect; try source=depth if cached metric depth is valid")
        raise ValueError("No valid cached metric depth to backproject; HTML cannot invent geometry")
    object_count = None
    if preset is not None:
        union = np.zeros(total, dtype=bool)
        for box in preset:
            union |= box_mask(points, box["bounds"])
        object_count = int(union.sum())
    if total > max_points:
        if preset is None:
            selected = np.linspace(0, total - 1, max_points, dtype=int)
        else:
            # Preserve small object crops before spending the HTML budget on
            # background; exported clouds still keep every cached source point.
            groups = (np.flatnonzero(union), np.flatnonzero(~union))
            selected = []
            remaining = max_points
            for indices in groups:
                count = min(remaining, len(indices))
                if count:
                    selected.extend(indices[np.linspace(0, len(indices)-1, count, dtype=int)])
                remaining -= count
            selected = np.asarray(selected, dtype=int)
        points, colors, camera_ids = points[selected], colors[selected], camera_ids[selected]

    def encoded(values, dtype):
        return base64.b64encode(np.asarray(values, dtype=dtype).tobytes()).decode("ascii")

    def marker(value):
        return None if value is None else array(value, name="TCP/goal").reshape(-1)[:3].tolist()

    return dict(sample_id=str(sample["id"]), frame=None if sample.get("frame") is None else int(sample["frame"]),
                target_frame=None if sample.get("target_frame") is None else int(sample["target_frame"]),
                diagnostic_note=sample.get("diagnostic_note", ""), scene_bounds=bounds.tolist(),
                objects=object_payload(sample), object_source=sample.get("diagnostic_objects"),
                observation_sha256=sample.get("observation_sha256"),
                contract_sha256=sample.get("diagnostic_contract_sha256"),
                object_box_preset=preset, object_box_union_points=object_count,
                object_colors=OBJECT_COLORS, object_overlap_color=OBJECT_OVERLAP,
                point_cloud_filter=options, cameras=cameras, total_points=total, displayed_points=len(points),
                xyz=encoded(points, "<f4"), rgb=encoded(colors, "u1"), camera_ids=encoded(camera_ids, "u1"),
                geometry_source="cached_xyz" if source == "xyz" else "cached_metric_depth_backprojection",
                current_tcp=marker(sample.get("current_tcp")),
                current_position_label=current_position_label(config),
                goal=marker(sample.get("labels", {}).get("gripper_pose")),
                source_note=("Cached world XYZ only. Previously filtered/invalid points cannot be recovered. " if source == "xyz" else
                             "Display-only backprojection of cached metric depth/K/optical extrinsics, without ROI masking. "
                             "Can restore ROI-masked XYZ where cached depth is still valid, not unobserved surfaces. "
                             "Does not fix historical depth encoding, calibration or synchronization errors. ") +
                            "Counts describe the display sample, not full-dataset coverage. "
                            "Goal is future GT for diagnostics only; no policy conditioning or cache edits.")


def save_point_cloud_html(path, observation, config, sample, max_points=40_000, *, source="xyz", object_boxes=None,
                          preview_image=None):
    """Write offline HTML, optionally linking its local preview PNG; never overwrite."""
    payload = point_cloud_payload(observation, config, sample, max_points, source=source, object_boxes=object_boxes)
    if preview_image is not None:
        image_path = Path(preview_image).resolve()
        if image_path.suffix.lower() != ".png" or not image_path.is_file():
            raise ValueError("HTML preview image must be an existing PNG beside the HTML")
        relative = image_path.relative_to(Path(path).parent.resolve())
        payload["preview_image"] = quote(relative.as_posix(), safe="/")
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


def bind_point_cloud_html_contract(path, checksum):
    """Atomically bind an owned build preview to the finished replay contract."""
    path = Path(path)
    document = path.read_text(encoding="utf-8")
    marker = '"contract_sha256":null'
    if document.count(marker) != 1:
        raise ValueError("Build HTML must have exactly one unbound contract hash")
    temporary = path.with_suffix(".html.tmp")
    temporary.write_text(document.replace(marker, '"contract_sha256":'+json.dumps(checksum)), encoding="utf-8")
    temporary.replace(path)
