"""Approximate object point clouds from world-axis boxes, not instance masks."""
from pathlib import Path

import numpy as np

from .cloud_source import camera_cloud
from .common import digest, write_json
from .geometry import array
from .object_preview import object_payload


def resolve_object_boxes(specification, sample, contract_sha256, *, allow_empty=False):
    if not isinstance(specification, dict):
        raise ValueError("Object boxes configuration must be a JSON object")
    if (specification.get("schema") != "oht_object_boxes_v1" or
            type(specification.get("frame")) is not int or
            specification.get("sample_id") != sample["id"] or specification.get("frame") != sample.get("frame")):
        raise ValueError("Object boxes must match the current sample ID/frame and schema=oht_object_boxes_v1")
    for key, expected in (("observation_sha256", sample.get("observation_sha256")),
                          ("contract_sha256", contract_sha256)):
        if specification.get(key) is not None and specification[key] != expected:
            raise ValueError(f"Object boxes {key} does not match replay")
    source = specification.get("source")
    if source not in ("xyz", "depth"):
        raise ValueError("Object boxes source must be xyz or depth")
    objects = object_payload(sample)
    entries = specification.get("objects")
    if not isinstance(entries, list) or (not entries and not allow_empty):
        raise ValueError("Select at least one object box")
    result, seen = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each object box must be a JSON object")
        index = entry.get("index")
        if type(index) is not int or not 0 <= index < len(objects) or index in seen:
            raise ValueError("Object box indices must be available and unique")
        seen.add(index)
        size = array(entry.get("size"), (3,), "object box full XYZ size")
        if np.any(size <= 0):
            raise ValueError("Object box size must be positive world metres")
        center = np.asarray(objects[index]["position"])
        bounds = array(np.r_[center-size/2, center+size/2], (6,), "object box bounds")
        if np.any(bounds[:3] >= bounds[3:]):
            raise ValueError("Object box must have positive extents")
        name = entry.get("name", objects[index]["name"])
        if not isinstance(name, str) or not name.strip() or len(name) > 128:
            raise ValueError("Object box name must be nonempty text, at most 128 characters")
        result.append(dict(index=index, name=name, center=center.tolist(), size=size.tolist(), bounds=bounds.tolist()))
    return result, source


def box_mask(points, bounds):
    bounds = np.asarray(bounds)
    return np.isfinite(points).all(axis=1) & (points >= bounds[:3]).all(axis=1) & (points < bounds[3:]).all(axis=1)


def _write_ply(path, points, colors):
    header = ("ply\nformat ascii 1.0\ncomment Approximate world-axis box crop; not instance segmentation\n"
              f"element vertex {len(points)}\nproperty double x\nproperty double y\nproperty double z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with path.open("x", encoding="ascii", newline="\n") as stream:
        stream.write(header)
        for point, color in zip(points, colors):
            stream.write(" ".join([*(format(float(v), ".17g") for v in point), *(str(int(v)) for v in color)])+"\n")


def export_object_clouds(output, observation, config, sample, specification, contract_sha256, validation,
                         *, allow_empty=False):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new object cloud directory: {output}")
    boxes, source = resolve_object_boxes(specification, sample, contract_sha256, allow_empty=allow_empty)
    # Keep every finite point from each camera, before HTML display sampling.
    chunks, color_chunks, ids = [], [], []
    cameras = list(config["cameras"])
    for index, camera in enumerate(cameras):
        points, colors = camera_cloud(observation, config, camera, source)
        finite = np.isfinite(points).all(axis=1)
        chunks.append(points[finite])
        color_chunks.append(colors[finite])
        ids.append(np.full(int(finite.sum()), index, np.int32))
    points, colors, camera_ids = np.concatenate(chunks), np.concatenate(color_chunks), np.concatenate(ids)
    report = dict(schema="oht_object_clouds_v1", approximate=True, method="world_axis_box",
                  sample_id=sample["id"], frame=sample["frame"], units="world_metres", source=source,
                  contract_sha256=contract_sha256, observation_sha256=sample.get("observation_sha256"),
                  specification_sha256=digest(specification), validation=validation, cameras=cameras,
                  diagnostic_objects=sample.get("diagnostic_objects"), total_source_points=len(points),
                  display_sampling=False, objects=[],
                  note="Box crops can contain background/neighbours and overlap. No instance labels, deduplication, "
                       "unobserved surfaces or replay edits. Cached resolution/depth/calibration still apply.")
    output.mkdir(parents=True, exist_ok=False)
    for box in boxes:
        keep = box_mask(points, box["bounds"])
        xyz, rgb, provenance = points[keep], colors[keep], camera_ids[keep]
        stem = f"object_{box['index']}"
        _write_ply(output / (stem+".ply"), xyz, rgb)
        with (output / (stem+".npz")).open("xb") as stream:
            np.savez_compressed(stream, xyz=xyz, rgb=rgb, camera_ids=provenance)
        report["objects"].append(dict(box, points=int(keep.sum()), valid=bool(keep.any()),
                                      ply=stem+".ply", npz=stem+".npz",
                                      per_camera={camera: int((provenance == i).sum()) for i, camera in enumerate(cameras)}))
    write_json(output / "object-clouds.json", report)
    return report
