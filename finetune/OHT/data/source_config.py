"""Resolve per-dataset exporter metadata once, before writing any replay."""
from copy import deepcopy
import json
from pathlib import Path
import re
import numpy as np
from .common import inside, file_digest
from .observation import validate_data_config
from .video import validate_quantization


def _name(value):
    return re.sub(r"[^a-z0-9]+", ".", str(value).lower()).strip(".")


def _intrinsic(metadata, camera):
    if not isinstance(metadata, dict) or not isinstance(metadata.get("cameras", {}), dict):
        raise ValueError("camera_intrinsics.json must contain a camera mapping")
    top = {_name(key): value for key, value in metadata.items()}
    cameras = {_name(key): value for key, value in metadata.get("cameras", {}).items()}
    entry = cameras.get(_name(camera), top.get(_name(camera)))
    if isinstance(entry, dict):
        entry = entry.get("intrinsic", entry.get("intrinsics", entry.get("K")))
    if entry is None:
        entry = top.get(_name(camera + "_intrinsic"))
    matrix = np.asarray(entry, dtype=float)
    if matrix.shape == (9,):
        matrix = matrix.reshape(3, 3)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(f"Missing/invalid {camera} K in meta/camera_intrinsics.json")
    return matrix.tolist()


def _boolean(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.lower().strip()
    if value in (0, "0", "false", "no", "off"):
        return False
    if value in (1, "1", "true", "yes", "on"):
        return True
    raise ValueError(f"Invalid depth use_log: {value!r}")


def resolve_dataset_config(dataset, config):
    """Return canonical, self-contained config plus metadata file fingerprints.

    Explicit YAML remains supported. No per-episode range fitting, quaternion
    guessing or inferred camera focal lengths. Depth metadata overrides the
    explicit YAML quantization defaults exactly as in the reference converter.
    """
    dataset = Path(dataset)
    result, sources, loaded = deepcopy(config), {}, {}

    def read(relative):
        if relative not in loaded:
            path = inside(dataset, relative)
            loaded[relative] = json.loads(path.read_text(encoding="utf-8"))
            sources[relative] = file_digest(path)
        return loaded[relative]

    if result.get("intrinsics_source", "config") == "metadata":
        metadata = read("meta/camera_intrinsics.json")
        for camera, calibration in result["cameras"].items():
            calibration["intrinsics"] = _intrinsic(metadata, camera)
        result["intrinsics_source"] = "config"

    gripper = result["gripper"]
    if gripper["source"] == "metadata":
        # Same dataset-global endpoints as the reference converter: smaller
        # motor position is open. Never fit min/max to an individual episode.
        candidates = (("meta/info.json", ("features_stats", "observation.state")),
                      ("meta/stats.json", ("observation.state",)))
        for relative, keys in candidates:
            if not inside(dataset, relative).is_file():
                continue
            node = read(relative)
            for key in keys:
                node = node.get(key, {})
            if "min" not in node or "max" not in node:
                continue
            low = np.asarray(node["min"], dtype=float).reshape(-1)
            high = np.asarray(node["max"], dtype=float).reshape(-1)
            if low.size != 7 or high.size != 7:
                raise ValueError(f"{relative}: expected observation.state min/max width 7")
            if not np.isfinite([low[-1], high[-1]]).all() or low[-1] >= high[-1]:
                raise ValueError(f"{relative}: invalid gripper min/max endpoints")
            gripper.update(source="config", open=float(low[-1]), close=float(high[-1]))
            break
        else:
            raise ValueError("Gripper endpoints unavailable in meta stats/info; set gripper.source=config and explicit open/close")

    depth = result["depth"]
    if depth.pop("metadata", False):
        if depth["encoding"] != "quantized":
            raise ValueError("Depth metadata resolution requires encoding=quantized")
        info = read("meta/info.json")
        if not isinstance(info, dict) or not isinstance(info.get("features", {}), dict):
            raise ValueError("meta/info.json must contain a features mapping")
        for camera, calibration in result["cameras"].items():
            spec = deepcopy(depth)
            feature = info.get("features", {}).get(f"observation.depth.{camera}", {})
            metadata = feature.get("info", {}) if isinstance(feature, dict) else {}
            if not isinstance(metadata, dict):
                metadata = {}
            for key in ("depth_min", "depth_max", "shift", "use_log", "qmax", "pix_fmt"):
                value = metadata.get("video." + key, metadata.get(key))
                if value is not None:
                    spec["pixel_format" if key == "pix_fmt" else key] = _boolean(value) if key == "use_log" else value
            validate_quantization(spec)
            if spec.get("pixel_format") != "gray12le":
                raise ValueError(f"{camera}: quantized depth requires native gray12le")
            calibration["depth"] = spec
    validate_data_config(result, resolved=True)
    return result, sources


def sample_data_config(contract, sample):
    profiles = contract.get("source_data_configs", {})
    if profiles:
        if sample.get("data_profile") not in profiles:
            raise ValueError(f"Unknown/missing source data profile: {sample.get('data_profile')}")
        return profiles[sample["data_profile"]]["data_config"]
    return contract["data_config"]
