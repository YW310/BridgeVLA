"""Resolve per-dataset exporter metadata once, before writing any replay."""
from copy import deepcopy
import json
from pathlib import Path
import re
import numpy as np
from .common import inside, file_digest
from .actions import keypoint_options
from .point_filter import point_filter_options
from .observation import validate_data_config
from .video import validate_quantization, video_alignment


def _name(value):
    return re.sub(r"[^a-z0-9]+", ".", str(value).lower()).strip(".")


def _intrinsic(metadata, camera, source="meta/camera_intrinsics.json"):
    if isinstance(metadata, dict):
        metadata = metadata.get("camera_intrinsics", metadata)
    if not isinstance(metadata, dict) or not isinstance(metadata.get("cameras", {}), dict):
        raise ValueError(f"{source} must contain a camera intrinsic mapping")
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
        raise ValueError(f"Missing/invalid {camera} K in {source}")
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
    guessing or inferred camera focal lengths. Quantized depth metadata must
    specify the complete writer contract; converter fallback values do not
    establish how a dataset's numeric depth was encoded.
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
        relative = ("meta/camera_intrinsics.json"
                    if inside(dataset, "meta/camera_intrinsics.json").is_file()
                    else "meta/info.json")
        metadata = read(relative)
        if relative == "meta/info.json" and "camera_intrinsics" not in metadata:
            raise FileNotFoundError(
                "Camera intrinsics unavailable: need meta/camera_intrinsics.json "
                "or meta/info.json camera_intrinsics; no focal-length defaults are used")
        for camera, calibration in result["cameras"].items():
            calibration["intrinsics"] = _intrinsic(metadata, camera, relative)
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
            keys = ("depth_min", "depth_max", "shift", "use_log", "qmax", "pix_fmt")
            missing = [key for key in keys
                       if metadata.get("video." + key, metadata.get(key)) is None]
            if missing:
                raise ValueError(
                    f"{camera}: missing depth quantization metadata {missing} in "
                    f"meta/info.json features.observation.depth.{camera}.info. "
                    "RGB codec metadata does not specify numeric depth encoding; "
                    "converter log defaults are not evidence of the writer contract. "
                    "For the confirmed millimetre export, replace the entire depth "
                    "block with encoding=scaled_integer, scale=0.001, offset=0, "
                    "kind=z (reference pinhole projection), invalid_values=[0,4095], metadata=false. For an "
                    "independently verified quantized writer, set metadata=false "
                    "and supply all quantization parameters explicitly. Update "
                    "the file passed to --config, not an existing replay contract.")
            for key in keys:
                value = metadata.get("video." + key, metadata.get(key))
                spec["pixel_format" if key == "pix_fmt" else key] = _boolean(value) if key == "use_log" else value
            validate_quantization(spec)
            if spec.get("pixel_format") != "gray12le":
                raise ValueError(f"{camera}: quantized depth requires native gray12le")
            calibration["depth"] = spec
    # Persist extraction and XYZ-filter defaults so future YAML changes cannot
    # reinterpret buffer provenance. Legacy geometric extraction stays explicit.
    result["keypoints"] = keypoint_options(result.get("keypoints"))
    result["point_cloud_filter"] = point_filter_options(result.get("point_cloud_filter"))
    result["video_alignment"] = video_alignment(result)
    result.setdefault("camera_extrinsic_direction", "camera_to_world")
    validate_data_config(result, resolved=True)
    return result, sources


def sample_data_config(contract, sample):
    profiles = contract.get("source_data_configs", {})
    if profiles:
        if sample.get("data_profile") not in profiles:
            raise ValueError(f"Unknown/missing source data profile: {sample.get('data_profile')}")
        return profiles[sample["data_profile"]]["data_config"]
    return contract["data_config"]
