"""Read-only single-frame inspection, separate from completed-replay loaders."""
import hashlib
import io
import json
from pathlib import Path
import warnings
from zipfile import BadZipFile

import numpy as np

from .common import SCHEMA, digest, inside
from .geometry import array, transform_matrix
from .observation import validate_data_config
from .replay import load_contract
from .source_config import sample_data_config


def _diagnostic_contract(root):
    # Intentionally not a relaxed load_contract(): training/teacher/evaluation
    # must keep their existing completion, full-index and split checks.
    contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
    value = dict(contract)
    if value.pop("sha256", None) != digest(value):
        raise ValueError("Replay contract checksum mismatch")
    if contract.get("schema") != SCHEMA:
        raise ValueError("Unsupported replay schema for geometry diagnostics")
    validate_data_config(contract["data_config"], resolved=True)
    for profile in contract.get("source_data_configs", {}).values():
        validate_data_config(profile["data_config"], resolved=True)
    return contract


def _validate_frame(observation, config, row):
    """Reject unfinished/malformed arrays before any diagnostic output is made."""
    for camera in config["cameras"]:
        names = [f"{camera}_{suffix}" for suffix in
                 ("rgb", "depth", "point_cloud", "camera_intrinsics", "camera_extrinsics")]
        if any(name not in observation for name in names):
            raise ValueError(f"{camera}: incomplete observation NPZ; required RGB/depth/XYZ/K/extrinsics")
        rgb, depth, xyz, K, transform = (observation[name] for name in names)
        if (rgb.ndim != 3 or rgb.shape[0] != 3 or rgb.dtype != np.uint8 or
                xyz.shape != rgb.shape or depth.shape != (1, *rgb.shape[1:]) or
                not np.issubdtype(xyz.dtype, np.floating) or
                not np.issubdtype(depth.dtype, np.floating)):
            raise ValueError(f"{camera}: expected uint8 RGB[3,H,W], float XYZ[3,H,W] and metric depth[1,H,W]")
        if list(rgb.shape[1:]) != list(config.get("image_size", [128, 128])):
            raise ValueError(f"{camera}: observation resolution disagrees with data profile")
        K = array(K, (3, 3), f"{camera} cached intrinsics")
        if K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1]):
            raise ValueError(f"{camera}: invalid cached intrinsics")
        transform_matrix(transform, f"{camera} cached optical extrinsics")
    for name, pose in (("current_tcp", row.get("current_tcp")),
                       ("goal", row.get("labels", {}).get("gripper_pose"))):
        if pose is not None:
            array(pose, (7,), name)


def load_geometry_diagnostic(root, *, sample_id=None, observation_path=None,
                             data_profile=None, allow_incomplete=False):
    """Load one immutable NPZ snapshot; never write markers, indexes or caches."""
    root = Path(root).resolve()
    if (sample_id is None) == (observation_path is None):
        raise ValueError("Choose exactly one of --sample-id or --observation")
    if observation_path is not None and not allow_incomplete:
        raise ValueError("Direct --observation inspection requires --allow-incomplete")
    if data_profile is not None and observation_path is None:
        raise ValueError("--data-profile is only used with --observation")
    contract = _diagnostic_contract(root) if allow_incomplete else load_contract(root)
    report = dict(completion_checked=not allow_incomplete, index_checksum_verified=False,
                  observation_checksum_verified=False, metadata_source="sample_index")
    if observation_path is None:
        index = root / "samples.jsonl"
        if not index.is_file():
            raise ValueError("No samples.jsonl yet; inspect a saved NPZ with --observation and --allow-incomplete")
        content = index.read_bytes()
        expected = contract.get("index_sha256")
        if expected is not None:
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError("Replay sample index changed")
            report["index_checksum_verified"] = True
        matches = [row for row in (json.loads(line) for line in content.decode("utf-8").splitlines() if line.strip())
                   if row["id"] == sample_id]
        if not matches:
            raise ValueError(f"Unknown sample ID: {sample_id}")
        if len(matches) != 1:
            raise ValueError(f"Duplicate replay sample ID: {sample_id}")
        row = dict(matches[0])
        path = inside(root, row["observation"])
        config = sample_data_config(contract, row)
    else:
        path = inside(root, observation_path)
        profiles = contract.get("source_data_configs", {})
        if profiles:
            if data_profile is None:
                if len(profiles) != 1:
                    raise ValueError("Multiple source data profiles; select --data-profile from contract.json: " +
                                     ", ".join(profiles))
                data_profile = next(iter(profiles))
            if data_profile not in profiles:
                raise ValueError(f"Unknown source data profile: {data_profile}")
        elif data_profile is not None:
            raise ValueError("This contract has no source data profiles; omit --data-profile")
        row = dict(id=path.relative_to(root).as_posix(), observation=path.relative_to(root).as_posix(),
                   data_profile=data_profile, frame=None, target_frame=None, labels={})
        config = sample_data_config(contract, row)
        report.update(metadata_source="direct_npz", data_profile=data_profile)
    if path.suffix.lower() != ".npz":
        raise ValueError("Geometry diagnostics require an observation .npz file")
    # A single byte snapshot ties the displayed arrays to the recorded digest.
    # A partially written archive is rejected; retry after the writer finishes.
    content = path.read_bytes()
    checksum = hashlib.sha256(content).hexdigest()
    if observation_path is None:
        if checksum != row["observation_sha256"]:
            raise ValueError("Observation changed")
        report["observation_checksum_verified"] = True
    try:
        with np.load(io.BytesIO(content), allow_pickle=False) as source:
            observation = {key: source[key] for key in source.files}
    except (BadZipFile, EOFError, OSError, ValueError) as error:
        raise ValueError("Observation NPZ is incomplete or unreadable; wait for its write to finish and retry") from error
    _validate_frame(observation, config, row)
    report["observation_sha256"] = checksum
    if allow_incomplete:
        report["note"] = "Diagnostic only: replay completion not checked; not proof of training-data integrity."
        row["diagnostic_note"] = "仅诊断：未检查整批完成状态。"
        if observation_path is not None:
            row["diagnostic_note"] += " 直接读取 NPZ，无索引校验；TCP/goal/帧时序未知，省略其标记及 GT 局部视图。"
        warnings.warn(report["note"], UserWarning, stacklevel=2)
    return contract, config, row, observation, report
