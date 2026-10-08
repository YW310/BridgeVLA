"""Read all Parquet columns directly; LeRobot feature metadata is incomplete."""
from pathlib import Path
import json
import numpy as np
from .common import TASKS, inside, file_digest
from .geometry import array, quaternion

SHAPES = {
    "action": 7, "observation.state": 7, "observation.gripper_joints": 2,
    "observation.ee_pos_world": 3, "observation.ee_quat_world": 4,
    "observation.objects_pos": 12, "observation.objects_quat": 16,
}
REQUIRED = ("frame_index", "episode_index", "timestamp", "instruction_id", "instruction",
            "next.done", "next.success", *SHAPES)


def discover(root):
    root = Path(root).resolve()
    records = []
    for task in TASKS:
        dataset = root / task / "lerobot_dataset"
        for path in sorted((dataset / "data").glob("chunk-*/episode_*.parquet")):
            try:
                episode = int(path.stem.split("_")[-1])
            except ValueError as exc:
                raise ValueError(f"Invalid episode filename: {path}") from exc
            records.append(dict(task=task, task_index=TASKS.index(task), episode_index=episode,
                                path=path.relative_to(root).as_posix(),
                                dataset=dataset.relative_to(root).as_posix(),
                                metadata=(Path(task) / f"episode_{episode:06d}" / "metadata.json").as_posix()))
    keys = [(r["task"], r["episode_index"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate task/episode Parquet files across chunks")
    return records


def read_episode(root, record):
    import pyarrow.parquet as pq
    path = inside(root, record["path"])
    return pq.read_table(path).to_pydict()


def validate_episode(columns, record):
    missing = set(REQUIRED) - set(columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    n = len(columns["timestamp"])
    if n < 2 or any(len(value) != n for value in columns.values()):
        raise ValueError("Empty/unaligned episode columns")
    frames = np.asarray(columns["frame_index"])
    if not np.array_equal(frames, np.arange(n)):
        raise ValueError("frame_index must be contiguous and start at zero")
    timestamps = array(columns["timestamp"], (n,), "timestamp")
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamp must be strictly increasing")
    if not np.all(np.asarray(columns["episode_index"]) == record["episode_index"]):
        raise ValueError("Episode index does not match its file")
    for name, width in SHAPES.items():
        array(columns[name], (n, width), name)
    quaternion(columns["observation.ee_quat_world"])
    quaternion(np.asarray(columns["observation.objects_quat"]).reshape(n, 4, 4))
    commands = np.asarray(columns["action"])[:, 6]
    if not np.isin(commands, [-1, 0, 1]).all():
        raise ValueError("Invalid gripper commands")
    # Content grouping prevents exact duplicate trajectories crossing splits.
    from .common import digest
    trajectory_hash = digest({key: columns[key] for key in (
        "observation.state", "observation.ee_pos_world", "observation.ee_quat_world")})
    positions = np.asarray(columns["observation.ee_pos_world"])
    ee_command = np.asarray(columns["action"])[:, :6]
    zero = np.linalg.norm(ee_command, axis=1) < 1e-8
    displacement = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    quality = dict(columns=sorted(columns), timestamp_dt_min=float(np.diff(timestamps).min()),
                   timestamp_dt_max=float(np.diff(timestamps).max()),
                   ee_world_min=positions.min(axis=0).tolist(), ee_world_max=positions.max(axis=0).tolist(),
                   state_gripper_min=float(np.asarray(columns["observation.state"])[:, 6].min()),
                   state_gripper_max=float(np.asarray(columns["observation.state"])[:, 6].max()),
                   raw_ee_command_zero_fraction=float(zero.mean()),
                   moving_despite_zero_command_frames=int((zero[:-1] & (displacement > 1e-4)).sum()),
                   gripper_command_counts={str(value): int((commands == value).sum()) for value in (-1, 0, 1)})
    return dict(frames=n, duration=float(timestamps[-1] - timestamps[0]),
                success=bool(any(columns["next.success"])), trajectory_hash=trajectory_hash, quality=quality)


def inspect_episode(root, record, cameras):
    columns = read_episode(root, record)
    info = validate_episode(columns, record)
    dataset = inside(root, record["dataset"])
    for camera in cameras:
        extrinsic = array(columns[f"observation.{camera}_extrinsic"], (info["frames"], 7), camera)
        quaternion(extrinsic[:, 3:])
        for modality in ("images", "depth"):
            name = f"observation.{modality}.{camera}"
            if name not in columns:
                raise ValueError(f"Missing video column {name}")
            for reference in columns[name]:
                if not isinstance(reference, dict) or "Path" not in reference or "Timestamp" not in reference:
                    raise ValueError(f"Invalid video reference {name}")
                if not inside(dataset, reference["Path"]).is_file():
                    raise FileNotFoundError(reference["Path"])
                stamp = np.asarray(reference["Timestamp"], dtype=float).reshape(-1)
                if stamp.size != 1 or not np.isfinite(stamp).all() or stamp[0] < 0:
                    raise ValueError(f"Expected one nonnegative video timestamp in {name}")
    metadata_path = inside(root, record["metadata"])
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("all_segments_succeeded") is False:
            raise ValueError("metadata reports failed segments")
    if not info["success"]:
        raise ValueError("Episode has no next.success marker")
    info["parquet_sha256"] = file_digest(inside(root, record["path"]))
    return info
