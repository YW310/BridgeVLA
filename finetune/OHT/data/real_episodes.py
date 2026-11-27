"""Read real OHT episode lineage and plan measured V4 action windows."""

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .actions import gripper_states, keypoints
from .audit import split_records
from .chunk_actions import make_chunk_targets, resolve_chunk_timing
from .common import REAL_TASK, file_digest, inside
from .real_rgbd import FIXED_CAMERAS


DATASET_PATH = Path("assemble_wheel/lerobot_dataset")


def _recorded_parent(dataset, episode_index, source):
    path = dataset / "provenance" / f"episode_{episode_index:06d}" / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    parents = set()
    for workflow in ("replay", "dagger"):
        reference = ((metadata.get(workflow) or {}).get("source") or {}).get("episode")
        if reference and reference not in ("manual", source):
            parents.add(reference)
    if len(parents) > 1:
        raise ValueError(f"Conflicting source episodes in {path}")
    return next(iter(parents), None)


def _lineage_root(source, parents):
    visited = set()
    current = source
    while current in parents:
        if current in visited:
            raise ValueError(f"Replay source cycle for real episode {source}")
        visited.add(current)
        current = parents[current]
    return current


def episode_records(export_root, seed):
    """Keep recordings that share a replay source in one split."""
    export_root = Path(export_root)
    dataset = export_root / DATASET_PATH
    report = json.loads((export_root / "conversion_report.json").read_text(encoding="utf-8"))
    receipts = report["receipts"]
    if report["status"] != "converted" or not receipts or len(receipts) != report["episodes"]:
        raise ValueError("Expected a complete real OHT conversion report")
    names = {receipt["episode"]: receipt["group"] for receipt in receipts}
    ids = {receipt["episode_index"] for receipt in receipts}
    if len(names) != len(receipts) or ids != set(range(len(receipts))):
        raise ValueError("Duplicate or missing real OHT episode identities")
    if Counter(names.values()) != report["groups"]:
        raise ValueError("Conversion report cohort counts do not match episode receipts")

    records, parents = [], {}
    for receipt in receipts:
        episode_index = receipt["episode_index"]
        source = receipt["episode"]
        cohort = receipt["group"]
        parent = _recorded_parent(dataset, episode_index, source)
        if cohort == "augmentation":
            if names.get(parent) != "manual":
                raise ValueError(f"Augmented episode {source} has no manual source in the export")
        if parent is not None:
            parents[source] = parent
        cameras = {stream["camera"]: stream for stream in receipt["cameras"]}
        if not set(FIXED_CAMERAS).issubset(cameras):
            raise ValueError(f"Real episode {source} is missing a fixed camera")
        records.append(dict(
            task=REAL_TASK, episode_index=episode_index, source_episode=source,
            cohort=cohort, source_parent=parent or source, trajectory_hash=receipt["raw_index_sha256"],
            path=receipt["parquet"], parquet_sha256=receipt["parquet_sha256"],
            rows=receipt["rows"], cameras={camera: cameras[camera] for camera in FIXED_CAMERAS},
        ))
    groups = {}
    for record in records:
        root = _lineage_root(record["source_episode"], parents)
        record["source_root"] = root
        groups[f"{REAL_TASK}/{record['episode_index']}"] = root
    return split_records(records, seed, groups=groups)


def _t58_bounds(path, source_rows, selected):
    with np.load(path, allow_pickle=False) as t58:
        before, after, valid = (t58[key] for key in ("source_row", "next_source_row", "supervision_valid"))
    length = len(source_rows)
    if (before.ndim != 1 or not len(before) or before.shape != after.shape
            or before.shape != valid.shape or valid.dtype != np.bool_ or not valid.all()
            or before.dtype.kind not in "iu" or after.dtype.kind not in "iu"
            or not np.all(before[1:] > before[:-1]) or np.any(before < 0)
            or np.any(after <= before) or np.any(after >= length)):
        raise ValueError(f"Invalid T58 measured source links in {path}")
    if (not np.array_equal(source_rows, np.arange(length))
            or not np.array_equal(source_rows[selected], before)):
        raise ValueError(f"T58 source indices disagree with real Parquet rows in {path}")
    # Each link covers real intermediate rows. Connected links define one
    # physical-time span independently of the T58 30 Hz training clock.
    if np.any(before[1:] > np.maximum.accumulate(after)[:-1]):
        raise ValueError(f"Discontinuous T58 source spans in {path}")
    return int(before[0]), int(after.max()) + 1


def _gripper_runs(states, poses, valid, timestamps, config):
    """Compute binary transitions only within connected measured state runs."""
    opening = np.full(len(states), np.nan, dtype=np.float32)
    binary = np.full(len(states), -1, dtype=np.int64)
    run_end = np.full(len(states), -1, dtype=np.int64)
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    keys = []
    for start, stop in edges.reshape(-1, 2):
        opening[start:stop], binary[start:stop] = gripper_states(states[start:stop], config)
        run_end[start:stop] = stop
        if stop - start >= 2:
            transitions = keypoints(poses[start:stop], binary[start:stop],
                                    timestamps=timestamps[start:stop], method="gripper")
            keys.extend(start + transition for transition in transitions)
    return opening, binary, run_end, np.asarray(keys, dtype=np.int64)


def read_episode(dataset, record, options, sample_stride, gripper):
    """Plan V4 samples on raw robot ticks, carrying the original row numbers."""
    if type(sample_stride) is not int or sample_stride < 1:
        raise ValueError("sample_stride must be a positive integer")
    path = inside(dataset, record["path"])
    if file_digest(path) != record["parquet_sha256"]:
        raise ValueError(f"Real Parquet checksum mismatch in {path}")
    columns = [
        "source_row", "source_timestamp_ns", "t58_selected", "observation.state",
        "observation.ee_pos", "observation.ee_quat",
    ]
    columns += [f"{camera}.{field}" for camera in FIXED_CAMERAS
                for field in ("capture_index", "age_ms")]
    table = pq.read_table(path, columns=columns)
    if table.num_rows != record["rows"]:
        raise ValueError(f"Unexpected real episode row count in {path}")

    def column(name):
        return table[name].to_numpy()

    def vector(name):
        values = table[name].combine_chunks()
        return values.values.to_numpy().reshape(table.num_rows, values.type.list_size)

    episode_dir = Path(dataset) / "provenance" / f"episode_{record['episode_index']:06d}"
    start, length = _t58_bounds(episode_dir / "t58_training_selection.npz",
                                column("source_row"), column("t58_selected"))
    clock_ns = column("source_timestamp_ns")[:length]
    clock = (clock_ns - clock_ns[0]).astype(np.float64) * 1e-9
    options = resolve_chunk_timing(clock, options)
    states = vector("observation.state")[:length]
    joints = states[:, :6]
    positions = vector("observation.ee_pos")[:length]
    quats = vector("observation.ee_quat")[:length, [1, 2, 3, 0]].copy()
    norms = np.linalg.norm(quats, axis=-1)
    valid = (np.isfinite(states).all(axis=-1) & (states[:, 6] >= 0) & (states[:, 6] <= 1)
             & np.isfinite(positions).all(axis=-1) & np.isfinite(norms) & (norms > 0))
    quats[valid] /= norms[valid, None]
    poses = np.column_stack((positions, quats))
    opening, binary, run_end, keys = _gripper_runs(states, poses, valid, clock, gripper)

    captures = {camera: column(f"{camera}.capture_index")[:length] for camera in FIXED_CAMERAS}
    camera_valid = np.ones(length, dtype=bool)
    for camera, indices in captures.items():
        ages = column(f"{camera}.age_ms")[:length]
        if np.any(indices >= record["cameras"][camera]["frames"]):
            raise ValueError(f"Out-of-range {camera} capture index in {path}")
        camera_valid &= (indices >= 0) & (ages >= 0) & (ages <= 150)
        if np.any(np.diff(indices[indices >= 0]) < 0):
            raise ValueError(f"Out-of-order {camera} capture index in {path}")

    frames = set(range(start, length - 1, sample_stride))
    frames.update(key for key in keys.tolist() if start <= key < length - 1)
    samples, skipped = [], Counter()
    stride = options["frame_stride"]
    for frame in sorted(frames):
        count = min(options["horizon"], (length - 1 - frame) // stride)
        if not count:
            skipped["no_future_target"] += 1
            continue
        if run_end[frame] <= frame + count * stride:
            skipped["invalid_measured_state"] += 1
            continue
        if not camera_valid[frame]:
            skipped["missing_or_stale_rgbd"] += 1
            continue
        next_key = keys[np.searchsorted(keys, frame, side="right"):]
        next_key = next_key[:1]
        targets, reason = make_chunk_targets(poses, joints, binary, clock, frame,
                                             next_key[next_key < run_end[frame]], options)
        if targets is None:
            skipped[reason] += 1
            continue
        samples.append(dict(
            frame=frame, episode_length=length, timestamp=float(clock[frame]),
            tcp=poses[frame], joints=joints[frame], opening=opening[frame],
            gripper=int(binary[frame]), captures={name: int(index[frame]) for name, index in captures.items()},
            targets=targets,
        ))
    return samples, skipped
