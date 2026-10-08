"""Shared observation NPZ cache plus lightweight transition index."""
import bisect
import json
from pathlib import Path
import numpy as np
from .common import SCHEMA, GOALS, digest, file_digest, inside, write_json, read_jsonl
from .reader import read_episode, validate_episode
from .actions import gripper_states, low_dim, world_tcp_poses, keypoints, target_labels
from .observation import validate_data_config, camera_observation
from .video import EpisodeVideos, metric_depth


def build(root, manifest_path, config, output, sample_stride=10):
    validate_data_config(config)
    if sample_stride < 1:
        raise ValueError("sample_stride must be positive")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    original = dict(manifest)
    recorded_hash = original.pop("manifest_sha256", None)
    if recorded_hash != digest(original):
        raise ValueError("Audit manifest checksum mismatch")
    if manifest.get("schema") != "oht_audit_v1" or not manifest.get("episodes"):
        raise ValueError("Audit must contain valid OHT episodes")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Use a new replay directory: {output}")
    output.mkdir(parents=True)
    contract = dict(schema=SCHEMA, data_config=config, audit_manifest_sha256=recorded_hash,
                    policy_frame="world", quaternion_order="xyzw", target="future_absolute_tcp",
                    language="task_goal", rotation_classes=int(config.get("rotation_classes", 72)),
                    sample_stride=sample_stride)
    contract["sha256"] = digest(contract)
    write_json(output / "contract.json", contract)
    rows = []
    for record in manifest["episodes"]:
        if file_digest(inside(root, record["path"])) != record["parquet_sha256"]:
            raise ValueError(f"Raw Parquet changed after audit: {record['path']}")
        columns = read_episode(root, record)
        validate_episode(columns, record)
        measured, desired = gripper_states(columns["observation.state"],
                                           np.asarray(columns["action"])[:, 6])
        poses = world_tcp_poses(columns, config["link_to_tcp"])
        keys = keypoints(poses, desired, columns["instruction_id"], **config.get("keypoints", {}))
        frames = sorted(set(range(0, len(poses) - 1, sample_stride)) | {0} | set(keys[:-1]))
        dataset = inside(root, record["dataset"])
        videos = EpisodeVideos(dataset, config.get("video_timestamp_tolerance", 1/120 + .0001))
        try:
            for frame in frames:
                target = keys[bisect.bisect_right(keys, frame)]
                sample_id = f'{record["task"]}/{record["episode_index"]:06d}/{frame:06d}'
                observation = {"low_dim_state": low_dim(measured[frame],
                                    columns["observation.gripper_joints"][frame])}
                for camera in config["cameras"]:
                    rgb = videos.read(columns[f"observation.images.{camera}"][frame])
                    depth = metric_depth(root, record, columns, camera, frame, videos, config["depth"])
                    observation.update(camera_observation(
                        camera, rgb, depth, columns[f"observation.{camera}_extrinsic"][frame], config))
                clouds = np.concatenate([observation[f"{c}_point_cloud"].reshape(3, -1).T for c in config["cameras"]])
                bounds = np.asarray(config["scene_bounds"])
                finite = np.isfinite(clouds).all(axis=1)
                in_bounds = finite & (clouds >= bounds[:3]).all(axis=1) & (clouds < bounds[3:]).all(axis=1)
                if not in_bounds.any():
                    raise ValueError(f"{sample_id}: no scene points within scene_bounds")
                relative = f"observations/{sample_id}.npz"
                path = output / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path, **observation)
                labels = target_labels(poses[target], int(desired[target]), config["scene_bounds"],
                                       int(config.get("rotation_classes", 72)))
                rows.append(dict(id=sample_id, task=record["task"], split=record["split"],
                                 episode_index=record["episode_index"], frame=frame, target_frame=target,
                                 timestamp=float(columns["timestamp"][frame]), group=record["group"],
                                 observation=relative, observation_sha256=file_digest(path),
                                 goal=GOALS[record["task"]],
                                 labels={key: value.tolist() for key, value in labels.items()}))
        finally:
            videos.close()
    (output / "samples.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    # Bind checkpoints/role caches to the actual observations and transition
    # schedule, not only to calibration and the source Parquet manifest.
    contract["index_sha256"] = file_digest(output / "samples.jsonl")
    contract.pop("sha256")
    contract["sha256"] = digest(contract)
    write_json(output / "contract.json", contract)
    write_json(output / "complete.json", dict(contract_sha256=contract["sha256"], samples=len(rows),
                                               index_sha256=contract["index_sha256"]))
    return len(rows)


def load_contract(root):
    root = Path(root)
    contract = json.loads((root / "contract.json").read_text(encoding="utf-8"))
    complete = json.loads((root / "complete.json").read_text(encoding="utf-8"))
    value = dict(contract)
    expected = value.pop("sha256", None)
    if expected != digest(value) or complete["contract_sha256"] != expected:
        raise ValueError("Replay contract checksum mismatch")
    if contract.get("schema") != SCHEMA:
        raise ValueError("Unsupported replay schema")
    if file_digest(root / "samples.jsonl") != complete["index_sha256"]:
        raise ValueError("Replay sample index changed")
    if contract.get("index_sha256") != complete["index_sha256"]:
        raise ValueError("Replay sample index is not bound to its contract")
    rows = read_jsonl(root / "samples.jsonl")
    if not rows or complete["samples"] != len(rows):
        raise ValueError("Replay sample count mismatch or empty index")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate replay sample IDs")
    groups = {}
    for row in rows:
        if row["split"] not in ("train", "val", "test") or row["target_frame"] <= row["frame"]:
            raise ValueError("Invalid replay split or non-future action target")
        previous = groups.setdefault(row["group"], row["split"])
        if previous != row["split"]:
            raise ValueError("Scene/trajectory group crosses dataset splits")
    return contract
