#!/usr/bin/env python3
"""Rewrite Oracle replay fields from strict RLBench semantic-role manifests.

Generate manifests by replaying stored demonstrations in the simulator with
``eval.py --ground-truth --oracle-provider rlbench_gt``.  This tool then keeps
every baseline field unchanged and replaces only Oracle/audit fields using raw
GT masks and depth-derived point clouds.
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import pickle
import shutil
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from augment_replay_with_oracle_objects import (
    DEFAULT_CAMERAS,
    ORACLE_KEYS,
    ORACLE_ROLE_REFERENCE,
    ORACLE_ROLE_TARGET,
    OracleObjects,
    _copy_metadata,
    _load_low_dim_observations,
    _numeric_replay_files,
    _same_original_value,
    _stable_frame_rng,
    discover_task_directories,
    empty_oracle_objects,
    load_frame_masks,
    load_raw_frame_point_clouds,
    resolve_episode_dir,
    validate_oracle_objects,
)


AUDIT_KEYS = (
    "oracle_role_schema_version",
    "oracle_phase_id",
    "oracle_target_name",
    "oracle_reference_name",
    "oracle_target_kind",
    "oracle_reference_kind",
    "oracle_target_handles",
    "oracle_reference_handles",
    "oracle_target_role_valid",
    "oracle_reference_role_valid",
)
SEMANTIC_ROLE_SCHEMA = "rlbench_o2_semantic_roles_v1"


class FrameCache:
    def __init__(self, capacity: int):
        self.capacity = max(0, int(capacity))
        self.values: OrderedDict[Tuple[str, int, int], OracleObjects] = OrderedDict()

    def get(self, key):
        value = self.values.get(key)
        if value is not None:
            self.values.move_to_end(key)
        return value

    def put(self, key, value):
        if self.capacity == 0:
            return
        self.values[key] = value
        self.values.move_to_end(key)
        while len(self.values) > self.capacity:
            self.values.popitem(last=False)


@lru_cache(maxsize=None)
def _manifest_task_dirs(root: str, task: str):
    return tuple(
        path for path in Path(root).glob(f"**/semantic_role_manifests/{task}")
        if path.is_dir()
    )


def _manifest_path(root: Path, task: str, episode_idx: int) -> Path:
    candidates = (
        root / task / f"episode_{episode_idx}.json",
        root / "semantic_role_manifests" / task / f"episode_{episode_idx}.json",
        root / "semantic_oracle" / "semantic_role_manifests" / task
        / f"episode_{episode_idx}.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    recursive = [
        directory / f"episode_{episode_idx}.json"
        for directory in _manifest_task_dirs(str(root.resolve()), task)
        if (directory / f"episode_{episode_idx}.json").is_file()
    ]
    if len(recursive) == 1:
        return recursive[0]
    if len(recursive) > 1:
        raise ValueError(
            f"Ambiguous semantic manifests for {task} episode {episode_idx}: "
            f"{recursive}. Point --manifest-dir at one checkpoint run."
        )
    raise FileNotFoundError(
        f"Missing semantic manifest for {task} episode {episode_idx}; tried: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_manifest(root: Path, task: str, episode_idx: int):
    path = _manifest_path(root, task, episode_idx)
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    schema = str(manifest.get("schema_version", ""))
    if schema != SEMANTIC_ROLE_SCHEMA:
        raise ValueError(
            f"Unsupported semantic role schema {schema!r} in {path}; "
            f"expected {SEMANTIC_ROLE_SCHEMA!r}"
        )
    entries = manifest.get("entries", ())
    if not entries:
        raise ValueError(f"Semantic manifest has no entries: {path}")
    if any(entry.get("sample_frame") is None for entry in entries):
        raise ValueError(
            f"Manifest {path} has no raw sample_frame values. Regenerate it with "
            "eval.py --ground-truth --oracle-provider rlbench_gt."
        )
    entries = sorted(entries, key=lambda entry: int(entry["sample_frame"]))
    frames = [int(entry["sample_frame"]) for entry in entries]
    if frames != sorted(set(frames)):
        raise ValueError(f"Manifest sample_frame values must be unique: {path}")
    expected = [int(value) for value in manifest.get("expected_sample_frames", ())]
    missing = sorted(set(expected).difference(frames))
    if not expected or missing:
        raise ValueError(
            f"Incomplete expert replay manifest {path}; expected keypoints={expected}, "
            f"missing={missing}. Increase --episode-length and regenerate."
        )
    if not bool(entries[-1].get("completion_satisfied", False)):
        raise ValueError(
            f"Expert replay did not satisfy the final task condition: {path}. "
            "Do not use a failed simulator replay as semantic-GT training data."
        )
    return schema, frames, entries


def _entry_for_frame(frames, entries, sample_frame: int):
    index = bisect.bisect_right(frames, sample_frame) - 1
    if index < 0:
        raise ValueError(
            f"No semantic phase is defined at or before raw frame {sample_frame}"
        )
    return entries[index], frames[index] == sample_frame


def _role_points(role, masks, point_clouds):
    if role is None:
        return np.empty((0, 3), dtype=np.float32)
    if role["kind"] == "site":
        position = np.asarray(role.get("site_position"), dtype=np.float32).reshape(-1)
        if position.size != 3 or not np.isfinite(position).all():
            raise ValueError(f"Invalid semantic site position: {position}")
        return position.reshape(1, 3)
    handles = np.asarray(role.get("handles", ()), dtype=np.int64)
    if handles.size == 0:
        raise ValueError(f"Semantic object has no handles: {role}")
    values = []
    for camera, mask in masks.items():
        cloud = np.asarray(point_clouds[camera])
        if cloud.shape[:2] != mask.shape or cloud.shape[-1] != 3:
            raise ValueError(
                f"Raw mask/point-cloud mismatch for {camera}: {mask.shape}/{cloud.shape}"
            )
        points = cloud[np.isin(mask, handles)]
        points = points[np.isfinite(points).all(axis=1)]
        if points.size:
            values.append(points.astype(np.float32, copy=False))
    if not values:
        return np.empty((0, 3), dtype=np.float32)
    return np.concatenate(values, axis=0)


def _fill_slot(oracle, slot, role_code, role, raw_points, num_points, rng):
    if role is None or raw_points.size == 0:
        return False, 0
    if role["kind"] == "site":
        sampled = np.repeat(raw_points[:1], num_points, axis=0)
    else:
        indices = rng.choice(
            len(raw_points), size=num_points, replace=len(raw_points) < num_points
        )
        sampled = raw_points[indices]
    oracle.points[slot] = sampled.astype(np.float32, copy=False)
    oracle.centers[slot] = raw_points.mean(axis=0, dtype=np.float64).astype(np.float32)
    oracle.sizes[slot] = np.ptp(raw_points, axis=0).astype(np.float32)
    oracle.ids[slot] = slot
    oracle.valid[slot] = True
    oracle.roles[slot] = role_code
    return True, len(raw_points)


def _build_oracle(
    task, episode_idx, sample_frame, entry, exact_manifest_frame,
    episode_dir, observation, cameras, max_objects, num_points, seed,
):
    masks = load_frame_masks(episode_dir, sample_frame, cameras)
    point_clouds = load_raw_frame_point_clouds(
        episode_dir, sample_frame, cameras, observation
    )
    oracle = empty_oracle_objects(max_objects, num_points)
    target_raw = _role_points(entry["target"], masks, point_clouds)
    reference_raw = _role_points(entry.get("reference"), masks, point_clouds)
    rng = _stable_frame_rng(seed, task, episode_idx, sample_frame)
    target_valid, target_count = _fill_slot(
        oracle, 0, ORACLE_ROLE_TARGET, entry["target"], target_raw, num_points, rng
    )
    reference_valid, reference_count = _fill_slot(
        oracle, 1, ORACLE_ROLE_REFERENCE, entry.get("reference"),
        reference_raw, num_points, rng,
    )
    oracle = OracleObjects(
        points=oracle.points,
        centers=oracle.centers,
        sizes=oracle.sizes,
        ids=oracle.ids,
        valid=oracle.valid,
        roles=oracle.roles,
        raw_point_counts=(target_count, reference_count),
        discovered_objects=int(target_valid) + int(reference_valid),
        filtered_objects=0,
    )
    validate_oracle_objects(oracle, max_objects, num_points)
    for key, valid in (("target", target_valid), ("reference", reference_valid)):
        live_valid = bool(entry.get(f"{key}_valid", False))
        role = entry.get(key)
        if (
            exact_manifest_frame
            and role is not None
            and live_valid != bool(valid)
        ):
            raise ValueError(
                f"Live/saved mask handle mismatch at {task} episode={episode_idx} "
                f"frame={sample_frame} role={key}; live_visible={live_valid}, "
                f"saved_visible={bool(valid)}, handles={role.get('handles', [])}"
            )
    return oracle, target_valid, reference_valid


def _raw_handles(role):
    if role is None:
        return np.empty((0,), dtype=np.int64)
    return np.asarray(role.get("handles", ()), dtype=np.int64)


def _audit_fields(schema, entry, target_valid, reference_valid, max_objects):
    target = entry["target"]
    reference = entry.get("reference")
    text = lambda value: np.asarray([value], dtype=object)
    return {
        "oracle_role_schema_version": text(schema),
        "oracle_phase_id": text(entry["phase_id"]),
        "oracle_target_name": text(target["semantic_name"]),
        "oracle_reference_name": text(
            "" if reference is None else reference["semantic_name"]
        ),
        "oracle_target_kind": text(target["kind"]),
        "oracle_reference_kind": text("none" if reference is None else reference["kind"]),
        "oracle_target_handles": _raw_handles(target),
        "oracle_reference_handles": _raw_handles(reference),
        "oracle_target_role_valid": np.asarray(target_valid, dtype=np.bool_),
        "oracle_reference_role_valid": np.asarray(reference_valid, dtype=np.bool_),
    }


def _empty_audit(max_objects):
    entry = {
        "phase_id": "",
        "target": {"semantic_name": "", "kind": "none", "handles": []},
        "reference": None,
    }
    return _audit_fields(SEMANTIC_ROLE_SCHEMA, entry, False, False, max_objects)


def _atomic_write(destination, original, migrated, oracle):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            pickle.dump(migrated, stream, protocol=pickle.HIGHEST_PROTOCOL)
        with temporary.open("rb") as stream:
            reloaded = pickle.load(stream)
        replaced = set(ORACLE_KEYS) | set(AUDIT_KEYS)
        for key, value in original.items():
            if key not in replaced and not _same_original_value(value, reloaded[key]):
                raise ValueError(f"Semantic rewrite changed baseline field {key!r}")
        for key, value in oracle.as_replay_fields().items():
            if not _same_original_value(value, reloaded[key]):
                raise ValueError(f"Failed to verify rewritten field {key!r}")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def process_task(args, task, source_dir, destination_dir):
    files = _numeric_replay_files(source_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    manifest_cache = {}
    observation_cache = OrderedDict()
    frame_cache = FrameCache(args.cache_frames)
    stats = {
        "task": task, "files": 0, "mapping_error": 0,
        "not_visible_target": 0, "not_visible_reference": 0,
        "no_reference": 0,
    }
    for index, source in enumerate(files):
        destination = destination_dir / source.name
        if args.resume and destination.is_file():
            continue
        if destination.exists() and not args.overwrite:
            raise FileExistsError(f"Output exists: {destination}")
        with source.open("rb") as stream:
            original = pickle.load(stream)
        terminal = int(np.asarray(original.get("terminal", -1)).item())
        if terminal == -1:
            oracle = empty_oracle_objects(args.max_objects, args.num_points)
            audit = _empty_audit(args.max_objects)
        else:
            episode_idx = int(np.asarray(original["episode_idx"]).item())
            sample_frame = int(np.asarray(original["sample_frame"]).item())
            if episode_idx not in manifest_cache:
                manifest_cache[episode_idx] = _load_manifest(
                    args.manifest_dir, task, episode_idx
                )
            schema, frames, entries = manifest_cache[episode_idx]
            entry, exact = _entry_for_frame(frames, entries, sample_frame)
            if episode_idx not in observation_cache:
                episode_dir = resolve_episode_dir(args.raw_data_dir, task, episode_idx)
                observation_cache[episode_idx] = (
                    episode_dir, _load_low_dim_observations(episode_dir)
                )
                while len(observation_cache) > args.cache_episodes:
                    observation_cache.popitem(last=False)
            else:
                observation_cache.move_to_end(episode_idx)
            episode_dir, observations = observation_cache[episode_idx]
            if not 0 <= sample_frame < len(observations):
                stats["mapping_error"] += 1
                raise ValueError(
                    f"Replay/raw frame mismatch for {task} episode={episode_idx}: "
                    f"sample_frame={sample_frame}, observations={len(observations)}"
                )
            cache_key = (task, episode_idx, sample_frame)
            oracle = frame_cache.get(cache_key)
            if oracle is None:
                oracle, target_valid, reference_valid = _build_oracle(
                    task, episode_idx, sample_frame, entry, exact, episode_dir,
                    observations[sample_frame], args.cameras, args.max_objects,
                    args.num_points, args.seed,
                )
                frame_cache.put(cache_key, oracle)
            else:
                target_valid = bool(
                    np.any(oracle.valid & (oracle.roles == ORACLE_ROLE_TARGET))
                )
                reference_valid = bool(
                    np.any(oracle.valid & (oracle.roles == ORACLE_ROLE_REFERENCE))
                )
            audit = _audit_fields(
                schema, entry, target_valid, reference_valid, args.max_objects
            )
            stats["not_visible_target"] += int(not target_valid)
            if entry.get("reference") is None:
                stats["no_reference"] += 1
            else:
                stats["not_visible_reference"] += int(not reference_valid)
        migrated = dict(original)
        migrated.update(oracle.as_replay_fields())
        migrated.update(audit)
        _atomic_write(destination, original, migrated, oracle)
        stats["files"] += 1
        if (index + 1) % 100 == 0 or index + 1 == len(files):
            print(f"{task}: {index + 1}/{len(files)}", flush=True)
    _copy_metadata(source_dir, destination_dir, overwrite=args.overwrite, resume=args.resume)
    with (destination_dir / "semantic_role_rewrite_stats.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(stats, stream, indent=2, sort_keys=True)
    return stats["files"]


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--raw-data-dir", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--cameras", nargs="+", default=list(DEFAULT_CAMERAS))
    parser.add_argument("--max-objects", type=int, default=32)
    parser.add_argument("--num-points", type=int, default=512)
    parser.add_argument("--cache-frames", type=int, default=128)
    parser.add_argument(
        "--cache-episodes", type=int, default=2,
        help="Maximum low-dimensional episodes retained in memory.",
    )
    parser.add_argument("--seed", type=int, default=0)
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument("--overwrite", action="store_true")
    policy.add_argument("--resume", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None):
    args = build_parser().parse_args(argv)
    args.replay_dir = args.replay_dir.resolve()
    args.raw_data_dir = args.raw_data_dir.resolve()
    args.manifest_dir = args.manifest_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    if (
        args.max_objects < 2 or args.num_points <= 0
        or args.cache_frames < 0 or args.cache_episodes < 1
    ):
        raise ValueError(
            "--max-objects must be at least 2; points must be positive and "
            "frame cache non-negative; episode cache must be positive"
        )
    task_dirs = discover_task_directories(args.replay_dir, args.task or ["all"])
    direct = bool(_numeric_replay_files(args.replay_dir))
    total = 0
    for task, source_dir in task_dirs:
        destination = args.output_dir if direct else args.output_dir / task
        try:
            total += process_task(args, task, source_dir, destination)
        except Exception as exc:
            destination.mkdir(parents=True, exist_ok=True)
            failure = {
                "task": task,
                "mapping_error": 1,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            with (destination / "semantic_role_rewrite_failure.json").open(
                "w", encoding="utf-8"
            ) as stream:
                json.dump(failure, stream, indent=2, sort_keys=True)
            raise
    print(f"Done: {total} semantic-GT replay files", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
