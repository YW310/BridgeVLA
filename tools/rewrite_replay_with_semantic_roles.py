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
import hashlib
import json
import os
import pickle
import shutil
import sys
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RLBENCH_ROOT = _REPO_ROOT / "finetune" / "RLBench"
if str(_RLBENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_RLBENCH_ROOT))

from utils.site_geometry import (
    SEMANTIC_ROLE_SCHEMA,
    SiteGeometry,
    sample_site_geometry,
)

from augment_replay_with_oracle_objects import (
    DEFAULT_CAMERAS,
    ORACLE_KEYS,
    ORACLE_ROLE_REFERENCE,
    ORACLE_ROLE_TARGET,
    OracleObjects,
    _copy_metadata,
    _load_low_dim_observations,
    _numeric_replay_files,
    _scene_points_for_visualization,
    _same_original_value,
    _stable_frame_rng,
    discover_task_directories,
    empty_oracle_objects,
    load_frame_rgb_images,
    load_frame_masks,
    load_raw_frame_point_clouds,
    resolve_episode_dir,
    validate_oracle_objects,
    visualize_oracle_objects,
)


AUDIT_KEYS = (
    "oracle_role_schema_version",
    "oracle_phase_source",
    "oracle_phase_id",
    "oracle_target_name",
    "oracle_reference_name",
    "oracle_target_kind",
    "oracle_reference_kind",
    "oracle_target_handles",
    "oracle_reference_handles",
    "oracle_target_role_valid",
    "oracle_reference_role_valid",
    "oracle_target_geometry_source",
    "oracle_reference_geometry_source",
)
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


def _semantic_entity_evidence_is_certified(evidence):
    if not isinstance(evidence, Mapping):
        return False
    views = evidence.get("views", {})
    if not isinstance(views, Mapping):
        return False
    source = evidence.get("source")
    if source == "semantic_entity_union_mask_overlap":
        return sum(
            bool(view.get("passed")) for view in views.values()
            if isinstance(view, Mapping)) >= 2
    if source == "semantic_entity_union_asymmetric_multiview_mask_overlap":
        supporting = [
            view for view in views.values()
            if isinstance(view, Mapping) and view.get("identity_support")]
        strong = [
            view for view in views.values()
            if isinstance(view, Mapping) and view.get("strong_identity_support")]
        conflicts = [
            view for view in views.values()
            if isinstance(view, Mapping) and view.get("hard_mask_conflict")]
        return len(supporting) >= 2 and bool(strong) and not conflicts
    if source == "semantic_entity_union_thin_exact_multiview_mask_overlap":
        supporting = [
            view for view in views.values()
            if isinstance(view, Mapping) and view.get("thin_identity_support")]
        strong = [
            view for view in views.values()
            if isinstance(view, Mapping)
            and view.get("thin_strong_identity_support")]
        conflicts = [
            view for view in views.values()
            if isinstance(view, Mapping) and view.get("hard_mask_conflict")]
        return len(supporting) >= 2 and bool(strong) and not conflicts
    return False


def _load_manifest(root: Path, task: str, episode_idx: int, allow_mask_verified=False):
    path = _manifest_path(root, task, episode_idx)
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    schema = str(manifest.get("schema_version", ""))
    if schema != SEMANTIC_ROLE_SCHEMA:
        raise ValueError(
            f"Unsupported semantic role schema {schema!r} in {path}; "
            f"expected {SEMANTIC_ROLE_SCHEMA!r}"
        )
    phase_source = str(manifest.get("phase_source", "sim_replay"))
    if (
        phase_source == "demo_events"
        and not bool(manifest.get("source_alignment_validated", False))
    ):
        raise ValueError(
            f"Demo-event manifest did not validate live/stored handle alignment: {path}"
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
    for entry in entries:
        for key in ("target", "reference"):
            role = entry.get(key)
            if role and role.get("kind") == "site":
                try:
                    position = np.asarray(
                        role.get("site_position"), dtype=np.float64
                    )
                    if position.shape != (3,) or not np.all(np.isfinite(position)):
                        raise ValueError(
                            "site_position must be a finite [3] vector"
                        )
                    SiteGeometry.from_mapping(role.get("site_geometry"))
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid v2 {key} site_geometry in {path}: {exc}"
                    ) from exc
    if manifest.get("handle_namespace") == "stored":
        alignment = manifest.get("handle_alignment", {})
        mask_verified = alignment.get('status') == 'mask_verified'
        if mask_verified and not allow_mask_verified:
            raise ValueError('Manifest geometry is not verified; review its audit, then '
                             'explicitly use --allow-mask-verified-handles to accept: ' + str(path))
        if (alignment.get("status") not in ("verified", "mask_verified")
                or not manifest.get("source_frame0_masks")):
            raise ValueError(f"Missing verified stored-handle provenance: {path}")
        if mask_verified:
            print(f'[WARNING] Using mask-only identity mapping; geometry not certified: {path}', flush=True)
        mapped = set(alignment.get("live_to_stored", {}).values())
        entity_mapping = alignment.get("semantic_entity_to_stored", {})
        if not isinstance(entity_mapping, Mapping):
            raise ValueError(
                f"Invalid semantic entity handle mapping in {path}")
        entity_mapped = set()
        for handles in entity_mapping.values():
            if (not isinstance(handles, (list, tuple)) or not handles
                    or any(isinstance(handle, bool)
                           or not isinstance(handle, int) or handle <= 0
                           for handle in handles)):
                raise ValueError(
                    f"Invalid semantic entity stored handles in {path}: "
                    f"{handles!r}")
            entity_mapped.update(handles)
        if entity_mapped:
            alignment_scope = alignment.get("alignment_scope")
            if alignment_scope == "mixed_entity_certificates":
                certificates = alignment.get("evidence", {}).get(
                    "entity_certificates", {})
                if not isinstance(certificates, Mapping):
                    raise ValueError(
                        f"Missing per-entity alignment certificates in {path}")
                live_to_stored = alignment.get("live_to_stored", {})
                for live_handles, stored_handles in entity_mapping.items():
                    certificate = certificates.get(live_handles)
                    if not isinstance(certificate, Mapping):
                        raise ValueError(
                            f"Missing certificate for semantic entity "
                            f"{live_handles!r} in {path}")
                    certified_handles = set(certificate.get("stored_handles", ()))
                    if certificate.get("source") == "individual_handles":
                        try:
                            expected = {
                                live_to_stored[str(int(handle))]
                                for handle in live_handles.split(",")
                                if str(int(handle)) in live_to_stored
                            }
                        except (TypeError, ValueError):
                            expected = set()
                        valid = expected == set(stored_handles)
                    else:
                        valid = _semantic_entity_evidence_is_certified(certificate)
                    if not valid or certified_handles != set(stored_handles):
                        raise ValueError(
                            f"Uncertified semantic entity mapping "
                            f"{live_handles!r} in {path}")
            elif alignment_scope == "semantic_entity_union":
                entities = alignment.get("evidence", {}).get("entities", {})
                if not isinstance(entities, Mapping):
                    raise ValueError(
                        f"Missing semantic entity union evidence in {path}")
                certified = {
                    handle
                    for evidence in entities.values()
                    if _semantic_entity_evidence_is_certified(evidence)
                    for handle in evidence.get("stored_handles", ())
                }
                if not entity_mapped <= certified:
                    raise ValueError(
                        f"Uncertified semantic entity handles in {path}: "
                        f"{sorted(entity_mapped.difference(certified))}")
            elif not entity_mapped <= mapped:
                raise ValueError(
                    f"Semantic entity mapping is not backed by individual "
                    f"handle verification in {path}")
            mapped.update(entity_mapped)
        for entry in entries:
            for key in ("target", "reference"):
                role = entry.get(key)
                if role and role["kind"] == "object" and not set(role["handles"]) <= mapped:
                    raise ValueError(f"Unverified stored handles in {path}: {role}")
    if manifest.get("source_frame0_masks"):
        entries[0] = dict(entries[0], source_frame0_masks=manifest["source_frame0_masks"])
    frames = [int(entry["sample_frame"]) for entry in entries]
    if frames != sorted(set(frames)):
        raise ValueError(f"Manifest sample_frame values must be unique: {path}")
    expected = [int(value) for value in manifest.get("expected_sample_frames", ())]
    missing = sorted(set(expected).difference(frames))
    if not expected or missing:
        raise ValueError(
            f"Incomplete semantic manifest {path}; expected keypoints={expected}, "
            f"missing={missing}. Regenerate the manifest from its configured "
            "phase source."
        )
    if not bool(entries[-1].get("completion_satisfied", False)):
        raise ValueError(
            f"Semantic manifest did not satisfy the final task condition: {path}. "
            "Do not use an incomplete phase trace as semantic-GT training data."
        )
    return schema, frames, entries


def _entry_for_frame(frames, entries, sample_frame: int):
    index = bisect.bisect_right(frames, sample_frame) - 1
    if index < 0:
        raise ValueError(
            f"No semantic phase is defined at or before raw frame {sample_frame}"
        )
    return entries[index], frames[index] == sample_frame


def _validate_source_masks(episode_dir, entries):
    expected = entries[0].get("source_frame0_masks", {})
    if not expected:
        return  # Legacy manifests predate fingerprint recording.
    masks = load_frame_masks(episode_dir, 0, tuple(expected))
    for camera, fingerprint in expected.items():
        mask = masks[camera]
        actual = hashlib.sha256(
            str(mask.shape).encode("ascii")
            + np.asarray(mask, dtype="<i8").tobytes()).hexdigest()
        if actual != fingerprint:
            raise ValueError(
                f"Manifest/raw source mask mismatch: {episode_dir}, {camera}, frame=0. "
                "Use the same raw dataset and mask resolution as manifest generation.")


def _role_points(role, masks, point_clouds, num_points):
    if role is None:
        return np.empty((0, 3), dtype=np.float32)
    if role["kind"] == "site":
        geometry = SiteGeometry.from_mapping(role.get("site_geometry"))
        return sample_site_geometry(geometry, num_points)
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
        geometry = SiteGeometry.from_mapping(role.get("site_geometry"))
        sampled = raw_points
        center = geometry.center_world.astype(np.float32)
        size = geometry.extent.astype(np.float32)
    else:
        indices = rng.choice(
            len(raw_points), size=num_points, replace=len(raw_points) < num_points
        )
        sampled = raw_points[indices]
        center = raw_points.mean(axis=0, dtype=np.float64).astype(np.float32)
        size = np.ptp(raw_points, axis=0).astype(np.float32)
    oracle.points[slot] = sampled.astype(np.float32, copy=False)
    oracle.centers[slot] = center
    oracle.sizes[slot] = size
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
    target_raw = _role_points(
        entry["target"], masks, point_clouds, num_points
    )
    reference_raw = _role_points(
        entry.get("reference"), masks, point_clouds, num_points
    )
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


def _geometry_source(role):
    if role is None:
        return "none"
    kind = role.get("kind")
    if kind == "site":
        return SiteGeometry.from_mapping(
            role.get("site_geometry")
        ).source
    if kind == "object":
        return "object_mask"
    return "none"


def _audit_fields(schema, entry, target_valid, reference_valid, max_objects):
    target = entry["target"]
    reference = entry.get("reference")
    text = lambda value: np.asarray([value], dtype=object)
    return {
        "oracle_role_schema_version": text(schema),
        "oracle_phase_source": text(entry.get("phase_source", "sim_replay")),
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
        "oracle_target_geometry_source": text(_geometry_source(target)),
        "oracle_reference_geometry_source": text(_geometry_source(reference)),
    }


def _empty_audit(max_objects):
    entry = {
        "phase_source": "",
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


def _scalar(value, key):
    values = np.asarray(value).reshape(-1)
    if values.size != 1:
        raise ValueError(f'{key} must contain exactly one value; got {values.shape}')
    return values[0]


def _oracle_from_replay(transition, max_objects, num_points):
    missing = sorted(set(ORACLE_KEYS).difference(transition))
    if missing:
        raise ValueError(f'Semantic replay is missing Oracle fields: {missing}')
    valid = np.asarray(transition['oracle_object_valid'])
    oracle = OracleObjects(
        points=np.asarray(transition['oracle_object_points']),
        centers=np.asarray(transition['oracle_object_centers']),
        sizes=np.asarray(transition['oracle_object_sizes']),
        ids=np.asarray(transition['oracle_object_ids']),
        valid=valid,
        roles=np.asarray(transition['oracle_object_roles']),
        raw_point_counts=(),
        discovered_objects=int(valid.sum()),
        filtered_objects=0,
    )
    validate_oracle_objects(oracle, max_objects, num_points)
    return oracle


def _validate_semantic_transition(transition, max_objects, num_points, source=None):
    missing = sorted(set(AUDIT_KEYS).difference(transition))
    if missing:
        raise ValueError(f'Semantic replay is missing audit fields: {missing}')
    schema = str(_scalar(
        transition['oracle_role_schema_version'], 'oracle_role_schema_version'))
    if schema != SEMANTIC_ROLE_SCHEMA:
        raise ValueError(
            f'Unsupported semantic replay schema {schema!r}; '
            f'expected {SEMANTIC_ROLE_SCHEMA!r}')
    oracle = _oracle_from_replay(transition, max_objects, num_points)
    target_present = bool(np.any(
        oracle.valid & (oracle.roles == ORACLE_ROLE_TARGET)))
    reference_present = bool(np.any(
        oracle.valid & (oracle.roles == ORACLE_ROLE_REFERENCE)))
    target_audit = bool(_scalar(
        transition['oracle_target_role_valid'], 'oracle_target_role_valid'))
    reference_audit = bool(_scalar(
        transition['oracle_reference_role_valid'],
        'oracle_reference_role_valid'))
    if target_present != target_audit:
        raise ValueError(
            'Target role-valid audit disagrees with oracle_object_roles')
    if reference_present != reference_audit:
        raise ValueError(
            'Reference role-valid audit disagrees with oracle_object_roles')
    for label, role_code in (
        ('target', ORACLE_ROLE_TARGET),
        ('reference', ORACLE_ROLE_REFERENCE),
    ):
        kind = str(_scalar(
            transition[f'oracle_{label}_kind'], f'oracle_{label}_kind'))
        geometry_source = str(_scalar(
            transition[f'oracle_{label}_geometry_source'],
            f'oracle_{label}_geometry_source'))
        slots = np.flatnonzero(
            oracle.valid & (oracle.roles == role_code))
        if len(slots) > 1:
            raise ValueError(f'Semantic replay has multiple {label} slots')
        if kind == 'site':
            if geometry_source not in ('object_bbox', 'fallback_box'):
                raise ValueError(
                    f'{label} site has invalid geometry source '
                    f'{geometry_source!r}')
            if len(slots) == 1:
                unique = np.unique(
                    np.round(oracle.points[int(slots[0])], decimals=7), axis=0)
                if len(unique) < 2:
                    raise ValueError(
                        f'{label} site still uses a repeated center point')
        elif kind == 'object':
            if geometry_source != 'object_mask':
                raise ValueError(
                    f'{label} object has invalid geometry source '
                    f'{geometry_source!r}')
        elif kind == 'none':
            if geometry_source != 'none' or len(slots):
                raise ValueError(
                    f'{label} none role contains geometry or a valid slot')
        else:
            raise ValueError(f'Unknown {label} kind {kind!r}')
    if source is not None:
        replaced = set(ORACLE_KEYS) | set(AUDIT_KEYS)
        for key, value in source.items():
            if (
                key not in replaced
                and (
                    key not in transition
                    or not _same_original_value(value, transition[key])
                )
            ):
                raise ValueError(
                    f'Semantic rewrite changed baseline field {key!r}')
    return oracle


def _validate_task_output(args, task, source_dir, destination_dir):
    source_files = _numeric_replay_files(source_dir)
    output_files = _numeric_replay_files(destination_dir)
    source_names = {path.name for path in source_files}
    output_names = {path.name for path in output_files}
    missing = sorted(source_names - output_names)
    extra = sorted(output_names - source_names)
    if missing or extra:
        raise ValueError(
            f'{task} replay file set mismatch: missing={missing[:5]}, '
            f'extra={extra[:5]}')
    report = {
        'task': task,
        'schema_version': SEMANTIC_ROLE_SCHEMA,
        'files': len(output_files),
        'nonterminal_files': 0,
        'target_valid': 0,
        'reference_valid': 0,
        'site_roles': 0,
        'fallback_box_roles': 0,
        'raw_fallback_files': 0,
        'phase_sources': {},
        'valid': True,
    }
    source_by_name = {path.name: path for path in source_files}
    for output_path in output_files:
        with source_by_name[output_path.name].open('rb') as stream:
            source = pickle.load(stream)
        with output_path.open('rb') as stream:
            transition = pickle.load(stream)
        oracle = _validate_semantic_transition(
            transition, args.max_objects, args.num_points, source=source)
        terminal = int(np.asarray(transition.get('terminal', -1)).item())
        if terminal == -1:
            continue
        report['nonterminal_files'] += 1
        report['target_valid'] += int(np.any(
            oracle.valid & (oracle.roles == ORACLE_ROLE_TARGET)))
        report['reference_valid'] += int(np.any(
            oracle.valid & (oracle.roles == ORACLE_ROLE_REFERENCE)))
        phase_source = str(_scalar(
            transition['oracle_phase_source'], 'oracle_phase_source'))
        report['raw_fallback_files'] += int(not phase_source)
        report['phase_sources'][phase_source] = (
            report['phase_sources'].get(phase_source, 0) + 1)
        for label in ('target', 'reference'):
            kind = str(_scalar(
                transition[f'oracle_{label}_kind'],
                f'oracle_{label}_kind'))
            geometry_source = str(_scalar(
                transition[f'oracle_{label}_geometry_source'],
                f'oracle_{label}_geometry_source'))
            report['site_roles'] += int(kind == 'site')
            report['fallback_box_roles'] += int(
                geometry_source == 'fallback_box')
    report_path = destination_dir / 'semantic_role_validation.json'
    with report_path.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
    files_count = report['files']
    nonterminal_count = report['nonterminal_files']
    site_count = report['site_roles']
    fallback_count = report['raw_fallback_files']
    print(
        f'[VALIDATED] {task}: {files_count} replay files; '
        f'nonterminal={nonterminal_count}; site_roles={site_count}; '
        f'raw_fallback={fallback_count}; report={report_path}',
        flush=True)
    return report


def _visualization_files(files, index, every):
    if index is not None:
        matches = [path for path in files if int(path.stem) == index]
        if not matches:
            raise FileNotFoundError(f'Missing replay index {index}')
        return matches
    if every > 0:
        return list(files[::every])
    return []


def _visualize_task_output(args, task, destination_dir):
    files = _numeric_replay_files(destination_dir)
    selected = _visualization_files(
        files, args.visualize_index, args.visualize_every)
    output_dir = args.visualize_output_dir / task
    count = 0
    for replay_path in selected:
        replay_index = int(replay_path.stem)
        with replay_path.open('rb') as stream:
            transition = pickle.load(stream)
        oracle = _validate_semantic_transition(
            transition, args.max_objects, args.num_points)
        terminal = int(np.asarray(transition.get('terminal', -1)).item())
        episode_idx = None
        sample_frame = None
        camera_images = {}
        camera_masks = {}
        group_by_id = {}
        if terminal != -1:
            episode_idx = int(np.asarray(transition['episode_idx']).item())
            sample_frame = int(np.asarray(transition['sample_frame']).item())
            episode_dir = resolve_episode_dir(
                args.raw_data_dir, task, episode_idx)
            camera_images = load_frame_rgb_images(
                episode_dir, sample_frame, args.cameras)
            camera_masks = load_frame_masks(
                episode_dir, sample_frame, args.cameras)
            for handle in np.asarray(
                    transition['oracle_target_handles']).reshape(-1):
                group_by_id[int(handle)] = 0
            for handle in np.asarray(
                    transition['oracle_reference_handles']).reshape(-1):
                group_by_id[int(handle)] = 1
        scene_points = None
        if not args.visualize_objects_only:
            scene_points = _scene_points_for_visualization(
                transition, args.cameras)
        output_path = visualize_oracle_objects(
            oracle,
            task,
            replay_index,
            output_dir,
            scene_points=scene_points,
            terminal=terminal,
            episode_idx=episode_idx,
            sample_frame=sample_frame,
            camera_images=camera_images,
            camera_masks=camera_masks,
            group_by_id=group_by_id,
            stable_object_ids=(0, 1),
        )
        metadata = {
            'task': task,
            'replay_index': replay_index,
            'episode_idx': episode_idx,
            'sample_frame': sample_frame,
            'phase_source': str(_scalar(
                transition['oracle_phase_source'], 'oracle_phase_source')),
            'phase_id': str(_scalar(
                transition['oracle_phase_id'], 'oracle_phase_id')),
            'target_name': str(_scalar(
                transition['oracle_target_name'], 'oracle_target_name')),
            'target_kind': str(_scalar(
                transition['oracle_target_kind'], 'oracle_target_kind')),
            'target_geometry_source': str(_scalar(
                transition['oracle_target_geometry_source'],
                'oracle_target_geometry_source')),
            'reference_name': str(_scalar(
                transition['oracle_reference_name'],
                'oracle_reference_name')),
            'reference_kind': str(_scalar(
                transition['oracle_reference_kind'],
                'oracle_reference_kind')),
            'reference_geometry_source': str(_scalar(
                transition['oracle_reference_geometry_source'],
                'oracle_reference_geometry_source')),
        }
        with output_path.with_suffix('.json').open(
                'w', encoding='utf-8') as stream:
            json.dump(metadata, stream, indent=2, sort_keys=True)
        count += 1
    return count


def process_task(args, task, source_dir, destination_dir):
    files = _numeric_replay_files(source_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    manifest_cache = {}
    invalid_manifests = {}
    observation_cache = OrderedDict()
    frame_cache = FrameCache(args.cache_frames)
    stats = {
        "task": task, "files": 0, "mapping_error": 0,
        "not_visible_target": 0, "not_visible_reference": 0,
        "no_reference": 0, "invalid_manifest_episodes": 0,
        "invalid_manifest_transitions": 0, "invalid_manifest_errors": {},
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
            if (episode_idx not in manifest_cache
                    and episode_idx not in invalid_manifests):
                try:
                    manifest_cache[episode_idx] = _load_manifest(
                        args.manifest_dir, task, episode_idx,
                        allow_mask_verified=getattr(
                            args, 'allow_mask_verified_handles', False)
                    )
                except (FileNotFoundError, ValueError) as exc:
                    if not getattr(
                            args, 'fallback_invalid_manifests_to_raw', False):
                        raise
                    invalid_manifests[episode_idx] = str(exc)
                    stats["invalid_manifest_episodes"] += 1
                    stats["invalid_manifest_errors"][str(episode_idx)] = str(exc)
                    print(
                        f"[WARNING] {task} episode {episode_idx}: invalid semantic "
                        "manifest; writing empty Oracle fields so training uses the "
                        f"raw branch. Reason: {exc}", flush=True)
            if episode_idx in invalid_manifests:
                oracle = empty_oracle_objects(args.max_objects, args.num_points)
                audit = _empty_audit(args.max_objects)
                stats["invalid_manifest_transitions"] += 1
            else:
                schema, frames, entries = manifest_cache[episode_idx]
                entry, exact = _entry_for_frame(frames, entries, sample_frame)
                if episode_idx not in observation_cache:
                    episode_dir = resolve_episode_dir(
                        args.raw_data_dir, task, episode_idx)
                    _validate_source_masks(episode_dir, entries)
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
    parser.add_argument('--allow-mask-verified-handles', action='store_true',
                        help='Explicitly accept mask-only identity provenance after geometry audit.')
    parser.add_argument(
        '--fallback-invalid-manifests-to-raw', action='store_true',
        help=(
            'Preserve transitions whose manifest is missing or invalid, but write '
            'empty Oracle fields so O2 falls back to the raw branch. Fail-fast is '
            'the default; ignored episode/transition counts are written to stats.'))
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
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Number of task-level worker processes. Each worker has independent "
            "frame/episode caches; start with 2 on disk-backed datasets."),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        '--validate-output', action='store_true',
        help=(
            'After rewriting, validate every output replay against its source '
            'and write semantic_role_validation.json per task.'),
    )
    visualization = parser.add_mutually_exclusive_group()
    visualization.add_argument(
        '--visualize-index', type=int,
        help='Visualize this numeric replay index for every selected task.',
    )
    visualization.add_argument(
        '--visualize-every', type=int, default=0, metavar='N',
        help='Visualize every Nth sorted semantic replay; 0 disables.',
    )
    parser.add_argument(
        '--visualize-output-dir', type=Path,
        help=(
            'PNG/JSON output root; defaults to '
            '<output-dir>/semantic_role_visualizations.'),
    )
    parser.add_argument(
        '--visualize-objects-only', action='store_true',
        help='Hide the gray full-scene point cloud in semantic replay PNGs.',
    )
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
    if args.visualize_output_dir is None:
        args.visualize_output_dir = (
            args.output_dir / 'semantic_role_visualizations')
    else:
        args.visualize_output_dir = args.visualize_output_dir.resolve()
    if (
        args.visualize_every < 0
        or (
            args.visualize_index is not None
            and args.visualize_index < 0
        )
    ):
        raise ValueError('Visualization index/interval must be non-negative')
    if (
        args.max_objects < 2 or args.num_points <= 0
        or args.cache_frames < 0 or args.cache_episodes < 1 or args.workers < 1
    ):
        raise ValueError(
            "--max-objects must be at least 2; points must be positive and "
            "frame cache non-negative; episode cache and workers must be positive"
        )
    task_dirs = discover_task_directories(args.replay_dir, args.task or ["all"])
    direct = bool(_numeric_replay_files(args.replay_dir))
    jobs = [
        (task, source_dir, args.output_dir if direct else args.output_dir / task)
        for task, source_dir in task_dirs
    ]
    total = 0

    def record_failure(task, destination, exc):
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

    if args.workers == 1 or len(jobs) <= 1:
        for task, source_dir, destination in jobs:
            try:
                total += process_task(args, task, source_dir, destination)
            except Exception as exc:
                record_failure(task, destination, exc)
                raise
    else:
        worker_count = min(args.workers, len(jobs))
        print(
            f"Parallel semantic rewrite: {worker_count} task workers; each "
            "worker owns independent caches.", flush=True)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(
                    process_task, args, task, source_dir, destination
                ): (task, destination)
                for task, source_dir, destination in jobs
            }
            for future in as_completed(futures):
                task, destination = futures[future]
                try:
                    count = future.result()
                except Exception as exc:
                    record_failure(task, destination, exc)
                    for pending in futures:
                        pending.cancel()
                    raise
                total += count
                print(
                    f"[DONE] {task}: {count} semantic-GT replay files",
                    flush=True)
    print(f"Done: {total} semantic-GT replay files", flush=True)
    if args.validate_output:
        for task, source_dir, destination in jobs:
            _validate_task_output(args, task, source_dir, destination)
    visualized = 0
    if args.visualize_index is not None or args.visualize_every > 0:
        for task, _, destination in jobs:
            visualized += _visualize_task_output(args, task, destination)
        print(
            f'Visualized: {visualized} semantic-GT replay files; '
            f'output={args.visualize_output_dir}',
            flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
