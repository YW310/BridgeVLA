"""Detached PR3 geometry for reviewed activation and offline action experiments.

No TASK imports, model weights, learnable parameters, or episode readers.
The acceptance report is an explicit trusted reviewer attestation, not a claim
that software tests establish robot performance.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


ROLES = ('target', 'reference')
STATUSES = ('VISIBLE', 'NULL', 'NOT_VISIBLE', 'LOW_CONFIDENCE')
REQUIRED_METRICS = ('role_accuracy', 'prompt_instance_hit_rate', 'mask_iou',
                    'multiview_consistency', 'center_error_m',
                    'action_translation_error_m', 'paired_task_success_rate',
                    'latency_ms', 'peak_gpu_memory_bytes')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{name} must be a lowercase SHA256 digest')
    return value


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


@dataclass(frozen=True)
class ObjectGeometry:
    object_points: torch.Tensor  # [B,2,N,3], world meters before to_cube
    object_point_valid: torch.Tensor  # [B,2,N]
    object_role_valid: torch.Tensor  # [B,2], geometric availability
    reference_null_probability: torch.Tensor  # [B]
    object_geometry_confidence: torch.Tensor  # [B,2]
    statuses: tuple[tuple[str, str], ...]
    provenance: tuple[Mapping, ...]

    def validate(self):
        p = self.object_points
        if p.dtype == torch.bool or p.is_complex():
            raise ValueError('object points must contain real numeric coordinates')
        if p.ndim != 4 or p.shape[1] != 2 or p.shape[-1] != 3 or p.shape[2] < 1:
            raise ValueError('object_points must be [B,2,N,3], N>0')
        b, _, n, _ = p.shape
        if b < 1:
            raise ValueError('object geometry batch must be nonempty')
        for value, shape, name in (
            (self.object_point_valid, (b, 2, n), 'object_point_valid'),
            (self.object_role_valid, (b, 2), 'object_role_valid'),
            (self.reference_null_probability, (b,), 'reference_null_probability'),
            (self.object_geometry_confidence, (b, 2), 'object_geometry_confidence')):
            if value.shape != shape:
                raise ValueError(f'{name} must have shape {shape}')
        if self.object_point_valid.dtype != torch.bool or self.object_role_valid.dtype != torch.bool:
            raise ValueError('validity masks must be bool')
        if not torch.isfinite(p[self.object_point_valid]).all():
            raise ValueError('valid points must be finite')
        for value in (self.reference_null_probability, self.object_geometry_confidence):
            if not torch.isfinite(value).all() or not ((value >= 0) & (value <= 1)).all():
                raise ValueError('probability/confidence must be finite in [0,1]')
        if len(self.statuses) != b or len(self.provenance) != b:
            raise ValueError('statuses/provenance must have B entries')
        for batch, statuses in enumerate(self.statuses):
            if len(statuses) != 2 or any(s not in STATUSES for s in statuses) or statuses[0] == 'NULL':
                raise ValueError('invalid Target/Reference statuses')
            for role, status in enumerate(statuses):
                available = bool(self.object_role_valid[batch, role])
                if available != bool(self.object_point_valid[batch, role].any()):
                    raise ValueError('role validity must match point support')
                if available and status != 'VISIBLE':
                    raise ValueError('unseen/uncertain/NULL roles cannot carry valid geometry')
        return self

    def detached(self, device=None, dtype=None):
        self.validate()
        p = self.object_points.detach().to(device=device, dtype=dtype).clone()
        mask = self.object_point_valid.detach().to(device=p.device).clone()
        return replace(self, object_points=torch.where(mask[..., None], p, 0.),
                       object_point_valid=mask,
                       object_role_valid=self.object_role_valid.detach().to(p.device).clone(),
                       reference_null_probability=self.reference_null_probability.detach().to(p).clone(),
                       object_geometry_confidence=self.object_geometry_confidence.detach().to(p).clone())

    def descriptor(self):
        """Masked centers/stddevs; zero padding never becomes an object point."""
        mask = self.object_point_valid[..., None]
        p = torch.where(mask, self.object_points, 0.)
        count = mask.sum(2).clamp_min(1)
        center = p.sum(2) / count
        variance = torch.where(mask, (p - center[:, :, None]).square(), 0.).sum(2) / count
        spread = variance.sqrt()
        valid = self.object_role_valid
        delta = torch.where(valid.all(1, keepdim=True), center[:, 1] - center[:, 0], 0.)
        return torch.cat((center.flatten(1), spread.flatten(1), delta, valid.to(p.dtype)), 1).detach()

    def transformed(self, function):
        p = function(self.object_points.detach())
        return replace(self, object_points=torch.where(self.object_point_valid[..., None], p, 0.)).detached()


def sample_geometry(point_sets, statuses, confidence, null_probability, provenance, count=128):
    """Deterministic fixed-size sampling, preserving the actual valid-point mask."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('point count must be a positive integer')
    points = torch.zeros(len(point_sets), 2, count, 3, dtype=torch.float32)
    valid = torch.zeros(len(point_sets), 2, count, dtype=torch.bool)
    for batch, roles in enumerate(point_sets):
        for role, cloud in enumerate(roles):
            try:
                cloud = torch.as_tensor(cloud).detach()
            except (TypeError, ValueError) as exc:
                raise ValueError('role cloud must contain real numeric coordinates') from exc
            if cloud.dtype == torch.bool or cloud.is_complex():
                raise ValueError('role cloud must contain real numeric coordinates')
            cloud = cloud.to(dtype=torch.float32, device='cpu')
            if cloud.ndim != 2 or cloud.shape[-1] != 3 or not torch.isfinite(cloud).all():
                raise ValueError('role cloud must be finite [N,3]')
            if len(cloud) and statuses[batch][role] != 'VISIBLE':
                raise ValueError('unavailable roles must have empty clouds')
            n = min(count, len(cloud))
            if n:
                idx = torch.linspace(0, len(cloud) - 1, n).long()
                points[batch, role, :n] = cloud[idx]
                valid[batch, role, :n] = True
    return ObjectGeometry(points, valid, valid.any(2),
                          torch.as_tensor(null_probability, dtype=torch.float32).detach(),
                          torch.as_tensor(confidence, dtype=torch.float32).detach(),
                          tuple(tuple(s) for s in statuses), tuple(provenance)).validate()


def load_pr3_geometry(path, count=128):
    """Read one cryptographically linked current-frame JSON/NPZ, never an episode."""
    path = Path(path)
    meta = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_json_object)
    if (not isinstance(meta, dict) or type(meta.get('schema_version')) is not int
            or meta['schema_version'] != 1 or meta.get('artifact_type') != 'slot_object_geometry'):
        raise ValueError('expected PR3 slot_object_geometry schema 1')
    if meta.get('coordinate_frame') != 'rlbench_world' or meta.get('coordinate_units') != 'm':
        raise ValueError('external geometry must be RLBench world meters')
    name = meta.get('geometry_path')
    if not isinstance(name, str) or Path(name).name != name or not name.endswith('.npz'):
        raise ValueError('geometry_path must be a frame-local NPZ basename')
    archive = path.parent / name
    archive_bytes = archive.read_bytes()
    if hashlib.sha256(archive_bytes).hexdigest() != _digest(meta.get('geometry_sha256'), 'geometry_sha256'):
        raise ValueError('geometry NPZ digest mismatch')
    clouds, statuses, confidence = [], [], []
    data = np.load(io.BytesIO(archive_bytes), allow_pickle=False)
    if not isinstance(data, np.lib.npyio.NpzFile):
        raise ValueError('geometry archive must be an NPZ, not an NPY array')
    try:
        expected_keys = {'target_points_world', 'reference_points_world'}
        if len(data.files) != 2 or set(data.files) != expected_keys:
            raise ValueError('geometry NPZ must contain exactly the canonical two role arrays')
        for role in ROLES:
            record = meta['roles'][role]
            if record.get('semantic_valid') is not (record.get('status') != 'NULL'):
                raise ValueError('semantic_valid must distinguish NULL from present roles')
            if record.get('points_key') != f'{role}_points_world':
                raise ValueError('role points_key does not match the canonical role')
            if type(record.get('point_count')) is not int or record['point_count'] < 0:
                raise ValueError('point_count must be a nonnegative integer')
            cloud = np.asarray(data[record['points_key']])
            if cloud.dtype.kind not in 'fiu':
                raise ValueError('geometry arrays must contain real numeric coordinates')
            if cloud.shape != (record['point_count'], 3):
                raise ValueError('NPZ point_count/shape mismatch')
            clouds.append(cloud)
            statuses.append(record['status'])
            confidence.append(record['geometry_confidence'])
    finally:
        data.close()
    return sample_geometry([clouds], [statuses], [confidence],
                           [meta['reference_null_probability']], [meta], count)


def validate_real_acceptance(report, sources, calibrations):
    """Fail closed: a synthetic/mock report cannot authorize real activation."""
    if (report.get('evidence_kind') != 'REAL' or report.get('verified') is not True
            or report.get('backend_kind') != 'sam3' or report.get('mock') is not False
            or report.get('accepted') is not True):
        raise ValueError('activation requires an accepted verified REAL SAM3 report')
    reviewer = report.get('reviewed_by')
    reviewed_at = report.get('reviewed_at')
    if not isinstance(reviewer, str) or not reviewer.strip() or not isinstance(reviewed_at, str):
        raise ValueError('acceptance report needs explicit reviewer and review timestamp')
    try:
        if datetime.fromisoformat(reviewed_at.replace('Z', '+00:00')).tzinfo is None:
            raise ValueError('review timestamp needs a timezone')
    except ValueError as exc:
        raise ValueError('review timestamp must be explicit and timezone-aware') from exc
    if report.get('sources') != sources:
        raise ValueError('acceptance source/checkpoint mismatch')
    _validate_sources(sources)
    approved = report.get('calibration_sha256', [])
    if not isinstance(approved, list) or not approved:
        raise ValueError('calibration_sha256 must be a nonempty list of digests')
    for digest in approved:
        _digest(digest, 'calibration_sha256')
    if not calibrations or any(_digest(c, 'calibration_sha256') not in approved for c in calibrations):
        raise ValueError('current calibration not covered by acceptance report')
    metrics = report.get('metrics', {})
    for name in REQUIRED_METRICS:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f'real acceptance metric {name} is missing/invalid')


@dataclass(frozen=True)
class AuthorizedGeometry:
    geometry: ObjectGeometry
    report: Mapping
    sources: Mapping

    def verify(self):
        self.geometry.validate()
        validate_real_acceptance(self.report, self.sources,
                                 [p['calibration_sha256'] for p in self.geometry.provenance])
        return self

    def transformed(self, function):
        return replace(self, geometry=self.geometry.transformed(function))


def _validate_sources(sources, *, allow_uncheckpointed_guidance=False):
    if set(sources) != {'bridge_git_sha', 'bridge_checkpoint_sha256', 'guidance_git_sha',
                        'guidance_checkpoint_sha256', 'sam3_checkpoint_sha256'}:
        raise ValueError('acceptance source/checkpoint fields are incomplete')
    for key, value in sources.items():
        if key == 'guidance_checkpoint_sha256' and value is None and allow_uncheckpointed_guidance:
            continue
        if key.endswith('sha256'):
            _digest(value, key)
        elif key.endswith('git_sha') and (not isinstance(value, str) or len(value) not in (40, 64)
                                         or any(c not in '0123456789abcdef' for c in value)):
            raise ValueError(f'{key} must be a git SHA')


def _validate_current_frames(geometry, current_frames, *, offline=False):
    geometry.validate()
    if len(current_frames) != len(geometry.provenance):
        raise ValueError('current frame context batch mismatch')
    for source, current in zip(geometry.provenance, current_frames):
        for key in ('episode_id', 'frame_id', 'source_frame_index', 'observed_at', 'calibration_sha256'):
            if key not in source or source[key] != current.get(key):
                raise ValueError(f'current frame provenance mismatch: {key}')
        index = source['source_frame_index']
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError('source_frame_index must be a nonnegative episode index')
        read_index = current.get('max_source_frame_index_read')
        if type(read_index) is not int or read_index != index:
            raise ValueError('current-frame read receipt is missing or contains future data')
        try:
            timestamp = datetime.fromisoformat(source['observed_at'].replace('Z', '+00:00'))
            if timestamp.tzinfo is None:
                raise ValueError('timestamp needs a timezone')
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError('observed_at must be an explicit timestamp with timezone') from exc
        allowed = ('oracle_gt', 'predicted_coarse_slot') if offline else ('predicted_coarse_slot',)
        if source.get('guidance_source') not in allowed or source.get('backend_kind') != 'sam3':
            if offline:
                raise ValueError('offline action experiments require real oracle/coarse-slot SAM3 geometry')
            raise ValueError('action path requires real predicted coarse-slot SAM3 geometry')
        if source.get('mock', False) is not False:
            raise ValueError('mock geometry cannot enter an action experiment')
        if source.get('coordinate_frame') != 'rlbench_world' or source.get('coordinate_units') != 'm':
            raise ValueError('action geometry must originate in RLBench world meters')
        _digest(source['calibration_sha256'], 'calibration_sha256')


def _geometry_sources(geometry, bridge_checkpoint_path, bridge_git_sha):
    first = geometry.provenance[0]
    sources = dict(bridge_git_sha=bridge_git_sha,
                   bridge_checkpoint_sha256=sha256_file(bridge_checkpoint_path),
                   guidance_git_sha=first.get('source_git_sha'),
                   guidance_checkpoint_sha256=first.get('source_checkpoint_sha256'),
                   sam3_checkpoint_sha256=first.get('sam3_checkpoint_sha256'))
    for meta in geometry.provenance:
        if any(meta.get(key) != first.get(key) for key in
               ('guidance_source', 'source_git_sha', 'source_checkpoint_sha256', 'sam3_checkpoint_sha256')):
            raise ValueError('mixed model sources in a geometry batch')
    return sources


def authorize_geometry(geometry, current_frames, report, bridge_checkpoint_path, bridge_git_sha):
    """Authorize reviewed activation; offline experiments never satisfy this gate."""
    geometry = geometry.detached()
    _validate_current_frames(geometry, current_frames)
    sources = _geometry_sources(geometry, bridge_checkpoint_path, bridge_git_sha)
    return AuthorizedGeometry(geometry, report, sources).verify()


@dataclass(frozen=True)
class OfflineGeometry:
    """Current-frame SAM3 geometry scoped to action prediction without a controller.

    This object carries no acceptance report and never authorizes reviewed/live
    activation. Checkpoint hashes describe the actual model files used by the
    offline runner; they are not evidence of manipulation performance.
    """
    geometry: ObjectGeometry
    current_frames: tuple[Mapping, ...]
    sources: Mapping
    scope: str = 'offline_experiment'

    def verify(self):
        if self.scope != 'offline_experiment':
            raise ValueError('offline geometry scope must be offline_experiment')
        _validate_current_frames(self.geometry, self.current_frames, offline=True)
        first = self.geometry.provenance[0]
        oracle = first['guidance_source'] == 'oracle_gt'
        _validate_sources(self.sources, allow_uncheckpointed_guidance=oracle)
        for meta in self.geometry.provenance:
            if meta['guidance_source'] != first['guidance_source']:
                raise ValueError('mixed model sources in an offline geometry batch')
            for field, key in (('source_git_sha', 'guidance_git_sha'),
                               ('source_checkpoint_sha256', 'guidance_checkpoint_sha256'),
                               ('sam3_checkpoint_sha256', 'sam3_checkpoint_sha256')):
                if meta.get(field) != self.sources[key]:
                    raise ValueError('offline geometry source/checkpoint mismatch')
        if not oracle and self.sources['guidance_checkpoint_sha256'] != self.sources['bridge_checkpoint_sha256']:
            raise ValueError('predicted coarse-slot guidance must use the actual BridgeVLA checkpoint')
        return self

    def transformed(self, function):
        return replace(self, geometry=self.geometry.transformed(function))


def authorize_offline_geometry(geometry, current_frames, bridge_checkpoint_path, bridge_git_sha):
    """Validate real SAM3 A/B inputs without claiming a reviewed success report."""
    geometry = geometry.detached()
    current_frames = tuple(dict(current) for current in current_frames)
    _validate_current_frames(geometry, current_frames, offline=True)
    sources = _geometry_sources(geometry, bridge_checkpoint_path, bridge_git_sha)
    return OfflineGeometry(geometry, current_frames, sources).verify()


def freeze_for_geometry_inference(network):
    network.eval()
    for parameter in network.parameters():
        parameter.requires_grad_(False)


def geometry_to_cube(authorized, scene_world, place_pc_in_cube, with_mean, bounds):
    """Use the exact scene transform; do not fit a new cube to object points."""
    authorized.verify()
    if not scene_world:
        raise ValueError('scene batch must be nonempty')
    geom = authorized.geometry.detached(device=scene_world[0].device, dtype=scene_world[0].dtype)
    if len(scene_world) != len(geom.provenance):
        raise ValueError('scene/geometry batch mismatch')
    if any(p.ndim != 2 or p.shape[-1] != 3 or not len(p) or not torch.isfinite(p).all()
           for p in scene_world):
        raise ValueError('scene transform requires finite nonempty world points')
    if with_mean and any((p.amax(0) - p.amin(0)).amax() <= 0 for p in scene_world):
        raise ValueError('scene cube scale is degenerate')
    points = []
    for scene, obj in zip(scene_world, geom.object_points):
        points.append(place_pc_in_cube(scene, app_pc=obj.reshape(-1, 3),
                      with_mean_or_bounds=with_mean,
                      scene_bounds=None if with_mean else bounds)[0].reshape_as(obj))
    return replace(authorized, geometry=replace(geom, object_points=torch.stack(points)).detached())


def geometry_for_render(authorized, scene_points, *, norm_corr, clean_xyz):
    """Mirror legacy per-scene correlation normalization for this render stage."""
    if norm_corr and not clean_xyz:
        scales = torch.stack([p.abs().max() if len(p) else p.new_tensor(1.) for p in scene_points])
        if not torch.isfinite(scales).all() or (scales <= 0).any():
            raise ValueError('invalid correlation normalization scale')
        return authorized.transformed(lambda p: p / scales[:, None, None, None])
    return authorized


def adapter_geometry_kwargs(authorized, *, device, dtype):
    """Override only separate geometry arguments; never modify role priors/tokens."""
    authorized.verify()
    geom = authorized.geometry.detached(device=device, dtype=dtype)
    # A low geometry-confidence score cannot silently become trusted geometry.
    reliable = geom.object_role_valid & (geom.object_geometry_confidence >= .25)
    geom = replace(geom, object_role_valid=reliable,
                   object_point_valid=geom.object_point_valid & reliable[..., None])
    return dict(instance_valid=reliable,
                relation_points=geom.object_points,
                geometry=geom.descriptor(),
                reference_null_probability=geom.reference_null_probability,
                reference_is_null=torch.tensor([s[1] == 'NULL' for s in geom.statuses],
                                               device=device, dtype=torch.bool))
