#!/usr/bin/env python3
'''Append Oracle RLBench instance point clouds to BridgeVLA replay files.

The script migrates existing per-transition YARR *.replay pickle files. It
does not call ReplayBuffer.add() and does not regenerate BridgeVLA features.
'''

from __future__ import annotations

import argparse
import colorsys
import hashlib
import os
import pickle
import shutil
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from zipfile import BadZipFile

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from tools.rlbench_task_object_priors import select_task_relevant_instances
except ModuleNotFoundError:  # Direct execution: python tools/<script>.py
    from rlbench_task_object_priors import select_task_relevant_instances
try:
    from tools.rlbench_task_handle_detector import (
        TaskHandleDetection,
        build_task_frame_evidence,
        detect_task_handles,
        load_task_handle_detection,
        save_task_handle_detection,
    )
except ModuleNotFoundError:  # Direct execution: python tools/<script>.py
    from rlbench_task_handle_detector import (
        TaskHandleDetection,
        build_task_frame_evidence,
        detect_task_handles,
        load_task_handle_detection,
        save_task_handle_detection,
    )
try:
    from tools.rlbench_robot_handle_detector import (
        RobotHandleDetection,
        build_robot_frame_evidence,
        detect_robot_handles,
        load_robot_handle_detection,
        save_robot_handle_detection,
    )
except ModuleNotFoundError:  # Direct execution: python tools/<script>.py
    from rlbench_robot_handle_detector import (
        RobotHandleDetection,
        build_robot_frame_evidence,
        detect_robot_handles,
        load_robot_handle_detection,
        save_robot_handle_detection,
    )


# Oracle object experiment
DEFAULT_CAMERAS = ('front', 'left_shoulder', 'right_shoulder', 'wrist')
DEFAULT_MAX_OBJECTS = 32
DEFAULT_NUM_POINTS = 512
RLBENCH_DEPTH_SCALE = 2**24 - 1
REPLAY_METADATA_CACHE_VERSION = 1
REPLAY_METADATA_CACHE_NAME = '.oracle_replay_metadata_v1.npz'
MAX_VISUALIZATION_SCENE_POINTS = 30000
# Same metric workspace used by finetune/RLBench/utils/peract_utils_rlbench.py.
# Fixed limits prevent per-frame Matplotlib autoscaling from changing apparent
# object size, including when --visualize-objects-only is enabled.
VISUALIZATION_SCENE_BOUNDS = (-0.3, -0.5, 0.6, 0.7, 0.5, 1.6)
ORACLE_KEYS = (
    'oracle_object_points',
    'oracle_object_centers',
    'oracle_object_sizes',
    'oracle_object_ids',
    'oracle_object_valid',
    'oracle_object_roles',
)
ORACLE_ROLE_UNKNOWN = 0
ORACLE_ROLE_TARGET = 1
ORACLE_ROLE_REFERENCE = 2


ReplayMetadataArrays = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
_REPLAY_METADATA_MEMORY_CACHE: Dict[
    Tuple[str, str], ReplayMetadataArrays
] = {}


@dataclass(frozen=True)
class OracleObjects:
    points: np.ndarray
    centers: np.ndarray
    sizes: np.ndarray
    ids: np.ndarray
    valid: np.ndarray
    roles: np.ndarray
    raw_point_counts: Tuple[int, ...]
    discovered_objects: int
    filtered_objects: int
    prior_filtered_objects: int = 0
    excluded_objects: int = 0
    excluded_object_ids: Tuple[int, ...] = ()
    no_finite_point_object_ids: Tuple[int, ...] = ()
    small_object_ids: Tuple[int, ...] = ()
    prior_filtered_object_ids: Tuple[int, ...] = ()
    truncated_object_ids: Tuple[int, ...] = ()
    temporal_filtered_objects: int = 0
    temporal_filtered_object_ids: Tuple[int, ...] = ()
    thin_plane_objects: int = 0
    thin_plane_object_ids: Tuple[int, ...] = ()
    protected_thin_plane_object_ids: Tuple[int, ...] = ()

    def as_replay_fields(self) -> Dict[str, np.ndarray]:
        return {
            'oracle_object_points': self.points,
            'oracle_object_centers': self.centers,
            'oracle_object_sizes': self.sizes,
            'oracle_object_ids': self.ids,
            'oracle_object_valid': self.valid,
            'oracle_object_roles': self.roles,
        }


@dataclass(frozen=True)
class RobotFrameSelection:
    '''Adaptive raw-frame prefix used to identify the robot.'''

    sample_frames: Tuple[int, ...]
    max_gripper_displacement: float
    motion_sufficient: bool
    stopped_on_close: bool


class OracleFrameCache:
    '''Thread-safe LRU cache with single-flight computation per raw frame.'''

    def __init__(self, capacity: int):
        if capacity < 0:
            raise ValueError('cache capacity must be non-negative')
        self.capacity = capacity
        self._entries: OrderedDict[Tuple[object, ...], Future] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get_or_compute(
        self,
        key: Tuple[object, ...],
        compute: Callable[[], OracleObjects],
    ) -> OracleObjects:
        if self.capacity == 0:
            with self._lock:
                self._misses += 1
            return compute()

        with self._lock:
            future = self._entries.get(key)
            if future is None:
                future = Future()
                self._entries[key] = future
                self._misses += 1
                owner = True
            else:
                self._entries.move_to_end(key)
                self._hits += 1
                owner = False

        if not owner:
            # A concurrent request for the same frame waits for the first
            # worker instead of decoding the four masks a second time.
            return future.result()

        try:
            value = compute()
        except BaseException as exc:
            future.set_exception(exc)
            with self._lock:
                if self._entries.get(key) is future:
                    del self._entries[key]
            raise

        future.set_result(value)
        with self._lock:
            if self._entries.get(key) is future:
                self._entries.move_to_end(key)
            self._evict_locked(protected_key=key)
        return value

    def _evict_locked(self, protected_key: Tuple[object, ...]) -> None:
        while len(self._entries) > self.capacity:
            evicted = False
            for candidate, future in list(self._entries.items()):
                if candidate != protected_key and future.done():
                    del self._entries[candidate]
                    evicted = True
                    break
            if not evicted:
                # Pending computations are never evicted. The cache may exceed
                # its capacity briefly until one of them completes.
                break

    def stats(self) -> Tuple[int, int, int]:
        with self._lock:
            return self._hits, self._misses, len(self._entries)


def empty_oracle_objects(max_objects: int, num_points: int) -> OracleObjects:
    '''Return padding for a YARR final-observation sentinel.'''
    return OracleObjects(
        points=np.zeros((max_objects, num_points, 3), dtype=np.float32),
        centers=np.zeros((max_objects, 3), dtype=np.float32),
        sizes=np.zeros((max_objects, 3), dtype=np.float32),
        ids=np.full((max_objects,), -1, dtype=np.int32),
        valid=np.zeros((max_objects,), dtype=np.bool_),
        roles=np.zeros((max_objects,), dtype=np.int8),
        raw_point_counts=(),
        discovered_objects=0,
        filtered_objects=0,
        prior_filtered_objects=0,
        excluded_objects=0,
    )


def _point_cloud_hwc(point_cloud: np.ndarray, name: str) -> np.ndarray:
    point_cloud = np.asarray(point_cloud)
    if point_cloud.ndim != 3:
        raise ValueError(f'{name} must have 3 dimensions; got {point_cloud.shape}')
    # BridgeVLA replay point clouds are channel-first. Check that convention
    # before HWC so a small test/input shaped [3, H, 3] is not ambiguous.
    if point_cloud.shape[0] == 3:
        return np.moveaxis(point_cloud, 0, -1)
    if point_cloud.shape[-1] == 3:
        return point_cloud
    raise ValueError(
        f'{name} must be [H, W, 3] or [3, H, W]; got {point_cloud.shape}'
    )


def _rlbench_mask_decoder() -> Callable[[np.ndarray], np.ndarray]:
    try:
        from rlbench.backend.utils import rgb_handles_to_mask
    except ImportError as exc:
        raise RuntimeError(
            'RGB masks require rlbench.backend.utils.rgb_handles_to_mask. '
            'Run this script in the BridgeVLA RLBench environment.'
        ) from exc
    return rgb_handles_to_mask


def decode_mask_image(
    image: np.ndarray,
    decoder: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    '''Decode raw RLBench handles without treating RGB values as IDs.'''
    image = np.asarray(image)
    if image.ndim == 2:
        mask = image
    elif image.ndim == 3 and image.shape[-1] in (3, 4):
        # Oracle object experiment: RLBench masks are coded RGB handle PNGs.
        decoder = decoder or _rlbench_mask_decoder()
        # PIL-backed np.asarray values are read-only, while RLBench's decoder
        # multiplies its input in place. The decoder also expects RGB in [0, 1],
        # whereas PNG pixels loaded by PIL are uint8 in [0, 255].
        encoded_rgb = image[..., :3]
        rgb = np.array(encoded_rgb, dtype=np.float32, copy=True)
        if np.issubdtype(encoded_rgb.dtype, np.integer):
            rgb /= 255.0
        elif rgb.size and np.max(rgb) > 1.0:
            rgb /= 255.0
        mask = decoder(rgb)
    else:
        raise ValueError(f'Unsupported RLBench mask image shape: {image.shape}')
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f'Decoded mask must be [H, W]; got {mask.shape}')
    return mask.astype(np.int64, copy=False)


def load_frame_masks(
    episode_dir: Path,
    sample_frame: int,
    cameras: Sequence[str],
    decoder: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    for camera in cameras:
        mask_path = episode_dir / f'{camera}_mask' / f'{sample_frame}.png'
        if not mask_path.is_file():
            raise FileNotFoundError(
                f'Missing {camera} mask for frame {sample_frame}: {mask_path}'
            )
        with Image.open(mask_path) as image:
            masks[camera] = decode_mask_image(np.asarray(image), decoder=decoder)
    return masks


def decode_depth_image(image: np.ndarray) -> np.ndarray:
    '''Decode an RLBench fixed-point depth PNG to normalized depth in [0, 1].'''
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[-1] in (3, 4):
        rgb = image[..., :3].astype(np.float64, copy=False)
        fixed = np.sum(rgb * np.array([65536.0, 256.0, 1.0]), axis=-1)
        depth = fixed / float(RLBENCH_DEPTH_SCALE)
    elif image.ndim == 2:
        if np.issubdtype(image.dtype, np.integer):
            scale = float(np.iinfo(image.dtype).max)
        else:
            scale = 1.0
        depth = image.astype(np.float64, copy=False) / scale
    else:
        raise ValueError(f'Unsupported RLBench depth image shape: {image.shape}')
    return depth.astype(np.float32, copy=False)


def point_cloud_from_depth_and_camera_params(
    depth: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    '''Convert metric camera depth to world coordinates using pure NumPy.'''
    depth = np.asarray(depth, dtype=np.float64)
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError(f'depth must be [H, W]; got {depth.shape}')
    if extrinsics.shape != (4, 4):
        raise ValueError(f'extrinsics must be [4, 4]; got {extrinsics.shape}')
    if intrinsics.shape != (3, 3):
        raise ValueError(f'intrinsics must be [3, 3]; got {intrinsics.shape}')

    rows, columns = np.indices(depth.shape, dtype=np.float64)
    projected = np.stack(
        (columns * depth, rows * depth, depth), axis=-1
    )
    camera_points = projected @ np.linalg.inv(intrinsics).T
    rotation = extrinsics[:3, :3]
    translation = extrinsics[:3, 3]
    world_points = camera_points @ rotation.T + translation
    return world_points.astype(np.float32, copy=False)


def load_raw_frame_point_clouds(
    episode_dir: Path,
    sample_frame: int,
    cameras: Sequence[str],
    observation: object,
) -> Dict[str, np.ndarray]:
    '''Reconstruct raw RLBench point clouds aligned with the raw mask PNGs.'''
    misc = getattr(observation, 'misc', None)
    if not isinstance(misc, Mapping):
        raise ValueError(
            f'Raw observation frame {sample_frame} has no camera metadata'
        )
    point_clouds: Dict[str, np.ndarray] = {}
    for camera in cameras:
        depth_path = episode_dir / f'{camera}_depth' / f'{sample_frame}.png'
        if not depth_path.is_file():
            raise FileNotFoundError(
                f'Missing {camera} depth for frame {sample_frame}: {depth_path}'
            )
        with Image.open(depth_path) as image:
            normalized_depth = decode_depth_image(np.asarray(image))
        try:
            near = float(misc[f'{camera}_camera_near'])
            far = float(misc[f'{camera}_camera_far'])
            extrinsics = misc[f'{camera}_camera_extrinsics']
            intrinsics = misc[f'{camera}_camera_intrinsics']
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f'Invalid {camera} camera metadata at frame {sample_frame}'
            ) from exc
        metric_depth = near + normalized_depth * (far - near)
        point_clouds[camera] = point_cloud_from_depth_and_camera_params(
            metric_depth, extrinsics, intrinsics
        )
    return point_clouds


def load_frame_rgb_images(
    episode_dir: Path,
    sample_frame: int,
    cameras: Sequence[str],
) -> Dict[str, np.ndarray]:
    '''Load available raw RLBench RGB views for a visualization frame.'''
    images: Dict[str, np.ndarray] = {}
    for camera in cameras:
        image_path = episode_dir / f'{camera}_rgb' / f'{sample_frame}.png'
        if not image_path.is_file():
            continue
        with Image.open(image_path) as image:
            images[camera] = np.asarray(image.convert('RGB')).copy()
    return images


def _robust_oriented_extents(points: np.ndarray) -> np.ndarray:
    '''Return PCA-aligned 2%-98% extents, sorted from thin to long axis.'''
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 3:
        return np.sort(np.ptp(points, axis=0))
    centered = points - np.median(points, axis=0)
    try:
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        projected = centered @ axes.T
        lower, upper = np.quantile(projected, (0.02, 0.98), axis=0)
        return np.sort(upper - lower)
    except np.linalg.LinAlgError:
        return np.sort(np.ptp(points, axis=0))


def extract_oracle_objects(
    transition: Mapping[str, object],
    masks: Mapping[str, np.ndarray],
    cameras: Sequence[str] = DEFAULT_CAMERAS,
    max_objects: int = DEFAULT_MAX_OBJECTS,
    num_points: int = DEFAULT_NUM_POINTS,
    excluded_ids: Iterable[int] = (0,),
    included_ids: Optional[Iterable[int]] = None,
    slot_ids: Optional[Sequence[int]] = None,
    min_object_points: int = 20,
    rng: Optional[np.random.Generator] = None,
    task_name: Optional[str] = None,
    task_prior_filter: bool = False,
    action_position: Optional[np.ndarray] = None,
    task_prior_radius: Optional[float] = None,
    task_prior_max_instances: Optional[int] = None,
    task_prior_background_extent: float = 0.60,
    task_prior_strict: bool = False,
    role_by_id: Optional[Mapping[int, int]] = None,
    group_by_id: Optional[Mapping[int, int]] = None,
    filter_thin_planes: bool = False,
    thin_plane_max_thickness: float = 0.005,
    thin_plane_min_extent: float = 0.08,
    filter_thin_planes_all_roles: bool = False,
) -> OracleObjects:
    '''Fuse decoded instance masks with the existing replay point clouds.'''
    if max_objects <= 0 or num_points <= 0 or min_object_points <= 0:
        raise ValueError(
            'max_objects, num_points, and min_object_points must be positive'
        )
    if thin_plane_max_thickness <= 0 or thin_plane_min_extent <= 0:
        raise ValueError('Thin-plane thresholds must be positive')
    rng = rng or np.random.default_rng()
    excluded = {int(value) for value in excluded_ids}
    included = (
        None if included_ids is None else {int(value) for value in included_ids}
    )
    points_by_id: Dict[int, List[np.ndarray]] = {}
    observed_ids = set()

    for camera in cameras:
        point_key = f'{camera}_point_cloud'
        if point_key not in transition:
            raise KeyError(f'Replay transition is missing {point_key!r}')
        if camera not in masks:
            raise KeyError(f'Decoded masks are missing camera {camera!r}')
        point_cloud = _point_cloud_hwc(np.asarray(transition[point_key]), point_key)
        mask = np.asarray(masks[camera])
        if mask.shape != point_cloud.shape[:2]:
            raise ValueError(
                f'Pixel alignment mismatch for {camera}: mask {mask.shape}, '
                f'point cloud {point_cloud.shape[:2]}'
            )

        for object_id_value in np.unique(mask):
            raw_object_id = int(object_id_value)
            object_id = (
                int(group_by_id.get(raw_object_id, raw_object_id))
                if group_by_id is not None
                else raw_object_id
            )
            if object_id >= 0:
                observed_ids.add(object_id)
            if (
                raw_object_id in excluded
                or object_id in excluded
                or object_id < 0
            ):
                continue
            if (
                included is not None
                and raw_object_id not in included
                and object_id not in included
            ):
                continue
            object_points = point_cloud[mask == raw_object_id]
            object_points = object_points[np.isfinite(object_points).all(axis=1)]
            if object_points.size:
                points_by_id.setdefault(object_id, []).append(object_points)

    merged = [
        (object_id, np.concatenate(camera_points, axis=0))
        for object_id, camera_points in points_by_id.items()
    ]
    temporal_filtered_object_ids = tuple(sorted(
        observed_ids - excluded - (included if included is not None else observed_ids)
    ))
    no_finite_point_object_ids = tuple(sorted(
        observed_ids
        - excluded
        - set(temporal_filtered_object_ids)
        - set(points_by_id)
    ))
    small_object_ids = tuple(sorted(
        object_id
        for object_id, object_points in merged
        if len(object_points) < min_object_points
    ))
    filtered_objects = sum(
        len(object_points) < min_object_points
        for _, object_points in merged
    )
    merged = [
        (object_id, object_points)
        for object_id, object_points in merged
        if len(object_points) >= min_object_points
    ]
    thin_plane_object_ids: Tuple[int, ...] = ()
    protected_thin_plane_object_ids: Tuple[int, ...] = ()
    if filter_thin_planes:
        protected_roles = role_by_id or {}
        thin_candidates = {
            object_id
            for object_id, object_points in merged
            if (
                lambda ordered: (
                    ordered[0] <= thin_plane_max_thickness
                    and ordered[1] >= thin_plane_min_extent
                    and ordered[2] >= thin_plane_min_extent
                )
            )(_robust_oriented_extents(object_points))
        }
        protected_thin_plane_object_ids = tuple(
            sorted(
                object_id
                for object_id in thin_candidates
                if not filter_thin_planes_all_roles
                and int(
                    protected_roles.get(object_id, ORACLE_ROLE_UNKNOWN)
                )
                != ORACLE_ROLE_UNKNOWN
            )
        )
        thin_plane_object_ids = tuple(
            sorted(
                thin_candidates - set(protected_thin_plane_object_ids)
            )
        )
        thin_plane_set = set(thin_plane_object_ids)
        merged = [
            (object_id, object_points)
            for object_id, object_points in merged
            if object_id not in thin_plane_set
        ]
    prior_filtered_objects = 0
    prior_filtered_object_ids: Tuple[int, ...] = ()
    if task_prior_filter:
        if task_name is None:
            raise ValueError('task_name is required with task-prior filtering')
        if action_position is None:
            raise ValueError(
                'action_position is required with task-prior filtering'
            )
        before_prior_ids = {object_id for object_id, _ in merged}
        merged = select_task_relevant_instances(
            task_name,
            merged,
            action_position,
            interaction_radius=task_prior_radius,
            max_instances=task_prior_max_instances,
            background_extent=task_prior_background_extent,
            strict_action_filter=task_prior_strict,
        )
        after_prior_ids = {object_id for object_id, _ in merged}
        prior_filtered_object_ids = tuple(sorted(
            before_prior_ids - after_prior_ids
        ))
        prior_filtered_objects = len(prior_filtered_object_ids)
    discovered_objects = len(merged)

    placements: List[Tuple[int, int, np.ndarray]] = []
    if slot_ids is None:
        # Fixed storage requires truncation. Prefer the strongest point
        # support, then use handle ID as a deterministic tie breaker.
        # Task-prior results are already ordered by action proximity.
        if not task_prior_filter:
            merged.sort(key=lambda item: (-len(item[1]), item[0]))
        truncated_object_ids = tuple(
            object_id for object_id, _ in merged[max_objects:]
        )
        placements = [
            (slot, object_id, object_points)
            for slot, (object_id, object_points) in enumerate(
                merged[:max_objects]
            )
        ]
    else:
        # Temporal mode is an episode-local ID-to-slot association, not an
        # object filter. Reserve the same slot even when its handle is absent
        # in this frame; visible handles outside the sampled episode map fill
        # remaining capacity without displacing known handles.
        stable_ids = tuple(
            dict.fromkeys(
                int(group_by_id.get(int(object_id), int(object_id)))
                if group_by_id is not None
                else int(object_id)
                for object_id in slot_ids
            )
        )
        slot_by_id = {
            object_id: slot
            for slot, object_id in enumerate(stable_ids[:max_objects])
        }
        merged_by_id = dict(merged)
        placed_ids = set()
        for object_id, slot in slot_by_id.items():
            object_points = merged_by_id.get(object_id)
            if object_points is None:
                continue
            placements.append((slot, object_id, object_points))
            placed_ids.add(object_id)

        reserved_slots = set(slot_by_id.values())
        free_slots = iter(
            slot
            for slot in range(max_objects)
            if slot not in reserved_slots
        )
        for object_id, object_points in sorted(merged, key=lambda item: item[0]):
            if object_id in placed_ids or object_id in slot_by_id:
                continue
            try:
                slot = next(free_slots)
            except StopIteration:
                break
            placements.append((slot, object_id, object_points))
            placed_ids.add(object_id)
        placements.sort(key=lambda item: item[0])
        truncated_object_ids = tuple(
            sorted(
                object_id
                for object_id, _ in merged
                if object_id not in placed_ids
            )
        )

    padded = empty_oracle_objects(max_objects, num_points)
    raw_counts: List[int] = []
    for slot, object_id, object_points in placements:
        count = len(object_points)
        indices = rng.choice(count, size=num_points, replace=count < num_points)
        padded.points[slot] = object_points[indices].astype(np.float32, copy=False)
        padded.centers[slot] = np.mean(
            object_points, axis=0, dtype=np.float64
        ).astype(np.float32)
        padded.sizes[slot] = (
            np.max(object_points, axis=0)
            - np.min(object_points, axis=0)
        ).astype(np.float32)
        padded.ids[slot] = object_id
        padded.valid[slot] = True
        if role_by_id is not None:
            padded.roles[slot] = int(
                role_by_id.get(object_id, ORACLE_ROLE_UNKNOWN)
            )
        raw_counts.append(count)

    oracle = OracleObjects(
        points=padded.points,
        centers=padded.centers,
        sizes=padded.sizes,
        ids=padded.ids,
        valid=padded.valid,
        roles=padded.roles,
        raw_point_counts=tuple(raw_counts),
        discovered_objects=discovered_objects,
        filtered_objects=filtered_objects,
        prior_filtered_objects=prior_filtered_objects,
        excluded_objects=len(observed_ids.intersection(excluded)),
        excluded_object_ids=tuple(sorted(observed_ids.intersection(excluded))),
        no_finite_point_object_ids=no_finite_point_object_ids,
        small_object_ids=small_object_ids,
        prior_filtered_object_ids=prior_filtered_object_ids,
        truncated_object_ids=truncated_object_ids,
        temporal_filtered_objects=len(temporal_filtered_object_ids),
        temporal_filtered_object_ids=temporal_filtered_object_ids,
        thin_plane_objects=len(thin_plane_object_ids),
        thin_plane_object_ids=thin_plane_object_ids,
        protected_thin_plane_object_ids=protected_thin_plane_object_ids,
    )
    validate_oracle_objects(oracle, max_objects, num_points)
    return oracle


def validate_oracle_objects(
    oracle: OracleObjects, max_objects: int, num_points: int
) -> None:
    expected = {
        'points': ((max_objects, num_points, 3), np.dtype(np.float32)),
        'centers': ((max_objects, 3), np.dtype(np.float32)),
        'sizes': ((max_objects, 3), np.dtype(np.float32)),
        'ids': ((max_objects,), np.dtype(np.int32)),
        'valid': ((max_objects,), np.dtype(np.bool_)),
        'roles': ((max_objects,), np.dtype(np.int8)),
    }
    for name, (shape, dtype) in expected.items():
        value = getattr(oracle, name)
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(
                f'Oracle {name} is {value.shape}/{value.dtype}; '
                f'expected {shape}/{dtype}'
            )
    if not np.isfinite(oracle.points).all():
        raise ValueError('oracle_object_points contains NaN or Inf')
    if not np.isfinite(oracle.centers).all():
        raise ValueError('oracle_object_centers contains NaN or Inf')
    if not np.isfinite(oracle.sizes).all():
        raise ValueError('oracle_object_sizes contains NaN or Inf')
    if np.any(oracle.sizes < 0):
        raise ValueError('oracle_object_sizes contains negative values')
    if np.any(oracle.ids[~oracle.valid] != -1):
        raise ValueError('Invalid Oracle slots must use object ID -1')
    if np.any(oracle.roles[~oracle.valid] != ORACLE_ROLE_UNKNOWN):
        raise ValueError('Invalid Oracle slots must use role 0')
    if not np.isin(
        oracle.roles,
        (ORACLE_ROLE_UNKNOWN, ORACLE_ROLE_TARGET, ORACLE_ROLE_REFERENCE),
    ).all():
        raise ValueError('oracle_object_roles contains an unknown role code')


def _same_original_value(before: object, after: object) -> bool:
    if isinstance(before, np.ndarray):
        if not isinstance(after, np.ndarray):
            return False
        if before.shape != after.shape or before.dtype != after.dtype:
            return False
        try:
            np.testing.assert_array_equal(before, after)
        except AssertionError:
            return False
        return True
    if type(before) is not type(after):
        return False
    try:
        return bool(np.all(before == after))
    except (TypeError, ValueError):
        return pickle.dumps(before) == pickle.dumps(after)


def validate_migrated_transition(
    original: Mapping[str, object],
    migrated: Mapping[str, object],
    oracle: OracleObjects,
) -> None:
    for key, before in original.items():
        if key not in migrated:
            raise ValueError(f'Migration removed original replay key {key!r}')
        if not _same_original_value(before, migrated[key]):
            raise ValueError(f'Migration changed original replay field {key!r}')
    for key, expected in oracle.as_replay_fields().items():
        if key not in migrated or not _same_original_value(expected, migrated[key]):
            raise ValueError(f'Oracle replay field failed validation: {key}')


def _stable_frame_rng(
    seed: int,
    task: str,
    episode_idx: int,
    sample_frame: int,
) -> np.random.Generator:
    digest = hashlib.sha256(task.encode('utf-8')).digest()
    task_seed = int.from_bytes(digest[:8], 'little')
    return np.random.default_rng(
        np.random.SeedSequence(
            [seed, task_seed, episode_idx, sample_frame]
        )
    )


def resolve_episode_dir(raw_data_dir: Path, task: str, episode_idx: int) -> Path:
    candidates = (
        raw_data_dir / task / 'all_variations' / 'episodes' / f'episode{episode_idx}',
        raw_data_dir / 'train' / task / 'all_variations' / 'episodes' / f'episode{episode_idx}',
        raw_data_dir / 'all_variations' / 'episodes' / f'episode{episode_idx}',
        raw_data_dir / f'episode{episode_idx}',
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    tried = '\n  '.join(str(path) for path in candidates)
    raise FileNotFoundError(
        f'Could not locate episode {episode_idx} for {task}. Tried:\n  {tried}'
    )


def augment_transition(
    original: Mapping[str, object],
    raw_data_dir: Path,
    task: str,
    replay_index: int,
    cameras: Sequence[str],
    max_objects: int,
    num_points: int,
    excluded_ids: Iterable[int],
    seed: int,
    included_ids: Optional[Iterable[int]] = None,
    slot_ids: Optional[Sequence[int]] = None,
    min_object_points: int = 20,
    frame_cache: Optional[OracleFrameCache] = None,
    task_prior_filter: bool = False,
    task_prior_radius: Optional[float] = None,
    task_prior_max_instances: Optional[int] = None,
    task_prior_background_extent: float = 0.60,
    task_prior_strict: bool = False,
    role_by_id: Optional[Mapping[int, int]] = None,
    group_by_id: Optional[Mapping[int, int]] = None,
    filter_thin_planes: bool = False,
    thin_plane_max_thickness: float = 0.005,
    thin_plane_min_extent: float = 0.08,
    filter_thin_planes_all_roles: bool = False,
) -> Tuple[Dict[str, object], OracleObjects, Optional[Path]]:
    existing_oracle_keys = set(ORACLE_KEYS).intersection(original)
    if existing_oracle_keys:
        raise ValueError(
            f'Replay {replay_index} already has Oracle fields: '
            f'{sorted(existing_oracle_keys)}'
        )
    # YARR final observations have terminal == -1 and intentionally undefined
    # metadata. They are not sampleable current transitions.
    terminal = int(np.asarray(original.get('terminal', -1)).item())
    if terminal == -1:
        oracle = empty_oracle_objects(max_objects, num_points)
        episode_dir = None
    else:
        for key in ('episode_idx', 'sample_frame'):
            if key not in original:
                raise KeyError(f'Replay {replay_index} is missing {key!r}')
        episode_idx = int(np.asarray(original['episode_idx']).item())
        sample_frame = int(np.asarray(original['sample_frame']).item())
        if episode_idx < 0 or sample_frame < 0:
            raise ValueError(
                f'Replay {replay_index} alignment is invalid: '
                f'episode_idx={episode_idx}, sample_frame={sample_frame}'
            )
        episode_dir = resolve_episode_dir(raw_data_dir, task, episode_idx)
        action_position = None
        if task_prior_filter:
            if 'gripper_pose' not in original:
                raise KeyError(
                    'Replay task-prior filtering requires gripper_pose'
                )
            action_position = np.asarray(original['gripper_pose']).reshape(-1)[:3]

        def build_oracle() -> OracleObjects:
            masks = load_frame_masks(episode_dir, sample_frame, cameras)
            return extract_oracle_objects(
                original,
                masks,
                cameras=cameras,
                max_objects=max_objects,
                num_points=num_points,
                excluded_ids=excluded_ids,
                included_ids=included_ids,
                slot_ids=slot_ids,
                min_object_points=min_object_points,
                rng=_stable_frame_rng(
                    seed, task, episode_idx, sample_frame
                ),
                task_name=task,
                task_prior_filter=task_prior_filter,
                action_position=action_position,
                task_prior_radius=task_prior_radius,
                task_prior_max_instances=task_prior_max_instances,
                task_prior_background_extent=task_prior_background_extent,
                task_prior_strict=task_prior_strict,
                role_by_id=role_by_id,
                group_by_id=group_by_id,
                filter_thin_planes=filter_thin_planes,
                thin_plane_max_thickness=thin_plane_max_thickness,
                thin_plane_min_extent=thin_plane_min_extent,
                filter_thin_planes_all_roles=filter_thin_planes_all_roles,
            )

        cache_key: Tuple[object, ...] = (task, episode_idx, sample_frame)
        if included_ids is not None:
            cache_key += (
                'included-ids',
                tuple(sorted(int(value) for value in included_ids)),
            )
        if slot_ids is not None:
            cache_key += ('slot-ids', tuple(int(value) for value in slot_ids))
        if task_prior_filter:
            assert action_position is not None
            cache_key += (
                'task-prior',
                tuple(np.round(action_position.astype(np.float64), 6)),
                task_prior_radius,
                task_prior_max_instances,
                task_prior_background_extent,
                task_prior_strict,
            )
        if role_by_id is not None:
            cache_key += (
                'roles',
                tuple(sorted((int(key), int(value)) for key, value in role_by_id.items())),
            )
        if group_by_id is not None:
            cache_key += (
                'groups',
                tuple(
                    sorted(
                        (int(key), int(value))
                        for key, value in group_by_id.items()
                    )
                ),
            )
        if filter_thin_planes:
            cache_key += (
                'thin-planes',
                float(thin_plane_max_thickness),
                float(thin_plane_min_extent),
                bool(filter_thin_planes_all_roles),
            )
        oracle = (
            frame_cache.get_or_compute(cache_key, build_oracle)
            if frame_cache is not None
            else build_oracle()
        )

    migrated = dict(original)
    migrated.update(oracle.as_replay_fields())
    validate_migrated_transition(original, migrated, oracle)
    return migrated, oracle, episode_dir


def _final_observation_oracle_for_visualization(
    final_transition: Mapping[str, object],
    previous_source: Optional[Path],
    raw_data_dir: Path,
    task: str,
    cameras: Sequence[str],
    max_objects: int,
    num_points: int,
    excluded_ids: Iterable[int],
    seed: int,
    min_object_points: int,
    slot_ids_by_episode: Optional[Mapping[int, Sequence[int]]] = None,
    task_prior_filter: bool = False,
    task_prior_radius: Optional[float] = None,
    task_prior_max_instances: Optional[int] = None,
    task_prior_background_extent: float = 0.60,
    task_prior_strict: bool = False,
    robot_handles_by_episode: Optional[Mapping[int, Sequence[int]]] = None,
    roles_by_episode: Optional[Mapping[int, Mapping[int, int]]] = None,
    groups_by_episode: Optional[Mapping[int, Mapping[int, int]]] = None,
    filter_thin_planes: bool = False,
    thin_plane_max_thickness: float = 0.005,
    thin_plane_min_extent: float = 0.08,
    filter_thin_planes_all_roles: bool = False,
) -> Optional[Tuple[OracleObjects, int, int]]:
    '''Recover a final observation's GT instances from prior alignment data.'''
    if previous_source is None or not previous_source.is_file():
        return None
    with previous_source.open('rb') as stream:
        previous = pickle.load(stream)
    if int(np.asarray(previous.get('terminal', -1)).item()) == -1:
        return None
    if 'episode_idx' not in previous or 'next_keypoint_frame' not in previous:
        return None
    episode_idx = int(np.asarray(previous['episode_idx']).item())
    sample_frame = int(np.asarray(previous['next_keypoint_frame']).item())
    if episode_idx < 0 or sample_frame < 0:
        return None
    if task_prior_filter and 'gripper_pose' not in final_transition:
        return None
    episode_dir = resolve_episode_dir(raw_data_dir, task, episode_idx)
    masks = load_frame_masks(episode_dir, sample_frame, cameras)
    effective_excluded_ids = list(excluded_ids)
    if robot_handles_by_episode is not None:
        effective_excluded_ids.extend(
            robot_handles_by_episode.get(episode_idx, ())
        )
    oracle = extract_oracle_objects(
        final_transition,
        masks,
        cameras=cameras,
        max_objects=max_objects,
        num_points=num_points,
        excluded_ids=effective_excluded_ids,
        slot_ids=(
            slot_ids_by_episode.get(episode_idx, ())
            if slot_ids_by_episode is not None
            else None
        ),
        min_object_points=min_object_points,
        rng=_stable_frame_rng(seed, task, episode_idx, sample_frame),
        task_name=task,
        task_prior_filter=task_prior_filter,
        action_position=(
            np.asarray(final_transition['gripper_pose']).reshape(-1)[:3]
            if task_prior_filter and 'gripper_pose' in final_transition
            else None
        ),
        task_prior_radius=task_prior_radius,
        task_prior_max_instances=task_prior_max_instances,
        task_prior_background_extent=task_prior_background_extent,
        task_prior_strict=task_prior_strict,
        role_by_id=(
            roles_by_episode.get(episode_idx, {})
            if roles_by_episode is not None
            else None
        ),
        group_by_id=(
            groups_by_episode.get(episode_idx, {})
            if groups_by_episode is not None
            else None
        ),
        filter_thin_planes=filter_thin_planes,
        thin_plane_max_thickness=thin_plane_max_thickness,
        thin_plane_min_extent=thin_plane_min_extent,
        filter_thin_planes_all_roles=filter_thin_planes_all_roles,
    )
    return oracle, episode_idx, sample_frame


def atomic_write_replay(
    destination: Path,
    original: Mapping[str, object],
    migrated: Mapping[str, object],
    oracle: OracleObjects,
    durable_write: bool = False,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f'{destination}.tmp')
    try:
        with temporary.open('wb') as stream:
            pickle.dump(migrated, stream, protocol=pickle.HIGHEST_PROTOCOL)
            if durable_write:
                # Oracle object experiment: forcing every file to stable storage
                # is useful for maximum durability, but is very slow on NFS and
                # network-mounted dataset directories.
                stream.flush()
                os.fsync(stream.fileno())
        with temporary.open('rb') as stream:
            reloaded = pickle.load(stream)
        validate_migrated_transition(original, reloaded, oracle)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _numeric_replay_files(directory: Path) -> List[Path]:
    files = list(directory.glob('*.replay'))
    invalid = [path.name for path in files if not path.stem.isdigit()]
    if invalid:
        raise ValueError(f'Non-numeric replay filenames in {directory}: {invalid}')
    return sorted(files, key=lambda path: int(path.stem))


def _parse_tasks(values: Sequence[str]) -> List[str]:
    tasks: List[str] = []
    for value in values:
        tasks.extend(part.strip() for part in value.split(',') if part.strip())
    return list(dict.fromkeys(tasks))


def _resolve_task_cache_directory(
    configured: Optional[Path],
    task_storage_dir: Path,
    task: str,
    default_name: str,
) -> Path:
    if configured is not None:
        return (configured.resolve() / task)
    return (task_storage_dir.resolve() / default_name)


def discover_task_directories(
    replay_dir: Path, requested_tasks: Sequence[str]
) -> List[Tuple[str, Path]]:
    tasks = _parse_tasks(requested_tasks)
    direct_files = _numeric_replay_files(replay_dir)
    if tasks == ['all'] or not tasks:
        if direct_files:
            return [(replay_dir.name, replay_dir)]
        discovered = [
            (child.name, child)
            for child in replay_dir.iterdir()
            if child.is_dir() and _numeric_replay_files(child)
        ]
        if not discovered:
            raise FileNotFoundError(
                f'No task replay directories found under {replay_dir}'
            )
        return sorted(discovered)
    if 'all' in tasks:
        raise ValueError('--task all cannot be combined with named tasks')
    if direct_files:
        if len(tasks) != 1:
            raise ValueError(
                'A direct task replay directory accepts exactly one --task'
            )
        return [(tasks[0], replay_dir)]
    resolved = [(task, replay_dir / task) for task in tasks]
    for task, directory in resolved:
        if not _numeric_replay_files(directory):
            raise FileNotFoundError(
                f'No *.replay files found for {task}: {directory}'
            )
    return resolved


def _describe(
    task: str,
    replay_index: int,
    transition: Mapping[str, object],
    oracle: OracleObjects,
) -> None:
    sentinel = int(np.asarray(transition.get('terminal', -1)).item()) == -1
    print(f'task={task} replay_index={replay_index} sentinel={sentinel}')
    if not sentinel:
        episode_idx = int(np.asarray(transition['episode_idx']).item())
        sample_frame = int(np.asarray(transition['sample_frame']).item())
        print(
            f'  episode_idx={episode_idx} sample_frame={sample_frame}'
        )
    count = int(oracle.valid.sum())
    print(
        f'  gt_objects={oracle.discovered_objects} stored_objects={count} '
        f'object_ids={oracle.ids[oracle.valid].tolist()}'
    )
    print(f'  object_roles={oracle.roles[oracle.valid].tolist()}')
    print(f'  raw_points_per_object={list(oracle.raw_point_counts)}')
    print(f'  sampled_points_per_object={[oracle.points.shape[1]] * count}')
    print(f'  centers={oracle.centers[oracle.valid].tolist()}')
    print(f'  sizes={oracle.sizes[oracle.valid].tolist()}')
    print(f'  filtered_small_objects={oracle.filtered_objects}')
    print(f'  filtered_thin_planes={oracle.thin_plane_objects}')
    print(f'  filtered_by_task_prior={oracle.prior_filtered_objects}')
    print(f'  excluded_by_id={oracle.excluded_objects}')
    print(f'  excluded_object_ids={list(oracle.excluded_object_ids)}')
    print(
        '  no_finite_point_object_ids='
        f'{list(oracle.no_finite_point_object_ids)}'
    )
    print(f'  small_object_ids={list(oracle.small_object_ids)}')
    print(f'  thin_plane_object_ids={list(oracle.thin_plane_object_ids)}')
    print(
        '  protected_thin_plane_object_ids='
        f'{list(oracle.protected_thin_plane_object_ids)}'
    )
    print(
        '  task_prior_filtered_object_ids='
        f'{list(oracle.prior_filtered_object_ids)}'
    )
    print(f'  truncated_object_ids={list(oracle.truncated_object_ids)}')
    print(
        '  temporal_filtered_object_ids='
        f'{list(oracle.temporal_filtered_object_ids)}'
    )
    print(
        f'  shapes=points{oracle.points.shape}, '
        f'centers{oracle.centers.shape}, sizes{oracle.sizes.shape}, '
        f'valid{oracle.valid.shape} '
        f'finite_points={bool(np.isfinite(oracle.points).all())}'
    )


def _scene_points_for_visualization(
    transition: Mapping[str, object],
    cameras: Sequence[str],
    max_points: int = MAX_VISUALIZATION_SCENE_POINTS,
) -> np.ndarray:
    '''Collect a bounded full-scene point cloud for visualization only.'''
    if max_points <= 0:
        raise ValueError('max_points must be positive')
    camera_points: List[np.ndarray] = []
    for camera in cameras:
        point_key = f'{camera}_point_cloud'
        if point_key not in transition:
            continue
        point_cloud = _point_cloud_hwc(
            np.asarray(transition[point_key]), point_key
        )
        points = point_cloud.reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        # Padding in YARR final-observation sentinels may be all zero. It does
        # not describe scene geometry and would collapse the plot to one dot.
        points = points[np.any(points != 0, axis=1)]
        if points.size:
            camera_points.append(points)
    if not camera_points:
        return np.empty((0, 3), dtype=np.float32)
    scene_points = np.concatenate(camera_points, axis=0)
    if len(scene_points) > max_points:
        # Evenly spaced deterministic sampling keeps visualization fast and
        # covers all cameras without retaining a large replay transition.
        indices = np.linspace(
            0, len(scene_points) - 1, num=max_points, dtype=np.int64
        )
        scene_points = scene_points[indices]
    return scene_points.astype(np.float32, copy=False)


def _instance_color(object_id: int) -> Tuple[float, float, float]:
    '''Return a deterministic categorical color for a decoded handle ID.'''
    # Golden-ratio hue spacing keeps adjacent simulator handles visually apart.
    hue = (int(object_id) * 0.618033988749895) % 1.0
    return colorsys.hsv_to_rgb(hue, 0.72, 0.92)


def _object_visualization_label(object_id: int, role: int) -> str:
    role_prefix = {
        ORACLE_ROLE_TARGET: 'T',
        ORACLE_ROLE_REFERENCE: 'R',
    }.get(int(role))
    return (
        f'{role_prefix}_{int(object_id)}'
        if role_prefix is not None
        else str(int(object_id))
    )


def _instance_boxes_for_mask(
    mask: np.ndarray,
    object_ids: Iterable[int],
    group_by_id: Optional[Mapping[int, int]] = None,
) -> Dict[int, Tuple[int, int, int, int]]:
    '''Return inclusive pixel boxes by handle or merged group representative.'''
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f'Instance box mask must be [H, W]; got {mask.shape}')
    boxes: Dict[int, Tuple[int, int, int, int]] = {}
    for object_id_value in object_ids:
        object_id = int(object_id_value)
        members = (
            [
                int(handle)
                for handle, representative in group_by_id.items()
                if int(representative) == object_id
            ]
            if group_by_id is not None
            else [object_id]
        )
        if object_id not in members:
            members.append(object_id)
        pixels = np.argwhere(np.isin(mask, members))
        if not pixels.size:
            continue
        y_min, x_min = np.min(pixels, axis=0)
        y_max, x_max = np.max(pixels, axis=0)
        boxes[object_id] = (
            int(x_min),
            int(y_min),
            int(x_max),
            int(y_max),
        )
    return boxes


def visualize_oracle_objects(
    oracle: OracleObjects,
    task: str,
    replay_index: int,
    output_dir: Path,
    scene_points: Optional[np.ndarray] = None,
    terminal: Optional[int] = None,
    episode_idx: Optional[int] = None,
    sample_frame: Optional[int] = None,
    camera_images: Optional[Mapping[str, np.ndarray]] = None,
    camera_masks: Optional[Mapping[str, np.ndarray]] = None,
    group_by_id: Optional[Mapping[int, int]] = None,
) -> Path:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError as exc:
        raise RuntimeError('--visualize-index requires matplotlib') from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{task}_replay_{replay_index}.png'
    figure = plt.figure(figsize=(18, 9))
    grid = figure.add_gridspec(
        2,
        4,
        height_ratios=(0.85, 1.15),
        hspace=0.24,
        wspace=0.18,
    )
    image_axes = [figure.add_subplot(grid[0, index]) for index in range(4)]
    axes = figure.add_subplot(grid[1, 0], projection='3d')
    top_axes = figure.add_subplot(grid[1, 1])
    front_axes = figure.add_subplot(grid[1, 2])
    side_axes = figure.add_subplot(grid[1, 3])
    camera_images = dict(camera_images or {})
    camera_masks = dict(camera_masks or {})
    retained_ids = [int(oracle.ids[slot]) for slot in np.flatnonzero(oracle.valid)]
    role_by_object = {
        int(oracle.ids[slot]): int(oracle.roles[slot])
        for slot in np.flatnonzero(oracle.valid)
    }
    camera_order = list(DEFAULT_CAMERAS)
    camera_order.extend(
        camera for camera in camera_images if camera not in camera_order
    )
    camera_titles = {
        'front': 'Front RGB',
        'left_shoulder': 'Left shoulder RGB',
        'right_shoulder': 'Right shoulder RGB',
        'wrist': 'Wrist RGB',
    }
    for image_axes_value, camera in zip(image_axes, camera_order[:4]):
        image = camera_images.get(camera)
        if image is not None:
            image_axes_value.imshow(image)
            mask = camera_masks.get(camera)
            if mask is not None and np.asarray(mask).shape == image.shape[:2]:
                boxes = _instance_boxes_for_mask(
                    mask,
                    retained_ids,
                    group_by_id=group_by_id,
                )
                for object_id, (x_min, y_min, x_max, y_max) in boxes.items():
                    color = _instance_color(object_id)
                    label = _object_visualization_label(
                        object_id,
                        role_by_object.get(object_id, ORACLE_ROLE_UNKNOWN),
                    )
                    image_axes_value.add_patch(
                        Rectangle(
                            (x_min, y_min),
                            x_max - x_min + 1,
                            y_max - y_min + 1,
                            fill=False,
                            edgecolor=color,
                            linewidth=2.0,
                            alpha=0.65,
                        )
                    )
                    image_axes_value.text(
                        x_min,
                        max(0, y_min - 2),
                        label,
                        color='black',
                        fontsize=8,
                        fontweight='bold',
                        ha='left',
                        va='bottom',
                        bbox={
                            'facecolor': color,
                            'edgecolor': color,
                            'alpha': 0.50,
                            'pad': 1.5,
                        },
                    )
        else:
            image_axes_value.text(
                0.5,
                0.5,
                f'{camera}\nRGB unavailable',
                transform=image_axes_value.transAxes,
                ha='center',
                va='center',
                color='gray',
            )
        image_axes_value.set_title(camera_titles.get(camera, f'{camera} RGB'))
        image_axes_value.set_axis_off()
    scene_points = (
        np.asarray(scene_points)
        if scene_points is not None
        else np.empty((0, 3), dtype=np.float32)
    )
    if scene_points.size:
        axes.scatter(
            scene_points[:, 0],
            scene_points[:, 1],
            scene_points[:, 2],
            c='lightgray',
            s=0.25,
            alpha=0.18,
            depthshade=False,
            label='scene',
        )
        scene_style = {
            'c': 'lightgray',
            's': 0.25,
            'alpha': 0.18,
        }
        top_axes.scatter(
            scene_points[:, 0], scene_points[:, 1], **scene_style
        )
        front_axes.scatter(
            scene_points[:, 0], scene_points[:, 2], **scene_style
        )
        side_axes.scatter(
            scene_points[:, 1], scene_points[:, 2], **scene_style
        )
    for slot in np.flatnonzero(oracle.valid):
        points = oracle.points[slot]
        object_id = int(oracle.ids[slot])
        role = int(oracle.roles[slot])
        label = _object_visualization_label(object_id, role)
        color = _instance_color(object_id)
        axes.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=[color],
            s=2,
            label=label,
        )
        object_style = {'c': [color], 's': 2}
        top_axes.scatter(points[:, 0], points[:, 1], **object_style)
        front_axes.scatter(points[:, 0], points[:, 2], **object_style)
        side_axes.scatter(points[:, 1], points[:, 2], **object_style)
        center = oracle.centers[slot]
        label_style = {
            'color': color,
            'fontsize': 7,
            'bbox': {
                'facecolor': 'white',
                'edgecolor': color,
                'alpha': 0.35,
                'pad': 1.0,
            },
        }
        axes.text(center[0], center[1], center[2], label, **label_style)
        top_axes.text(center[0], center[1], label, **label_style)
        front_axes.text(center[0], center[2], label, **label_style)
        side_axes.text(center[1], center[2], label, **label_style)
    sentinel = terminal == -1
    alignment = ''
    if episode_idx is not None and sample_frame is not None:
        alignment = f' ep={episode_idx} frame={sample_frame}'
    figure.suptitle(
        f'{task} replay {replay_index}{alignment}: Oracle GT instances '
        f'(valid={int(oracle.valid.sum())}, '
        f'target={int(np.sum(oracle.roles == ORACLE_ROLE_TARGET))}, '
        f'reference={int(np.sum(oracle.roles == ORACLE_ROLE_REFERENCE))}'
        + (', final sentinel' if sentinel else '')
        + ')'
    )
    axes.set_title('3D perspective')
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')
    x_min, y_min, z_min, x_max, y_max, z_max = VISUALIZATION_SCENE_BOUNDS
    axes.set_xlim(x_min, x_max)
    axes.set_ylim(y_min, y_max)
    axes.set_zlim(z_min, z_max)
    axes.set_box_aspect(
        (x_max - x_min, y_max - y_min, z_max - z_min)
    )
    top_axes.set_title('Top orthographic (XY, view along Z)')
    top_axes.set_xlabel('x')
    top_axes.set_ylabel('y')
    top_axes.set_xlim(x_min, x_max)
    top_axes.set_ylim(y_min, y_max)
    top_axes.set_aspect('equal', adjustable='box')
    top_axes.grid(True, alpha=0.2)

    front_axes.set_title('Front orthographic (XZ, view along Y)')
    front_axes.set_xlabel('x')
    front_axes.set_ylabel('z')
    front_axes.set_xlim(x_min, x_max)
    front_axes.set_ylim(z_min, z_max)
    front_axes.set_aspect('equal', adjustable='box')
    front_axes.grid(True, alpha=0.2)

    side_axes.set_title('Side orthographic (YZ, view along X)')
    side_axes.set_xlabel('y')
    side_axes.set_ylabel('z')
    side_axes.set_xlim(y_min, y_max)
    side_axes.set_ylim(z_min, z_max)
    side_axes.set_aspect('equal', adjustable='box')
    side_axes.grid(True, alpha=0.2)
    if scene_points.size or oracle.valid.any():
        axes.legend(title='Num / role')
    else:
        reason = (
            'Final-observation sentinel:\nno scene or Oracle points available'
            if sentinel
            else 'No finite scene or valid Oracle points available'
        )
        axes.text2D(
            0.5,
            0.5,
            reason,
            transform=axes.transAxes,
            ha='center',
            va='center',
        )
        for projection_axes in (top_axes, front_axes, side_axes):
            projection_axes.text(
                0.5,
                0.5,
                reason,
                transform=projection_axes.transAxes,
                ha='center',
                va='center',
            )
    figure.subplots_adjust(top=0.90, bottom=0.06, left=0.04, right=0.98)
    figure.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(figure)
    print(f'Oracle visualization saved: {output_path}', flush=True)
    return output_path


def _copy_metadata(
    source_dir: Path, destination_dir: Path, overwrite: bool
) -> None:
    if source_dir.resolve() == destination_dir.resolve():
        return
    for source in source_dir.iterdir():
        if (
            source.is_dir()
            or source.suffix == '.replay'
            or source.name == REPLAY_METADATA_CACHE_NAME
            or source.name.endswith('.tmp')
        ):
            continue
        destination = destination_dir / source.name
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f'Output metadata exists: {destination}; use --overwrite'
            )
        destination_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _select_dry_run_files(
    files: Sequence[Path],
    seed: int,
    sample_count: int,
    visualize_index: Optional[int],
) -> List[Path]:
    rng = np.random.default_rng(seed)
    selected: List[Path] = []
    for index in rng.permutation(len(files)):
        source = files[int(index)]
        with source.open('rb') as stream:
            transition = pickle.load(stream)
        if int(np.asarray(transition.get('terminal', -1)).item()) != -1:
            selected.append(source)
            # Stop as soon as enough ordinary transitions have been found.
            # This avoids opening every replay merely to run a small dry-run.
            if len(selected) == sample_count:
                break
    if visualize_index is not None:
        visual = files[0].parent / f'{visualize_index}.replay'
        if not visual.is_file():
            raise FileNotFoundError(f'Visualization replay is missing: {visual}')
        if visual not in selected:
            selected.append(visual)
    return selected


def _select_visualization_files(
    files: Sequence[Path],
    visualize_index: Optional[int],
    visualize_every: int,
) -> List[Path]:
    if visualize_index is not None:
        by_index = {int(path.stem): path for path in files}
        if visualize_index not in by_index:
            missing = files[0].parent / (
                str(visualize_index) + '.replay'
            )
            raise FileNotFoundError(
                f'Visualization replay is missing: {missing}'
            )
        return [by_index[visualize_index]]
    if visualize_every > 0:
        return list(files[::visualize_every])
    return []


def _bounded_thread_map(function, items: Sequence[Path], workers: int):
    '''Yield completed results while keeping only 2 * workers tasks in flight.'''
    if workers == 1:
        for item in items:
            yield function(item)
        return

    item_iterator = iter(items)
    executor = ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix='oracle-replay'
    )
    pending = set()
    try:
        for _ in range(min(len(items), workers * 2)):
            pending.add(executor.submit(function, next(item_iterator)))

        while pending:
            completed, pending = wait(
                pending, return_when=FIRST_COMPLETED
            )
            for future in completed:
                yield future.result()
                try:
                    item = next(item_iterator)
                except StopIteration:
                    continue
                pending.add(executor.submit(function, item))
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def _replay_file_name_fingerprint(files: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for source in files:
        digest.update(source.name.encode('utf-8'))
        digest.update(b'\0')
    return digest.hexdigest()


def _load_or_scan_replay_metadata(
    files: Sequence[Path],
    *,
    show_progress: bool,
    refresh_cache: bool,
    cache_dir: Optional[Path] = None,
) -> ReplayMetadataArrays:
    '''Load a compact replay index or deserialize every replay once.'''
    if not files:
        empty = np.empty((0,), dtype=np.int64)
        return empty, empty.copy(), empty.copy(), empty.copy()

    source_dir = files[0].parent
    fingerprint = _replay_file_name_fingerprint(files)
    memory_key = (str(source_dir.resolve()), fingerprint)
    memory_cached = _REPLAY_METADATA_MEMORY_CACHE.get(memory_key)
    if memory_cached is not None:
        if show_progress:
            tqdm.write(
                'episode detection: replay metadata memory cache hit '
                f'({len(files)} replay files)'
            )
        return memory_cached

    cache_path = (cache_dir or source_dir) / REPLAY_METADATA_CACHE_NAME
    if cache_path.is_file() and not refresh_cache:
        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                version = int(np.asarray(payload['version']).item())
                cached_fingerprint = str(
                    np.asarray(payload['file_name_fingerprint']).item()
                )
                file_count = int(np.asarray(payload['file_count']).item())
                if (
                    version != REPLAY_METADATA_CACHE_VERSION
                    or cached_fingerprint != fingerprint
                    or file_count != len(files)
                ):
                    raise ValueError('stale replay metadata cache')
                metadata = (
                    np.asarray(payload['terminal'], dtype=np.int64).copy(),
                    np.asarray(payload['episode_idx'], dtype=np.int64).copy(),
                    np.asarray(payload['sample_frame'], dtype=np.int64).copy(),
                    np.asarray(
                        payload['next_keypoint_frame'], dtype=np.int64
                    ).copy(),
                )
            if any(len(values) != len(files) for values in metadata):
                raise ValueError('replay metadata cache length mismatch')
        except (BadZipFile, EOFError, KeyError, OSError, ValueError):
            if show_progress:
                tqdm.write(
                    'episode detection: stale replay metadata cache; '
                    'rebuilding'
                )
        else:
            _REPLAY_METADATA_MEMORY_CACHE[memory_key] = metadata
            if show_progress:
                tqdm.write(
                    'episode detection: replay metadata disk cache hit '
                    f'({len(files)} replay files)'
                )
            return metadata

    terminal = np.full(len(files), -1, dtype=np.int64)
    episode_idx = np.full(len(files), -1, dtype=np.int64)
    sample_frame = np.full(len(files), -1, dtype=np.int64)
    next_keypoint_frame = np.full(len(files), -1, dtype=np.int64)
    progress = tqdm(
        enumerate(files),
        total=len(files),
        desc='episode detection: replay metadata scan',
        unit='replay',
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for index, source in progress:
        with source.open('rb') as stream:
            transition = pickle.load(stream)
        terminal[index] = int(
            np.asarray(transition.get('terminal', -1)).item()
        )
        if terminal[index] == -1:
            continue
        if 'episode_idx' not in transition or 'sample_frame' not in transition:
            continue
        episode_idx[index] = int(
            np.asarray(transition['episode_idx']).item()
        )
        sample_frame[index] = int(
            np.asarray(transition['sample_frame']).item()
        )
        next_keypoint_frame[index] = int(
            np.asarray(
                transition.get('next_keypoint_frame', sample_frame[index])
            ).item()
        )

    metadata = terminal, episode_idx, sample_frame, next_keypoint_frame
    _REPLAY_METADATA_MEMORY_CACHE[memory_key] = metadata
    temporary = Path(f'{cache_path}.tmp')
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open('wb') as stream:
            np.savez_compressed(
                stream,
                version=np.asarray(REPLAY_METADATA_CACHE_VERSION),
                file_name_fingerprint=np.asarray(fingerprint),
                file_count=np.asarray(len(files)),
                terminal=terminal,
                episode_idx=episode_idx,
                sample_frame=sample_frame,
                next_keypoint_frame=next_keypoint_frame,
            )
        os.replace(temporary, cache_path)
        if show_progress:
            tqdm.write(
                f'episode detection: replay metadata cache saved: {cache_path}'
            )
    except OSError as exc:
        if show_progress:
            tqdm.write(
                'episode detection: could not save replay metadata cache '
                f'({exc})'
            )
    finally:
        if temporary.exists():
            temporary.unlink()
    return metadata


def _limit_episode_candidates(
    candidates: Sequence[object],
    limit: int,
    strategy: str,
) -> List[object]:
    '''Deterministically limit episode evidence without random sampling.'''
    if strategy not in ('early', 'uniform'):
        raise ValueError(f'Unknown episode sampling strategy: {strategy}')
    values = list(candidates)
    if len(values) <= limit:
        return values
    if strategy == 'early':
        return values[:limit]
    indices = np.linspace(
        0, len(values) - 1, num=limit, dtype=np.int64
    )
    return [values[int(index)] for index in indices]


def _open_gripper_prefix(
    frame_sources: Sequence[Tuple[int, Path]],
    gripper_openings: Mapping[int, float],
) -> List[Tuple[int, Path]]:
    '''Keep consecutive episode-start evidence before the first close event.'''
    prefix = []
    for sample_frame, source in frame_sources:
        if gripper_openings[sample_frame] < 0.5:
            break
        prefix.append((sample_frame, source))
    # One frame cannot provide motion evidence. Keep the original early sample
    # as a compatibility fallback for unusual episodes that begin closed.
    return prefix if len(prefix) >= 2 else list(frame_sources)


def _episode_detection_sources(
    files: Sequence[Path],
    frames_per_episode: int,
    requested_episode_ids: Optional[Iterable[int]] = None,
    cached_episode_ids: Iterable[int] = (),
    show_progress: bool = True,
    preserve_action_edges: bool = False,
    refresh_metadata_cache: bool = False,
    metadata_cache_dir: Optional[Path] = None,
    sampling_strategy: str = 'uniform',
) -> Dict[int, List[Tuple[int, Path]]]:
    '''Merge augmented replay segments, then sample frames per raw episode.'''
    requested = (
        None
        if requested_episode_ids is None
        else {int(value) for value in requested_episode_ids}
    )
    cached = {int(value) for value in cached_episode_ids}
    replay_info_path = files[0].parent / 'replay_info.npy' if files else None
    if replay_info_path is not None and replay_info_path.is_file():
        try:
            replay_info = np.asarray(
                np.load(replay_info_path, allow_pickle=False)
            ).reshape(-1)
        except (OSError, ValueError):
            replay_info = np.empty((0,), dtype=np.int8)
        if len(replay_info) == len(files):
            segments: List[List[int]] = []
            segment: List[int] = []
            for index, terminal_value in enumerate(replay_info):
                segment.append(index)
                if int(terminal_value) == -1:
                    segments.append(segment)
                    segment = []
            if segment:
                segments.append(segment)

            grouped: Dict[int, Dict[Tuple[int, int], Path]] = {}
            progress = tqdm(
                segments,
                desc='episode detection: replay segments',
                unit='segment',
                dynamic_ncols=True,
                disable=not show_progress,
            )
            for indices in progress:
                ordinary = [
                    index
                    for index in indices
                    if int(replay_info[index]) != -1
                ]
                if not ordinary:
                    continue
                anchor_source = files[ordinary[0]]
                with anchor_source.open('rb') as stream:
                    anchor = pickle.load(stream)
                if 'episode_idx' not in anchor:
                    continue
                episode_idx = int(np.asarray(anchor['episode_idx']).item())
                if requested is not None and episode_idx not in requested:
                    continue
                if episode_idx in cached:
                    grouped.setdefault(episode_idx, {})
                    continue
                # Each demo-augmentation start produces another replay segment
                # with the same raw episode_idx. Sample each segment first, but
                # merge all of them before applying the episode-wide limit.
                ordinary = _limit_episode_candidates(
                    ordinary, frames_per_episode, sampling_strategy
                )
                frame_sources = grouped.setdefault(episode_idx, {})
                for index in ordinary:
                    source = files[index]
                    transition = anchor if source == anchor_source else None
                    if transition is None:
                        with source.open('rb') as stream:
                            transition = pickle.load(stream)
                    if int(np.asarray(transition.get('terminal', -1)).item()) == -1:
                        continue
                    sample_frame = int(
                        np.asarray(transition['sample_frame']).item()
                    )
                    next_keypoint_frame = sample_frame
                    if (
                        preserve_action_edges
                        and 'next_keypoint_frame' in transition
                    ):
                        next_keypoint_frame = int(
                            np.asarray(transition['next_keypoint_frame']).item()
                        )
                    frame_sources.setdefault(
                        (sample_frame, next_keypoint_frame), source
                    )

            selected: Dict[int, List[Tuple[int, Path]]] = {}
            for episode_idx, by_edge in grouped.items():
                candidates = [
                    (key[0], source)
                    for key, source in sorted(by_edge.items())
                ]
                candidates = _limit_episode_candidates(
                    candidates, frames_per_episode, sampling_strategy
                )
                selected[episode_idx] = candidates
            return selected

    # Compatibility fallback for old replay folders without replay_info.npy.
    # The first scan opens every large replay pickle; later runs use the compact
    # metadata cache unless the ordered replay filename set changes.
    (
        terminal_values,
        episode_indices,
        sample_frame_values,
        next_keypoint_frames,
    ) = _load_or_scan_replay_metadata(
        files,
        show_progress=show_progress,
        refresh_cache=refresh_metadata_cache,
        cache_dir=metadata_cache_dir,
    )
    grouped: Dict[int, Dict[Tuple[int, int], Path]] = {}
    for index, source in enumerate(files):
        if terminal_values[index] == -1:
            continue
        if episode_indices[index] < 0 or sample_frame_values[index] < 0:
            continue
        episode_idx = int(episode_indices[index])
        if requested is not None and episode_idx not in requested:
            continue
        if episode_idx in cached:
            grouped.setdefault(episode_idx, {})
            continue
        sample_frame = int(sample_frame_values[index])
        next_keypoint_frame = (
            int(next_keypoint_frames[index])
            if preserve_action_edges
            else sample_frame
        )
        grouped.setdefault(episode_idx, {}).setdefault(
            (sample_frame, next_keypoint_frame), source
        )

    selected: Dict[int, List[Tuple[int, Path]]] = {}
    for episode_idx, by_edge in grouped.items():
        candidates = [
            (key[0], source)
            for key, source in sorted(by_edge.items())
        ]
        candidates = _limit_episode_candidates(
            candidates, frames_per_episode, sampling_strategy
        )
        selected[episode_idx] = candidates
    return selected


def _episode_ids_for_selected_files(
    files: Sequence[Path],
    previous_file_by_index: Mapping[int, Path],
) -> Tuple[int, ...]:
    '''Find only the episodes needed by a dry-run selection.'''
    episode_ids = set()
    for source in files:
        with source.open('rb') as stream:
            transition = pickle.load(stream)
        if int(np.asarray(transition.get('terminal', -1)).item()) == -1:
            previous = previous_file_by_index.get(int(source.stem))
            if previous is None:
                continue
            with previous.open('rb') as stream:
                transition = pickle.load(stream)
        if 'episode_idx' in transition:
            episode_ids.add(int(np.asarray(transition['episode_idx']).item()))
    return tuple(sorted(episode_ids))


def _load_low_dim_observations(episode_dir: Path) -> Sequence[object]:
    low_dim_path = episode_dir / 'low_dim_obs.pkl'
    if not low_dim_path.is_file():
        raise FileNotFoundError(
            f'Robot handle detection requires {low_dim_path}'
        )
    with low_dim_path.open('rb') as stream:
        observations = pickle.load(stream)
    try:
        observation_count = len(observations)
    except TypeError as exc:
        raise ValueError(f'Invalid observations in {low_dim_path}') from exc
    if observation_count == 0:
        raise ValueError(f'No observations in {low_dim_path}')
    return observations


def _gripper_states_from_observations(
    observations: Sequence[object],
    sample_frames: Sequence[int],
    source_name: object = 'low_dim_obs.pkl',
) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    positions: Dict[int, np.ndarray] = {}
    openings: Dict[int, float] = {}
    for sample_frame in sample_frames:
        try:
            observation = observations[sample_frame]
            pose = np.asarray(observation.gripper_pose).reshape(-1)
            gripper_open = float(observation.gripper_open)
        except (IndexError, TypeError, AttributeError) as exc:
            raise ValueError(
                f'Cannot read current gripper_pose for frame {sample_frame} '
                f'from {source_name}'
            ) from exc
        if pose.size < 3 or not np.isfinite(pose[:3]).all():
            raise ValueError(
                f'Invalid current gripper_pose at frame {sample_frame}: {pose}'
            )
        positions[sample_frame] = pose[:3].astype(np.float32)
        if not np.isfinite(gripper_open):
            raise ValueError(
                f'Invalid gripper_open at frame {sample_frame}: '
                f'{gripper_open}'
            )
        openings[sample_frame] = gripper_open
    return positions, openings


def _load_current_gripper_states(
    episode_dir: Path, sample_frames: Sequence[int]
) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
    observations = _load_low_dim_observations(episode_dir)
    return _gripper_states_from_observations(
        observations,
        sample_frames,
        episode_dir / 'low_dim_obs.pkl',
    )


def _select_adaptive_robot_frames(
    observations: Sequence[object],
    *,
    stride: int = 5,
    initial_window: int = 100,
    max_frames: int = 64,
    motion_threshold: float = 0.02,
) -> RobotFrameSelection:
    '''Sample an open-gripper raw prefix and extend it only when motion is weak.'''
    if stride <= 0 or max_frames <= 0:
        raise ValueError('stride and max_frames must be positive')
    if initial_window < 0:
        raise ValueError('initial_window must be non-negative')
    if motion_threshold < 0:
        raise ValueError('motion_threshold must be non-negative')
    if len(observations) == 0:
        raise ValueError('observations must not be empty')

    first_close: Optional[int] = None
    for frame, observation in enumerate(observations):
        try:
            gripper_open = float(observation.gripper_open)
        except (TypeError, AttributeError, ValueError) as exc:
            raise ValueError(f'Invalid gripper_open at frame {frame}') from exc
        if not np.isfinite(gripper_open):
            raise ValueError(f'Invalid gripper_open at frame {frame}')
        if gripper_open < 0.5:
            first_close = frame
            break

    open_stop = len(observations) if first_close is None else first_close
    candidates = list(range(0, open_stop, stride))
    # Unusual episodes can begin closed. Preserve one frame so the wrist-view
    # geometric seed can still produce a conservative result.
    if not candidates:
        candidates = [0]
    initial = [frame for frame in candidates if frame <= initial_window]
    selected = (initial or candidates[:1])[:max_frames]
    positions, _ = _gripper_states_from_observations(observations, selected)
    origin = positions[selected[0]]
    max_displacement = max(
        float(np.linalg.norm(positions[frame] - origin)) for frame in selected
    )

    motion_sufficient = max_displacement + 1e-8 >= motion_threshold
    if not motion_sufficient and len(selected) < max_frames:
        for frame in candidates[len(initial):]:
            selected.append(frame)
            position, _ = _gripper_states_from_observations(
                observations, [frame]
            )
            max_displacement = max(
                max_displacement,
                float(np.linalg.norm(position[frame] - origin)),
            )
            motion_sufficient = max_displacement + 1e-8 >= motion_threshold
            if motion_sufficient or len(selected) >= max_frames:
                break

    stopped_on_close = bool(
        first_close is not None
        and selected
        and selected[-1] + stride >= first_close
    )
    return RobotFrameSelection(
        sample_frames=tuple(selected),
        max_gripper_displacement=max_displacement,
        motion_sufficient=motion_sufficient,
        stopped_on_close=stopped_on_close,
    )


def _load_current_gripper_positions(
    episode_dir: Path, sample_frames: Sequence[int]
) -> Dict[int, np.ndarray]:
    positions, _ = _load_current_gripper_states(episode_dir, sample_frames)
    return positions


def _detect_task_robot_handles(
    task: str,
    files: Sequence[Path],
    raw_data_dir: Path,
    cameras: Sequence[str],
    excluded_ids: Iterable[int],
    frames_per_episode: int,
    frame_stride: int,
    initial_window: int,
    motion_threshold: float,
    link_motion_threshold: float,
    adjacency_distance: float,
    cache_dir: Path,
    refresh_cache: bool,
    write_cache: bool,
    requested_episode_ids: Optional[Iterable[int]] = None,
    show_progress: bool = True,
    refresh_metadata_cache: bool = False,
    metadata_cache_dir: Optional[Path] = None,
) -> Dict[int, Tuple[int, ...]]:
    if 'wrist' not in cameras:
        raise ValueError(
            '--detect-robot-handles requires wrist in the selected cameras'
        )
    requested_episode_set = (
        None
        if requested_episode_ids is None
        else {int(value) for value in requested_episode_ids}
    )
    cache_config = {
        'source': 'raw_depth',
        'stride': int(frame_stride),
        'initial_window': int(initial_window),
        'max_frames': int(frames_per_episode),
        'motion_threshold': float(motion_threshold),
        'link_motion_threshold': float(link_motion_threshold),
        'adjacency_distance': float(adjacency_distance),
    }
    cached_detections: Dict[int, RobotHandleDetection] = {}
    stale_cache_count = 0
    if not refresh_cache:
        for path in cache_dir.glob('episode_*.json'):
            try:
                episode_idx = int(path.stem.rsplit('_', 1)[1])
            except (IndexError, ValueError):
                continue
            if (
                requested_episode_set is not None
                and episode_idx not in requested_episode_set
            ):
                continue
            try:
                detection = load_robot_handle_detection(
                    path, expected_sampling_config=cache_config
                )
                if detection.episode_idx != episode_idx:
                    raise ValueError(
                        f'cache episode {detection.episode_idx} does not '
                        f'match filename episode {episode_idx}'
                    )
            except (OSError, KeyError, TypeError, ValueError):
                stale_cache_count += 1
                continue
            cached_detections[episode_idx] = detection
    if stale_cache_count:
        tqdm.write(
            f'{task}: ignoring {stale_cache_count} stale robot-handle '
            'cache file(s); affected episodes will be detected again'
        )
    selected = _episode_detection_sources(
        files,
        1,
        requested_episode_ids=requested_episode_set,
        cached_episode_ids=cached_detections,
        show_progress=show_progress,
        refresh_metadata_cache=refresh_metadata_cache,
        metadata_cache_dir=metadata_cache_dir,
        sampling_strategy='early',
    )
    handles_by_episode: Dict[int, Tuple[int, ...]] = {}
    progress = tqdm(
        sorted(selected.items()),
        desc=f'{task}: robot handle detection',
        unit='episode',
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for episode_idx, frame_sources in progress:
        cache_path = cache_dir / f'episode_{episode_idx:04d}.json'
        detection = cached_detections.get(episode_idx)
        if detection is None:
            if not frame_sources:
                raise RuntimeError(
                    f'Robot cache missing frame sources for episode {episode_idx}'
                )
            episode_dir = resolve_episode_dir(raw_data_dir, task, episode_idx)
            observations = _load_low_dim_observations(episode_dir)
            selection = _select_adaptive_robot_frames(
                observations,
                stride=frame_stride,
                initial_window=initial_window,
                max_frames=frames_per_episode,
                motion_threshold=motion_threshold,
            )
            sample_frames = list(selection.sample_frames)
            gripper_positions, gripper_openings = (
                _gripper_states_from_observations(
                    observations,
                    sample_frames,
                    episode_dir / 'low_dim_obs.pkl',
                )
            )
            tqdm.write(
                f'{task} episode={episode_idx}: raw robot sampling '
                f'count={len(sample_frames)} range='
                f'{sample_frames[0]}-{sample_frames[-1]} stride={frame_stride} '
                f'motion={selection.max_gripper_displacement:.4f}m '
                f'sufficient={selection.motion_sufficient} '
                f'stopped_on_close={selection.stopped_on_close}'
            )
            evidence = []
            frame_progress = tqdm(
                sample_frames,
                desc=f'{task} episode={episode_idx}: robot raw frames',
                unit='frame',
                dynamic_ncols=True,
                leave=False,
                disable=not show_progress,
            )
            for sample_frame in frame_progress:
                masks = load_frame_masks(episode_dir, sample_frame, cameras)
                point_clouds = load_raw_frame_point_clouds(
                    episode_dir,
                    sample_frame,
                    cameras,
                    observations[sample_frame],
                )
                evidence.append(
                    build_robot_frame_evidence(
                        sample_frame,
                        gripper_positions[sample_frame],
                        masks,
                        point_clouds,
                        gripper_open=gripper_openings[sample_frame],
                        excluded_ids=excluded_ids,
                    )
                )
            detection = detect_robot_handles(
                episode_idx,
                evidence,
                min_link_motion=link_motion_threshold,
                adjacency_distance=adjacency_distance,
            )
            if write_cache:
                save_robot_handle_detection(
                    cache_path,
                    detection,
                    sampling_config=cache_config,
                )
            if not selection.motion_sufficient:
                tqdm.write(
                    f'{task} episode={episode_idx}: WARNING gripper moved only '
                    f'{selection.max_gripper_displacement:.4f} m in the sampled '
                    'open prefix; motion-based robot evidence is weak'
                )
        handles_by_episode[episode_idx] = detection.robot_handles
        observed_handles = sorted(
            set(detection.robot_handles)
            | set(detection.ambiguous_handles)
            | set(detection.grasped_handles)
        )
        tqdm.write(
            f'{task} episode={episode_idx}: detected robot handles='
            f'{list(detection.robot_handles)} gripper='
            f'{list(detection.gripper_handles)} arm='
            f'{list(detection.arm_handles)} ambiguous='
            f'{list(detection.ambiguous_handles)} grasped='
            f'{list(detection.grasped_handles)} classified='
            f'{len(observed_handles)} frames='
            f'{list(detection.sampled_frames)}',
        )
        if not detection.gripper_handles:
            tqdm.write(
                f'{task} episode={episode_idx}: WARNING no gripper seed was '
                'detected; no arm chain can be excluded for this episode'
            )
    return handles_by_episode


def _detect_task_relevant_handles(
    task: str,
    files: Sequence[Path],
    raw_data_dir: Path,
    cameras: Sequence[str],
    excluded_ids: Iterable[int],
    robot_handles_by_episode: Mapping[int, Sequence[int]],
    frames_per_episode: int,
    cache_dir: Path,
    refresh_cache: bool,
    write_cache: bool,
    task_prior_radius: Optional[float],
    task_prior_max_instances: Optional[int],
    task_prior_background_extent: float,
    requested_episode_ids: Optional[Iterable[int]] = None,
    show_progress: bool = True,
    refresh_metadata_cache: bool = False,
    metadata_cache_dir: Optional[Path] = None,
) -> Tuple[
    Dict[int, Tuple[int, ...]],
    Dict[int, Dict[int, int]],
    Dict[int, Dict[int, int]],
]:
    '''Detect one stable task-handle whitelist per episode.'''
    requested_episode_set = (
        None
        if requested_episode_ids is None
        else {int(value) for value in requested_episode_ids}
    )
    cached_detections: Dict[int, TaskHandleDetection] = {}
    stale_cache_count = 0
    if not refresh_cache:
        for path in cache_dir.glob('episode_*.json'):
            try:
                episode_idx = int(path.stem.rsplit('_', 1)[1])
            except (IndexError, ValueError):
                continue
            if (
                requested_episode_set is not None
                and episode_idx not in requested_episode_set
            ):
                continue
            try:
                detection = load_task_handle_detection(path)
                if detection.episode_idx != episode_idx:
                    raise ValueError(
                        f'cache episode {detection.episode_idx} does not '
                        f'match filename episode {episode_idx}'
                    )
            except (OSError, KeyError, TypeError, ValueError):
                stale_cache_count += 1
                continue
            cached_detections[episode_idx] = detection
    if stale_cache_count:
        tqdm.write(
            f'{task}: ignoring {stale_cache_count} stale task-handle '
            'cache file(s); affected episodes will be detected again'
        )

    selected = _episode_detection_sources(
        files,
        frames_per_episode,
        requested_episode_ids=requested_episode_set,
        cached_episode_ids=cached_detections,
        show_progress=show_progress,
        preserve_action_edges=True,
        refresh_metadata_cache=refresh_metadata_cache,
        metadata_cache_dir=metadata_cache_dir,
    )
    handles_by_episode: Dict[int, Tuple[int, ...]] = {}
    roles_by_episode: Dict[int, Dict[int, int]] = {}
    groups_by_episode: Dict[int, Dict[int, int]] = {}
    progress = tqdm(
        sorted(selected.items()),
        desc=f'{task}: task handle detection',
        unit='episode',
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for episode_idx, frame_sources in progress:
        cache_path = cache_dir / f'episode_{episode_idx:04d}.json'
        detection = cached_detections.get(episode_idx)
        if detection is None:
            if not frame_sources:
                raise RuntimeError(
                    f'Task cache missing frame sources for episode {episode_idx}'
                )
            episode_dir = resolve_episode_dir(raw_data_dir, task, episode_idx)
            sample_frames = [frame for frame, _ in frame_sources]
            gripper_positions, gripper_openings = _load_current_gripper_states(
                episode_dir, sample_frames
            )
            effective_excluded_ids = list(excluded_ids)
            effective_excluded_ids.extend(
                robot_handles_by_episode.get(episode_idx, ())
            )
            evidence = []
            for sample_frame, source in frame_sources:
                with source.open('rb') as stream:
                    transition = pickle.load(stream)
                if 'gripper_pose' not in transition:
                    raise KeyError(
                        f'Task handle detection requires gripper_pose in {source}'
                    )
                action_position = np.asarray(
                    transition['gripper_pose']
                ).reshape(-1)[:3]
                masks = load_frame_masks(episode_dir, sample_frame, cameras)
                point_clouds = {
                    camera: _point_cloud_hwc(
                        np.asarray(transition[f'{camera}_point_cloud']),
                        f'{camera}_point_cloud',
                    )
                    for camera in cameras
                }
                evidence.append(
                    build_task_frame_evidence(
                        sample_frame,
                        gripper_positions[sample_frame],
                        action_position,
                        masks,
                        point_clouds,
                        excluded_ids=effective_excluded_ids,
                        gripper_open=gripper_openings[sample_frame],
                    )
                )
            detection = detect_task_handles(
                task,
                episode_idx,
                evidence,
                interaction_radius=task_prior_radius,
                max_instances=task_prior_max_instances,
                background_extent=task_prior_background_extent,
            )
            if write_cache:
                save_task_handle_detection(cache_path, detection)
        handles_by_episode[episode_idx] = detection.grouped_slot_handles
        roles_by_episode[episode_idx] = detection.role_by_group
        groups_by_episode[episode_idx] = detection.group_by_handle
        tqdm.write(
            f'{task} episode={episode_idx}: task handles='
            f'{list(detection.task_handles)} interaction='
            f'{list(detection.interaction_handles)} adjacent='
            f'{list(detection.adjacent_handles)} rejected_dynamic='
            f'{list(detection.rejected_dynamic_handles)} target='
            f'{list(detection.target_handles)} reference='
            f'{list(detection.reference_handles)} groups='
            f'{[list(group) for group in detection.object_groups if len(group) > 1]} '
            f'slots={list(detection.grouped_slot_handles)} frames='
            f'{list(detection.sampled_frames)}'
        )
    return handles_by_episode, roles_by_episode, groups_by_episode


def process_task(
    task: str,
    source_dir: Path,
    destination_dir: Optional[Path],
    raw_data_dir: Path,
    cameras: Sequence[str],
    max_objects: int,
    num_points: int,
    excluded_ids: Iterable[int],
    seed: int,
    dry_run: bool,
    dry_run_samples: int,
    visualize_index: Optional[int],
    visualize_every: int,
    visualize_output_dir: Path,
    visualize_objects_only: bool,
    overwrite: bool,
    durable_write: bool,
    show_progress: bool,
    workers: int,
    cache_frames: int,
    min_object_points: int,
    filter_thin_planes: bool,
    thin_plane_max_thickness: float,
    thin_plane_min_extent: float,
    filter_thin_planes_all_roles: bool,
    task_prior_filter: bool,
    task_prior_radius: Optional[float],
    task_prior_max_instances: Optional[int],
    task_prior_background_extent: float,
    task_prior_strict: bool,
    temporal_task_filter: bool,
    task_handle_cache_dir: Path,
    task_detection_frames: int,
    refresh_task_handle_cache: bool,
    auto_detect_robot_handles: bool,
    robot_handle_cache_dir: Path,
    robot_detection_frames: int,
    robot_detection_stride: int,
    robot_detection_window: int,
    robot_motion_threshold: float,
    robot_link_motion_threshold: float,
    robot_adjacency_distance: float,
    refresh_robot_handle_cache: bool,
    refresh_replay_metadata_cache: bool,
    replay_metadata_cache_dir: Optional[Path],
) -> int:
    all_files = _numeric_replay_files(source_dir)
    if not all_files:
        raise FileNotFoundError(f'No *.replay files found in {source_dir}')
    visualization_files = _select_visualization_files(
        all_files,
        visualize_index,
        visualize_every,
    )
    visualization_indices = {
        int(path.stem) for path in visualization_files
    }
    previous_file_by_index = {
        int(current.stem): previous
        for previous, current in zip(all_files, all_files[1:])
    }
    files = all_files
    if dry_run:
        if visualize_every > 0:
            # Interval visualization is itself the dry-run selection. Skipped
            # replay files do not need mask decoding or point-cloud fusion.
            files = visualization_files
        else:
            files = _select_dry_run_files(
                files, seed, dry_run_samples, visualize_index
            )

    cameras = tuple(cameras)
    excluded_ids = tuple(excluded_ids)
    replay_metadata_cache_dir = (
        replay_metadata_cache_dir or destination_dir or source_dir
    )
    robot_handles_by_episode: Dict[int, Tuple[int, ...]] = {}
    if auto_detect_robot_handles:
        requested_robot_episodes = (
            _episode_ids_for_selected_files(files, previous_file_by_index)
            if dry_run
            else None
        )
        robot_handles_by_episode = _detect_task_robot_handles(
            task,
            all_files,
            raw_data_dir,
            cameras,
            excluded_ids,
            robot_detection_frames,
            robot_detection_stride,
            robot_detection_window,
            robot_motion_threshold,
            robot_link_motion_threshold,
            robot_adjacency_distance,
            robot_handle_cache_dir,
            refresh_robot_handle_cache,
            write_cache=not dry_run,
            requested_episode_ids=requested_robot_episodes,
            show_progress=show_progress,
            refresh_metadata_cache=refresh_replay_metadata_cache,
            metadata_cache_dir=replay_metadata_cache_dir,
        )
    task_slot_ids_by_episode: Dict[int, Tuple[int, ...]] = {}
    task_roles_by_episode: Dict[int, Dict[int, int]] = {}
    task_groups_by_episode: Dict[int, Dict[int, int]] = {}
    if temporal_task_filter:
        requested_task_episodes = (
            _episode_ids_for_selected_files(files, previous_file_by_index)
            if dry_run
            else None
        )
        (
            task_slot_ids_by_episode,
            task_roles_by_episode,
            task_groups_by_episode,
        ) = _detect_task_relevant_handles(
            task,
            all_files,
            raw_data_dir,
            cameras,
            excluded_ids,
            robot_handles_by_episode,
            task_detection_frames,
            task_handle_cache_dir,
            refresh_task_handle_cache,
            write_cache=not dry_run,
            task_prior_radius=task_prior_radius,
            task_prior_max_instances=task_prior_max_instances,
            task_prior_background_extent=task_prior_background_extent,
            requested_episode_ids=requested_task_episodes,
            show_progress=show_progress,
            refresh_metadata_cache=refresh_replay_metadata_cache,
            metadata_cache_dir=replay_metadata_cache_dir,
        )
    frame_cache = OracleFrameCache(cache_frames)

    def process_one(source: Path):
        replay_index = int(source.stem)
        try:
            with source.open('rb') as stream:
                original = pickle.load(stream)
            terminal = int(
                np.asarray(original.get('terminal', -1)).item()
            )
            effective_excluded_ids = list(excluded_ids)
            slot_ids = None
            role_by_id = None
            group_by_id = None
            if terminal != -1 and 'episode_idx' in original:
                original_episode_idx = int(
                    np.asarray(original['episode_idx']).item()
                )
                effective_excluded_ids.extend(
                    robot_handles_by_episode.get(original_episode_idx, ())
                )
                if temporal_task_filter:
                    slot_ids = task_slot_ids_by_episode.get(
                        original_episode_idx, ()
                    )
                    role_by_id = task_roles_by_episode.get(
                        original_episode_idx, {}
                    )
                    group_by_id = task_groups_by_episode.get(
                        original_episode_idx, {}
                    )
            migrated, oracle, _ = augment_transition(
                original,
                raw_data_dir,
                task,
                replay_index,
                cameras,
                max_objects,
                num_points,
                effective_excluded_ids,
                seed,
                slot_ids=slot_ids,
                min_object_points=min_object_points,
                frame_cache=frame_cache,
                task_prior_filter=task_prior_filter,
                task_prior_radius=task_prior_radius,
                task_prior_max_instances=task_prior_max_instances,
                task_prior_background_extent=task_prior_background_extent,
                task_prior_strict=task_prior_strict,
                role_by_id=role_by_id,
                group_by_id=group_by_id,
                filter_thin_planes=filter_thin_planes,
                thin_plane_max_thickness=thin_plane_max_thickness,
                thin_plane_min_extent=thin_plane_min_extent,
                filter_thin_planes_all_roles=filter_thin_planes_all_roles,
            )
            if not dry_run:
                assert destination_dir is not None
                atomic_write_replay(
                    destination_dir / source.name,
                    original,
                    migrated,
                    oracle,
                    durable_write=durable_write,
                )
            alignment = {
                key: original[key]
                for key in ('terminal', 'episode_idx', 'sample_frame')
                if key in original
            }
            scene_points = None
            camera_images: Dict[str, np.ndarray] = {}
            camera_masks: Dict[str, np.ndarray] = {}
            visualization_oracle = oracle
            visualization_episode_idx = (
                int(np.asarray(original['episode_idx']).item())
                if 'episode_idx' in original and terminal != -1
                else None
            )
            visualization_sample_frame = (
                int(np.asarray(original['sample_frame']).item())
                if 'sample_frame' in original and terminal != -1
                else None
            )
            if replay_index in visualization_indices:
                if not visualize_objects_only:
                    scene_points = _scene_points_for_visualization(
                        original, cameras
                    )
                if terminal == -1:
                    recovered = _final_observation_oracle_for_visualization(
                        original,
                        previous_file_by_index.get(replay_index),
                        raw_data_dir,
                        task,
                        cameras,
                        max_objects,
                        num_points,
                        excluded_ids,
                        seed,
                        min_object_points,
                        slot_ids_by_episode=(
                            task_slot_ids_by_episode
                            if temporal_task_filter
                            else None
                        ),
                        task_prior_filter=task_prior_filter,
                        task_prior_radius=task_prior_radius,
                        task_prior_max_instances=task_prior_max_instances,
                        task_prior_background_extent=(
                            task_prior_background_extent
                        ),
                        task_prior_strict=task_prior_strict,
                        robot_handles_by_episode=robot_handles_by_episode,
                        roles_by_episode=(
                            task_roles_by_episode
                            if temporal_task_filter
                            else None
                        ),
                        groups_by_episode=(
                            task_groups_by_episode
                            if temporal_task_filter
                            else None
                        ),
                        filter_thin_planes=filter_thin_planes,
                        thin_plane_max_thickness=thin_plane_max_thickness,
                        thin_plane_min_extent=thin_plane_min_extent,
                        filter_thin_planes_all_roles=(
                            filter_thin_planes_all_roles
                        ),
                    )
                    if recovered is not None:
                        (
                            visualization_oracle,
                            visualization_episode_idx,
                            visualization_sample_frame,
                        ) = recovered
                if (
                    visualization_episode_idx is not None
                    and visualization_sample_frame is not None
                ):
                    visualization_episode_dir = resolve_episode_dir(
                        raw_data_dir,
                        task,
                        visualization_episode_idx,
                    )
                    camera_images = load_frame_rgb_images(
                        visualization_episode_dir,
                        visualization_sample_frame,
                        cameras,
                    )
                    camera_masks = load_frame_masks(
                        visualization_episode_dir,
                        visualization_sample_frame,
                        cameras,
                    )
            return (
                replay_index,
                alignment,
                oracle,
                visualization_oracle,
                scene_points,
                camera_images,
                camera_masks,
                visualization_episode_idx,
                visualization_sample_frame,
            )
        except Exception as exc:
            raise RuntimeError(
                f'Failed to process replay {source}'
            ) from exc

    truncated = 0
    filtered = 0
    prior_filtered = 0
    excluded = 0
    temporal_filtered = 0
    thin_planes = 0
    visualized = 0
    progress = tqdm(
        total=len(files),
        desc=f'{task}: Oracle replay',
        unit='replay',
        dynamic_ncols=True,
        disable=not show_progress,
    )
    try:
        for (
            replay_index,
            alignment,
            oracle,
            visualization_oracle,
            scene_points,
            camera_images,
            camera_masks,
            visualization_episode_idx,
            visualization_sample_frame,
        ) in _bounded_thread_map(process_one, files, workers):
            truncated += int(oracle.discovered_objects > max_objects)
            filtered += oracle.filtered_objects
            prior_filtered += oracle.prior_filtered_objects
            excluded += oracle.excluded_objects
            temporal_filtered += oracle.temporal_filtered_objects
            thin_planes += oracle.thin_plane_objects
            cache_hits, cache_misses, _ = frame_cache.stats()
            progress.set_postfix(
                objects=int(oracle.valid.sum()),
                truncated=truncated,
                filtered=filtered,
                prior_filtered=prior_filtered,
                excluded=excluded,
                temporal_filtered=temporal_filtered,
                thin_planes=thin_planes,
                workers=workers,
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                refresh=False,
            )
            progress.update(1)
            if dry_run:
                _describe(task, replay_index, alignment, oracle)
            if replay_index in visualization_indices:
                visualize_oracle_objects(
                    visualization_oracle,
                    task,
                    replay_index,
                    visualize_output_dir,
                    scene_points=scene_points,
                    terminal=int(
                        np.asarray(alignment.get('terminal', -1)).item()
                    ),
                    episode_idx=visualization_episode_idx,
                    sample_frame=visualization_sample_frame,
                    camera_images=camera_images,
                    camera_masks=camera_masks,
                    group_by_id=(
                        task_groups_by_episode.get(
                            visualization_episode_idx, {}
                        )
                        if (
                            temporal_task_filter
                            and visualization_episode_idx is not None
                        )
                        else None
                    ),
                )
                visualized += 1
    finally:
        progress.close()

    if not dry_run:
        assert destination_dir is not None
        _copy_metadata(source_dir, destination_dir, overwrite)
    mode = 'validated' if dry_run else 'migrated'
    cache_hits, cache_misses, cache_entries = frame_cache.stats()
    print(
        f'{task}: {mode} {len(files)} replay files; truncated={truncated}; '
        f'filtered={filtered}; '
        f'prior_filtered={prior_filtered}; '
        f'excluded={excluded}; '
        f'temporal_filtered={temporal_filtered}; '
        f'thin_planes={thin_planes}; '
        f'visualized={visualized}; '
        f'cache_hits={cache_hits}; cache_misses={cache_misses}; '
        f'cache_entries={cache_entries}'
    )
    return len(files)


def _preflight_output(
    task_directories: Sequence[Tuple[str, Path]],
    replay_dir: Path,
    output_dir: Path,
    direct_input: bool,
    overwrite: bool,
) -> None:
    if output_dir == replay_dir:
        raise ValueError(
            '--output-dir must differ from --replay-dir; use --in-place'
        )
    if overwrite:
        return
    conflicts: List[Path] = []
    for task, source_dir in task_directories:
        destination_dir = output_dir if direct_input else output_dir / task
        for source in source_dir.iterdir():
            if source.name == REPLAY_METADATA_CACHE_NAME:
                continue
            if source.is_file() and (destination_dir / source.name).exists():
                conflicts.append(destination_dir / source.name)
    if conflicts:
        raise FileExistsError(
            f'Output already contains {conflicts[0]}; use --overwrite'
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-dir', type=Path, required=True)
    parser.add_argument('--raw-data-dir', type=Path, required=True)
    parser.add_argument(
        '--task',
        action='append',
        default=[],
        help='Task name, comma-separated names, or all for discovery',
    )
    output = parser.add_mutually_exclusive_group()
    output.add_argument('--output-dir', type=Path)
    output.add_argument('--in-place', action='store_true')
    parser.add_argument(
        '--max-objects', type=int, default=DEFAULT_MAX_OBJECTS
    )
    parser.add_argument('--num-points', type=int, default=DEFAULT_NUM_POINTS)
    parser.add_argument(
        '--min-object-points',
        type=int,
        default=20,
        help='discard decoded instances with fewer fused finite points',
    )
    parser.add_argument('--camera', action='append', dest='cameras')
    parser.add_argument(
        '--exclude-object-id',
        '--exclude-robot-id',
        action='append',
        type=int,
        default=[0],
        help=(
            'decoded handle to exclude; --exclude-robot-id is an alias; '
            'repeat as needed'
        ),
    )
    parser.add_argument(
        '--task-prior-filter',
        action='store_true',
        help=(
            'rank GT handles by the task-specific next-action spatial prior '
            'and remove obvious planar background'
        ),
    )
    parser.add_argument(
        '--task-prior-strict',
        action='store_true',
        help=(
            'also discard handles outside the task interaction radius; '
            'lower recall and intended only for explicit strict filtering'
        ),
    )
    parser.add_argument(
        '--task-prior-radius',
        type=float,
        help='override the configured task interaction radius in metres',
    )
    parser.add_argument(
        '--task-prior-max-instances',
        type=int,
        help='override the configured maximum retained simulator handles',
    )
    parser.add_argument(
        '--task-prior-background-extent',
        type=float,
        default=0.60,
        help=(
            'reject obvious planar background spanning at least this many '
            'metres on two axes (default: 0.60)'
        ),
    )
    parser.add_argument(
        '--filter-thin-planes',
        action='store_true',
        help=(
            'discard unknown-role instances whose shortest extent is very '
            'thin while both planar extents are large'
        ),
    )
    parser.add_argument(
        '--thin-plane-max-thickness',
        type=float,
        default=0.005,
        help='maximum thin-plane thickness in metres (default: 0.005)',
    )
    parser.add_argument(
        '--thin-plane-min-extent',
        type=float,
        default=0.08,
        help='minimum size of both planar axes in metres (default: 0.08)',
    )
    parser.add_argument(
        '--filter-thin-planes-all-roles',
        action='store_true',
        help=(
            'also discard geometrically thin target/reference instances; '
            'use only after visually verifying role false positives'
        ),
    )
    parser.add_argument(
        '--temporal-task-filter',
        '--temporal-id-matching',
        dest='temporal_task_filter',
        action='store_true',
        help=(
            'assign episode-local stable object slots from temporal handle '
            'evidence without filtering visible objects'
        ),
    )
    parser.add_argument(
        '--task-handle-cache-dir',
        type=Path,
        default=None,
        help=(
            'root for per-task stable-slot and task-handle JSON caches; '
            'defaults to <task-output-dir>/task_handle_maps'
        ),
    )
    parser.add_argument(
        '--task-detection-frames',
        type=int,
        default=16,
        help='maximum temporally spread replay frames inspected per episode',
    )
    parser.add_argument(
        '--refresh-task-handle-cache',
        action='store_true',
        help='ignore existing task-handle JSON files and detect again',
    )
    parser.add_argument(
        '--detect-robot-handles',
        action='store_true',
        help=(
            'detect gripper/arm handles once per episode from raw current '
            'gripper poses and temporal GT masks, then exclude them'
        ),
    )
    parser.add_argument(
        '--robot-handle-cache-dir',
        type=Path,
        default=None,
        help=(
            'root for per-task robot-handle JSON caches; defaults to '
            '<task-output-dir>/robot_handle_maps'
        ),
    )
    parser.add_argument(
        '--robot-detection-frames',
        type=int,
        default=64,
        help=(
            'maximum raw frames inspected per episode after adaptive extension '
            '(default: 64)'
        ),
    )
    parser.add_argument(
        '--robot-detection-stride',
        type=int,
        default=5,
        help='raw-frame interval for robot evidence (default: 5)',
    )
    parser.add_argument(
        '--robot-detection-window',
        type=int,
        default=100,
        help=(
            'initial inclusive raw-frame window starting at frame 0; extends '
            'when gripper motion is insufficient (default: 100)'
        ),
    )
    parser.add_argument(
        '--robot-motion-threshold',
        type=float,
        default=0.02,
        help=(
            'minimum gripper displacement in metres before adaptive sampling '
            'can stop (default: 0.02)'
        ),
    )
    parser.add_argument(
        '--robot-link-motion-threshold',
        type=float,
        default=0.008,
        help=(
            'minimum arm-link motion in metres for the first kinematic-chain '
            'hop (default: 0.008)'
        ),
    )
    parser.add_argument(
        '--robot-adjacency-distance',
        type=float,
        default=0.05,
        help=(
            'maximum AABB gap in metres between connected robot handles '
            '(default: 0.05)'
        ),
    )
    parser.add_argument(
        '--refresh-robot-handle-cache',
        action='store_true',
        help='ignore existing robot-handle JSON files and detect again',
    )
    parser.add_argument(
        '--refresh-replay-metadata-cache',
        action='store_true',
        help=(
            'ignore the compact replay metadata index and deserialize all '
            'replay files again'
        ),
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--workers',
        type=int,
        default=1,
        help='number of replay files to process concurrently with threads',
    )
    parser.add_argument(
        '--cache-frames',
        type=int,
        default=128,
        help='number of completed raw-frame Oracle results kept in LRU cache; 0 disables',
    )
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--dry-run-samples', type=int, default=5)
    visualization = parser.add_mutually_exclusive_group()
    visualization.add_argument('--visualize-index', type=int)
    visualization.add_argument(
        '--visualize-every',
        type=int,
        default=0,
        metavar='N',
        help='save one visualization for every Nth sorted replay file',
    )
    parser.add_argument(
        '--visualize-output-dir',
        type=Path,
        default=Path('oracle_visualizations'),
        help=(
            'directory for visualization PNG output '
            '(default: ./oracle_visualizations)'
        ),
    )
    parser.add_argument(
        '--visualize-objects-only',
        action='store_true',
        help=(
            'draw only retained Oracle instances without the gray full-scene '
            'point cloud'
        ),
    )
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--durable-write',
        action='store_true',
        help='fsync every temporary replay before rename (safer but slower)',
    )
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='disable the per-task tqdm progress bar',
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    replay_dir = args.replay_dir.resolve()
    raw_data_dir = args.raw_data_dir.resolve()
    visualize_output_dir = args.visualize_output_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else None
    if not replay_dir.is_dir():
        raise FileNotFoundError(f'Replay directory does not exist: {replay_dir}')
    if not raw_data_dir.is_dir():
        raise FileNotFoundError(f'Raw directory does not exist: {raw_data_dir}')
    if (
        args.max_objects <= 0
        or args.num_points <= 0
        or args.min_object_points <= 0
    ):
        raise ValueError(
            '--max-objects, --num-points, and --min-object-points '
            'must be positive'
        )
    if args.dry_run_samples <= 0:
        raise ValueError('--dry-run-samples must be positive')
    if args.visualize_every < 0:
        raise ValueError('--visualize-every must be non-negative')
    if args.workers <= 0:
        raise ValueError('--workers must be positive')
    if args.cache_frames < 0:
        raise ValueError('--cache-frames must be non-negative')
    if args.robot_detection_frames <= 0:
        raise ValueError('--robot-detection-frames must be positive')
    if args.robot_detection_stride <= 0:
        raise ValueError('--robot-detection-stride must be positive')
    if args.robot_detection_window < 0:
        raise ValueError('--robot-detection-window must be non-negative')
    if args.robot_motion_threshold < 0:
        raise ValueError('--robot-motion-threshold must be non-negative')
    if args.robot_link_motion_threshold <= 0:
        raise ValueError('--robot-link-motion-threshold must be positive')
    if args.robot_adjacency_distance <= 0:
        raise ValueError('--robot-adjacency-distance must be positive')
    if args.task_detection_frames <= 0:
        raise ValueError('--task-detection-frames must be positive')
    if args.task_prior_radius is not None and args.task_prior_radius <= 0:
        raise ValueError('--task-prior-radius must be positive')
    if (
        args.task_prior_max_instances is not None
        and args.task_prior_max_instances <= 0
    ):
        raise ValueError('--task-prior-max-instances must be positive')
    if args.task_prior_background_extent <= 0:
        raise ValueError('--task-prior-background-extent must be positive')
    if args.thin_plane_max_thickness <= 0:
        raise ValueError('--thin-plane-max-thickness must be positive')
    if args.thin_plane_min_extent <= 0:
        raise ValueError('--thin-plane-min-extent must be positive')
    if not args.dry_run and not args.in_place and args.output_dir is None:
        raise ValueError('Choose --output-dir or explicit --in-place')

    task_directories = discover_task_directories(
        replay_dir, args.task or ['all']
    )
    direct_input = bool(_numeric_replay_files(replay_dir))
    if output_dir is not None and not args.dry_run:
        _preflight_output(
            task_directories,
            replay_dir,
            output_dir,
            direct_input,
            args.overwrite,
        )

    total = 0
    for task, source_dir in task_directories:
        if args.dry_run:
            destination_dir = None
        elif args.in_place:
            destination_dir = source_dir
        elif direct_input:
            destination_dir = output_dir
        else:
            destination_dir = output_dir / task
        replay_metadata_cache_dir = source_dir
        if output_dir is not None:
            replay_metadata_cache_dir = (
                output_dir if direct_input else output_dir / task
            )
        robot_handle_cache_dir = _resolve_task_cache_directory(
            args.robot_handle_cache_dir,
            replay_metadata_cache_dir,
            task,
            'robot_handle_maps',
        )
        task_handle_cache_dir = _resolve_task_cache_directory(
            args.task_handle_cache_dir,
            replay_metadata_cache_dir,
            task,
            'task_handle_maps',
        )
        total += process_task(
            task=task,
            source_dir=source_dir,
            destination_dir=destination_dir,
            raw_data_dir=raw_data_dir,
            cameras=tuple(args.cameras or DEFAULT_CAMERAS),
            max_objects=args.max_objects,
            num_points=args.num_points,
            excluded_ids=args.exclude_object_id,
            seed=args.seed,
            dry_run=args.dry_run,
            dry_run_samples=args.dry_run_samples,
            visualize_index=args.visualize_index,
            visualize_every=args.visualize_every,
            visualize_output_dir=visualize_output_dir,
            visualize_objects_only=args.visualize_objects_only,
            overwrite=args.overwrite,
            durable_write=args.durable_write,
            show_progress=not args.no_progress,
            workers=args.workers,
            cache_frames=args.cache_frames,
            min_object_points=args.min_object_points,
            filter_thin_planes=args.filter_thin_planes,
            thin_plane_max_thickness=args.thin_plane_max_thickness,
            thin_plane_min_extent=args.thin_plane_min_extent,
            filter_thin_planes_all_roles=args.filter_thin_planes_all_roles,
            task_prior_filter=args.task_prior_filter,
            task_prior_radius=args.task_prior_radius,
            task_prior_max_instances=args.task_prior_max_instances,
            task_prior_background_extent=(
                args.task_prior_background_extent
            ),
            task_prior_strict=args.task_prior_strict,
            temporal_task_filter=args.temporal_task_filter,
            task_handle_cache_dir=task_handle_cache_dir,
            task_detection_frames=args.task_detection_frames,
            refresh_task_handle_cache=args.refresh_task_handle_cache,
            auto_detect_robot_handles=args.detect_robot_handles,
            robot_handle_cache_dir=robot_handle_cache_dir,
            robot_detection_frames=args.robot_detection_frames,
            robot_detection_stride=args.robot_detection_stride,
            robot_detection_window=args.robot_detection_window,
            robot_motion_threshold=args.robot_motion_threshold,
            robot_link_motion_threshold=args.robot_link_motion_threshold,
            robot_adjacency_distance=args.robot_adjacency_distance,
            refresh_robot_handle_cache=args.refresh_robot_handle_cache,
            refresh_replay_metadata_cache=(
                args.refresh_replay_metadata_cache
            ),
            replay_metadata_cache_dir=replay_metadata_cache_dir,
        )
    print(f'Done: {total} replay files')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
