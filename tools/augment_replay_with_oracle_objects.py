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

import numpy as np
from PIL import Image
from tqdm import tqdm

try:
    from tools.rlbench_task_object_priors import select_task_relevant_instances
except ModuleNotFoundError:  # Direct execution: python tools/<script>.py
    from rlbench_task_object_priors import select_task_relevant_instances
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
)


@dataclass(frozen=True)
class OracleObjects:
    points: np.ndarray
    centers: np.ndarray
    sizes: np.ndarray
    ids: np.ndarray
    valid: np.ndarray
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

    def as_replay_fields(self) -> Dict[str, np.ndarray]:
        return {
            'oracle_object_points': self.points,
            'oracle_object_centers': self.centers,
            'oracle_object_sizes': self.sizes,
            'oracle_object_ids': self.ids,
            'oracle_object_valid': self.valid,
        }


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


def extract_oracle_objects(
    transition: Mapping[str, object],
    masks: Mapping[str, np.ndarray],
    cameras: Sequence[str] = DEFAULT_CAMERAS,
    max_objects: int = DEFAULT_MAX_OBJECTS,
    num_points: int = DEFAULT_NUM_POINTS,
    excluded_ids: Iterable[int] = (0,),
    min_object_points: int = 20,
    rng: Optional[np.random.Generator] = None,
    task_name: Optional[str] = None,
    task_prior_filter: bool = False,
    action_position: Optional[np.ndarray] = None,
    task_prior_radius: Optional[float] = None,
    task_prior_max_instances: Optional[int] = None,
    task_prior_background_extent: float = 0.60,
    task_prior_strict: bool = False,
) -> OracleObjects:
    '''Fuse decoded instance masks with the existing replay point clouds.'''
    if max_objects <= 0 or num_points <= 0 or min_object_points <= 0:
        raise ValueError(
            'max_objects, num_points, and min_object_points must be positive'
        )
    rng = rng or np.random.default_rng()
    excluded = {int(value) for value in excluded_ids}
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
            object_id = int(object_id_value)
            if object_id >= 0:
                observed_ids.add(object_id)
            if object_id in excluded or object_id < 0:
                continue
            object_points = point_cloud[mask == object_id]
            object_points = object_points[np.isfinite(object_points).all(axis=1)]
            if object_points.size:
                points_by_id.setdefault(object_id, []).append(object_points)

    merged = [
        (object_id, np.concatenate(camera_points, axis=0))
        for object_id, camera_points in points_by_id.items()
    ]
    no_finite_point_object_ids = tuple(sorted(
        observed_ids - excluded - set(points_by_id)
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
    # Fixed storage requires truncation. Prefer the strongest point support,
    # then use handle ID as a deterministic tie breaker. Task-prior results
    # are already ordered by action proximity and must retain that ordering.
    if not task_prior_filter:
        merged.sort(key=lambda item: (-len(item[1]), item[0]))
    discovered_objects = len(merged)
    truncated_object_ids = tuple(
        object_id for object_id, _ in merged[max_objects:]
    )
    merged = merged[:max_objects]

    padded = empty_oracle_objects(max_objects, num_points)
    raw_counts: List[int] = []
    for slot, (object_id, object_points) in enumerate(merged):
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
        raw_counts.append(count)

    oracle = OracleObjects(
        padded.points,
        padded.centers,
        padded.sizes,
        padded.ids,
        padded.valid,
        tuple(raw_counts),
        discovered_objects,
        filtered_objects,
        prior_filtered_objects,
        len(observed_ids.intersection(excluded)),
        tuple(sorted(observed_ids.intersection(excluded))),
        no_finite_point_object_ids,
        small_object_ids,
        prior_filtered_object_ids,
        truncated_object_ids,
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
    min_object_points: int = 20,
    frame_cache: Optional[OracleFrameCache] = None,
    task_prior_filter: bool = False,
    task_prior_radius: Optional[float] = None,
    task_prior_max_instances: Optional[int] = None,
    task_prior_background_extent: float = 0.60,
    task_prior_strict: bool = False,
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
            )

        cache_key: Tuple[object, ...] = (task, episode_idx, sample_frame)
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
    task_prior_filter: bool = False,
    task_prior_radius: Optional[float] = None,
    task_prior_max_instances: Optional[int] = None,
    task_prior_background_extent: float = 0.60,
    task_prior_strict: bool = False,
    robot_handles_by_episode: Optional[Mapping[int, Sequence[int]]] = None,
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
    print(f'  raw_points_per_object={list(oracle.raw_point_counts)}')
    print(f'  sampled_points_per_object={[oracle.points.shape[1]] * count}')
    print(f'  centers={oracle.centers[oracle.valid].tolist()}')
    print(f'  sizes={oracle.sizes[oracle.valid].tolist()}')
    print(f'  filtered_small_objects={oracle.filtered_objects}')
    print(f'  filtered_by_task_prior={oracle.prior_filtered_objects}')
    print(f'  excluded_by_id={oracle.excluded_objects}')
    print(f'  excluded_object_ids={list(oracle.excluded_object_ids)}')
    print(
        '  no_finite_point_object_ids='
        f'{list(oracle.no_finite_point_object_ids)}'
    )
    print(f'  small_object_ids={list(oracle.small_object_ids)}')
    print(
        '  task_prior_filtered_object_ids='
        f'{list(oracle.prior_filtered_object_ids)}'
    )
    print(f'  truncated_object_ids={list(oracle.truncated_object_ids)}')
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


def visualize_oracle_objects(
    oracle: OracleObjects,
    task: str,
    replay_index: int,
    output_dir: Path,
    scene_points: Optional[np.ndarray] = None,
    terminal: Optional[int] = None,
    episode_idx: Optional[int] = None,
    sample_frame: Optional[int] = None,
) -> Path:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('--visualize-index requires matplotlib') from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{task}_replay_{replay_index}.png'
    figure = plt.figure(figsize=(12, 10))
    axes = figure.add_subplot(221, projection='3d')
    top_axes = figure.add_subplot(222)
    front_axes = figure.add_subplot(223)
    side_axes = figure.add_subplot(224)
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
        color = _instance_color(object_id)
        axes.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            c=[color],
            s=2,
            label=str(object_id),
        )
        object_style = {'c': [color], 's': 2}
        top_axes.scatter(points[:, 0], points[:, 1], **object_style)
        front_axes.scatter(points[:, 0], points[:, 2], **object_style)
        side_axes.scatter(points[:, 1], points[:, 2], **object_style)
    sentinel = terminal == -1
    alignment = ''
    if episode_idx is not None and sample_frame is not None:
        alignment = f' ep={episode_idx} frame={sample_frame}'
    figure.suptitle(
        f'{task} replay {replay_index}{alignment}: Oracle GT instances '
        f'(valid={int(oracle.valid.sum())}'
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
        axes.legend(title='instance ID')
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
    figure.tight_layout(rect=(0, 0, 1, 0.96))
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


def _episode_detection_sources(
    files: Sequence[Path],
    frames_per_episode: int,
    requested_episode_ids: Optional[Iterable[int]] = None,
    cached_episode_ids: Iterable[int] = (),
    show_progress: bool = True,
) -> Dict[int, List[Tuple[int, Path]]]:
    '''Select deterministic, temporally spread replay frames per episode.'''
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

            selected: Dict[int, List[Tuple[int, Path]]] = {}
            progress = tqdm(
                segments,
                desc='robot detection: episode index',
                unit='episode',
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
                    selected[episode_idx] = []
                    continue
                if len(ordinary) > frames_per_episode:
                    positions = np.linspace(
                        0,
                        len(ordinary) - 1,
                        num=frames_per_episode,
                        dtype=np.int64,
                    )
                    ordinary = [ordinary[int(position)] for position in positions]
                frame_sources: Dict[int, Path] = {}
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
                    frame_sources.setdefault(sample_frame, source)
                selected[episode_idx] = sorted(frame_sources.items())
            return selected

    # Compatibility fallback for old replay folders without replay_info.npy.
    # This is slower because every large replay pickle must be opened.
    grouped: Dict[int, Dict[int, Path]] = {}
    progress = tqdm(
        files,
        desc='robot detection: replay metadata scan',
        unit='replay',
        dynamic_ncols=True,
        disable=not show_progress,
    )
    for source in progress:
        with source.open('rb') as stream:
            transition = pickle.load(stream)
        if int(np.asarray(transition.get('terminal', -1)).item()) == -1:
            continue
        if 'episode_idx' not in transition or 'sample_frame' not in transition:
            continue
        episode_idx = int(np.asarray(transition['episode_idx']).item())
        if requested is not None and episode_idx not in requested:
            continue
        if episode_idx in cached:
            grouped.setdefault(episode_idx, {})
            continue
        sample_frame = int(np.asarray(transition['sample_frame']).item())
        grouped.setdefault(episode_idx, {}).setdefault(sample_frame, source)

    selected: Dict[int, List[Tuple[int, Path]]] = {}
    for episode_idx, by_frame in grouped.items():
        candidates = sorted(by_frame.items())
        if len(candidates) > frames_per_episode:
            indices = np.linspace(
                0,
                len(candidates) - 1,
                num=frames_per_episode,
                dtype=np.int64,
            )
            candidates = [candidates[int(index)] for index in indices]
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


def _load_current_gripper_positions(
    episode_dir: Path, sample_frames: Sequence[int]
) -> Dict[int, np.ndarray]:
    low_dim_path = episode_dir / 'low_dim_obs.pkl'
    if not low_dim_path.is_file():
        raise FileNotFoundError(
            f'Robot handle detection requires {low_dim_path}'
        )
    with low_dim_path.open('rb') as stream:
        observations = pickle.load(stream)
    positions: Dict[int, np.ndarray] = {}
    for sample_frame in sample_frames:
        try:
            observation = observations[sample_frame]
            pose = np.asarray(observation.gripper_pose).reshape(-1)
        except (IndexError, TypeError, AttributeError) as exc:
            raise ValueError(
                f'Cannot read current gripper_pose for frame {sample_frame} '
                f'from {low_dim_path}'
            ) from exc
        if pose.size < 3 or not np.isfinite(pose[:3]).all():
            raise ValueError(
                f'Invalid current gripper_pose at frame {sample_frame}: {pose}'
            )
        positions[sample_frame] = pose[:3].astype(np.float32)
    return positions


def _detect_task_robot_handles(
    task: str,
    files: Sequence[Path],
    raw_data_dir: Path,
    cameras: Sequence[str],
    excluded_ids: Iterable[int],
    frames_per_episode: int,
    cache_dir: Path,
    refresh_cache: bool,
    write_cache: bool,
    requested_episode_ids: Optional[Iterable[int]] = None,
    show_progress: bool = True,
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
    cached_detections: Dict[int, RobotHandleDetection] = {}
    stale_cache_count = 0
    if not refresh_cache:
        for path in (cache_dir / task).glob('episode_*.json'):
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
                detection = load_robot_handle_detection(path)
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
        frames_per_episode,
        requested_episode_ids=requested_episode_set,
        cached_episode_ids=cached_detections,
        show_progress=show_progress,
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
        cache_path = cache_dir / task / f'episode_{episode_idx:04d}.json'
        detection = cached_detections.get(episode_idx)
        if detection is None:
            if not frame_sources:
                raise RuntimeError(
                    f'Robot cache missing frame sources for episode {episode_idx}'
                )
            episode_dir = resolve_episode_dir(raw_data_dir, task, episode_idx)
            sample_frames = [frame for frame, _ in frame_sources]
            gripper_positions = _load_current_gripper_positions(
                episode_dir, sample_frames
            )
            evidence = []
            for sample_frame, source in frame_sources:
                with source.open('rb') as stream:
                    transition = pickle.load(stream)
                masks = load_frame_masks(episode_dir, sample_frame, cameras)
                point_clouds = {
                    camera: _point_cloud_hwc(
                        np.asarray(transition[f'{camera}_point_cloud']),
                        f'{camera}_point_cloud',
                    )
                    for camera in cameras
                }
                evidence.append(
                    build_robot_frame_evidence(
                        sample_frame,
                        gripper_positions[sample_frame],
                        masks,
                        point_clouds,
                        excluded_ids=excluded_ids,
                    )
                )
            detection = detect_robot_handles(episode_idx, evidence)
            if write_cache:
                save_robot_handle_detection(cache_path, detection)
        handles_by_episode[episode_idx] = detection.robot_handles
        tqdm.write(
            f'{task} episode={episode_idx}: detected robot handles='
            f'{list(detection.robot_handles)} gripper='
            f'{list(detection.gripper_handles)} frames='
            f'{list(detection.sampled_frames)}',
        )
    return handles_by_episode


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
    task_prior_filter: bool,
    task_prior_radius: Optional[float],
    task_prior_max_instances: Optional[int],
    task_prior_background_extent: float,
    task_prior_strict: bool,
    auto_detect_robot_handles: bool,
    robot_handle_cache_dir: Path,
    robot_detection_frames: int,
    refresh_robot_handle_cache: bool,
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
            robot_handle_cache_dir,
            refresh_robot_handle_cache,
            write_cache=not dry_run,
            requested_episode_ids=requested_robot_episodes,
            show_progress=show_progress,
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
            if terminal != -1 and 'episode_idx' in original:
                original_episode_idx = int(
                    np.asarray(original['episode_idx']).item()
                )
                effective_excluded_ids.extend(
                    robot_handles_by_episode.get(original_episode_idx, ())
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
                min_object_points=min_object_points,
                frame_cache=frame_cache,
                task_prior_filter=task_prior_filter,
                task_prior_radius=task_prior_radius,
                task_prior_max_instances=task_prior_max_instances,
                task_prior_background_extent=task_prior_background_extent,
                task_prior_strict=task_prior_strict,
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
                        task_prior_filter=task_prior_filter,
                        task_prior_radius=task_prior_radius,
                        task_prior_max_instances=task_prior_max_instances,
                        task_prior_background_extent=(
                            task_prior_background_extent
                        ),
                        task_prior_strict=task_prior_strict,
                        robot_handles_by_episode=robot_handles_by_episode,
                    )
                    if recovered is not None:
                        (
                            visualization_oracle,
                            visualization_episode_idx,
                            visualization_sample_frame,
                        ) = recovered
            return (
                replay_index,
                alignment,
                oracle,
                visualization_oracle,
                scene_points,
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
            visualization_episode_idx,
            visualization_sample_frame,
        ) in _bounded_thread_map(process_one, files, workers):
            truncated += int(oracle.discovered_objects > max_objects)
            filtered += oracle.filtered_objects
            prior_filtered += oracle.prior_filtered_objects
            excluded += oracle.excluded_objects
            cache_hits, cache_misses, _ = frame_cache.stats()
            progress.set_postfix(
                objects=int(oracle.valid.sum()),
                truncated=truncated,
                filtered=filtered,
                prior_filtered=prior_filtered,
                excluded=excluded,
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
        default=Path('robot_handle_maps'),
        help='episode robot-handle JSON cache directory',
    )
    parser.add_argument(
        '--robot-detection-frames',
        type=int,
        default=8,
        help='maximum temporally spread replay frames inspected per episode',
    )
    parser.add_argument(
        '--refresh-robot-handle-cache',
        action='store_true',
        help='ignore existing robot-handle JSON files and detect again',
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
    robot_handle_cache_dir = args.robot_handle_cache_dir.resolve()
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
    if args.task_prior_radius is not None and args.task_prior_radius <= 0:
        raise ValueError('--task-prior-radius must be positive')
    if (
        args.task_prior_max_instances is not None
        and args.task_prior_max_instances <= 0
    ):
        raise ValueError('--task-prior-max-instances must be positive')
    if args.task_prior_background_extent <= 0:
        raise ValueError('--task-prior-background-extent must be positive')
    if not args.dry_run and not args.in_place and args.output_dir is None:
        raise ValueError('Choose --output-dir or explicit --in-place')

    task_directories = discover_task_directories(
        replay_dir, args.task or ['all']
    )
    direct_input = bool(_numeric_replay_files(replay_dir))
    output_dir = args.output_dir.resolve() if args.output_dir else None
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
            task_prior_filter=args.task_prior_filter,
            task_prior_radius=args.task_prior_radius,
            task_prior_max_instances=args.task_prior_max_instances,
            task_prior_background_extent=(
                args.task_prior_background_extent
            ),
            task_prior_strict=args.task_prior_strict,
            auto_detect_robot_handles=args.detect_robot_handles,
            robot_handle_cache_dir=robot_handle_cache_dir,
            robot_detection_frames=args.robot_detection_frames,
            refresh_robot_handle_cache=args.refresh_robot_handle_cache,
        )
    print(f'Done: {total} replay files')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
