#!/usr/bin/env python3
'''Append Oracle RLBench instance point clouds to BridgeVLA replay files.

The script migrates existing per-transition YARR *.replay pickle files. It
does not call ReplayBuffer.add() and does not regenerate BridgeVLA features.
'''

from __future__ import annotations

import argparse
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


# Oracle object experiment
DEFAULT_CAMERAS = ('front', 'left_shoulder', 'right_shoulder', 'wrist')
DEFAULT_MAX_OBJECTS = 32
DEFAULT_NUM_POINTS = 512
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
        self._entries: OrderedDict[Tuple[str, int, int], Future] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get_or_compute(
        self,
        key: Tuple[str, int, int],
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

    def _evict_locked(self, protected_key: Tuple[str, int, int]) -> None:
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
    )


def _point_cloud_hwc(point_cloud: np.ndarray, name: str) -> np.ndarray:
    point_cloud = np.asarray(point_cloud)
    if point_cloud.ndim != 3:
        raise ValueError(f'{name} must have 3 dimensions; got {point_cloud.shape}')
    if point_cloud.shape[-1] == 3:
        return point_cloud
    if point_cloud.shape[0] == 3:
        return np.moveaxis(point_cloud, 0, -1)
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
) -> OracleObjects:
    '''Fuse decoded instance masks with the existing replay point clouds.'''
    if max_objects <= 0 or num_points <= 0 or min_object_points <= 0:
        raise ValueError(
            'max_objects, num_points, and min_object_points must be positive'
        )
    rng = rng or np.random.default_rng()
    excluded = {int(value) for value in excluded_ids}
    points_by_id: Dict[int, List[np.ndarray]] = {}

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
    filtered_objects = sum(
        len(object_points) < min_object_points
        for _, object_points in merged
    )
    merged = [
        (object_id, object_points)
        for object_id, object_points in merged
        if len(object_points) >= min_object_points
    ]
    # Fixed storage requires truncation. Prefer the strongest point support,
    # then use handle ID as a deterministic tie breaker.
    merged.sort(key=lambda item: (-len(item[1]), item[0]))
    discovered_objects = len(merged)
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
            )

        cache_key = (task, episode_idx, sample_frame)
        oracle = (
            frame_cache.get_or_compute(cache_key, build_oracle)
            if frame_cache is not None
            else build_oracle()
        )

    migrated = dict(original)
    migrated.update(oracle.as_replay_fields())
    validate_migrated_transition(original, migrated, oracle)
    return migrated, oracle, episode_dir


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
    print(
        f'  shapes=points{oracle.points.shape}, '
        f'centers{oracle.centers.shape}, sizes{oracle.sizes.shape}, '
        f'valid{oracle.valid.shape} '
        f'finite_points={bool(np.isfinite(oracle.points).all())}'
    )


def visualize_oracle_objects(
    oracle: OracleObjects,
    task: str,
    replay_index: int,
    output_dir: Path,
) -> Path:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('--visualize-index requires matplotlib') from exc
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{task}_replay_{replay_index}.png'
    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    for slot in np.flatnonzero(oracle.valid):
        points = oracle.points[slot]
        axes.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=2,
            label=str(oracle.ids[slot]),
        )
    axes.set_title(f'{task} replay {replay_index}: Oracle GT instances')
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')
    if oracle.valid.any():
        axes.legend(title='instance ID')
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
    overwrite: bool,
    durable_write: bool,
    show_progress: bool,
    workers: int,
    cache_frames: int,
    min_object_points: int,
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
    frame_cache = OracleFrameCache(cache_frames)

    def process_one(source: Path):
        replay_index = int(source.stem)
        try:
            with source.open('rb') as stream:
                original = pickle.load(stream)
            migrated, oracle, _ = augment_transition(
                original,
                raw_data_dir,
                task,
                replay_index,
                cameras,
                max_objects,
                num_points,
                excluded_ids,
                seed,
                min_object_points=min_object_points,
                frame_cache=frame_cache,
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
            return replay_index, alignment, oracle
        except Exception as exc:
            raise RuntimeError(
                f'Failed to process replay {source}'
            ) from exc

    truncated = 0
    filtered = 0
    visualized = 0
    progress = tqdm(
        total=len(files),
        desc=f'{task}: Oracle replay',
        unit='replay',
        dynamic_ncols=True,
        disable=not show_progress,
    )
    try:
        for replay_index, alignment, oracle in _bounded_thread_map(
            process_one, files, workers
        ):
            truncated += int(oracle.discovered_objects > max_objects)
            filtered += oracle.filtered_objects
            cache_hits, cache_misses, _ = frame_cache.stats()
            progress.set_postfix(
                objects=int(oracle.valid.sum()),
                truncated=truncated,
                filtered=filtered,
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
                    oracle,
                    task,
                    replay_index,
                    visualize_output_dir,
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
        action='append',
        type=int,
        default=[0],
        help='Decoded handle to exclude; repeat as needed',
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
            overwrite=args.overwrite,
            durable_write=args.durable_write,
            show_progress=not args.no_progress,
            workers=args.workers,
            cache_frames=args.cache_frames,
            min_object_points=args.min_object_points,
        )
    print(f'Done: {total} replay files')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
