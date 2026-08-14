'''Episode-level temporal selection of task-relevant RLBench handles.'''

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from tools.rlbench_task_object_priors import get_task_object_prior
except ModuleNotFoundError:  # Direct execution: python tools/<script>.py
    from rlbench_task_object_priors import get_task_object_prior


Bounds = Tuple[np.ndarray, np.ndarray]
TASK_HANDLE_DETECTOR_METHOD = 'episode_action_trajectory_v2'


@dataclass(frozen=True)
class TaskFrameEvidence:
    sample_frame: int
    gripper_position: np.ndarray
    action_position: np.ndarray
    bounds_by_id: Mapping[int, Bounds]
    centers_by_id: Mapping[int, np.ndarray]
    point_counts_by_id: Mapping[int, int]


@dataclass(frozen=True)
class TaskHandleDetection:
    episode_idx: int
    task_handles: Tuple[int, ...]
    interaction_handles: Tuple[int, ...]
    adjacent_handles: Tuple[int, ...]
    rejected_dynamic_handles: Tuple[int, ...]
    background_handles: Tuple[int, ...]
    sampled_frames: Tuple[int, ...]

    def as_json(self) -> Dict[str, object]:
        return {
            'episode_idx': self.episode_idx,
            'task_handles': list(self.task_handles),
            'interaction_handles': list(self.interaction_handles),
            'adjacent_handles': list(self.adjacent_handles),
            'rejected_dynamic_handles': list(self.rejected_dynamic_handles),
            'background_handles': list(self.background_handles),
            'sampled_frames': list(self.sampled_frames),
            'method': TASK_HANDLE_DETECTOR_METHOD,
        }


def build_task_frame_evidence(
    sample_frame: int,
    gripper_position: np.ndarray,
    action_position: np.ndarray,
    masks: Mapping[str, np.ndarray],
    point_clouds: Mapping[str, np.ndarray],
    *,
    excluded_ids: Iterable[int] = (0,),
) -> TaskFrameEvidence:
    '''Fuse multi-view geometry needed by the episode-level selector.'''
    excluded = {int(value) for value in excluded_ids}
    points_by_id: Dict[int, List[np.ndarray]] = {}
    for camera, mask_value in masks.items():
        if camera not in point_clouds:
            continue
        mask = np.asarray(mask_value)
        point_cloud = np.asarray(point_clouds[camera])
        if mask.shape != point_cloud.shape[:2] or point_cloud.shape[-1] != 3:
            raise ValueError(
                f'Task detector alignment mismatch for {camera}: '
                f'mask={mask.shape}, point_cloud={point_cloud.shape}'
            )
        for object_id_value in np.unique(mask):
            object_id = int(object_id_value)
            if object_id < 0 or object_id in excluded:
                continue
            points = point_cloud[mask == object_id]
            points = points[np.isfinite(points).all(axis=1)]
            if points.size:
                points_by_id.setdefault(object_id, []).append(points)

    bounds_by_id: Dict[int, Bounds] = {}
    centers_by_id: Dict[int, np.ndarray] = {}
    point_counts_by_id: Dict[int, int] = {}
    for object_id, camera_points in points_by_id.items():
        points = np.concatenate(camera_points, axis=0)
        bounds_by_id[object_id] = (
            np.min(points, axis=0),
            np.max(points, axis=0),
        )
        centers_by_id[object_id] = np.mean(
            points, axis=0, dtype=np.float64
        ).astype(np.float32)
        point_counts_by_id[object_id] = int(len(points))

    gripper = np.asarray(gripper_position, dtype=np.float32).reshape(-1)
    action = np.asarray(action_position, dtype=np.float32).reshape(-1)
    if (
        gripper.size < 3
        or action.size < 3
        or not np.isfinite(gripper[:3]).all()
        or not np.isfinite(action[:3]).all()
    ):
        raise ValueError('Task detector poses must contain 3 finite values')
    return TaskFrameEvidence(
        sample_frame=int(sample_frame),
        gripper_position=gripper[:3],
        action_position=action[:3],
        bounds_by_id=bounds_by_id,
        centers_by_id=centers_by_id,
        point_counts_by_id=point_counts_by_id,
    )


def _point_to_bounds_distance(point: np.ndarray, bounds: Bounds) -> float:
    minimum, maximum = bounds
    outside = np.maximum(np.maximum(minimum - point, point - maximum), 0.0)
    return float(np.linalg.norm(outside))


def _bounds_distance(left: Bounds, right: Bounds) -> float:
    left_min, left_max = left
    right_min, right_max = right
    outside = np.maximum(
        np.maximum(left_min - right_max, right_min - left_max), 0.0
    )
    return float(np.linalg.norm(outside))


def _is_planar_background(sizes: Sequence[np.ndarray], extent: float) -> bool:
    if not sizes:
        return False
    median_size = np.median(np.stack(sizes), axis=0)
    ordered = np.sort(median_size)
    return bool(ordered[-1] >= extent and ordered[-2] >= extent)


def detect_task_handles(
    task_name: str,
    episode_idx: int,
    frames: Sequence[TaskFrameEvidence],
    *,
    interaction_radius: Optional[float] = None,
    max_instances: Optional[int] = None,
    background_extent: float = 0.60,
    motion_threshold: float = 0.03,
    adjacency_distance: float = 0.04,
    adjacency_ratio: float = 0.50,
) -> TaskHandleDetection:
    '''Build one stable handle whitelist for an entire demonstration.

    Motion is never required for retention. A moving handle is rejected as
    unexplained only when no sampled current/next gripper pose approaches it.
    This preserves static targets and non-grasp pushing/articulation tasks.
    '''
    if not frames:
        raise ValueError('Task handle detection requires at least one frame')
    if min(background_extent, motion_threshold, adjacency_distance) <= 0:
        raise ValueError('Task detector distance thresholds must be positive')
    if not 0 < adjacency_ratio <= 1:
        raise ValueError('adjacency_ratio must be in (0, 1]')

    prior = get_task_object_prior(task_name)
    radius = prior.interaction_radius if interaction_radius is None else float(
        interaction_radius
    )
    limit = prior.max_instances if max_instances is None else int(max_instances)
    if radius <= 0 or limit <= 0:
        raise ValueError('Task detector radius and max instances must be positive')

    frames = sorted(frames, key=lambda frame: frame.sample_frame)
    all_ids = sorted(set().union(*(set(frame.bounds_by_id) for frame in frames)))
    sizes: Dict[int, List[np.ndarray]] = {object_id: [] for object_id in all_ids}
    centers: Dict[int, List[Tuple[int, np.ndarray]]] = {
        object_id: [] for object_id in all_ids
    }
    minimum_distance: Dict[int, float] = {
        object_id: float('inf') for object_id in all_ids
    }
    point_support: Dict[int, int] = {object_id: 0 for object_id in all_ids}
    near_in_frame: Dict[Tuple[int, int], bool] = {}

    for frame_index, frame in enumerate(frames):
        for object_id, bounds in frame.bounds_by_id.items():
            sizes[object_id].append(bounds[1] - bounds[0])
            centers[object_id].append((frame_index, frame.centers_by_id[object_id]))
            point_support[object_id] += frame.point_counts_by_id.get(object_id, 0)
            distance = min(
                _point_to_bounds_distance(frame.gripper_position, bounds),
                _point_to_bounds_distance(frame.action_position, bounds),
            )
            minimum_distance[object_id] = min(minimum_distance[object_id], distance)
            near_in_frame[(frame_index, object_id)] = distance <= radius

    background = {
        object_id
        for object_id in all_ids
        if _is_planar_background(sizes[object_id], background_extent)
    }
    interaction = {
        object_id
        for object_id in all_ids
        if object_id not in background and minimum_distance[object_id] <= radius
    }

    rejected_dynamic = set()
    for object_id in all_ids:
        observations = centers[object_id]
        if object_id in background or len(observations) < 3:
            continue
        significant_steps = []
        for (left_frame, left), (right_frame, right) in zip(
            observations, observations[1:]
        ):
            if float(np.linalg.norm(right - left)) < motion_threshold:
                continue
            explained = (
                near_in_frame.get((left_frame, object_id), False)
                or near_in_frame.get((right_frame, object_id), False)
            )
            significant_steps.append(explained)
        if significant_steps and not any(significant_steps):
            rejected_dynamic.add(object_id)

    # Keep one hop of persistent geometry attached to an interacted handle.
    # This retains static targets/containers composed of several simulator
    # handles without recursively swallowing an entire scene hierarchy.
    adjacent = set()
    for candidate in all_ids:
        if candidate in interaction or candidate in background or candidate in rejected_dynamic:
            continue
        for seed in interaction:
            co_visible = 0
            touching = 0
            for frame in frames:
                if candidate not in frame.bounds_by_id or seed not in frame.bounds_by_id:
                    continue
                co_visible += 1
                if _bounds_distance(
                    frame.bounds_by_id[candidate], frame.bounds_by_id[seed]
                ) <= adjacency_distance:
                    touching += 1
            if co_visible >= 2 and touching / co_visible >= adjacency_ratio:
                adjacent.add(candidate)
                break

    ranked = sorted(
        interaction | adjacent,
        key=lambda object_id: (
            object_id not in interaction,
            minimum_distance[object_id],
            -point_support[object_id],
            object_id,
        ),
    )
    if not ranked:
        # Never silently emit an empty episode map if all sampled poses are
        # slightly outside the configured radius.
        ranked = sorted(
            (object_id for object_id in all_ids if object_id not in background),
            key=lambda object_id: (
                minimum_distance[object_id],
                -point_support[object_id],
                object_id,
            ),
        )[: min(2, limit)]
    selected = tuple(sorted(ranked[:limit]))
    selected_set = set(selected)
    return TaskHandleDetection(
        episode_idx=int(episode_idx),
        task_handles=selected,
        interaction_handles=tuple(sorted(interaction & selected_set)),
        adjacent_handles=tuple(sorted(adjacent & selected_set)),
        rejected_dynamic_handles=tuple(sorted(rejected_dynamic)),
        background_handles=tuple(sorted(background)),
        sampled_frames=tuple(frame.sample_frame for frame in frames),
    )


def save_task_handle_detection(path: Path, detection: TaskHandleDetection) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f'{path}.tmp')
    temporary.write_text(
        json.dumps(detection.as_json(), indent=2, sort_keys=True),
        encoding='utf-8',
    )
    temporary.replace(path)


def load_task_handle_detection(path: Path) -> TaskHandleDetection:
    payload = json.loads(path.read_text(encoding='utf-8'))
    method = payload.get('method')
    if method != TASK_HANDLE_DETECTOR_METHOD:
        raise ValueError(
            f'Stale task-handle cache method {method!r}; '
            f'expected {TASK_HANDLE_DETECTOR_METHOD!r}'
        )
    return TaskHandleDetection(
        episode_idx=int(payload['episode_idx']),
        task_handles=tuple(int(value) for value in payload['task_handles']),
        interaction_handles=tuple(
            int(value) for value in payload['interaction_handles']
        ),
        adjacent_handles=tuple(int(value) for value in payload['adjacent_handles']),
        rejected_dynamic_handles=tuple(
            int(value) for value in payload['rejected_dynamic_handles']
        ),
        background_handles=tuple(
            int(value) for value in payload['background_handles']
        ),
        sampled_frames=tuple(int(value) for value in payload['sampled_frames']),
    )
