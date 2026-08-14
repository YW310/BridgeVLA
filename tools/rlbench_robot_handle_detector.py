'''Offline temporal detection of RLBench robot/gripper mask handles.'''

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


Bounds = Tuple[np.ndarray, np.ndarray]
ROBOT_DETECTOR_METHOD = 'wrist_pose_temporal_adjacency_v3'


@dataclass(frozen=True)
class RobotFrameEvidence:
    sample_frame: int
    gripper_position: np.ndarray
    bounds_by_id: Mapping[int, Bounds]
    centers_by_id: Mapping[int, np.ndarray]
    wrist_centroids_by_id: Mapping[int, np.ndarray]


@dataclass(frozen=True)
class RobotHandleDetection:
    episode_idx: int
    gripper_handles: Tuple[int, ...]
    arm_handles: Tuple[int, ...]
    confidence: Mapping[int, float]
    sampled_frames: Tuple[int, ...]

    @property
    def robot_handles(self) -> Tuple[int, ...]:
        return tuple(sorted(set(self.gripper_handles + self.arm_handles)))

    def as_json(self) -> Dict[str, object]:
        return {
            'episode_idx': self.episode_idx,
            'gripper_handles': list(self.gripper_handles),
            'arm_handles': list(self.arm_handles),
            'robot_handles': list(self.robot_handles),
            'confidence': {
                str(handle): float(value)
                for handle, value in sorted(self.confidence.items())
            },
            'sampled_frames': list(self.sampled_frames),
            'method': ROBOT_DETECTOR_METHOD,
        }


def build_robot_frame_evidence(
    sample_frame: int,
    gripper_position: np.ndarray,
    masks: Mapping[str, np.ndarray],
    point_clouds: Mapping[str, np.ndarray],
    *,
    excluded_ids: Iterable[int] = (0,),
) -> RobotFrameEvidence:
    '''Build per-handle geometry and wrist-image evidence for one raw frame.'''
    excluded = {int(value) for value in excluded_ids}
    points_by_id: Dict[int, List[np.ndarray]] = {}
    wrist_centroids: Dict[int, np.ndarray] = {}
    for camera, mask_value in masks.items():
        if camera not in point_clouds:
            continue
        mask = np.asarray(mask_value)
        point_cloud = np.asarray(point_clouds[camera])
        if mask.shape != point_cloud.shape[:2] or point_cloud.shape[-1] != 3:
            raise ValueError(
                f'Robot detector alignment mismatch for {camera}: '
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
            if camera == 'wrist':
                pixels = np.argwhere(mask == object_id)
                if pixels.size:
                    scale = np.maximum(np.asarray(mask.shape) - 1, 1)
                    wrist_centroids[object_id] = (
                        np.mean(pixels, axis=0) / scale
                    ).astype(np.float32)

    bounds_by_id: Dict[int, Bounds] = {}
    centers_by_id: Dict[int, np.ndarray] = {}
    for object_id, camera_points in points_by_id.items():
        points = np.concatenate(camera_points, axis=0)
        bounds_by_id[object_id] = (
            np.min(points, axis=0),
            np.max(points, axis=0),
        )
        centers_by_id[object_id] = np.mean(
            points, axis=0, dtype=np.float64
        ).astype(np.float32)
    gripper_position = np.asarray(gripper_position, dtype=np.float32).reshape(-1)
    if gripper_position.size < 3 or not np.isfinite(gripper_position[:3]).all():
        raise ValueError('Current gripper position must contain 3 finite values')
    return RobotFrameEvidence(
        int(sample_frame),
        gripper_position[:3],
        bounds_by_id,
        centers_by_id,
        wrist_centroids,
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


def detect_robot_handles(
    episode_idx: int,
    frames: Sequence[RobotFrameEvidence],
    *,
    gripper_radius: float = 0.14,
    wrist_centroid_std: float = 0.08,
    relative_offset_std: float = 0.06,
    min_visibility_ratio: float = 0.75,
    adjacency_distance: float = 0.05,
    adjacency_ratio: float = 0.70,
    min_link_motion: float = 0.008,
) -> RobotHandleDetection:
    '''Detect gripper seeds and expand persistent moving robot-link neighbors.'''
    if not frames:
        raise ValueError('Robot detection requires at least one frame')
    if not 0 < min_visibility_ratio <= 1 or not 0 < adjacency_ratio <= 1:
        raise ValueError('visibility and adjacency ratios must be in (0, 1]')
    if min(
        gripper_radius,
        wrist_centroid_std,
        relative_offset_std,
        adjacency_distance,
        min_link_motion,
    ) <= 0:
        raise ValueError('Robot detector distance thresholds must be positive')

    frames = sorted(frames, key=lambda frame: frame.sample_frame)
    frame_count = len(frames)
    all_ids = sorted(
        set().union(*(set(frame.bounds_by_id) for frame in frames))
    )
    visibility: Dict[int, int] = {object_id: 0 for object_id in all_ids}
    distances: Dict[int, List[float]] = {object_id: [] for object_id in all_ids}
    wrist_centroids: Dict[int, List[np.ndarray]] = {
        object_id: [] for object_id in all_ids
    }
    centers: Dict[int, List[np.ndarray]] = {object_id: [] for object_id in all_ids}
    relative_offsets: Dict[int, List[np.ndarray]] = {
        object_id: [] for object_id in all_ids
    }
    first_seen: Dict[int, int] = {}

    for frame_number, frame in enumerate(frames):
        for object_id, bounds in frame.bounds_by_id.items():
            visibility[object_id] += 1
            first_seen.setdefault(object_id, frame_number)
            distances[object_id].append(
                _point_to_bounds_distance(frame.gripper_position, bounds)
            )
            centers[object_id].append(frame.centers_by_id[object_id])
            relative_offsets[object_id].append(
                frame.centers_by_id[object_id] - frame.gripper_position
            )
        for object_id, centroid in frame.wrist_centroids_by_id.items():
            wrist_centroids.setdefault(object_id, []).append(centroid)

    gripper_handles: List[int] = []
    confidence: Dict[int, float] = {}
    minimum_visible = min(
        frame_count,
        max(2, int(np.ceil(frame_count * min_visibility_ratio))),
    )
    first_frame = frames[0]
    for object_id in all_ids:
        wrist_values = wrist_centroids.get(object_id, [])
        # A gripper link must already be visible beside the gripper at the
        # beginning. A manipulated object may follow the wrist later, but must
        # never become a gripper seed for that reason.
        if (
            object_id not in first_frame.bounds_by_id
            or object_id not in first_frame.wrist_centroids_by_id
            or first_seen.get(object_id) != 0
            or visibility[object_id] < minimum_visible
            or len(wrist_values) < minimum_visible
            or _point_to_bounds_distance(
                first_frame.gripper_position,
                first_frame.bounds_by_id[object_id],
            ) > gripper_radius
        ):
            continue
        wrist_array = np.stack(wrist_values)
        centroid_spread = float(
            np.sqrt(np.mean(np.sum((wrist_array - wrist_array.mean(0)) ** 2, axis=1)))
        )
        offset_array = np.stack(relative_offsets[object_id])
        offset_spread = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (offset_array - offset_array.mean(0)) ** 2,
                        axis=1,
                    )
                )
            )
        )
        median_distance = float(np.median(distances[object_id]))
        if (
            centroid_spread <= wrist_centroid_std
            and offset_spread <= relative_offset_std
            and median_distance <= gripper_radius
        ):
            visibility_score = visibility[object_id] / frame_count
            position_score = max(0.0, 1.0 - centroid_spread / wrist_centroid_std)
            offset_score = max(0.0, 1.0 - offset_spread / relative_offset_std)
            distance_score = max(0.0, 1.0 - median_distance / gripper_radius)
            confidence[object_id] = float(
                0.30 * visibility_score
                + 0.25 * position_score
                + 0.25 * offset_score
                + 0.20 * distance_score
            )
            gripper_handles.append(object_id)

    if not gripper_handles:
        # Conservative fallback: choose persistent wrist-visible handles nearest
        # the current gripper, never arbitrary scene objects.
        candidates = []
        for object_id in all_ids:
            wrist_values = wrist_centroids.get(object_id, [])
            if (
                first_seen.get(object_id) != 0
                or object_id not in first_frame.bounds_by_id
                or object_id not in first_frame.wrist_centroids_by_id
                or len(wrist_values) < minimum_visible
            ):
                continue
            first_distance = _point_to_bounds_distance(
                first_frame.gripper_position,
                first_frame.bounds_by_id[object_id],
            )
            offset_array = np.stack(relative_offsets[object_id])
            offset_spread = float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (offset_array - offset_array.mean(0)) ** 2,
                            axis=1,
                        )
                    )
                )
            )
            if (
                first_distance <= gripper_radius * 1.5
                and offset_spread <= relative_offset_std * 1.5
            ):
                candidates.append(
                    (float(np.median(distances[object_id])), object_id)
                )
        for median_distance, object_id in sorted(candidates)[:2]:
            if median_distance <= gripper_radius * 1.5:
                gripper_handles.append(object_id)
                confidence[object_id] = 0.35

    motion: Dict[int, float] = {}
    for object_id, values in centers.items():
        if len(values) < 2:
            motion[object_id] = 0.0
            continue
        array = np.stack(values)
        motion[object_id] = float(np.linalg.norm(np.max(array, 0) - np.min(array, 0)))

    adjacency: Dict[int, set] = {object_id: set() for object_id in all_ids}
    early_frame_count = min(
        frame_count,
        max(2, int(np.ceil(frame_count * 0.35))),
    )
    early_frames = frames[:early_frame_count]
    minimum_early_co_visible = min(2, early_frame_count)
    for left_index, left_id in enumerate(all_ids):
        for right_id in all_ids[left_index + 1:]:
            co_visible = 0
            adjacent = 0
            for frame in frames:
                if left_id not in frame.bounds_by_id or right_id not in frame.bounds_by_id:
                    continue
                co_visible += 1
                if _bounds_distance(
                    frame.bounds_by_id[left_id], frame.bounds_by_id[right_id]
                ) <= adjacency_distance:
                    adjacent += 1
            early_co_visible = 0
            early_adjacent = 0
            for frame in early_frames:
                if left_id not in frame.bounds_by_id or right_id not in frame.bounds_by_id:
                    continue
                early_co_visible += 1
                if _bounds_distance(
                    frame.bounds_by_id[left_id], frame.bounds_by_id[right_id]
                ) <= adjacency_distance:
                    early_adjacent += 1
            if (
                co_visible >= 2
                and adjacent / co_visible >= adjacency_ratio
                and early_co_visible >= minimum_early_co_visible
                and early_adjacent / early_co_visible >= adjacency_ratio
            ):
                adjacency[left_id].add(right_id)
                adjacency[right_id].add(left_id)

    robot_handles = set(gripper_handles)
    queue = list(gripper_handles)
    early_limit = max(1, int(np.ceil(frame_count * 0.25)))
    while queue:
        object_id = queue.pop()
        for neighbor in adjacency.get(object_id, ()):
            if neighbor in robot_handles:
                continue
            # Robot links are present from the beginning and move over the
            # episode. Objects that only enter the gripper later are rejected.
            if first_seen.get(neighbor, frame_count) > early_limit:
                continue
            if visibility[neighbor] < minimum_visible:
                continue
            if motion[neighbor] < min_link_motion:
                continue
            robot_handles.add(neighbor)
            confidence[neighbor] = min(
                0.90,
                0.55 + min(0.35, motion[neighbor]),
            )
            queue.append(neighbor)

    gripper_set = set(gripper_handles)
    arm_handles = sorted(robot_handles - gripper_set)
    return RobotHandleDetection(
        episode_idx=int(episode_idx),
        gripper_handles=tuple(sorted(gripper_set)),
        arm_handles=tuple(arm_handles),
        confidence=confidence,
        sampled_frames=tuple(frame.sample_frame for frame in frames),
    )


def save_robot_handle_detection(
    path: Path, detection: RobotHandleDetection
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f'{path}.tmp')
    temporary.write_text(
        json.dumps(detection.as_json(), indent=2, sort_keys=True),
        encoding='utf-8',
    )
    temporary.replace(path)


def load_robot_handle_detection(path: Path) -> RobotHandleDetection:
    payload = json.loads(path.read_text(encoding='utf-8'))
    method = payload.get('method')
    if method != ROBOT_DETECTOR_METHOD:
        raise ValueError(
            f'Stale robot-handle cache method {method!r}; '
            f'expected {ROBOT_DETECTOR_METHOD!r}'
        )
    return RobotHandleDetection(
        episode_idx=int(payload['episode_idx']),
        gripper_handles=tuple(int(value) for value in payload['gripper_handles']),
        arm_handles=tuple(int(value) for value in payload['arm_handles']),
        confidence={
            int(handle): float(value)
            for handle, value in payload.get('confidence', {}).items()
        },
        sampled_frames=tuple(int(value) for value in payload['sampled_frames']),
    )
