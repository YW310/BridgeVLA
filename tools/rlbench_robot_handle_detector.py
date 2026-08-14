'''Offline temporal detection of RLBench robot/gripper mask handles.'''

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


Bounds = Tuple[np.ndarray, np.ndarray]
ROBOT_DETECTOR_METHOD = 'wrist_pose_temporal_adjacency_v12_early_seed_visibility'


@dataclass(frozen=True)
class RobotFrameEvidence:
    sample_frame: int
    gripper_position: np.ndarray
    bounds_by_id: Mapping[int, Bounds]
    centers_by_id: Mapping[int, np.ndarray]
    wrist_centroids_by_id: Mapping[int, np.ndarray]
    gripper_open: float = 1.0


@dataclass(frozen=True)
class RobotHandleDetection:
    episode_idx: int
    gripper_handles: Tuple[int, ...]
    arm_handles: Tuple[int, ...]
    ambiguous_handles: Tuple[int, ...]
    grasped_handles: Tuple[int, ...]
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
            'ambiguous_handles': list(self.ambiguous_handles),
            'grasped_handles': list(self.grasped_handles),
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
    gripper_open: float = 1.0,
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
    gripper_open = float(gripper_open)
    if not np.isfinite(gripper_open):
        raise ValueError('Current gripper_open must be finite')
    return RobotFrameEvidence(
        int(sample_frame),
        gripper_position[:3],
        bounds_by_id,
        centers_by_id,
        wrist_centroids,
        gripper_open,
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


def _detect_grasped_handles(
    frames: Sequence[RobotFrameEvidence],
    object_ids: Sequence[int],
    *,
    gripper_radius: float,
    follow_tolerance: float = 0.035,
    min_follow_motion: float = 0.01,
) -> set:
    '''Find handles that begin following the gripper after it closes.'''
    grasped = set()
    for close_index in range(1, len(frames)):
        if not (
            frames[close_index - 1].gripper_open >= 0.5
            and frames[close_index].gripper_open < 0.5
        ):
            continue
        closed_end = close_index + 1
        while (
            closed_end < len(frames)
            and frames[closed_end].gripper_open < 0.5
        ):
            closed_end += 1
        pre_start = max(0, close_index - 3)

        for object_id in object_ids:
            close_frame = frames[close_index]
            bounds = close_frame.bounds_by_id.get(object_id)
            if bounds is None or _point_to_bounds_distance(
                close_frame.gripper_position, bounds
            ) > gripper_radius:
                continue

            def motion_pairs(start: int, stop: int):
                values = []
                for right_index in range(start + 1, stop):
                    left = frames[right_index - 1]
                    right = frames[right_index]
                    if (
                        object_id not in left.centers_by_id
                        or object_id not in right.centers_by_id
                    ):
                        continue
                    object_delta = (
                        right.centers_by_id[object_id]
                        - left.centers_by_id[object_id]
                    )
                    gripper_delta = (
                        right.gripper_position - left.gripper_position
                    )
                    values.append(
                        (
                            float(np.linalg.norm(object_delta - gripper_delta)),
                            float(np.linalg.norm(object_delta)),
                            float(np.linalg.norm(gripper_delta)),
                        )
                    )
                return values

            before = motion_pairs(pre_start, close_index)
            after = motion_pairs(close_index, closed_end)
            if not before:
                continue
            before_error = float(np.mean([value[0] for value in before]))
            before_object_motion = float(sum(value[1] for value in before))
            before_gripper_motion = float(sum(value[2] for value in before))
            did_not_follow_before = (
                before_error > follow_tolerance
                or (
                    before_gripper_motion >= min_follow_motion
                    and before_object_motion < before_gripper_motion * 0.6
                )
            )
            if not did_not_follow_before:
                continue
            if not after:
                # A close event at the end of the sampled sequence still
                # protects a nearby non-robot object. Later motion is useful
                # confirmation, but is not required for safe exclusion.
                grasped.add(object_id)
                continue
            after_error = float(np.mean([value[0] for value in after]))
            after_object_motion = float(sum(value[1] for value in after))
            after_gripper_motion = float(sum(value[2] for value in after))
            if (
                after_error <= follow_tolerance
                and after_object_motion >= min_follow_motion * 0.5
                and after_gripper_motion >= min_follow_motion
            ):
                grasped.add(object_id)
    return grasped


def detect_robot_handles(
    episode_idx: int,
    frames: Sequence[RobotFrameEvidence],
    *,
    gripper_radius: float = 0.14,
    wrist_centroid_std: float = 0.08,
    relative_offset_std: float = 0.12,
    min_visibility_ratio: float = 0.75,
    adjacency_distance: float = 0.05,
    adjacency_ratio: float = 0.60,
    min_link_motion: float = 0.008,
    min_hard_confidence: float = 0.75,
    arm_min_visibility_ratio: float = 0.50,
) -> RobotHandleDetection:
    '''Detect gripper seeds and expand persistent moving robot-link neighbors.'''
    if not frames:
        raise ValueError('Robot detection requires at least one frame')
    if not all(
        0 < value <= 1
        for value in (
            min_visibility_ratio,
            adjacency_ratio,
            min_hard_confidence,
            arm_min_visibility_ratio,
        )
    ):
        raise ValueError(
            'visibility, adjacency, and confidence ratios must be in (0, 1]'
        )
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
    grasped_handles = _detect_grasped_handles(
        frames,
        all_ids,
        gripper_radius=gripper_radius,
    )
    first_closed_index = next(
        (
            index
            for index, frame in enumerate(frames)
            if frame.gripper_open < 0.5
        ),
        frame_count,
    )
    pre_grasp_frames = frames[:first_closed_index]

    def pre_grasp_motion(object_id: int) -> Tuple[float, float]:
        object_motion = 0.0
        gripper_motion = 0.0
        for left, right in zip(pre_grasp_frames, pre_grasp_frames[1:]):
            if (
                object_id not in left.centers_by_id
                or object_id not in right.centers_by_id
            ):
                continue
            object_motion += float(
                np.linalg.norm(
                    right.centers_by_id[object_id]
                    - left.centers_by_id[object_id]
                )
            )
            gripper_motion += float(
                np.linalg.norm(
                    right.gripper_position - left.gripper_position
                )
            )
        return object_motion, gripper_motion
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
    ambiguous_handles = set()
    confidence: Dict[int, float] = {}
    minimum_visible = min(
        frame_count,
        max(2, int(np.ceil(frame_count * min_visibility_ratio))),
    )
    first_frame = frames[0]
    early_seed_limit = max(1, int(np.ceil(frame_count * 0.25)))
    for object_id in all_ids:
        if object_id in grasped_handles:
            ambiguous_handles.add(object_id)
            continue
        object_pre_motion, gripper_pre_motion = pre_grasp_motion(object_id)
        if (
            gripper_pre_motion >= min_link_motion
            and object_pre_motion < min_link_motion
            and object_pre_motion < gripper_pre_motion * 0.25
        ):
            # A task object is commonly static while the open gripper moves
            # toward it. Robot geometry must already move before grasping.
            ambiguous_handles.add(object_id)
            continue
        wrist_values = wrist_centroids.get(object_id, [])
        seed_frame_number = first_seen.get(object_id, frame_count)
        seed_frame = (
            frames[seed_frame_number]
            if seed_frame_number < frame_count
            else first_frame
        )
        # A gripper link may be hidden in raw frame 0 and first become visible
        # a few samples later. Accept that early appearance, but reject handles
        # that only enter the gripper after the early prefix.
        if (
            seed_frame_number > early_seed_limit
            or object_id not in seed_frame.bounds_by_id
            or object_id not in seed_frame.wrist_centroids_by_id
            or visibility[object_id] < minimum_visible
            or len(wrist_values) < minimum_visible
            or _point_to_bounds_distance(
                seed_frame.gripper_position,
                seed_frame.bounds_by_id[object_id],
            ) > gripper_radius
            or sum(
                distance <= gripper_radius
                for distance in distances[object_id]
            ) / len(distances[object_id]) < min_visibility_ratio
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
            score = float(
                0.30 * visibility_score
                + 0.25 * position_score
                + 0.25 * offset_score
                + 0.20 * distance_score
            )
            if score >= min_hard_confidence:
                confidence[object_id] = score
                gripper_handles.append(object_id)
            else:
                ambiguous_handles.add(object_id)

    if not gripper_handles:
        # Recover a wrist-stable gripper seed when strict scoring misses due to
        # rotation, partial masks, or point-cloud offset noise. Grasped and
        # static-before-grasp objects remain protected.
        candidates = []
        for object_id in all_ids:
            if object_id in grasped_handles:
                continue
            object_pre_motion, gripper_pre_motion = pre_grasp_motion(object_id)
            if (
                gripper_pre_motion >= min_link_motion
                and object_pre_motion < min_link_motion
                and object_pre_motion < gripper_pre_motion * 0.25
            ):
                ambiguous_handles.add(object_id)
                continue
            wrist_values = wrist_centroids.get(object_id, [])
            seed_frame_number = first_seen.get(object_id, frame_count)
            seed_frame = (
                frames[seed_frame_number]
                if seed_frame_number < frame_count
                else first_frame
            )
            if (
                seed_frame_number > early_seed_limit
                or object_id not in seed_frame.bounds_by_id
                or object_id not in seed_frame.wrist_centroids_by_id
                or len(wrist_values) < minimum_visible
            ):
                continue
            first_distance = _point_to_bounds_distance(
                seed_frame.gripper_position,
                seed_frame.bounds_by_id[object_id],
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
            wrist_array = np.stack(wrist_values)
            centroid_spread = float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (wrist_array - wrist_array.mean(0)) ** 2,
                            axis=1,
                        )
                    )
                )
            )
            if (
                first_distance <= gripper_radius * 1.5
                and offset_spread <= relative_offset_std * 1.5
                and centroid_spread <= wrist_centroid_std * 1.5
            ):
                candidates.append(
                    (
                        centroid_spread,
                        offset_spread,
                        float(np.median(distances[object_id])),
                        object_id,
                    )
                )
        for (
            centroid_spread,
            offset_spread,
            median_distance,
            object_id,
        ) in sorted(candidates)[:2]:
            if median_distance <= gripper_radius * 1.5:
                gripper_handles.append(object_id)
                confidence[object_id] = max(
                    0.55,
                    0.75
                    - 0.10 * centroid_spread / wrist_centroid_std
                    - 0.10 * offset_spread / relative_offset_std,
                )
                ambiguous_handles.discard(object_id)

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

    gripper_set = set(gripper_handles)
    robot_handles = set(gripper_set)
    queue = list(gripper_handles)
    early_limit = max(1, int(np.ceil(frame_count * 0.25)))
    minimum_arm_visible = min(
        frame_count,
        max(2, int(np.ceil(frame_count * arm_min_visibility_ratio))),
    )
    while queue:
        object_id = queue.pop()
        for neighbor in adjacency.get(object_id, ()):
            if neighbor in robot_handles:
                continue
            if neighbor in grasped_handles:
                ambiguous_handles.add(neighbor)
                continue
            # Robot links are present from the beginning and move over the
            # episode. Objects that only enter the gripper later are rejected.
            if first_seen.get(neighbor, frame_count) > early_limit:
                continue
            early_centers = [
                frame.centers_by_id[neighbor]
                for frame in early_frames
                if neighbor in frame.centers_by_id
            ]
            early_motion = 0.0
            if len(early_centers) >= 2:
                early_array = np.stack(early_centers)
                early_motion = float(
                    np.linalg.norm(
                        np.max(early_array, 0) - np.min(early_array, 0)
                    )
                )
            first_adjacent = (
                neighbor in first_frame.bounds_by_id
                and object_id in first_frame.bounds_by_id
                and _bounds_distance(
                    first_frame.bounds_by_id[object_id],
                    first_frame.bounds_by_id[neighbor],
                ) <= adjacency_distance
            )
            first_bounds = first_frame.bounds_by_id.get(neighbor)
            maximum_extent = float('inf')
            if first_bounds is not None:
                maximum_extent = float(
                    np.max(first_bounds[1] - first_bounds[0])
                )
            persistent_chain_evidence = (
                visibility[neighbor] >= minimum_arm_visible
                and len(early_centers) >= minimum_early_co_visible
                and first_adjacent
                and maximum_extent <= 0.50
            )
            moving_link_evidence = (
                motion[neighbor] >= min_link_motion
                and early_motion >= min_link_motion * 0.5
            )
            # The first hop beside a gripper seed must already move; this
            # protects a static task object that the open gripper approaches.
            # Once a moving arm link is established, persistent adjacent links
            # may be static (for example the robot base).
            hard_robot_evidence = persistent_chain_evidence and (
                moving_link_evidence or object_id not in gripper_set
            )
            object_pre_motion, gripper_pre_motion = pre_grasp_motion(neighbor)
            if (
                gripper_pre_motion >= min_link_motion
                and object_pre_motion < min_link_motion
                and object_pre_motion < gripper_pre_motion * 0.25
                and object_id in gripper_set
            ):
                hard_robot_evidence = False
            if not hard_robot_evidence:
                ambiguous_handles.add(neighbor)
                continue
            robot_handles.add(neighbor)
            confidence[neighbor] = min(
                0.90,
                0.75 + min(0.15, early_motion),
            )
            queue.append(neighbor)

    gripper_set.difference_update(grasped_handles)
    robot_handles.difference_update(grasped_handles)
    arm_handles = sorted(robot_handles - gripper_set)
    return RobotHandleDetection(
        episode_idx=int(episode_idx),
        gripper_handles=tuple(sorted(gripper_set)),
        arm_handles=tuple(arm_handles),
        ambiguous_handles=tuple(sorted(ambiguous_handles - robot_handles)),
        grasped_handles=tuple(sorted(grasped_handles)),
        confidence=confidence,
        sampled_frames=tuple(frame.sample_frame for frame in frames),
    )


def save_robot_handle_detection(
    path: Path,
    detection: RobotHandleDetection,
    sampling_config: Optional[Mapping[str, object]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f'{path}.tmp')
    payload = detection.as_json()
    if sampling_config is not None:
        payload['sampling_config'] = dict(sampling_config)
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding='utf-8',
    )
    temporary.replace(path)


def load_robot_handle_detection(
    path: Path,
    expected_sampling_config: Optional[Mapping[str, object]] = None,
) -> RobotHandleDetection:
    payload = json.loads(path.read_text(encoding='utf-8'))
    method = payload.get('method')
    if method != ROBOT_DETECTOR_METHOD:
        raise ValueError(
            f'Stale robot-handle cache method {method!r}; '
            f'expected {ROBOT_DETECTOR_METHOD!r}'
        )
    cached_sampling_config = payload.get('sampling_config')
    if (
        expected_sampling_config is not None
        and cached_sampling_config != dict(expected_sampling_config)
    ):
        raise ValueError(
            'Stale robot-handle cache sampling configuration '
            f'{cached_sampling_config!r}; '
            f'expected {dict(expected_sampling_config)!r}'
        )
    return RobotHandleDetection(
        episode_idx=int(payload['episode_idx']),
        gripper_handles=tuple(int(value) for value in payload['gripper_handles']),
        arm_handles=tuple(int(value) for value in payload['arm_handles']),
        ambiguous_handles=tuple(
            int(value) for value in payload.get('ambiguous_handles', ())
        ),
        grasped_handles=tuple(
            int(value) for value in payload.get('grasped_handles', ())
        ),
        confidence={
            int(handle): float(value)
            for handle, value in payload.get('confidence', {}).items()
        },
        sampled_frames=tuple(int(value) for value in payload['sampled_frames']),
    )
