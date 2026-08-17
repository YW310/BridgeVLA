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
TASK_HANDLE_DETECTOR_METHOD = 'episode_action_trajectory_v8_structural_components'


@dataclass(frozen=True)
class TaskFrameEvidence:
    sample_frame: int
    gripper_position: np.ndarray
    action_position: np.ndarray
    bounds_by_id: Mapping[int, Bounds]
    centers_by_id: Mapping[int, np.ndarray]
    point_counts_by_id: Mapping[int, int]
    gripper_open: float = 1.0


@dataclass(frozen=True)
class TaskHandleDetection:
    episode_idx: int
    task_handles: Tuple[int, ...]
    observed_handles: Tuple[int, ...]
    interaction_handles: Tuple[int, ...]
    adjacent_handles: Tuple[int, ...]
    rejected_dynamic_handles: Tuple[int, ...]
    background_handles: Tuple[int, ...]
    sampled_frames: Tuple[int, ...]
    target_handles: Tuple[int, ...] = ()
    reference_handles: Tuple[int, ...] = ()
    object_groups: Tuple[Tuple[int, ...], ...] = ()

    @property
    def slot_handles(self) -> Tuple[int, ...]:
        '''Return a stable, high-recall episode ordering for Oracle slots.'''
        task_handle_set = set(self.task_handles)
        return self.task_handles + tuple(
            handle
            for handle in self.observed_handles
            if handle not in task_handle_set
        )

    @property
    def role_by_handle(self) -> Dict[int, int]:
        '''Return replay role codes: 1=target, 2=reference.'''
        roles = {handle: 1 for handle in self.target_handles}
        roles.update({handle: 2 for handle in self.reference_handles})
        return roles

    @property
    def group_by_handle(self) -> Dict[int, int]:
        '''Map every raw handle to its stable representative handle ID.'''
        mapping: Dict[int, int] = {}
        for group in self.object_groups:
            representative = min(group)
            mapping.update({handle: representative for handle in group})
        for handle in self.observed_handles:
            mapping.setdefault(handle, handle)
        return mapping

    @property
    def grouped_slot_handles(self) -> Tuple[int, ...]:
        mapping = self.group_by_handle
        return tuple(
            dict.fromkeys(mapping.get(handle, handle) for handle in self.slot_handles)
        )

    @property
    def role_by_group(self) -> Dict[int, int]:
        mapping = self.group_by_handle
        roles: Dict[int, int] = {}
        for handle, role in self.role_by_handle.items():
            representative = mapping.get(handle, handle)
            roles[representative] = min(roles.get(representative, role), role)
        return roles

    def as_json(self) -> Dict[str, object]:
        return {
            'episode_idx': self.episode_idx,
            'task_handles': list(self.task_handles),
            'observed_handles': list(self.observed_handles),
            'slot_handles': list(self.slot_handles),
            'interaction_handles': list(self.interaction_handles),
            'adjacent_handles': list(self.adjacent_handles),
            'rejected_dynamic_handles': list(self.rejected_dynamic_handles),
            'background_handles': list(self.background_handles),
            'target_handles': list(self.target_handles),
            'reference_handles': list(self.reference_handles),
            'object_groups': [list(group) for group in self.object_groups],
            'group_by_handle': {
                str(handle): representative
                for handle, representative in self.group_by_handle.items()
            },
            'grouped_slot_handles': list(self.grouped_slot_handles),
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
    gripper_open: float = 1.0,
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
    gripper_open = float(gripper_open)
    if not np.isfinite(gripper_open):
        raise ValueError('Task detector gripper_open must be finite')
    return TaskFrameEvidence(
        sample_frame=int(sample_frame),
        gripper_position=gripper[:3],
        action_position=action[:3],
        bounds_by_id=bounds_by_id,
        centers_by_id=centers_by_id,
        point_counts_by_id=point_counts_by_id,
        gripper_open=gripper_open,
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


def _detect_rigid_groups(
    object_ids: Sequence[int],
    frames: Sequence[TaskFrameEvidence],
    *,
    excluded_ids: Iterable[int],
    adjacency_distance: float,
    adjacency_ratio: float,
    relative_distance_std: float,
    motion_threshold: float,
) -> Tuple[Tuple[int, ...], ...]:
    '''Merge persistently touching handles with stable relative geometry.'''
    excluded = {int(value) for value in excluded_ids}
    compatible_pairs = set()

    visibility = {
        object_id: sum(
            object_id in frame.centers_by_id for frame in frames
        )
        for object_id in object_ids
    }
    eligible = [
        object_id for object_id in object_ids if object_id not in excluded
    ]
    for left_index, left_id in enumerate(eligible):
        for right_id in eligible[left_index + 1:]:
            records = []
            for frame in frames:
                if (
                    left_id not in frame.centers_by_id
                    or right_id not in frame.centers_by_id
                ):
                    continue
                left_center = frame.centers_by_id[left_id]
                right_center = frame.centers_by_id[right_id]
                records.append(
                    (
                        left_center,
                        right_center,
                        _bounds_distance(
                            frame.bounds_by_id[left_id],
                            frame.bounds_by_id[right_id],
                        ),
                    )
                )
            required_co_visible = max(
                2,
                int(
                    np.ceil(
                        min(visibility[left_id], visibility[right_id]) * 0.75
                    )
                ),
            )
            if len(records) < required_co_visible:
                continue
            bounds_distances = np.asarray(
                [record[2] for record in records], dtype=np.float32
            )
            if float(np.mean(bounds_distances <= adjacency_distance)) < adjacency_ratio:
                continue
            relative_distances = np.asarray(
                [
                    np.linalg.norm(right_center - left_center)
                    for left_center, right_center, _ in records
                ],
                dtype=np.float32,
            )
            if float(np.std(relative_distances)) > relative_distance_std:
                continue
            shared_motion = 0.0
            for left_record, right_record in zip(records, records[1:]):
                left_delta = right_record[0] - left_record[0]
                right_delta = right_record[1] - left_record[1]
                left_motion = float(np.linalg.norm(left_delta))
                right_motion = float(np.linalg.norm(right_delta))
                if min(left_motion, right_motion) <= 1e-4:
                    continue
                if float(np.linalg.norm(left_delta - right_delta)) > max(
                    relative_distance_std,
                    max(left_motion, right_motion) * 0.5,
                ):
                    continue
                shared_motion += min(left_motion, right_motion)
            tightly_connected_while_static = bool(
                np.max(bounds_distances) <= min(0.005, adjacency_distance)
            )
            if (
                shared_motion < motion_threshold * 0.5
                and not tightly_connected_while_static
            ):
                continue
            compatible_pairs.add((min(left_id, right_id), max(left_id, right_id)))

    # A simulator assembly can be a branching structure: two rack tips may
    # never touch each other, while both remain rigidly attached to one base.
    # Connected components recover that one logical object. Late task-object
    # contacts do not qualify because pair compatibility already requires
    # persistent co-visibility, adjacency and stable relative geometry.
    parent = {object_id: object_id for object_id in object_ids}

    def find(object_id: int) -> int:
        while parent[object_id] != object_id:
            parent[object_id] = parent[parent[object_id]]
            object_id = parent[object_id]
        return object_id

    def union(left_id: int, right_id: int) -> None:
        left_root = find(left_id)
        right_root = find(right_id)
        if left_root == right_root:
            return
        representative = min(left_root, right_root)
        parent[left_root] = representative
        parent[right_root] = representative

    for left_id, right_id in sorted(compatible_pairs):
        union(left_id, right_id)
    groups: Dict[int, set] = {}
    for object_id in object_ids:
        groups.setdefault(find(object_id), set()).add(object_id)
    return tuple(
        tuple(sorted(group))
        for group in sorted(groups.values(), key=lambda value: min(value))
    )


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
    group_adjacency_distance: float = 0.02,
    group_adjacency_ratio: float = 0.80,
    group_relative_distance_std: float = 0.01,
) -> TaskHandleDetection:
    '''Build one stable handle whitelist for an entire demonstration.

    Motion is never required for retention. A moving handle is rejected as
    unexplained only when no sampled current/next gripper pose approaches it.
    This preserves static targets and non-grasp pushing/articulation tasks.
    '''
    if not frames:
        raise ValueError('Task handle detection requires at least one frame')
    if min(
        background_extent,
        motion_threshold,
        adjacency_distance,
        group_adjacency_distance,
        group_relative_distance_std,
    ) <= 0:
        raise ValueError('Task detector distance thresholds must be positive')
    if not 0 < min(adjacency_ratio, group_adjacency_ratio) <= 1:
        raise ValueError('adjacency ratios must be in (0, 1]')

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
    gripper_distance: Dict[Tuple[int, int], float] = {}

    for frame_index, frame in enumerate(frames):
        for object_id, bounds in frame.bounds_by_id.items():
            sizes[object_id].append(bounds[1] - bounds[0])
            centers[object_id].append((frame_index, frame.centers_by_id[object_id]))
            point_support[object_id] += frame.point_counts_by_id.get(object_id, 0)
            current_gripper_distance = _point_to_bounds_distance(
                frame.gripper_position, bounds
            )
            distance = min(
                current_gripper_distance,
                _point_to_bounds_distance(frame.action_position, bounds),
            )
            minimum_distance[object_id] = min(minimum_distance[object_id], distance)
            near_in_frame[(frame_index, object_id)] = distance <= radius
            gripper_distance[(frame_index, object_id)] = current_gripper_distance

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

    object_groups = _detect_rigid_groups(
        all_ids,
        frames,
        excluded_ids=background,
        adjacency_distance=group_adjacency_distance,
        adjacency_ratio=group_adjacency_ratio,
        relative_distance_std=group_relative_distance_std,
        motion_threshold=motion_threshold,
    )
    group_by_handle = {
        handle: min(group)
        for group in object_groups
        for handle in group
    }
    members_by_group = {
        min(group): set(group)
        for group in object_groups
    }
    candidate_groups = {
        group_by_handle[handle]
        for handle in interaction | adjacent
    }
    ranked_groups = sorted(
        candidate_groups,
        key=lambda group_id: (
            not bool(members_by_group[group_id].intersection(interaction)),
            min(minimum_distance[handle] for handle in members_by_group[group_id]),
            -sum(point_support[handle] for handle in members_by_group[group_id]),
            group_id,
        ),
    )
    if not ranked_groups:
        # Never silently emit an empty episode map if all sampled poses are
        # slightly outside the configured radius.
        ranked_groups = sorted(
            (
                group_id
                for group_id, members in members_by_group.items()
                if not members.intersection(background)
            ),
            key=lambda group_id: (
                min(
                    minimum_distance[handle]
                    for handle in members_by_group[group_id]
                ),
                -sum(
                    point_support[handle]
                    for handle in members_by_group[group_id]
                ),
                group_id,
            ),
        )[: min(2, limit)]
    selected_group_ids = tuple(ranked_groups[:limit])
    selected = tuple(sorted(
        handle
        for group_id in selected_group_ids
        for handle in members_by_group[group_id]
    ))
    selected_set = set(selected)

    close_indices = [
        index
        for index in range(1, len(frames))
        if frames[index - 1].gripper_open >= 0.5
        and frames[index].gripper_open < 0.5
    ]
    close_distance = {
        object_id: min(
            (
                min(
                    gripper_distance.get(
                        (index - 1, object_id), float('inf')
                    ),
                    gripper_distance.get((index, object_id), float('inf')),
                )
                for index in close_indices
            ),
            default=float('inf'),
        )
        for object_id in selected
    }
    grasp_contact_radius = min(
        radius,
        max(adjacency_distance, 0.06),
    )
    close_contact = {
        object_id
        for object_id in selected
        if close_distance[object_id] <= grasp_contact_radius
    }
    total_motion: Dict[int, float] = {object_id: 0.0 for object_id in all_ids}
    causal_motion: Dict[int, float] = {object_id: 0.0 for object_id in all_ids}
    closed_follow_motion: Dict[int, float] = {
        object_id: 0.0 for object_id in all_ids
    }
    for object_id, observations in centers.items():
        for (left_index, left), (right_index, right) in zip(
            observations, observations[1:]
        ):
            displacement = float(np.linalg.norm(right - left))
            total_motion[object_id] += displacement
            if displacement <= 1e-4:
                continue
            object_delta = right - left
            gripper_delta = (
                frames[right_index].gripper_position
                - frames[left_index].gripper_position
            )
            gripper_motion = float(np.linalg.norm(gripper_delta))
            near_gripper = min(
                gripper_distance.get((left_index, object_id), float('inf')),
                gripper_distance.get((right_index, object_id), float('inf')),
            ) <= radius
            if near_gripper and gripper_motion > 1e-4:
                alignment = float(
                    np.dot(object_delta, gripper_delta)
                    / (displacement * gripper_motion)
                )
                residual = float(np.linalg.norm(object_delta - gripper_delta))
                if alignment >= 0.5 and residual <= max(
                    adjacency_distance,
                    displacement * 0.75,
                ):
                    causal_motion[object_id] += displacement
            if (
                near_gripper
                and frames[left_index].gripper_open < 0.5
                and frames[right_index].gripper_open < 0.5
            ):
                left_offset = left - frames[left_index].gripper_position
                right_offset = right - frames[right_index].gripper_position
                if float(np.linalg.norm(right_offset - left_offset)) <= max(
                    adjacency_distance, motion_threshold
                ):
                    closed_follow_motion[object_id] += displacement

    target_groups = {
        group_id
        for group_id in selected_group_ids
        if max(
            causal_motion[handle]
            for handle in members_by_group[group_id]
        ) >= motion_threshold
        or (
            bool(members_by_group[group_id].intersection(close_contact))
            and max(
                closed_follow_motion[handle]
                for handle in members_by_group[group_id]
            ) >= motion_threshold
        )
    }
    if not target_groups and selected_group_ids:
        # Static reach/press targets may not exhibit measurable object motion.
        # Retain exactly the strongest directly interacted object group.
        direct_groups = [
            group_id
            for group_id in ranked_groups
            if group_id in selected_group_ids
            and members_by_group[group_id].intersection(interaction)
        ]
        close_groups = [
            group_id
            for group_id in selected_group_ids
            if members_by_group[group_id].intersection(close_contact)
        ]
        fallback = close_groups or direct_groups or list(selected_group_ids)
        target_groups.add(
            min(
                fallback,
                key=lambda group_id: (
                    min(
                        close_distance.get(handle, float('inf'))
                        for handle in members_by_group[group_id]
                    ),
                    min(
                        minimum_distance[handle]
                        for handle in members_by_group[group_id]
                    ),
                    -sum(
                        point_support[handle]
                        for handle in members_by_group[group_id]
                    ),
                    group_id,
                ),
            )
        )

    reference_groups = set()
    if target_groups:
        for candidate_group in set(selected_group_ids) - target_groups:
            candidate_members = members_by_group[candidate_group]
            candidate_motion = max(
                total_motion[handle] for handle in candidate_members
            )
            strongest_target_motion = max(
                (
                    total_motion[handle]
                    for group_id in target_groups
                    for handle in members_by_group[group_id]
                ),
                default=0.0,
            )
            is_static_reference = (
                candidate_motion < motion_threshold
                or (
                    strongest_target_motion >= motion_threshold
                    and candidate_motion <= strongest_target_motion * 0.25
                )
            )
            if not is_static_reference:
                continue
            related = bool(candidate_members.intersection(adjacent))
            for frame in frames:
                for candidate in candidate_members:
                    if candidate not in frame.bounds_by_id:
                        continue
                    for target_group in target_groups:
                        for target_id in members_by_group[target_group]:
                            if target_id not in frame.bounds_by_id:
                                continue
                            if _bounds_distance(
                                frame.bounds_by_id[candidate],
                                frame.bounds_by_id[target_id],
                            ) <= max(adjacency_distance, radius * 0.5):
                                related = True
                                break
                        if related:
                            break
                    if related:
                        break
                if related:
                    break
            if related:
                reference_groups.add(candidate_group)

    target = {
        handle
        for group_id in target_groups
        for handle in members_by_group[group_id]
    }
    reference = {
        handle
        for group_id in reference_groups
        for handle in members_by_group[group_id]
    }

    return TaskHandleDetection(
        episode_idx=int(episode_idx),
        task_handles=selected,
        observed_handles=tuple(all_ids),
        interaction_handles=tuple(sorted(interaction & selected_set)),
        adjacent_handles=tuple(sorted(adjacent & selected_set)),
        rejected_dynamic_handles=tuple(sorted(rejected_dynamic)),
        background_handles=tuple(sorted(background)),
        sampled_frames=tuple(frame.sample_frame for frame in frames),
        target_handles=tuple(sorted(target)),
        reference_handles=tuple(sorted(reference)),
        object_groups=object_groups,
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
        observed_handles=tuple(
            int(value) for value in payload['observed_handles']
        ),
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
        target_handles=tuple(int(value) for value in payload['target_handles']),
        reference_handles=tuple(
            int(value) for value in payload['reference_handles']
        ),
        object_groups=tuple(
            tuple(int(value) for value in group)
            for group in payload['object_groups']
        ),
    )
