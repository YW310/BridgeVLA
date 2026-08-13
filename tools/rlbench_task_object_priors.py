'''Task priors for filtering offline RLBench GT-handle point clouds.

This module deliberately uses only information already present in a BridgeVLA
replay transition. It does not require Qwen, SAM, simulator reloads, or object
names. A prior limits GT handles to the neighborhood of the supervised next
keyframe action while rejecting obvious large planar background geometry.
'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


InstancePoints = Tuple[int, np.ndarray]


@dataclass(frozen=True)
class TaskObjectPrior:
    action_family: str
    interaction_radius: float
    max_instances: int


# max_instances counts simulator shape handles, not semantic/logical objects.
TASK_OBJECT_PRIORS: Dict[str, TaskObjectPrior] = {
    'close_jar': TaskObjectPrior('articulate', 0.16, 8),
    'reach_and_drag': TaskObjectPrior('tool_use', 0.25, 10),
    'insert_onto_square_peg': TaskObjectPrior('insert', 0.18, 8),
    'meat_off_grill': TaskObjectPrior('pick', 0.20, 8),
    'open_drawer': TaskObjectPrior('articulate', 0.16, 8),
    'place_cups': TaskObjectPrior('multi_place', 0.22, 12),
    'place_wine_at_rack_location': TaskObjectPrior('place', 0.22, 10),
    'push_buttons': TaskObjectPrior('press', 0.12, 8),
    'put_groceries_in_cupboard': TaskObjectPrior('multi_place', 0.25, 16),
    'put_item_in_drawer': TaskObjectPrior('place', 0.22, 10),
    'put_money_in_safe': TaskObjectPrior('place', 0.22, 10),
    'light_bulb_in': TaskObjectPrior('insert', 0.18, 8),
    'slide_block_to_color_target': TaskObjectPrior('push', 0.22, 8),
    'place_shape_in_shape_sorter': TaskObjectPrior('insert', 0.20, 10),
    'stack_blocks': TaskObjectPrior('multi_place', 0.22, 12),
    'stack_cups': TaskObjectPrior('multi_place', 0.22, 10),
    'sweep_to_dustpan_of_size': TaskObjectPrior('tool_use', 0.30, 12),
    'turn_tap': TaskObjectPrior('articulate', 0.15, 8),
}


def get_task_object_prior(task_name: str) -> TaskObjectPrior:
    try:
        return TASK_OBJECT_PRIORS[task_name]
    except KeyError as exc:
        known = ', '.join(sorted(TASK_OBJECT_PRIORS))
        raise ValueError(
            f'No task-object prior for {task_name!r}; known tasks: {known}'
        ) from exc


def _distance_to_aabb(
    point: np.ndarray, minimum: np.ndarray, maximum: np.ndarray
) -> float:
    outside = np.maximum(np.maximum(minimum - point, point - maximum), 0.0)
    return float(np.linalg.norm(outside))


def _is_obvious_planar_background(
    size: np.ndarray, background_extent: float
) -> bool:
    # Tables/floors cover two large axes. Do not reject cabinets/racks merely
    # because one axis is long; those may be task references.
    ordered = np.sort(size)
    return bool(
        ordered[-1] >= background_extent
        and ordered[-2] >= background_extent
    )


def select_task_relevant_instances(
    task_name: str,
    instances: Sequence[InstancePoints],
    action_position: np.ndarray,
    *,
    interaction_radius: Optional[float] = None,
    max_instances: Optional[int] = None,
    background_extent: float = 0.60,
) -> List[InstancePoints]:
    '''Rank GT handles using a task-specific next-action spatial prior.

    The returned list contains only existing handles and preserves their full
    fused point clouds. Robot handles must be supplied to the caller's exact
    exclusion list because offline mask IDs contain no semantic class names.
    '''
    prior = get_task_object_prior(task_name)
    radius = (
        prior.interaction_radius
        if interaction_radius is None
        else float(interaction_radius)
    )
    limit = prior.max_instances if max_instances is None else int(max_instances)
    if radius <= 0 or limit <= 0 or background_extent <= 0:
        raise ValueError(
            'task-prior radius, max instances, and background extent '
            'must be positive'
        )
    action_position = np.asarray(action_position, dtype=np.float64).reshape(-1)
    if action_position.size < 3 or not np.isfinite(action_position[:3]).all():
        raise ValueError('action_position must contain three finite values')
    action_position = action_position[:3]

    ranked = []
    for object_id, object_points in instances:
        points = np.asarray(object_points)
        if not points.size:
            continue
        minimum = np.min(points, axis=0)
        maximum = np.max(points, axis=0)
        size = maximum - minimum
        if _is_obvious_planar_background(size, background_extent):
            continue
        distance = _distance_to_aabb(action_position, minimum, maximum)
        # Prefer handles intersecting the action neighborhood. Point support is
        # a stable tie breaker; it is not the primary relevance criterion.
        ranked.append((distance, -len(points), int(object_id), object_points))

    ranked.sort(key=lambda item: item[:3])
    nearby = [item for item in ranked if item[0] <= radius]
    if not nearby:
        # Keep a small diagnostic fallback instead of silently producing an
        # empty Oracle tensor when calibration is slightly outside the radius.
        nearby = ranked[: min(2, limit)]
    return [
        (object_id, points)
        for _, _, object_id, points in nearby[:limit]
    ]

