"""Reporting helpers without simulator/model dependencies."""

import math
from numbers import Real


MANIFEST_FIELDS = [
    'task', 'generated coverage', 'generated episodes',
    'requested episodes', 'logical transitions',
]


def manifest_result(task, generated, requested, logical_transitions):
    if requested <= 0 or not 0 <= generated <= requested:
        raise ValueError('Manifest reporting requires 0 <= generated <= requested and requested > 0')
    return dict(zip(MANIFEST_FIELDS, (
        task, 100.0 * generated / requested, generated, requested, logical_transitions)))


def numeric_task_scores(scores):
    # Missing metrics are not zero success and must not reach add_scalar.
    return {task: float(value) for task, value in scores.items()
            if isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(float(value))}
