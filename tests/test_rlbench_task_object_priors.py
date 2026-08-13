import unittest

import numpy as np

from tools.rlbench_task_object_priors import (
    TASK_OBJECT_PRIORS,
    select_task_relevant_instances,
)


class RLBenchTaskObjectPriorsTest(unittest.TestCase):
    def test_all_bridgevla_tasks_have_priors(self):
        self.assertEqual(len(TASK_OBJECT_PRIORS), 18)

    def test_selects_near_action_and_rejects_large_planar_background(self):
        near = np.array(
            [[0.00, 0.00, 0.80], [0.04, 0.04, 0.84]], dtype=np.float32
        )
        far = near + np.array([0.50, 0.0, 0.0], dtype=np.float32)
        table = np.array(
            [[-0.5, -0.5, 0.7], [0.5, 0.5, 0.72]], dtype=np.float32
        )
        selected = select_task_relevant_instances(
            'stack_blocks',
            [(10, table), (11, far), (12, near)],
            np.array([0.02, 0.02, 0.82]),
            strict_action_filter=True,
        )
        self.assertEqual([object_id for object_id, _ in selected], [12])

    def test_default_keeps_far_objects_and_uses_distance_only_for_ranking(self):
        near = np.array(
            [[0.00, 0.00, 0.80], [0.04, 0.04, 0.84]], dtype=np.float32
        )
        far = near + np.array([0.50, 0.0, 0.0], dtype=np.float32)
        selected = select_task_relevant_instances(
            'stack_blocks',
            [(11, far), (12, near)],
            np.array([0.02, 0.02, 0.82]),
        )
        self.assertEqual([object_id for object_id, _ in selected], [12, 11])

    def test_nearest_fallback_avoids_empty_result(self):
        points = np.array(
            [[0.4, 0.0, 0.8], [0.42, 0.02, 0.82]], dtype=np.float32
        )
        selected = select_task_relevant_instances(
            'push_buttons',
            [(5, points)],
            np.array([0.0, 0.0, 0.8]),
            interaction_radius=0.01,
        )
        self.assertEqual([object_id for object_id, _ in selected], [5])


if __name__ == '__main__':
    unittest.main()
