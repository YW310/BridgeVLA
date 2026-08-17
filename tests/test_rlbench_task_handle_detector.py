import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.rlbench_task_handle_detector import (
    TaskFrameEvidence,
    detect_task_handles,
    load_task_handle_detection,
    save_task_handle_detection,
)


def bounds(center, half_extent=0.01):
    center = np.asarray(center, dtype=np.float32)
    return center - half_extent, center + half_extent


def frame(index, gripper, action, centers, gripper_open=1.0):
    return TaskFrameEvidence(
        sample_frame=index,
        gripper_position=np.asarray(gripper, dtype=np.float32),
        action_position=np.asarray(action, dtype=np.float32),
        bounds_by_id={key: bounds(value) for key, value in centers.items()},
        centers_by_id={
            key: np.asarray(value, dtype=np.float32)
            for key, value in centers.items()
        },
        point_counts_by_id={key: 20 for key in centers},
        gripper_open=gripper_open,
    )


class RLBenchTaskHandleDetectorTest(unittest.TestCase):
    def test_keeps_static_future_target_and_rejects_unexplained_motion(self):
        frames = []
        for index in range(4):
            frames.append(
                frame(
                    index,
                    gripper=[index * 0.05, 0.0, 0.8],
                    action=[0.30, 0.0, 0.8],
                    centers={
                        10: [0.30, 0.0, 0.8],  # static placement target
                        20: [0.70, index * 0.05, 0.8],  # moving, no interaction
                        30: [0.80, 0.30, 0.8],  # unrelated static object
                    },
                )
            )

        detection = detect_task_handles('stack_blocks', 2, frames)
        self.assertIn(10, detection.task_handles)
        self.assertNotIn(20, detection.task_handles)
        self.assertIn(20, detection.rejected_dynamic_handles)
        self.assertNotIn(30, detection.task_handles)
        self.assertEqual(detection.observed_handles, (10, 20, 30))
        self.assertEqual(detection.slot_handles, (10, 20, 30))

    def test_keeps_ungrasped_object_moved_by_nearby_action(self):
        frames = []
        for index in range(4):
            center = [0.10 + index * 0.04, 0.0, 0.8]
            frames.append(
                frame(
                    index,
                    gripper=[center[0] - 0.03, 0.0, 0.8],
                    action=center,
                    centers={11: center},
                )
            )

        detection = detect_task_handles(
            'slide_block_to_color_target', 3, frames
        )
        self.assertEqual(detection.task_handles, (11,))
        self.assertEqual(detection.target_handles, (11,))
        self.assertEqual(detection.reference_handles, ())
        self.assertNotIn(11, detection.rejected_dynamic_handles)

    def test_close_and_follow_marks_target_and_static_contact_as_reference(self):
        frames = []
        grippers = [0.00, 0.00, 0.05, 0.10, 0.10]
        openings = [1.0, 0.0, 0.0, 0.0, 1.0]
        targets = [0.01, 0.01, 0.06, 0.11, 0.11]
        for index, (gripper_x, gripper_open, target_x) in enumerate(
            zip(grippers, openings, targets)
        ):
            frames.append(
                frame(
                    index,
                    gripper=[gripper_x, 0.0, 0.8],
                    action=[gripper_x, 0.0, 0.8],
                    centers={
                        10: [target_x, 0.0, 0.8],
                        20: [0.055, 0.0, 0.8],
                    },
                    gripper_open=gripper_open,
                )
            )

        detection = detect_task_handles(
            'stack_blocks',
            6,
            frames,
            interaction_radius=0.08,
            adjacency_distance=0.05,
        )
        self.assertIn(10, detection.target_handles)
        self.assertIn(20, detection.reference_handles)
        self.assertEqual(detection.role_by_handle[10], 1)
        self.assertEqual(detection.role_by_handle[20], 2)

    def test_reference_is_optional(self):
        frames = [
            frame(
                index,
                gripper=[index * 0.04, 0.0, 0.8],
                action=[index * 0.04, 0.0, 0.8],
                centers={10: [index * 0.04 + 0.01, 0.0, 0.8]},
            )
            for index in range(4)
        ]
        detection = detect_task_handles(
            'slide_block_to_color_target', 7, frames
        )
        self.assertEqual(detection.target_handles, (10,))
        self.assertEqual(detection.reference_handles, ())

    def test_nearby_independent_motion_does_not_make_every_object_target(self):
        frames = []
        for index in range(4):
            gripper_x = index * 0.04
            frames.append(
                frame(
                    index,
                    gripper=[gripper_x, 0.0, 0.8],
                    action=[gripper_x, 0.0, 0.8],
                    centers={
                        10: [gripper_x + 0.01, 0.0, 0.8],
                        20: [0.06, index * 0.04 + 0.06, 0.8],
                    },
                )
            )
        detection = detect_task_handles(
            'stack_blocks',
            10,
            frames,
            interaction_radius=0.20,
        )
        self.assertEqual(detection.target_handles, (10,))
        self.assertNotIn(20, detection.target_handles)

    def test_persistent_rigid_neighbors_merge_and_share_target_role(self):
        frames = []
        for index in range(5):
            target_x = index * 0.04
            frames.append(
                frame(
                    index,
                    gripper=[target_x, 0.0, 0.8],
                    action=[target_x, 0.0, 0.8],
                    centers={
                        10: [target_x, 0.0, 0.8],
                        11: [target_x + 0.025, 0.0, 0.8],
                    },
                )
            )
        detection = detect_task_handles(
            'slide_block_to_color_target',
            8,
            frames,
            interaction_radius=0.01,
        )
        self.assertIn((10, 11), detection.object_groups)
        self.assertEqual(detection.group_by_handle[11], 10)
        self.assertEqual(detection.grouped_slot_handles, (10,))
        self.assertEqual(detection.target_handles, (10, 11))
        self.assertEqual(detection.role_by_group, {10: 1})

    def test_handles_that_only_touch_late_do_not_merge(self):
        frames = []
        for index in range(5):
            target_x = index * 0.04
            frames.append(
                frame(
                    index,
                    gripper=[target_x, 0.0, 0.8],
                    action=[target_x, 0.0, 0.8],
                    centers={
                        10: [target_x, 0.0, 0.8],
                        20: [0.18, 0.0, 0.8],
                    },
                )
            )
        detection = detect_task_handles(
            'stack_blocks',
            9,
            frames,
            interaction_radius=0.05,
        )
        self.assertNotEqual(
            detection.group_by_handle[10],
            detection.group_by_handle[20],
        )

    def test_static_touching_handles_merge_before_role_assignment(self):
        frames = [
            frame(
                index,
                gripper=[0.0, 0.0, 0.8],
                action=[0.0, 0.0, 0.8],
                centers={10: [0.0, 0.0, 0.8], 20: [0.02, 0.0, 0.8]},
            )
            for index in range(4)
        ]
        detection = detect_task_handles(
            'stack_blocks', 11, frames, interaction_radius=0.03
        )
        self.assertEqual(
            detection.group_by_handle[10],
            detection.group_by_handle[20],
        )
        self.assertEqual(detection.target_handles, (10, 20))
        self.assertEqual(detection.role_by_group, {10: 1})

    def test_static_structural_chain_merges_as_one_assembly(self):
        frames = [
            frame(
                index,
                gripper=[0.0, 0.0, 0.8],
                action=[0.0, 0.0, 0.8],
                centers={
                    10: [0.0, 0.0, 0.8],
                    20: [0.02, 0.0, 0.8],
                    30: [0.04, 0.0, 0.8],
                },
            )
            for index in range(4)
        ]
        detection = detect_task_handles(
            'stack_blocks', 12, frames, interaction_radius=0.05
        )
        self.assertEqual(detection.group_by_handle[10], 10)
        self.assertEqual(detection.group_by_handle[20], 10)
        self.assertEqual(detection.group_by_handle[30], 10)
        self.assertIn((10, 20, 30), detection.object_groups)

    def test_place_cups_merges_separated_static_rack_regions(self):
        frames = [
            frame(
                index,
                gripper=[0.0, -0.10, 0.8],
                action=[0.0, -0.10, 0.8],
                centers={
                    10: [0.00, 0.0, 0.8],
                    20: [0.06, 0.0, 0.8],
                    30: [0.12, 0.0, 0.8],
                },
            )
            for index in range(4)
        ]
        rack = detect_task_handles('place_cups', 13, frames)
        ordinary = detect_task_handles('stack_blocks', 13, frames)
        self.assertIn((10, 20, 30), rack.object_groups)
        self.assertNotEqual(
            ordinary.group_by_handle[10],
            ordinary.group_by_handle[20],
        )

    def test_keeps_static_handle_persistently_adjacent_to_interaction_seed(self):
        frames = [
            frame(
                index,
                gripper=[0.0, 0.0, 0.8],
                action=[0.0, 0.0, 0.8],
                centers={10: [0.0, 0.0, 0.8], 12: [0.035, 0.0, 0.8]},
            )
            for index in range(3)
        ]
        detection = detect_task_handles(
            'stack_blocks', 4, frames, interaction_radius=0.02
        )
        self.assertEqual(detection.interaction_handles, (10,))
        self.assertEqual(detection.adjacent_handles, (12,))
        self.assertEqual(detection.task_handles, (10, 12))

    def test_json_cache_round_trip(self):
        detection = detect_task_handles(
            'stack_blocks',
            5,
            [
                frame(
                    index,
                    gripper=[0.0, 0.0, 0.8],
                    action=[0.0, 0.0, 0.8],
                    centers={10: [0.0, 0.0, 0.8]},
                )
                for index in range(3)
            ],
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'episode_0005.json'
            save_task_handle_detection(path, detection)
            loaded = load_task_handle_detection(path)
        self.assertEqual(loaded, detection)

    def test_old_cache_method_is_rejected(self):
        payload = {
            'episode_idx': 5,
            'task_handles': [10],
            'interaction_handles': [10],
            'adjacent_handles': [],
            'rejected_dynamic_handles': [],
            'background_handles': [],
            'sampled_frames': [0, 5],
            'method': 'episode_action_trajectory_v2',
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'episode_0005.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Stale task-handle cache'):
                load_task_handle_detection(path)


if __name__ == '__main__':
    unittest.main()
