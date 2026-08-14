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


def frame(index, gripper, action, centers):
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
        self.assertNotIn(11, detection.rejected_dynamic_handles)

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
