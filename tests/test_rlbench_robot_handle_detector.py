import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.rlbench_robot_handle_detector import (
    RobotFrameEvidence,
    detect_robot_handles,
    load_robot_handle_detection,
    save_robot_handle_detection,
)


def bounds(center, half_extent=0.01):
    center = np.asarray(center, dtype=np.float32)
    return center - half_extent, center + half_extent


class RLBenchRobotHandleDetectorTest(unittest.TestCase):
    def test_temporal_detection_separates_robot_from_late_grasped_object(self):
        frames = []
        for frame_index in range(4):
            gripper = np.array(
                [frame_index * 0.05, 0.0, 0.8], dtype=np.float32
            )
            centers = {
                10: gripper,
                11: gripper + np.array([0.03, 0.0, 0.0]),
                12: gripper - np.array([0.07, 0.0, 0.0]),
                20: (
                    np.array([0.50, 0.0, 0.8], dtype=np.float32)
                    if frame_index < 2
                    else gripper + np.array([0.04, 0.0, 0.0])
                ),
                30: np.array([0.0, 0.5, 0.7], dtype=np.float32),
            }
            frames.append(
                RobotFrameEvidence(
                    sample_frame=frame_index,
                    gripper_position=gripper,
                    bounds_by_id={
                        object_id: bounds(center, 0.015)
                        for object_id, center in centers.items()
                    },
                    centers_by_id=centers,
                    wrist_centroids_by_id={
                        10: np.array([0.50, 0.45]),
                        11: np.array([0.50, 0.55]),
                        20: np.array([0.15 + frame_index * 0.2, 0.20]),
                    },
                )
            )

        detection = detect_robot_handles(3, frames)
        self.assertEqual(detection.gripper_handles, (10, 11))
        self.assertIn(12, detection.arm_handles)
        self.assertNotIn(20, detection.robot_handles)
        self.assertNotIn(30, detection.robot_handles)

    def test_json_cache_round_trip(self):
        frames = [
            RobotFrameEvidence(
                sample_frame=index,
                gripper_position=np.array([0.0, 0.0, 0.8]),
                bounds_by_id={5: bounds([0.0, 0.0, 0.8])},
                centers_by_id={5: np.array([0.0, 0.0, 0.8])},
                wrist_centroids_by_id={5: np.array([0.5, 0.5])},
            )
            for index in range(2)
        ]
        detection = detect_robot_handles(7, frames)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'episode_0007.json'
            save_robot_handle_detection(path, detection)
            loaded = load_robot_handle_detection(path)
        self.assertEqual(loaded, detection)


if __name__ == '__main__':
    unittest.main()
