import json
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

    def test_late_grasped_object_cannot_become_gripper_seed(self):
        frames = []
        for frame_index in range(8):
            gripper = np.array(
                [frame_index * 0.04, 0.0, 0.8], dtype=np.float32
            )
            manipulated = (
                np.array([0.55, 0.0, 0.8], dtype=np.float32)
                if frame_index < 2
                else gripper + np.array([0.03, 0.0, 0.0])
            )
            centers = {10: gripper, 20: manipulated}
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
                        10: np.array([0.5, 0.5]),
                        # It looks wrist-stable after being grasped, but was
                        # not physically beside the gripper at the start.
                        20: np.array([0.2, 0.2]),
                    },
                )
            )

        detection = detect_robot_handles(4, frames)
        self.assertEqual(detection.gripper_handles, (10,))
        self.assertNotIn(20, detection.robot_handles)

    def test_gripper_seed_tolerates_rotation_and_one_distance_outlier(self):
        frames = []
        for frame_index in range(5):
            gripper = np.array(
                [frame_index * 0.04, 0.0, 0.8], dtype=np.float32
            )
            offset = (
                np.array([0.16, 0.0, 0.0], dtype=np.float32)
                if frame_index == 4
                else np.zeros(3, dtype=np.float32)
            )
            center = gripper + offset
            frames.append(
                RobotFrameEvidence(
                    sample_frame=frame_index,
                    gripper_position=gripper,
                    bounds_by_id={10: bounds(center, 0.01)},
                    centers_by_id={10: center},
                    wrist_centroids_by_id={10: np.array([0.5, 0.5])},
                )
            )

        detection = detect_robot_handles(8, frames)
        self.assertEqual(detection.gripper_handles, (10,))

    def test_gripper_handle_may_first_appear_at_raw_frame_ten(self):
        frames = []
        for frame_index in range(8):
            gripper = np.array(
                [frame_index * 0.03, 0.0, 0.8], dtype=np.float32
            )
            centers = {10: gripper}
            wrist_centroids = {10: np.array([0.45, 0.5])}
            if frame_index >= 2:
                centers[11] = gripper + np.array(
                    [0.03, 0.0, 0.0], dtype=np.float32
                )
                wrist_centroids[11] = np.array([0.55, 0.5])
            frames.append(
                RobotFrameEvidence(
                    # With the default raw stride, evidence index 2 is frame 10.
                    sample_frame=frame_index * 5,
                    gripper_position=gripper,
                    bounds_by_id={
                        object_id: bounds(center, 0.015)
                        for object_id, center in centers.items()
                    },
                    centers_by_id=centers,
                    wrist_centroids_by_id=wrist_centroids,
                )
            )

        detection = detect_robot_handles(16, frames)
        self.assertEqual(detection.gripper_handles, (10, 11))

    def test_low_confidence_wrist_candidate_recovers_gripper_seed(self):
        frames = []
        for frame_index in range(5):
            gripper = np.array(
                [frame_index * 0.04, 0.0, 0.8], dtype=np.float32
            )
            center = gripper + np.array(
                [0.0, 0.13 if frame_index % 2 == 0 else -0.13, 0.0],
                dtype=np.float32,
            )
            frames.append(
                RobotFrameEvidence(
                    sample_frame=frame_index,
                    gripper_position=gripper,
                    bounds_by_id={10: bounds(center, 0.01)},
                    centers_by_id={10: center},
                    wrist_centroids_by_id={10: np.array([0.5, 0.5])},
                )
            )

        detection = detect_robot_handles(11, frames)
        self.assertEqual(detection.gripper_handles, (10,))

    def test_kinematic_chain_reaches_static_robot_base(self):
        frames = []
        for frame_index in range(6):
            gripper = np.array(
                [frame_index * 0.02, 0.0, 0.8], dtype=np.float32
            )
            moving_link = gripper + np.array([0.04, 0.0, 0.0])
            static_base = np.array([0.16, 0.0, 0.8], dtype=np.float32)
            frames.append(
                RobotFrameEvidence(
                    sample_frame=frame_index,
                    gripper_position=gripper,
                    bounds_by_id={
                        10: bounds(gripper, 0.015),
                        11: bounds(moving_link, 0.015),
                        12: bounds(static_base, 0.08),
                    },
                    centers_by_id={
                        10: gripper,
                        11: moving_link,
                        12: static_base,
                    },
                    wrist_centroids_by_id={10: np.array([0.5, 0.5])},
                )
            )

        detection = detect_robot_handles(12, frames)
        self.assertEqual(detection.gripper_handles, (10,))
        self.assertEqual(detection.arm_handles, (11, 12))

    def test_meaningful_slow_arm_motion_is_not_rejected_by_ratio_alone(self):
        frames = []
        gripper_positions = (0.0, 0.05, 0.10)
        link_positions = (0.04, 0.05, 0.06)
        for frame_index, (gripper_x, link_x) in enumerate(
            zip(gripper_positions, link_positions)
        ):
            gripper = np.array([gripper_x, 0.0, 0.8], dtype=np.float32)
            link = np.array([link_x, 0.0, 0.8], dtype=np.float32)
            frames.append(
                RobotFrameEvidence(
                    sample_frame=frame_index,
                    gripper_position=gripper,
                    bounds_by_id={
                        10: bounds(gripper, 0.015),
                        11: bounds(link, 0.015),
                    },
                    centers_by_id={10: gripper, 11: link},
                    wrist_centroids_by_id={10: np.array([0.5, 0.5])},
                )
            )

        detection = detect_robot_handles(13, frames)
        self.assertEqual(detection.gripper_handles, (10,))
        self.assertEqual(detection.arm_handles, (11,))

    def test_link_motion_threshold_can_recover_low_amplitude_arm_link(self):
        frames = []
        gripper_positions = (0.0, 0.05, 0.10)
        link_positions = (0.04, 0.043, 0.046)
        for frame_index, (gripper_x, link_x) in enumerate(
            zip(gripper_positions, link_positions)
        ):
            gripper = np.array([gripper_x, 0.0, 0.8], dtype=np.float32)
            link = np.array([link_x, 0.0, 0.8], dtype=np.float32)
            frames.append(
                RobotFrameEvidence(
                    sample_frame=frame_index,
                    gripper_position=gripper,
                    bounds_by_id={
                        10: bounds(gripper, 0.02),
                        11: bounds(link, 0.02),
                    },
                    centers_by_id={10: gripper, 11: link},
                    wrist_centroids_by_id={10: np.array([0.5, 0.5])},
                )
            )

        conservative = detect_robot_handles(14, frames)
        relaxed = detect_robot_handles(
            14, frames, min_link_motion=0.002
        )
        self.assertNotIn(11, conservative.arm_handles)
        self.assertIn(11, relaxed.arm_handles)

    def test_adjacency_distance_can_bridge_segmented_robot_link_gap(self):
        frames = []
        for frame_index in range(3):
            gripper = np.array(
                [frame_index * 0.03, 0.0, 0.8], dtype=np.float32
            )
            link = gripper + np.array([0.09, 0.0, 0.0], dtype=np.float32)
            frames.append(
                RobotFrameEvidence(
                    sample_frame=frame_index,
                    gripper_position=gripper,
                    bounds_by_id={
                        10: bounds(gripper, 0.01),
                        11: bounds(link, 0.01),
                    },
                    centers_by_id={10: gripper, 11: link},
                    wrist_centroids_by_id={10: np.array([0.5, 0.5])},
                )
            )

        conservative = detect_robot_handles(15, frames)
        relaxed = detect_robot_handles(
            15, frames, adjacency_distance=0.08
        )
        self.assertNotIn(11, conservative.arm_handles)
        self.assertIn(11, relaxed.arm_handles)

    def test_late_adjacency_cannot_expand_robot_chain(self):
        frames = []
        for frame_index in range(8):
            gripper = np.array(
                [frame_index * 0.04, 0.0, 0.8], dtype=np.float32
            )
            manipulated = (
                np.array([0.55, 0.0, 0.8], dtype=np.float32)
                if frame_index < 2
                else gripper + np.array([0.03, 0.0, 0.0])
            )
            centers = {10: gripper, 20: manipulated}
            frames.append(
                RobotFrameEvidence(
                    sample_frame=frame_index,
                    gripper_position=gripper,
                    bounds_by_id={
                        object_id: bounds(center, 0.015)
                        for object_id, center in centers.items()
                    },
                    centers_by_id=centers,
                    wrist_centroids_by_id={10: np.array([0.5, 0.5])},
                    gripper_open=1.0 if frame_index < 3 else 0.0,
                )
            )

        detection = detect_robot_handles(5, frames)
        self.assertEqual(detection.gripper_handles, (10,))
        self.assertNotIn(20, detection.arm_handles)

    def test_static_early_neighbor_is_ambiguous_not_hard_excluded(self):
        frames = []
        for frame_index in range(8):
            gripper = np.array(
                [frame_index * 0.04, 0.0, 0.8], dtype=np.float32
            )
            neighbor = (
                np.array([0.04, 0.0, 0.8], dtype=np.float32)
                if frame_index < 3
                else gripper + np.array([0.04, 0.0, 0.0])
            )
            centers = {10: gripper, 20: neighbor}
            frames.append(
                RobotFrameEvidence(
                    sample_frame=frame_index,
                    gripper_position=gripper,
                    bounds_by_id={
                        object_id: bounds(center, 0.015)
                        for object_id, center in centers.items()
                    },
                    centers_by_id=centers,
                    wrist_centroids_by_id={10: np.array([0.5, 0.5])},
                    gripper_open=1.0 if frame_index < 3 else 0.0,
                )
            )

        detection = detect_robot_handles(6, frames)
        self.assertEqual(detection.gripper_handles, (10,))
        self.assertNotIn(20, detection.robot_handles)
        self.assertIn(20, detection.ambiguous_handles)

    def test_object_that_follows_after_gripper_closes_is_never_deleted(self):
        frames = []
        for frame_index in range(6):
            gripper = np.array(
                [frame_index * 0.05, 0.0, 0.8], dtype=np.float32
            )
            if frame_index < 3:
                manipulated = np.array(
                    [0.06 + frame_index * 0.005, 0.0, 0.8],
                    dtype=np.float32,
                )
            else:
                manipulated = gripper + np.array([0.01, 0.0, 0.0])
            centers = {10: gripper, 20: manipulated}
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
                        10: np.array([0.5, 0.5]),
                        20: np.array([0.1 + frame_index * 0.1, 0.2]),
                    },
                    gripper_open=1.0 if frame_index < 3 else 0.0,
                )
            )

        detection = detect_robot_handles(9, frames)
        self.assertEqual(detection.gripper_handles, (10,))
        self.assertNotIn(20, detection.robot_handles)
        self.assertIn(20, detection.grasped_handles)
        self.assertIn(20, detection.ambiguous_handles)

    def test_close_event_protects_object_without_post_close_motion(self):
        frames = []
        for frame_index in range(4):
            gripper = np.array(
                [frame_index * 0.04, 0.0, 0.8], dtype=np.float32
            )
            object_center = np.array([0.13, 0.0, 0.8], dtype=np.float32)
            centers = {10: gripper, 20: object_center}
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
                        10: np.array([0.5, 0.5]),
                        20: np.array([0.5, 0.5]),
                    },
                    gripper_open=1.0 if frame_index < 3 else 0.0,
                )
            )

        detection = detect_robot_handles(10, frames)
        self.assertEqual(detection.gripper_handles, (10,))
        self.assertNotIn(20, detection.robot_handles)
        self.assertIn(20, detection.grasped_handles)

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

    def test_json_cache_rejects_changed_sampling_configuration(self):
        frames = [
            RobotFrameEvidence(
                sample_frame=0,
                gripper_position=np.array([0.0, 0.0, 0.8]),
                bounds_by_id={5: bounds([0.0, 0.0, 0.8])},
                centers_by_id={5: np.array([0.0, 0.0, 0.8])},
                wrist_centroids_by_id={5: np.array([0.5, 0.5])},
            )
        ]
        detection = detect_robot_handles(7, frames)
        config = {
            'source': 'raw_depth',
            'stride': 5,
            'initial_window': 100,
            'max_frames': 64,
            'motion_threshold': 0.02,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'episode_0007.json'
            save_robot_handle_detection(path, detection, config)
            loaded = load_robot_handle_detection(path, config)
            self.assertEqual(loaded, detection)
            changed = dict(config, stride=10)
            with self.assertRaisesRegex(
                ValueError, 'sampling configuration'
            ):
                load_robot_handle_detection(path, changed)

    def test_old_cache_method_is_rejected(self):
        payload = {
            'episode_idx': 7,
            'gripper_handles': [5],
            'arm_handles': [],
            'robot_handles': [5],
            'confidence': {'5': 0.9},
            'sampled_frames': [0, 1],
            'method': 'wrist_pose_temporal_adjacency_v5_balanced',
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'episode_0007.json'
            path.write_text(json.dumps(payload), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'Stale robot-handle cache'):
                load_robot_handle_detection(path)


if __name__ == '__main__':
    unittest.main()
