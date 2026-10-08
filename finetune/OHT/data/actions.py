"""Labels use reached future TCP poses, never the unreliable EE action columns."""
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import array, quaternion, check_bounds, rotation_error, tcp_pose


def gripper_states(states, commands):
    states = array(states, name="states")
    commands = array(commands, (len(states),), "gripper commands")
    if states.ndim != 2 or states.shape[1] != 7 or not np.isin(commands, [-1, 0, 1]).all():
        raise ValueError("Expected state[N,7] and gripper commands in {-1,0,1}")
    measured = np.clip((states[:, 6] + 0.040) / 0.034, 0, 1)
    desired = np.empty(len(states), dtype=np.int64)
    current = int(measured[0] >= 0.5)
    for i, command in enumerate(commands):
        if command:
            current = int(command == -1)
        desired[i] = current
    return measured.astype(np.float32), desired


def low_dim(measured, finger_joints):
    fingers = array(finger_joints, (2,), "finger joints")
    # Compatibility features in [0,.04], not a claim of physical metre opening.
    opening = np.clip((fingers + 0.020) / 0.017, 0, 1) * 0.04
    return np.asarray([float(measured >= 0.5), *opening, 0], dtype=np.float32)


def world_tcp_poses(columns, link_to_tcp):
    positions = array(columns["observation.ee_pos_world"], name="world EE positions")
    orientations = quaternion(columns["observation.ee_quat_world"])
    return np.asarray([tcp_pose(p, q, link_to_tcp) for p, q in zip(positions, orientations)])


def keypoints(poses, desired_gripper, instruction_ids, max_translation=0.04,
              max_rotation_degrees=8, max_frames=30):
    poses = array(poses, name="TCP poses")
    n = len(poses)
    if n < 2 or len(desired_gripper) != n or len(instruction_ids) != n:
        raise ValueError("An episode requires at least two aligned frames")
    if max_translation <= 0 or max_rotation_degrees <= 0 or max_frames < 1:
        raise ValueError("Keypoint thresholds must be positive")
    boundaries = {n - 1}
    for i in range(1, n):
        if desired_gripper[i] != desired_gripper[i - 1] or instruction_ids[i] != instruction_ids[i - 1]:
            boundaries.update((max(1, i - 1), i))
    result, last = [], 0
    for i in range(1, n):
        moved = np.linalg.norm(poses[i, :3] - poses[last, :3]) >= max_translation
        rotated = rotation_error(poses[last, 3:], poses[i, 3:]) >= np.deg2rad(max_rotation_degrees)
        if i in boundaries or moved or rotated or i - last >= max_frames:
            result.append(i)
            last = i
    return result


def target_labels(pose, gripper_open, bounds, rotation_classes=72, voxel_size=100):
    pose = array(pose, (7,), "TCP target").copy()
    pose[3:] = quaternion(pose[3:])
    if pose[6] < 0:
        pose[3:] *= -1
    bounds = check_bounds(pose[:3], bounds)
    if rotation_classes < 1 or voxel_size < 1 or gripper_open not in (0, 1):
        raise ValueError("Invalid action discretization")
    degrees = Rotation.from_quat(pose[3:]).as_euler("xyz", degrees=True)
    bins = np.round((degrees + 180) / (360 / rotation_classes)).astype(np.int32) % rotation_classes
    indices = np.floor((pose[:3] - bounds[:3]) / (bounds[3:] - bounds[:3]) * voxel_size).astype(np.int32)
    return {
        "action": np.r_[pose, gripper_open].astype(np.float32),
        "gripper_pose": pose.astype(np.float32),
        "rot_grip_action_indicies": np.r_[bins, gripper_open].astype(np.int32),
        "trans_action_indicies": indices,
        "ignore_collisions": np.zeros(1, dtype=np.int32),
    }
