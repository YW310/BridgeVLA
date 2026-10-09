"""Default OHT workspace covers the v423 description, not measured full data."""
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from finetune.OHT.data.actions import world_tcp_poses
from finetune.OHT.data.geometry import check_bounds, transform_matrix
from finetune.OHT.data.observation import camera_observation, validate_data_config
from finetune.OHT.data.point_filter import point_cloud_mask
from finetune.OHT.data.visualization import project_world
from finetune.OHT.data.video import decode_depth


CONFIG = Path(__file__).resolve().parents[1] / "finetune/OHT/configs/dataset.yaml"


@pytest.fixture
def dataset_config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture
def bounds(dataset_config):
    return dataset_config["scene_bounds"]


@pytest.fixture
def documented_positions():
    # Data description sections 8.4 (objects) and 10.2 (waypoints).
    return np.array([
        [.283, .480, .681], [.780, .183, 1.388],
        [.374, -.548, .936], [.779, -.182, 1.388],
        [.466, .334, 1.540], [.374, -.548, 1.039],
        [.374, -.548, 1.005], [.712, -.183, 1.388],
        [.507, -.183, 1.388],
    ])


def test_default_workspace_contains_documented_v423_examples(bounds, documented_positions):
    check_bounds(documented_positions, bounds)
    assert np.all(documented_positions > np.asarray(bounds[:3]))
    assert np.all(documented_positions < np.asarray(bounds[3:]))


def test_provisional_roi_preserves_documented_coordinate_examples(dataset_config, documented_positions):
    assert point_cloud_mask(documented_positions, dataset_config["point_cloud_filter"]).all()


def test_default_workspace_keeps_out_of_range_guard(bounds):
    with pytest.raises(ValueError, match="outside scene_bounds"):
        check_bounds(bounds[3:], bounds)


def test_default_keypoints_use_only_measured_gripper_changes(dataset_config):
    validate_data_config(dataset_config)
    assert dataset_config["keypoints"] == dict(method="gripper")


def test_default_camera_transforms_convert_optical_to_usd(dataset_config):
    assert dataset_config["camera_extrinsic_direction"] == "camera_to_world"
    assert dataset_config["video_alignment"] == "frame_index"
    assert set(dataset_config["cameras"]) == {
        "global_left", "global_right", "local_left", "local_right", "wrist",
    }
    for camera in dataset_config["cameras"].values():
        np.testing.assert_array_equal(
            transform_matrix(camera["optical_to_sensor"]), np.diag([1, -1, -1, 1]),
        )


@pytest.mark.parametrize("camera", ["global_left", "global_right", "local_left", "local_right", "wrist"])
def test_missing_camera_transform_explains_config_fix(dataset_config, camera):
    dataset_config["intrinsics_source"] = "metadata"
    dataset_config["cameras"][camera]["optical_to_sensor"] = None
    with pytest.raises(ValueError, match=f"{camera} optical_to_sensor is missing") as error:
        validate_data_config(dataset_config)
    assert "--config" in str(error.value)
    assert "OpenCV" in str(error.value)


def test_default_depth_contract_and_missing_definitions(dataset_config):
    validate_data_config(dataset_config)
    assert dataset_config["depth"]["encoding"] == "scaled_integer"
    assert dataset_config["depth"]["metadata"] is False
    assert dataset_config["depth"]["scale"] == .001
    assert dataset_config["depth"]["invalid_values"] == [0, 4095]
    assert dataset_config["depth"]["limits"] == [.001, 4.094]
    assert dataset_config["intrinsics_source"] == "metadata"
    dataset_config["depth"]["encoding"] = None
    with pytest.raises(ValueError, match="depth encoding"):
        validate_data_config(dataset_config)
    dataset_config["depth"]["encoding"] = "scaled_integer"
    dataset_config["depth"]["kind"] = None
    with pytest.raises(ValueError, match="depth.kind"):
        validate_data_config(dataset_config)
    dataset_config["depth"]["kind"] = "z"
    validate_data_config(dataset_config)


def test_documented_raw_depth_897_uses_mm_not_converter_log_default(dataset_config):
    raw = np.array([[0, 1, 897, 4094, 4095]], dtype=np.uint16)
    decoded = decode_depth(raw, dataset_config["depth"])
    np.testing.assert_allclose(decoded, [[np.nan, .001, .897, 4.094, np.nan]],
                               equal_nan=True)


@pytest.mark.parametrize("camera", ["global_left", "global_right", "local_left", "local_right", "wrist"])
def test_usd_camera_frame_preserves_ray_distance(dataset_config, camera):
    # Synthetic camera at (1,2,3) lies outside the provisional OHT task ROI.
    dataset_config["point_cloud_filter"]["enabled"] = False
    dataset_config["image_size"] = [2, 2]
    dataset_config["cameras"][camera]["intrinsics"] = np.eye(3).tolist()
    obs = camera_observation(
        camera, np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2)),
        [1, 2, 3, 1, 0, 0, 0], dataset_config,
    )
    cloud = obs[f"{camera}_point_cloud"]
    np.testing.assert_allclose(cloud[:, 0, 0], [1, 2, 2])
    d = 1 / np.sqrt(3)
    np.testing.assert_allclose(cloud[:, 1, 1], [1+d, 2-d, 3-d])
    np.testing.assert_allclose(np.linalg.norm(cloud - np.array([1, 2, 3])[:, None, None], axis=0), 1)


@pytest.mark.parametrize("order", [None, "unknown", ""])
def test_camera_order_must_be_explicit(dataset_config, order):
    if order is None:
        dataset_config.pop("camera_quaternion_order")
    else:
        dataset_config["camera_quaternion_order"] = order
    with pytest.raises(ValueError, match="camera_quaternion_order explicitly"):
        validate_data_config(dataset_config)
    with pytest.raises(ValueError, match="camera_quaternion_order explicitly"):
        camera_observation("wrist", np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)),
                           [0, 0, 0, 1, 0, 0, 0], dataset_config)


@pytest.mark.parametrize("order", ["wxyz", "xyzw"])
def test_raw_camera_order_to_world_xyz_and_provider_projection(dataset_config, order):
    # Isolate camera math; these arbitrary world points are not OHT fixtures.
    dataset_config["point_cloud_filter"]["enabled"] = False
    dataset_config["camera_quaternion_order"] = order
    dataset_config["image_size"] = [8, 8]
    K = np.array([[4., 0, 4], [0, 4., 4], [0, 0, 1]])
    dataset_config["cameras"]["wrist"]["intrinsics"] = K.tolist()
    # Deliberately non-symmetric, non-unit rotation: identity-only tests cannot
    # establish a quaternion ordering or the direction of the camera transform.
    rotation = Rotation.from_euler("xyz", [23, -41, 67], degrees=True)
    xyzw = rotation.as_quat()
    raw_q = xyzw[[3, 0, 1, 2]] if order == "wxyz" else xyzw
    position = np.array([.4, -.2, 1.6])
    obs = camera_observation("wrist", np.zeros((8, 8, 3), np.uint8), np.full((8, 8), 2.),
                             np.r_[position, raw_q], dataset_config)
    expected_transform = np.eye(4)
    expected_transform[:3, :3] = rotation.as_matrix() @ np.diag([1, -1, -1])
    expected_transform[:3, 3] = position
    np.testing.assert_allclose(obs["wrist_camera_extrinsics"], expected_transform, atol=1e-6)
    # Pixel (6,5), ray length 2m, independently transformed into the world.
    ray = np.array([.5, .25, 1.])
    optical = ray / np.linalg.norm(ray) * 2
    world = rotation.apply(optical * [1, -1, -1]) + position
    np.testing.assert_allclose(obs["wrist_point_cloud"][:, 5, 6], world, atol=1e-6)
    # Reference formula supplied by the data provider, in column-vector form.
    camera_usd = rotation.as_matrix().T @ (world - position)
    camera_optical = camera_usd * [1, -1, -1]
    expected_pixel = (K @ camera_optical)[:2] / camera_optical[2]
    projected, valid = project_world(world, K, obs["wrist_camera_extrinsics"], (8, 8))
    assert valid[0]
    np.testing.assert_allclose(projected[0], expected_pixel, atol=1e-5)
    np.testing.assert_allclose(projected[0], [6, 5], atol=1e-5)


@pytest.mark.parametrize("order", ["xyzw", "wxyz"])
def test_v423_recorded_tcp_is_not_offset_again_and_ee_order_is_explicit(dataset_config, order):
    assert dataset_config["camera_quaternion_order"] == "wxyz"
    np.testing.assert_array_equal(dataset_config["link_to_tcp"], np.eye(4))
    orientation = Rotation.from_euler("xyz", [19, 37, -64], degrees=True).as_quat()
    position = [.622, .183, 1.388]
    raw = orientation[[3, 0, 1, 2]] if order == "wxyz" else orientation
    columns = {"observation.ee_pos_world": [position], "observation.ee_quat_world": [raw]}
    pose = world_tcp_poses(columns, dataset_config["link_to_tcp"], order)[0]
    np.testing.assert_allclose(pose[:3], position, atol=1e-9)
    np.testing.assert_allclose(Rotation.from_quat(pose[3:]).as_matrix(),
                               Rotation.from_quat(orientation).as_matrix(), atol=1e-9)
