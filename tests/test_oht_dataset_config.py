"""Default OHT workspace covers the v423 description, not measured full data."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from finetune.OHT.data.geometry import check_bounds, transform_matrix
from finetune.OHT.data.observation import camera_observation, validate_data_config


CONFIG = Path(__file__).resolve().parents[1] / "finetune/OHT/configs/dataset.yaml"


@pytest.fixture
def dataset_config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture
def bounds(dataset_config):
    return dataset_config["scene_bounds"]


def test_default_workspace_contains_documented_v423_examples(bounds):
    # Data description sections 8.4 (objects) and 10.2 (waypoints).
    points = np.array([
        [.283, .480, .681], [.780, .183, 1.388],
        [.374, -.548, .936], [.779, -.182, 1.388],
        [.466, .334, 1.540], [.374, -.548, 1.039],
        [.374, -.548, 1.005], [.712, -.183, 1.388],
        [.507, -.183, 1.388],
    ])
    check_bounds(points, bounds)
    assert np.all(points > np.asarray(bounds[:3]))
    assert np.all(points < np.asarray(bounds[3:]))


def test_default_workspace_keeps_out_of_range_guard(bounds):
    with pytest.raises(ValueError, match="outside scene_bounds"):
        check_bounds(bounds[3:], bounds)


def test_default_camera_transforms_convert_optical_to_usd(dataset_config):
    assert set(dataset_config["cameras"]) == {
        "global_left", "global_right", "local_left", "local_right", "wrist",
    }
    for camera in dataset_config["cameras"].values():
        np.testing.assert_array_equal(
            transform_matrix(camera["optical_to_sensor"]), np.diag([1, -1, -1, 1]),
        )


@pytest.mark.parametrize("camera", ["global_left", "global_right", "local_left", "local_right", "wrist"])
def test_missing_camera_transform_explains_config_fix(dataset_config, camera):
    dataset_config["cameras"][camera]["optical_to_sensor"] = None
    with pytest.raises(ValueError, match=f"{camera} optical_to_sensor is missing") as error:
        validate_data_config(dataset_config)
    assert "--config" in str(error.value)
    assert "OpenCV" in str(error.value)


def test_default_depth_contract_and_missing_definitions(dataset_config):
    validate_data_config(dataset_config)
    assert dataset_config["depth"] == {
        "encoding": "scaled_integer", "pixel_format": "gray12le", "scale": .001,
        "offset": 0., "kind": "ray", "invalid_values": [0, 4095],
        "path_pattern": None, "limits": [.001, 4.094],
    }
    dataset_config["depth"]["encoding"] = None
    with pytest.raises(ValueError, match="depth encoding"):
        validate_data_config(dataset_config)
    dataset_config["depth"]["encoding"] = "scaled_integer"
    dataset_config["depth"]["kind"] = None
    with pytest.raises(ValueError, match="depth.kind"):
        validate_data_config(dataset_config)
    dataset_config["depth"]["kind"] = "z"
    validate_data_config(dataset_config)


@pytest.mark.parametrize("camera", ["global_left", "global_right", "local_left", "local_right", "wrist"])
def test_usd_camera_frame_preserves_ray_distance(dataset_config, camera):
    dataset_config["image_size"] = [2, 2]
    dataset_config["cameras"][camera]["intrinsics"] = np.eye(3).tolist()
    obs = camera_observation(
        camera, np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2)),
        [1, 2, 3, 0, 0, 0, 1], dataset_config,
    )
    cloud = obs[f"{camera}_point_cloud"]
    np.testing.assert_allclose(cloud[:, 0, 0], [1, 2, 2])
    d = 1 / np.sqrt(3)
    np.testing.assert_allclose(cloud[:, 1, 1], [1+d, 2-d, 3-d])
    np.testing.assert_allclose(np.linalg.norm(cloud - np.array([1, 2, 3])[:, None, None], axis=0), 1)
