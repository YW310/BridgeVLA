"""Manual background boxes: shared world geometry, no RGB/action/role relabeling."""
from copy import deepcopy
from pathlib import Path
import json

import numpy as np
import pytest
import yaml

from finetune.OHT.data.common import read_jsonl
from finetune.OHT.data.dataset import OHTDataset, collate
from finetune.OHT.data.observation import camera_observation, validate_data_config, validate_observation
from finetune.OHT.data.point_filter import (
    filter_observation_points, filter_world_points, point_cloud_mask, point_filter_options,
)
from finetune.OHT.data.replay import build, load_contract
from finetune.OHT.data.role_cache import RoleCache, role_fields
from finetune.OHT.data.role_teacher import build_teacher
from finetune.OHT.runtime.policy import Policy
from finetune.OHT.runtime.predicted_wrapper import PredictedObjectWrapper
from tests.test_oht_migration import FakeAgent, replay_fixture


def test_default_filter_is_off_and_legacy_configs_are_supported():
    path = Path(__file__).resolve().parents[1] / "finetune/OHT/configs/dataset.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["point_cloud_filter"] == point_filter_options() == dict(
        enabled=False, keep_bounds=None, exclude_boxes=[])
    validate_data_config(config)
    config.pop("point_cloud_filter")
    validate_data_config(config)


@pytest.mark.parametrize("options", [
    [], True, dict(enabled="false"), dict(enabled=1), dict(exclude_box=[]),
    dict(keep_bounds=[0, 0, 0, 1, 1]),
    dict(keep_bounds=[0, 0, 0, 0, 1, 1]),
    dict(keep_bounds=[0, 0, 0, 1, -1, 1]),
    dict(keep_bounds=[0, 0, 0, np.inf, 1, 1]),
    dict(keep_bounds=[0, 0, 0, "x", 1, 1]),
    dict(exclude_boxes=None), dict(exclude_boxes=[0, 0, 0, 1, 1, 1]),
    dict(exclude_boxes=[[0, 0, 0, 1, 1, np.nan]]),
    dict(exclude_boxes=[[0, 0, 0, 1, 1, 0]]),
])
def test_invalid_filter_options_fail_early(options):
    with pytest.raises(ValueError, match="point_cloud_filter"):
        point_filter_options(options)


def test_roi_and_exclusion_boxes_have_explicit_half_open_boundaries():
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 1, 1], [3, 1, 1],
                       [-1, 1, 1], [np.nan, 1, 1], [np.inf, 1, 1]], dtype=float)
    options = dict(enabled=True, keep_bounds=[0, 0, 0, 3, 3, 3],
                   exclude_boxes=[[1, 1, 1, 2, 2, 2]])
    np.testing.assert_array_equal(point_cloud_mask(points, options), [1, 0, 1, 0, 0, 0, 0])


def test_exclusions_are_a_union_with_no_required_roi():
    points = np.array([[-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    options = dict(enabled=True, exclude_boxes=[[-1, -1, -1, 0, 1, 1], [1, -1, -1, 2, 1, 1]])
    np.testing.assert_array_equal(point_cloud_mask(points, options), [0, 1, 0, 1])
    assert point_cloud_mask(points, dict(enabled=True)).all()


def test_disabled_filter_preserves_input_and_ignores_configured_boxes():
    points = np.array([[.5, .5, .5], [np.nan, np.nan, np.nan]], dtype=np.float32)
    options = dict(enabled=False, exclude_boxes=[[0, 0, 0, 1, 1, 1]])
    assert filter_world_points(points, options) is points
    np.testing.assert_array_equal(point_cloud_mask(points, options), [1, 0])
    observation = dict(cam_point_cloud=points.T[:, None])
    assert filter_observation_points(observation, dict(cameras={"cam": {}}, point_cloud_filter=options)) is observation


def test_filter_preserves_organized_layout_dtype_and_input_values():
    points = np.array([[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 1]]], np.float32)
    original = points.copy()
    filtered = filter_world_points(points, dict(enabled=True, keep_bounds=[0, 0, 0, 1, 2, 2]))
    assert filtered.shape == points.shape and filtered.dtype == points.dtype
    assert np.isnan(filtered[:, 1]).all()
    np.testing.assert_array_equal(filtered[:, 0], points[:, 0])
    np.testing.assert_array_equal(points, original)
    np.testing.assert_array_equal(filter_world_points(filtered, dict(enabled=True)), filtered)
    with pytest.raises(ValueError, match="world points"):
        point_cloud_mask(np.zeros((3, 2)))
    with pytest.raises(ValueError, match="floating-point"):
        filter_world_points(np.zeros((2, 3), dtype=int), dict(enabled=True))
    with pytest.raises(ValueError, match="never infinity"):
        filter_world_points(np.full((2, 3), np.inf), dict(enabled=True))


def test_filter_uses_world_xyz_after_extrinsics_and_keeps_raw_rgb_depth():
    config = dict(camera_quaternion_order="wxyz", image_size=[2, 2],
                  cameras={"cam": dict(intrinsics=np.eye(3).tolist(), optical_to_sensor=np.eye(4).tolist())},
                  depth=dict(kind="z"))
    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    depth = np.ones((2, 2), np.float32)
    pose = [10, 0, 0, 1, 0, 0, 0]
    raw = camera_observation("cam", rgb, depth, pose, config)
    config["point_cloud_filter"] = dict(enabled=True, keep_bounds=[10, 0, 0, 10.5, 2, 2],
                                        exclude_boxes=[[10, 1, 0, 10.5, 2, 2]])
    filtered = camera_observation("cam", rgb, depth, pose, config)
    expected = filter_observation_points(raw, config)
    for name, value in filtered.items():
        np.testing.assert_array_equal(value, expected[name])
        if not name.endswith("point_cloud"):
            np.testing.assert_array_equal(value, raw[name])
    assert np.isfinite(filtered["cam_point_cloud"]).all(axis=0).sum() == 1
    np.testing.assert_array_equal(filtered["cam_point_cloud"][:, 0, 0], [10, 0, 1])
    assert np.isfinite(raw["cam_point_cloud"]).all()  # neither source nor raw RGB is mutated.
    # Removing one camera is allowed; whole-scene rejection occurs after fusion.
    config["point_cloud_filter"]["keep_bounds"] = [20, 0, 0, 21, 2, 2]
    assert np.isnan(camera_observation("cam", rgb, depth, pose, config)["cam_point_cloud"]).all()


@pytest.fixture(scope="module")
def filtered_replay(replay_fixture, tmp_path_factory):
    config = deepcopy(replay_fixture.config)
    config["point_cloud_filter"] = dict(enabled=True, keep_bounds=[-.5, -.5, .3, .75, .75, 1.5],
                                        exclude_boxes=[[-.5, -.5, .3, 0, 0, 1.5]])
    output = tmp_path_factory.mktemp("manual-background") / "replay"
    build(replay_fixture.root, replay_fixture.manifest, config, output, sample_stride=2)
    return output


def test_replay_filter_changes_only_xyz_not_targets_or_action_bounds(replay_fixture, filtered_replay):
    f = replay_fixture
    before, after = load_contract(f.replay), load_contract(filtered_replay)
    assert before["sha256"] != after["sha256"]
    assert after["data_config"]["scene_bounds"] == before["data_config"]["scene_bounds"]
    options = after["data_config"]["point_cloud_filter"]
    assert options["enabled"]
    for profile in after["source_data_configs"].values():
        assert profile["data_config"]["point_cloud_filter"] == options
    raw_rows = read_jsonl(f.replay / "samples.jsonl")
    rows = read_jsonl(filtered_replay / "samples.jsonl")
    assert len(rows) == len(raw_rows)
    for raw_row, row in zip(raw_rows, rows):
        for key in set(row) - {"observation_sha256"}:
            assert row[key] == raw_row[key]
    with np.load(f.replay / raw_rows[0]["observation"], allow_pickle=False) as raw, \
            np.load(filtered_replay / rows[0]["observation"], allow_pickle=False) as filtered:
        expected = filter_observation_points(raw, after["data_config"])
        for name in raw.files:
            np.testing.assert_array_equal(filtered[name], expected[name])
            if not name.endswith("point_cloud"):
                np.testing.assert_array_equal(filtered[name], raw[name])
        raw_count = np.isfinite(raw["global_left_point_cloud"]).all(axis=0).sum()
        kept_count = np.isfinite(filtered["global_left_point_cloud"]).all(axis=0).sum()
        assert 0 < kept_count < raw_count


def test_training_renderer_receives_filtered_xyz_and_aligned_rgb(filtered_replay):
    from finetune.bridgevla.data.observations import preprocess_inputs
    from finetune.bridgevla.utils.rvt_utils import get_pc_img_feat, move_pc_in_bound
    data = OHTDataset(filtered_replay, "train")
    row = data[0]
    config = data.contract["data_config"]
    obs, pcd = preprocess_inputs(collate([row]), config["cameras"])
    clouds, colors = get_pc_img_feat(obs, pcd)
    clouds, colors = move_pc_in_bound(clouds, colors, config["scene_bounds"])
    assert len(clouds[0]) == len(colors[0]) > 0
    assert point_cloud_mask(clouds[0].numpy(), config["point_cloud_filter"]).all()
    expected_colors = []
    for camera in config["cameras"]:
        valid = np.isfinite(row[f"{camera}_point_cloud"]).all(axis=0)
        expected_colors.append(row[f"{camera}_rgb"].transpose(1, 2, 0)[valid] / 255.)
    np.testing.assert_allclose(colors[0].numpy(), np.concatenate(expected_colors), atol=1e-7)


@pytest.mark.parametrize("mode", ["baseline", "role_queries", "predicted_external"])
def test_online_raw_input_is_filtered_like_training_before_predictor_and_agent(
        replay_fixture, filtered_replay, mode):
    raw_data, filtered_data = OHTDataset(replay_fixture.replay, "train"), OHTDataset(filtered_replay, "train")
    raw, expected = raw_data[0], filtered_data[0]
    original_cloud = raw["global_left_point_cloud"].copy()
    predictor_inputs = []
    def predictor(observation, goal):
        predictor_inputs.append(observation)
        assert not any(key.startswith("oracle") for key in observation)
        return role_fields("predicted", np.zeros((2, 8, 3)), np.zeros(2, bool),
                           np.zeros(2, bool), confidence=[.5, .5])
    agent = FakeAgent()
    config = filtered_data.contract["data_config"]
    wrapper = PredictedObjectWrapper(predictor, config["cameras"], 8) if mode == "predicted_external" else None
    policy = Policy(agent, filtered_data.contract, "cpu", mode, wrapper)
    policy.act(raw, raw["goal"], 0)
    for camera in config["cameras"]:
        for suffix in ("point_cloud", "rgb", "depth"):
            name = f"{camera}_{suffix}"
            np.testing.assert_array_equal(agent.inputs[0][name][0, 0].numpy(), expected[name])
            if predictor_inputs:
                np.testing.assert_array_equal(predictor_inputs[0][name], expected[name])
    assert "action" not in agent.inputs[0] and "gripper_pose" not in agent.inputs[0]
    np.testing.assert_array_equal(raw["global_left_point_cloud"], original_cloud)


def test_filtered_mask_teacher_remains_present_when_geometry_is_removed(filtered_replay, tmp_path):
    row = read_jsonl(filtered_replay / "samples.jsonl")[0]
    config = load_contract(filtered_replay)["data_config"]
    masks = {}
    for camera in config["cameras"]:
        mask = np.zeros(config["image_size"], bool)
        mask[3, 3] = True  # world (-.25, -.25, 1), inside the excluded box.
        masks[camera] = mask
    np.savez_compressed(tmp_path / "masks.npz", **masks)
    annotation = dict(id=row["id"], target=dict(present=True, known=True, source="visible_surface", mask_path="masks.npz"),
                      reference=dict(present=False, known=True, source="none"))
    path = tmp_path / "annotations.jsonl"
    path.write_text(json.dumps(annotation) + "\n", encoding="utf-8")
    output = tmp_path / "teacher"
    build_teacher(filtered_replay, path, output, point_count=8)
    fields = RoleCache(output, "teacher", load_contract(filtered_replay)["sha256"], 8).read(row["id"])
    assert fields["oracle_target_present"] and not fields["oracle_target_object_valid"]
    assert not fields["oracle_reference_present"]


def test_filter_can_empty_one_camera_but_rejects_an_empty_whole_scene(replay_fixture):
    data = OHTDataset(replay_fixture.replay, "train")
    raw = data[0]
    config = deepcopy(data.contract["data_config"])
    config["point_cloud_filter"] = dict(enabled=True, keep_bounds=[-.5, -.5, .3, .75, .75, 1.5])
    raw["global_left_point_cloud"] = np.full_like(raw["global_left_point_cloud"], 10)
    filtered = filter_observation_points(raw, config)
    # Remove training-only labels before the online observation contract check.
    from finetune.OHT.runtime.predicted_wrapper import policy_observation
    validate_observation(policy_observation(filtered, config["cameras"]), config)
    assert np.isnan(filtered["global_left_point_cloud"]).all()
    config["point_cloud_filter"]["exclude_boxes"] = [config["point_cloud_filter"]["keep_bounds"]]
    agent = FakeAgent()
    contract = deepcopy(data.contract)
    contract["data_config"] = config
    with pytest.raises(ValueError, match="No finite scene points.*filtering"):
        Policy(agent, contract, "cpu", "baseline").act(raw, raw["goal"], 0)
    assert not agent.inputs


def test_invalid_filter_fails_before_output_and_total_removal_cannot_complete(replay_fixture, tmp_path):
    f = replay_fixture
    config = deepcopy(f.config)
    config["point_cloud_filter"] = dict(enabled=True, exclude_boxes=[[0, 0, 0, 0, 1, 1]])
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="positive extents"):
        build(f.root, f.manifest, config, output)
    assert not output.exists()
    config["point_cloud_filter"]["exclude_boxes"] = [config["scene_bounds"]]
    output = tmp_path / "all-removed"
    with pytest.raises(ValueError, match="no scene points.*filtering"):
        build(f.root, f.manifest, config, output)
    assert not (output / "complete.json").exists()


def test_global_and_local_previews_filter_before_sampling_without_mutating_inputs(monkeypatch, tmp_path):
    from finetune.OHT.data import visualization
    from tests.test_oht_visualization import _minimal_preview
    points = np.array([[.45, .5, .5], [.55, .5, .5], [.65, .5, .5]], np.float32)
    observation, config, sample = _minimal_preview(points, [.55, .5, .5])
    observation["wrist_rgb"] = np.array([[[10, 20, 30]], [[40, 50, 60]], [[70, 80, 90]]], np.uint8)
    config["point_cloud_filter"] = dict(enabled=True, keep_bounds=[.4, .4, .4, .8, .6, .6],
                                        exclude_boxes=[[.5, .4, .4, .6, .6, .6]])
    original = {name: value.copy() for name, value in observation.items()}
    sample_before = deepcopy(sample)
    calls = []
    render = visualization._orthographic
    def record(cloud, rgb, *args, **kwargs):
        calls.append((cloud.copy(), rgb.copy()))
        return render(cloud, rgb, *args, **kwargs)
    monkeypatch.setattr(visualization, "_orthographic", record)
    monkeypatch.setattr(visualization, "MAX_PREVIEW_POINTS", 2)
    visualization.save_preview(tmp_path / "filtered.png", observation, config, sample)
    assert len(calls) == 6
    for cloud, rgb in calls:
        np.testing.assert_array_equal(cloud, points[[0, 2]])
        np.testing.assert_array_equal(rgb, observation["wrist_rgb"].reshape(3, -1).T[[0, 2]])
    for name, value in observation.items():
        np.testing.assert_array_equal(value, original[name])
    assert sample == sample_before
