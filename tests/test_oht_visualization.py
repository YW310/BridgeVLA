"""Preview geometry and generation-time isolation using real OHT cache fixtures."""
import json
import numpy as np
from PIL import Image, ImageDraw
import pytest
from scipy.spatial.transform import Rotation

from tests.test_oht_migration import replay_fixture, teacher_annotations
from finetune.OHT.cli import main
from finetune.OHT.data.common import read_jsonl
from finetune.OHT.data.geometry import backproject
from finetune.OHT.data.replay import build, load_contract
from finetune.OHT.data.role_teacher import build_teacher
from finetune.OHT.data.visualization import (
    TARGET, REFERENCE, OVERLAP, PreviewWriter, depth_colors, project_world, role_overlay,
)
from finetune.OHT.data import visualization as preview_module


def test_rgb_role_blending_preserves_background_and_input():
    rgb = np.full((2, 3, 3), 100, np.uint8)
    target = np.array([[True, False, True], [False, False, False]])
    reference = np.array([[False, True, True], [False, False, False]])
    result = role_overlay(rgb, dict(target=target, reference=reference))
    for index, color in enumerate((TARGET, REFERENCE, OVERLAP)):
        np.testing.assert_array_equal(result[0, index], (.3 * 100 + .7 * color).astype(np.uint8))
    np.testing.assert_array_equal(result[1], rgb[1])
    assert np.all(rgb == 100)
    with pytest.raises(ValueError, match="resolution"):
        role_overlay(rgb, dict(target=np.ones((3, 3), bool)))


def test_depth_invalid_values_are_black_and_meter_range_is_preserved():
    depth = np.array([[np.nan, np.inf, 0.], [1., 2., 12.]])
    result, scale = depth_colors(depth, [.001, 10])
    assert scale == (1., 2.)
    np.testing.assert_array_equal(result[0], 0)
    np.testing.assert_array_equal(result[1, 2], 0)
    np.testing.assert_array_equal(result[1, 0], [0, 0, 255])
    np.testing.assert_array_equal(result[1, 1], [255, 0, 0])
    empty, scale = depth_colors(np.full((2, 2), np.nan), [.001, 10])
    assert not empty.any() and scale is None


def test_camera_projection_roundtrips_world_extrinsics_and_rejects_behind_camera():
    K = np.array([[2., 0, 1], [0, 3, 1], [0, 0, 1]])
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", [20, 30, 80], degrees=True).as_matrix()
    transform[:3, 3] = [1., 2., 3.]
    points = backproject(np.ones((3, 4)), K, transform)
    xy, valid = project_world(points.reshape(-1, 3), K, transform, (3, 4))
    # Use interior pixels to avoid floating point roundoff at the frustum edge.
    np.testing.assert_allclose(xy[5:7], [[1, 1], [2, 1]], atol=1e-6)
    assert valid[5:7].all()
    behind = np.array([[0., 0., -1.]]) @ transform[:3, :3].T + transform[:3, 3]
    xy, valid = project_world(behind, K, transform, (3, 4))
    assert not valid.any() and np.isnan(xy).all()


def test_buffer_cli_previews_do_not_change_contract_labels_or_observations(replay_fixture, tmp_path):
    import yaml
    f = replay_fixture
    config = tmp_path / "dataset.yaml"
    config.write_text(yaml.safe_dump(f.config), encoding="utf-8")
    output, previews = tmp_path / "replay", tmp_path / "previews"
    assert main(["build", "--root", str(f.root), "--manifest", str(f.manifest),
                 "--config", str(config), "--output", str(output), "--sample-stride", "2",
                 "--visualize-every", "100", "--visualize-output-dir", str(previews)]) == 0
    assert load_contract(output) == load_contract(f.replay)
    assert (output / "samples.jsonl").read_bytes() == (f.replay / "samples.jsonl").read_bytes()
    rows = read_jsonl(output / "samples.jsonl")
    images = sorted(previews.rglob("*.png"))
    episodes = {(row["task"], row["episode_index"]) for row in rows}
    assert len(images) == len(episodes) == 12  # First emitted sample of every episode.
    for path in images:
        assert path.name == "000000.png"
        with Image.open(path) as image:
            assert image.format == "PNG" and image.size == (1024, 1490)
            image.verify()
    assert not (f.replay / "visualizations").exists()
    assert not (output / "visualizations").exists()
    path = previews / (rows[0]["id"] + ".png")
    original = path.read_bytes()
    with np.load(output / rows[0]["observation"], allow_pickle=False) as observation:
        with pytest.raises(FileExistsError):
            PreviewWriter(previews, 1).write(observation, f.config, rows[0])
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="nonnegative"):
        build(f.root, f.manifest, f.config, tmp_path / "invalid", visualize_every=-1)
    assert not (tmp_path / "invalid").exists()


def test_teacher_previews_cover_unknown_site_null_without_changing_cache(replay_fixture, tmp_path):
    f = replay_fixture
    annotations_path = tmp_path / "annotations.jsonl"
    annotations = teacher_annotations(f, annotations_path)[:3]
    annotations[0]["reference"] = dict(present=True, known=True, source="unknown")
    transform = np.eye(4)
    transform[:3, 3] = [.2, .2, 1.]
    annotations[1]["reference"] = dict(present=True, known=True, source="site_region",
                                       world_from_site=transform.tolist(), size=[.2, .2, .2])
    annotations_path.write_text("".join(json.dumps(row) + "\n" for row in annotations), encoding="utf-8")
    baseline = build_teacher(f.replay, annotations_path, tmp_path / "baseline", point_count=8)
    output = tmp_path / "teacher"
    assert main(["teacher", "--replay", str(f.replay), "--annotations", str(annotations_path),
                 "--output", str(output), "--point-count", "8", "--visualize-every", "1"]) == 0
    visualized = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert visualized == baseline
    for row in annotations:
        with Image.open(output / "visualizations" / (row["id"] + ".png")) as image:
            assert image.size[0] == 1024
            image.verify()
    assert not (tmp_path / "baseline" / "visualizations").exists()


def test_preview_writer_is_disabled_by_default(tmp_path):
    writer = PreviewWriter(tmp_path / "disabled")
    assert writer.write({}, {}, {}) is None
    assert not writer.output.exists()
    with pytest.raises(ValueError, match="nonnegative"):
        PreviewWriter(tmp_path, -1)


def test_global_preview_keeps_more_than_twenty_thousand_points():
    points = np.zeros((24001, 3))
    colors = np.zeros((24001, 3), dtype=np.uint8)
    shown, rgb = preview_module._sample_preview_cloud(points, colors)
    assert len(shown) == len(rgb) == len(points)
    assert preview_module.MAX_PREVIEW_POINTS == 200_000


def test_small_splats_keep_nearest_surface_and_do_not_propagate():
    points = np.array([[.5, .5, .2], [.5, .5, .8]])
    colors = np.array([[255, 0, 0], [0, 0, 255]], dtype=np.uint8)
    bounds = [0, 0, 0, 1, 1, 1]
    image = np.asarray(preview_module._orthographic(points, colors, bounds, (0, 1), None, None, None, None))
    np.testing.assert_array_equal(image[126:129, 168:171], np.tile([0, 0, 255], (3, 3, 1)))
    assert (image == [0, 0, 255]).all(axis=-1).sum() == 9
    # A nearer splat also wins over a farther neighbouring centre pixel.
    points = np.r_[points, [[.5 + 1/339, .5, .1]]]
    colors = np.r_[colors, [[0, 255, 0]]].astype(np.uint8)
    first = np.asarray(preview_module._orthographic(points, colors, bounds, (0, 1), None, None, None, None))
    second = np.asarray(preview_module._orthographic(points[::-1], colors[::-1], bounds, (0, 1), None, None, None, None))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first[127, 170], [0, 0, 255])


def _minimal_preview(points, center):
    count = len(points)
    observation = {
        "wrist_point_cloud": points.T[:, None, :],
        "wrist_rgb": np.full((3, 1, count), 160, dtype=np.uint8),
        "wrist_depth": np.ones((1, 1, count)),
        "wrist_camera_intrinsics": np.eye(3),
        "wrist_camera_extrinsics": np.eye(4),
    }
    config = dict(cameras={"wrist": {}}, scene_bounds=[0, 0, 0, 1, 1, 1], depth=dict(limits=[.001, 4.094]))
    sample = dict(id="task/000000/000000", split="train", frame=0, target_frame=1,
                  timestamp=0., goal="test", labels=dict(gripper_pose=[*center, 0, 0, 0, 1]))
    return observation, config, sample


@pytest.mark.parametrize("center", [[.5, .5, .5], [.98, .5, .5]])
def test_local_views_select_before_global_sampling_and_keep_gt_center(monkeypatch, tmp_path, center):
    center = np.array(center)
    local_points = center + np.array([[0, 0, 0], [-.01, 0, 0], [-.02, 0, 0]])
    points = np.r_[np.tile([.9, .9, .9], (100, 1)), local_points]
    observation, config, sample = _minimal_preview(points, center)
    original_arrays = {name: value.copy() for name, value in observation.items()}
    calls = []
    original = preview_module._orthographic
    def record(cloud, rgb, bounds, *args, **kwargs):
        calls.append((cloud.copy(), np.asarray(bounds).copy(), kwargs.get("region_bounds")))
        return original(cloud, rgb, bounds, *args, **kwargs)
    monkeypatch.setattr(preview_module, "MAX_PREVIEW_POINTS", 5)
    monkeypatch.setattr(preview_module, "_orthographic", record)
    path = preview_module.save_preview(tmp_path / "preview.png", observation, config, sample)
    assert len(calls) == 6
    expected_bounds = np.r_[center - .20, center + .20]
    for cloud, bounds, region in calls[:3]:
        assert len(cloud) == 5
        np.testing.assert_allclose(region, expected_bounds)
    for cloud, bounds, region in calls[3:]:
        np.testing.assert_allclose(cloud, local_points)
        np.testing.assert_allclose(bounds, expected_bounds)
        assert region is None
    for name, value in observation.items():
        np.testing.assert_array_equal(value, original_arrays[name])
    with Image.open(path) as image:
        assert image.size == (1024, 1018)  # One physical camera, two orthographic rows.


def test_empty_local_views_are_explicitly_labelled(monkeypatch, tmp_path):
    observation, config, sample = _minimal_preview(np.array([[.9, .9, .9]]), [.5, .5, .5])
    texts = []
    original = ImageDraw.ImageDraw.text
    def record(self, xy, text, *args, **kwargs):
        texts.append(text)
        return original(self, xy, text, *args, **kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record)
    preview_module.save_preview(tmp_path / "empty.png", observation, config, sample)
    assert any("GT-centered refine diagnostic (NOT model stage2)" in text for text in texts)
    assert any("No observed points in local cube" in text for text in texts)
    assert any("in-bounds points 1 -> shown 1" in text for text in texts)


def test_camera_coloring_changes_only_orthographic_display(monkeypatch, tmp_path):
    observation, config, sample = _minimal_preview(np.array([[.5, .5, .5]]), [.5, .5, .5])
    original_arrays = {key: value.copy() for key, value in observation.items()}
    colors_seen = []
    original = preview_module._orthographic

    def record(points, colors, *args, **kwargs):
        colors_seen.append(colors.copy())
        return original(points, colors, *args, **kwargs)

    monkeypatch.setattr(preview_module, "_orthographic", record)
    preview_module.save_preview(tmp_path / "camera-colors.png", observation, config, sample, color_by_camera=True)
    assert len(colors_seen) == 6
    for colors in colors_seen:
        np.testing.assert_array_equal(colors, [preview_module.CAMERA_COLORS["wrist"]])
    for key, value in observation.items():
        np.testing.assert_array_equal(value, original_arrays[key])


def test_geometry_cli_exports_each_physical_camera_without_changing_cache(replay_fixture, tmp_path):
    from finetune.OHT.data.common import file_digest

    f = replay_fixture
    row = read_jsonl(f.replay / "samples.jsonl")[0]
    before = file_digest(f.replay / row["observation"])
    output = tmp_path / "geometry"
    args = ["diagnose-geometry", "--replay", str(f.replay), "--sample-id", row["id"], "--output", str(output)]
    assert main(args) == 0
    assert {path.name for path in output.glob("*.png")} == {
        "fused_rgb.png", "fused_camera_colors.png", *(camera + ".png" for camera in f.config["cameras"])}
    summary = json.loads((output / "geometry.json").read_text(encoding="utf-8"))
    assert summary["sample_id"] == row["id"]
    assert summary["video_alignment"] == "timestamp"  # Legacy config is not silently reinterpreted.
    for camera in f.config["cameras"]:
        np.testing.assert_array_equal(summary["cameras"][camera]["world_from_optical"], np.eye(4))
    assert file_digest(f.replay / row["observation"]) == before
    with pytest.raises(FileExistsError):
        main(args)
    invalid = tmp_path / "missing"
    with pytest.raises(ValueError, match="Unknown sample ID"):
        main(["diagnose-geometry", "--replay", str(f.replay), "--sample-id", "missing", "--output", str(invalid)])
    assert not invalid.exists()
