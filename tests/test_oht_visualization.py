"""Preview geometry and generation-time isolation using real OHT cache fixtures."""
import json
import numpy as np
from PIL import Image
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
            assert image.format == "PNG" and image.size[0] == 1024
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
