"""Metric scene/path alignment, bounded sensor-only snapshots and empty depth."""
import numpy as np
import pytest

from finetune.OHT.chunk_visualization import path_panel, scene_background


def test_camera_snapshot_preserves_rgb_xyz_pairs_and_ignores_labels():
    cloud = np.arange(300, dtype=np.float32).reshape(3, 10, 10)
    rgb = np.arange(300, dtype=np.uint16).reshape(3, 10, 10) % 256
    cloud[:, 0, 0] = np.nan
    sample = dict(front_point_cloud=cloud, front_rgb=rgb.astype(np.uint8),
                  wrist_point_cloud=cloud + 500, wrist_rgb=rgb.astype(np.uint8),
                  oracle_target_object_points=np.full((512, 3), 9999),
                  action_chunk=np.full((16, 8), 9999))
    before = sample['front_point_cloud'].copy()
    scene = scene_background(sample, max_points=18)
    assert scene['scene_points'].shape == scene['scene_colors'].shape == (18, 3)
    assert np.isfinite(scene['scene_points']).all()
    for i, name in enumerate(('front', 'wrist')):
        observed = sample[name + '_point_cloud'].reshape(3, -1).T
        colors = sample[name + '_rgb'].reshape(3, -1).T
        for point, color in zip(scene['scene_points'][i * 9:(i + 1) * 9],
                                scene['scene_colors'][i * 9:(i + 1) * 9]):
            index = np.flatnonzero((observed == point).all(axis=1)).item()
            np.testing.assert_array_equal(color, colors[index])
    np.testing.assert_array_equal(sample['front_point_cloud'], before)
    np.testing.assert_array_equal(scene_background(sample, max_points=18)['scene_points'], scene['scene_points'])
    sample['front_point_cloud'][:] = 9000
    assert (scene['scene_points'] < 9000).all()  # Queued snapshots own their arrays.


def test_scene_and_paths_share_world_projection_and_translation():
    path = np.array([[0., 0., 0.], [.1, .1, .1]])
    background = np.array([[.017, .071, .033]])
    colors = np.array([[40, 80, 200]], np.uint8)
    image = np.asarray(path_panel(path, path, current=path[0],
                                 scene_points=background, scene_colors=colors))
    center, span = np.array([.05] * 3), .2
    color = (.7 * colors[0] + .3 * np.array([246, 248, 250])).astype(np.uint8)
    for panel, axes in enumerate(((0, 1), (0, 2), (1, 2))):
        local = (background[0, list(axes)] - center[list(axes)]) * 240 / span
        x, y = np.rint(local * [1, -1] + [120, 120]).astype(int)
        np.testing.assert_array_equal(image[110 + y, panel * 300 + 30 + x], color)
    shift = np.array([3., -2., 5.])
    shifted = np.asarray(path_panel(path + shift, path + shift, current=path[0] + shift,
                                   scene_points=background + shift, scene_colors=colors))
    np.testing.assert_array_equal(image[110:351], shifted[110:351])


def test_far_background_does_not_shrink_paths_and_invalid_points_are_ignored():
    path = np.array([[0., 0., 0.], [.1, .1, .1]])
    scene = dict(scene_points=np.array([[.017, .071, .033]]), scene_colors=np.array([[40, 80, 200]]))
    expected = np.asarray(path_panel(path, **scene))
    extended = dict(scene_points=np.r_[scene['scene_points'], [[1e6] * 3, [np.nan] * 3]],
                    scene_colors=np.r_[scene['scene_colors'], [[255] * 3, [0] * 3]])
    np.testing.assert_array_equal(expected, np.asarray(path_panel(path, **extended)))


def test_no_valid_scene_uses_light_grid_and_still_draws_paths():
    path = np.array([[0., 0., 0.], [.1, .1, .1]])
    scene = scene_background(dict(front_point_cloud=np.full((3, 4, 4), np.nan),
                                 front_rgb=np.zeros((3, 4, 4), np.uint8)))
    assert scene['scene_points'].shape == (0, 3)
    image = np.asarray(path_panel(path, **scene))
    np.testing.assert_array_equal(image, np.asarray(path_panel(path)))
    assert (image[110:351].mean(axis=2) > 200).mean() > .9
    assert ((image[..., 0] == 255) & (image[..., 1] == 96) & (image[..., 2] == 96)).any()
    with pytest.raises(ValueError, match='aligned CHW'):
        scene_background(dict(front_point_cloud=np.zeros((3, 4, 4)), front_rgb=np.zeros((3, 2, 4))))
