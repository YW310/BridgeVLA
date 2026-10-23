"""Sensor diagnostic geometry has no simulator, policy, or renderer dependency."""
import json

import numpy as np
from PIL import Image
import pytest

from finetune.OHT.runtime.sensor_visualization import (
    MAX_SCENE_POINTS, export_sensor_scene, prepared_sensor_snapshot, raw_sensor_snapshot,
)


def raw_inputs(name='front', height=4, width=6):
    rgb = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    depth = np.full((height, width), 2.0012345, np.float32)
    intrinsics = np.array([[2., 0., width / 2], [0., 2., height / 2], [0., 0., 1.]])
    values = {f'{name}_rgb': rgb, f'{name}_depth': depth,
              f'{name}_intrinsics': intrinsics, f'{name}_pose': np.array([0., 0., 0., 1., 0., 0., 0.]),
              'untrusted_label': np.array([999]), 'teacher': {'untrusted': True}}
    config = dict(cameras={name: {}}, depth={'kind': 'ray', 'limits': [.001, 10.], 'round_to_mm': True},
                  image_size=[2, 3], point_cloud_filter={'enabled': True})
    return values, config


def canonical_snapshot(kind='z', depth=None):
    depth = np.full((3, 4), 2., np.float32) if depth is None else np.asarray(depth, np.float32)
    height, width = depth.shape
    return {'source': 'prepared_observation', 'cameras': {'front': {
        'rgb': np.full((height, width, 3), [110, 150, 210], np.uint8), 'depth': depth,
        'intrinsics': np.eye(3), 'world_from_optical': np.eye(4),
        'depth_kind': kind, 'depth_limits': [.001, 10.]}}}


def read_scene(folder):
    return json.loads((folder / 'scene.json').read_text(encoding='utf-8'))


def test_raw_snapshot_preserves_resolution_values_and_owned_whitelist():
    values, config = raw_inputs()
    tcp = np.array([1., 2., 3., 0., 0., 0., 1.])
    snapshot = raw_sensor_snapshot(values, config, tcp)
    camera = snapshot['cameras']['front']
    assert set(snapshot) == {'source', 'cameras', 'tcp_pose_xyzw'}
    assert set(camera) == {'rgb', 'depth', 'intrinsics', 'world_from_optical', 'depth_kind', 'depth_limits'}
    assert snapshot['source'] == 'raw_sensor'
    assert camera['rgb'].shape == (4, 6, 3)
    assert camera['depth_kind'] == 'z'  # Configured training ray depth does not reinterpret wire Z.
    assert camera['depth'][0, 0] == values['front_depth'][0, 0]
    assert camera['depth'][0, 0] != np.round(camera['depth'][0, 0] * 1000) / 1000
    np.testing.assert_array_equal(camera['world_from_optical'], np.diag([1., -1., -1., 1.]))
    for suffix, key in [('rgb', 'rgb'), ('depth', 'depth'), ('intrinsics', 'intrinsics')]:
        assert not np.shares_memory(values[f'front_{suffix}'], camera[key])
    values['front_rgb'][:] = 0
    values['front_depth'][:] = 0
    tcp[:] = 0
    assert camera['rgb'].max() > 0
    assert camera['depth'].min() > 2
    assert snapshot['tcp_pose_xyzw'][0] == 1


def test_raw_pose_rotation_is_applied_before_optical_axis_conversion():
    values, config = raw_inputs()
    values['front_pose'] = np.array([1., 2., 3., np.sqrt(.5), 0., 0., np.sqrt(.5)])
    snapshot = raw_sensor_snapshot(values, config)
    expected = np.array([[0., 1., 0., 1.], [1., 0., 0., 2.],
                         [0., 0., -1., 3.], [0., 0., 0., 1.]])
    np.testing.assert_allclose(snapshot['cameras']['front']['world_from_optical'], expected, atol=1e-12)


def test_raw_camera_depth_override_controls_limits_only():
    values, config = raw_inputs()
    config['cameras']['front']['depth'] = {'kind': 'ray', 'limits': [.1, 3.]}
    camera = raw_sensor_snapshot(values, config)['cameras']['front']
    assert camera['depth_kind'] == 'z'
    assert camera['depth_limits'] == [.1, 3.]


def test_prepared_fallback_preserves_canonical_axes_and_ray_semantics():
    observation = dict(front_rgb=np.zeros((3, 3, 4), np.uint8),
                       front_depth=np.full((1, 3, 4), 2., np.float32),
                       front_camera_intrinsics=np.eye(3), front_camera_extrinsics=np.eye(4),
                       front_point_cloud=np.zeros((3, 3, 4)), teacher=np.zeros((5,)),
                       tcp_pose=np.array([0., 0., 1., 0., 0., 0., 1.]))
    config = dict(cameras={'front': {'depth': {'kind': 'ray', 'limits': [.01, 5.]}}})
    snapshot = prepared_sensor_snapshot(observation, config)
    assert snapshot['source'] == 'prepared_observation'
    camera = snapshot['cameras']['front']
    assert camera['rgb'].shape == (3, 4, 3)
    assert camera['depth_kind'] == 'ray'
    np.testing.assert_array_equal(camera['world_from_optical'], np.eye(4))
    assert not np.shares_memory(camera['rgb'], observation['front_rgb'])
    assert not np.shares_memory(snapshot['tcp_pose_xyzw'], observation['tcp_pose'])
    assert 'teacher' not in snapshot


def test_prepared_fallback_uses_global_depth_kind_and_discovers_camera_names():
    observation = dict(wrist_rgb=np.zeros((3, 2, 5), np.uint8), wrist_depth=np.ones((1, 2, 5)),
                       wrist_camera_intrinsics=np.eye(3), wrist_camera_extrinsics=np.eye(4))
    assert prepared_sensor_snapshot(observation)['cameras']['wrist']['depth_kind'] == 'z'
    assert prepared_sensor_snapshot(observation, {'depth': {'kind': 'ray'}})['cameras']['wrist']['depth_kind'] == 'ray'
    assert prepared_sensor_snapshot({'low_dim_state': np.zeros(4)}) is None


def test_export_raw_sensors_projects_before_resize_and_converts_opengl_once(tmp_path):
    values, config = raw_inputs()
    snapshot = raw_sensor_snapshot(values, config, [0., 0., -2., 0., 0., 0., 1.])
    sensor, scene_meta = export_sensor_scene(tmp_path, 'abc', snapshot,
                                            [0., 0., -2., 0., 0., 0., 1.], [3., 0., -1.])
    assert sensor['source'] == 'raw_sensor'
    record = sensor['cameras'][0]
    assert (record['width'], record['height']) == (6, 4)
    assert record['projections']['policy']['pixel'] == [3., 2.]
    assert record['projections']['policy']['status'] == 'visible'
    assert record['projections']['heatmap']['pixel'] == [9., 2.]
    assert record['projections']['heatmap']['status'] == 'outside'
    assert record['projections']['measured']['status'] == 'visible'
    assert sensor['montage_url'] == '/visualization/files/abc/sensors.png'
    assert scene_meta['url'] == '/visualization/files/abc/scene.json'
    scene = read_scene(tmp_path)
    assert scene['coordinate_frame'] == 'env_origin_relative_world'
    assert scene['units'] == 'metres'
    assert {marker['name'] for marker in scene['markers']} == {'policy', 'heatmap', 'measured'}
    np.testing.assert_allclose(scene['points']['positions'][0], [-3.001852, 2.001235, -2.001235], atol=1e-6)
    with np.load(tmp_path / 'sensors.npz', allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['camera_0_rgb'], values['front_rgb'])
        np.testing.assert_array_equal(arrays['camera_0_depth'], values['front_depth'])
        np.testing.assert_array_equal(arrays['camera_0_intrinsics'], values['front_intrinsics'])
        np.testing.assert_array_equal(arrays['camera_0_policy_pixel'], [3., 2.])
        assert arrays['camera_0_heatmap_status'].item() == 'outside'
        assert not any('teacher' in key or 'untrusted' in key for key in arrays.files)
    with Image.open(tmp_path / 'sensors.png') as image:
        assert image.size == (12, 89)  # Two unscaled native tiles plus headers.


def test_occlusion_uses_optical_z_and_keeps_unclipped_offscreen_coordinates(tmp_path):
    snapshot = canonical_snapshot()
    sensor, _ = export_sensor_scene(tmp_path, 'entry', snapshot,
                                   [3., 0., 3., 0., 0., 0., 1.], [100., 0., 1.])
    projection = sensor['cameras'][0]['projections']
    assert projection['policy']['status'] == 'occluded'
    assert projection['policy']['sample_depth_z'] == 2.
    assert projection['policy']['pixel'] == [1., 0.]
    assert projection['heatmap']['status'] == 'outside'
    assert projection['heatmap']['pixel'] == [100., 0.]


def test_ray_distance_is_converted_to_z_for_projection_occlusion(tmp_path):
    snapshot = canonical_snapshot('ray', [[1., np.sqrt(2), np.sqrt(5)]])
    sensor, _ = export_sensor_scene(tmp_path, 'entry', snapshot,
                                   [2., 0., 2., 0., 0., 0., 1.], [1., 0., 1.])
    projection = sensor['cameras'][0]['projections']
    assert projection['policy']['status'] == 'occluded'
    assert projection['heatmap']['status'] == 'visible'
    assert projection['heatmap']['sample_depth_z'] == pytest.approx(1., abs=1e-6)
    points = read_scene(tmp_path)['points']['positions']
    np.testing.assert_allclose(points[1], [1., 0., 1.], atol=1e-6)
    np.testing.assert_allclose(points[2], [2., 0., 1.], atol=1e-6)


@pytest.mark.parametrize('point,status', [([0., 0., -1.], 'behind'), ([0., 0., 0.], 'behind'),
                                         ([-1., 0., 1.], 'outside'), ([0., 0., 1.], 'invalid_depth')])
def test_visibility_statuses_remain_explicit(tmp_path, point, status):
    snapshot = canonical_snapshot(depth=np.full((3, 4), np.nan))
    sensor, _ = export_sensor_scene(tmp_path, 'entry', snapshot, None, point)
    projection = sensor['cameras'][0]['projections']['heatmap']
    assert projection['status'] == status
    assert projection['sample_depth_z'] is None
    assert projection['pixel'] is None if status == 'behind' else projection['pixel'] is not None


def test_joint_policy_does_not_fabricate_tcp_or_orientation(tmp_path):
    export_sensor_scene(tmp_path, 'entry', canonical_snapshot(), None, [0., 0., 1.])
    scene = read_scene(tmp_path)
    assert [marker['name'] for marker in scene['markers']] == ['heatmap']
    assert scene['markers'][0]['pose_xyzw'] is None
    assert 'policy' not in scene['cameras'][0]['projections']
    with np.load(tmp_path / 'sensors.npz', allow_pickle=False) as arrays:
        assert 'policy_world' not in arrays.files


def test_point_cloud_is_bounded_camera_balanced_and_deterministic(tmp_path):
    source = canonical_snapshot(depth=np.ones((200, 200)))['cameras']['front']
    snapshot = dict(source='raw_sensor', cameras={f'camera_{index}': source for index in range(3)})
    _, metadata = export_sensor_scene(tmp_path, 'entry', snapshot, None, [0., 0., 1.])
    scene = read_scene(tmp_path)
    assert metadata['point_count'] == MAX_SCENE_POINTS
    assert len(scene['points']['positions']) == MAX_SCENE_POINTS
    assert len(scene['points']['colors']) == MAX_SCENE_POINTS
    assert np.bincount(scene['points']['camera_indices']).tolist() == [8000, 8000, 8000]
    first = (tmp_path / 'scene.json').read_bytes()
    export_sensor_scene(tmp_path, 'other_entry', snapshot, None, [0., 0., 1.])
    assert (tmp_path / 'scene.json').read_bytes() == first


def test_small_camera_budget_is_redistributed_without_oversampling(tmp_path):
    small = canonical_snapshot(depth=np.ones((1, 2)))['cameras']['front']
    large = canonical_snapshot(depth=np.ones((160, 160)))['cameras']['front']
    empty = canonical_snapshot(depth=np.full((10, 10), np.nan))['cameras']['front']
    snapshot = dict(source='raw_sensor', cameras={'small': small, 'large': large, 'empty': empty})
    export_sensor_scene(tmp_path, 'entry', snapshot, None, [0., 0., 1.])
    scene = read_scene(tmp_path)
    assert np.bincount(scene['points']['camera_indices']).tolist() == [2, MAX_SCENE_POINTS - 2]
    assert len(scene['cameras']) == 3


def test_all_invalid_depth_still_exports_images_and_empty_scene(tmp_path):
    depth = np.array([[np.nan, np.inf], [0., -1.]], np.float32)
    snapshot = canonical_snapshot(depth=depth)
    _, metadata = export_sensor_scene(tmp_path, 'entry', snapshot, None, [0., 0., 1.])
    assert metadata['point_count'] == 0
    assert read_scene(tmp_path)['points'] == dict(positions=[], colors=[], camera_indices=[])
    with np.load(tmp_path / 'sensors.npz', allow_pickle=False) as arrays:
        np.testing.assert_equal(arrays['camera_0_depth'], depth)


def test_camera_name_never_becomes_filename_or_npz_key(tmp_path):
    snapshot = canonical_snapshot()
    camera = snapshot['cameras'].pop('front')
    snapshot['cameras']['../../external/teacher'] = camera
    sensor, _ = export_sensor_scene(tmp_path, 'entry', snapshot, None, [0., 0., 1.])
    assert {path.name for path in tmp_path.iterdir()} == {'sensors.png', 'sensors.npz', 'scene.json'}
    assert sensor['cameras'][0]['name'] == '../../external/teacher'
    with np.load(tmp_path / 'sensors.npz', allow_pickle=False) as arrays:
        assert arrays['camera_names'][0] == '../../external/teacher'
        assert all('/' not in key and '..' not in key for key in arrays.files)


@pytest.mark.parametrize('mutation', [
    lambda values: values.__setitem__('front_pose', np.zeros(7)),
    lambda values: values.__setitem__('front_intrinsics', np.zeros((3, 3))),
    lambda values: values.__setitem__('front_rgb', np.zeros((4, 6, 4), np.uint8)),
    lambda values: values.__setitem__('front_depth', np.zeros((2, 3))),
])
def test_invalid_snapshot_is_a_local_diagnostic_error(mutation):
    values, config = raw_inputs()
    mutation(values)
    with pytest.raises(ValueError):
        raw_sensor_snapshot(values, config)


def test_no_sensor_data_produces_no_artifacts(tmp_path):
    assert export_sensor_scene(tmp_path, 'entry', None, None, [0., 0., 1.]) == (None, None)
    assert not list(tmp_path.iterdir())
