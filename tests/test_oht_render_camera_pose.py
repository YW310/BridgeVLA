"""Render extrinsics match the RGB-D frame, without a real simulator dependency."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from finetune.OHT.data.geometry import pose_matrix
from finetune.OHT.data.observation import camera_observation
from finetune.OHT.runtime.usd_cameras import ReplicatorCamera, render_camera_matrix


def camera_params(world_from_camera, units=1.):
    scene = world_from_camera.copy()
    scene[:3, 3] /= units
    return dict(cameraViewTransform=np.linalg.inv(scene).T.reshape(-1),
                metersPerSceneUnit=units)


def runtime(origin=(0., 0., 0.)):
    data = SimpleNamespace(body_pos_w=np.array([[[1., 2., 3.]]]),
                           body_quat_w=np.array([[[1., 0., 0., 0.]]]))
    return SimpleNamespace(env=SimpleNamespace(scene={'robot': SimpleNamespace(data=data)}),
                           _origin=lambda: np.array(origin), _wxyz=lambda value: np.asarray(value))


def camera(params, *, units=1., body=False, fallback=None):
    value = object.__new__(ReplicatorCamera)
    value.name = 'wrist' if body else 'global_left'
    value.metres_per_unit = units
    value.annotators = {'camera_params': SimpleNamespace(get_data=lambda: params)}
    value.body_index = 0 if body else None
    value.body_from_camera = pose_matrix([.1, -.02, .03],
                                        Rotation.from_euler('xyz', [12, -7, 18], degrees=True).as_quat())
    value.prim = object()
    value.usd_matrix = lambda prim: np.eye(4) if fallback is None else fallback
    value.pose_source, value.pose_diagnostics = None, {}
    return value


def as_matrix(pose):
    return pose_matrix(pose[:3], pose[3:][[1, 2, 3, 0]])


@pytest.mark.parametrize('units', [1., .01, 100.])
def test_inverse_transpose_and_scene_units_recover_full_rigid_transform(units):
    expected = pose_matrix([2.8, -1.4, .7],
                           Rotation.from_euler('xyz', [23, -41, 67], degrees=True).as_quat())
    params = camera_params(expected, units)
    actual = render_camera_matrix(params, units)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
    # A missing transpose incorrectly puts translation in the homogeneous row;
    # transposing without inversion produces world->camera instead of its inverse.
    assert not np.allclose(actual, np.linalg.inv(params['cameraViewTransform'].reshape(4, 4)))
    assert not np.allclose(actual, params['cameraViewTransform'].reshape(4, 4).T)


def test_published_replicator_example_is_world_to_camera_in_scene_units():
    # NVIDIA's example creates camera(position=(100,0,0)) in a centimetre stage.
    # Despite some API prose calling the field camera->world, it contains -100.
    view = np.eye(4)
    view[3, 0] = -100.
    actual = render_camera_matrix(dict(cameraViewTransform=view.reshape(-1),
                                       metersPerSceneUnit=.01), .01)
    np.testing.assert_allclose(actual[:3, 3], [1., 0., 0.])


def test_render_pose_wins_over_usd_ninety_degree_difference_and_projects_correctly():
    origin = np.array([10., -4., 0.])
    expected = pose_matrix(origin + [.2, .3, 1.],
                           Rotation.from_euler('x', 90, degrees=True).as_quat())
    fallback = expected.copy()
    fallback[:3, :3] = np.eye(3)
    selected = camera(camera_params(expected), fallback=fallback)
    pose = selected.pose(runtime(origin))
    assert selected.pose_source == 'camera_params'
    assert selected.pose_diagnostics['fallback_reason'] is None
    np.testing.assert_allclose(selected.pose_diagnostics['camera_pose_wxyz'], pose)
    expected[:3, 3] -= origin
    np.testing.assert_allclose(as_matrix(pose), expected, atol=1e-7)

    config = dict(camera_quaternion_order='wxyz', image_size=[2, 2],
                  cameras={'front': dict(intrinsics=np.eye(3), optical_to_sensor=np.diag([1., -1., -1., 1.]))},
                  depth=dict(kind='z', limits=[.001, 3.]))
    observation = camera_observation('front', np.zeros((2, 2, 3), np.uint8),
                                     np.full((2, 2), 2., np.float32), pose, config)
    np.testing.assert_allclose(observation['front_point_cloud'][:, 0, 0], [.2, 2.3, 1.], atol=1e-7)
    assert np.linalg.norm(observation['front_point_cloud'][:, 0, 0] - [.2, .3, -1.]) > 2.


def test_origin_is_subtracted_once_after_converting_view_scene_units():
    origin = [4., -3., .5]
    world = pose_matrix([4.7, -2.8, 1.4], Rotation.from_euler('z', 30, degrees=True).as_quat())
    selected = camera(camera_params(world, .01), units=.01)
    pose = selected.pose(runtime(origin))
    np.testing.assert_allclose(pose[:3], [.7, .2, .9], atol=1e-7)
    np.testing.assert_allclose(as_matrix(pose)[:3, :3], world[:3, :3], atol=1e-7)


def test_render_view_without_unit_metadata_uses_declared_stage_units():
    expected = pose_matrix([2., -1., .5], [0., 0., 0., 1.])
    params = camera_params(expected, .01)
    params.pop('metersPerSceneUnit')
    np.testing.assert_allclose(render_camera_matrix(params, .01), expected)


@pytest.mark.parametrize('bad', [None, {}, {'cameraViewTransform': None},
                               {'cameraViewTransform': np.zeros((4, 4))},
                               {'cameraViewTransform': np.ones(15)},
                               {'cameraViewTransform': np.full((4, 4), np.nan)},
                               {'cameraViewTransform': np.full((4, 4), np.inf)},
                               {'cameraViewTransform': np.diag([1., 1., -1., 1.])},
                               {'cameraViewTransform': np.diag([2., 1., 1., 1.])},
                               {'cameraViewTransform': np.eye(4), 'metersPerSceneUnit': 0.},
                               {'cameraViewTransform': np.eye(4), 'metersPerSceneUnit': .01}])
def test_invalid_view_falls_back_to_existing_static_usd_pose(bad):
    fallback = pose_matrix([1.2, -.4, .8],
                           Rotation.from_euler('xyz', [10, 20, 30], degrees=True).as_quat())
    selected = camera(bad, fallback=fallback)
    pose = selected.pose(runtime())
    np.testing.assert_allclose(as_matrix(pose), fallback, atol=1e-7)
    assert selected.pose_source == 'usd'
    assert selected.pose_diagnostics['fallback_reason']


def test_column_vector_input_is_rejected_instead_of_guessing_a_transpose():
    world = pose_matrix([1., 2., 3.], Rotation.from_euler('z', 30, degrees=True).as_quat())
    with pytest.raises(ValueError, match='homogeneous'):
        render_camera_matrix(dict(cameraViewTransform=np.linalg.inv(world)), 1.)


@pytest.mark.parametrize('units', [0., -1., np.nan, np.inf])
def test_invalid_stage_units_are_rejected(units):
    with pytest.raises(ValueError, match='units'):
        render_camera_matrix(dict(cameraViewTransform=np.eye(4)), units)


def test_wrist_valid_view_is_current_and_already_contains_mount_offset():
    state = runtime([1., 0., 0.])
    world = pose_matrix([2., 3., 4.], Rotation.from_euler('xyz', [20, 30, 40], degrees=True).as_quat())
    params = camera_params(world)
    selected = camera(params, body=True)
    selected.body_from_camera[:3, 3] = [99., 99., 99.]
    selected.usd_matrix = lambda prim: pytest.fail('Valid wrist view must not read USD pose')
    first = selected.pose(state)
    expected = world.copy()
    expected[0, 3] -= 1.
    np.testing.assert_allclose(as_matrix(first), expected, atol=1e-7)
    world = pose_matrix([2.2, 3.1, 3.8], Rotation.from_euler('xyz', [-15, 25, 70], degrees=True).as_quat())
    params.update(camera_params(world))
    second = selected.pose(state)
    expected = world.copy()
    expected[0, 3] -= 1.
    np.testing.assert_allclose(as_matrix(second), expected, atol=1e-7)
    assert selected.pose_source == 'camera_params'
    assert not np.allclose(first, second)


def test_singular_wrist_view_keeps_live_physics_and_full_mount_transform():
    state = runtime([1., 0., 0.])
    selected = camera(dict(cameraViewTransform=np.zeros((4, 4))), body=True)
    selected.usd_matrix = lambda prim: pytest.fail('Wrist fallback must use live physics pose')
    first = selected.pose(state)
    state.env.scene['robot'].data.body_pos_w[0, 0] = [2., 1., .5]
    body_rotation = Rotation.from_euler('z', 90, degrees=True).as_quat()
    state.env.scene['robot'].data.body_quat_w[0, 0] = body_rotation[[3, 0, 1, 2]]
    second = selected.pose(state)
    expected = pose_matrix([2., 1., .5], body_rotation) @ selected.body_from_camera
    expected[0, 3] -= 1.
    np.testing.assert_allclose(as_matrix(second), expected, atol=1e-7)
    assert selected.pose_source == 'physics_offset'
    assert selected.pose_diagnostics['fallback_reason']
    assert not np.allclose(first, second)


def test_pose_source_can_recover_after_temporary_invalid_render_view(capsys):
    expected = pose_matrix([.2, .3, 1.], Rotation.from_euler('x', 90, degrees=True).as_quat())
    params = dict(cameraViewTransform=np.zeros((4, 4)))
    selected = camera(params)
    selected.pose(runtime())
    assert selected.pose_source == 'usd'
    params.update(camera_params(expected))
    np.testing.assert_allclose(as_matrix(selected.pose(runtime())), expected, atol=1e-7)
    assert selected.pose_source == 'camera_params'
    assert selected.pose_diagnostics['fallback_reason'] is None
    output = capsys.readouterr().out
    assert 'pose_source=usd' in output and 'pose_source=camera_params' in output
