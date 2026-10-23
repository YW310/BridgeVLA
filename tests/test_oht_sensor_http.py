"""HTTP sensor diagnostics preserve raw camera geometry and never alter control."""
from copy import deepcopy
import io

import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from tests.test_oht_migration import replay_fixture  # noqa: F401
from tests.test_oht_vla_server import fixture, prediction  # noqa: F401
from tests.test_oht_isaaclab_http import Opener, raw_sample
from tests.test_oht_live_server import enable_images, wait_for_state
from finetune.OHT.runtime.http_client import BridgeVLAClient
from finetune.OHT.runtime.raw_observation import data_profiles, decode_raw
from finetune.OHT.runtime.transport import pack_observation


def raw_cameras(replay_fixture, policy, shape=(16, 24)):
    """Use full-resolution metric-Z cameras whose optical frame is world aligned."""
    tensors, row, packed = raw_sample(replay_fixture.replay, policy)
    config = data_profiles(policy)[packed['data_profile']]
    offset = np.eye(4)
    offset[:3, 3] = [.03, .02, .04]
    config['link_to_tcp'] = offset.tolist()
    tensors['ee_link_pose'] = np.array([.1, 0, .5, 1, 0, 0, 0], np.float32)
    height, width = shape
    yy, xx = np.indices(shape)
    rgb = np.stack((xx % 256, yy % 256, (xx + yy) % 256), axis=-1).astype(np.uint8)
    depth = np.ones(shape, np.float32)
    depth[0, 0] = np.nan
    intrinsics = np.array([[width / 2, 0, width / 2], [0, height * .625, height / 2],
                           [0, 0, 1]], np.float32)
    for name in config['cameras']:
        tensors.update({name + '_rgb': rgb.copy(), name + '_depth': depth.copy(),
                        name + '_intrinsics': intrinsics.copy(),
                        name + '_pose': np.array([0, 0, 0, 0, 1, 0, 0], np.float32)})
    packed['tensors'] = pack_observation(tensors)
    return tensors, row['goal'], packed


def publish(client, response):
    assert response['visualization']['status'] == 'queued'
    entry_id = response['visualization']['entry_id']
    return wait_for_state(client, lambda state: state['latest'] is not None
                          and state['latest']['entry_id'] == entry_id)['latest']


def assert_label_free(record):
    if isinstance(record, dict):
        assert not any(str(key).startswith(('gt_', 'teacher_', 'oracle_')) for key in record)
        assert 'target_action' not in record
        for value in record.values():
            assert_label_free(value)
    elif isinstance(record, list):
        for value in record:
            assert_label_free(value)


@pytest.mark.parametrize('shape', [(16, 24), (160, 240)])
def test_raw_http_exports_full_sensor_resolution_once_and_correct_world_geometry(
        fixture, replay_fixture, monkeypatch, shape):
    from finetune.OHT.runtime import raw_observation

    tensors, instruction, packed = raw_cameras(replay_fixture, fixture.policy, shape)
    original = {name: value.copy() for name, value in tensors.items()}
    expected, _ = decode_raw(deepcopy(packed), fixture.policy)
    unpack_calls = []
    ordinary_unpack = raw_observation.unpack_observation

    def audit_unpack(values):
        unpack_calls.append(set(values))
        return ordinary_unpack(values)

    monkeypatch.setattr(raw_observation, 'unpack_observation', audit_unpack)
    captures = enable_images(fixture)
    with TestClient(fixture.app(visualize_every=1)) as server:
        client = BridgeVLAClient('http://testserver', opener=Opener(server))
        client.reset()
        action, response = client.predict(tensors, instruction, 0., task=packed['task'],
                                          data_profile=packed['data_profile'])
        item = publish(server, response)
        assert captures == [True] and len(fixture.policy.calls) == len(unpack_calls) == 1
        np.testing.assert_allclose(action, [.2, 0, .5, 0, 0, 0, 1, 1])
        observed = fixture.policy.calls[0][0]
        assert set(observed) == set(expected)
        for name in expected:
            np.testing.assert_allclose(observed[name], expected[name], equal_nan=True)
        for name, value in original.items():
            np.testing.assert_array_equal(tensors[name], value)
        assert item['sensors']['source'] == item['scene']['source'] == 'raw_sensor'
        assert len(item['sensors']['cameras']) == len(fixture.policy.contract['data_config']['cameras'])
        assert item['scene']['point_count'] <= 24000
        with Image.open(io.BytesIO(server.get(item['sensors']['montage_url']).content)) as image:
            image.verify()
        arrays_response = server.get(item['sensors']['arrays_url'])
        assert arrays_response.status_code == 200
        scene_response = server.get(item['scene']['url'])
        assert scene_response.status_code == 200
        scene = scene_response.json()
        points = np.asarray(scene['points']['positions'])
        assert len(points) == item['scene']['point_count'] and points.shape[1] == 3
        assert len(points) > 0 and np.isfinite(points).all()
        np.testing.assert_allclose(points[:, 2], 1., atol=1e-7)
        assert len(scene['points']['colors']) == len(scene['points']['camera_indices']) == len(points)
        for camera in scene['cameras']:
            assert (camera['height'], camera['width']) == shape
            np.testing.assert_allclose(camera['world_from_optical'], np.eye(4), atol=1e-7)
            projection = camera['projections']['policy']
            np.testing.assert_allclose(projection['pixel'], [.7 * shape[1], shape[0] / 2], atol=1e-5)
            assert projection['optical_depth'] == pytest.approx(.5)
        measured = next(marker for marker in scene['markers'] if marker['name'] == 'measured')
        np.testing.assert_allclose(measured['position'], [.13, .02, .54], atol=1e-7)
        assert_label_free(item)
        assert_label_free(scene)
        with np.load(io.BytesIO(arrays_response.content), allow_pickle=False) as arrays:
            assert all(not name.startswith(('gt_', 'teacher_', 'oracle_')) for name in arrays.files)
            assert arrays['source'].item() == 'raw_sensor'
            for index, camera in enumerate(scene['cameras']):
                name = camera['name']
                np.testing.assert_array_equal(arrays[f'camera_{index}_rgb'], tensors[name + '_rgb'])
                np.testing.assert_array_equal(arrays[f'camera_{index}_depth'], tensors[name + '_depth'])
                np.testing.assert_array_equal(arrays[f'camera_{index}_intrinsics'], tensors[name + '_intrinsics'])
                assert arrays[f'camera_{index}_rgb'].shape[:2] == shape
            np.testing.assert_allclose(arrays['measured_world'], [.13, .02, .54], atol=1e-7)
        for filename in ('private.npy', 'info.json', 'sensors.txt', 'scene.npz'):
            assert server.get(f'/visualization/files/{item["entry_id"]}/{filename}').status_code == 404
        for entry_id in ('not-a-token', '%2e%2e', 'a' * 31, 'a' * 33):
            assert server.get(f'/visualization/files/{entry_id}/sensors.png').status_code == 404


def test_prepared_http_labels_sensor_resolution_as_prepared_fallback(fixture):
    enable_images(fixture)
    with TestClient(fixture.app(visualize_every=1)) as server:
        info = server.get('/model_info').json()
        reply = server.post('/predict', json=prediction(fixture, info))
        assert reply.status_code == 200, reply.text
        item = publish(server, reply.json())
        assert item['sensors']['source'] == item['scene']['source'] == 'prepared_observation'
        scene = server.get(item['scene']['url']).json()
        assert all((camera['height'], camera['width']) == (8, 8) for camera in scene['cameras'])
        assert len(fixture.policy.calls) == 1
        assert_label_free(scene)


@pytest.mark.parametrize('failure', ['raw_sensor_snapshot', 'export_sensor_scene'])
def test_sensor_diagnostic_failure_keeps_control_and_heatmaps_without_retry(
        fixture, replay_fixture, monkeypatch, failure):
    from finetune.OHT.runtime import sensor_visualization

    tensors, instruction, packed = raw_cameras(replay_fixture, fixture.policy)
    captures = enable_images(fixture)
    ordinary = getattr(sensor_visualization, failure)
    failures = []

    def unavailable(*args, **kwargs):
        failures.append(failure)
        raise RuntimeError('Sensor diagnostic unavailable')

    monkeypatch.setattr(sensor_visualization, failure, unavailable)
    with TestClient(fixture.app(visualize_every=1)) as server:
        client = BridgeVLAClient('http://testserver', opener=Opener(server))
        client.reset()
        action, response = client.predict(tensors, instruction, 0., task=packed['task'],
                                          data_profile=packed['data_profile'])
        item = publish(server, response)
        assert failures == [failure] and len(fixture.policy.calls) == 1
        assert not server.get('/health').json()['episode_failed']
        assert item['images'] and all(server.get(url).status_code == 200 for url in item['images'].values())
        np.testing.assert_allclose(action, [.2, 0, .5, 0, 0, 0, 1, 1])
        if failure == 'raw_sensor_snapshot':
            # Raw capture failure may fall back to already prepared RGB-D.
            assert item['sensors'] is None or item['sensors']['source'] == 'prepared_observation'
        else:
            assert 'Sensor diagnostic unavailable' in item['sensors']['error']
            assert item['scene'] is None
            for filename in ('sensors.png', 'sensors.npz', 'scene.json'):
                assert server.get(f'/visualization/files/{item["entry_id"]}/{filename}').status_code == 404
        monkeypatch.setattr(sensor_visualization, failure, ordinary)
        _, recovered_response = client.predict(tensors, instruction, .1, task=packed['task'],
                                                data_profile=packed['data_profile'])
        recovered = publish(server, recovered_response)
        assert recovered['sensors']['source'] == 'raw_sensor' and recovered['scene']['point_count'] > 0
        assert captures == [True, True] and len(fixture.policy.calls) == 2


def test_raw_sensor_snapshots_follow_existing_online_sampling_interval(
        fixture, replay_fixture, monkeypatch):
    from finetune.OHT.runtime import sensor_visualization

    tensors, instruction, packed = raw_cameras(replay_fixture, fixture.policy)
    snapshots = []
    ordinary = sensor_visualization.raw_sensor_snapshot

    def snapshot(*args, **kwargs):
        snapshots.append(True)
        return ordinary(*args, **kwargs)

    monkeypatch.setattr(sensor_visualization, 'raw_sensor_snapshot', snapshot)
    captures = enable_images(fixture)
    with TestClient(fixture.app(visualize_every=2)) as server:
        client = BridgeVLAClient('http://testserver', opener=Opener(server))
        client.reset()
        for step in range(3):
            _, response = client.predict(tensors, instruction, step * .1, task=packed['task'],
                                          data_profile=packed['data_profile'])
            if step % 2 == 0:
                item = publish(server, response)
                assert item['step'] == step
            else:
                assert 'visualization' not in response
        assert snapshots == [True, True] and captures == [True, False, True]
        assert len(fixture.policy.calls) == 3


def test_raw_snapshot_failure_preserves_ray_depth_semantics_in_prepared_fallback(
        fixture, replay_fixture, monkeypatch):
    from finetune.OHT.runtime import sensor_visualization

    tensors, instruction, packed = raw_cameras(replay_fixture, fixture.policy)
    profile = data_profiles(fixture.policy)[packed['data_profile']]
    profile['depth']['kind'] = 'ray'
    for camera in profile['cameras'].values():
        if 'depth' in camera:
            camera['depth']['kind'] = 'ray'
    enable_images(fixture)

    def unavailable(*args, **kwargs):
        raise RuntimeError('Raw sensor snapshot unavailable')

    monkeypatch.setattr(sensor_visualization, 'raw_sensor_snapshot', unavailable)
    with TestClient(fixture.app(visualize_every=1)) as server:
        client = BridgeVLAClient('http://testserver', opener=Opener(server))
        client.reset()
        _, response = client.predict(tensors, instruction, 0., task=packed['task'],
                                      data_profile=packed['data_profile'])
        item = publish(server, response)
        assert item['sensors']['source'] == 'prepared_observation'
        scene = server.get(item['scene']['url']).json()
        assert all(camera['depth_kind'] == 'ray' for camera in scene['cameras'])
        # Wire depth is Z=1; canonical ray depth must not be reinterpreted as Z.
        np.testing.assert_allclose(np.asarray(scene['points']['positions'])[:, 2], 1., atol=.001)
        assert len(fixture.policy.calls) == 1
