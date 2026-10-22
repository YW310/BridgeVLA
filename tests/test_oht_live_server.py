"""Online HTTP diagnostics use one forward and never become control inputs."""
import io
import time

import numpy as np
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from tests.test_oht_migration import replay_fixture  # noqa: F401
from tests.test_oht_vla_server import fixture, prediction  # noqa: F401


def image_payload():
    rgb = np.full((3, 8, 8, 3), .5, np.float32)
    projection = np.zeros((3, 2, 4), np.float32)
    projection[:, :, 3] = 4
    stage = dict(rgb=rgb, heatmap=np.full((3, 8, 8), 1 / 64, np.float32), world_to_pixel=projection)
    return dict(stages={'mvt1': stage, 'mvt2': stage}, heatmap_waypoint_world=np.array([.2, 0, .5]))


def wait_for_state(client, predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = client.get('/visualization/latest').json()
        if predicate(state):
            return state
        time.sleep(.01)
    raise AssertionError(f'Visualization did not finish: {state}')


def enable_images(fixture, *, fail=False):
    ordinary = fixture.policy.act
    captures = []
    def act(observation, goal, step, *, visualization_callback=None):
        result = ordinary(observation, goal, step)
        captures.append(visualization_callback is not None)
        assert not any('teacher' in name or name in ('action', 'gripper_pose') for name in observation)
        if visualization_callback is not None:
            if fail:
                visualization_callback.on_error(ValueError('projection unavailable'))
            else:
                visualization_callback(image_payload())
        return result
    fixture.policy.act = act
    return captures


def test_http_live_viewer_samples_same_forward_and_serves_only_visualization_files(fixture):
    captures = enable_images(fixture)
    with TestClient(fixture.app(visualize_every=2)) as client:
        info = client.get('/model_info').json()
        assert info['visualization']['enabled'] and info['visualization']['every'] == 2
        page = client.get('/visualization')
        assert page.status_code == 200 and 'text/html' in page.headers['content-type']
        assert 'textContent' in page.text and '/visualization/latest' in page.text
        for step in range(3):
            body = prediction(fixture, info, step=step, episode='../<script>episode</script>')
            response = client.post('/predict', json=body)
            assert response.status_code == 200, response.text
            if step % 2 == 0:
                assert response.json()['visualization']['status'] == 'queued'
                wait_for_state(client, lambda state: state['latest'] is not None and state['latest']['step'] == step)
            else:
                assert 'visualization' not in response.json()
        state = client.get('/visualization/latest').json()
        item = state['latest']
        assert item['label_free'] and item['step'] == 2
        assert item['episode_id'] == '../<script>episode</script>'
        assert len(item['entry_id']) == 32 and '/' not in item['entry_id']
        assert item['policy_tcp_world'] == pytest.approx([.2, 0, .5])
        assert not item['heatmap_is_auxiliary']
        for url in item['images'].values():
            reply = client.get(url)
            assert reply.status_code == 200
            with Image.open(io.BytesIO(reply.content)) as image:
                image.verify()
        record = client.get(item['prediction_url']).json()
        assert record['entry_id'] == item['entry_id']
        array_url = record['stages']['mvt1']['arrays']
        with np.load(io.BytesIO(client.get(array_url).content), allow_pickle=False) as arrays:
            assert 'gt_heatmap' not in arrays and 'gt_pixels' not in arrays
            np.testing.assert_allclose(arrays['heatmap'].sum(axis=(1, 2)), 1.)
        for url in (f'/visualization/files/{item["entry_id"]}/info.json',
                    '/visualization/files/not-a-token/mvt1.png',
                    '/visualization/files/%2e%2e/info.json'):
            assert client.get(url).status_code == 404
    assert captures == [True, False, True] and len(fixture.policy.calls) == 3


@pytest.mark.parametrize('disabled', [False, True])
def test_unsupported_or_disabled_visualization_keeps_original_policy_compatible(fixture, disabled):
    captures = enable_images(fixture) if disabled else []
    with TestClient(fixture.app(visualize_every=0 if disabled else 10)) as client:
        info = client.get('/model_info').json()
        assert not info['visualization']['enabled']
        assert client.get('/visualization/latest').json()['latest'] is None
        reply = client.post('/predict', json=prediction(fixture, info))
        assert reply.status_code == 200 and 'visualization' not in reply.json()
        assert not (fixture.logs / 'visualizations').exists()
    assert len(fixture.policy.calls) == 1
    if disabled:
        assert captures == [False]


def test_failed_capture_is_reported_without_failing_or_retrying_policy(fixture):
    captures = enable_images(fixture, fail=True)
    with TestClient(fixture.app(visualize_every=1)) as client:
        info = client.get('/model_info').json()
        for step in range(2):
            reply = client.post('/predict', json=prediction(fixture, info, step=step))
            assert reply.status_code == 200
            assert reply.json()['visualization']['status'] == 'error'
            assert client.get('/health').json()['episode_failed'] is False
        assert client.get('/visualization/latest').json()['capture_error']
    assert captures == [True, True] and len(fixture.policy.calls) == 2


def test_visualization_startup_failure_does_not_disable_control(fixture, monkeypatch):
    from finetune.OHT.runtime import online_visualization
    captures = enable_images(fixture)
    def unavailable(*args, **kwargs):
        raise OSError('Image directory unavailable')
    monkeypatch.setattr(online_visualization, 'OnlineVisualizer', unavailable)
    with TestClient(fixture.app()) as client:
        assert client.get('/health').json()['ready']
        info = client.get('/model_info').json()
        assert not info['visualization']['enabled']
        assert 'initialization failed' in info['visualization']['reason']
        response = client.post('/predict', json=prediction(fixture, info))
        assert response.status_code == 200
    assert captures == [False] and len(fixture.policy.calls) == 1


@pytest.mark.parametrize('setting', [{'visualize_every': -1}, {'visualize_keep': 0}])
def test_invalid_visualization_options_fail_before_loading(fixture, setting):
    with pytest.raises(ValueError, match='visualize_'):
        with TestClient(fixture.app(**setting)):
            pass
    assert not fixture.loads
