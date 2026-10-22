"""Actual BridgeVLA forwards through HTTP, with optional label-free diagnostics."""
import io
from pathlib import Path
import time

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from tests.test_oht_migration import replay_fixture  # noqa: F401
from tests.test_oht_action_modes import caches  # noqa: F401
from tests.test_oht_eval_visualization import ImageMVT
from tests.test_oht_vla_server import fixture, prediction  # noqa: F401
from finetune.OHT.action_modes import MODES, apply_eef_delta, resolve_action
from finetune.OHT.config import load
from finetune.OHT.data.dataset import OHTDataset
from finetune.OHT.model import build_agent
from finetune.OHT.runtime.policy import Policy
from finetune.OHT.runtime.predicted_wrapper import PredictedObjectWrapper, policy_observation
from finetune.OHT.vla_server import create_app


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {'action', 'gripper_pose', 'trans_action_indicies', 'rot_grip_action_indicies',
             'ignore_collisions', 'joint_action', 'objects_pos', 'objects_quat'}


def _build(replay_fixture, caches, monkeypatch, *, action_mode='eef', object_mode='baseline'):
    from bridgevla.mvt import mvt
    monkeypatch.setattr(mvt, 'MVT', ImageMVT)
    config = load(ROOT / f'finetune/OHT/configs/{object_mode}.yaml')
    config.update(action_mode=action_mode, point_count=8)
    config['action']['hidden_dim'] = 16
    options = resolve_action(config)
    dataset = OHTDataset(replay_fixture.replay, 'val', action_config=options)
    agent = build_agent(config, dataset.contract, torch.device('cpu'), training=False)
    agent.eval()
    observed_batches, predictor_calls = [], []
    original_act = agent.act

    def audit_act(step, observation, **kwargs):
        assert not FORBIDDEN.intersection(observation)
        assert not any(key.startswith(('teacher_', 'next.')) for key in observation)
        observed_batches.append(set(observation))
        return original_act(step, observation, **kwargs)

    monkeypatch.setattr(agent, 'act', audit_act)
    wrapper = None
    if object_mode == 'predicted_external':
        def predictor(observation, goal):
            assert not FORBIDDEN.intersection(observation)
            assert not any(key.startswith(('teacher_', 'oracle_')) for key in observation)
            predictor_calls.append((set(observation), goal))
            return caches[1]
        wrapper = PredictedObjectWrapper(predictor, dataset.contract['data_config']['cameras'], 8)
    policy = Policy(agent, dataset.contract, 'cpu', object_mode, wrapper)
    policy.training_config = config
    return policy, dataset[0], observed_batches, predictor_calls


def _published(client):
    deadline = time.monotonic() + 5
    state = None
    while time.monotonic() < deadline:
        state = client.get('/visualization/latest').json()
        if state['latest'] is not None:
            assert state['writer_status'] != 'error', state
            return state['latest']
        if state['writer_status'] == 'error':
            raise AssertionError(state)
        time.sleep(.01)
    raise AssertionError(f'No online diagnostic was published: {state}')


@pytest.mark.parametrize('action_mode', MODES)
@pytest.mark.parametrize('object_mode', ['baseline', 'role_queries', 'predicted_external'])
def test_actual_http_policy_visualization_preserves_action_and_single_forward(
        fixture, replay_fixture, caches, monkeypatch, tmp_path, action_mode, object_mode):
    policy, row, observed, predictions = _build(replay_fixture, caches, monkeypatch,
                                               action_mode=action_mode, object_mode=object_mode)
    options = policy.action_config
    fixture.observation = policy_observation(row, policy.contract['data_config']['cameras'], options)
    fixture.goal = row['goal']
    # Passing a training row here also verifies that Policy strips labels before inference.
    expected = policy.act(row, row['goal'], 0)
    initial_calls = policy.agent._net_mod.calls
    assert initial_calls == 1
    enabled = create_app(fixture.checkpoint, output_dir=tmp_path/'enabled', policy_loader=lambda *a: policy)
    with TestClient(enabled) as client:
        info = client.get('/model_info').json()
        assert info['visualization']['enabled'] and info['visualization']['every'] == 10
        reply = client.post('/predict', json=prediction(fixture, info))
        assert reply.status_code == 200, reply.text
        np.testing.assert_array_equal(np.asarray(reply.json()['action'], np.float32), expected)
        assert policy.agent._net_mod.calls == initial_calls + 1
        assert reply.json()['visualization']['status'] == 'queued'
        item = _published(client)
        assert item['label_free'] is True and item['heatmap_is_auxiliary'] == (action_mode != 'eef')
        assert 'target_action' not in item and 'gt_tcp_world' not in item
        if action_mode == 'eef':
            np.testing.assert_allclose(item['policy_tcp_world'], expected[:3], atol=1e-7)
        elif action_mode == 'relative_eef':
            decoded = apply_eef_delta(row['tcp_pose'], expected[:6], options)
            np.testing.assert_allclose(item['policy_tcp_world'], decoded[:3], atol=1e-7)
        else:
            assert item['policy_tcp_world'] is None and item['policy_pose_xyzw'] is None
        assert set(item['stages']) == {'mvt1', 'mvt2'}
        for stage in item['stages'].values():
            response = client.get(stage['arrays'])
            assert response.status_code == 200
            with np.load(io.BytesIO(response.content), allow_pickle=False) as arrays:
                assert not any(key.startswith(('gt_', 'teacher_')) for key in arrays.files)
                assert arrays['world_to_pixel'].shape == (3, 2, 4)
                np.testing.assert_allclose(arrays['heatmap'].sum(axis=(1, 2)), 1., atol=1e-6)
                assert ('policy_pixels' in arrays) == ('joint' not in action_mode)
                if object_mode != 'baseline':
                    assert {'target_prior', 'reference_prior'} <= set(arrays.files)
    disabled = create_app(fixture.checkpoint, output_dir=tmp_path/'disabled',
                          policy_loader=lambda *a: policy, visualize_every=0)
    with TestClient(disabled) as client:
        info = client.get('/model_info').json()
        assert info['visualization']['enabled'] is False
        reply = client.post('/predict', json=prediction(fixture, info))
        assert reply.status_code == 200, reply.text
        np.testing.assert_array_equal(np.asarray(reply.json()['action'], np.float32), expected)
        assert 'visualization' not in reply.json()
        assert policy.agent._net_mod.calls == initial_calls + 2
        assert not (tmp_path/'disabled/visualizations').exists()
    assert len(observed) == 3
    assert len(predictions) == (3 if object_mode == 'predicted_external' else 0)


def test_real_diagnostic_build_failure_keeps_http_action_and_offline_callback_strict(
        fixture, replay_fixture, caches, monkeypatch, tmp_path):
    from bridgevla.models import bridgevla_agent
    policy, row, observed, _ = _build(replay_fixture, caches, monkeypatch)
    fixture.observation = policy_observation(row, policy.contract['data_config']['cameras'], policy.action_config)
    fixture.goal = row['goal']
    expected = policy.act(row, row['goal'], 0)
    build_calls = []

    def broken(*args, **kwargs):
        build_calls.append(True)
        raise RuntimeError('private projection diagnostic failure')

    monkeypatch.setattr(bridgevla_agent, 'build_policy_visualization_payload', broken)
    app = create_app(fixture.checkpoint, output_dir=tmp_path/'broken', policy_loader=lambda *a: policy)
    with TestClient(app) as client:
        info = client.get('/model_info').json()
        assert info['visualization']['enabled']
        for step in range(2):
            reply = client.post('/predict', json=prediction(fixture, info, step=step))
            assert reply.status_code == 200, reply.text
            np.testing.assert_array_equal(np.asarray(reply.json()['action'], np.float32), expected)
            assert policy.agent._net_mod.calls == step + 2
            health = client.get('/health').json()
            assert health['episode_failed'] is False and health['next_step'] == step + 1
            if step == 0:
                assert reply.json()['visualization']['status'] == 'error'
            else:
                assert 'visualization' not in reply.json()
        assert len(build_calls) == 1
        assert client.get('/visualization/latest').json()['capture_error']
    # Existing offline evaluation uses a plain callback and must still fail strictly.
    with pytest.raises(RuntimeError, match='private projection diagnostic failure'):
        policy.act(row, row['goal'], 0, visualization_callback=lambda payload: None)
    assert len(build_calls) == 2 and policy.agent._net_mod.calls == 4 and len(observed) == 4
