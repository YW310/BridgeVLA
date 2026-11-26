"""Continuous, periodic joint features shared by chunk training and serving.

Toy decoders verify contracts and gradients; these tests do not establish task
performance or migrate old scalar checkpoints to new feature semantics.
"""
import copy

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from finetune.OHT.chunk_config import chunk_settings
from finetune.OHT.config import load
from finetune.OHT.model import load_weights, model_state, resume_fingerprint
from finetune.OHT.runtime.inference_api import OBSERVATION_FORMAT
from finetune.OHT.runtime.policy import Policy
from finetune.OHT.runtime.transport import pack_observation
from finetune.OHT.spatial_chunk import SpatialChunkDecoder
from finetune.OHT.vla_server import create_app
from tests.test_oht_chunk_state import state_config
from tests.test_oht_chunk_temporal import temporal_agent
from tests.test_oht_chunk_replay import chunk_fixture, replay_fixture  # noqa: F401
from tests.test_oht_spatial_chunk import inputs
from tests.test_oht_vla_server import envelope


KINDS = [('regression', 'relative_pose'), ('regression', 'delta_pose'),
         ('flow_matching', 'relative_pose')]
DEPLOYED_CENTER = [.5836237419396639, -2.033438813686371, 1.777019736170769,
                   -2.7411230653524394, -1.7244551002979278, -2.5896496739238497, .5]
DEPLOYED_SCALE = [.5356327902525664, .5973584771156311, .5068113118410108,
                  1.8865561157464985, .8967279732227327, 2.561074999347329, .5]


def periodic_config(kind='regression', representation='delta_pose'):
    config = state_config(kind, representation)
    config['chunk_policy'].update(joint_encoding='sin_cos',
                                  proprio_center=DEPLOYED_CENTER.copy(),
                                  proprio_scale=DEPLOYED_SCALE.copy())
    return config


def test_periodic_presets_opt_in_without_mutating_legacy_experiments():
    source = load('finetune/OHT/configs/oht_chunk_v4_delta_state_tcp_guided_calibrated_8x40.yaml')
    periodic = load('finetune/OHT/configs/oht_chunk_v4_delta_state_tcp_guided_periodic_8x40.yaml')
    repair = load('finetune/OHT/configs/oht_chunk_v4_delta_state_tcp_guided_repair_8x40.yaml')
    original, updated = chunk_settings(source), chunk_settings(periodic)
    assert original['joint_encoding'] == 'scalar'
    assert updated['joint_encoding'] == chunk_settings(repair)['joint_encoding'] == 'sin_cos'
    assert {key for key in original if original[key] != updated[key]} == {'joint_encoding'}
    assert source['exp_id'] != periodic['exp_id']
    for key in ('epochs', 'global_batch_size', 'learning_rate', 'freeze_gemma_prefix_layers',
                'freeze_vision_tower', 'freeze_multimodal_projector'):
        assert source[key] == periodic[key]
    assert repair['chunk_policy']['max_grad_norm'] == 1.


@pytest.mark.parametrize('value', [None, True, 1, '', 'sincos', 'raw', [], {}])
def test_invalid_joint_encodings_fail_before_model_construction(value):
    config = periodic_config()
    config['chunk_policy']['joint_encoding'] = value
    with pytest.raises(ValueError, match='joint_encoding'):
        chunk_settings(config)


def test_periodic_features_require_spatial_decoder():
    config = dict(mode='baseline', chunk_policy=dict(enabled=True, joint_encoding='sin_cos'))
    with pytest.raises(ValueError, match='spatial'):
        chunk_settings(config)


@pytest.mark.parametrize('normalized', [True, False])
def test_periodic_state_layout_uses_radians_and_preserves_measurements(normalized):
    config = periodic_config()
    if not normalized:
        config['chunk_policy'].update(proprio_normalization='none', proprio_center=None, proprio_scale=None)
    decoder = SpatialChunkDecoder(4, chunk_settings(config), 3, 1 / 60)
    _, context, _, _ = inputs('regression')
    state = context[2].clone()
    state[:, 7:13] = torch.tensor([.551843, -.8, 2.3, -5.8, 6.1, .552043])
    state[:, 13] = .8
    before = state.clone()
    encoded = decoder.state_features(state)
    assert decoder.state.in_features == encoded.shape[-1] == 20
    torch.testing.assert_close(encoded[:, :7], torch.tensor([0., 0., 0., 0., 0., 0., 1.]).expand(2, -1))
    torch.testing.assert_close(encoded[:, 7:13], torch.sin(state[:, 7:13]))
    torch.testing.assert_close(encoded[:, 13:19], torch.cos(state[:, 7:13]))
    torch.testing.assert_close(encoded[:, 19], torch.full((2,), .6 if normalized else .8))
    torch.testing.assert_close(state, before, rtol=0, atol=0)
    if normalized:
        # Joint quantiles no longer change periodic angles. Opening retains
        # its own physical normalization; raw quantiles remain diagnostic data.
        with torch.no_grad():
            decoder.proprio_center[:6].add_(9.)
            decoder.proprio_scale[:6].mul_(3.)
        torch.testing.assert_close(decoder.state_features(state), encoded, rtol=0, atol=0)


@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('kind,representation', KINDS)
@pytest.mark.parametrize('side', [-1, 1])
def test_every_old_joint_cut_is_continuous_with_finite_gradients(training, kind, representation, side):
    decoder = SpatialChunkDecoder(4, chunk_settings(periodic_config(kind, representation)), 3, 1 / 60).train(training)
    # Include the measured q6 cut +0.55194298 and each joint's lower cut.
    centers = torch.tensor(DEPLOYED_CENTER[:6], dtype=torch.float64)
    for joint in range(6):
        state = torch.zeros(2, 14, dtype=torch.float64)
        state[:, 6] = 1.
        state[:, 7:13] = centers
        state[:, 13] = .75
        state[:, 7 + joint] = centers[joint] + side * np.pi + torch.tensor([-1.e-4, 1.e-4])
        state.requires_grad_()
        features = decoder.state_features(state)
        assert float(torch.max(torch.abs(features[1] - features[0]))) < 2.01e-4
        # A periodic embedding must retain local sensitivity, rather than
        # replacing a discontinuity by clamping or a constant feature.
        weights = torch.arange(1., 13., dtype=features.dtype)
        gradient = torch.autograd.grad((features[:, 7:19] * weights).sum(), state)[0]
        assert torch.isfinite(gradient).all()
        expected = ((joint + 1) * torch.cos(state[:, 7 + joint])
                    - (joint + 7) * torch.sin(state[:, 7 + joint]))
        torch.testing.assert_close(gradient[:, 7 + joint], expected)
        torch.testing.assert_close(gradient[0, 7:13], gradient[1, 7:13], atol=.003, rtol=0)


@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('kind,representation', KINDS)
def test_periodic_features_and_predictions_match_equivalent_joint_turns(training, kind, representation):
    decoder = SpatialChunkDecoder(4, chunk_settings(periodic_config(kind, representation)), 3, 1 / 60).train(training)
    _, context, _, _ = inputs('regression')
    state = context[2].clone()
    state[:, 7:13] = torch.tensor([.55, -.8, 1.4, -2.7, 4.8, -.3])
    context = (*context[:2], state, *context[3:])
    shifted = state.clone()
    shifted[:, 7:13] += torch.tensor([1, -1, 2, -2, 3, -3]) * (2 * np.pi)
    altered = (*context[:2], shifted, *context[3:])
    torch.testing.assert_close(decoder.state_features(state), decoder.state_features(shifted), atol=2e-6, rtol=1e-5)
    for first, second in zip(decoder(*context), decoder(*altered)):
        torch.testing.assert_close(first, second, atol=3e-6, rtol=1e-5)


def test_absent_encoding_preserves_legacy_scalar_cut_and_checkpoint_dimensions():
    config = periodic_config()
    config['chunk_policy'].pop('joint_encoding')
    decoder = SpatialChunkDecoder(4, chunk_settings(config), 3, 1 / 60)
    state = torch.zeros(2, 14)
    state[:, 6] = 1.
    state[:, 7:13] = torch.tensor(DEPLOYED_CENTER[:6])
    cut = DEPLOYED_CENTER[5] + np.pi
    state[:, 12] = torch.tensor([cut - 1.e-4, cut + 1.e-4])
    features = decoder.state_features(state)
    assert decoder.state.in_features == features.shape[-1] == 14
    assert float(features[0, 12] - features[1, 12]) > 2.4
    config['chunk_policy']['joint_encoding'] = 'scalar'
    explicit = SpatialChunkDecoder(4, chunk_settings(config), 3, 1 / 60)
    explicit.load_state_dict(decoder.state_dict(), strict=True)
    torch.testing.assert_close(features, explicit.state_features(state), rtol=0, atol=0)


@pytest.mark.parametrize('source_encoding,target_encoding', [('scalar', 'sin_cos'), ('sin_cos', 'scalar')])
def test_encoding_change_rejects_resume_and_initializes_only_encoder(chunk_fixture, source_encoding, target_encoding):
    config = periodic_config()
    config['chunk_policy']['joint_encoding'] = source_encoding
    source, data, config = temporal_agent(chunk_fixture.root, config=config)
    with torch.no_grad():
        for parameter in source._net_mod.encoder.parameters():
            parameter.add_(.05)
        source._net_mod.decoder.head.bias.add_(.2)
    saved = dict(config=copy.deepcopy(config), model_state=copy.deepcopy(model_state(source)))
    target_config = copy.deepcopy(config)
    target_config['chunk_policy']['joint_encoding'] = target_encoding
    target, _, target_config = temporal_agent(data.root, config=target_config)
    before = copy.deepcopy(model_state(target))
    with pytest.raises(ValueError, match='mismatch'):
        load_weights(target, saved)
    for name, value in model_state(target).items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert resume_fingerprint(config, data.contract, None, 1) != resume_fingerprint(target_config, data.contract, None, 1)
    load_weights(target, saved, initialize=True)
    for name, value in model_state(target).items():
        expected = saved['model_state'][name] if name.startswith('encoder.') else before[name]
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


@pytest.mark.parametrize('encoding', [None, 'sin_cos'])
def test_checkpoint_roundtrip_preserves_selected_encoding_and_predictions(chunk_fixture, tmp_path, encoding):
    config = periodic_config()
    if encoding is None:
        config['chunk_policy'].pop('joint_encoding')
    source, data, config = temporal_agent(chunk_fixture.root, config=config)
    sample = data[0]
    expected = source.predict(sample, sample['goal'])['actions']
    path = tmp_path / 'checkpoint.pth'
    torch.save(dict(config=config, oht_contract=data.contract, model_state=model_state(source)), path)
    saved = torch.load(path, map_location='cpu', weights_only=False)
    target, _, _ = temporal_agent(data.root, config=copy.deepcopy(saved['config']))
    load_weights(target, saved)
    assert target.settings['joint_encoding'] == (encoding or 'scalar')
    np.testing.assert_array_equal(expected, target.predict(sample, sample['goal'])['actions'])


def test_http_advertises_periodic_encoding_and_matches_direct_policy(chunk_fixture, tmp_path):
    agent, data, config = temporal_agent(chunk_fixture.root, config=periodic_config())
    policy = Policy(agent, data.contract, 'cpu', 'baseline')
    policy.training_config = config
    sample = data[0]
    names = {'low_dim_state', 'tcp_pose', 'joint_positions', 'gripper_open01'}
    observation = {key:value for key,value in sample.items() if key in names or any(
        key.endswith('_' + suffix) for suffix in ('rgb', 'depth', 'point_cloud', 'camera_intrinsics', 'camera_extrinsics'))}
    policy.reset()
    expected = policy.act(observation, sample['goal'], 0)
    path = tmp_path/'checkpoint.pth'
    path.write_bytes(b'injected CPU policy')
    app = create_app(path, policy_loader=lambda *args:policy, output_dir=tmp_path/'logs', visualize_every=0)
    with TestClient(app) as client:
        info = client.get('/model_info').json()
        assert info['chunk_policy']['joint_encoding'] == 'sin_cos'
        assert info['observation_schema']['joint_positions']['shape'] == [6]
        body = dict(**envelope(info), step=0, timestamp=0., instruction=sample['goal'],
                    prediction_schema=info['chunk_spec']['schema'],
                    observation_format=OBSERVATION_FORMAT, observation=pack_observation(observation))
        response = client.post('/predict', json=body)
        assert response.status_code == 200, response.text
        np.testing.assert_allclose(response.json()['actions'], expected['actions'], atol=1e-7, rtol=1e-6)
