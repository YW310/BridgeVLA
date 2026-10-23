"""CPU decoder, checkpoint, HTTP, held-out metrics and timed-control contracts."""
import copy
import io
import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from fastapi.testclient import TestClient

from tests.test_oht_migration import replay_fixture  # noqa: F401
from tests.test_oht_chunk_replay import chunk_fixture  # noqa: F401
from tests.test_oht_vla_server import envelope
from finetune.OHT.chunk_config import chunk_settings
from finetune.OHT.action_modes import action_options
from finetune.OHT.chunk_model import ChunkAgent, ChunkDecoder, chunk_loss, quaternion_matrix, rotation_6d
from finetune.OHT.config import load
from finetune.OHT.data.chunk_dataset import OHTChunkDataset, collate_chunk
from finetune.OHT.runtime.policy import Policy
from finetune.OHT.runtime.chunk_protocol import chunk_spec, validate_prediction
from finetune.OHT.runtime.chunk_executor import ChunkExecutor
from finetune.OHT.runtime.http_client import BridgeVLAClient
from finetune.OHT.runtime.transport import pack_observation
from finetune.OHT.runtime.inference_api import OBSERVATION_FORMAT
from finetune.OHT.vla_server import create_app
from finetune.OHT.eval import open_loop


def settings_config():
    config = load('finetune/OHT/configs/oht_chunk_v4_8x40.yaml')
    config['chunk_policy'].update(stage='frozen_key', horizon=4, execute_horizon=2, hidden_dim=16, layers=1, heads=2)
    return config


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mvt1 = nn.Linear(8, 8)
        self.mvt1.vlm_dim = 8

    def forward(self, observation):
        assert not self.training
        assert not {'key_action', 'key_valid', 'action_chunk', 'action_valid'} & set(observation)
        tokens = self.mvt1(torch.ones(1, 6, 8))
        return dict(oht_chunk_tokens=tokens, mvt2=dict(oht_chunk_tokens=tokens + .1))


class KeyAgent:
    def __init__(self):
        self._net_mod = Encoder()

    def reset(self):
        pass

    def act(self, step, observation, **kwargs):
        self._net_mod(observation)
        callback = kwargs.get('visualization_callback')
        if callback:
            callback(dict(stages={}, heatmap_waypoint_world=np.array([.2, 0, .5])))
        return np.array([.2, 0, .5, 0, 0, 0, 1, 1], np.float32)


def agent_data(chunk_fixture, training=True):
    data = OHTChunkDataset(chunk_fixture.root)
    config = settings_config()
    agent = ChunkAgent(KeyAgent(), chunk_settings(config, data.contract), data.contract,
                       config, torch.device('cpu'), training=training)
    return agent, data, config


def test_chunk_config_inherits_epoch_adamw_and_rejects_inconsistent_modes():
    config = load('finetune/OHT/configs/oht_chunk_v4_8x40.yaml')
    assert config['epochs'] == 100 and config['global_batch_size'] == 192
    assert config['agent']['optimizer_type'] == 'adamw'
    assert config['freeze_vision_tower'] and config['freeze_gemma_prefix_layers'] == 6
    assert not config['agent']['transform_augmentation']
    assert not config['freeze_language_model'] and config['chunk_policy']['stage'] == 'joint'
    for patch in ({'execute_horizon': 17}, {'stage': 'unknown'}, {'heads': 3}, {'dropout': 1.}):
        altered = copy.deepcopy(config)
        altered['chunk_policy'].update(patch)
        with pytest.raises(ValueError):
            chunk_settings(altered)


def test_masked_loss_ignores_padding_and_quaternion_sign_and_has_gradients():
    settings = chunk_settings(settings_config())
    decoder = ChunkDecoder(8, settings)
    state = torch.zeros(2, 14)
    state[:, 6] = 1
    prediction = decoder(torch.randn(2, 3, 8), torch.randn(2, 3, 8), state, torch.zeros(2, 8))
    target = torch.zeros(2, 4, 8)
    target[..., 6] = 1
    valid = torch.tensor([[True, True, False, False], [True, False, False, False]])
    target[~valid] = float('nan')
    loss, _ = chunk_loss(prediction, target, valid, settings)
    flipped = target.clone()
    flipped[..., 3:7] *= -1
    second, _ = chunk_loss(prediction, flipped, valid, settings)
    torch.testing.assert_close(loss, second)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in decoder.parameters())


def test_continuous_rotation_convention_matches_scipy_world_xyzw():
    from scipy.spatial.transform import Rotation
    expected = Rotation.from_euler('xyz', [[90, 0, 0], [0, 90, 0], [0, 0, -90]], degrees=True)
    matrix = quaternion_matrix(torch.tensor(expected.as_quat()))
    np.testing.assert_allclose(matrix.numpy(), expected.as_matrix(), atol=1e-7)
    six = torch.cat((matrix[..., :, 0], matrix[..., :, 1]), -1)
    torch.testing.assert_close(rotation_6d(six), matrix)


def test_decoder_training_freezes_key_encoder_and_cannot_read_future_labels(chunk_fixture):
    agent, data, _ = agent_data(chunk_fixture)
    before = {key: value.clone() for key, value in agent._net_mod.encoder.state_dict().items()}
    decoder = agent._net_mod.decoder.head.weight.detach().clone()
    sample = data[0]
    initial = agent.predict(sample, sample['goal'])
    altered = copy.deepcopy(sample)
    altered['key_action'][:] = 900
    altered['action_chunk'][:] = -900
    after = agent.predict(altered, sample['goal'])
    np.testing.assert_array_equal(initial['actions'], after['actions'])
    metrics = agent.update(collate_chunk([data[0], data[-1]]))
    assert np.isfinite(metrics['total_loss'])
    assert not torch.equal(decoder, agent._net_mod.decoder.head.weight)
    assert all(torch.equal(before[key], value) for key, value in agent._net_mod.encoder.state_dict().items())
    agent._net_mod.train()
    assert agent._net_mod.decoder.training and not agent._net_mod.encoder.training


def test_chunk_checkpoint_roundtrip_and_old_resume_rejection(chunk_fixture):
    from finetune.OHT.model import load_weights, model_state
    first, data, config = agent_data(chunk_fixture)
    first.update(collate_chunk([data[0]]))
    saved = dict(config=config, model_state=model_state(first), oht_contract=data.contract)
    second, _, _ = agent_data(chunk_fixture)
    load_weights(second, saved)
    np.testing.assert_array_equal(first.predict(data[0], data[0]['goal'])['actions'],
                                  second.predict(data[0], data[0]['goal'])['actions'])
    second._optimizer.load_state_dict(copy.deepcopy(first._optimizer.state_dict()))
    batch = collate_chunk([data[0]])
    first.update(batch)
    second.update(batch)
    for key, value in first._net_mod.decoder.state_dict().items():
        torch.testing.assert_close(value, second._net_mod.decoder.state_dict()[key])
    with pytest.raises(ValueError, match='Single-key checkpoint'):
        load_weights(second, dict(config={'mode': 'baseline'}, model_state={}))
    previous = dict(config={'mode': 'baseline'}, model_state=copy.deepcopy(first.base._net_mod.state_dict()),
                    oht_contract=data.contract)
    decoder = second._net_mod.decoder.head.weight.detach().clone()
    load_weights(second, previous, initialize=True)
    assert torch.equal(decoder, second._net_mod.decoder.head.weight)
    changed = copy.deepcopy(previous)
    profile = next(iter(changed['oht_contract']['source_data_configs'].values()))
    profile['data_config']['image_size'] = [4, 4]
    with pytest.raises(ValueError, match='calibration/data profiles'):
        load_weights(second, changed, initialize=True)


def test_chunk_factory_uses_real_key_agent_with_separately_trainable_decoder(chunk_fixture, monkeypatch):
    from tests.test_oht_action_modes import TinyMVT
    from finetune.OHT import bootstrap  # noqa: F401
    from bridgevla.mvt import mvt
    from bridgevla.models.bridgevla_agent import RVTAgent
    from finetune.OHT.model import build_agent
    monkeypatch.setattr(mvt, 'MVT', TinyMVT)
    data = OHTChunkDataset(chunk_fixture.root)
    agent = build_agent(settings_config(), data.contract, torch.device('cpu'))
    assert isinstance(agent.base, RVTAgent)
    assert agent.base._net_mod is agent._net_mod.encoder
    assert all(not parameter.requires_grad for parameter in agent._net_mod.encoder.parameters())
    assert all(name.startswith('decoder.') for name, parameter in agent._net_mod.named_parameters() if parameter.requires_grad)


def test_v4_training_validate_only_and_offline_metrics(chunk_fixture, tmp_path, capsys):
    from finetune.OHT.train import main
    import yaml
    config = settings_config()
    config['extends'] = None
    path = tmp_path / 'train.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    assert main(['--config', str(path), '--replay', str(chunk_fixture.root), '--validate-only']) == 0
    assert json.loads(capsys.readouterr().out)['valid']
    agent, data, _ = agent_data(chunk_fixture, training=False)
    policy = Policy(agent, data.contract, 'cpu', 'baseline')
    report = open_loop(policy, chunk_fixture.root, 'val', limit=2)
    assert report['kind'] == 'open_loop_chunk_errors'
    assert report['per_offset']['1']['samples'] == 2
    assert report['all_actions']['samples'] >= 2


def test_chunk_prepared_http_and_native_client_negotiate_explicit_actions(chunk_fixture, tmp_path):
    agent, data, _ = agent_data(chunk_fixture, training=False)
    policy = Policy(agent, data.contract, 'cpu', 'baseline')
    sample = data[0]
    observation = {key: value for key, value in sample.items() if key in
        {'low_dim_state', 'tcp_pose', 'joint_positions', 'gripper_open01'} or any(key.endswith('_' + suffix)
        for suffix in ('rgb', 'depth', 'point_cloud', 'camera_intrinsics', 'camera_extrinsics'))}
    checkpoint = tmp_path / 'checkpoint.pth'
    checkpoint.write_bytes(b'injected loader')
    app = create_app(checkpoint, policy_loader=lambda *args: policy,
                     output_dir=tmp_path / 'logs', visualize_every=0)
    with TestClient(app) as client:
        info = client.get('/model_info').json()
        assert info['chunk_spec']['schema'] == 'bridgevla_oht_eef_chunk_v1'
        body = dict(**envelope(info), step=0, timestamp=0., instruction=sample['goal'],
                    observation_format=OBSERVATION_FORMAT, observation=pack_observation(observation))
        assert client.post('/predict', json=body).status_code == 422
        body['prediction_schema'] = info['chunk_spec']['schema']
        response = client.post('/predict', json=body)
        assert response.status_code == 200, response.text
        result = response.json()
        assert 'action' not in result and np.shape(result['actions']) == (4, 8)
        validate_prediction(result, info['chunk_spec'])
        assert client.post('/predict', json=body).status_code == 409

        class Opener:
            def open(self, request, timeout):
                response = client.request(request.get_method(), request.full_url.replace('http://local', ''),
                                           content=request.data, headers=dict(request.header_items()))
                assert response.status_code == 200, response.text
                return io.BytesIO(response.content)
        native = BridgeVLAClient('http://local', opener=Opener())
        assert native.chunk_spec == info['chunk_spec']
        # Use the client reset/identity checks; a malformed chunk must leave it uncertain.
        native.reset('fresh-native')
        from scipy.spatial.transform import Rotation
        from finetune.OHT.runtime.raw_observation import OPENGL_TO_OPTICAL
        profile = data.samples[0]['data_profile']
        config = data.contract['source_data_configs'][profile]['data_config']
        tcp = sample['tcp_pose']
        opening = float(sample['gripper_open01'])
        raw = dict(joint_positions=sample['joint_positions'],
                   ee_link_pose=np.r_[tcp[:3], tcp[3:][[3, 0, 1, 2]]].astype(np.float32),
                   gripper_position=np.array([config['gripper']['close'] -
                       opening * (config['gripper']['close'] - config['gripper']['open'])], np.float32))
        for name in config['cameras']:
            transform = sample[name + '_camera_extrinsics'] @ OPENGL_TO_OPTICAL
            quat = Rotation.from_matrix(transform[:3, :3]).as_quat()
            raw.update({name + '_rgb': sample[name + '_rgb'].transpose(1, 2, 0),
                        name + '_depth': sample[name + '_depth'][0],
                        name + '_intrinsics': sample[name + '_camera_intrinsics'],
                        name + '_pose': np.r_[transform[:3, 3], quat[[3, 0, 1, 2]]].astype(np.float32)})
        actions, reply = native.predict(raw, sample['goal'], 0., task=data.samples[0]['task'], data_profile=profile)
        assert actions.shape == (4, 8) and native.step == 1 and not native.uncertain
        request = dict(**envelope(info, 'other', 'different'), step=0, timestamp=1., instruction=sample['goal'],
                       observation_format=OBSERVATION_FORMAT, observation=pack_observation(observation),
                       prediction_schema=info['chunk_spec']['schema'])
        assert client.post('/predict', json=request).status_code == 409


def fake_bridge(control_dt=.01):
    state = np.array([0., 0, .5, 0, 0, 0, 1])
    sent = []
    bridge = SimpleNamespace(client=SimpleNamespace(action_config={'mode': 'eef'}),
        env=SimpleNamespace(step_dt=control_dt), last_done=False, last_stopped=False, last_target=None)
    bridge.measured_state = lambda *args: dict(tcp_pose_xyzw=state.copy(), joint_positions=[0.] * 6)
    bridge.measured_gripper = lambda *args: (1., 1)
    bridge.tracking_errors = lambda current: dict(translation_m=0., rotation_rad=0.)
    def step(target, opening, callback):
        state[:] = target[:7]
        sent.append((target.copy(), opening))
        bridge.last_stopped = callback is not None and bool(callback(2 * opening - 1))
        return [target.tolist()]
    bridge.step_target = step
    return bridge, sent


def test_chunk_timing_absolute_targets_gripper_ticks_and_prefix_discard():
    bridge, sent = fake_bridge()
    settings = chunk_settings(settings_config())
    spec = chunk_spec(settings, {'chunk': {'action_dt': .02}})
    actions = np.tile([.02, 0, .5, 0, 0, 0, 1, 0], (4, 1)).astype(np.float32)
    actions[1, 0], actions[1, 7] = .04, 1
    executor = ChunkExecutor(bridge, spec)
    executor.execute(actions, {})
    assert len(sent) == 4  # H=2, two control steps per target
    assert [opening for _, opening in sent] == [1, 0, 0, 1]
    np.testing.assert_allclose([row[0][0] for row in sent], [.01, .02, .03, .04])
    assert executor.last_result['consumed_actions'] == 2
    assert executor.last_result['discarded_suffix'] == 2
    with pytest.raises(ValueError, match='integer multiple'):
        ChunkExecutor(fake_bridge(.03)[0], spec)


def test_chunk_executor_does_not_continue_after_tracking_or_task_stop():
    settings = chunk_settings(settings_config())
    spec = chunk_spec(settings, {'chunk': {'action_dt': .01}})
    actions = np.tile([.02, 0, .5, 0, 0, 0, 1, 0], (4, 1)).astype(np.float32)
    bridge, sent = fake_bridge()
    bridge.tracking_errors = lambda state: dict(translation_m=.2, rotation_rad=0)
    executor = ChunkExecutor(bridge, spec)
    executor.execute(actions, {})
    assert len(sent) == 1 and executor.last_result['status'] == 'tracking_error'
    bridge, sent = fake_bridge()
    executor = ChunkExecutor(bridge, spec)
    executor.execute(actions, {}, lambda grip: True)
    assert len(sent) == 1 and executor.last_result['completed']


def test_chunk_online_export_keeps_predictions_without_future_labels(chunk_fixture, tmp_path):
    from finetune.OHT.runtime.online_visualization import OnlineVisualizer
    policy_payload = dict(stages={'mvt1': dict(rgb=np.zeros((3, 8, 8, 3), np.float32),
        heatmap=np.full((3, 8, 8), 1 / 64, np.float32), world_to_pixel=np.tile(np.eye(4)[:2], (3, 1, 1)))},
        heatmap_waypoint_world=np.array([.2, 0, .5]))
    settings = chunk_settings(settings_config())
    actions = np.tile([.02, 0, .5, 0, 0, 0, 1, 1], (4, 1)).astype(np.float32)
    policy_payload['chunk'] = dict(spec=chunk_spec(settings, {'chunk': {'action_dt': .01}}),
                                   actions=actions, key_action=actions[0])
    data = OHTChunkDataset(chunk_fixture.root)
    writer = OnlineVisualizer(tmp_path / 'viz', every=1)
    job = writer.submit(policy_payload, dict(step=0), actions[0], action_options(), data[0])
    writer.close(wait=True)
    assert writer.latest()['error'] is None
    record = json.loads((writer.root / job['entry_id'] / 'prediction.json').read_text(encoding='utf-8'))
    assert len(record['chunk']['actions']) == 4
    assert 'target' not in record['chunk']
    assert (writer.root / job['entry_id'] / 'chunk.png').exists()
    scene = json.loads((writer.root / job['entry_id'] / 'scene.json').read_text())
    assert len(scene['trajectories']) == 2
