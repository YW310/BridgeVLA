"""Measured-state conditioning and cumulative delta actions without future leakage."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
import yaml

from finetune.OHT.chunk_config import chunk_settings
from finetune.OHT.chunk_model import chunk_loss, quaternion_matrix
from finetune.OHT.chunk_normalization import proprio_statistics, resolve_training_normalization
from finetune.OHT.config import load
from finetune.OHT.data.chunk_dataset import OHTChunkDataset
from finetune.OHT.model import build_agent, load_weights, model_state, resume_fingerprint
from finetune.OHT.spatial_chunk import SpatialChunkDecoder
from tests.test_oht_chunk_temporal import temporal_config, temporal_agent
from tests.test_oht_spatial_chunk import inputs
from tests.test_oht_chunk_joint import JointMVT, mixed_batch
from tests.test_oht_chunk_replay import chunk_fixture, replay_fixture  # noqa: F401


def state_config(kind='regression', representation='relative_pose', resolved=True):
    config = temporal_config()
    config['chunk_policy'].update(decoder_type=kind, state_conditioning=True,
        proprio_normalization='train_quantile', key_condition_gate=True,
        prediction_representation=representation)
    if resolved:
        config['chunk_policy'].update(proprio_center=[.2] * 6 + [.5], proprio_scale=[.5] * 7)
    return config


def test_state_presets_isolate_heads_without_changing_data_budget_or_freezing():
    previous = load('finetune/OHT/configs/oht_chunk_v4_flow_temporal_8x40.yaml')
    variants = [load(f'finetune/OHT/configs/oht_chunk_v4_{name}_state_8x40.yaml')
                for name in ('flow', 'regression', 'delta')]
    assert len({config['exp_id'] for config in variants}) == 3
    for config in variants:
        settings = chunk_settings(config)
        assert settings['state_conditioning'] and settings['key_condition_gate']
        assert settings['proprio_normalization'] == 'train_quantile'
        for name in ('epochs', 'global_batch_size', 'learning_rate', 'freeze_gemma_prefix_layers',
                     'freeze_vision_tower', 'freeze_multimodal_projector', 'freeze_language_model'):
            assert config[name] == previous[name]
    a, b, c = [chunk_settings(config) for config in variants]
    assert {key for key in a if a[key] != b[key]} == {'decoder_type'}
    assert {key for key in b if b[key] != c[key]} == {'prediction_representation'}
    assert not chunk_settings(previous)['state_conditioning']


@pytest.mark.parametrize('patch', [dict(state_conditioning=1), dict(key_condition_gate=1),
    dict(adaln=False), dict(proprio_normalization='auto'), dict(proprio_min_scale=0),
    dict(key_condition_gate_init=0), dict(key_condition_gate_init=1), dict(key_condition_gate_init=float('nan')),
    dict(proprio_center=[0.] * 6), dict(proprio_scale=[0.] * 7), dict(proprio_scale=[True] * 7),
    dict(proprio_center=[float('inf')] * 7), dict(proprio_scale=None),
    dict(proprio_normalization='none'), dict(normalization_quantile=.5),
    dict(prediction_representation='delta_euler'), dict(decoder_type='flow_matching', prediction_representation='delta_pose')])
def test_invalid_state_contracts_fail_early(patch):
    config = state_config()
    config['chunk_policy'].update(patch)
    with pytest.raises(ValueError):
        chunk_settings(config)


def test_unresolved_proprio_statistics_cannot_silently_reach_inference():
    with pytest.raises(ValueError, match='Resolve proprio normalization'):
        SpatialChunkDecoder(4, chunk_settings(state_config(resolved=False)), 3, 1 / 60)


@pytest.mark.parametrize('kind,representation', [('regression', 'relative_pose'),
    ('regression', 'delta_pose'), ('flow_matching', 'relative_pose')])
def test_live_joint_branches_match_training_state_features_without_changing_measurements(kind, representation, caplog):
    # Statistics and measured q6 from the deployed policy's first failed chunk.
    center = [.5836237419396639, -2.033438813686371, 1.777019736170769,
              -2.7411230653524394, -1.7244551002979278, -2.5896496739238497, .5]
    scale = [.5356327902525664, .5973584771156311, .5068113118410108,
             1.8865561157464985, .8967279732227327, 2.561074999347329, .5]
    measured = [-.01349194347858429, -2.3390045166015625, -4.599393844604492,
                -2.3961946964263916, 4.640030384063721, .10907460749149323]
    config = state_config(kind, representation)
    config['chunk_policy'].update(proprio_center=center, proprio_scale=scale)
    decoder = SpatialChunkDecoder(4, chunk_settings(config), 3, 1 / 60).eval()
    _, context, _, _ = inputs()
    state = context[2]
    state[:, 7:13] = torch.tensor(measured)
    state[:, 13] = .38315762584537016
    original = state.clone()
    captured = []
    handle = decoder.state.register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
    try:
        live_memory = decoder.memory(*context)
        canonical = state.clone()
        canonical[:, 9] += 2 * np.pi
        canonical[:, 11] -= 2 * np.pi
        canonical_memory = decoder.memory(*(*context[:2], canonical, *context[3:]))
    finally:
        handle.remove()
    torch.testing.assert_close(state, original, rtol=0, atol=0)
    expected = (canonical[:, 7:] - torch.tensor(center)) / torch.tensor(scale)
    torch.testing.assert_close(captured[0][:, 7:], expected, atol=3e-6, rtol=1e-5)
    # Slightly out-of-quantile joints remain out of range; nothing is clipped.
    assert captured[0][0, 7] < -1 and captured[0][0, 12] > 1
    assert captured[0][0, 9].item() == pytest.approx(-.18395066, abs=3e-6)
    assert captured[0][0, 11].item() == pytest.approx(.09066314, abs=3e-6)
    for live, recorded in zip(live_memory, canonical_memory):
        torch.testing.assert_close(live, recorded, atol=5e-6, rtol=1e-5)
    decoder.memory(*context)
    assert caplog.text.count('Chunk joint input aligned to training branch') == 1


def test_joint_branch_alignment_preserves_unsupported_and_ambiguous_states():
    config = state_config()
    config['chunk_policy'].update(proprio_center=[0.] * 6 + [.5],
                                  proprio_scale=[.5, 4., .5, .5, .5, .5, .5])
    decoder = SpatialChunkDecoder(4, chunk_settings(config), 3, 1 / 60)
    _, context, _, _ = inputs()
    state = context[2]
    state[:, 7:13] = torch.tensor([7., 2 * np.pi + .1, -4 * np.pi + .2, .3, -.4, .1])
    state[:, 13] = .8
    original = state.clone()
    captured = []
    handle = decoder.state.register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
    try:
        decoder.memory(*context)
    finally:
        handle.remove()
    expected = original[:, 7:].clone()
    expected[:, 2] = .2
    expected = (expected - decoder.proprio_center) / decoder.proprio_scale
    torch.testing.assert_close(captured[0][:, 7:], expected, atol=3e-6, rtol=1e-5)
    torch.testing.assert_close(state, original, rtol=0, atol=0)


def test_proprio_fit_is_train_only_repeatable_replay_bound_and_ignores_future_labels(chunk_fixture):
    data = OHTChunkDataset(chunk_fixture.root)
    config = state_config(resolved=False)
    config['chunk_policy'].update(normalization_max_samples=3, normalization_quantile=.8)
    settings = chunk_settings(config)
    states = []
    for index in np.linspace(0, len(data) - 1, 3, dtype=int):
        with np.load(data.root / data.samples[index]['observation'], allow_pickle=False) as source:
            states.append(source['joint_positions'])
    lower, upper = np.quantile(states, [.2, .8], axis=0)
    # No target file access is possible through this input, including key labels.
    rows = [{key: value for key, value in row.items() if not key.startswith('target')}
            for row in data.samples]
    measured_only = SimpleNamespace(root=data.root, samples=rows, contract=data.contract)
    stats = proprio_statistics(measured_only, settings)
    np.testing.assert_allclose(stats['proprio_center'], np.r_[(lower + upper) / 2, .5])
    np.testing.assert_allclose(stats['proprio_scale'], np.r_[np.maximum((upper - lower) / 2, .1), .5])
    assert stats['samples'] == 3 and stats['replay_sha256'] == data.contract['sha256']
    resolve_training_normalization(config, measured_only)
    assert config['chunk_proprio_normalization'] == stats
    repeated = state_config(resolved=False)
    repeated['chunk_policy'].update(normalization_max_samples=3, normalization_quantile=.8)
    resolve_training_normalization(repeated, measured_only)
    assert resume_fingerprint(config, data.contract, None, 2) == resume_fingerprint(repeated, data.contract, None, 2)
    saved = copy.deepcopy(config)
    resolve_training_normalization(config, SimpleNamespace(contract=data.contract))
    assert saved == config  # Already resolved checkpoints need no replay file IO.
    with pytest.raises(ValueError, match='training split'):
        proprio_statistics(OHTChunkDataset(data.root, 'val'), settings)
    rows[0]['observation_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='checksum'):
        proprio_statistics(measured_only, settings)


def test_proprio_success_and_failures_reach_all_ranks_without_peer_file_access(chunk_fixture):
    class Broadcast:
        def broadcast_object_list(self, values, src):
            if values == [None, None]:
                values[:] = copy.deepcopy(self.values)
            else:
                self.values = copy.deepcopy(values)
    data, communication = OHTChunkDataset(chunk_fixture.root), Broadcast()
    config = state_config(resolved=False)
    config['chunk_policy']['action_position_scale'] = 'train_quantile'
    peer = copy.deepcopy(config)
    resolve_training_normalization(config, data, distributed=communication)
    resolve_training_normalization(peer, SimpleNamespace(contract=data.contract), distributed=communication, rank=1)
    assert config == peer
    broken = copy.deepcopy(data.samples)
    broken[0]['observation_sha256'] = '0' * 64
    failed = state_config(resolved=False)
    with pytest.raises(ValueError, match='checksum'):
        resolve_training_normalization(failed, SimpleNamespace(root=data.root, samples=broken, contract=data.contract),
                                       distributed=communication)
    with pytest.raises(ValueError, match='checksum'):
        resolve_training_normalization(state_config(resolved=False), SimpleNamespace(contract=data.contract),
                                       distributed=communication, rank=1)


@pytest.mark.parametrize('kind', ['regression', 'flow_matching'])
def test_normalized_state_reaches_queries_and_every_adaln_block_with_a_gated_key(kind):
    _, context, _, _ = inputs()
    config = state_config(kind)
    config['chunk_policy']['layers'] = 2
    decoder = SpatialChunkDecoder(4, chunk_settings(config), 3, 1 / 60)
    captured_state, captured_conditions = [], []
    handles = [decoder.state.register_forward_pre_hook(lambda module, args: captured_state.append(args[0]))]
    for block in decoder.blocks:
        handles.append(block.modulation.register_forward_pre_hook(lambda module, args: captured_conditions.append(args[0])))
    try:
        memory = decoder.memory(*context)
        expected_state = context[2].clone()
        expected_state[:, :7] = torch.tensor([0., 0., 0., 0., 0., 0., 1.])
        expected_state[:, 7:] = (context[2][:, 7:] - decoder.proprio_center) / decoder.proprio_scale
        torch.testing.assert_close(captured_state[0], expected_state)
        assert memory[0].shape[1] == context[0].shape[1] + context[1].shape[1] + 1
        state_token = decoder.state(expected_state)
        goal_token = decoder.goal(decoder.goal_condition(context[2], context[3]))
        torch.testing.assert_close(memory[3], state_token + .1 * goal_token)
        pose = torch.zeros(2, 4, 9)
        tau = torch.ones(2)
        raw, grip = decoder.field(pose, tau, memory)
        from finetune.OHT.spatial_chunk import time_features
        expected_condition = decoder.flow_time(time_features(tau)) + memory[3]
        assert len(captured_conditions) == 2
        for condition in captured_conditions:
            torch.testing.assert_close(condition, expected_condition)
        (raw.square().mean() + grip.square().mean()).backward()
        for parameter in (decoder.state.weight, decoder.key_gate_logit, decoder.goal.weight):
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    finally:
        for handle in handles:
            handle.remove()
    # State is readable immediately, even with all visual tokens invalid and
    # zero-initialized AdaLN residual gates; it is not attention-only input.
    masked = list(context)
    masked[6], masked[7] = torch.zeros_like(context[6]), torch.zeros_like(context[7])
    first = decoder(*masked)[0]
    masked[2] = context[2].clone()
    masked[2][:, 7:13] += .3
    assert not torch.allclose(first, decoder(*masked)[0], atol=1e-7, rtol=0)
    # Disabling the explicit goal gate removes every direct key route.
    with torch.no_grad():
        decoder.key_gate_logit.fill_(-100.)
    first = decoder(*masked)
    masked[3] = context[3].clone()
    masked[3][:, :3] += 5.
    for a, b in zip(first, decoder(*masked)):
        torch.testing.assert_close(a, b)


def test_delta_pose_roundtrip_noncommuting_rotations_and_cumulative_trajectory_loss():
    _, context, target, valid = inputs('regression')
    decoder = SpatialChunkDecoder(4, chunk_settings(state_config(representation='delta_pose')), 3, 1 / 60)
    target[..., 3:7] = torch.tensor(Rotation.from_euler('xyz',
        [[15, 10, 3], [-60, 80, 30], [20, 40, -30], [90, -70, 10]], degrees=True).as_quat())
    encoded = decoder.encode_target(target, context[2])
    prediction = decoder.decode(encoded, torch.zeros_like(target[..., 7]), context[2])
    torch.testing.assert_close(prediction[0], target[..., :3])
    torch.testing.assert_close(prediction[1], quaternion_matrix(target[..., 3:7]), atol=1e-6, rtol=1e-5)
    # A one-mm error at each tick accumulates; supervise the resulting path.
    raw = encoded.detach().clone().requires_grad_()
    changed = raw + torch.tensor([.001 / .02, 0., 0., 0., 0., 0., 0., 0., 0.])
    predicted = decoder.decode(changed, torch.zeros_like(target[..., 7]), context[2])
    torch.testing.assert_close(torch.linalg.vector_norm(predicted[0] - target[..., :3], dim=-1),
                              torch.arange(1, 5).expand(2, -1) * .001, atol=1e-7, rtol=1e-4)
    loss, metrics = chunk_loss(predicted, target, valid, decoder.settings, current_position=context[2][:, :3])
    assert metrics['chunk_position_loss'] > 0 and metrics['chunk_displacement_loss'] > 0
    loss.backward()
    assert raw.grad[:, 0, :3].abs().sum() > raw.grad[:, -1, :3].abs().sum()


@pytest.mark.parametrize('kind,representation', [('flow_matching', 'relative_pose'), ('regression', 'relative_pose'),
                                               ('regression', 'delta_pose')])
def test_state_training_checkpoint_and_deployed_predictions_exclude_future_labels(chunk_fixture, kind, representation):
    config = state_config(kind, representation, resolved=False)
    data = OHTChunkDataset(chunk_fixture.root)
    resolve_training_normalization(config, data)
    agent, data, _ = temporal_agent(data.root, config=config)
    sample, batch = data[0], mixed_batch(data)
    first = agent.predict(sample, sample['goal'])['actions']
    changed = copy.deepcopy(sample)
    for name in ('action_chunk', 'key_action', 'joint_chunk'):
        changed[name][...] = float('nan')
    np.testing.assert_array_equal(first, agent.predict(changed, sample['goal'])['actions'])
    before = copy.deepcopy(model_state(agent))
    metrics = agent.update(batch)
    assert metrics['chunk_key_condition_gate'] == pytest.approx(.1)
    assert not torch.equal(before['decoder.state.weight'], model_state(agent)['decoder.state.weight'])
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in agent._net_mod.parameters())
    checkpoint = dict(config=config, model_state=copy.deepcopy(model_state(agent)))
    other, _, _ = temporal_agent(data.root, config=copy.deepcopy(config))
    load_weights(other, checkpoint)
    np.testing.assert_array_equal(agent.predict(sample, sample['goal'])['actions'], other.predict(sample, sample['goal'])['actions'])
    # Changing normalization/representation must not silently reuse old decoder weights.
    other.settings['proprio_center'] = [v + .01 for v in other.settings['proprio_center']]
    with pytest.raises(ValueError, match='mismatch'):
        load_weights(other, checkpoint)
    torch.nn.init.zeros_(other._net_mod.decoder.head.weight)
    load_weights(other, checkpoint, initialize=True)
    assert not other._net_mod.decoder.head.weight.any()


def test_cli_saves_proprio_statistics_for_inference_without_training_replay(chunk_fixture, tmp_path, monkeypatch):
    from finetune.OHT import train
    from bridgevla.mvt import mvt
    from torch.utils.data import DataLoader
    config = state_config(resolved=False)
    config['chunk_policy']['action_position_scale'] = 'train_quantile'
    config.update(batch_size=2, num_workers=0, global_batch_size=2, epochs=1, train_iter=2,
                  max_optimizer_steps=1, checkpoint_every_epochs=1)
    config['train_visualization']['enabled'] = False
    path, output = tmp_path / 'state.yaml', tmp_path / 'run'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    def build(settings, contract, device, pretrain_path, **kwargs):
        assert isinstance(settings['chunk_policy']['proprio_scale'], list)
        return build_agent(settings, contract, torch.device('cpu'), pretrain_path, **kwargs)
    monkeypatch.setattr(mvt, 'MVT', JointMVT)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda device: None)
    monkeypatch.setattr(torch.utils.data, 'DataLoader', lambda *args, **kwargs:
                        DataLoader(*args, **dict(kwargs, pin_memory=False)))
    monkeypatch.setattr(train, 'to_device', lambda batch, device: batch)
    monkeypatch.setattr(train, 'build_agent', build)
    for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'):
        monkeypatch.delenv(name, raising=False)
    assert train.main(['--config', str(path), '--replay', str(chunk_fixture.root),
                       '--output', str(output), '--log-backend', 'none']) == 0
    saved = torch.load(output / 'model_last.pth', weights_only=False)
    stats = json.loads((output / 'proprio_normalization.json').read_text())
    assert saved['config']['chunk_proprio_normalization'] == stats
    assert saved['config']['chunk_policy']['proprio_scale'] == stats['proprio_scale']
    restored = build_agent(saved['config'], saved['oht_contract'], torch.device('cpu'), training=False)
    load_weights(restored, saved)
    sample = OHTChunkDataset(chunk_fixture.root)[0]
    assert np.isfinite(restored.predict(sample, sample['goal'])['actions']).all()


def _state_ddp_worker(rank, root, directory, kind, stage, representation='relative_pose'):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=(Path(directory) / 'rendezvous').as_uri(), rank=rank, world_size=2)
    try:
        torch.manual_seed(100 + rank)
        config, data = state_config(kind, representation, resolved=False), OHTChunkDataset(root)
        config['chunk_policy']['action_position_scale'] = 'train_quantile'
        resolve_training_normalization(config, data, distributed=dist, rank=rank)
        agent, _, _ = temporal_agent(root, config=config, stage=stage, distributed=True)
        batch = mixed_batch(data)
        if rank:
            batch['key_valid'][:] = False
            batch['key_action'][:] = float('nan')
        for _ in range(2):
            with agent._network.no_sync():
                agent.update(batch, loss_scale=.5, step_optimizer=False)
            agent.update(batch, loss_scale=.5, reset_gradients=False)
        torch.save(model_state(agent), Path(directory) / f'rank{rank}.pth')
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('kind,stage,representation', [('regression', 'frozen_key', 'delta_pose'),
                                                     ('flow_matching', 'joint', 'relative_pose')])
@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(), reason='Gloo unavailable')
def test_state_ddp_with_zero_key_rank_and_gradient_accumulation(chunk_fixture, tmp_path, kind, stage, representation):
    torch.multiprocessing.spawn(_state_ddp_worker,
        args=(str(chunk_fixture.root), str(tmp_path), kind, stage, representation), nprocs=2, join=True)
    first, second = [torch.load(tmp_path / f'rank{rank}.pth', weights_only=True) for rank in range(2)]
    for name in first:
        torch.testing.assert_close(first[name], second[name])
