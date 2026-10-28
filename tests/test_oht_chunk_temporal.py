"""Physical normalization, high-noise flow training and temporal decoder contracts."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from finetune.OHT.chunk_config import chunk_settings, chunk_scale
from finetune.OHT.chunk_model import chunk_loss, quaternion_matrix
from finetune.OHT.chunk_normalization import action_scale_statistics, resolve_training_normalization
from finetune.OHT.config import load
from finetune.OHT.data.chunk_dataset import OHTChunkDataset
from finetune.OHT.model import build_agent, load_weights, model_state
from finetune.OHT.spatial_chunk import SpatialChunkDecoder, SpatialBlock
from tests.test_oht_spatial_chunk import inputs, small_config
from tests.test_oht_chunk_joint import JointMVT, mixed_batch
from tests.test_oht_chunk_replay import chunk_fixture, replay_fixture  # noqa: F401


def temporal_config():
    config = small_config()
    config['chunk_policy'].update(action_position_scale=.02, geometry_scale=.25,
        position_loss_scale=.01, flow_time_sampling='beta_high_noise', adaln=True,
        temporal_kernel_size=3, second_difference_weight=.1)
    return config


def temporal_agent(root, stage='joint', distributed=False, config=None):
    from bridgevla.mvt import mvt
    original, mvt.MVT = mvt.MVT, JointMVT
    try:
        config = temporal_config() if config is None else config
        data = OHTChunkDataset(root)
        config['chunk_policy']['stage'] = stage
        return build_agent(config, data.contract, torch.device('cpu'), distributed=distributed), data, config
    finally:
        mvt.MVT = original


def test_presets_keep_legacy_checkpoint_defaults_and_independent_experiments():
    old = chunk_settings(load('finetune/OHT/configs/oht_chunk_v4_flow_8x40.yaml'))
    scaled = load('finetune/OHT/configs/oht_chunk_v4_flow_scaled_8x40.yaml')
    temporal = load('finetune/OHT/configs/oht_chunk_v4_flow_temporal_8x40.yaml')
    assert not old['adaln'] and old['temporal_kernel_size'] == 1 and old['flow_time_sampling'] == 'uniform'
    assert all(chunk_scale(old, name) == .25 for name in ('action_position_scale', 'geometry_scale', 'position_loss_scale'))
    assert scaled['chunk_policy']['action_position_scale'] == 'train_quantile'
    assert temporal['chunk_policy']['adaln'] and temporal['chunk_policy']['second_difference_weight'] > 0
    assert scaled['freeze_gemma_prefix_layers'] == temporal['freeze_gemma_prefix_layers'] == 16
    assert scaled['epochs'] == temporal['epochs'] == 60
    assert scaled['exp_id'] != temporal['exp_id']


@pytest.mark.parametrize('patch', [dict(action_position_scale='auto'), dict(action_position_scale=0),
    dict(geometry_scale=-1), dict(position_loss_scale=float('nan')), dict(adaln=1),
    dict(temporal_kernel_size=2), dict(flow_time_sampling='beta_low_noise'),
    dict(second_difference_weight=-1), dict(second_difference_scale=0),
    dict(normalization_quantile=0), dict(normalization_quantile=1.1),
    dict(normalization_min_scale=0), dict(normalization_max_samples=False)])
def test_new_options_are_strictly_validated(patch):
    config = temporal_config()
    config['chunk_policy'].update(patch)
    with pytest.raises(ValueError):
        chunk_settings(config)


def test_normalization_reads_only_train_labels_and_persists_resolved_scale(chunk_fixture):
    data = OHTChunkDataset(chunk_fixture.root)
    config = temporal_config()
    config['chunk_policy'].update(action_position_scale='train_quantile', normalization_max_samples=3,
                                   normalization_quantile=.8, normalization_min_scale=.0001)
    settings = chunk_settings(config)
    expected = []
    for index in np.linspace(0, len(data) - 1, 3, dtype=int):
        row = data.samples[index]
        with np.load(data.root / row['targets']) as source:
            expected.extend(np.linalg.norm(source['action_chunk'][:4, :3][source['action_valid'][:4]].astype(float)
                                          - np.array(row['current_tcp'][:3]), axis=-1))
    # Observations/validation futures are not read for statistics.
    label_only = SimpleNamespace(root=data.root, samples=data.samples, contract=data.contract)
    statistics = action_scale_statistics(label_only, settings)
    assert statistics['quantile_m'] == pytest.approx(np.quantile(expected, .8))
    assert statistics['samples'] == 3 and statistics['valid_ticks'] == len(expected)
    resolve_training_normalization(config, label_only)
    assert config['chunk_policy']['action_position_scale'] == statistics['action_position_scale']
    assert config['chunk_normalization'] == statistics
    resolved = copy.deepcopy(config)
    resolve_training_normalization(config, label_only)
    assert config == resolved
    with pytest.raises(ValueError, match='training split'):
        action_scale_statistics(OHTChunkDataset(data.root, 'val'), settings)
    # A floor prevents small/near-constant motions from causing huge scales.
    assert action_scale_statistics(label_only, dict(settings, normalization_min_scale=1.))['action_position_scale'] == 1.


def test_rank_zero_statistics_and_failures_are_broadcast_to_peers(chunk_fixture):
    data = OHTChunkDataset(chunk_fixture.root)
    config = temporal_config()
    config['chunk_policy']['action_position_scale'] = 'train_quantile'

    class Broadcast:
        def broadcast_object_list(self, values, src):
            assert src == 0
            if values == [None, None]:
                values[:] = copy.deepcopy(self.values)
            else:
                self.values = copy.deepcopy(values)

    communication = Broadcast()
    peer = copy.deepcopy(config)
    resolve_training_normalization(config, data, distributed=communication)
    # A peer with no file access still receives exactly the same statistics.
    resolve_training_normalization(peer, SimpleNamespace(contract=data.contract), distributed=communication, rank=1)
    assert peer == config
    broken = temporal_config()
    broken['chunk_policy']['action_position_scale'] = 'train_quantile'
    rows = copy.deepcopy(data.samples)
    rows[0]['targets_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='checksum'):
        resolve_training_normalization(broken, SimpleNamespace(root=data.root, samples=rows, contract=data.contract),
                                       distributed=communication)
    with pytest.raises(ValueError, match='checksum'):
        resolve_training_normalization(copy.deepcopy(broken), SimpleNamespace(contract=data.contract),
                                       distributed=communication, rank=1)


def test_train_statistics_are_repeatable_for_resume_and_legacy_scale_defaults_are_exact(chunk_fixture):
    from finetune.OHT.model import resume_fingerprint
    data = OHTChunkDataset(chunk_fixture.root)
    initial, resumed = temporal_config(), temporal_config()
    for config in (initial, resumed):
        config['chunk_policy']['action_position_scale'] = 'train_quantile'
        resolve_training_normalization(config, data)
    assert resume_fingerprint(initial, data.contract, None, 2) == resume_fingerprint(resumed, data.contract, None, 2)
    decoder, context, _, _ = inputs()
    decoder.eval()
    first = decoder(*context)
    assert not any('modulation' in name or 'temporal' in name for name in decoder.state_dict())
    decoder.settings.update(action_position_scale=.25, geometry_scale=.25, position_loss_scale=.25)
    for a, b in zip(first, decoder(*context)):
        assert torch.equal(a, b)


def test_separate_action_geometry_and_loss_scales_share_one_physical_frame(monkeypatch):
    decoder, context, target, _ = inputs()
    decoder.settings.update(action_position_scale=.02, geometry_scale=.25, position_loss_scale=.005)
    state = context[2]
    encoded = decoder.encode_target(target, state)
    position, rotation, _ = decoder.decode(encoded, target[..., 7], state)
    torch.testing.assert_close(position, target[..., :3])
    torch.testing.assert_close(rotation, quaternion_matrix(target[..., 3:7]))
    memory = decoder.memory(*context)
    torch.testing.assert_close(memory[1][:, -1], decoder.goal_condition(state, context[3])[:, :3])
    captured = []
    def block(query, xyz, features, positions, valid, time):
        captured.append(xyz)
        return query
    monkeypatch.setattr(decoder.blocks[0], 'forward', block)
    decoder.field(encoded, torch.zeros(2), memory)
    torch.testing.assert_close(captured[0], encoded[..., :3] * .02 / .25)
    pending = chunk_settings(temporal_config())
    pending['action_position_scale'] = 'train_quantile'
    with pytest.raises(ValueError, match='Resolve train_quantile'):
        SpatialChunkDecoder(4, pending, 3, 1 / 60)


def test_physical_loss_scale_changes_position_loss_without_changing_actions_or_flow_scale():
    decoder, context, target, valid = inputs()
    settings = dict(decoder.settings, position_loss_scale=.01)
    shifted = target[..., :3].clone()
    shifted[..., 0] += .03
    prediction = shifted, quaternion_matrix(target[..., 3:7]), torch.zeros_like(target[..., 7])
    _, metrics = chunk_loss(prediction, target, valid, settings, current_position=context[2][:, :3])
    assert metrics['chunk_position_loss'] == pytest.approx((3. - .5) / 3, rel=1e-5)
    assert decoder.settings['position_scale'] == .25


def test_second_differences_match_real_turns_and_ignore_invalid_padding():
    decoder, context, target, valid = inputs()
    target[:, :, 0] += torch.tensor([.001, .004, .009, .016])
    settings = dict(decoder.settings, displacement_weight=0., second_difference_weight=1., second_difference_scale=.001)
    rotation, grip = quaternion_matrix(target[..., 3:7]), torch.zeros_like(target[..., 7])
    _, exact = chunk_loss((target[..., :3], rotation, grip), target, valid, settings, current_position=context[2][:, :3])
    assert exact['chunk_second_difference_loss'] == 0.  # Curvature itself is not penalized.
    stationary = context[2][:, None, :3].expand(-1, 4, -1).clone().requires_grad_()
    loss, incorrect = chunk_loss((stationary, rotation, grip), target, valid, settings, current_position=context[2][:, :3])
    assert incorrect['chunk_second_difference_loss'] > 0
    loss.backward()
    assert torch.isfinite(stationary.grad).all()
    valid[:, 2:] = False
    target[:, 2:] = float('nan')
    loss, metrics = chunk_loss((stationary, rotation, grip), target, valid, settings, current_position=context[2][:, :3])
    assert torch.isfinite(loss) and metrics['chunk_second_difference_loss'] > 0
    valid[:, 1:] = False
    _, metrics = chunk_loss((stationary, rotation, grip), target, valid, settings, current_position=context[2][:, :3])
    assert metrics['chunk_second_difference_loss'] == 0.


def test_beta_time_sampling_emphasizes_high_noise_and_tau_bins_account_for_valid_ticks():
    decoder, context, target, valid = inputs()
    decoder.settings['flow_time_sampling'] = 'beta_high_noise'
    torch.manual_seed(9)
    tau = decoder.sample_flow_time(torch.zeros(30000, 14))
    assert 0 <= tau.min() and tau.max() <= .999
    assert tau.mean().item() == pytest.approx(.3996, abs=.005)
    assert (tau < 1 / 3).float().mean() > .43
    prediction, extra = decoder.training_prediction(*context, target=target, valid=valid)
    counts = [extra[f'chunk_flow_tau_{name}_count'] for name in ('high_noise', 'mid_noise', 'low_noise')]
    losses = [extra[f'chunk_flow_tau_{name}_position_loss'] for name in ('high_noise', 'mid_noise', 'low_noise')]
    assert sum(counts) == valid.sum().item()
    assert sum(c * v for c, v in zip(counts, losses)) / sum(counts) == pytest.approx(extra['chunk_flow_position_loss'])


def test_adaln_zero_initialization_and_conditioned_residuals():
    block = SpatialBlock(8, 2, 0., adaln=True)
    query, xyz, memory = torch.randn(2, 4, 8), torch.randn(2, 4, 3), torch.randn(2, 6, 8)
    positions, valid = torch.randn(2, 6, 3), torch.ones(2, 6, dtype=torch.bool)
    first = block(query, xyz, memory, positions, valid, torch.randn(2, 8))
    torch.testing.assert_close(first, query)
    first.square().mean().backward()
    assert block.modulation[-1].weight.grad.abs().sum() > 0
    with torch.no_grad():
        block.modulation[-1].weight.normal_(std=.1)
    assert not torch.allclose(block(query, xyz, memory, positions, valid, torch.zeros(2, 8)),
                              block(query, xyz, memory, positions, valid, torch.ones(2, 8)))


def test_temporal_decoder_world_frame_equivariance_and_padding_isolation():
    from scipy.spatial.transform import Rotation
    _, context, target, valid = inputs()
    decoder = SpatialChunkDecoder(4, chunk_settings(temporal_config()), 3, 1 / 60).eval()
    # Activate AdaLN attention and temporal convolution to test real routing,
    # rather than just their zero-initialized identity paths.
    with torch.no_grad():
        for block in decoder.blocks:
            block.modulation[-1].weight.normal_(std=.02)
        decoder.temporal.weight.normal_(std=.02)
    first = decoder(*context)
    world = Rotation.from_euler('xyz', [25, 60, -15], degrees=True)
    rotation = torch.tensor(world.as_matrix(), dtype=torch.float32)
    translation = torch.tensor([1., -.5, .3])
    changed = list(context)
    for index in (2, 3):
        changed[index] = context[index].clone()
        changed[index][:, :3] = context[index][:, :3] @ rotation.T + translation
        changed[index][:, 3:7] = torch.from_numpy((world * Rotation.from_quat(context[index][:, 3:7])).as_quat())
    for index in (4, 5):
        changed[index] = context[index] @ rotation.T + translation
    second = decoder(*changed)
    torch.testing.assert_close(second[0], first[0] @ rotation.T + translation, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(second[1], rotation @ first[1], atol=2e-6, rtol=1e-5)
    masked = list(context)
    masked[6], masked[7] = torch.zeros_like(context[6]), torch.zeros_like(context[7])
    clean = decoder(*masked)
    for index in (0, 1, 4, 5):
        masked[index] = torch.full_like(masked[index], float('nan'))
    for a, b in zip(clean, decoder(*masked)):
        torch.testing.assert_close(a, b)
    valid[:, 2:] = False
    target[:, 2:] = float('nan')
    decoder.train()
    torch.manual_seed(7)
    prediction, extra = decoder.training_prediction(*context, target=target, valid=valid)
    altered = target.clone()
    altered[:, 2:] = 10000.
    torch.manual_seed(7)
    other, details = decoder.training_prediction(*context, target=altered, valid=valid)
    torch.testing.assert_close(extra['flow_loss'], details['flow_loss'])
    for a, b in zip(prediction, other):
        torch.testing.assert_close(a, b)


def test_temporal_training_checkpoint_and_sampling_have_no_future_label_leak(chunk_fixture, monkeypatch):
    from bridgevla.models import bridgevla_agent
    monkeypatch.setattr(bridgevla_agent, 'build_policy_visualization_payload', lambda *args, **kwargs: dict(stages={}))
    agent, data, config = temporal_agent(chunk_fixture.root)
    sample, batch = data[0], mixed_batch(data)
    initial = agent.predict(sample, sample['goal'])
    changed = copy.deepcopy(sample)
    changed['action_chunk'][:] = 1000.
    np.testing.assert_array_equal(initial['actions'], agent.predict(changed, sample['goal'])['actions'])
    before = copy.deepcopy(model_state(agent))
    for _ in range(3):
        metrics = agent.update(batch, return_visualization=True)
    assert metrics['chunk_second_difference_weighted_loss'] >= 0
    assert metrics['chunk_sampled_position_loss'] >= 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in agent._net_mod.parameters())
    assert not torch.equal(before['decoder.blocks.0.modulation.1.weight'], model_state(agent)['decoder.blocks.0.modulation.1.weight'])
    assert not torch.equal(before['decoder.temporal.weight'], model_state(agent)['decoder.temporal.weight'])
    checkpoint = dict(config=config, model_state=copy.deepcopy(model_state(agent)))
    other, _, _ = temporal_agent(chunk_fixture.root)
    load_weights(other, checkpoint)
    np.testing.assert_array_equal(agent.predict(sample, sample['goal'])['actions'], other.predict(sample, sample['goal'])['actions'])
    # Weight/sampling-only changes preserve the head; semantics changes do not.
    other.settings['flow_time_sampling'] = 'uniform'
    other.settings['position_loss_scale'] = .005
    load_weights(other, checkpoint, initialize=True)
    torch.testing.assert_close(model_state(other)['decoder.head.weight'], model_state(agent)['decoder.head.weight'])
    other.settings['action_position_scale'] = .03
    nn.init.zeros_(other._net_mod.decoder.head.weight)
    load_weights(other, checkpoint, initialize=True)
    assert not other._net_mod.decoder.head.weight.any()


def test_cli_resolves_training_scale_and_checkpoint_before_model_construction(chunk_fixture, tmp_path, monkeypatch):
    from finetune.OHT import train
    from bridgevla.mvt import mvt
    from torch.utils.data import DataLoader
    config = temporal_config()
    config['chunk_policy']['action_position_scale'] = 'train_quantile'
    config.update(batch_size=2, num_workers=0, global_batch_size=2, epochs=1, train_iter=2,
                  max_optimizer_steps=1, checkpoint_every_epochs=1)
    config['train_visualization']['enabled'] = False
    path, output = tmp_path / 'temporal.yaml', tmp_path / 'run'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    captured = {}
    def build(settings, contract, device, pretrain_path, **kwargs):
        assert isinstance(settings['chunk_policy']['action_position_scale'], float)
        captured.update(copy.deepcopy(settings))
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
    assert saved['config']['chunk_policy'] == captured['chunk_policy']
    statistics = json.loads((output / 'chunk_normalization.json').read_text())
    assert saved['config']['chunk_normalization'] == statistics
    assert saved['config']['chunk_policy']['action_position_scale'] == statistics['action_position_scale']
    # Inference loads the numeric scale saved with the checkpoint, without
    # needing access to the training replay or recalculating statistics.
    restored = build_agent(saved['config'], saved['oht_contract'], torch.device('cpu'), training=False)
    load_weights(restored, saved)
    sample = OHTChunkDataset(chunk_fixture.root)[0]
    assert np.isfinite(restored.predict(sample, sample['goal'])['actions']).all()


def _temporal_ddp_worker(rank, root, directory, stage):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=(Path(directory) / 'rendezvous').as_uri(), rank=rank, world_size=2)
    try:
        torch.manual_seed(100 + rank)
        config, data = temporal_config(), OHTChunkDataset(root)
        config['chunk_policy']['action_position_scale'] = 'train_quantile'
        resolve_training_normalization(config, data, distributed=dist, rank=rank)
        agent, data, _ = temporal_agent(root, stage=stage, distributed=True, config=config)
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


@pytest.mark.parametrize('stage', ['joint', 'frozen_key'])
@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(), reason='Gloo unavailable')
def test_temporal_flow_ddp_accumulation_and_zero_key_rank(chunk_fixture, tmp_path, stage):
    torch.multiprocessing.spawn(_temporal_ddp_worker, args=(str(chunk_fixture.root), str(tmp_path), stage), nprocs=2, join=True)
    first, second = [torch.load(tmp_path / f'rank{rank}.pth', weights_only=True) for rank in range(2)]
    for name in first:
        torch.testing.assert_close(first[name], second[name])
