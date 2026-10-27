"""Joint key/chunk gradients with the real agent and a small renderer/VLM stand-in."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from tests.test_oht_chunk_replay import chunk_fixture, replay_fixture  # noqa: F401
from tests.test_oht_action_modes import TinyMVT
from tests.test_oht_chunk_pipeline import settings_config
from finetune.OHT.chunk_model import load_chunk_weights
from finetune.OHT.config import load
from finetune.OHT.data.chunk_dataset import OHTChunkDataset, collate_chunk
from finetune.OHT.model import build_agent, model_state, resume_fingerprint


class TinyGemma(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.layers = nn.ModuleList(nn.Linear(4, 4) for _ in range(8))
        self.norm = nn.LayerNorm(4)

    def forward(self, features):
        for layer in self.layers:
            features = features + torch.tanh(layer(features))
        return self.norm(features)


class JointMVT(TinyMVT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mvt1.model = nn.Module()
        for name in ('vision_tower', 'multi_modal_projector'):
            setattr(self.mvt1.model, name, nn.Sequential(nn.Linear(4, 4), nn.Dropout(.1)))
        self.mvt1.model.language_model = TinyGemma()
        self.mvt1.feature_norm = nn.BatchNorm1d(4)
        self.calls = []

    def forward(self, pc, **kwargs):
        self.calls.append(dict(training=self.training, grad=torch.is_grad_enabled(),
                               gt=kwargs.get('wpt_local') is not None,
                               vlm_training=self.mvt1.model.training))
        output = super().forward(pc, **kwargs)
        features = self.mvt1.features.reshape(1, 6, 4).expand(len(pc), -1, -1)
        for module in self.mvt1.model.children():
            features = module(features)
        features = self.mvt1.feature_norm(features.reshape(-1, 4)).reshape(len(pc), 6, 4)
        for stage in (output, output['mvt2']):
            logits = stage['feat']
            stage.update(feat_x=logits[:, :72], feat_y=logits[:, 72:144],
                         feat_z=logits[:, 144:216], feat_ex_rot=logits[:, 216:])
        output['oht_chunk_tokens'] = features
        output['mvt2']['oht_chunk_tokens'] = features + .1
        return output


def joint_agent(root, distributed=False, overrides=None):
    from bridgevla.mvt import mvt
    original = mvt.MVT
    mvt.MVT = JointMVT
    try:
        config = settings_config()
        config['chunk_policy']['stage'] = 'joint'
        config.update(overrides or {})
        data = OHTChunkDataset(root)
        return build_agent(config, data.contract, torch.device('cpu'), distributed=distributed), data, config
    finally:
        mvt.MVT = original


def mixed_batch(data):
    # One supervised key plus one tail without a future key, which still has
    # dense continuation labels. Invalid key padding must never reach the loss.
    first = next(i for i in range(len(data)) if data[i]['key_valid'])
    last = next(i for i in range(len(data)) if not data[i]['key_valid'])
    samples = [data[first], data[last]]
    samples[1]['key_action'][:] = float('nan')
    return collate_chunk(samples)


@pytest.mark.parametrize('pretrain', [None, '/local/bridgevla-pretrain'])
def test_cli_joint_without_eef_checkpoint_trains_and_saves_both_heads(
        chunk_fixture, tmp_path, monkeypatch, capsys, pretrain):
    from finetune.OHT import train
    from bridgevla.mvt import mvt
    from torch.utils.data import DataLoader

    config = settings_config()
    config['chunk_policy']['stage'] = 'joint'
    config.update(batch_size=2, num_workers=0, global_batch_size=2,
                  epochs=1, train_iter=2, checkpoint_every_epochs=1)
    config['train_visualization']['enabled'] = False
    path = tmp_path / 'joint.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    output = tmp_path / 'run'
    captured = {}

    def build(settings, contract, device, pretrain_path, **kwargs):
        captured['pretrain'] = pretrain_path
        agent = build_agent(settings, contract, torch.device('cpu'), pretrain_path, **kwargs)
        captured['before'] = copy.deepcopy(model_state(agent))
        return agent

    def unexpected_checkpoint(*args, **kwargs):
        raise AssertionError('No task checkpoint should be loaded')

    monkeypatch.setattr(mvt, 'MVT', JointMVT)
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_device', lambda device: None)
    monkeypatch.setattr(torch.utils.data, 'DataLoader', lambda *args, **kwargs:
                        DataLoader(*args, **dict(kwargs, pin_memory=False)))
    monkeypatch.setattr(train, 'to_device', lambda batch, device: batch)
    monkeypatch.setattr(train, 'build_agent', build)
    monkeypatch.setattr(train, 'load_weights', unexpected_checkpoint)
    for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'):
        monkeypatch.delenv(name, raising=False)
    args = ['--config', str(path), '--replay', str(chunk_fixture.root),
            '--output', str(output), '--log-backend', 'none']
    if pretrain:
        args += ['--pretrain-path', pretrain]
    assert train.main(args) == 0
    assert captured['pretrain'] == pretrain
    metrics = json.loads((output / 'metrics.jsonl').read_text().strip())
    assert metrics['key_valid_samples'] > 0
    assert np.isfinite(metrics['total_loss'])
    checkpoint = torch.load(output / 'model_last.pth', weights_only=False)
    assert checkpoint['optimizer_step'] == 1
    for name in ('encoder.trans', 'encoder.feat', 'decoder.head.weight'):
        assert not torch.equal(checkpoint['model_state'][name], captured['before'][name]), name
    assert torch.equal(checkpoint['model_state']['encoder.mvt1.model.vision_tower.0.weight'],
                       captured['before']['encoder.mvt1.model.vision_tower.0.weight'])
    assert f"stage=joint source={'bridgevla_pretrain' if pretrain else 'paligemma_base'}" in capsys.readouterr().out


@pytest.mark.parametrize('pretrain', [None, '/local/bridgevla-pretrain'])
def test_cli_frozen_key_still_requires_trained_checkpoint(chunk_fixture, tmp_path, capsys, pretrain):
    from finetune.OHT.train import main
    path = tmp_path / 'frozen.yaml'
    config = settings_config()
    config['train_visualization']['enabled'] = False
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    args = ['--config', str(path), '--replay', str(chunk_fixture.root), '--log-backend', 'none']
    if pretrain:
        args += ['--pretrain-path', pretrain]
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
    assert 'frozen_key chunk training requires --init-checkpoint' in capsys.readouterr().err


@pytest.mark.parametrize('prefix', [0, 2, 6])
@pytest.mark.parametrize('goal_conditioning', ['absolute', 'relative_tcp'])
def test_joint_layer_count_freezes_prefix_but_trains_later_layers_and_both_heads(chunk_fixture, prefix, goal_conditioning):
    agent, data, config = joint_agent(chunk_fixture.root, overrides={'freeze_gemma_prefix_layers': prefix})
    agent.settings['goal_conditioning'] = goal_conditioning
    assert not config['freeze_language_model']
    before = copy.deepcopy(model_state(agent))
    frozen = [name for name, p in agent._net_mod.named_parameters() if not p.requires_grad]
    assert frozen and all('mvt1.model.' in name for name in frozen)
    language = agent.base._net_mod.mvt1.model.language_model
    for index, layer in enumerate(language.layers):
        assert all(p.requires_grad == (index >= prefix) for p in layer.parameters())
    assert not language.embed_tokens.weight.requires_grad
    result = agent.update(mixed_batch(data))
    assert result['key_valid_samples'] == 1
    assert result['total_loss'] == pytest.approx(result['chunk_total_loss'] + result['key_total_loss'])
    assert all(np.isfinite(value) for value in result.values())
    after = model_state(agent)
    assert all(torch.equal(before[name], after[name]) for name in frozen)
    assert torch.equal(before['encoder.mvt1.feature_norm.running_mean'], after['encoder.mvt1.feature_norm.running_mean'])
    for name in ('encoder.trans', 'encoder.feat', 'encoder.mvt1.features', 'decoder.head.weight'):
        assert not torch.equal(before[name], after[name]), name
    for index in range(prefix, len(language.layers)):
        name = f'encoder.mvt1.model.language_model.layers.{index}.weight'
        assert not torch.equal(before[name], after[name]), name
    assert all(p.grad is None for name, p in agent._net_mod.named_parameters() if name in frozen)
    calls = agent.base._net_mod.calls
    assert calls[0] == dict(training=True, grad=True, gt=True, vlm_training=True)
    assert all(call == dict(training=False, grad=True, gt=False, vlm_training=False) for call in calls[1:])


def test_joint_context_keeps_feature_gradients_and_ignores_all_future_labels(chunk_fixture):
    agent, data, _ = joint_agent(chunk_fixture.root)
    sample = data[0]
    initial = agent._context(sample, with_grad=True)
    changed = copy.deepcopy(sample)
    changed['key_action'][:] = 900
    changed['action_chunk'][:] = -900
    changed['key_valid'] = False
    other = agent._context(changed, with_grad=True)
    for a, b in zip(initial, other):
        torch.testing.assert_close(a, b)
    assert initial[0].requires_grad and not initial[3].requires_grad
    initial[0].square().sum().backward()
    assert agent.base._net_mod.mvt1.features.grad.abs().sum() > 0
    assert all(p.grad is None for p in agent.base._net_mod.mvt1.model.parameters() if not p.requires_grad)
    assert agent.base._net_mod.mvt1.model.language_model.layers[-1].weight.grad.abs().sum() > 0
    deployed = agent._context(sample)
    for a, b in zip(initial, deployed):
        torch.testing.assert_close(a, b)
    assert not deployed[0].requires_grad


def test_explicit_full_language_freeze_remains_available(chunk_fixture):
    agent, data, _ = joint_agent(chunk_fixture.root, overrides={'freeze_language_model': True})
    language = agent.base._net_mod.mvt1.model.language_model
    assert all(not p.requires_grad for p in language.parameters())
    before = copy.deepcopy(language.state_dict())
    agent.update(mixed_batch(data))
    assert all(torch.equal(before[name], value) for name, value in language.state_dict().items())
    assert agent.base._net_mod.calls[0]['vlm_training'] is False


def test_joint_no_key_samples_still_train_chunk_and_resume_both_heads(chunk_fixture):
    agent, data, config = joint_agent(chunk_fixture.root)
    batch = mixed_batch(data)
    batch['key_valid'][:] = False
    batch['key_action'][:] = float('nan')
    before = agent._net_mod.decoder.head.weight.detach().clone()
    result = agent.update(batch)
    assert result['key_valid_samples'] == 0 and result['key_total_loss'] == 0
    assert not torch.equal(before, agent._net_mod.decoder.head.weight)
    saved = dict(config=config, model_state=copy.deepcopy(model_state(agent)))
    other, _, _ = joint_agent(chunk_fixture.root)
    load_chunk_weights(other, saved)
    other._optimizer.load_state_dict(copy.deepcopy(agent._optimizer.state_dict()))
    agent.update(mixed_batch(data))
    other.update(mixed_batch(data))
    for name, value in model_state(agent).items():
        torch.testing.assert_close(value, model_state(other)[name])


def test_joint_stage_change_requires_init_and_preserves_trained_chunk(chunk_fixture):
    agent, data, config = joint_agent(chunk_fixture.root)
    frozen = copy.deepcopy(config)
    frozen['chunk_policy']['stage'] = 'frozen_key'
    saved = dict(config=frozen, model_state=copy.deepcopy(model_state(agent)))
    before = agent._net_mod.decoder.head.weight.detach().clone()
    with pytest.raises(ValueError, match='mismatch'):
        load_chunk_weights(agent, saved)
    load_chunk_weights(agent, saved, initialize=True)
    assert torch.equal(before, agent._net_mod.decoder.head.weight)
    assert resume_fingerprint(config, data.contract, None, 1) != resume_fingerprint(frozen, data.contract, None, 1)


def test_joint_freezing_and_loss_options_are_validated(tmp_path):
    import yaml
    config = load('finetune/OHT/configs/oht_chunk_v4_8x40.yaml')
    for patch in ({'freeze_language_model': 'true'}, {'freeze_gemma_prefix_layers': -1},
                  {'freeze_gemma_prefix_layers': 1.5}, {'freeze_gemma_prefix_layers': True},
                  {'chunk_policy': {'stage': 'joint', 'key_loss_weight': 0}}):
        changed = copy.deepcopy(config)
        if 'chunk_policy' in patch:
            changed['chunk_policy'].update(patch['chunk_policy'])
        else:
            changed.update(patch)
        path = tmp_path / 'invalid.yaml'
        path.write_text(yaml.safe_dump(changed), encoding='utf-8')
        with pytest.raises(ValueError):
            load(path)


def _ddp_worker(rank, root, directory):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=(Path(directory) / 'rendezvous').as_uri(), rank=rank, world_size=2)
    try:
        torch.manual_seed(123)
        agent, data, _ = joint_agent(root, distributed=True)
        # Exercise no_sync accumulation, rank-specific unused heads, and a
        # subsequent iteration after DDP has finalized its first reduction.
        for iteration in range(2):
            batch = mixed_batch(data)
            if rank != iteration:
                batch['key_valid'][:] = False
                batch['key_action'][:] = float('nan')
            with agent._network.no_sync():
                agent.update(batch, loss_scale=.5, step_optimizer=False)
            agent.update(batch, loss_scale=.5, reset_gradients=False)
        torch.save(dict(model=model_state(agent), gradients={
            name: p.grad for name, p in agent._net_mod.named_parameters() if p.grad is not None}),
            Path(directory) / f'rank{rank}.pth')
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(), reason='Gloo unavailable')
def test_joint_ddp_handles_empty_key_rank_and_gradient_accumulation(chunk_fixture, tmp_path):
    torch.multiprocessing.spawn(_ddp_worker, args=(str(chunk_fixture.root), str(tmp_path)), nprocs=2, join=True)
    first, second = [torch.load(tmp_path / f'rank{rank}.pth', weights_only=True) for rank in range(2)]
    torch.manual_seed(123)
    reference, data, _ = joint_agent(chunk_fixture.root)
    for _ in range(2):
        reference.update(mixed_batch(data), loss_scale=.5, step_optimizer=False)
        reference.update(mixed_batch(data), loss_scale=.5, reset_gradients=False)
    for name, value in model_state(reference).items():
        torch.testing.assert_close(first['model'][name], second['model'][name])
        torch.testing.assert_close(first['model'][name], value)
    for name, p in reference._net_mod.named_parameters():
        if p.grad is not None:
            torch.testing.assert_close(first['gradients'][name], p.grad)
