"""Repair objectives must train safely without changing legacy inference."""
import copy

import pytest
import torch

from finetune.OHT.chunk_config import chunk_settings
from finetune.OHT.config import load
from finetune.OHT.data.chunk_dataset import collate_chunk
from finetune.OHT.model import load_weights, model_state
from tests.test_oht_chunk_pipeline import agent_data
from tests.test_oht_chunk_replay import chunk_fixture, replay_fixture  # noqa: F401


@pytest.mark.parametrize('value', [0, -1, True, float('inf'), float('nan')])
def test_invalid_gradient_limits_are_rejected(value):
    config = load('finetune/OHT/configs/oht_chunk_v4_delta_state_tcp_guided_repair_8x40.yaml')
    config['chunk_policy']['max_grad_norm'] = value
    with pytest.raises(ValueError, match='max_grad_norm'):
        chunk_settings(config)


def test_clips_complete_accumulation_only_at_optimizer_boundary(chunk_fixture, monkeypatch):
    agent, data, _ = agent_data(chunk_fixture)
    agent.settings['max_grad_norm'] = .001
    batch = collate_chunk([data[0]])
    clip = torch.nn.utils.clip_grad_norm_
    calls = []

    def capture(parameters, *args, **kwargs):
        calls.append(True)
        return clip(parameters, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', capture)
    before = copy.deepcopy(model_state(agent))
    metrics = agent.update(batch, loss_scale=.5, step_optimizer=False)
    assert not calls and 'chunk_grad_norm_before_clip' not in metrics
    for name, value in model_state(agent).items():
        torch.testing.assert_close(value, before[name])
    metrics = agent.update(batch, loss_scale=.5, reset_gradients=False)
    assert len(calls) == 1 and metrics['chunk_grad_norm_before_clip'] > .001
    norm = torch.linalg.vector_norm(torch.cat([
        p.grad.flatten() for p in agent._net_mod.parameters() if p.grad is not None]))
    assert norm <= .001001
    assert not torch.equal(model_state(agent)['decoder.head.weight'], before['decoder.head.weight'])


def test_nonfinite_gradients_cannot_update_parameters_or_optimizer(chunk_fixture):
    agent, data, _ = agent_data(chunk_fixture)
    agent.settings['max_grad_norm'] = 1.
    parameter = agent._net_mod.decoder.head.weight
    parameter.register_hook(lambda gradient: gradient * float('inf'))
    before = copy.deepcopy(model_state(agent))
    with pytest.raises(RuntimeError, match='non-finite'):
        agent.update(collate_chunk([data[0]]))
    assert not agent._optimizer.state
    for name, value in model_state(agent).items():
        torch.testing.assert_close(value, before[name])


def test_gradient_limit_change_retains_compatible_trained_decoder(chunk_fixture):
    agent, data, config = agent_data(chunk_fixture, training=False)
    checkpoint = dict(config=copy.deepcopy(config), model_state=copy.deepcopy(model_state(agent)))
    sample = data[0]
    prediction = agent.predict(sample, sample['goal'])['actions']
    agent.settings['max_grad_norm'] = 1.
    # Inference is independent of this optimizer setting, including old files
    # without the new key. Initialization must not reset a compatible decoder.
    with torch.no_grad():
        agent._net_mod.decoder.head.weight.zero_()
    load_weights(agent, checkpoint, initialize=True)
    torch.testing.assert_close(agent._net_mod.decoder.head.weight,
                               checkpoint['model_state']['decoder.head.weight'])
    torch.testing.assert_close(torch.as_tensor(agent.predict(sample, sample['goal'])['actions']),
                               torch.as_tensor(prediction))
