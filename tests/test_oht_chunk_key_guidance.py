"""Predicted spatial key guidance, gated independently of scene token count."""
import copy

import numpy as np
import pytest
import torch

from finetune.OHT.chunk_config import chunk_settings
from finetune.OHT.config import load
from finetune.OHT.model import load_weights, model_state
from finetune.OHT.spatial_chunk import RelativeAttention, SpatialChunkDecoder
from tests.test_oht_chunk_state import state_config
from tests.test_oht_chunk_temporal import temporal_agent
from tests.test_oht_chunk_joint import mixed_batch
from tests.test_oht_spatial_chunk import inputs
from tests.test_oht_chunk_replay import chunk_fixture, replay_fixture  # noqa: F401


def guided_config(kind='regression', representation='delta_pose'):
    config = state_config(kind, representation)
    config['chunk_policy']['spatial_key_token'] = True
    return config


def test_guided_preset_only_adds_spatial_key_route_to_delta_baseline():
    parent = load('finetune/OHT/configs/oht_chunk_v4_delta_state_8x40.yaml')
    guided = load('finetune/OHT/configs/oht_chunk_v4_delta_state_guided_8x40.yaml')
    assert parent['exp_id'] != guided['exp_id']
    old, new = chunk_settings(parent), chunk_settings(guided)
    assert {key for key in old if old[key] != new[key]} == {'spatial_key_token'}
    assert new['spatial_key_token'] and new['prediction_representation'] == 'delta_pose'
    for key in set(parent) - {'chunk_policy', 'exp_id'}:
        assert parent[key] == guided[key]


@pytest.mark.parametrize('patch', [dict(spatial_key_token=1), dict(key_condition_gate=False),
                                  dict(state_conditioning=False), dict(spatial_conditioning=False)])
def test_guidance_requires_boolean_flag_and_gated_spatial_state_decoder(patch):
    config = guided_config()
    config['chunk_policy'].update(patch)
    with pytest.raises(ValueError):
        chunk_settings(config)


def test_key_token_has_tcp_local_geometry_and_reaches_every_cross_attention():
    _, context, _, _ = inputs()
    config = guided_config()
    config['chunk_policy']['layers'] = 2
    decoder = SpatialChunkDecoder(4, chunk_settings(config), 3, 1 / 60)
    memory = decoder.memory(*context)
    assert len(memory) == 5
    expected_goal = decoder.goal_condition(context[2], context[3])
    torch.testing.assert_close(memory[0][:, -1], decoder.goal(expected_goal))
    torch.testing.assert_close(memory[1][:, -1], expected_goal[:, :3])
    assert memory[2][:, -1].all()
    expected_prior = (memory[2][:, :-1].sum(-1) * .1).log()
    torch.testing.assert_close(memory[4][:, -1], expected_prior)
    assert not memory[4][:, :-1].any()
    captured = []
    handles = [block.cross_attention.register_forward_pre_hook(
        lambda module, args: captured.append(args)) for block in decoder.blocks]
    try:
        decoder.sample(memory, context[2])
    finally:
        for handle in handles:
            handle.remove()
    assert len(captured) == 2
    for args in captured:
        torch.testing.assert_close(args[3][:, -1], expected_goal[:, :3])
        torch.testing.assert_close(args[5], memory[4])


@pytest.mark.parametrize('count', [6, 60])
def test_goal_attention_prior_is_not_diluted_by_patch_count_or_invalid_patches(count):
    config = guided_config()
    decoder = SpatialChunkDecoder(4, chunk_settings(config), 3, 1 / 60)
    _, context, _, _ = inputs()
    coarse, fine, state, goal, xyz, fine_xyz, valid, fine_valid = context
    repeat = count // coarse.shape[1]
    context = (coarse.repeat(1, repeat, 1), fine.repeat(1, repeat, 1), state, goal,
               xyz.repeat(1, repeat, 1), fine_xyz.repeat(1, repeat, 1),
               valid.repeat(1, repeat), fine_valid.repeat(1, repeat))
    context[-1][0] = False
    memory = decoder.memory(*context)
    attention = RelativeAttention(4, 2, 0.)
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.zero_()
        attention.value.weight.copy_(torch.eye(4))
        attention.output.weight.copy_(torch.eye(4))
    values = torch.zeros(2, memory[0].shape[1], 4)
    values[:, -1] = 1.
    prediction = attention(torch.zeros(2, 1, 4), values, torch.zeros(2, 1, 3),
                           memory[1], memory[2], memory[4])
    torch.testing.assert_close(prediction, torch.full_like(prediction, .1 / 1.1))
    prediction.sum().backward()
    assert decoder.key_gate_logit.grad is not None and decoder.key_gate_logit.grad.abs() > 0


def test_spatial_route_uses_predicted_goal_and_closes_without_xyz_leakage():
    _, context, _, _ = inputs()
    decoder = SpatialChunkDecoder(4, chunk_settings(guided_config()), 3, 1 / 60).eval()
    # Open the zero-initialized cross-attention residual for this route test.
    dim = decoder.settings['hidden_dim']
    with torch.no_grad():
        decoder.blocks[0].modulation[-1].bias[5 * dim:6 * dim] = 1.
    changed = list(context)
    changed[3] = context[3].clone()
    changed[3][:, :3] += .3
    def spatial_only(context):
        memory = decoder.memory(*context)
        # Remove the original goal->AdaLN route to isolate the new token.
        return decoder.sample((*memory[:3], torch.zeros_like(memory[3]), memory[4]), context[2])
    first = spatial_only(context)
    assert not torch.allclose(first[0], spatial_only(changed)[0], atol=1e-7, rtol=0)
    first[0].square().mean().backward()
    assert decoder.goal.weight.grad.abs().sum() > 0
    assert decoder.blocks[0].cross_attention.bias[-1].weight.grad.abs().sum() > 0
    # Closing the gate removes both semantic and XYZ influence, even with no
    # valid visual surfaces. The state token remains a finite fallback.
    context = list(context)
    context[6], context[7] = torch.zeros_like(context[6]), torch.zeros_like(context[7])
    changed[6], changed[7] = context[6], context[7]
    with torch.no_grad():
        decoder.key_gate_logit.fill_(-1000.)
    for a, b in zip(decoder(*context), decoder(*changed)):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=0, rtol=0)


@pytest.mark.parametrize('kind,representation', [('regression', 'delta_pose'),
    ('regression', 'relative_pose'), ('flow_matching', 'relative_pose')])
def test_guided_training_and_checkpoint_predictions_ignore_future_labels(chunk_fixture, kind, representation):
    config = guided_config(kind, representation)
    agent, data, _ = temporal_agent(chunk_fixture.root, config=config)
    sample = data[0]
    expected = agent.predict(sample, sample['goal'])['actions']
    altered = copy.deepcopy(sample)
    for name in ('action_chunk', 'key_action', 'joint_chunk'):
        altered[name][...] = float('nan')
    np.testing.assert_array_equal(expected, agent.predict(altered, sample['goal'])['actions'])
    result = agent.update(mixed_batch(data))
    assert np.isfinite(result['total_loss'])
    saved = dict(config=config, model_state=copy.deepcopy(model_state(agent)))
    restored, _, _ = temporal_agent(data.root, config=copy.deepcopy(config))
    load_weights(restored, saved)
    np.testing.assert_array_equal(agent.predict(sample, sample['goal'])['actions'],
                                  restored.predict(sample, sample['goal'])['actions'])


def test_guidance_requires_new_run_but_can_initialize_from_compatible_delta_weights(chunk_fixture):
    config = state_config(representation='delta_pose')
    legacy, data, _ = temporal_agent(chunk_fixture.root, config=config)
    with torch.no_grad():
        legacy._net_mod.decoder.key_gate_logit.fill_(-20.)
    saved = dict(config=config, model_state=copy.deepcopy(model_state(legacy)))
    guided = copy.deepcopy(config)
    guided['chunk_policy']['spatial_key_token'] = True
    agent, _, _ = temporal_agent(data.root, config=guided)
    with pytest.raises(ValueError, match='mismatch'):
        load_weights(agent, saved)
    load_weights(agent, saved, initialize=True)
    for name, value in model_state(agent).items():
        if name == 'decoder.key_gate_logit':
            assert value.sigmoid().item() == pytest.approx(.1)
        else:
            torch.testing.assert_close(value, saved['model_state'][name], rtol=0, atol=0)
    assert np.isfinite(agent.predict(data[0], data[0]['goal'])['actions']).all()
