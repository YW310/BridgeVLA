"""Position weighting changes gradients without discarding learned trajectories."""
import copy

import pytest
import torch

from tests.test_oht_chunk_pipeline import KeyAgent, agent_data
from tests.test_oht_chunk_replay import chunk_fixture, replay_fixture  # noqa: F401
from finetune.OHT.chunk_config import chunk_settings
from finetune.OHT.chunk_model import ChunkAgent, chunk_loss
from finetune.OHT.config import load
from finetune.OHT.model import load_weights, model_state, resume_fingerprint


def test_full_chunk_config_weights_position_but_legacy_default_stays_one():
    config = load('finetune/OHT/configs/oht_chunk_v4_8x40.yaml')
    assert chunk_settings(config)['position_weight'] == 5.
    del config['chunk_policy']['position_weight']
    assert chunk_settings(config)['position_weight'] == 1.


@pytest.mark.parametrize('value', [-1., float('nan'), float('inf'), True, '5', None])
def test_position_weight_rejects_invalid_values(value):
    config = load('finetune/OHT/configs/oht_chunk_v4_8x40.yaml')
    config['chunk_policy']['position_weight'] = value
    with pytest.raises(ValueError, match='position_weight'):
        chunk_settings(config)


def test_position_weight_scales_only_position_gradient_and_excludes_padding():
    settings = chunk_settings(load('finetune/OHT/configs/oht_chunk_v4_8x40.yaml'))
    target = torch.tensor([[[0., 0., 0., 0., 0., 0., 1., 1.], [float('nan')] * 8]])
    valid = torch.tensor([[True, False]])
    gradients, reports = [], []
    for weight in (1., 5.):
        position = torch.tensor([[[.01, -.02, .03], [9., 9., 9.]]], requires_grad=True)
        rotation = torch.eye(3).repeat(1, 2, 1, 1).requires_grad_()
        grip = torch.zeros(1, 2, requires_grad=True)
        loss, metrics = chunk_loss((position, rotation, grip), target, valid,
                                   dict(settings, position_weight=weight))
        loss.backward()
        gradients.append((position.grad, rotation.grad, grip.grad))
        reports.append((loss.item(), metrics))
        assert torch.count_nonzero(position.grad[:, 1]) == 0
        assert torch.isfinite(position.grad).all()
        assert metrics['chunk_valid_actions'] == 1
    torch.testing.assert_close(gradients[1][0], gradients[0][0] * 5)
    torch.testing.assert_close(gradients[1][1], gradients[0][1])
    torch.testing.assert_close(gradients[1][2], gradients[0][2])
    original, weighted = reports[0][1], reports[1][1]
    assert weighted['chunk_position_loss'] == original['chunk_position_loss']
    assert weighted['chunk_position_weighted_loss'] == pytest.approx(5 * original['chunk_position_loss'])
    assert reports[1][0] - reports[0][0] == pytest.approx(4 * original['chunk_position_loss'], abs=1e-7)


@pytest.mark.parametrize('weight', ['position_weight', 'rotation_weight', 'gripper_weight'])
def test_loss_weight_change_requires_new_run_and_preserves_full_model(chunk_fixture, weight):
    agent, data, config = agent_data(chunk_fixture)
    old_config = copy.deepcopy(config)
    old_config['chunk_policy'].pop('position_weight')  # Older checkpoints predate the option.
    saved_agent = ChunkAgent(KeyAgent(), chunk_settings(old_config, data.contract), data.contract,
                             old_config, torch.device('cpu'))
    with torch.no_grad():
        saved_agent._net_mod.decoder.head.bias.add_(.02)
    checkpoint = dict(config=old_config, model_state=copy.deepcopy(model_state(saved_agent)))
    # Isolate each weight change, including implicit legacy position_weight=1.
    config = copy.deepcopy(old_config)
    config['chunk_policy'][weight] = 5.
    agent = ChunkAgent(KeyAgent(), chunk_settings(config, data.contract), data.contract,
                       config, torch.device('cpu'))
    with pytest.raises(ValueError, match='mismatch'):
        load_weights(agent, checkpoint)
    load_weights(agent, checkpoint, initialize=True)
    for name, value in model_state(saved_agent).items():
        torch.testing.assert_close(model_state(agent)[name], value)
    assert resume_fingerprint(config, data.contract, None, 1) != resume_fingerprint(old_config, data.contract, None, 1)


def test_legacy_checkpoint_still_loads_with_implicit_position_weight_one(chunk_fixture):
    _, data, config = agent_data(chunk_fixture)
    config['chunk_policy'].pop('position_weight')
    settings = chunk_settings(config, data.contract)
    agent = ChunkAgent(KeyAgent(), settings, data.contract, config, torch.device('cpu'))
    checkpoint = dict(config=config, model_state=copy.deepcopy(model_state(agent)))
    load_weights(agent, checkpoint)
    assert agent.settings['position_weight'] == 1.
