"""TCP-local key conditioning without changing world EEF labels or deployment."""
import copy

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from tests.test_oht_chunk_pipeline import KeyAgent, agent_data, settings_config
from tests.test_oht_chunk_replay import chunk_fixture, replay_fixture  # noqa: F401
from finetune.OHT.chunk_config import chunk_settings
from finetune.OHT.chunk_model import ChunkAgent, ChunkDecoder, quaternion_matrix
from finetune.OHT.config import load
from finetune.OHT.data.chunk_dataset import collate_chunk
from finetune.OHT.model import load_weights, model_state, resume_fingerprint


def decoder():
    config = settings_config()
    config['chunk_policy']['goal_conditioning'] = 'relative_tcp'
    return ChunkDecoder(8, chunk_settings(config))


def poses():
    state = torch.zeros(2, 14, dtype=torch.float64)
    state[:, :3] = torch.tensor([[.5, -.2, 1.3], [-.1, .4, .8]])
    current = Rotation.from_euler('xyz', [[20, -40, 90], [-60, 10, -20]], degrees=True)
    target = Rotation.from_euler('xyz', [[-35, 25, -10], [15, -25, 140]], degrees=True)
    state[:, 3:7] = torch.from_numpy(current.as_quat())
    goal = torch.tensor([[.6, -.25, 1.32, 0., 0., 0., 1., 1.],
                         [-.15, .42, .7, 0., 0., 0., 1., 0.]], dtype=torch.float64)
    goal[:, 3:7] = torch.from_numpy(target.as_quat())
    return state, goal, current, target


def test_relative_config_is_separate_and_legacy_stays_absolute():
    parent = load('finetune/OHT/configs/oht_chunk_v4_8x40.yaml')
    config = load('finetune/OHT/configs/oht_chunk_v4_relative_goal_8x40.yaml')
    assert chunk_settings(parent)['goal_conditioning'] == 'absolute'
    settings = chunk_settings(config)
    assert settings['goal_conditioning'] == 'relative_tcp'
    assert settings['displacement_weight'] == 0.
    assert settings['position_weight'] == 5.
    assert config['epochs'] == parent['epochs']
    assert config['freeze_gemma_prefix_layers'] == parent['freeze_gemma_prefix_layers']


@pytest.mark.parametrize('value', [None, True, 'relative_eef', 'world_delta', 1, {}])
def test_invalid_conditioning_rejected(value):
    config = settings_config()
    config['chunk_policy']['goal_conditioning'] = value
    with pytest.raises(ValueError, match='goal_conditioning'):
        chunk_settings(config)


def test_relative_pose_matches_se3_inverse_and_preserves_inputs():
    model = decoder()
    state, goal, current, target = poses()
    before = state.clone(), goal.clone()
    condition = model.goal_condition(state, goal)
    expected_position = current.inv().apply((goal[:, :3] - state[:, :3]).numpy()) / .25
    np.testing.assert_allclose(condition[:, :3].numpy(), expected_position, atol=1e-12)
    np.testing.assert_allclose(quaternion_matrix(condition[:, 3:7]).numpy(),
                               (current.inv() * target).as_matrix(), atol=1e-12)
    torch.testing.assert_close(condition[:, 7], goal[:, 7])
    torch.testing.assert_close(state, before[0])
    torch.testing.assert_close(goal, before[1])
    same_pose = torch.cat((state[:, :7], goal[:, 7:8]), dim=-1)
    expected = torch.zeros_like(same_pose)
    expected[:, 6] = 1.
    expected[:, 7] = goal[:, 7]
    torch.testing.assert_close(model.goal_condition(state, same_pose), expected)


def test_local_condition_invariant_to_common_world_transform_and_quaternion_signs():
    model = decoder()
    state, goal, current, target = poses()
    expected = model.goal_condition(state, goal)
    world = Rotation.from_euler('xyz', [23, 70, -110], degrees=True)
    translation = np.array([2., -1., .3])
    moved_state, moved_goal = state.clone(), goal.clone()
    for value, orientation in ((moved_state, current), (moved_goal, target)):
        value[:, :3] = torch.from_numpy(world.apply(value[:, :3].numpy()) + translation)
        value[:, 3:7] = torch.from_numpy((world * orientation).as_quat())
    torch.testing.assert_close(model.goal_condition(moved_state, moved_goal), expected)
    for flip_current, flip_goal in ((True, False), (False, True), (True, True)):
        signed_state, signed_goal = state.clone(), goal.clone()
        if flip_current:
            signed_state[:, 3:7] *= -1
        if flip_goal:
            signed_goal[:, 3:7] *= -1
        torch.testing.assert_close(model.goal_condition(signed_state, signed_goal), expected)
    # Exact half-turn must not produce sign-dependent tokens either.
    state[:, 3:7] = torch.tensor([0., 0., 0., 1.])
    goal[:, 3:7] = torch.tensor([1., 0., 0., 0.])
    expected = model.goal_condition(state, goal)
    goal[:, 3:7] *= -1
    torch.testing.assert_close(model.goal_condition(state, goal), expected)


def test_decoder_uses_local_condition_but_outputs_absolute_world_actions():
    model = decoder().double()
    state, goal, _, _ = poses()
    seen = []
    handle = model.goal.register_forward_pre_hook(lambda module, args: seen.append(args[0].clone()))
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias[:3] = torch.tensor([.1, -.2, .3])
    position, rotation, _ = model(torch.randn(2, 3, 8), torch.randn(2, 3, 8), state, goal)
    handle.remove()
    torch.testing.assert_close(seen[0], model.goal_condition(state, goal))
    expected = state[:, None, :3] + model.head.bias[:3] * .25
    torch.testing.assert_close(position, expected.expand_as(position))
    torch.testing.assert_close(rotation, torch.eye(3, dtype=torch.float64).expand_as(rotation))
    (position.square().mean() + rotation.square().mean()).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


@pytest.mark.parametrize('source_mode,target_mode', [('absolute', 'relative_tcp'), ('relative_tcp', 'absolute')])
def test_condition_change_resets_only_goal_projection_and_requires_new_run(chunk_fixture, source_mode, target_mode):
    _, data, config = agent_data(chunk_fixture)
    config['chunk_policy']['goal_conditioning'] = source_mode
    source = ChunkAgent(KeyAgent(), chunk_settings(config, data.contract), data.contract, config, torch.device('cpu'))
    source.update(collate_chunk([data[0], data[-1]]))
    old_config = copy.deepcopy(config)
    if source_mode == 'absolute':
        old_config['chunk_policy'].pop('goal_conditioning')  # Legacy checkpoint.
    checkpoint = dict(config=old_config, model_state=copy.deepcopy(model_state(source)))
    config['chunk_policy']['goal_conditioning'] = target_mode
    target = ChunkAgent(KeyAgent(), chunk_settings(config, data.contract), data.contract, config, torch.device('cpu'))
    before = copy.deepcopy(model_state(target))
    with pytest.raises(ValueError, match='mismatch'):
        load_weights(target, checkpoint)
    load_weights(target, checkpoint, initialize=True)
    for name, value in model_state(target).items():
        expected = before[name] if name.startswith('decoder.goal.') else model_state(source)[name]
        torch.testing.assert_close(value, expected)
    assert not target._optimizer.state
    assert resume_fingerprint(config, data.contract, None, 1) != resume_fingerprint(old_config, data.contract, None, 1)
    assert np.isfinite(target.update(collate_chunk([data[0]]))['total_loss'])


def test_relative_training_and_prediction_share_condition_and_keep_key_world_pose(chunk_fixture):
    _, data, config = agent_data(chunk_fixture)
    config['chunk_policy']['goal_conditioning'] = 'relative_tcp'
    agent = ChunkAgent(KeyAgent(), chunk_settings(config, data.contract), data.contract, config, torch.device('cpu'))
    sample = data[0]
    context = agent._context(sample)
    expected = agent._net_mod.decoder.goal_condition(context[2], context[3])
    seen = []
    handle = agent._net_mod.decoder.goal.register_forward_pre_hook(
        lambda module, args: seen.append(args[0].detach().clone()))
    agent.update(collate_chunk([sample]), step_optimizer=False)
    predicted = agent.predict(sample, sample['goal'])
    handle.remove()
    assert len(seen) == 2
    for condition in seen:
        torch.testing.assert_close(condition, expected)
    np.testing.assert_array_equal(predicted['key_action'], context[3][0].numpy())
    changed = copy.deepcopy(sample)
    changed['key_action'][:] = 999.
    changed['action_chunk'][:] = -999.
    np.testing.assert_array_equal(agent.predict(changed, sample['goal'])['actions'], predicted['actions'])
    checkpoint = dict(config=config, model_state=copy.deepcopy(model_state(agent)))
    other = ChunkAgent(KeyAgent(), chunk_settings(config, data.contract), data.contract, config, torch.device('cpu'))
    load_weights(other, checkpoint)
    np.testing.assert_array_equal(other.predict(sample, sample['goal'])['actions'], predicted['actions'])


def test_legacy_absolute_condition_is_passed_through_unchanged():
    model = ChunkDecoder(8, chunk_settings(settings_config()))
    state, goal, _, _ = poses()
    assert model.goal_condition(state, goal) is goal
