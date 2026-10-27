"""Measured-state anchoring, temporal masks and loss-only checkpoint migration."""
import copy

import pytest
import torch
from torch.nn import functional as F

from tests.test_oht_chunk_pipeline import KeyAgent, agent_data
from tests.test_oht_chunk_replay import chunk_fixture, replay_fixture  # noqa: F401
from finetune.OHT.chunk_config import chunk_settings
from finetune.OHT.chunk_model import ChunkAgent, chunk_loss
from finetune.OHT.config import load
from finetune.OHT.data.chunk_dataset import collate_chunk
from finetune.OHT.model import load_weights, model_state, resume_fingerprint


def example(predicted, observed, current, valid=None):
    position = torch.tensor([predicted], dtype=torch.float32, requires_grad=True)
    target = torch.zeros(1, len(observed), 8)
    target[0, :, :3] = torch.tensor(observed)
    target[..., 6] = 1.
    rotation = torch.eye(3).repeat(1, len(observed), 1, 1)
    grip = torch.zeros(1, len(observed))
    mask = torch.ones(1, len(observed), dtype=torch.bool) if valid is None else torch.tensor([valid])
    current = torch.tensor([current], dtype=torch.float32)
    config = load('finetune/OHT/configs/oht_chunk_v4_displacement_8x40.yaml')
    settings = chunk_settings(config)
    # Isolate auxiliary gradients while still computing the original losses.
    settings.update(position_weight=0., rotation_weight=0., gripper_weight=0.)
    return (position, rotation, grip), target, mask, current, settings


def test_experiment_inherits_baseline_and_legacy_displacement_is_disabled():
    parent = load('finetune/OHT/configs/oht_chunk_v4_8x40.yaml')
    experiment = load('finetune/OHT/configs/oht_chunk_v4_displacement_8x40.yaml')
    assert chunk_settings(parent)['displacement_weight'] == 0.
    assert chunk_settings(experiment)['displacement_weight'] == 1.
    assert chunk_settings(experiment)['displacement_scale'] == .01
    for name in ('position_weight', 'position_scale', 'horizon', 'stage'):
        assert chunk_settings(parent)[name] == chunk_settings(experiment)[name]
    for name in ('learning_rate', 'epochs', 'freeze_gemma_prefix_layers', 'freeze_vision_tower'):
        assert parent[name] == experiment[name]


@pytest.mark.parametrize('key,value', [
    ('displacement_weight', -1), ('displacement_weight', float('nan')),
    ('displacement_weight', True), ('displacement_weight', '1'),
    ('displacement_scale', 0), ('displacement_scale', -1),
    ('displacement_scale', float('inf')), ('displacement_scale', True),
])
def test_invalid_displacement_config_rejected(key, value):
    config = load('finetune/OHT/configs/oht_chunk_v4_displacement_8x40.yaml')
    config['chunk_policy'][key] = value
    with pytest.raises(ValueError, match=key):
        chunk_settings(config)


def test_exact_observed_motion_has_zero_loss_and_padding_has_zero_gradient():
    observed = [[.101, .2, .3], [.102, .2, .3], [float('nan')] * 3]
    prediction, target, valid, current, settings = example(observed, observed, [.1, .2, .3], [True, True, False])
    target[0, 2] = float('nan')
    loss, metrics = chunk_loss(prediction, target, valid, settings, current_position=current)
    assert loss.item() == 0.
    assert metrics['chunk_displacement_loss'] == 0.
    assert metrics['chunk_displacement_error_m'] == 0.
    assert metrics['chunk_stationary_displacement_error_m'] == pytest.approx(.001, abs=1e-8)
    loss.backward()
    assert torch.isfinite(prediction[0].grad).all()
    assert torch.count_nonzero(prediction[0].grad) == 0


def test_constant_bias_is_penalized_at_current_to_first_transition():
    observed = [[.101, 0., 0.], [.102, 0., 0.], [.103, 0., 0.]]
    predicted = [[x + .003, y, z] for x, y, z in observed]
    prediction, target, valid, current, settings = example(predicted, observed, [.1, 0., 0.])
    loss, metrics = chunk_loss(prediction, target, valid, settings, current_position=current)
    # Later differences are correct; a first-step jump must still be supervised.
    assert loss.item() == pytest.approx(.005, abs=1e-7)
    assert metrics['chunk_displacement_error_m'] == pytest.approx(.001, abs=1e-8)
    loss.backward()
    assert prediction[0].grad[0, 0, 0] > 0


@pytest.mark.parametrize('predicted', [
    [[0., 0., 0.], [0., 0., 0.]],  # Copying current state is not correct motion.
    [[-.001, 0., 0.], [-.002, 0., 0.]],  # Reversing direction must be penalized.
    [[.001, .001, -.001], [.002, .002, -.002]],  # Extra Y/Z movement.
])
def test_stationary_reversed_and_off_axis_motion_have_auxiliary_gradients(predicted):
    observed = [[.001, 0., 0.], [.002, 0., 0.]]
    prediction, target, valid, current, settings = example(predicted, observed, [0., 0., 0.])
    loss, metrics = chunk_loss(prediction, target, valid, settings, current_position=current)
    assert loss.item() > 0 and metrics['chunk_displacement_error_m'] > 0
    loss.backward()
    assert torch.isfinite(prediction[0].grad).all()
    assert torch.count_nonzero(prediction[0].grad) > 0


def test_displacement_never_bridges_an_invalid_tick():
    observed = [[.001, 0., 0.], [float('nan')] * 3, [.003, 0., 0.], [.004, 0., 0.]]
    predicted = [[.002, 0., 0.], [float('nan')] * 3, [.01, 0., 0.], [.012, 0., 0.]]
    prediction, target, valid, current, settings = example(predicted, observed, [0., 0., 0.], [True, False, True, True])
    loss, _ = chunk_loss(prediction, target, valid, settings, current_position=current)
    expected = F.smooth_l1_loss(torch.tensor([[.2, 0., 0.], [.2, 0., 0.]]),
                              torch.tensor([[.1, 0., 0.], [.1, 0., 0.]]))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(prediction[0].grad).all()
    assert torch.count_nonzero(prediction[0].grad[0, 1]) == 0


def test_single_tick_is_anchored_and_isolated_late_tick_has_no_auxiliary_pair():
    args = example([[.003, 0., 0.]], [[.001, 0., 0.]], [0., 0., 0.])
    prediction, target, valid, current, settings = args
    loss, _ = chunk_loss(prediction, target, valid, settings, current_position=current)
    assert loss.item() == pytest.approx(.5 * .2 ** 2 / 3)
    args = example([[float('nan')] * 3, [.003, 0., 0.]],
                   [[float('nan')] * 3, [.001, 0., 0.]], [0., 0., 0.], [False, True])
    prediction, target, valid, current, settings = args
    loss, metrics = chunk_loss(prediction, target, valid, settings, current_position=current)
    assert loss.item() == 0. and metrics['chunk_displacement_error_m'] == 0.
    loss.backward()
    assert torch.isfinite(prediction[0].grad).all()


def test_auxiliary_weight_scales_gradients_without_altering_raw_losses():
    gradients, reports = [], []
    for weight in (0., 1., 3.):
        args = example([[.002, .001, 0.], [.004, .002, 0.]],
                       [[.001, 0., 0.], [.002, 0., 0.]], [0., 0., 0.])
        prediction, target, valid, current, settings = args
        settings['displacement_weight'] = weight
        loss, metrics = chunk_loss(prediction, target, valid, settings, current_position=current)
        loss.backward()
        gradients.append(prediction[0].grad)
        reports.append(metrics)
    assert torch.count_nonzero(gradients[0]) == 0
    torch.testing.assert_close(gradients[2], gradients[1] * 3)
    assert reports[0]['chunk_displacement_loss'] == reports[2]['chunk_displacement_loss']
    assert reports[2]['chunk_displacement_weighted_loss'] == pytest.approx(3 * reports[1]['chunk_displacement_loss'])


def test_enabled_loss_requires_finite_measured_anchor():
    prediction, target, valid, current, settings = example([[.001, 0., 0.]], [[.001, 0., 0.]], [0., 0., 0.])
    with pytest.raises(ValueError, match='measured current TCP'):
        chunk_loss(prediction, target, valid, settings)
    for value in (torch.zeros(3), torch.full((1, 3), float('nan'))):
        with pytest.raises(ValueError, match='finite'):
            chunk_loss(prediction, target, valid, settings, current_position=value)
    # Legacy direct callers are still supported with the auxiliary disabled.
    loss, metrics = chunk_loss(prediction, target, valid, dict(settings, displacement_weight=0.))
    assert torch.isfinite(loss) and metrics['chunk_displacement_weighted_loss'] == 0.


@pytest.mark.parametrize('patch', [{'displacement_weight': 1.}, {'displacement_scale': .005}])
def test_auxiliary_changes_preserve_encoder_and_decoder_on_initialization(chunk_fixture, patch):
    source, data, old_config = agent_data(chunk_fixture)
    source.update(collate_chunk([data[0], data[-1]]))
    checkpoint = dict(config=old_config, model_state=copy.deepcopy(model_state(source)))
    config = copy.deepcopy(old_config)
    config['chunk_policy'].update(patch)
    target = ChunkAgent(KeyAgent(), chunk_settings(config, data.contract), data.contract,
                        config, torch.device('cpu'))
    with pytest.raises(ValueError, match='mismatch'):
        load_weights(target, checkpoint)
    load_weights(target, checkpoint, initialize=True)
    for name, value in model_state(source).items():
        torch.testing.assert_close(model_state(target)[name], value)
    assert not target._optimizer.state
    assert resume_fingerprint(config, data.contract, None, 1) != resume_fingerprint(old_config, data.contract, None, 1)
    metrics = target.update(collate_chunk([data[0], data[-1]]))
    assert metrics['chunk_displacement_error_m'] >= 0
    assert metrics['chunk_displacement_weighted_loss'] == pytest.approx(
        target.settings['displacement_weight'] * metrics['chunk_displacement_loss'])
