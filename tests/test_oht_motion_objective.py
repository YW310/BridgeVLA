"""Executed-prefix supervision and motion baselines for sub-millimetre actions."""
import copy

import pytest
import torch
from torch.nn import functional as F

from finetune.OHT.chunk_config import chunk_settings
from finetune.OHT.chunk_config import chunk_scale
from finetune.OHT.chunk_eval import summarize_motion
from finetune.OHT.chunk_model import chunk_loss, quaternion_matrix
from finetune.OHT.config import load
from tests.test_oht_chunk_pipeline import settings_config


def loss_inputs():
    settings = chunk_settings(settings_config())
    target = torch.zeros(1, 4, 8)
    target[..., 6] = 1
    valid = torch.tensor([[True, True, True, False]])
    target[~valid] = float('nan')
    position = torch.zeros(1, 4, 3, requires_grad=True)
    rotation = torch.eye(3).repeat(1, 4, 1, 1).requires_grad_()
    grip = torch.zeros(1, 4, requires_grad=True)
    return settings, (position, rotation, grip), target, valid


def test_default_objective_matches_existing_checkpoint_loss():
    settings, pred, target, valid = loss_inputs()
    pred = (torch.full_like(pred[0], .003, requires_grad=True),
            (pred[1].detach() + .001).requires_grad_(), torch.full_like(pred[2], .2, requires_grad=True))
    scale = chunk_scale(settings, 'position_loss_scale')
    expected = (settings['position_weight'] * F.smooth_l1_loss(pred[0][valid] / scale, target[..., :3][valid] / scale)
                + settings['rotation_weight'] * F.mse_loss(pred[1][valid], quaternion_matrix(target[..., 3:7][valid]))
                + settings['gripper_weight'] * F.binary_cross_entropy_with_logits(pred[2][valid], target[..., 7][valid]))
    actual, _ = chunk_loss(pred, target, valid, settings)
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in pred)


def test_prefix_weight_prioritizes_executed_ticks_and_ignores_padding():
    settings, pred, target, valid = loss_inputs()
    settings.update(execute_horizon=1, prefix_loss_weight=4.)
    errors = []
    for tick in (0, 2):
        position = pred[0].detach().clone()
        position[0, tick] = .001
        _, metrics = chunk_loss((position, *pred[1:]), target, valid, settings)
        errors.append(metrics['chunk_position_loss'])
    assert errors[0] == pytest.approx(4 * errors[1])


def test_rotation_scale_exposes_small_errors_with_finite_gradients():
    settings, pred, target, valid = loss_inputs()
    angle = torch.tensor(.01, requires_grad=True)
    zero, one = angle * 0, angle * 0 + 1
    rotation = torch.stack((angle.cos(), -angle.sin(), zero,
                            angle.sin(), angle.cos(), zero, zero, zero, one)).reshape(3, 3).expand(1, 4, 3, 3)
    _, old = chunk_loss((pred[0], rotation, pred[2]), target, valid, settings)
    settings['rotation_loss_scale'] = .02
    loss, new = chunk_loss((pred[0], rotation, pred[2]), target, valid, settings)
    assert new['chunk_rotation_loss'] == pytest.approx(old['chunk_rotation_loss'] / .02 ** 2)
    loss.backward()
    assert torch.isfinite(angle.grad) and angle.grad > 0


@pytest.mark.parametrize('key,value', [('rotation_loss_scale', 0), ('rotation_loss_scale', float('nan')),
                                      ('prefix_loss_weight', -1), ('prefix_loss_weight', True)])
def test_objective_configuration_rejects_invalid_scales(key, value):
    config = settings_config()
    config['chunk_policy'][key] = value
    with pytest.raises(ValueError):
        chunk_settings(config)


def test_repair_preset_keeps_action_and_encoder_contract():
    path = 'finetune/OHT/configs/oht_chunk_v4_delta_state_tcp_guided'
    old, new = chunk_settings(load(path + '_8x40.yaml')), chunk_settings(load(path + '_motion_8x40.yaml'))
    assert {key for key in old if old[key] != new[key]} == {
        'position_loss_scale', 'displacement_scale', 'rotation_loss_scale', 'prefix_loss_weight'}
    assert new['horizon'] == 16 and new['execute_horizon'] == 4 and new['local_context'] == 'tcp'


def test_calibrated_baseline_changes_no_model_or_objective_settings():
    path = 'finetune/OHT/configs/oht_chunk_v4_delta_state_tcp_guided'
    old = load(path + '_8x40.yaml')
    calibrated = load(path + '_calibrated_8x40.yaml')
    assert chunk_settings(old) == chunk_settings(calibrated)
    assert calibrated['required_gripper_calibration'] == dict(open=-.04, close=-.006)
    assert {k for k in old if old[k] != calibrated[k]} == {'exp_id'}


def test_repair_preset_rejects_old_replay_before_building_model():
    config = load('finetune/OHT/configs/oht_chunk_v4_delta_state_tcp_guided_motion_8x40.yaml')
    contract = dict(schema='oht_bridgevla_v4', chunk=dict(horizon=16),
                    data_config=dict(gripper=dict(open=-.095094, close=-.0057778)))
    with pytest.raises(ValueError, match='rebuild raw V4 replay'):
        chunk_settings(config, contract)
    contract['data_config']['gripper'].update(open=-.04, close=-.006)
    chunk_settings(config, contract)


def test_motion_report_flags_forecasts_worse_than_holding_still():
    row = dict(translation_m=.0006, rotation_degrees=.2, stationary_translation_m=.00025,
               stationary_rotation_degrees=.1, gripper_correct=True)
    report = summarize_motion([row])
    assert report['translation_m_skill_vs_stationary'] == pytest.approx(-1.4)
    assert report['rotation_degrees_skill_vs_stationary'] == pytest.approx(-1.)
    row.update(stationary_translation_m=0., stationary_rotation_degrees=0.)
    assert summarize_motion([row])['translation_m_skill_vs_stationary'] is None
    assert summarize_motion([])['samples'] == 0


def test_explicit_nominal_gripper_preserves_release_despite_extreme_measurement():
    from finetune.OHT.data.actions import gripper_states
    # Two normal open plateaus surround a close; a later spike must not define
    # the normalization of every otherwise normal open frame in the episode.
    physical = [-.04] * 5 + torch.linspace(-.04, -.006, 30).tolist() + [-.006] * 5
    physical += torch.linspace(-.006, -.04, 30).tolist() + [-.04] * 5 + [-.095094]
    state = torch.zeros(len(physical), 7).numpy()
    state[:, -1] = physical
    _, wrong = gripper_states(state, dict(open=-.095094, close=-.006))
    opening, fixed = gripper_states(state, dict(open=-.04, close=-.006))
    assert wrong[-2] == 0 and fixed[-2] == 1 and opening[-2] == pytest.approx(1.)


def test_new_objective_initialization_preserves_weights_but_strict_resume_rejects_change(
        chunk_fixture, monkeypatch):
    from bridgevla.mvt import mvt
    from finetune.OHT.data.chunk_dataset import OHTChunkDataset
    from finetune.OHT.model import build_agent, load_weights, model_state
    from tests.test_oht_chunk_joint import JointMVT
    monkeypatch.setattr(mvt, 'MVT', JointMVT)
    config = settings_config()
    data = OHTChunkDataset(chunk_fixture.root)
    original = build_agent(config, data.contract, torch.device('cpu'))
    saved = dict(config=copy.deepcopy(config), model_state=copy.deepcopy(model_state(original)))
    config['chunk_policy'].update(rotation_loss_scale=.02, prefix_loss_weight=4.)
    new = build_agent(config, data.contract, torch.device('cpu'))
    with pytest.raises(ValueError, match='mismatch'):
        load_weights(new, saved)
    load_weights(new, saved, initialize=True)
    for name, value in model_state(new).items():
        torch.testing.assert_close(value, saved['model_state'][name])


from tests.test_oht_chunk_replay import chunk_fixture, replay_fixture  # noqa: E402,F401
