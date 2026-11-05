"""Flow training/inference consistency: shared reconstruction path, oracle round trip and the diagnostic report."""
import pytest
import torch

from finetune.OHT.flow_diagnostics import flow_consistency_report
from tests.test_oht_chunk_replay import chunk_fixture, replay_fixture  # noqa: F401
from tests.test_oht_spatial_chunk import inputs


def oracle_decoder(delta_pose):
    decoder, context, target, valid = inputs()
    decoder.delta_pose = delta_pose
    state = context[2]
    clean = decoder.encode_target(target, state)

    # Exact marginal velocity of a point-mass target: Euler lands on it for any step count.
    def field(pose, tau, memory):
        velocity = (clean - pose) / (1 - tau)[:, None, None]
        return velocity, (target[..., 7] * 2 - 1) * 10
    decoder.field = field
    return decoder, context, target, valid


@pytest.mark.parametrize('delta_pose', [False, True])
@pytest.mark.parametrize('steps', [1, 3, 8])
def test_euler_sampling_inverts_target_encoding_for_any_step_count(delta_pose, steps):
    decoder, context, target, valid = oracle_decoder(delta_pose)
    state = context[2]
    position, rotation, grip = decoder.sample(None, state, steps=steps, seed=5)
    torch.testing.assert_close(position, target[..., :3], atol=1e-5, rtol=0)
    expected = decoder.decode(decoder.encode_target(target, state), grip, state)[1]
    torch.testing.assert_close(rotation, expected, atol=1e-5, rtol=0)
    assert ((grip > 0) == (target[..., 7] > .5)).all()


def test_sample_defaults_match_configured_steps_and_seed_but_accept_overrides():
    decoder, context, _, _ = inputs()
    memory = decoder.memory(*context)
    state = context[2]
    default = decoder.sample(memory, state)
    explicit = decoder.sample(memory, state, steps=decoder.settings['flow_steps'], seed=decoder.settings['flow_inference_seed'])
    for left, right in zip(default, explicit):
        torch.testing.assert_close(left, right)
    other = decoder.sample(memory, state, seed=decoder.settings['flow_inference_seed'] + 1)
    assert not torch.allclose(default[0], other[0])


def test_training_prediction_uses_the_shared_reconstruction_path():
    decoder, context, target, valid = inputs()
    decoder.eval()
    memory = decoder.memory(*context)
    state = context[2]
    torch.manual_seed(3)
    prediction, extra = decoder.training_prediction(*context, target=target, valid=valid)
    torch.manual_seed(3)
    noise = torch.randn(len(state), 4, 9)
    parts = decoder.flow_reconstruction(memory, state, target, valid, noise, decoder.sample_flow_time(state))
    expected = decoder.decode(parts['reconstruction'], parts['grip'], state)
    for left, right in zip(prediction, expected):
        torch.testing.assert_close(left, right)
    torch.testing.assert_close(extra['flow_loss'], decoder.settings['position_weight'] * torch.nn.functional.mse_loss(
        parts['velocity'][..., :3][valid], parts['expected'][..., :3][valid]) + decoder.settings['rotation_weight'] *
        torch.nn.functional.mse_loss(parts['velocity'][..., 3:9][valid], parts['expected'][..., 3:9][valid]))


def test_report_is_exact_for_oracle_field_and_restores_training_mode():
    decoder, context, target, valid = oracle_decoder(False)
    decoder.train()
    report = flow_consistency_report(decoder, context, target, valid, steps=(1, 2), seeds=3)
    assert decoder.training
    for row in report['sampling_by_steps'].values():
        assert row['position_error_m'] == pytest.approx(0., abs=1e-5)
        assert row['rotation_error_degrees'] == pytest.approx(0., abs=.1)
        assert row['gripper_accuracy'] == 1.
        assert row['position_error_seed_std_m'] == pytest.approx(0., abs=1e-5)
    # At tau<1 the oracle velocity still yields the exact one-step clean estimate.
    for row in report['reconstruction_by_tau'].values():
        assert row['position_error_m'] == pytest.approx(0., abs=1e-5)
    assert report['gripper']['oracle_tau1_accuracy'] == 1.
    assert sum(report['training_tau_weights'].values()) == pytest.approx(1.)


def test_report_exposes_high_noise_weighting_and_untrained_gap():
    decoder, context, target, valid = inputs()
    decoder.settings['flow_time_sampling'] = 'beta_high_noise'
    torch.manual_seed(1)
    report = flow_consistency_report(decoder, context, target, valid, steps=(1, decoder.settings['flow_steps']), seeds=2)
    weights = report['training_tau_weights']
    assert weights['0.1'] + weights['0.3'] > weights['0.7'] + weights['0.9']
    assert 0 <= report['gripper']['training_tau_mass_above_0_9'] < .1
    assert report['deployed_position_error_m'] == report['sampling_by_steps'][str(decoder.settings['flow_steps'])]['position_error_m']
    assert report['reconstruction_minus_deployed_m'] == pytest.approx(
        report['training_weighted_reconstruction_position_error_m'] - report['deployed_position_error_m'])


def test_report_rejects_regression_decoder():
    decoder, context, target, valid = inputs('regression')
    with pytest.raises(ValueError):
        flow_consistency_report(decoder, context, target, valid)


def test_replay_flow_consistency_uses_deployed_context_and_cli_report(chunk_fixture, tmp_path):
    import json
    from finetune.OHT.chunk_eval import flow_consistency_chunk
    from finetune.OHT.runtime.policy import Policy
    from tests.test_oht_spatial_chunk import spatial_agent
    agent, data, _ = spatial_agent(chunk_fixture.root, 'flow_matching')
    policy = Policy(agent, data.contract, 'cpu', 'baseline')
    report = flow_consistency_chunk(policy, chunk_fixture.root, 'val', limit=2, steps=(1, 2), seeds=2)
    assert report['kind'] == 'flow_consistency' and report['samples'] == 2
    assert set(report['report']['sampling_by_steps']) == {'1', '2'}
    json.dumps(report)
    sample = data.__class__(chunk_fixture.root, 'val')[0]
    context = policy.chunk_decoder_context(sample, sample['goal'])
    assert len(context) == 8 and context[2].shape == (1, 14)
    regression, _, _ = spatial_agent(chunk_fixture.root, 'regression')
    with pytest.raises(ValueError, match='flow_matching'):
        flow_consistency_chunk(Policy(regression, data.contract, 'cpu', 'baseline'), chunk_fixture.root, 'val', limit=1)


def test_cli_registers_flow_consistency_options():
    from finetune.OHT import eval as evaluation
    with pytest.raises(SystemExit):
        evaluation.main(['flow-consistency'])
