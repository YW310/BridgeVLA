"""Quantifies how well the flow training objective predicts deployed Euler sampling.

Training optimizes a one-step clean estimate from a noisy mixture at a random
``tau`` (and a gripper logit at that same ``tau``); deployment integrates from
pure noise and reads the gripper at ``tau=1`` on the sampled pose. The report
exposes each gap separately so ablations can attribute success-rate changes.
"""
import torch
import torch.nn.functional as F

from .chunk_model import quaternion_matrix

TAU_GRID = (.1, .3, .5, .7, .9)


def _errors(position, rotation, grip, target, valid):
    translation = torch.linalg.vector_norm(position - target[..., :3], dim=-1)[valid]
    relative = rotation.transpose(-1, -2) @ quaternion_matrix(target[..., 3:7])
    trace = (relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    degrees = torch.rad2deg(torch.acos(trace.clamp(-1., 1.)))[valid]
    correct = ((grip > 0) == (target[..., 7] > .5))[valid].float()
    return translation, degrees, correct


def _summary(position, rotation, grip, target, valid):
    translation, degrees, correct = _errors(position, rotation, grip, target, valid)
    return dict(position_error_m=translation.mean().item(), rotation_error_degrees=degrees.mean().item(),
                gripper_accuracy=correct.mean().item())


@torch.no_grad()
def flow_consistency_report(decoder, context, target, valid, *, steps=(1, 2, 4, 8, 16), seeds=8,
                            tau_grid=TAU_GRID, noise_seed=0):
    """Compare one-step training reconstruction with multi-step sampling on one decoder context.

    ``context`` is the decoder input tuple ``(coarse, fine, state, goal, coarse_xyz, fine_xyz,
    coarse_valid, fine_valid)`` as produced by ``ChunkAgent._context`` (one sample per row).
    """
    if not decoder.is_flow:
        raise ValueError('Flow consistency requires decoder_type=flow_matching')
    horizon = decoder.settings['horizon']
    target, valid = target[:, :horizon], valid[:, :horizon]
    if not valid.any():
        raise ValueError('Flow consistency requires valid future supervision')
    training = decoder.training
    decoder.eval()
    try:
        state = context[2]
        memory = decoder.memory(*context)
        generator = torch.Generator(device=state.device).manual_seed(noise_seed)
        noise = torch.randn(len(state), horizon, 9, device=state.device, dtype=state.dtype, generator=generator)

        reconstruction = {}
        for tau_value in tau_grid:
            tau = state.new_full((len(state),), tau_value)
            parts = decoder.flow_reconstruction(memory, state, target, valid, noise, tau)
            decoded = decoder.decode(parts['reconstruction'], parts['grip'], state)
            reconstruction[f'{tau_value:g}'] = _summary(*decoded, target, valid)
        # Weighted by the configured training tau distribution, this is what
        # the reconstruction term of the training loss actually measures.
        draws = decoder.sample_flow_time(state.new_zeros(4096, state.shape[1]))
        grid = state.new_tensor(tau_grid)
        counts = torch.bincount((draws[:, None] - grid).abs().argmin(-1), minlength=len(tau_grid)).float()
        position_weights = {f'{value:g}': weight for value, weight in zip(tau_grid, (counts / counts.sum()).tolist())}
        training_weighted = sum(position_weights[name] * reconstruction[name]['position_error_m']
                                for name in reconstruction)

        sampling = {}
        for count in steps:
            runs = [_summary(*decoder.sample(memory, state, steps=count, seed=noise_seed + seed), target, valid)
                    for seed in range(seeds)]
            mean = {name: sum(run[name] for run in runs) / len(runs) for name in runs[0]}
            spread = torch.tensor([run['position_error_m'] for run in runs])
            mean['position_error_seed_std_m'] = spread.std(unbiased=False).item() if len(runs) > 1 else 0.
            sampling[str(count)] = mean

        # Gripper at deployment reads tau=1; the clean-target read is its upper bound, and the
        # tau mass above 0.9 shows how rarely training supervises that regime.
        clean = decoder.encode_target(
            torch.where(valid[..., None], target, torch.cat((state[:, None, :7], state.new_zeros(len(state), 1, 1)), -1)),
            state)
        _, oracle_grip = decoder.field(clean, state.new_ones(len(state)), memory)
        oracle = ((oracle_grip > 0) == (target[..., 7] > .5))[valid].float().mean().item()
        gripper = dict(oracle_tau1_accuracy=oracle,
                       deployed_accuracy=sampling.get(str(decoder.settings['flow_steps']), {}).get('gripper_accuracy'),
                       oracle_tau1_bce=F.binary_cross_entropy_with_logits(
                           oracle_grip[valid], target[..., 7][valid]).item(),
                       training_tau_mass_above_0_9=(draws > .9).float().mean().item())
    finally:
        decoder.train(training)
    deployed_error = sampling.get(str(decoder.settings['flow_steps']), {}).get('position_error_m')
    return dict(reconstruction_by_tau=reconstruction, training_tau_weights=position_weights, sampling_by_steps=sampling,
                training_weighted_reconstruction_position_error_m=training_weighted,
                deployed_position_error_m=deployed_error,
                reconstruction_minus_deployed_m=None if deployed_error is None else training_weighted - deployed_error,
                gripper=gripper)
