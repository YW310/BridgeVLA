"""Behavioral checks for the independent observed-geometry residual head."""

import math

import pytest
import torch


from finetune.bridgevla.models import local_geometry_refiner as module
LocalGeometryRefiner = module.LocalGeometryRefiner
pose_refinement_loss = module.pose_refinement_loss
pose_error_metrics = module.pose_error_metrics


@pytest.fixture(autouse=True, scope='module')
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def pose(batch=1, dtype=torch.float32):
    return torch.eye(4, dtype=dtype).repeat(batch, 1, 1)


def z_rotation(angle, dtype=torch.float32):
    cosine, sine = math.cos(angle), math.sin(angle)
    return torch.tensor([[cosine, -sine, 0.], [sine, cosine, 0.], [0., 0., 1.]], dtype=dtype)


def observed_field(batch=1):
    field = torch.zeros(batch, 5, 8, 8, 8)
    field[:, 3] = 1
    return field


def test_zero_initialization_exactly_preserves_nontrivial_candidate():
    torch.manual_seed(9)
    model = LocalGeometryRefiner(3, hidden_dim=8)
    candidate = pose(2)
    candidate[0, :3, :3] = z_rotation(.8)
    candidate[:, :3, 3] = torch.randn(2, 3)
    result = model(observed_field(2), candidate, torch.randn(2, 3))
    assert set(result) == {'pose', 'translation_local', 'rotation_vector'}
    torch.testing.assert_close(result['pose'], candidate, rtol=0, atol=0)
    assert result['translation_local'].count_nonzero() == 0
    assert result['rotation_vector'].count_nonzero() == 0


def test_synthetic_training_learns_opposite_spatial_corrections():
    torch.manual_seed(14)
    model = LocalGeometryRefiner(2, hidden_dim=8)
    field = observed_field(4)
    # Identical context and support mass; only the support's spatial location
    # identifies the correction. Global mean pooling alone would lose it.
    field[:2, 1, :2, 2:6, 2:6] = 1
    field[2:, 1, -2:, 2:6, 2:6] = 1
    candidate, target = pose(4), pose(4)
    target[:2, 0, 3], target[2:, 0, 3] = .006, -.006
    target[:2, :3, :3] = z_rotation(.04)
    target[2:, :3, :3] = z_rotation(-.04)
    context = torch.zeros(4, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=.02)
    initial = pose_refinement_loss(model(field, candidate, context)['pose'], target).item()
    for _ in range(65):
        optimizer.zero_grad(set_to_none=True)
        loss = pose_refinement_loss(model(field, candidate, context)['pose'], target)
        loss.backward()
        optimizer.step()
    result = model(field, candidate, context)
    final = pose_refinement_loss(result['pose'], target).item()
    assert final < initial * .1
    assert result['translation_local'][:2, 0].min() > 0
    assert result['translation_local'][2:, 0].max() < 0


def test_vector_norms_stay_strictly_bounded_for_extreme_finite_outputs():
    model = LocalGeometryRefiner(2, hidden_dim=4, max_translation_m=.013, max_rotation_rad=.12)
    with torch.no_grad():
        model.output.bias.fill_(1e25)
    result = model(observed_field(2), pose(2), torch.zeros(2, 2))
    assert torch.isfinite(result['pose']).all()
    assert torch.linalg.vector_norm(result['translation_local'], dim=-1).max() < .013
    assert torch.linalg.vector_norm(result['rotation_vector'], dim=-1).max() < .12
    torch.testing.assert_close(torch.linalg.det(result['pose'][:, :3, :3]), torch.ones(2))


def test_candidate_rotation_maps_local_translation_to_world():
    model = LocalGeometryRefiner(1, hidden_dim=4)
    with torch.no_grad():
        model.output.bias[0] = 1
    candidate = pose()
    candidate[0, :3, :3] = z_rotation(math.pi / 2)
    candidate[0, :3, 3] = torch.tensor([.2, -.1, .3])
    result = model(observed_field(), candidate, torch.zeros(1, 1))
    displacement = result['pose'][0, :3, 3] - candidate[0, :3, 3]
    expected = torch.stack((torch.zeros(()), result['translation_local'][0, 0], torch.zeros(())))
    torch.testing.assert_close(displacement, expected, rtol=1e-5, atol=1e-8)


def test_rotation_is_right_composed_in_candidate_frame():
    model = LocalGeometryRefiner(1, hidden_dim=4)
    with torch.no_grad():
        model.output.bias[5] = .7
    candidate = pose()
    candidate[0, :3, :3] = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    result = model(observed_field(), candidate, torch.zeros(1, 1))
    local_rotation = z_rotation(result['rotation_vector'][0, 2].item())
    torch.testing.assert_close(result['pose'][0, :3, :3], candidate[0, :3, :3] @ local_rotation)
    assert not torch.allclose(result['pose'][0, :3, :3], local_rotation @ candidate[0, :3, :3])


def test_missing_known_field_returns_base_even_with_nonzero_prediction():
    model = LocalGeometryRefiner(2, hidden_dim=4)
    with torch.no_grad():
        model.output.bias.fill_(1)
    field = observed_field(2)
    field[0, 3] = 0
    field[0, 1] = 1  # Support and context cannot override the known mask.
    candidate = pose(2)
    candidate[:, :3, 3] = .3
    result = model(field, candidate, torch.randn(2, 2))
    torch.testing.assert_close(result['pose'][0], candidate[0], rtol=0, atol=0)
    assert result['translation_local'][0].count_nonzero() == 0
    assert result['rotation_vector'][0].count_nonzero() == 0
    assert result['translation_local'][1].count_nonzero() == 3


def test_observation_is_detached_but_context_and_model_receive_gradients():
    torch.manual_seed(11)
    model = LocalGeometryRefiner(2, hidden_dim=4)
    with torch.no_grad():
        model.output.weight.normal_(std=.1)
    field = observed_field().requires_grad_()
    context = torch.randn(1, 2, requires_grad=True)
    target = pose()
    target[0, 0, 3] = .01
    loss = pose_refinement_loss(model(field, pose(), context)['pose'], target)
    loss.backward()
    assert field.grad is None
    assert context.grad is not None and torch.isfinite(context.grad).all()
    assert context.grad.abs().sum() > 0
    gradient = model.geometry_encoder[0].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


@pytest.mark.parametrize('angle', [0., 1e-10, math.pi - 1e-7, math.pi])
def test_so3_exp_and_gradients_are_stable_near_zero_and_pi(angle):
    omega = torch.tensor([[0., 0., angle]], dtype=torch.float64, requires_grad=True)
    rotation = module._so3_exp(omega)
    torch.testing.assert_close(rotation[0], z_rotation(angle, torch.float64), rtol=1e-10, atol=1e-10)
    assert torch.autograd.gradcheck(module._so3_exp, (omega,), eps=1e-6, atol=1e-6)
    predicted = torch.cat((torch.cat((rotation, torch.zeros(1, 3, 1, dtype=torch.float64)), dim=-1),
                           pose(dtype=torch.float64)[:, 3:4]), dim=1)
    loss = pose_refinement_loss(predicted, pose(dtype=torch.float64))
    loss.backward()
    assert torch.isfinite(omega.grad).all()
    error = pose_error_metrics(predicted.detach(), pose(dtype=torch.float64))
    torch.testing.assert_close(error['rotation_rad'], torch.tensor([angle], dtype=torch.float64),
                               rtol=1e-8, atol=1e-10)


def test_metric_units_and_loss_scaling_are_explicit():
    predicted, target = pose(2, torch.float64), pose(2, torch.float64)
    target[:, :3, 3] = torch.tensor([.003, .004, 0.], dtype=torch.float64)
    target[1, :3, :3] = z_rotation(math.pi / 2, torch.float64)
    metrics = pose_error_metrics(predicted, target)
    torch.testing.assert_close(metrics['translation_m'], torch.tensor([.005, .005], dtype=torch.float64))
    torch.testing.assert_close(metrics['rotation_rad'], torch.tensor([0., math.pi / 2], dtype=torch.float64))
    assert pose_refinement_loss(predicted, predicted).item() == 0
    translation_only = pose_refinement_loss(predicted[:1], target[:1], translation_scale_m=.01)
    torch.testing.assert_close(translation_only, torch.tensor(.125, dtype=torch.float64))
    assert pose_refinement_loss(predicted, target, rotation_scale_rad=.2) < pose_refinement_loss(predicted, target)


@pytest.mark.parametrize('kind', ['field_shape', 'empty', 'field_finite', 'context_shape',
                                 'context_finite', 'pose_batch', 'pose_finite', 'reflection',
                                 'scaled_rotation', 'homogeneous_row', 'dtype', 'integer'])
def test_malformed_inputs_are_rejected(kind):
    model = LocalGeometryRefiner(2, hidden_dim=4)
    field, candidate, context = observed_field(), pose(), torch.zeros(1, 2)
    if kind == 'field_shape':
        field = field[:, :4]
    elif kind == 'empty':
        field = field[:, :, :0]
    elif kind == 'field_finite':
        field[0, 0, 0, 0, 0] = float('nan')
    elif kind == 'context_shape':
        context = torch.zeros(1, 3)
    elif kind == 'context_finite':
        context[0, 0] = float('inf')
    elif kind == 'pose_batch':
        candidate = pose(2)
    elif kind == 'pose_finite':
        candidate[0, 0, 3] = float('inf')
    elif kind == 'reflection':
        candidate[0, 0, 0] = -1
    elif kind == 'scaled_rotation':
        candidate[0, 0, 0] = 2
    elif kind == 'homogeneous_row':
        candidate[0, 3, 0] = .2
    elif kind == 'dtype':
        context = context.double()
    elif kind == 'integer':
        field = field.long()
    with pytest.raises(ValueError):
        model(field, candidate, context)


@pytest.mark.parametrize('kwargs', [dict(context_dim=-1), dict(context_dim=True),
                                  dict(context_dim=1.2), dict(context_dim=1, hidden_dim=0),
                                  dict(context_dim=1, max_translation_m=0),
                                  dict(context_dim=1, max_rotation_rad=float('inf'))])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        LocalGeometryRefiner(**kwargs)


def test_loss_and_metrics_validate_poses_and_scales():
    bad = pose()
    bad[0, 1, 1] = 2
    for function in (pose_refinement_loss, pose_error_metrics):
        with pytest.raises(ValueError, match='rigid'):
            function(pose(), bad)
        with pytest.raises(ValueError, match='matching'):
            function(pose(), pose(2))
    with pytest.raises(ValueError, match='translation_scale_m'):
        pose_refinement_loss(pose(), pose(), translation_scale_m=0)
    with pytest.raises(ValueError, match='rotation_scale_rad'):
        pose_refinement_loss(pose(), pose(), rotation_scale_rad=float('nan'))


def test_zero_width_context_is_supported():
    result = LocalGeometryRefiner(0, hidden_dim=4)(observed_field(), pose(), torch.empty(1, 0))
    torch.testing.assert_close(result['pose'], pose(), rtol=0, atol=0)
