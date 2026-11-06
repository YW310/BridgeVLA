"""Coarse export uses actual predictor selection and measured render coverage."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'finetune'))
from bridgevla.models.coarse_slot_export import (
    build_coarse_slot_packet, coarse_point_support, cube_inverse_metadata,
)
from bridgevla.models.oracle_prior import InternalRoleQueryPredictor
from bridgevla.mvt.utils import place_pc_in_cube


class CoverageRenderer:
    default_color = 0.

    def __init__(self, coverage):
        self.coverage = coverage
        self.calls = 0

    def __call__(self, points, features, *, fix_cam, dyn_cam_info):
        self.calls += 1
        assert features.shape == (len(points), 1)
        assert torch.equal(features, torch.ones_like(features))
        assert fix_cam is True and dyn_cam_info is None
        return self.coverage[..., None]


def sample(with_mean=False):
    world = torch.tensor([[-.2, -.4, .7], [.4, .2, 1.4], [.1, .3, 1.2]])
    cube, reverse = place_pc_in_cube(world, with_mean_or_bounds=with_mean,
                                     scene_bounds=None if with_mean else [-.3, -.5, .6, .7, .5, 1.6])
    xyz = torch.zeros(1, 1, 3, 3, 4)
    for index in range(3):
        xyz[0, 0, :, 0, index] = cube[index]
    prior = torch.tensor([[[[[.9, .6], [.4, .2]], [[.8, .5], [.3, .1]]]]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    predictor = InternalRoleQueryPredictor(feature_channels=4, num_views=1, num_slots=2,
                 slot_dim=4, num_heads=2, decoder_layers=1, point_samples=5, soft_conditioning=True)
    points = predictor._extract_points(prior, xyz, valid)
    output = dict(object_slot_prior=prior.requires_grad_(), object_slot_points=points.requires_grad_(),
                  object_slot_confidence=torch.tensor([[.9, .8]], requires_grad=True),
                  object_slot_valid=valid, object_slot_role_token_valid=valid.clone(),
                  object_slot_reference_null_probability=torch.tensor([.1], requires_grad=True),
                  object_slot_reference_is_null=torch.tensor([False]),
                  object_slot_geometry=torch.zeros(1, 17, requires_grad=True),
                  mvt1_ori_img=torch.cat((xyz, torch.zeros_like(xyz)), 2))
    coverage = torch.zeros(1, 3, 4)
    coverage[0, 0, 0] = 1.
    coverage[0, 0, 2] = .5  # Antialiasing still supplies measured observation support.
    renderer = CoverageRenderer(coverage)
    network = SimpleNamespace(training=False, object_slot_predictor_type='role_queries',
                              object_slot_predictor1=predictor, renderer=renderer)
    return network, output, [cube], [reverse], [world]


@pytest.mark.parametrize('with_mean', [False, True])
def test_actual_cube_inverse_and_predictor_support_are_exported(with_mean):
    network, output, cube, reverse, world = sample(with_mean)
    result = build_coarse_slot_packet(network, output, cube, reverse, world)
    assert network.renderer.calls == 1
    assert result['source_frame'] == 'bridgevla_coarse'
    assert result['point_support_source'] == 'renderer_coverage_and_predictor_topk'
    assert result['slot_output']['points'].shape == (2, 5, 3)
    assert result['point_valid'].shape == (2, 5)
    assert result['point_valid'].dtype == np.bool_
    assert result['point_valid'].sum() == 4  # Not all finite, nonzero XYZ is supported.
    for role in range(2):
        selected = result['slot_output']['points'][role, result['point_valid'][role]]
        matrix = result['matrix_world_from_source']
        restored = np.c_[selected, np.ones(len(selected))] @ matrix.T
        for point in restored[:, :3]:
            assert np.min(np.linalg.norm(world[0].numpy()[[0, 2]] - point, axis=1)) < 2e-6
    # Mutating a callback's packet cannot mutate live model outputs.
    result['slot_output']['points'][:] = 100
    assert output['object_slot_points'].detach().abs().max() < 2
    assert result['slot_output']['role_token_valid'].dtype == np.bool_


def test_insufficient_support_never_exports_padding_as_world_center():
    network, output, cube, reverse, world = sample()
    packet = build_coarse_slot_packet(network, output, cube, reverse, world)
    raw = packet['slot_output']['points']
    padding = np.all(raw == 0, axis=-1)
    assert padding.any()
    assert not packet['point_valid'][padding].any()
    finite_nonzero = np.isfinite(raw).all(-1) & ~padding
    assert (finite_nonzero & ~packet['point_valid']).any()


def test_invalid_geometry_does_not_erase_semantic_token_validity():
    network, output, cube, reverse, world = sample()
    output['object_slot_valid'][0, 1] = False
    output['object_slot_points'] = network.object_slot_predictor1._extract_points(
        output['object_slot_prior'], output['mvt1_ori_img'][:, :, :3], output['object_slot_valid'])
    packet = build_coarse_slot_packet(network, output, cube, reverse, world)
    assert not packet['point_valid'][1].any()
    assert not packet['slot_output']['valid'][1]
    assert packet['slot_output']['role_token_valid'][1]
    assert not packet['slot_output']['reference_is_null']


def test_extraction_mismatch_and_invalid_coverage_fail_explicitly():
    network, output, cube, reverse, world = sample()
    output['object_slot_points'] = output['object_slot_points'].detach().clone() + .1
    with pytest.raises(ValueError, match='top-k'):
        build_coarse_slot_packet(network, output, cube, reverse, world)
    network, output, cube, reverse, world = sample()
    network.renderer.coverage[:] = 2.
    with pytest.raises(ValueError, match='coverage'):
        build_coarse_slot_packet(network, output, cube, reverse, world)


def test_wrong_inverse_or_non_isotropic_inverse_is_rejected():
    _, _, cube, reverse, world = sample()
    with pytest.raises(ValueError, match='restore'):
        cube_inverse_metadata(reverse[0], cube[0], world[0] + .1)
    with pytest.raises(ValueError, match='isotropic'):
        cube_inverse_metadata(lambda p: p * p.new_tensor([1., 2., 3.]), cube[0], world[0])


@pytest.mark.parametrize('failure', ['legacy_slots', 'training', 'batch', 'background'])
def test_unsupported_sources_fail_before_extra_render(failure):
    network, output, cube, reverse, world = sample()
    if failure == 'legacy_slots':
        network.object_slot_predictor_type = 'slots'
    elif failure == 'training':
        network.training = True
    elif failure == 'background':
        network.renderer.default_color = 1.
    else:
        cube = cube * 2
    with pytest.raises(ValueError):
        build_coarse_slot_packet(network, output, cube, reverse, world)
    assert network.renderer.calls == 0


def callback_block():
    source = (ROOT / 'finetune/bridgevla/models/bridgevla_agent.py').read_text(encoding='utf-8')
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'RVTAgent')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'act')
    defaults = dict(zip((a.arg for a in method.args.args[-len(method.args.defaults):]), method.args.defaults))
    assert ast.literal_eval(defaults['coarse_slot_callback']) is None
    block = next(n for n in method.body if isinstance(n, ast.If) and
                 any(isinstance(child, ast.ImportFrom) and child.module == 'bridgevla.models.coarse_slot_export'
                     for child in ast.walk(n)))
    return compile(ast.Module(body=[block], type_ignores=[]), '<actual_act_export_callback>', 'exec')


def test_actual_act_callback_control_is_opt_in_and_runs_once():
    network, output, cube, reverse, world = sample()
    scope = dict(self=SimpleNamespace(_net_mod=network), out=output, pc=cube,
                 rev_trans=reverse, coarse_scene_world=world, coarse_slot_callback=None)
    exec(callback_block(), scope)
    assert network.renderer.calls == 0
    collected = []
    scope['coarse_slot_callback'] = collected.append
    exec(callback_block(), scope)
    assert network.renderer.calls == 1 and len(collected) == 1
    assert collected[0]['slot_output']['points'].shape == (2, 5, 3)
