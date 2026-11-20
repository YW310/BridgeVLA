"""Original slots -> TASK prompts/masks -> original Anchor; software evidence only."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_coarse_slot_export import CoverageRenderer, sample
from test_online_slot_sam3_perception import scene, _run
from test_online_slot_sam3_policy import policy_fixture
from test_slot_guided_action_draft import (
    actual_anchor_block, packet, software_authorization, WrapperStub, source_method,
)
from bridgevla.config import get_cfg_defaults
from bridgevla.models.coarse_slot_export import build_coarse_slot_packet
from bridgevla.models.oracle_prior import InternalObjectSlotPredictor, OracleRelationAnchorFeatureAdapter
from bridgevla.models.slot_guided_geometry import adapter_geometry_kwargs
from bridgevla.online import slot_sam3_runtime as runtime

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / 'finetune/RLBench/configs/rlbench_slot_guided_sam3_internal_slots_online.yaml'


def test_original_sam3_profile_changes_only_runtime_not_training_or_model_architecture():
    baseline = get_cfg_defaults()
    baseline.merge_from_file(str(PROFILE.with_name('rlbench_o2_internal_slots.yaml')))
    candidate = runtime.load_online_profile(PROFILE)
    assert candidate.object_slots.predictor_type == 'slots'
    assert not candidate.object_conditioning.inherit_coarse_roles
    assert not candidate.object_conditioning.shared_action_features
    assert candidate.slot_guided_sam3.stage == 'action'
    assert runtime.resolve_runtime(candidate)['guidance_stage'] == 'coarse'
    candidate.exp_id = baseline.exp_id
    candidate.rvt.enable_slot_guided_sam3 = baseline.rvt.enable_slot_guided_sam3
    candidate.rvt.slot_guided_sam3_execution_mode = baseline.rvt.slot_guided_sam3_execution_mode
    candidate.slot_guided_sam3 = baseline.slot_guided_sam3.clone()
    assert candidate == baseline
    assert not baseline.rvt.enable_slot_guided_sam3
    with pytest.raises(ValueError, match='coarse guidance'):
        runtime.resolve_runtime(runtime.load_online_profile(PROFILE), {'guidance_stage': 'refine'})


@pytest.mark.parametrize('norm_corr', [False, True])
def test_original_normalized_correlations_are_inverted_only_in_detached_export(norm_corr):
    network, output, cube, reverse, world = sample()
    cube = [cube[0] * .6]
    original_reverse = reverse[0]
    reverse = [lambda p: original_reverse(p / .6)]
    network.mvt1 = SimpleNamespace(norm_corr=norm_corr)
    network.object_slot_predictor_type = 'slots'
    network.object_slot_predictor1 = InternalObjectSlotPredictor(
        feature_channels=4, num_views=1, num_slots=2, slot_dim=4,
        num_heads=2, decoder_layers=1, point_samples=5)
    scale = cube[0].abs().max() if norm_corr else 1.
    output['mvt1_ori_img'][:, :, :3] *= .6 / scale
    output['object_slot_points'] = network.object_slot_predictor1._extract_points(
        output['object_slot_prior'], output['mvt1_ori_img'][:, :, :3], output['object_slot_valid'])
    output['object_slot_geometry'] = None  # Actual original predictor contract.
    before = copy.deepcopy(output)
    weights = copy.deepcopy(network.object_slot_predictor1.state_dict())
    result = build_coarse_slot_packet(network, output, cube, reverse, world)
    assert result['predictor_type'] == 'slots'
    assert result['correlation_scale_to_cube'] == pytest.approx(float(scale))
    assert result['slot_output']['geometry'] is None
    valid = result['point_valid']
    np.testing.assert_allclose(result['slot_output']['points'][valid],
                               output['object_slot_points'].numpy()[0][valid] * float(scale))
    for points in result['slot_output']['points']:
        restored = np.c_[points, np.ones(len(points))] @ result['matrix_world_from_source'].T
        for p in restored[np.any(points != 0, axis=-1), :3]:
            assert np.linalg.norm(world[0].numpy() - p, axis=1).min() < 2e-6
    for name, value in before.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(output[name], value, rtol=0, atol=0)
    for name, value in weights.items():
        torch.testing.assert_close(network.object_slot_predictor1.state_dict()[name], value, rtol=0, atol=0)


def test_original_export_drives_real_task_mock_segmentation_and_multiview_fusion(scene):
    points = torch.tensor(scene['packet']['slot_output']['points'], dtype=torch.float32)
    center, cube_scale = torch.tensor([.4, -.2, 1.]), 2.
    cube = (points - center) / cube_scale
    corr_scale = cube.abs().max()
    count = points.shape[1]
    image = torch.zeros(1, 1, 6, 32, 32)
    image[0, 0, :3].reshape(3, -1)[:, :2 * count] = cube.reshape(-1, 3).T / corr_scale
    prior = torch.zeros(1, 1, 2, 32, 32)
    prior[0, 0, 0].flatten()[:count] = .9
    prior[0, 0, 1].flatten()[count:2 * count] = .9
    predictor = InternalObjectSlotPredictor(feature_channels=4, num_views=1, num_slots=2,
        slot_dim=4, num_heads=2, decoder_layers=1, point_samples=count)
    valid = torch.ones(1, 2, dtype=torch.bool)
    output = dict(object_slot_prior=prior, object_slot_points=predictor._extract_points(prior, image[:, :, :3], valid),
        object_slot_confidence=torch.full((1, 2), .9), object_slot_valid=valid,
        object_slot_role_token_valid=valid, object_slot_reference_null_probability=torch.tensor([.02]),
        object_slot_reference_is_null=torch.tensor([False]), object_slot_geometry=None, mvt1_ori_img=image)
    coverage = torch.zeros(1, 32, 32)
    coverage.flatten()[:2 * count] = 1.
    network = SimpleNamespace(training=False, mvt1=SimpleNamespace(norm_corr=True),
        object_slot_predictor_type='slots', object_slot_predictor1=predictor, renderer=CoverageRenderer(coverage))
    scene['packet'] = build_coarse_slot_packet(network, output, [cube.reshape(-1, 3)],
        [lambda p: p * cube_scale + center], [points.reshape(-1, 3)])
    result = _run(scene)
    assert result.backend_kind == 'mock'  # Never authorize this fixture for actions.
    assert result.geometry.object_role_valid.all()
    assert 'predictor=slots' in result.guidance.metadata['transform']['source']
    for name, backend in scene['backends'].items():
        assert backend.set_image_calls == 1
        for role in ('target', 'reference'):
            row = result.inference['cameras'][name]['roles'][role]
            assert row['prompt']['prompt_valid']
            assert not row['candidates'][0]['accepted']  # Wrong instance has higher SAM3 score.
            assert row['candidates'][1]['accepted']


def test_original_anchor_uses_masked_extent_not_stddev_and_preserves_old_weights(software_authorization):
    adapter = OracleRelationAnchorFeatureAdapter(4, rank=3, anchor_rank=3).eval()
    torch.nn.init.normal_(adapter.feature_expand.weight, std=.1)
    torch.nn.init.normal_(adapter.anchor_expand.weight, std=.1)
    weights = copy.deepcopy(adapter.state_dict())
    geom = packet()
    external = software_authorization(geom)
    kwargs = adapter_geometry_kwargs(external, device='cpu', dtype=torch.float32, role_conditioning=False)
    expected = adapter._anchor_geometry(torch.stack((geom.object_points[0, 0, :2],
        geom.object_points[0, 1, :1].expand(2, -1)))[None], geom.object_role_valid, torch.float32, 'cpu')
    torch.testing.assert_close(kwargs['geometry'], expected)
    inputs = dict(x=torch.rand(1, 4, 4, 4), oracle_prior_heatmap=torch.rand(1, 1, 2, 4, 4),
        oracle_prior_valid=geom.object_role_valid, oracle_relation_points=geom.object_points,
        oracle_relation_state=torch.zeros(1, 3), instruction_context=None, oracle_feature_adapter=adapter,
        oracle_adapter_translation_only=False, object_slot_output=dict(role_tokens=torch.rand(1, 2, 128),
        role_token_valid=geom.object_role_valid, reference_null_probability=torch.tensor([.2]), geometry=None),
        self=SimpleNamespace(training=False), torch=torch)
    prior = inputs['oracle_prior_heatmap'].clone()
    with torch.no_grad():
        actual = actual_anchor_block(inputs, external)
        reference = adapter.forward_with_anchor(inputs['x'], prior, **kwargs, current_state=inputs['oracle_relation_state'])
    for a, b in zip(actual, reference):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(prior, inputs['oracle_prior_heatmap'], rtol=0, atol=0)
    adapter.load_state_dict(weights, strict=True)
    for name, value in weights.items():
        torch.testing.assert_close(adapter.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize('status', ['NOT_VISIBLE', 'LOW_CONFIDENCE', 'NULL'])
def test_original_anchor_never_collapses_unseen_or_uncertain_reference_into_null(software_authorization, status):
    external = software_authorization(packet(status))
    if status == 'NULL':
        kwargs = adapter_geometry_kwargs(external, device='cpu', dtype=torch.float32, role_conditioning=False)
        assert kwargs['reference_is_null'].item() and not kwargs['instance_valid'][0, 1]
        assert torch.isfinite(kwargs['geometry']).all()
    else:
        with pytest.raises(ValueError, match='explicit NULL'):
            adapter_geometry_kwargs(external, device='cpu', dtype=torch.float32, role_conditioning=False)


def test_segmentation_mode_runs_sam3_but_keeps_exact_baseline_action_even_if_target_empty(tmp_path):
    policy, agent, perception, _ = policy_fixture(stage='segmentation', diagnostics_dir=tmp_path)
    perception.empty = True
    result = policy.act(5, {'current_index': 0})
    assert len(agent.calls) == 1 and len(perception.calls) == 1
    np.testing.assert_array_equal(result.action, agent.last_result.action)
    assert result.info['slot_guided_sam3']['status'] == 'segmentation_only'
    assert result.info['slot_guided_sam3']['sam3_executed']
    assert not result.replay_elements['slot_guided_sam3_used']
    assert not result.replay_elements['slot_guided_sam3_fallback']


def test_original_mvt_routes_fused_geometry_to_both_original_anchor_stages(software_authorization):
    from bridgevla.mvt import utils
    fn = source_method('finetune/bridgevla/mvt/mvt.py', 'MVT', 'forward', {'mvt_utils': utils})
    model = WrapperStub()
    model.oracle_prior_feature_adapter1 = OracleRelationAnchorFeatureAdapter(4, rank=3, anchor_rank=3).eval()
    model.oracle_prior_feature_adapter2 = OracleRelationAnchorFeatureAdapter(4, rank=3, anchor_rank=3).eval()
    weights = copy.deepcopy(model.state_dict())
    pc = [torch.tensor([[-.4, -.3, -.2], [.6, .7, .8]])]
    with torch.no_grad():
        original = fn(model, pc, [torch.ones(2, 3)])
        disabled = fn(model, pc, [torch.ones(2, 3)], slot_guided_geometry=None)
        fn(model, pc, [torch.ones(2, 3)], slot_guided_geometry=software_authorization(packet()))
    for stage in (original, original['mvt2']):
        torch.testing.assert_close(stage['trans'], (disabled if stage is original else disabled['mvt2'])['trans'], rtol=0, atol=0)
    assert all('slot_guided_geometry' not in call for call in model.mvt1.calls[:4])
    assert all('slot_guided_geometry' in call for call in model.mvt1.calls[-2:])
    model.load_state_dict(weights, strict=True)


def test_baseline_profile_cli_and_segmentation_stage_use_existing_launcher(monkeypatch, capsys):
    import json
    calls = []
    monkeypatch.setattr(runtime.runpy, 'run_path', lambda *a, **k: calls.append(list(runtime.sys.argv)))
    assert runtime.main(['--config', str(PROFILE), '--slot-sam3-stage', 'segmentation']) == 0
    assert calls[0][calls[0].index('--slot-sam3-stage') + 1] == 'segmentation'
    assert runtime.main(['--config', str(PROFILE), '--check-config']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['settings']['stage'] == 'action'
    assert report['settings']['guidance_stage'] == 'coarse'
