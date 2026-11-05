"""PR4 software contracts only. No real acceptance report or performance claim."""
import ast
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'finetune'))
from bridgevla.models.slot_guided_geometry import (
    AuthorizedGeometry, REQUIRED_METRICS, adapter_geometry_kwargs, authorize_geometry,
    freeze_for_geometry_inference, geometry_for_render, geometry_to_cube,
    load_pr3_geometry, sample_geometry, sha256_file, validate_real_acceptance,
)
from bridgevla.models.oracle_prior import OracleRelationAnchorFeatureAdapter
from bridgevla.mvt.utils import place_pc_in_cube


def method(path, cls, name):
    model = next(n for n in ast.parse((ROOT / path).read_text(encoding='utf-8')).body
                 if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == name)


def source_method(path, cls, name, scope=None):
    node = method(path, cls, name)
    node.decorator_list = []
    scope = {'torch': torch, **(scope or {})}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope[name]


def packet(reference='VISIBLE', confidence=.9):
    empty = torch.empty(0, 3)
    return sample_geometry(
        [[torch.tensor([[.2, .3, .8], [.4, .3, .8]]),
          torch.tensor([[.6, .3, .8]]) if reference == 'VISIBLE' else empty]],
        [('VISIBLE', reference)], [[confidence, confidence]], [.2],
        [dict(episode_id='episode0', frame_id='17', source_frame_index=17,
              observed_at='2026-11-05T00:00:00+00:00', calibration_sha256='a' * 64,
              coordinate_frame='rlbench_world', coordinate_units='m',
              guidance_source='predicted_coarse_slot', backend_kind='sam3',
              source_git_sha='b' * 40, source_checkpoint_sha256='c' * 64,
              sam3_checkpoint_sha256='d' * 64)], count=8)


@pytest.fixture
def software_authorization(monkeypatch):
    # Explicit test seam only. Production verify rejects this SOFTWARE_TEST report.
    monkeypatch.setattr(AuthorizedGeometry, 'verify', lambda self: self)
    return lambda geometry: AuthorizedGeometry(geometry, {'evidence_kind': 'SOFTWARE_TEST'}, {})


def test_fixed_sampler_padding_is_not_geometry_and_no_grad():
    geom = packet()
    assert geom.object_points.shape == (1, 2, 8, 3)
    assert geom.object_point_valid.sum().item() == 3
    torch.testing.assert_close(geom.descriptor()[0, :3], torch.tensor([.3, .3, .8]))
    assert torch.count_nonzero(geom.object_points[~geom.object_point_valid]) == 0
    external = replace(geom, object_points=geom.object_points.requires_grad_(),
                       reference_null_probability=geom.reference_null_probability.requires_grad_())
    detached = external.detached()
    assert not detached.object_points.requires_grad
    assert not detached.descriptor().requires_grad
    assert not detached.reference_null_probability.requires_grad


@pytest.mark.parametrize('status', ['NULL', 'NOT_VISIBLE', 'LOW_CONFIDENCE'])
@pytest.mark.parametrize('posterior', [0., .95, 1.])
def test_null_unseen_uncertain_preserve_distinct_status_and_posterior(status, posterior):
    geom = replace(packet(status), reference_null_probability=torch.tensor([posterior])).validate()
    assert geom.statuses[0][1] == status
    assert not geom.object_point_valid[0, 1].any()
    assert not geom.object_role_valid[0, 1]
    assert geom.reference_null_probability.item() == pytest.approx(posterior)


def test_semantically_visible_with_failed_geometry_is_allowed():
    geom = sample_geometry([[np.empty((0, 3)), np.empty((0, 3))]],
                           [('VISIBLE', 'VISIBLE')], [[0., 0.]], [1.], [{}])
    assert not geom.object_role_valid.any()
    assert geom.descriptor().count_nonzero() == 0


def test_null_points_and_nonfinite_valid_points_rejected():
    with pytest.raises(ValueError, match='unavailable roles'):
        sample_geometry([[np.ones((1, 3)), np.ones((1, 3))]],
                        [('VISIBLE', 'NULL')], [[1., 1.]], [1.], [{}])
    bad = packet()
    bad.object_points[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match='finite'):
        bad.validate()


def test_software_or_mock_reports_never_authorize_real_activation():
    sources = dict(bridge_git_sha='b' * 40, bridge_checkpoint_sha256='c' * 64,
                   guidance_git_sha='d' * 40, guidance_checkpoint_sha256='e' * 64,
                   sam3_checkpoint_sha256='f' * 64)
    base = dict(evidence_kind='REAL', verified=True, backend_kind='sam3', mock=False,
                accepted=True, reviewed_by='reviewer', reviewed_at='2026-11-05T00:00:00Z',
                sources=sources, calibration_sha256=['a' * 64],
                metrics={key: 1. for key in REQUIRED_METRICS})
    for alteration in ({'evidence_kind': 'SOFTWARE_TEST'}, {'mock': True},
                       {'backend_kind': 'mock'}, {'verified': False}, {'accepted': False},
                       {'reviewed_by': ''}, {'metrics': {}}):
        with pytest.raises(ValueError):
            validate_real_acceptance({**base, **alteration}, sources, ['a' * 64])
    with pytest.raises(ValueError, match='source/checkpoint'):
        validate_real_acceptance(base, {'bridge_checkpoint_sha256': 'e' * 64}, ['a' * 64])
    with pytest.raises(ValueError, match='calibration'):
        validate_real_acceptance(base, sources, ['e' * 64])
    with pytest.raises(ValueError, match='REAL'):
        AuthorizedGeometry(packet(), {'evidence_kind': 'SOFTWARE_TEST'}, {}).verify()


def test_stale_future_or_wrong_frame_rejected_before_acceptance(tmp_path):
    geom = packet()
    checkpoint = tmp_path / 'weights.pt'
    checkpoint.write_bytes(b'not real weights; never an accepted report')
    current = {**geom.provenance[0], 'max_source_frame_index_read': 17}
    for key, value in [('source_frame_index', 18), ('frame_id', '18'),
                       ('episode_id', 'another'), ('calibration_sha256', 'e' * 64),
                       ('max_source_frame_index_read', 18)]:
        with pytest.raises(ValueError, match='provenance|future'):
            authorize_geometry(geom, [{**current, key: value}], {}, checkpoint, 'f' * 40)
    with pytest.raises(ValueError, match='REAL'):
        authorize_geometry(geom, [current], {'evidence_kind': 'SOFTWARE_TEST'}, checkpoint, 'f' * 40)


def test_loader_digest_shape_role_and_meter_contract(tmp_path):
    geom = packet('NULL')
    archive = tmp_path / 'geometry.npz'
    np.savez(archive, target_points_world=np.array([[.1, .2, .3]]),
             reference_points_world=np.empty((0, 3)))
    meta = {**geom.provenance[0], 'schema_version': 1,
            'artifact_type': 'slot_object_geometry', 'geometry_path': archive.name,
            'geometry_sha256': sha256_file(archive), 'reference_null_probability': .91,
            'roles': {'target': dict(status='VISIBLE', semantic_valid=True, geometry_confidence=.8,
                                     points_key='target_points_world', point_count=1),
                      'reference': dict(status='NULL', semantic_valid=False, geometry_confidence=0.,
                                        points_key='reference_points_world', point_count=0)}}
    path = tmp_path / 'geometry.json'
    path.write_text(json.dumps(meta))
    loaded = load_pr3_geometry(path)
    assert loaded.object_point_valid.sum() == 1
    assert loaded.statuses == (('VISIBLE', 'NULL'),)
    meta['geometry_sha256'] = 'f' * 64
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='digest'):
        load_pr3_geometry(path)


def loader_fixture(tmp_path, **arrays):
    archive = tmp_path / 'geometry.npz'
    np.savez(archive, **({'target_points_world': np.array([[.1, .2, .3]]),
                        'reference_points_world': np.empty((0, 3)), **arrays}))
    meta = {**packet('NULL').provenance[0], 'schema_version': 1,
            'artifact_type': 'slot_object_geometry', 'geometry_path': archive.name,
            'geometry_sha256': sha256_file(archive), 'reference_null_probability': .91,
            'roles': {'target': dict(status='VISIBLE', semantic_valid=True, geometry_confidence=.8,
                                     points_key='target_points_world', point_count=1),
                      'reference': dict(status='NULL', semantic_valid=False, geometry_confidence=0.,
                                        points_key='reference_points_world', point_count=0)}}
    path = tmp_path / 'geometry.json'
    path.write_text(json.dumps(meta))
    return path, archive, meta


@pytest.mark.parametrize('version', [True, 1., '1', None])
def test_loader_requires_integer_schema_version(tmp_path, version):
    path, _, meta = loader_fixture(tmp_path)
    meta['schema_version'] = version
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='schema'):
        load_pr3_geometry(path)


def test_loader_duplicate_json_keys_rejected(tmp_path):
    path, _, _ = loader_fixture(tmp_path)
    path.write_text(path.read_text()[:-1] + ',"schema_version":1}')
    with pytest.raises(ValueError, match='duplicate JSON'):
        load_pr3_geometry(path)


@pytest.mark.parametrize('dtype', [bool, complex, object])
def test_loader_rejects_boolean_complex_object_coordinates(tmp_path, dtype):
    path, _, _ = loader_fixture(tmp_path, target_points_world=np.ones((1, 3), dtype=dtype))
    with pytest.raises(ValueError):
        load_pr3_geometry(path)


@pytest.mark.parametrize('dtype', [torch.bool, torch.complex64])
def test_tensor_sampler_rejects_non_real_coordinates(dtype):
    with pytest.raises(ValueError, match='real numeric'):
        sample_geometry([[torch.ones(1, 3, dtype=dtype), torch.empty(0, 3)]],
                        [('VISIBLE', 'NULL')], [[1., 0.]], [1.], [{}])


def test_loader_requires_npz_and_exact_canonical_keys(tmp_path):
    path, archive, meta = loader_fixture(tmp_path)
    with archive.open('wb') as stream:
        np.save(stream, np.ones((1, 3)))
    meta['geometry_sha256'] = sha256_file(archive)
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='NPZ'):
        load_pr3_geometry(path)
    np.savez(archive, target_points_world=np.ones((1, 3)),
             reference_points_world=np.empty((0, 3)), unrelated=np.ones(1))
    meta['geometry_sha256'] = sha256_file(archive)
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='canonical'):
        load_pr3_geometry(path)
    np.savez(archive, target_points_world=np.ones((1, 3)), reference_points_world=np.empty((0, 3)))
    meta['geometry_sha256'] = sha256_file(archive)
    meta['roles']['target']['points_key'] = 'reference_points_world'
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='canonical role'):
        load_pr3_geometry(path)


def test_loader_hashes_and_loads_identical_bytes(tmp_path, monkeypatch):
    path, archive, _ = loader_fixture(tmp_path)
    original_read = Path.read_bytes

    def read_then_replace(source):
        content = original_read(source)
        if source == archive:
            np.savez(archive, target_points_world=np.full((1, 3), .9), reference_points_world=np.empty((0, 3)))
        return content

    monkeypatch.setattr(Path, 'read_bytes', read_then_replace)
    loaded = load_pr3_geometry(path)
    torch.testing.assert_close(loaded.object_points[0, 0, 0], torch.tensor([.1, .2, .3]))


def test_git_sha256_repositories_supported_by_report_contract():
    sources = {'bridge_git_sha': 'b' * 64, 'guidance_git_sha': 'c' * 40,
               'bridge_checkpoint_sha256': 'd' * 64, 'guidance_checkpoint_sha256': 'e' * 64,
               'sam3_checkpoint_sha256': 'f' * 64}
    report = dict(evidence_kind='REAL', verified=True, backend_kind='sam3', mock=False,
                  accepted=True, reviewed_by='unit-test-reviewer', reviewed_at='2026-11-05T00:00:00Z',
                  sources=sources, calibration_sha256=['a' * 64],
                  metrics={key: 1. for key in REQUIRED_METRICS})
    # Validator shape coverage, not activation: no geometry or network is passed.
    validate_real_acceptance(report, sources, ['a' * 64])


def test_actual_scene_cube_crop_and_norm_corr_math(software_authorization):
    geom = packet()
    scene = [torch.tensor([[-.2, -.4, .65], [.6, .4, 1.55]])]
    authorized = software_authorization(geom)
    for with_mean in (False, True):
        bounds = [-.3, -.5, .6, .7, .5, 1.6]
        cube = geometry_to_cube(authorized, scene, place_pc_in_cube, with_mean, bounds)
        expected, reverse = place_pc_in_cube(scene[0], app_pc=geom.object_points[0].reshape(-1, 3),
                                            with_mean_or_bounds=with_mean,
                                            scene_bounds=None if with_mean else bounds)
        valid = geom.object_point_valid[0]
        torch.testing.assert_close(cube.geometry.object_points[0][valid], expected.reshape(2, 8, 3)[valid])
        torch.testing.assert_close(reverse(cube.geometry.object_points[0][valid]), geom.object_points[0][valid])
        center, scale = torch.tensor([[.1, -.2, .3]]), 4.
        crop = cube.transformed(lambda p: scale * (p - center[:, None, None]))
        restored = crop.geometry.object_points / scale + center[:, None, None]
        torch.testing.assert_close(restored[geom.object_point_valid], cube.geometry.object_points[geom.object_point_valid])
        normalized = geometry_for_render(crop, [scene[0] * 3], norm_corr=True, clean_xyz=False)
        torch.testing.assert_close(normalized.geometry.object_points,
                                   crop.geometry.object_points / (scene[0] * 3).abs().max())
        assert geometry_for_render(crop, scene, norm_corr=True, clean_xyz=True) is crop
        assert not cube.geometry.object_points[~geom.object_point_valid].any()


def anchor_inputs():
    torch.manual_seed(82)
    adapter = OracleRelationAnchorFeatureAdapter(4, rank=3, anchor_rank=3,
                                                role_conditioning=True, role_token_dim=8).eval()
    # Exercise geometry-sensitive behavior, not only zero-init residual identity.
    torch.nn.init.normal_(adapter.feature_expand.weight, std=.1)
    torch.nn.init.normal_(adapter.anchor_expand.weight, std=.1)
    inputs = dict(x=torch.rand(1, 4, 4, 4), oracle_prior_heatmap=torch.rand(1, 1, 2, 4, 4),
                  oracle_prior_valid=torch.ones(1, 2, dtype=torch.bool),
                  oracle_relation_points=torch.rand(1, 2, 8, 3), oracle_relation_state=torch.zeros(1, 3),
                  instruction_context=None, oracle_feature_adapter=adapter,
                  oracle_adapter_translation_only=False,
                  object_slot_output=dict(role_tokens=torch.rand(1, 2, 8),
                                          role_token_valid=torch.ones(1, 2, dtype=torch.bool),
                                          reference_null_probability=torch.tensor([.1]),
                                          geometry=packet().descriptor()), self=SimpleNamespace(training=False),
                  torch=torch)
    return adapter, inputs


def actual_anchor_block(inputs, external):
    node = method('finetune/bridgevla/mvt/mvt_single.py', 'MVT', 'forward')
    blocks = [n for n in node.body if isinstance(n, ast.If) and ast.unparse(n.test) in
              ('slot_guided_geometry is not None', 'oracle_feature_adapter is not None')]
    scope = {**inputs, 'slot_guided_geometry': external}
    exec(compile(ast.Module(body=blocks, type_ignores=[]), '<actual_anchor_route>', 'exec'), scope)
    return scope['translation_features'], scope['x'], scope['relation_anchor']


def test_actual_anchor_disabled_route_exact_equality_and_strict_state_dict():
    adapter, inputs = anchor_inputs()
    keys = tuple(adapter.state_dict())
    before_prior = inputs['oracle_prior_heatmap'].clone()
    slot = inputs['object_slot_output']
    with torch.no_grad():
        original = adapter.forward_with_anchor(
            inputs['x'], inputs['oracle_prior_heatmap'], inputs['oracle_prior_valid'],
            inputs['oracle_relation_points'], current_state=inputs['oracle_relation_state'],
            role_tokens=slot['role_tokens'], reference_null_probability=slot['reference_null_probability'],
            geometry=slot['geometry'], role_token_valid=slot['role_token_valid'])
        routed = actual_anchor_block(inputs, None)
    for old, new in zip(original, routed):
        assert torch.equal(old, new)
    assert torch.equal(inputs['oracle_prior_heatmap'], before_prior)
    assert tuple(adapter.state_dict()) == keys
    blob = io.BytesIO()
    torch.save(adapter.state_dict(), blob)
    blob.seek(0)
    clone, _ = anchor_inputs()
    clone.load_state_dict(torch.load(blob, weights_only=True), strict=True)
    assert tuple(clone.state_dict()) == keys


def test_actual_anchor_uses_separate_external_geometry_without_prior_mutation(software_authorization):
    adapter, inputs = anchor_inputs()
    prior = inputs['oracle_prior_heatmap'].clone()
    slot = inputs['object_slot_output']
    saved_tokens = slot['role_tokens'].clone()
    geom = packet('NOT_VISIBLE')
    external = software_authorization(geom)
    with torch.no_grad():
        actual = actual_anchor_block(inputs, external)
        kwargs = adapter_geometry_kwargs(external, device=inputs['x'].device, dtype=inputs['x'].dtype)
        expected = adapter.forward_with_anchor(inputs['x'], inputs['oracle_prior_heatmap'], **kwargs,
            current_state=inputs['oracle_relation_state'], role_tokens=slot['role_tokens'],
            role_token_valid=slot['role_token_valid'])
    for a, b in zip(actual, expected):
        assert torch.equal(a, b)
    assert torch.equal(prior, inputs['oracle_prior_heatmap'])
    assert torch.equal(saved_tokens, slot['role_tokens'])
    assert geom.statuses[0][1] == 'NOT_VISIBLE'


def test_explicit_null_and_unseen_have_distinct_action_tokens_at_same_posterior(software_authorization):
    adapter, inputs = anchor_inputs()
    with torch.no_grad():
        adapter.unknown_reference.fill_(.7)
        adapter.null_reference.fill_(-.4)
        adapter.role_token_projection.weight.zero_()
        adapter.role_token_projection.bias.zero_()
    action_queries = []
    hook = adapter.anchor_query_encoder[0].register_forward_pre_hook(
        lambda _, args: action_queries.append(args[0].detach().clone()))
    prior_before = inputs['oracle_prior_heatmap'].clone()
    try:
        with torch.no_grad():
            null = software_authorization(packet('NULL'))
            unseen = software_authorization(packet('NOT_VISIBLE'))
            null_out = actual_anchor_block(inputs, null)
            unseen_out = actual_anchor_block(inputs, unseen)
    finally:
        hook.remove()
    assert null.geometry.reference_null_probability.item() == unseen.geometry.reference_null_probability.item()
    null_token, unseen_token = [q[0, 0, adapter.rank:2 * adapter.rank] for q in action_queries]
    torch.testing.assert_close(null_token, torch.full_like(null_token, -.4))
    assert not torch.equal(null_token, unseen_token)
    assert not torch.equal(null_out[0], unseen_out[0])
    assert torch.equal(prior_before, inputs['oracle_prior_heatmap'])
    kwargs = adapter_geometry_kwargs(unseen, device='cpu', dtype=torch.float32)
    assert not kwargs['reference_is_null'].any()
    certain = replace(unseen.geometry, reference_null_probability=torch.ones(1)).validate()
    kwargs = adapter_geometry_kwargs(software_authorization(certain), device='cpu', dtype=torch.float32)
    assert kwargs['reference_null_probability'].item() == 1.
    assert not kwargs['reference_is_null'].any()


def test_disabled_agent_boundary_ignores_payload_without_freezing_or_parsing():
    fn = source_method('finetune/bridgevla/models/bridgevla_agent.py', 'RVTAgent', '_slot_guided_geometry_inputs')
    net = torch.nn.Linear(3, 3).train()
    agent = SimpleNamespace(enable_slot_guided_sam3=False, _network=net)
    original = net.state_dict()
    assert fn(agent, object(), object(), object(), True) == {}
    assert net.training and all(p.requires_grad for p in net.parameters())
    for key, value in original.items():
        assert torch.equal(value, net.state_dict()[key])


def test_frozen_path_and_train_guards():
    net = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(.8))
    freeze_for_geometry_inference(net)
    assert not net.training and not any(p.requires_grad for p in net.parameters())
    assert not net[1].training
    fn = source_method('finetune/bridgevla/models/bridgevla_agent.py', 'RVTAgent', 'train')
    with pytest.raises(ValueError, match='frozen'):
        fn(SimpleNamespace(enable_slot_guided_sam3=True, _network=net))
    fn = source_method('finetune/bridgevla/models/bridgevla_agent.py', 'RVTAgent', 'update')
    with pytest.raises(ValueError, match='expert action'):
        fn(SimpleNamespace(enable_slot_guided_sam3=True), {})


def test_low_geometry_confidence_disables_geometry(software_authorization):
    kwargs = adapter_geometry_kwargs(software_authorization(packet(confidence=.1)),
                                    device='cpu', dtype=torch.float32)
    assert not kwargs['instance_valid'].any()
    assert kwargs['geometry'].count_nonzero() == 0


class StageStub(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.7))
        self.norm_corr = True
        self.calls = []

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        return {'trans': kwargs['img'].sum(2) * self.weight}


class WrapperStub(torch.nn.Module):
    def __init__(self, second_stage=True):
        super().__init__()
        self.mvt1 = StageStub()
        self.oracle_prior_feature_adapter1, _ = anchor_inputs()
        self.oracle_prior_feature_adapter2, _ = anchor_inputs()
        self.img_aug_2 = 0
        self.object_slots_enabled = True
        self.object_slot_predictor_type = 'slots'
        self.object_slot_predictor1 = self.object_slot_predictor2 = None
        self.object_conditioning_inherit_coarse_roles = False
        self.object_conditioning_shared_action_features = False
        self.object_conditioning_use_context = False
        self.object_conditioning_refine_mask_detach = False
        self.oracle_adapter_translation_only = False
        self.oracle_relation_gated_adapter = True
        self.refine_role_mask_head = None
        self.stage_two = second_stage
        self.st_sca = 4.
        self.eval()

    def verify_inp(self, **kwargs):
        pass

    def render(self, **kwargs):
        return torch.ones(1, 1, 6, 4, 4)

    def _build_oracle_instance_prior(self, *args):
        return None

    def get_wpt(self, *args, **kwargs):
        return torch.tensor([[.1, .2, .3]])


def test_actual_mvt_forward_routes_both_stages_and_disabled_kwargs(software_authorization):
    from bridgevla.mvt import utils
    fn = source_method('finetune/bridgevla/mvt/mvt.py', 'MVT', 'forward',
                       {'mvt_utils': utils})
    model = WrapperStub()
    pc = [torch.tensor([[-.4, -.3, -.2], [.6, .7, .8]])]
    features = [torch.ones(2, 3)]
    state = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        baseline = fn(model, pc, features)
        explicit_disabled = fn(model, pc, features, slot_guided_geometry=None)
    assert torch.equal(baseline['trans'], explicit_disabled['trans'])
    assert torch.equal(baseline['mvt2']['trans'], explicit_disabled['mvt2']['trans'])
    assert all('slot_guided_geometry' not in call for call in model.mvt1.calls)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in state.items())
    assert all(p.requires_grad for p in model.parameters())
    model.mvt1.calls.clear()
    external = software_authorization(packet())
    with torch.no_grad():
        fn(model, pc, features, slot_guided_geometry=external)
    first, second = [call['slot_guided_geometry'].geometry for call in model.mvt1.calls]
    mask = external.geometry.object_point_valid
    torch.testing.assert_close(first.object_points[mask],
        external.geometry.object_points[mask] / pc[0].abs().max())
    crop = 4. * (pc[0] - torch.tensor([.1, .2, .3]))
    expected = 4. * (external.geometry.object_points - torch.tensor([.1, .2, .3])) / crop.abs().max()
    torch.testing.assert_close(second.object_points[mask], expected[mask])
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert tuple(model.state_dict()) == tuple(state)
    model.load_state_dict(state, strict=True)


def test_actual_mvt_rejects_training_grad_and_unreviewed_input():
    fn = source_method('finetune/bridgevla/mvt/mvt.py', 'MVT', 'forward')
    model = WrapperStub(False)
    geom = AuthorizedGeometry(packet(), {'evidence_kind': 'SOFTWARE_TEST'}, {})
    with pytest.raises(ValueError, match='eval'):
        fn(model, [], [], slot_guided_geometry=geom)
    with torch.no_grad(), pytest.raises(ValueError, match='REAL'):
        fn(model, [], [], slot_guided_geometry=geom)


@pytest.mark.parametrize('stage', [1, 2])
def test_enabled_route_rejects_anchors_without_existing_role_conditioning(software_authorization, stage):
    fn = source_method('finetune/bridgevla/mvt/mvt.py', 'MVT', 'forward')
    model = WrapperStub()
    getattr(model, f'oracle_prior_feature_adapter{stage}').role_conditioning = False
    with torch.no_grad(), pytest.raises(ValueError, match='role_conditioning=True'):
        fn(model, [], [], slot_guided_geometry=software_authorization(packet()))


def test_actual_pr3_writer_artifact_interoperability():
    supplied = os.environ.get('SLOT_GUIDED_PR3_ARTIFACT')
    if not supplied:
        pytest.skip('set SLOT_GUIDED_PR3_ARTIFACT to a real PR3-writer synthetic output')
    artifact = Path(supplied)
    meta = json.loads(artifact.read_text(encoding='utf-8'))
    geom = load_pr3_geometry(artifact, count=128)
    assert geom.object_points.shape == (1, 2, 128, 3)
    assert geom.object_point_valid.all() and geom.object_role_valid.all()
    assert geom.statuses == (('VISIBLE', 'VISIBLE'),)
    assert geom.provenance[0]['calibration_sha256'] == meta['calibration_sha256']
    assert geom.provenance[0]['source_frame_index'] == 7
    assert meta['backend_kind'] == 'mock'
    # Consume the artifact but never authorize mock-produced geometry for action.
    current = {**meta, 'max_source_frame_index_read': meta['source_frame_index']}
    with pytest.raises(ValueError, match='real predicted coarse-slot SAM3'):
        authorize_geometry(geom, [current], {}, artifact, 'f' * 40)
