"""Offline action prediction contracts, with no acceptance or robot evidence."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_slot_guided_action_draft import (
    WrapperStub, packet, source_method,
)
from bridgevla.models.slot_guided_geometry import (
    AuthorizedGeometry, OfflineGeometry, authorize_geometry,
    authorize_offline_geometry, sha256_file,
)
from bridgevla.mvt import utils as mvt_utils


@pytest.fixture
def offline_inputs(tmp_path):
    checkpoint = tmp_path / 'checkpoint.pth'
    # Hash contract fixture; this is never loaded as a real robot model.
    checkpoint.write_bytes(b'offline geometry unit-test checkpoint')
    geom = packet()
    provenance = {**geom.provenance[0], 'source_checkpoint_sha256': sha256_file(checkpoint)}
    geom = replace(geom, provenance=(provenance,))
    current = {**provenance, 'max_source_frame_index_read': 17}
    return geom, [current], checkpoint, 'e' * 40


@pytest.mark.parametrize('source', ['oracle_gt', 'predicted_coarse_slot'])
def test_offline_scope_allows_real_sam3_a_and_b_without_an_acceptance_report(offline_inputs, source):
    geom, current, checkpoint, git_sha = offline_inputs
    meta = {**geom.provenance[0], 'guidance_source': source}
    if source == 'oracle_gt':
        meta['source_checkpoint_sha256'] = None
    geom = replace(geom, provenance=(meta,))
    authorized = authorize_offline_geometry(geom, current, checkpoint, git_sha)
    assert isinstance(authorized, OfflineGeometry)
    assert authorized.scope == 'offline_experiment'
    assert authorized.sources['bridge_checkpoint_sha256'] == sha256_file(checkpoint)
    assert not authorized.geometry.object_points.requires_grad
    assert authorized.verify() is authorized
    assert not hasattr(authorized, 'report')
    with pytest.raises(ValueError, match='REAL'):
        AuthorizedGeometry(geom, {}, authorized.sources).verify()
    with pytest.raises(ValueError, match='scope'):
        replace(authorized, scope='reviewed').verify()


@pytest.mark.parametrize('field,value', [
    ('source_frame_index', 18), ('frame_id', '18'), ('episode_id', 'another'),
    ('calibration_sha256', 'f' * 64), ('observed_at', '2026-11-06T00:00:00Z'),
    ('max_source_frame_index_read', 18), ('max_source_frame_index_read', True),
])
def test_offline_scope_rejects_stale_future_and_mismatched_receipts(offline_inputs, field, value):
    geom, current, checkpoint, git_sha = offline_inputs
    current[0][field] = value
    with pytest.raises(ValueError, match='provenance|read receipt'):
        authorize_offline_geometry(geom, current, checkpoint, git_sha)


@pytest.mark.parametrize('field,value', [
    ('backend_kind', 'mock'), ('guidance_source', 'mock'), ('mock', True),
    ('coordinate_frame', 'normalized_cube'), ('coordinate_units', 'mm'),
    ('sam3_checkpoint_sha256', None), ('source_git_sha', 'main'),
    ('source_checkpoint_sha256', 'f' * 64),
    ('source_checkpoint_sha256', None),
])
def test_offline_scope_rejects_mock_bad_coordinates_and_unmatched_model_sources(offline_inputs, field, value):
    geom, current, checkpoint, git_sha = offline_inputs
    geom = replace(geom, provenance=({**geom.provenance[0], field: value},))
    with pytest.raises(ValueError):
        authorize_offline_geometry(geom, current, checkpoint, git_sha)


def test_offline_scope_revalidates_receipt_and_model_sources_after_transform(offline_inputs):
    authorized = authorize_offline_geometry(*offline_inputs)
    moved = authorized.transformed(lambda p: p * 2.)
    assert moved.verify() is moved
    current = ({**moved.current_frames[0], 'max_source_frame_index_read': 19},)
    with pytest.raises(ValueError, match='future'):
        replace(moved, current_frames=current).verify()
    with pytest.raises(ValueError, match='source/checkpoint'):
        replace(moved, sources={**moved.sources, 'sam3_checkpoint_sha256': 'f' * 64}).verify()


def test_offline_scope_never_bypasses_reviewed_authorization(offline_inputs):
    geom, current, checkpoint, git_sha = offline_inputs
    with pytest.raises(ValueError, match='REAL'):
        authorize_geometry(geom, current, {}, checkpoint, git_sha)


def test_agent_routes_only_explicit_matching_offline_mode_and_freezes_the_model(offline_inputs):
    geom, current, checkpoint, git_sha = offline_inputs
    fn = source_method('finetune/bridgevla/models/bridgevla_agent.py', 'RVTAgent',
                       '_slot_guided_geometry_inputs', {'mvt_utils': mvt_utils})
    net = torch.nn.Linear(3, 3).train()
    agent = SimpleNamespace(enable_slot_guided_sam3=True, _training=False,
                            slot_guided_sam3_execution_mode='offline_experiment',
                            _network=net, _place_with_mean=False,
                            scene_bounds=[-.3, -.5, .6, .7, .5, 1.6])
    payload = dict(execution_mode='offline_experiment', geometry=geom,
                   bridge_checkpoint_path=checkpoint, bridge_git_sha=git_sha)
    scene = [torch.tensor([[-.2, -.4, .65], [.6, .4, 1.55]])]
    inputs = fn(agent, payload, {'slot_guided_current_frames': current}, scene, False)
    authorized = inputs['slot_guided_geometry']
    assert isinstance(authorized, OfflineGeometry)
    assert not net.training and not any(p.requires_grad for p in net.parameters())
    expected = mvt_utils.place_pc_in_cube(scene[0], app_pc=geom.object_points[0].reshape(-1, 3),
                                        with_mean_or_bounds=False, scene_bounds=agent.scene_bounds)[0]
    mask = geom.object_point_valid[0]
    torch.testing.assert_close(authorized.geometry.object_points[0][mask], expected.reshape(2, 8, 3)[mask])
    for candidate in ({**payload, 'execution_mode': 'reviewed'},
                      {k: v for k, v in payload.items() if k != 'execution_mode'}):
        with pytest.raises(ValueError, match='execution_mode'):
            fn(agent, candidate, {'slot_guided_current_frames': current}, scene, False)
    with pytest.raises(ValueError, match='acceptance reports'):
        fn(agent, {**payload, 'acceptance_report': {}}, {}, scene, False)
    with pytest.raises(ValueError, match='frozen'):
        fn(agent, payload, {}, scene, True)
    agent.slot_guided_sam3_execution_mode = 'reviewed'
    with pytest.raises(ValueError, match='execution_mode'):
        fn(agent, payload, {}, scene, False)


def test_actual_mvt_accepts_offline_scope_under_existing_no_grad_eval_guards(offline_inputs):
    fn = source_method('finetune/bridgevla/mvt/mvt.py', 'MVT', 'forward',
                       {'mvt_utils': mvt_utils})
    model = WrapperStub()
    external = authorize_offline_geometry(*offline_inputs)
    scene = [torch.tensor([[-.4, -.3, -.2], [.6, .7, .8]])]
    features = [torch.ones(2, 3)]
    with pytest.raises(ValueError, match='eval'):
        fn(model, scene, features, slot_guided_geometry=external)
    with torch.no_grad():
        fn(model, scene, features, slot_guided_geometry=external)
    assert len(model.mvt1.calls) == 2
    assert all(isinstance(call['slot_guided_geometry'], OfflineGeometry) for call in model.mvt1.calls)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    model.train()
    with torch.no_grad(), pytest.raises(ValueError, match='eval'):
        fn(model, scene, features, slot_guided_geometry=external)
    model.eval()
    with torch.no_grad(), pytest.raises(ValueError, match='authorized'):
        fn(model, scene, features, slot_guided_geometry=external.geometry)
