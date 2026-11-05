"""Online action geometry requires fresh RGB-D and the actual loaded policy."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_slot_guided_action_draft import WrapperStub, packet, source_method
from bridgevla.models.slot_guided_geometry import (
    AuthorizedGeometry, OnlineGeometry, authorize_online_geometry, sha256_file,
)
from bridgevla.mvt import utils as mvt_utils


@pytest.fixture
def current_geometry(tmp_path):
    checkpoint = tmp_path / 'model.pth'
    checkpoint.write_bytes(b'online unit-test provenance; no actual model inference')
    geometry = packet()
    meta = {**geometry.provenance[0], 'source_checkpoint_sha256': sha256_file(checkpoint),
            'source_git_sha': 'b' * 40, 'snapshot_input_sha256': 'e' * 64,
            'max_source_frame_index_read': 17, 'execution_mode': 'online_experiment'}
    geometry = replace(geometry, provenance=(meta,))
    receipt = {key: meta[key] for key in ('episode_id', 'frame_id', 'source_frame_index',
                'observed_at', 'calibration_sha256', 'snapshot_input_sha256', 'max_source_frame_index_read')}
    return geometry, [receipt], checkpoint, 'b' * 40


def test_current_online_geometry_is_detached_and_keeps_explicit_scope(current_geometry):
    geometry, receipt, checkpoint, sha = current_geometry
    geometry = replace(geometry, object_points=geometry.object_points.requires_grad_())
    authorized = authorize_online_geometry(geometry, receipt, checkpoint, sha)
    assert isinstance(authorized, OnlineGeometry)
    assert not authorized.geometry.object_points.requires_grad
    assert authorized.verify() is authorized
    assert authorized.transformed(lambda p: p * 2).verify().scope == 'online_experiment'
    with pytest.raises(ValueError, match='REAL'):
        AuthorizedGeometry(geometry, {}, authorized.sources).verify()


@pytest.mark.parametrize('key,value', [
    ('source_frame_index', 18), ('snapshot_input_sha256', 'f' * 64),
    ('max_source_frame_index_read', 18), ('calibration_sha256', 'f' * 64),
])
def test_same_camera_or_step_number_cannot_reuse_an_old_rgbd_frame(current_geometry, key, value):
    geometry, receipt, checkpoint, sha = current_geometry
    with pytest.raises(ValueError):
        authorize_online_geometry(geometry, [{**receipt[0], key: value}], checkpoint, sha)


@pytest.mark.parametrize('updates', [
    {'guidance_source': 'oracle_gt'}, {'backend_kind': 'mock'},
    {'execution_mode': 'offline_experiment'}, {'mock': True},
    {'source_git_sha': 'c' * 40}, {'max_source_frame_index_read': 18},
])
def test_online_action_rejects_privileged_mock_offline_or_future_sources(current_geometry, updates):
    geometry, receipt, checkpoint, sha = current_geometry
    geometry = replace(geometry, provenance=({**geometry.provenance[0], **updates},))
    with pytest.raises(ValueError):
        authorize_online_geometry(geometry, receipt, checkpoint, sha)


def test_real_agent_boundary_routes_online_and_current_scene_transform(current_geometry):
    geometry, receipt, checkpoint, sha = current_geometry
    fn = source_method('finetune/bridgevla/models/bridgevla_agent.py', 'RVTAgent',
                       '_slot_guided_geometry_inputs', {'mvt_utils': mvt_utils})
    network = torch.nn.Linear(3, 3).train()
    agent = SimpleNamespace(enable_slot_guided_sam3=True, _training=False,
                            slot_guided_sam3_execution_mode='online_experiment',
                            _network=network, _place_with_mean=False,
                            scene_bounds=[-.3, -.5, .6, .7, .5, 1.6])
    payload = dict(execution_mode='online_experiment', geometry=geometry,
                   bridge_checkpoint_path=checkpoint, bridge_git_sha=sha)
    scene = [torch.tensor([[-.2, -.3, .7], [.6, .3, 1.5]])]
    result = fn(agent, payload, {'slot_guided_current_frames': receipt}, scene, False)['slot_guided_geometry']
    expected = mvt_utils.place_pc_in_cube(scene[0], app_pc=geometry.object_points.flatten(0, 2),
        with_mean_or_bounds=False, scene_bounds=agent.scene_bounds)[0].reshape_as(geometry.object_points)
    torch.testing.assert_close(result.geometry.object_points[geometry.object_point_valid],
                               expected[geometry.object_point_valid])
    assert not network.training and not any(p.requires_grad for p in network.parameters())
    with pytest.raises(ValueError, match='acceptance'):
        fn(agent, {**payload, 'acceptance_report': {}}, {}, scene, False)


def test_actual_two_stage_mvt_accepts_online_scope_with_no_grad(current_geometry):
    external = authorize_online_geometry(*current_geometry)
    fn = source_method('finetune/bridgevla/mvt/mvt.py', 'MVT', 'forward', {'mvt_utils': mvt_utils})
    model = WrapperStub()
    scene = [torch.tensor([[-.4, -.3, -.2], [.6, .7, .8]])]
    features = [torch.ones(2, 3)]
    with pytest.raises(ValueError, match='eval'):
        fn(model, scene, features, slot_guided_geometry=external)
    with torch.no_grad():
        fn(model, scene, features, slot_guided_geometry=external)
    assert len(model.mvt1.calls) == 2
    assert all(isinstance(call['slot_guided_geometry'], OnlineGeometry) for call in model.mvt1.calls)
