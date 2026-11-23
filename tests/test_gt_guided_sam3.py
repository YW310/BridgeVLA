"""Slot-free Oracle SAM3 software contracts; not real robot performance evidence."""
import copy
from dataclasses import replace
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_slot_guided_action_draft import packet, actual_anchor_block, source_method, WrapperStub
from test_online_slot_sam3_perception import scene
from test_online_slot_sam3_policy import receipt
from test_slot_sam3_observation import current, task_repo, prepped, isolated_env, BaseEnv
from bridgevla.config import get_cfg_defaults
from bridgevla.models.oracle_prior import OracleRelationGatedFeatureAdapter, route_oracle_adapter_features
from bridgevla.models.slot_guided_geometry import (
    OracleOnlineGeometry, authorize_oracle_online_geometry, authorize_online_geometry,
    oracle_sam3_prior_points, sha256_file,
)
from bridgevla.online.slot_sam3_perception import OnlineTaskPerception
from bridgevla.online.slot_sam3_policy import OnlineSlotSam3Policy
from bridgevla.online import slot_sam3_runtime as runtime
from bridgevla.utils.startup_progress import startup_phase
from bridgevla.utils import checkpoint_identity
from bridgevla.offline.slot_sam3_checkpoint import load_profile, load_saved_configs, model_arguments
from yarr.agents.agent import ActResult

ROOT = Path(__file__).resolve().parents[1]
ONLINE = ROOT / 'finetune/RLBench/configs/rlbench_gt_guided_sam3_online.yaml'
OFFLINE = ONLINE.with_name('rlbench_gt_guided_sam3_offline.yaml')


def authorized(tmp_path, reference='VISIBLE'):
    checkpoint = tmp_path / 'weights.pth'
    checkpoint.write_bytes(b'software-only checkpoint provenance')
    geom = packet(reference)
    meta = {**geom.provenance[0], 'guidance_source': 'oracle_gt', 'source_checkpoint_sha256': None,
        'snapshot_input_sha256': 'b' * 64, 'execution_mode': 'oracle_online_experiment',
        'max_source_frame_index_read': 17}
    geom = replace(geom, provenance=(meta,))
    return authorize_oracle_online_geometry(geom, [dict(meta)], checkpoint, 'b' * 40), checkpoint


def test_profiles_reuse_trained_semantic_gt_relation_model_without_slots_or_anchor(tmp_path):
    original = get_cfg_defaults()
    original.merge_from_file(str(ONLINE.with_name('rlbench_o2_semantic_gt.yaml')))
    online, offline = runtime.load_online_profile(ONLINE), load_profile(OFFLINE)
    assert online.oracle_relation_anchor_rank == 0
    assert not online.object_slots.enabled
    assert runtime.resolve_runtime(online)['guidance_source'] == 'oracle_gt'
    for cfg in (online, offline):
        compare = cfg.clone()
        compare.exp_id = original.exp_id
        compare.rvt.enable_slot_guided_sam3 = original.rvt.enable_slot_guided_sam3
        compare.rvt.slot_guided_sam3_execution_mode = original.rvt.slot_guided_sam3_execution_mode
        compare.slot_guided_sam3 = original.slot_guided_sam3.clone()
        assert compare == original
    (tmp_path / 'exp_cfg.yaml').write_bytes(ONLINE.with_name('rlbench_o2_semantic_gt.yaml').read_bytes())
    (tmp_path / 'mvt_cfg.yaml').write_bytes((ROOT / 'finetune/bridgevla/mvt/configs/rvt2.yaml').read_bytes())
    cfg, mvt, _, _ = load_saved_configs(tmp_path / 'weights.pth', offline)
    assert not model_arguments(cfg, mvt)['object_slots_enabled']
    (tmp_path / 'exp_cfg.yaml').write_bytes(ONLINE.with_name('rlbench_o2_internal_slots.yaml').read_bytes())
    with pytest.raises(ValueError, match='object_slots.enabled'):
        load_saved_configs(tmp_path / 'weights.pth', offline)
    with pytest.raises(ValueError, match='matching experiment config'):
        runtime.resolve_runtime(online, {'guidance_source': 'predicted_coarse_slot'})


@pytest.mark.parametrize('failure', ['future', 'stale_rgb', 'predicted', 'mock', 'wrong_mode', 'slot_checkpoint'])
def test_oracle_authorization_rejects_stale_mock_or_mislabeled_inputs(tmp_path, failure):
    good, checkpoint = authorized(tmp_path)
    geom = good.geometry
    meta = dict(geom.provenance[0])
    key, value = {
        'future': ('max_source_frame_index_read', 18),
        'stale_rgb': ('snapshot_input_sha256', 'f' * 64),
        'predicted': ('guidance_source', 'predicted_coarse_slot'),
        'mock': ('backend_kind', 'mock'),
        'wrong_mode': ('execution_mode', 'online_experiment'),
        'slot_checkpoint': ('source_checkpoint_sha256', 'e' * 64),
    }[failure]
    meta[key] = value
    with pytest.raises(ValueError):
        authorize_oracle_online_geometry(replace(geom, provenance=(meta,)), good.current_frames, checkpoint, 'b' * 40)
    with pytest.raises(ValueError, match='predicted coarse-slot'):
        authorize_online_geometry(geom, good.current_frames, checkpoint, 'b' * 40)


@pytest.mark.parametrize('reference', ['VISIBLE', 'NULL', 'NOT_VISIBLE', 'LOW_CONFIDENCE'])
def test_prior_sampling_never_projects_zero_padding_or_collapses_role_states(tmp_path, reference):
    auth, _ = authorized(tmp_path, reference)
    points, valid = oracle_sam3_prior_points(auth)
    for role in range(2):
        if valid[0, role]:
            cloud = auth.geometry.object_points[0, role, auth.geometry.object_point_valid[0, role]]
            assert torch.cdist(points[0, role], cloud).amin(-1).max() < 1e-6
    assert auth.geometry.statuses[0][1] == reference
    assert valid[0, 1].item() == (reference == 'VISIBLE')
    assert not points.requires_grad


def oracle_labels(scene):
    return {**{key: scene['snapshot'].metadata[key] for key in ('episode_id', 'frame_id', 'source_frame_index', 'observed_at')},
        'reference_is_null': False,
        'oracle_role_masks': {name: {} for name in scene['snapshot'].cameras}}


def test_gt_masks_drive_real_task_sam3_prompts_and_mock_fusion_without_slot_packet(scene):
    # Recover fixture masks directly from registered depth/world point identity.
    api = scene['api']
    labels = oracle_labels(scene)
    for name, frame in scene['snapshot'].cameras.items():
        for role, points in zip(('target', 'reference'), scene['packet']['slot_output']['points']):
            xy, positive = api.camera_geometry.project_points(points, frame.camera.intrinsics, frame.camera.extrinsics)
            mask = np.zeros(frame.depth_m.shape, dtype=bool)
            pixels = np.rint(xy[positive]).astype(int)
            mask[pixels[:, 1], pixels[:, 0]] = True
            labels['oracle_role_masks'][name][role] = mask
    client = OnlineTaskPerception(scene['repo'], 'configs/slot_guided_offline.json',
                                  backend=scene['backends'], guidance_source='oracle_gt')
    result = client.perceive(scene['snapshot'], None, instruction='move left block to right block',
        task_name='memory', checkpoint_path=scene['checkpoint'], bridge_git_sha='a' * 40, oracle_labels=labels)
    assert result.backend_kind == 'mock'  # This can never authorize an action.
    assert result.geometry.object_role_valid.all()
    assert result.guidance.metadata['guidance_source'] == 'oracle_gt'
    assert result.geometry.provenance[0]['source_checkpoint_sha256'] is None
    assert result.geometry.provenance[0]['execution_mode'] == 'oracle_online_experiment'
    for backend in scene['backends'].values():
        assert backend.set_image_calls == 1 and len({call['state_id'] for call in backend.predict_calls}) == 1
    client.reset()
    labels['source_frame_index'] += 1
    with pytest.raises(ValueError, match='current RGB-D frame'):
        client.perceive(scene['snapshot'], None, instruction='move block', task_name='memory',
            checkpoint_path=scene['checkpoint'], bridge_git_sha='a' * 40, oracle_labels=labels)


def test_actual_relation_adapter_receives_only_sam3_surface_points_and_loads_old_weights(tmp_path):
    auth, _ = authorized(tmp_path)
    adapter = OracleRelationGatedFeatureAdapter(4, rank=3).eval()
    torch.nn.init.normal_(adapter.feature_expand.weight, std=.1)
    weights = copy.deepcopy(adapter.state_dict())
    x, prior = torch.rand(1, 4, 4, 4), torch.rand(1, 1, 2, 4, 4)
    inputs = dict(x=x, oracle_prior_heatmap=prior, oracle_prior_valid=torch.ones(1, 2, dtype=torch.bool),
        oracle_relation_points=torch.full((1, 2, 8, 3), 999.), oracle_relation_state=torch.zeros(1, 3),
        instruction_context=None, oracle_feature_adapter=adapter, oracle_adapter_translation_only=False,
        object_slot_output=None, relation_anchor=None, self=SimpleNamespace(training=False), torch=torch,
        route_oracle_adapter_features=route_oracle_adapter_features)
    with torch.no_grad():
        actual = actual_anchor_block(inputs, auth)
        p, v = oracle_sam3_prior_points(auth)
        expected = adapter(x, prior, v, p)
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
    assert actual[2] is None
    clone = OracleRelationGatedFeatureAdapter(4, rank=3)
    clone.load_state_dict(weights, strict=True)
    for key, value in weights.items():
        torch.testing.assert_close(clone.state_dict()[key], value, rtol=0, atol=0)


def test_actual_mvt_generates_both_stage_priors_from_sam3_and_never_calls_slots(tmp_path):
    from bridgevla.mvt import utils
    auth, _ = authorized(tmp_path)
    fn = source_method('finetune/bridgevla/mvt/mvt.py', 'MVT', 'forward', {'mvt_utils': utils})
    model = WrapperStub()
    model.object_slots_enabled = False
    model.oracle_prior_feature_adapter1 = OracleRelationGatedFeatureAdapter(4, rank=3).eval()
    model.oracle_prior_feature_adapter2 = OracleRelationGatedFeatureAdapter(4, rank=3).eval()
    model.object_slot_predictor1 = lambda *a, **kw: pytest.fail('Slot prediction must not run')
    model.object_slot_predictor2 = model.object_slot_predictor1
    projected = []
    def prior(points, valid, stage, out, sigma):
        projected.append((points.clone(), valid.clone(), stage))
        return valid[:, None, :, None, None].expand(-1, 1, -1, 4, 4).float()
    model._build_oracle_instance_prior = prior
    model._attach_oracle_instance_prior = lambda *args: None
    crop_center = torch.tensor([[.1, -.2, .3]])
    model.get_wpt = lambda *args, **kwargs: crop_center
    with torch.no_grad():
        result = fn(model, [torch.tensor([[-.4, -.3, -.2], [.6, .7, .8]])], [torch.ones(2, 3)], slot_guided_geometry=auth)
    assert len(projected) == 2
    for call in model.mvt1.calls:
        assert call['object_slot_predictor'] is None
        assert call.get('inherited_object_roles') is None
        assert call['oracle_prior_heatmap'] is not None
    torch.testing.assert_close(model.mvt1.calls[0]['slot_guided_geometry'].geometry.object_points, auth.geometry.object_points)
    fine = model.mvt1.calls[1]['slot_guided_geometry'].geometry
    expected = model.st_sca * (auth.geometry.object_points - crop_center[:, None, None])
    torch.testing.assert_close(fine.object_points[fine.object_point_valid], expected[fine.object_point_valid])
    with torch.no_grad(), pytest.raises(ValueError, match='raw Oracle points'):
        fn(model, [], [], slot_guided_geometry=auth, oracle_prior_points=torch.ones(1, 2, 8, 3))


def test_current_gt_masks_stay_outside_extracted_policy_observation():
    events = []
    class Base(BaseEnv):
        def extract_obs(self, obs, *a, **kw):
            assert obs.front_mask is None
            return self.output
    provider = SimpleNamespace(requires_oracle_task=True, cameras=['front'],
        capture_raw=lambda obs, extracted, **kw: events.append((obs.front_mask.copy(), extracted, kw)),
        invalidate=lambda error: pytest.fail(str(error)))
    env = isolated_env(Base)(slot_sam3_observation_provider=provider)
    obs = SimpleNamespace(front_mask=np.ones((4, 4), dtype=int))
    extracted = env.extract_obs(obs)
    assert extracted is env.output and not any(key.startswith('oracle_') or key.endswith('_mask') for key in extracted)
    assert events[0][2]['task_environment'] is env._task
    assert obs.front_mask is not None


def test_oracle_capture_labels_bind_current_frame_and_are_readonly(current):
    provider, clock, raw, extracted = current
    provider.requires_oracle_task = True
    resets = []
    provider._prompt_oracle = SimpleNamespace(
        reset=lambda *args: resets.append(args),
        sam3_prompt_annotations=lambda obs: dict(reference_is_null=True,
            oracle_role_masks={name: {'target': np.ones(getattr(obs, f'{name}_rgb').shape[:2], dtype=bool)} for name in provider.cameras}))
    provider.capture_raw(raw, extracted, task_name='open_drawer', instruction='move the red object', task_environment=object())
    captured = provider.capture(prepped(extracted))
    assert len(resets) == 1
    assert captured.oracle_labels['source_frame_index'] == captured.receipt['source_frame_index']
    assert not captured.oracle_labels['oracle_role_masks']['front']['target'].flags.writeable


@pytest.mark.parametrize('stage,failure', [('action', None), ('segmentation', None),
                                        ('action', 'empty'), ('action', 'mock'), ('action', 'exception')])
def test_oracle_policy_uses_only_sam3_conditioning_or_current_unconditioned_fallback(tmp_path, stage, failure):
    auth, checkpoint = authorized(tmp_path, 'NULL')
    calls, perception_calls = [], []
    current_receipt = dict(auth.current_frames[0])
    labels = {'oracle_role_masks': object()}  # Must never appear in agent.act().
    observation = {'low_dim_state': torch.ones(1, 1, 4)}

    def act(step, observed, deterministic=False, **kwargs):
        assert torch.is_inference_mode_enabled()
        assert 'coarse_slot_callback' not in kwargs
        assert not any(k.startswith('oracle_') or k.endswith('_mask') for k in observed)
        calls.append((agent.enable_slot_guided_sam3, observed, kwargs))
        if agent.enable_slot_guided_sam3:
            assert kwargs['slot_guided_sam3']['execution_mode'] == 'oracle_online_experiment'
            assert all(current_receipt[key] == value
                       for key, value in observed['slot_guided_current_frames'][0].items())
            assert not kwargs['slot_guided_sam3']['geometry'].object_points.requires_grad
        return ActResult(np.array([.2 if agent.enable_slot_guided_sam3 else .1, 0, 0, 0, 0, 0, 1, 1, 0]))

    agent = SimpleNamespace(_training=False, enable_slot_guided_sam3=True,
        slot_guided_sam3_execution_mode='oracle_online_experiment',
        _net_mod=SimpleNamespace(object_slots_enabled=False), act=act)
    capture = SimpleNamespace(snapshot=object(), receipt=current_receipt, oracle_labels=labels,
        task_name='open_drawer', instruction='open drawer')

    def perceive(snapshot, packet, **kwargs):
        perception_calls.append(kwargs)
        assert snapshot is capture.snapshot and packet is None and kwargs['oracle_labels'] is labels
        if failure == 'exception':
            raise RuntimeError('empty SAM3 prediction')
        geom = auth.geometry
        if failure == 'empty':
            geom = replace(geom, object_points=torch.zeros_like(geom.object_points),
                object_point_valid=torch.zeros_like(geom.object_point_valid),
                object_role_valid=torch.zeros_like(geom.object_role_valid), statuses=(('NOT_VISIBLE', 'NULL'),))
        return SimpleNamespace(geometry=geom, receipt=current_receipt,
            backend_kind='mock' if failure == 'mock' else 'sam3', timings_ms={})

    policy = OnlineSlotSam3Policy(agent, SimpleNamespace(perceive=perceive),
        SimpleNamespace(capture=lambda obs: capture), checkpoint, 'b' * 40,
        enabled=True, stage=stage, guidance_source='oracle_gt', diagnostics_dir=tmp_path / 'diagnostics')
    result = policy.act(0, observation)
    conditioned = failure is None and stage == 'action'
    assert result.action[0] == (.2 if conditioned else .1), policy.last_diagnostics
    assert [call[0] for call in calls] == ([False, True] if conditioned else [False])
    assert policy.last_diagnostics['slot_prediction_used'] is False
    assert policy.last_diagnostics['oracle_conditioned'] is True
    assert policy.last_diagnostics['status'] == ('conditioned' if conditioned else (
        'baseline_fallback' if failure else 'segmentation_only'))
    assert agent.enable_slot_guided_sam3 is True
    with pytest.raises(ValueError, match='side channel'):
        policy.act(1, {**observation, 'front_mask': np.ones((3, 3))})
    assert len(perception_calls) == 1


@pytest.mark.parametrize('reference', ['VISIBLE', 'NULL', 'NOT_VISIBLE', 'SITE'])
def test_live_prompt_annotations_keep_instance_identity_and_absence_semantics(reference):
    from test_o2_semantic_roles import RLBenchGTOracleProvider, decode_handle_mask
    provider = object.__new__(RLBenchGTOracleProvider)
    provider.handle_alignment, provider.raw_data_root = 'identity', None
    provider.cameras, provider.role_config_sha256 = ['front', 'wrist'], 'c' * 64
    # The other instance of the same class (handle 30) must never enter Target.
    target = dict(kind='object', handles=[10, 11])
    ref = None if reference == 'NULL' else dict(kind='site' if reference == 'SITE' else 'object', handles=[20])
    provider._entries = [dict(target=target, reference=ref, phase_id=2)]
    enrich_calls = []
    provider.enrich = lambda obs, extracted: enrich_calls.append((obs, extracted))
    mask = np.array([[10, 11, 30], [0, 20 if reference == 'VISIBLE' else 0, 30]])
    obs = SimpleNamespace(front_mask=mask, wrist_mask=mask.copy())
    if reference == 'SITE':
        with pytest.raises(ValueError, match='nonvisual site'):
            provider.sam3_prompt_annotations(obs)
        return
    result = provider.sam3_prompt_annotations(obs)
    assert enrich_calls == [(obs, {})]
    assert result['reference_is_null'] == (reference == 'NULL')
    for roles in result['oracle_role_masks'].values():
        np.testing.assert_array_equal(roles['target'], np.isin(decode_handle_mask(mask), [10, 11]))
        if reference == 'NULL':
            assert 'reference' not in roles
        else:
            assert bool(roles['reference'].any()) == (reference == 'VISIBLE')
    provider.raw_data_root = Path('stored_demo_must_not_be_read')
    with pytest.raises(ValueError, match='no stored episode reads'):
        provider.sam3_prompt_annotations(obs)
    assert len(enrich_calls) == 1


def test_eval_preserves_oracle_sam3_mode_before_generic_gt_prior_disable(capsys):
    tree = ast.parse((ROOT / 'finetune/RLBench/eval.py').read_text(encoding='utf-8'))
    fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_eval')
    branch = next(node for node in fn.body if isinstance(node, ast.For))
    selector = next(node for node in branch.body if isinstance(node, ast.If)
        and ast.unparse(node.test) == "args.oracle_provider == 'rlbench_gt'")
    agent = SimpleNamespace(enable_slot_guided_sam3=True,
        slot_guided_sam3_execution_mode='oracle_online_experiment', oracle_prior_enabled=True,
        oracle_prior_mode='o2_gt_instance')
    exec(compile(ast.Module(body=[selector], type_ignores=[]), '<actual_eval_branch>', 'exec'),
         dict(agent=agent, args=SimpleNamespace(oracle_provider='none')))
    assert agent.oracle_prior_mode == 'o2_gt_instance'
    assert 'no Slot prediction' in capsys.readouterr().out


def test_real_agent_authorizes_oracle_scope_and_uses_original_world_to_cube_transform(tmp_path, monkeypatch):
    from bridgevla.mvt import utils
    auth, checkpoint = authorized(tmp_path)
    fn = source_method('finetune/bridgevla/models/bridgevla_agent.py', 'RVTAgent',
        '_slot_guided_geometry_inputs', {'mvt_utils': utils})
    net = torch.nn.Linear(3, 3).train()
    agent = SimpleNamespace(enable_slot_guided_sam3=True, _training=False,
        slot_guided_sam3_execution_mode='oracle_online_experiment', _network=net,
        _place_with_mean=False, scene_bounds=[-.3, -.5, .6, .7, .5, 1.6])
    agent.slot_guided_sam3_checkpoint_identity = checkpoint_identity.LoadedCheckpointIdentity(checkpoint)

    def forbidden(*args):
        raise AssertionError('Oracle conditioned policy must use its loaded checkpoint identity')

    monkeypatch.setattr(checkpoint_identity, 'sha256_file', forbidden)
    payload = dict(execution_mode='oracle_online_experiment', geometry=auth.geometry,
        bridge_checkpoint_path=checkpoint, bridge_git_sha='b' * 40)
    observation = dict(slot_guided_current_frames=auth.current_frames)
    scene = [torch.tensor([[-.2, -.3, .7], [.6, .3, 1.5]])]
    result = fn(agent, payload, observation, scene, False)['slot_guided_geometry']
    assert isinstance(result, OracleOnlineGeometry)
    expected = utils.place_pc_in_cube(scene[0], app_pc=auth.geometry.object_points.flatten(0, 2),
        with_mean_or_bounds=False, scene_bounds=agent.scene_bounds)[0].reshape_as(auth.geometry.object_points)
    mask = result.geometry.object_point_valid
    torch.testing.assert_close(result.geometry.object_points[mask], expected[mask])
    assert not net.training and not any(p.requires_grad for p in net.parameters())
    with pytest.raises(ValueError, match='frozen inference'):
        fn(agent, payload, observation, scene, True)


def test_eval_automatically_enables_current_gt_capture_with_default_role_yaml(monkeypatch, tmp_path):
    from test_online_slot_sam3_runtime import eval_tree
    events = []
    monkeypatch.setitem(sys.modules, 'utils.slot_sam3_observation', SimpleNamespace(
        SlotSam3ObservationProvider=lambda *a, **kw: events.append(kw) or object()))
    fn = eval_tree()
    block = next(node for node in fn.body if isinstance(node, ast.If) and any(
        isinstance(item, ast.ImportFrom) and item.module == 'utils.slot_sam3_observation' for item in node.body))
    code = compile(ast.Module(body=[block], type_ignores=[]), '<actual_oracle_observer_setup>', 'exec')
    scope = dict(online_slot_sam3=True, agent=SimpleNamespace(cameras=['front'],
        slot_guided_sam3_runtime=dict(guidance_source='oracle_gt', task_repo='TASK')),
        oracle_role_config=None, online_env_kwargs={}, Path=Path, startup_phase=startup_phase,
        __file__=str(ROOT / 'finetune/RLBench/eval.py'))
    exec(code, scope)
    assert events[0]['oracle_role_config'] == ROOT / 'finetune/RLBench/configs/rlbench_o2_semantic_roles.yaml'
    scope['oracle_role_config'] = tmp_path / 'missing.yaml'
    with pytest.raises(FileNotFoundError, match='semantic role config'):
        exec(code, scope)
    assert len(events) == 1
