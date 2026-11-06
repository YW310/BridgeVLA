"""Real loader/config contracts and controlled offline orchestrator seams."""
import ast
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
from bridgevla.config import get_cfg_defaults
from bridgevla.models.slot_guided_geometry import authorize_offline_geometry, sha256_file
from bridgevla.offline.slot_sam3_checkpoint import (
    CONDITIONING_FLAGS, load_profile, load_saved_configs, model_arguments,
    validate_checkpoint_metadata,
)
from bridgevla.offline.slot_sam3_runner import (
    guidance_export, load_current_state, load_oracle_labels, main, run_single_frame,
)

PROFILE = ROOT / 'finetune/RLBench/configs/rlbench_slot_guided_sam3_offline.yaml'
IDENTITY = dict(episode_id='episode0', frame_id='sample-2', source_frame_index=17,
                observed_at='2026-11-05T00:00:00+00:00')


def config_fixture(tmp_path):
    checkpoint = tmp_path / 'model.pt'
    checkpoint.write_bytes(b'software fixture, not model performance evidence')
    original = ROOT / 'finetune/RLBench/configs/rlbench_o2_role_queries.yaml'
    (tmp_path / 'exp_cfg.yaml').write_bytes(original.read_bytes())
    (tmp_path / 'mvt_cfg.yaml').write_bytes((ROOT / 'finetune/bridgevla/mvt/configs/rvt2.yaml').read_bytes())
    return checkpoint


def test_formal_profile_is_enabled_and_global_and_legacy_defaults_stay_off():
    assert load_profile(PROFILE).rvt.slot_guided_sam3_execution_mode == 'offline_experiment'
    assert not get_cfg_defaults().rvt.enable_slot_guided_sam3
    legacy = get_cfg_defaults()
    legacy.merge_from_file(str(PROFILE.with_name('rlbench_slot_guided_sam3_draft.yaml')))
    assert not legacy.rvt.enable_slot_guided_sam3
    with pytest.raises(ValueError, match='legacy draft'):
        load_profile(PROFILE.with_name('rlbench_slot_guided_sam3_draft.yaml'))


def test_saved_architecture_and_actual_eval_model_kwargs_match(tmp_path):
    checkpoint = config_fixture(tmp_path)
    cfg, mvt, _, _ = load_saved_configs(checkpoint, load_profile(PROFILE))
    actual = model_arguments(cfg, mvt)
    tree = ast.parse((ROOT / 'finetune/RLBench/eval.py').read_text(encoding='utf-8'))
    constructor = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                       and isinstance(n.func, ast.Name) and n.func.id == 'MVT')
    for kw in constructor.keywords:
        if kw.arg and 'exp_cfg' in ast.unparse(kw.value):
            expected = eval(compile(ast.Expression(kw.value), '<existing_eval_kw>', 'eval'), {'exp_cfg': cfg})
            assert actual[kw.arg] == expected, kw.arg
    assert cfg.rvt.enable_slot_guided_sam3 and cfg.rvt.slot_guided_sam3_execution_mode == 'offline_experiment'
    # The actual saved file was not rewritten by the runtime selection.
    assert 'enable_slot_guided_sam3: True' not in (tmp_path / 'exp_cfg.yaml').read_text()


def test_changed_checkpoint_architecture_and_missing_routing_metadata_fail(tmp_path):
    checkpoint = config_fixture(tmp_path)
    cfg, _, _, _ = load_saved_configs(checkpoint, load_profile(PROFILE))
    with pytest.raises(ValueError, match='predictor metadata'):
        validate_checkpoint_metadata({'object_predictor_type': 'slots'}, cfg)
    metadata = dict(object_predictor_type='role_queries', object_conditioning={
        name: bool(getattr(cfg.object_conditioning, name)) for name in CONDITIONING_FLAGS})
    validate_checkpoint_metadata(metadata, cfg)
    metadata['object_conditioning']['preserve_role_tokens'] = False
    with pytest.raises(ValueError, match='preserve_role_tokens'):
        validate_checkpoint_metadata(metadata, cfg)
    path = tmp_path / 'exp_cfg.yaml'
    path.write_text(path.read_text().replace('predictor_type: role_queries', 'predictor_type: slots'))
    with pytest.raises(ValueError, match='predictor_type'):
        load_saved_configs(checkpoint, load_profile(PROFILE))


def write_state(tmp_path, **updates):
    path = tmp_path / 'state.json'
    state = dict(schema_version=1, artifact_type='slot_policy_state', **IDENTITY,
                 low_dim_state=[1., .02, .02, .4], instruction='pick the cup', task_name='pick_cup')
    state.update(updates)
    path.write_text(json.dumps(state))
    return path


@pytest.mark.parametrize('updates', [dict(source_frame_index=18), dict(frame_id='17'),
                                    dict(low_dim_state=[True, .1, .1, .4]),
                                    dict(future_action=[0., 0., 0.]), dict(instruction='')])
def test_policy_state_rejects_future_or_unbound_inputs(tmp_path, updates):
    with pytest.raises(ValueError):
        load_current_state(write_state(tmp_path, **updates), SimpleNamespace(metadata=IDENTITY))


def test_oracle_labels_require_current_source_index_and_boolean_masks(tmp_path):
    snapshot = SimpleNamespace(metadata=IDENTITY,
                               cameras={'wrist': SimpleNamespace(depth_m=np.ones((3, 4)))})
    np.save(tmp_path / 'target.npy', np.ones((3, 4), dtype=bool))
    labels = dict(**IDENTITY, reference_is_null=True, role_masks={'wrist': {'target': 'target.npy'}})
    path = tmp_path / 'oracle.json'
    path.write_text(json.dumps(labels))
    assert load_oracle_labels(path, snapshot)['reference_is_null'] is True
    labels['source_frame_index'] = 18
    path.write_text(json.dumps(labels))
    with pytest.raises(ValueError, match='source_frame_index'):
        load_oracle_labels(path, snapshot)


class PerceptionSeam:
    """Unit-test artifact producer; not accessible from the command-line backend."""
    mode = 'point+box'
    def __init__(self, checkpoint, *, empty=False, kind='sam3', confidence=.9):
        self.checkpoint, self.empty, self.kind, self.confidence = checkpoint, empty, kind, confidence
    def load_snapshot(self, frame):
        return SimpleNamespace(metadata=IDENTITY)
    def snapshot_receipt(self, snapshot):
        return dict(**IDENTITY, calibration_sha256='a' * 64, snapshot_input_sha256='b' * 64,
                    max_source_frame_index_read=17)
    def snapshot_to_observation(self, snapshot, state, instruction, cameras, device):
        return {'opaque_current_observation': True}
    def perceive(self, frame, export, instruction, task_name, checkpoint, git_sha, output, **kwargs):
        assert kwargs['expected_snapshot_digest'] == 'b' * 64
        assert export['transform']['source_frame_index'] == 17
        assert export['transform']['timestamp'] == IDENTITY['observed_at']
        output.mkdir(parents=True)
        cloud = np.empty((0, 3)) if self.empty else np.array([[.2, .3, .8]])
        archive = output / 'geometry.npz'
        np.savez(archive, target_points_world=cloud, reference_points_world=np.empty((0, 3)))
        meta = dict(**IDENTITY, schema_version=1, artifact_type='slot_object_geometry',
                    geometry_path=archive.name, geometry_sha256=sha256_file(archive),
                    coordinate_frame='rlbench_world', coordinate_units='m',
                    calibration_sha256='a' * 64, guidance_source='predicted_coarse_slot',
                    backend_kind=self.kind, source_git_sha=git_sha,
                    source_checkpoint_sha256=sha256_file(self.checkpoint), sam3_checkpoint_sha256='d' * 64,
                    reference_null_probability=.9,
                    roles={'target': dict(status='VISIBLE', semantic_valid=True, geometry_confidence=self.confidence,
                                          points_key='target_points_world', point_count=len(cloud)),
                           'reference': dict(status='NULL', semantic_valid=False, geometry_confidence=0.,
                                             points_key='reference_points_world', point_count=0)})
        path = output / 'geometry.json'
        path.write_text(json.dumps(meta))
        return {'backend_kind': self.kind, 'object_geometry_path': str(path)}


class AgentSeam:
    cameras = ['wrist']
    enable_slot_guided_sam3 = True
    def __init__(self):
        self.calls = []
    def act(self, step, observation, deterministic, coarse_slot_callback=None, slot_guided_sam3=None):
        self.calls.append(self.enable_slot_guided_sam3)
        assert torch.is_inference_mode_enabled()
        if coarse_slot_callback is not None:
            coarse_slot_callback(dict(source_frame='bridgevla_coarse', matrix_world_from_source=np.eye(4),
                                      point_support_source='unit test controlled coverage seam',
                                      point_valid=np.ones((2, 1), dtype=bool),
                                      slot_output={'points': np.ones((2, 1, 3))}))
        if self.enable_slot_guided_sam3:
            assert slot_guided_sam3['execution_mode'] == 'offline_experiment'
            assert 'acceptance_report' not in slot_guided_sam3
            authorize_offline_geometry(slot_guided_sam3['geometry'], observation['slot_guided_current_frames'],
                                       slot_guided_sam3['bridge_checkpoint_path'], slot_guided_sam3['bridge_git_sha'])
        return SimpleNamespace(action=np.array([.1 + .02 * self.enable_slot_guided_sam3, 0., 0., 0., 0., 0., 1., 1., 0.]))


def test_runner_predicts_C_then_conditioned_B_with_real_authorization_contract(tmp_path):
    checkpoint = config_fixture(tmp_path)
    agent = AgentSeam()
    report = run_single_frame(agent=agent, perception=PerceptionSeam(checkpoint),
        frame_path='unused-frame-in-unit-seam.json', state_path=write_state(tmp_path),
        checkpoint_path=checkpoint, bridge_git_sha='c' * 40, output_dir=tmp_path / 'run')
    assert agent.calls == [False, True] and agent.enable_slot_guided_sam3 is True
    assert report['status'] == 'completed'
    assert report['action_translation_change_from_C_m'] == pytest.approx(.02)
    assert report['action_translation_error_m'] is None
    assert report['robot_task_success'] is None
    assert report['real_offline_acceptance_passed'] is False
    assert 'acceptance_report' not in report
    assert json.loads((tmp_path / 'run/action_report.json').read_text())['status'] == 'completed'


@pytest.mark.parametrize('options', [dict(empty=True), dict(kind='mock'), dict(confidence=.1)])
def test_runner_empty_or_mock_geometry_falls_back_without_second_action(tmp_path, options):
    checkpoint = config_fixture(tmp_path)
    agent = AgentSeam()
    report = run_single_frame(agent=agent, perception=PerceptionSeam(checkpoint, **options),
        frame_path='unit-seam.json', state_path=write_state(tmp_path), checkpoint_path=checkpoint,
        bridge_git_sha='c' * 40, output_dir=tmp_path / 'run')
    assert report['status'] == 'baseline_fallback' and report['conditioned_action'] is None
    assert agent.calls == [False] and agent.enable_slot_guided_sam3
    assert report['failure_reason'] and report['fallback_action'] == report['baseline_C_action']


def test_formal_command_config_check_needs_no_sam3_or_checkpoint(capsys):
    assert main(['--check-config', '--config', str(PROFILE)]) == 0
    assert json.loads(capsys.readouterr().out)['enabled'] is True


def test_real_command_input_preflight_without_loading_either_model(tmp_path, capsys):
    from PIL import Image
    task_repo = Path(os.environ.get('TASK_RELEVANT_OBJECT_REPO', ROOT.parent / 'TASK_Relevant_object-main'))
    if not (task_repo / 'slot_guidance_producers.py').is_file():
        pytest.skip('set TASK_RELEVANT_OBJECT_REPO for the actual single-frame CLI contract')
    checkpoint = config_fixture(tmp_path)
    sam_checkpoint = tmp_path / 'sam3.test'
    sam_checkpoint.write_bytes(b'preflight existence fixture only, never loaded')
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(tmp_path / 'rgb.png')
    np.save(tmp_path / 'depth.npy', np.ones((8, 8)))
    camera_rows = []
    for name in ('front', 'left_shoulder', 'right_shoulder', 'wrist'):
        pose = np.eye(4)
        pose[:3, 3] = [.2, 0., .1]
        camera_rows.append(dict(name=name, rgb_path='rgb.png', depth_path='depth.npy',
            depth_encoding='meters_npy', source_frame_index=17, observed_at=IDENTITY['observed_at'],
            calibration_source='current synthetic CLI fixture',
            intrinsics=[[100., 0., 4.], [0., 100., 4.], [0., 0., 1.]], extrinsics=pose.tolist()))
    frame = tmp_path / 'frame.json'
    frame_meta = dict(schema_version=1, artifact_type='slot_rgbd_frame', **IDENTITY, cameras=camera_rows)
    frame.write_text(json.dumps(frame_meta))
    arguments = ['--check-inputs', '--config', str(PROFILE), '--checkpoint', str(checkpoint),
                 '--task-repo', str(task_repo), '--sam3-checkpoint', str(sam_checkpoint),
                 '--frame', str(frame), '--state', str(write_state(tmp_path))]
    assert main(arguments) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['models_loaded'] is False and report['real_offline_acceptance_passed'] is False
    assert report['current_frame']['source_frame_index'] == 17
    frame_meta['cameras'][-1]['source_frame_index'] = 18
    frame.write_text(json.dumps(frame_meta))
    assert main(arguments) == 2
    assert 'source frame/time' in capsys.readouterr().err


def test_standard_rollout_entrypoint_rejects_offline_profile_before_model_build():
    tree = ast.parse((ROOT / 'finetune/RLBench/eval.py').read_text(encoding='utf-8'))
    loader = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'load_agent')
    guard = next(node for node in loader.body if isinstance(node, ast.If)
                 and 'slot_guided_sam3_execution_mode' in ast.unparse(node.test))
    compiled = compile(ast.Module(body=[guard], type_ignores=[]), '<actual_rollout_guard>', 'exec')
    with pytest.raises(ValueError, match='offline_experiment snapshots'):
        exec(compiled, {'exp_cfg': load_profile(PROFILE)})
    exec(compiled, {'exp_cfg': get_cfg_defaults()})


def test_checkpoint_change_after_model_loading_rejected(tmp_path):
    checkpoint = config_fixture(tmp_path)
    loaded_sha = sha256_file(checkpoint)
    checkpoint.write_bytes(b'changed after the policy loaded its weights')
    with pytest.raises(ValueError, match='no longer matches'):
        run_single_frame(agent=AgentSeam(), perception=PerceptionSeam(checkpoint),
            frame_path='unit-seam.json', state_path=write_state(tmp_path),
            checkpoint_path=checkpoint, bridge_git_sha='c' * 40, output_dir=tmp_path / 'run',
            model_config={'bridge_checkpoint_sha256': loaded_sha})
