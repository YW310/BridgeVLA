import ast
import csv
import io
import json
import sys
from multiprocessing import Lock
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'finetune' / 'RLBench'))
from utils.eval_reporting import (
    EVAL_FIELDS, MANIFEST_FIELDS, atomic_write_json, build_eval_run_signature,
    evaluation_result, generated_manifest_entry_count, manifest_result,
    numeric_task_scores, quarantine_file, resumable_eval_episode,
    resumable_manifest)


def accumulator():
    # Execute the actual dependency-free accumulator classes, omitting imports
    # of YARR agent/transition types that otherwise require torch in unit tests.
    path = ROOT / 'finetune/bridgevla/libs/YARR/yarr/utils/stat_accumulator.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    classes = ast.Module(body=[node for node in tree.body if isinstance(node, ast.ClassDef)],
                         type_ignores=[])
    scope = dict(np=np, Lock=Lock, List=List, Summary=object, ReplayTransition=object,
                 ScalarSummary=lambda name, value: SimpleNamespace(name=name, value=value))
    exec(compile(classes, str(path), 'exec'), scope)
    return scope['SimpleAccumulator']()


@pytest.mark.parametrize('rewards, expected', [([100.], 100.), ([0.], 0.), ([100., 0.], 50.)])
def test_completed_episode_metrics_are_numeric_and_drained(rewards, expected):
    stats = accumulator()
    assert stats.pop() == []
    for reward in rewards:
        stats.step(SimpleNamespace(reward=reward, terminal=True, summaries=[]), True)
    values = {s.name: s.value for s in stats.pop()}
    assert values['eval_envs/return'] == expected
    assert values['eval_envs/length'] == 1
    assert stats.pop() == []


def test_generated_manifest_entry_count_uses_sample_frames_not_rollout_length():
    assert generated_manifest_entry_count(
        {'sample_frames': [43, 57, 69, 100, 116]}) == 5


@pytest.mark.parametrize('info', [
    {},
    {'sample_frames': []},
    {'sample_frames': [2, 1]},
    {'sample_frames': [1, 1]},
    {'sample_frames': [False]},
])
def test_generated_manifest_entry_count_rejects_invalid_metadata(info):
    with pytest.raises(ValueError, match='sample_frames'):
        generated_manifest_entry_count(info)


def test_no_completed_episode_is_not_reported_as_success():
    stats = accumulator()
    stats.step(SimpleNamespace(reward=0., terminal=False, summaries=[]), True)
    assert stats.pop() == []


def test_manifest_coverage_is_separate_from_policy_success():
    result = manifest_result('close_jar', 1, 1, 1)
    assert result['generated coverage'] == 100.
    assert 'success rate' not in result
    assert manifest_result('close_jar', 1, 2, 1)['generated coverage'] == 50.
    assert manifest_result('close_jar', 0, 1, 0)['generated coverage'] == 0.
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
    writer.writeheader()
    writer.writerow(result)
    assert 'success rate' not in stream.getvalue()
    assert numeric_task_scores({'close_jar': result['generated coverage']}) == {'close_jar': 100.}


def test_missing_metrics_are_not_fabricated_or_written_as_strings():
    assert numeric_task_scores({'a': 'unknown', 'b': None, 'c': float('nan'),
                                'd': float('inf'), 'e': 0., 'f': np.float32(100.)}) == {
                                    'e': 0., 'f': 100.}
    with pytest.raises(ValueError):
        manifest_result('close_jar', 0, 0, 0)


def test_eval_wires_separate_manifest_csv_and_tensorboard_namespace():
    source = (ROOT / 'finetune/RLBench/eval.py').read_text(encoding='utf-8')
    ast.parse(source)
    assert 'manifest_results.csv' in source
    assert "'manifest_coverage'" in source
    assert 'numeric_task_scores(task_scores)' in source
    shell = (ROOT / 'finetune/RLBench/eval.sh').read_text(encoding='utf-8')
    assert 'merged_manifest_results.csv' in shell
    assert '${result_filename}' in shell
    assert 'START_EPISODE="${START_EPISODE:-0}"' in shell
    assert '--start-episode "${START_EPISODE}"' in shell
    assert 'EVAL_RESUME="${EVAL_RESUME:-${MANIFEST_RESUME}}"' in shell
    assert '--eval-resume' in shell
    assert 'MANIFEST_CONTINUE_ON_ERROR="${MANIFEST_CONTINUE_ON_ERROR:-0}"' in shell
    assert '--manifest-continue-on-error' in shell
    assert 'TASKS=all expanded into 18 isolated task processes' in shell
    assert 'close_jar' in shell and 'turn_tap' in shell
    assert 'eval_resume does not support sim_replay ground-truth execution' in source
    assert '[Manifest][FAILED]' in source
    assert 'rlbench_manifest_failure_v1' in source
    assert 'Manifest generation is model-free; skipping PaliGemma' in source
    assert 'demo_events manifest generation must run without a model agent' in source
    assert 'All requested episodes are complete' in source
    assert 'if environment_launched:' in source
    rollout_source = (
        ROOT / 'finetune/bridgevla/libs/YARR/yarr/utils/rollout_generator.py'
    ).read_text(encoding='utf-8')
    assert rollout_source.index('if manifest_phase_source == "demo_events":') < (
        rollout_source.index('agent.reset()'))
    parser_source = (
        ROOT / 'finetune/bridgevla/utils/rvt_utils.py'
    ).read_text(encoding='utf-8')
    assert '"--eval-resume", "--manifest-resume", dest="eval_resume"' in parser_source


def complete_manifest():
    return {
        'schema_version': 'rlbench_o2_semantic_roles_v1',
        'role_config_sha256': 'current',
        'task': 'close_jar', 'episode_idx': 3,
        'phase_source': 'demo_events', 'source_alignment_validated': True,
        'handle_namespace': 'stored',
        'handle_alignment': {
            'status': 'mask_verified', 'live_to_stored': {'87': 99}},
        'source_frame0_masks': {'front': 'digest'},
        'expected_sample_frames': [0, 5],
        'entries': [
            {'sample_frame': 0, 'completion_satisfied': False,
             'target': {'kind': 'object', 'handles': [99]}, 'reference': None},
            {'sample_frame': 5, 'completion_satisfied': True,
             'target': {'kind': 'object', 'handles': [99]},
             'reference': {'kind': 'site', 'site_position': [.1, .2, .3]}},
        ],
    }


def test_resume_accepts_only_complete_current_manifest(tmp_path):
    path = tmp_path / 'episode_3.json'
    path.write_text(__import__('json').dumps(complete_manifest()), encoding='utf-8')
    info, error = resumable_manifest(
        path, 'close_jar', 3, 'mask_verified', 'current')
    assert error is None and info == {
        'logical_transitions': 2, 'legacy_config_digest': False}


def test_resume_rejects_stale_task_resolver_version(tmp_path):
    value = complete_manifest()
    path = tmp_path / 'episode_3.json'
    path.write_text(__import__('json').dumps(value), encoding='utf-8')

    info, error = resumable_manifest(
        path, 'close_jar', 3, 'mask_verified', 'current',
        'push_buttons_contact_site_v2')

    assert info is None
    assert error == 'task resolver version mismatch'

    value['resolver_version'] = 'push_buttons_contact_site_v2'
    path.write_text(__import__('json').dumps(value), encoding='utf-8')
    info, error = resumable_manifest(
        path, 'close_jar', 3, 'mask_verified', 'current',
        'push_buttons_contact_site_v2')
    assert error is None
    assert info['logical_transitions'] == 2


@pytest.mark.parametrize('mutation, reason', [
    (lambda value: value.update(role_config_sha256='old'), 'digest'),
    (lambda value: value.update(source_alignment_validated=False), 'alignment'),
    (lambda value: value['entries'][-1].update(completion_satisfied=False), 'completion'),
    (lambda value: value.update(expected_sample_frames=[0, 4, 5]), 'coverage'),
    (lambda value: value['entries'][0]['target'].update(handles=[123]), 'handles'),
])
def test_resume_regenerates_stale_or_incomplete_manifest(tmp_path, mutation, reason):
    value = complete_manifest()
    mutation(value)
    path = tmp_path / 'episode_3.json'
    path.write_text(__import__('json').dumps(value), encoding='utf-8')
    info, error = resumable_manifest(
        path, 'close_jar', 3, 'mask_verified', 'current')
    assert info is None and reason in error


def test_standard_eval_resume_is_signature_bound_and_atomic(tmp_path):
    checkpoint = tmp_path / 'model.pth'
    exp_cfg = tmp_path / 'exp.yaml'
    mvt_cfg = tmp_path / 'mvt.yaml'
    role_cfg = tmp_path / 'roles.yaml'
    for path, contents in (
            (checkpoint, b'checkpoint'), (exp_cfg, b'exp: 1\n'),
            (mvt_cfg, b'mvt: 1\n'), (role_cfg, b'roles: 1\n')):
        path.write_bytes(contents)
    signature, digest = build_eval_run_signature(
        checkpoint, exp_cfg, mvt_cfg, tmp_path / 'data',
        episode_length=50, oracle_provider='rlbench_gt',
        oracle_role_config=role_cfg, oracle_num_points=512,
        oracle_strict=True, oracle_handle_alignment='verified',
        use_input_place_with_mean=False)
    path = tmp_path / 'episode_7.json'
    atomic_write_json(path, {
        'schema_version': 'rlbench_eval_episode_v1',
        'task': 'place_cups', 'episode_idx': 7,
        'reward': 100., 'length': 23, 'attempts_used': 1,
        'run_signature_sha256': digest, 'run_signature': signature,
    })
    info, error = resumable_eval_episode(path, 'place_cups', 7, digest)
    assert error is None
    assert info == {'reward': 100., 'length': 23, 'attempts_used': 1}
    assert not path.with_name(f'.{path.name}.tmp').exists()

    info, error = resumable_eval_episode(path, 'place_cups', 7, 'different')
    assert info is None and error == 'run signature mismatch'


def test_incompatible_manifest_is_quarantined_without_overwrite(tmp_path):
    source = tmp_path / 'semantic_role_manifests' / 'episode_85.json'
    source.parent.mkdir()
    source.write_text('old', encoding='utf-8')
    quarantine = tmp_path / 'rejected'
    first = quarantine_file(source, quarantine)
    assert first.name == 'episode_85.json'
    assert first.read_text(encoding='utf-8') == 'old'
    source.write_text('new', encoding='utf-8')
    second = quarantine_file(source, quarantine)
    assert second.name == 'episode_85.1.json'
    assert second.read_text(encoding='utf-8') == 'new'


def test_standard_eval_signature_changes_with_config_and_runtime_setting(tmp_path):
    checkpoint = tmp_path / 'model.pth'
    exp_cfg = tmp_path / 'exp.yaml'
    mvt_cfg = tmp_path / 'mvt.yaml'
    checkpoint.write_bytes(b'checkpoint')
    exp_cfg.write_text('exp: 1\n', encoding='utf-8')
    mvt_cfg.write_text('mvt: 1\n', encoding='utf-8')

    def signature(episode_length=50):
        return build_eval_run_signature(
            checkpoint, exp_cfg, mvt_cfg, tmp_path / 'data',
            episode_length=episode_length, oracle_provider='none',
            oracle_role_config=None, oracle_num_points=512,
            oracle_strict=False, oracle_handle_alignment='verified',
            use_input_place_with_mean=False)[1]

    original = signature()
    assert signature(episode_length=51) != original
    exp_cfg.write_text('exp: 2\n', encoding='utf-8')
    assert signature() != original


def test_standard_eval_resume_aggregation_includes_skipped_and_new_episodes():
    result = evaluation_result('place_cups', [100., 0., 100.], [10, 20, 30])
    assert list(result) == EVAL_FIELDS
    assert result == {
        'task': 'place_cups', 'success rate': pytest.approx(200 / 3),
        'length': 20, 'total_transitions': 60,
    }


@pytest.mark.parametrize('field, value, reason', [
    ('reward', '100', 'reward'),
    ('reward', float('nan'), 'reward'),
    ('length', 0, 'length'),
    ('attempts_used', 0, 'attempt'),
])
def test_standard_eval_resume_rejects_invalid_episode_journal(
        tmp_path, field, value, reason):
    result = {
        'schema_version': 'rlbench_eval_episode_v1',
        'task': 'place_cups', 'episode_idx': 7,
        'reward': 100., 'length': 23, 'attempts_used': 1,
        'run_signature_sha256': 'current',
    }
    result[field] = value
    path = tmp_path / 'episode_7.json'
    path.write_text(json.dumps(result), encoding='utf-8')
    info, error = resumable_eval_episode(path, 'place_cups', 7, 'current')
    assert info is None and reason in error
