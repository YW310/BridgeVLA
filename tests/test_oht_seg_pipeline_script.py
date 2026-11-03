import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

spec = importlib.util.spec_from_file_location('oht_prepare', Path(__file__).parents[1] / 'tools/prepare_oht_seg_data.py')
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


@pytest.mark.parametrize('version', [3, 4])
def test_pipeline_stage_order_and_v4_boundary(tmp_path, monkeypatch, version):
    root = tmp_path / 'raw with spaces'
    root.mkdir()
    output = tmp_path / 'generated'
    calls = []
    def run(command, log_path):
        assert log_path.parent == output / 'logs'
        assert (output / 'dataset.yaml').is_file()
        assert (output / 'seg_roles.yaml').is_file()
        calls.append(command)
        if 'build_oht_replay.py' == Path(command[2]).name:
            replay = output / f'replay-source-v{version}'
            replay.mkdir()
            for filename in ('contract.json', 'complete.json'):
                (replay / filename).touch()
    monkeypatch.setattr(pipeline, 'run_stage', run)
    # Default invocation must preserve dense targets; V3 is explicitly requested.
    options = ['--replay-version', '3'] if version == 3 else []
    assert pipeline.main(['--root', str(root), '--output', str(output), *options]) == 0
    status = json.loads((output / 'pipeline.json').read_text())
    expected = (['audit'] + (['workspace'] if version == 4 else []) +
                ['replay', 'validate', 'seg-roles'] + (['teacher'] if version == 3 else ['role-preview']))
    assert status['status'] == 'complete' and status['completed'] == expected
    assert len(calls) == len(expected)
    assert calls[0][calls[0].index('--root') + 1] == str(root.resolve())
    build = next(command for command in calls if Path(command[2]).name == 'build_oht_replay.py')
    assert ('--chunk-horizon' in build) == (version == 4)
    if version == 4:
        assert build[build.index('--replay-version') + 1] == '4'
        assert build[build.index('--chunk-horizon') + 1] == '16'
        assert build[build.index('--action-frame-stride') + 1] == '1'
    with pytest.raises(SystemExit):
        pipeline.main(['--root', str(root), '--output', str(output)])


def test_pipeline_stops_after_bad_audit_and_dry_run_writes_nothing(tmp_path, monkeypatch):
    output = tmp_path / 'output'
    args = ['--root', str(tmp_path), '--output', str(output)]
    calls = []
    def fail(command, log_path):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(pipeline, 'run_stage', fail)
    assert pipeline.main([*args, '--dry-run']) == 0
    assert not output.exists() and not calls
    assert pipeline.main(args) == 1
    assert len(calls) == 1
    status = json.loads((output / 'pipeline.json').read_text())
    assert status['status'] == 'failed' and status['failed_stage'] == 'audit'
    assert status['completed'] == []


def test_empty_shell_root_is_rejected_before_output_creation(tmp_path, capsys):
    output = tmp_path / 'empty-root'
    with pytest.raises(SystemExit):
        pipeline.main(['--root', '', '--output', str(output)])
    assert '--root is empty' in capsys.readouterr().err
    assert not output.exists()


def test_setup_error_is_recorded_before_audit(tmp_path, monkeypatch):
    output = tmp_path / 'copy-failed'
    def fail(*args):
        raise PermissionError('config copy denied')
    monkeypatch.setattr(pipeline.shutil, 'copyfile', fail)
    assert pipeline.main(['--root', str(tmp_path), '--output', str(output)]) == 1
    status = json.loads((output / 'pipeline.json').read_text())
    assert status['failed_stage'] == 'setup'
    assert status['error'] == 'config copy denied'
    assert status['completed'] == []


def test_missing_contract_is_a_build_failure_not_a_validation_error(tmp_path, monkeypatch):
    output = tmp_path / 'missing-contract'
    calls = []
    monkeypatch.setattr(pipeline, 'run_stage', lambda command, log_path: calls.append(command))
    assert pipeline.main(['--root', str(tmp_path), '--output', str(output)]) == 1
    status = json.loads((output / 'pipeline.json').read_text())
    assert status['failed_stage'] == 'replay'
    assert status['completed'] == ['audit', 'workspace']
    assert 'without creating' in status['error']
    assert str(output / 'replay-source-v4/contract.json') in status['error']
    assert len(calls) == 3  # Neither validation nor segmentation reads the absent contract.


def test_stage_output_is_streamed_and_full_traceback_saved(tmp_path, capsys):
    log = tmp_path / 'logs' / 'replay.log'
    command = [sys.executable, '-u', '-c',
               'import sys; print("building", flush=True); print("depth message", file=sys.stderr); '
               'raise ValueError("GT target outside bounds")']
    with pytest.raises(subprocess.CalledProcessError) as error:
        pipeline.run_stage(command, log)
    saved = log.read_text(encoding='utf-8')
    assert 'building' in saved and 'depth message' in saved
    assert 'Traceback' in saved and 'ValueError: GT target outside bounds' in saved
    assert capsys.readouterr().out == saved
    assert 'ValueError: GT target outside bounds' in error.value.output


def test_failed_replay_records_diagnostic_tail_and_log_path(tmp_path, monkeypatch):
    output = tmp_path / 'failed-replay'
    def run(command, log_path):
        if Path(command[2]).name == 'build_oht_replay.py':
            raise subprocess.CalledProcessError(1, command, output='ValueError: GT target outside bounds\n')
    monkeypatch.setattr(pipeline, 'run_stage', run)
    assert pipeline.main(['--root', str(tmp_path), '--output', str(output)]) == 1
    status = json.loads((output / 'pipeline.json').read_text())
    assert status['completed'] == ['audit', 'workspace'] and status['failed_stage'] == 'replay'
    assert status['error_tail'] == 'ValueError: GT target outside bounds\n'
    assert status['log_file'] == status['logs']['replay'] == str(output / 'logs/replay.log')


def test_reuse_audit_copies_manifest_and_starts_with_workspace(tmp_path, monkeypatch):
    manifest = tmp_path / 'previous-audit.json'
    manifest.write_text('{"copied": true}\n')
    output = tmp_path / 'retry'
    calls = []
    def stop(command, log_path):
        calls.append(command)
        assert (output / 'audit.json').read_bytes() == manifest.read_bytes()
        raise subprocess.CalledProcessError(1, command, output='coverage check failed')
    monkeypatch.setattr(pipeline, 'run_stage', stop)
    assert pipeline.main(['--root', str(tmp_path), '--output', str(output),
                          '--audit-manifest', str(manifest)]) == 1
    status = json.loads((output / 'pipeline.json').read_text())
    assert status['reused'] == status['completed'] == ['audit']
    assert status['failed_stage'] == 'workspace'
    assert len(calls) == 1 and Path(calls[0][2]).name == 'check_oht_workspace.py'
