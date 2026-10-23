import importlib.util
import json
from pathlib import Path
import subprocess

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
    def run(command, **kwargs):
        assert kwargs['check'] and kwargs['cwd'] == pipeline.REPO_ROOT
        assert (output / 'dataset.yaml').is_file()
        assert (output / 'seg_roles.yaml').is_file()
        calls.append(command)
        if 'build_oht_replay.py' == Path(command[2]).name:
            replay = output / f'replay-source-v{version}'
            replay.mkdir()
            for filename in ('contract.json', 'complete.json'):
                (replay / filename).touch()
    monkeypatch.setattr(pipeline.subprocess, 'run', run)
    # Default invocation must preserve dense targets; V3 is explicitly requested.
    options = ['--replay-version', '3'] if version == 3 else []
    assert pipeline.main(['--root', str(root), '--output', str(output), *options]) == 0
    status = json.loads((output / 'pipeline.json').read_text())
    expected = ['audit', 'replay', 'validate', 'seg-roles'] + (['teacher'] if version == 3 else [])
    assert status['status'] == 'complete' and status['completed'] == expected
    assert len(calls) == len(expected)
    assert calls[0][calls[0].index('--root') + 1] == str(root.resolve())
    assert ('--chunk-horizon' in calls[1]) == (version == 4)
    if version == 4:
        assert calls[1][calls[1].index('--replay-version') + 1] == '4'
        assert calls[1][calls[1].index('--chunk-horizon') + 1] == '16'
        assert calls[1][calls[1].index('--action-frame-stride') + 1] == '1'
    with pytest.raises(SystemExit):
        pipeline.main(['--root', str(root), '--output', str(output)])


def test_pipeline_stops_after_bad_audit_and_dry_run_writes_nothing(tmp_path, monkeypatch):
    output = tmp_path / 'output'
    args = ['--root', str(tmp_path), '--output', str(output)]
    calls = []
    def fail(command, **kwargs):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(pipeline.subprocess, 'run', fail)
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
    monkeypatch.setattr(pipeline.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    assert pipeline.main(['--root', str(tmp_path), '--output', str(output)]) == 1
    status = json.loads((output / 'pipeline.json').read_text())
    assert status['failed_stage'] == 'replay'
    assert status['completed'] == ['audit']
    assert 'without creating' in status['error']
    assert str(output / 'replay-source-v4/contract.json') in status['error']
    assert len(calls) == 2  # Neither validation nor segmentation reads the absent contract.
