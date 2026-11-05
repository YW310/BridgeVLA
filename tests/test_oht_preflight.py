"""The data-only training CLI must work without the model dependencies."""
import json
import os
from pathlib import Path
import subprocess
import sys

from tests.test_oht_migration import replay_fixture  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]


def run_without_torch(*arguments):
    script = """
import importlib.abc
import runpy
import sys

class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'torch':
            raise ModuleNotFoundError('PyTorch is unavailable in a data-only environment', name=fullname)

sys.meta_path.insert(0, NoTorch())
sys.argv = ['finetune.OHT.train', *sys.argv[1:]]
try:
    runpy.run_module('finetune.OHT.train', run_name='__main__')
except SystemExit as error:
    assert error.code == 0, error.code
assert 'torch' not in sys.modules
"""
    environment = dict(os.environ, RANK='0', LOCAL_RANK='0', WORLD_SIZE='1')
    return subprocess.run([sys.executable, '-c', script, *arguments], cwd=ROOT,
                          env=environment, capture_output=True, text=True, timeout=30)


def test_help_without_pytorch():
    result = run_without_torch('--help')
    assert result.returncode == 0, result.stderr
    assert '--validate-only' in result.stdout


def test_validate_only_reads_replay_without_pytorch(replay_fixture, tmp_path):
    output = tmp_path / 'unused'
    result = run_without_torch('--config', str(ROOT / 'finetune/OHT/configs/smoke.yaml'),
                               '--replay', str(replay_fixture.replay), '--output', str(output),
                               '--validate-only')
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    samples = (replay_fixture.replay / 'samples.jsonl').read_text(encoding='utf-8').splitlines()
    assert report['valid'] is True
    assert report['samples'] == sum(json.loads(row)['split'] == 'train' for row in samples) > 0
    assert report['training_plan']['optimizer_steps'] == 2
    assert not output.exists()
