"""Startup watchdogs report failures and never leak into normal evaluation."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'finetune'))
from bridgevla.utils import startup_progress


@pytest.mark.parametrize('fails', [False, True])
def test_watchdog_is_cancelled_on_success_and_error(monkeypatch, capsys, fails):
    events = []
    monkeypatch.setattr(startup_progress.faulthandler, 'dump_traceback_later',
                        lambda delay, **kw: events.append(('arm', delay, kw)))
    monkeypatch.setattr(startup_progress.faulthandler, 'cancel_dump_traceback_later',
                        lambda: events.append(('cancel',)))
    error = RuntimeError('backend load failed')

    def run():
        with startup_progress.startup_phase('SAM3 preload'):
            events.append(('work',))
            if fails:
                raise error

    if fails:
        with pytest.raises(RuntimeError) as caught:
            run()
        assert caught.value is error
    else:
        run()
    assert events == [('arm', 60, {'repeat': True}), ('work',), ('cancel',)]
    output = capsys.readouterr().out
    assert '[Startup] START SAM3 preload' in output
    assert ('FAILED' if fails else 'DONE') + ' SAM3 preload' in output


def test_real_watchdog_prints_waiting_stack_and_process_continues():
    # Use a child so the test cannot replace pytest's process-wide watchdog.
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'finetune'))
    result = subprocess.run([sys.executable, '-c', '''
import time
from bridgevla.utils.startup_progress import startup_phase
with startup_phase('slow backend', traceback_after=0.05):
    time.sleep(0.25)
print('evaluation continues', flush=True)
'''], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'Timeout' in result.stderr and '<module>' in result.stderr
    assert '[Startup] DONE slow backend' in result.stdout
    assert 'evaluation continues' in result.stdout
