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


@pytest.mark.parametrize('operation', ['startup', 'rollout'])
def test_real_watchdog_prints_waiting_stack_and_process_continues(operation):
    # Use a child so the test cannot replace pytest's process-wide watchdog.
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'finetune'))
    code = '''
import time
from bridgevla.utils.startup_progress import rollout_progress, startup_phase
def reset_to_demo():
    time.sleep(0.25)
def transitions():
    reset_to_demo()
    yield 'first transition'
'''
    if operation == 'startup':
        code += '''
with startup_phase('slow backend', traceback_after=0.05):
    reset_to_demo()
'''
    else:
        code += '''
assert list(rollout_progress(transitions(), 'episode=0', traceback_after=0.05)) == ['first transition']
'''
    code += '''
print('evaluation continues', flush=True)
'''
    result = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'Timeout' in result.stderr and 'reset_to_demo' in result.stderr
    assert ('[Startup] DONE slow backend' if operation == 'startup'
            else '[Rollout] COMPLETE episode=0; transitions=1') in result.stdout
    assert 'evaluation continues' in result.stdout


def test_rollout_watchdog_covers_lazy_reset_and_is_off_between_steps(monkeypatch, capsys):
    armed = []
    monkeypatch.setattr(startup_progress.faulthandler, 'dump_traceback_later',
                        lambda *a, **kw: armed.append(True))
    monkeypatch.setattr(startup_progress.faulthandler, 'cancel_dump_traceback_later',
                        lambda: armed.clear())
    first, second = object(), object()
    events = []

    def transitions():
        assert armed
        events.append('reset')
        yield first
        assert armed
        events.append('next step')
        yield second

    wrapped = startup_progress.rollout_progress(transitions(), 'task=stack_blocks, episode=0')
    assert events == [] and not armed  # constructing a generator must not reset the scene
    assert next(wrapped) is first
    assert events == ['reset'] and not armed
    assert next(wrapped) is second
    assert events == ['reset', 'next step'] and not armed
    assert list(wrapped) == [] and not armed
    output = capsys.readouterr().out
    assert 'reset + first policy/action step' in output and 'transitions=2' in output
    assert 'FAILED' not in output  # normal exhaustion is not a failed startup


def test_rollout_exception_keeps_identity_and_cancels_watchdog(monkeypatch):
    events = []
    monkeypatch.setattr(startup_progress.faulthandler, 'dump_traceback_later',
                        lambda *a, **kw: events.append('arm'))
    monkeypatch.setattr(startup_progress.faulthandler, 'cancel_dump_traceback_later',
                        lambda: events.append('cancel'))
    error = RuntimeError('scene reset failed')

    def transitions():
        raise error
        yield  # keep scene reset lazy, as in the real rollout generator

    with pytest.raises(RuntimeError) as caught:
        next(startup_progress.rollout_progress(transitions(), 'episode=0'))
    assert caught.value is error
    assert events == ['arm', 'cancel']
