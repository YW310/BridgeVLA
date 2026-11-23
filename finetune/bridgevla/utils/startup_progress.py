"""Progress and waiting-stack diagnostics for sequential evaluation operations."""
from contextlib import contextmanager
from contextvars import ContextVar
import faulthandler
import threading
import time


_LIVE_PROGRESS = ContextVar('evaluation_live_progress', default=None)


@contextmanager
def live_phase(label, *, prefix='Online', heartbeat_interval=5):
    """Flush phase changes immediately and report the deepest phase while waiting.

    Nested phases share one heartbeat thread. Only the evaluation thread runs
    model/simulator code; the background thread only reads labels and prints.
    """
    if heartbeat_interval <= 0:
        raise ValueError('heartbeat_interval must be positive')
    state = _LIVE_PROGRESS.get()
    owner = state is None
    if owner:
        state = dict(lock=threading.RLock(), stop=threading.Event(), current=None)
        token = _LIVE_PROGRESS.set(state)
    started = time.perf_counter()
    with state['lock']:
        previous = state['current']
        state['current'] = (prefix, label, started)
        print(f'[{prefix}] START {label}', flush=True)

    def heartbeat():
        while not state['stop'].wait(heartbeat_interval):
            with state['lock']:
                current_prefix, current_label, current_started = state['current']
                print(f'[{current_prefix}] WAIT {current_label}; '
                      f'elapsed={time.perf_counter() - current_started:.1f}s', flush=True)

    if owner:
        worker = threading.Thread(target=heartbeat, name='evaluation-progress', daemon=True)
        worker.start()
    outcome = 'DONE'
    try:
        yield
    except BaseException:
        outcome = 'FAILED'
        raise
    finally:
        if owner:
            state['stop'].set()
            worker.join()
        with state['lock']:
            print(f'[{prefix}] {outcome} {label} in {time.perf_counter() - started:.1f}s', flush=True)
            state['current'] = previous
        if owner:
            _LIVE_PROGRESS.reset(token)


@contextmanager
def startup_phase(label, *, traceback_after=60, prefix='Startup'):
    """Report a startup phase and dump stacks periodically until it finishes.

    Use sequentially, without nesting: faulthandler owns one process-wide timer.
    The timer diagnoses a slow operation; it never interrupts or terminates it.
    """
    with live_phase(label, prefix=prefix):
        faulthandler.dump_traceback_later(traceback_after, repeat=True)
        try:
            yield
        finally:
            faulthandler.cancel_dump_traceback_later()


def rollout_progress(transitions, label, *, traceback_after=60):
    """Watch actual generator advancement, including reset before its first yield.

    Cancel the watchdog before yielding to the caller, so its log writes and
    other work do not share this timer. Both ordinary and visualized rollouts
    retain their original transition objects and exception behavior.
    """
    iterator = iter(transitions)
    exhausted = object()
    step = 0
    while True:
        operation = ('reset + first policy/action step' if step == 0
                     else 'next policy/action step or episode completion')
        with startup_phase(f'{label}, step={step}: {operation}',
                           traceback_after=traceback_after, prefix='Rollout'):
            transition = next(iterator, exhausted)
        if transition is exhausted:
            print(f'[Rollout] COMPLETE {label}; transitions={step}', flush=True)
            return
        print(f'[Rollout] STEP {label}, step={step}; '
              f'reward={getattr(transition, "reward", None)}; '
              f'terminal={getattr(transition, "terminal", None)}; '
              f'timeout={getattr(transition, "timeout", None)}', flush=True)
        yield transition
        step += 1
