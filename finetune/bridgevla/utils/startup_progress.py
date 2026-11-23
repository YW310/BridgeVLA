"""Progress and waiting-stack diagnostics for sequential evaluation operations."""
from contextlib import contextmanager
import faulthandler
import time


@contextmanager
def startup_phase(label, *, traceback_after=60, prefix='Startup'):
    """Report a startup phase and dump stacks periodically until it finishes.

    Use sequentially, without nesting: faulthandler owns one process-wide timer.
    The timer diagnoses a slow operation; it never interrupts or terminates it.
    """
    started = time.perf_counter()
    print(f'[{prefix}] START {label} '
          f'(thread stacks every {traceback_after:g}s while waiting)', flush=True)
    faulthandler.dump_traceback_later(traceback_after, repeat=True)
    try:
        yield
    except BaseException:
        print(f'[{prefix}] FAILED {label} after {time.perf_counter() - started:.1f}s', flush=True)
        raise
    else:
        print(f'[{prefix}] DONE {label} in {time.perf_counter() - started:.1f}s', flush=True)
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
        yield transition
        step += 1
