"""Progress and bounded-wait diagnostics for sequential startup operations."""
from contextlib import contextmanager
import faulthandler
import time


@contextmanager
def startup_phase(label, *, traceback_after=60):
    """Report a startup phase and dump stacks periodically until it finishes.

    Use sequentially, without nesting: faulthandler owns one process-wide timer.
    The timer diagnoses a slow operation; it never interrupts or terminates it.
    """
    started = time.perf_counter()
    print(f'[Startup] START {label} '
          f'(thread stacks every {traceback_after:g}s while waiting)', flush=True)
    faulthandler.dump_traceback_later(traceback_after, repeat=True)
    try:
        yield
    except BaseException:
        print(f'[Startup] FAILED {label} after {time.perf_counter() - started:.1f}s', flush=True)
        raise
    else:
        print(f'[Startup] DONE {label} in {time.perf_counter() - started:.1f}s', flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()
