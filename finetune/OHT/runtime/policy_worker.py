"""One HTTP request at a time, with simulator work kept on the calling thread."""
from concurrent.futures import Future, wait
import math
from queue import Queue
from threading import Thread


class PolicyWorker:
    """Own HTTP calls only; observations must already be independent CPU snapshots.

    A daemon thread permits window-close cleanup without waiting for a socket
    timeout. An abandoned request is never retried and its result is not executed.
    The caller must consume each result before submitting another request.
    """

    def __init__(self, client, instruction, data_profile):
        self.client, self.instruction, self.data_profile = client, instruction, data_profile
        self._jobs = Queue()
        self._future = None
        self._closed = False
        self._thread = Thread(target=self._run, name='OHTPolicyHTTP', daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            job = self._jobs.get()
            if job is None:
                return
            future, function, args, kwargs = job
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(function(*args, **kwargs))
            except BaseException as exc:
                future.set_exception(exc)

    def _submit(self, function, *args, **kwargs):
        if self._closed:
            raise RuntimeError('Policy worker is closed')
        if self._future is not None:
            raise RuntimeError('Consume the pending policy result before submitting another request')
        self._future = Future()
        self._jobs.put((self._future, function, args, kwargs))

    def submit(self, observation, task, timestamp):
        self._submit(self.client.predict, observation, self.instruction, timestamp,
                     task=task, data_profile=self.data_profile)

    def submit_reset(self, episode=None):
        self._submit(self.client.reset, episode)

    def result(self, wait_callback=None, poll_interval=.05):
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError('poll_interval must be finite and positive')
        if self._closed:
            raise RuntimeError('Policy worker is closed')
        if self._future is None:
            raise RuntimeError('No pending policy request')
        future = self._future
        try:
            while not future.done():
                if wait_callback is not None:
                    wait_callback()
                wait((future,), timeout=poll_interval)
            return future.result()
        except BaseException:
            self.close()
            raise
        finally:
            self._future = None

    def close(self):
        if not self._closed:
            self._closed = True
            if self._future is not None:
                self._future.cancel()
            self._jobs.put(None)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def refresh_while_waiting(reference, env):
    """Pump GUI events using IsaacLab's render-only API, never env.step/app.update."""
    app = reference.simulation_app
    if not app.is_running():
        raise RuntimeError('Simulation application stopped while waiting for policy')
    sim = getattr(env, 'sim', None)
    has_gui = getattr(sim, 'has_gui', None)
    headless = getattr(getattr(reference, 'args_cli', None), 'headless', False)
    gui = bool(has_gui()) if callable(has_gui) else not headless
    if gui:
        render = getattr(sim, 'render', None)
        if callable(render):
            render()
        else:
            # Compatibility runtimes must implement the existing render-only hook.
            reference._replicator_render_tick(env)
    if not app.is_running():
        raise RuntimeError('Simulation application stopped while waiting for policy')
