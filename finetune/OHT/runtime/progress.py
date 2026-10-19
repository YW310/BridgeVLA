"""Periodic stderr diagnostics, including while the current operation is blocked."""
import math
import sys
import threading
import time


class ProgressReporter:
    def __init__(self, interval=10, enabled=True, stream=None):
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("Progress interval must be finite and positive")
        self.interval, self.enabled = interval, enabled
        self.stream = sys.stderr if stream is None else stream
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._stage = None
        self._details = {}
        self._started = self._stage_started = time.perf_counter()
        self._samples_started = None

    def __enter__(self):
        if self.enabled:
            self._thread = threading.Thread(target=self._heartbeat, name="oht-eval-progress", daemon=True)
            self._thread.start()
        return self

    def update(self, stage, *, emit=True, **details):
        if not self.enabled:
            return
        with self._lock:
            self._stage, self._details = stage, details
            self._stage_started = time.perf_counter()
            if "total" in details and self._samples_started is None:
                self._samples_started = self._stage_started
            if emit:
                self._print()

    def _print(self):
        now = time.perf_counter()
        fields = dict(stage=self._stage, elapsed_s=round(now - self._started, 1),
                      stage_s=round(now - self._stage_started, 1), **self._details)
        completed, total = fields.get("completed", 0), fields.get("total")
        if total is not None and completed and self._samples_started is not None:
            elapsed = now - self._samples_started
            fields.update(samples_per_s=round(completed / max(elapsed, 1e-9), 3),
                          eta_s=round((total - completed) * elapsed / completed, 1))
        print("[OHT eval] " + " ".join(f"{key}={value}" for key, value in fields.items()),
              file=self.stream, flush=True)

    def _heartbeat(self):
        while not self._stop.wait(self.interval):
            with self._lock:
                if self._stage is not None:
                    self._print()

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is not None:
                self.update("failed", error=f"{exc_type.__name__}: {exc}")
        finally:
            self._stop.set()
            if self._thread is not None:
                self._thread.join()
