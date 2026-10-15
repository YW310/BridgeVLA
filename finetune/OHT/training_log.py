"""Scalar TensorBoard logs, independent of model construction and replay data."""
import math
from numbers import Real
from pathlib import Path


def summary_writer_class():
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as exc:
        raise RuntimeError(
            "OHT TensorBoard logging requires tensorboard. Run: python -m pip install tensorboard "
            "or select --log-backend none to keep JSONL metrics only."
        ) from exc
    return SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir, *, rank=0, backend="tensorboard", flush_secs=10, resume_step=None):
        if backend not in ("tensorboard", "none"):
            raise ValueError("Unsupported OHT log backend")
        if not isinstance(flush_secs, Real) or not math.isfinite(flush_secs) or flush_secs <= 0:
            raise ValueError("TensorBoard flush_secs must be positive and finite")
        self.writer = None
        if rank == 0 and backend == "tensorboard":
            self.writer = summary_writer_class()(
                log_dir=str(Path(log_dir)), flush_secs=flush_secs,
                purge_step=resume_step + 1 if resume_step is not None else None,
            )

    def log(self, step, metrics, learning_rates=()):
        if self.writer is None:
            return
        for key, value in metrics.items():
            if isinstance(value, Real):
                self.writer.add_scalar(f"train/{key}", value, step)
        for index, value in enumerate(learning_rates):
            tag = "train/learning_rate" if index == 0 else f"train/learning_rate_group_{index}"
            self.writer.add_scalar(tag, value, step)

    def flush(self):
        if self.writer is not None:
            self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
