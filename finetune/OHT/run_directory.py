"""BridgeVLA-style timestamped run directories shared by all training ranks."""
from datetime import datetime, timedelta, timezone
from itertools import count
from pathlib import Path

from .data.common import write_json

RUN_TIMEZONE = timezone(timedelta(hours=8), "Asia/Shanghai")


def _component(value, label, allow_empty=False):
    if allow_empty and value == "":
        return value
    if (not isinstance(value, str) or not value.strip() or value in (".", "..")
            or "/" in value or "\\" in value):
        raise ValueError(f"{label} must be a single nonempty directory name")
    return value


def _create_or_resume(args, config, now):
    if args.resume:
        checkpoint = Path(args.resume).resolve(strict=True)
        if not checkpoint.is_file():
            raise ValueError("--resume must point to a checkpoint file")
        output = checkpoint.parent
        if args.output and Path(args.output).resolve() != output:
            raise ValueError("--output must match the resume checkpoint directory; omit it to resume in place")
        return output

    experiment = _component(args.exp_id or config.get("exp_id") or Path(args.config).stem, "exp_id")
    note = _component(args.exp_note, "exp_note", allow_empty=True)
    if args.output:
        output = Path(args.output).resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        parent = Path(args.log_dir or "exp/OHT").resolve() / "train" / experiment
        if note:
            parent /= note
        parent.mkdir(parents=True, exist_ok=True)
        timestamp = now.strftime("%m_%d_%H_%M")
        for index in count():
            output = parent / (timestamp if index == 0 else f"{timestamp}_{index:02d}")
            try:
                output.mkdir(exist_ok=False)
                break
            except FileExistsError:
                continue
    write_json(output / "run.json", dict(
        output=str(output), exp_id=experiment, exp_note=note,
        created_at=now.isoformat(), timezone="Asia/Shanghai",
        config=str(Path(args.config).resolve()), arguments=vars(args),
    ))
    return output


def prepare_run_directory(args, config, *, rank=0, world_size=1, distributed=None, now=None):
    """Only rank zero creates a run; broadcast its path or error to every rank."""
    if world_size == 1:
        return _create_or_resume(args, config, now or datetime.now(RUN_TIMEZONE))
    if distributed is None:
        raise ValueError("Distributed output selection requires a process group")
    message = [None]
    if rank == 0:
        try:
            output = _create_or_resume(args, config, now or datetime.now(RUN_TIMEZONE))
            message[0] = dict(output=str(output))
        except (OSError, ValueError) as exc:
            message[0] = dict(error=str(exc), error_type=type(exc).__name__)
    distributed.broadcast_object_list(message, src=0)
    if "error" in message[0]:
        error = {"FileExistsError": FileExistsError, "FileNotFoundError": FileNotFoundError,
                 "ValueError": ValueError}.get(message[0]["error_type"], OSError)
        raise error(message[0]["error"])
    return Path(message[0]["output"])
