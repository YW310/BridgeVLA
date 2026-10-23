"""Configurable training diagnostics using the shared RLBench montage writer."""
from pathlib import Path

DEFAULTS = dict(enabled=False, interval=500, save_png=True, tensorboard=True,
                output_dir="train_visualizations")


def visualization_settings(config, backend=None):
    provided = config.get("train_visualization", {})
    if not isinstance(provided, dict) or set(provided) - set(DEFAULTS):
        raise ValueError("Invalid train_visualization settings")
    settings = {**DEFAULTS, **provided}
    for key in ("enabled", "save_png", "tensorboard"):
        if type(settings[key]) is not bool:
            raise ValueError(f"train_visualization.{key} must be boolean")
    if type(settings["interval"]) is not int or settings["interval"] < 1:
        raise ValueError("train_visualization.interval must be a positive integer")
    if not isinstance(settings["output_dir"], str) or not settings["output_dir"].strip():
        raise ValueError("train_visualization.output_dir must be a nonempty path")
    if settings["enabled"]:
        if not (settings["save_png"] or settings["tensorboard"]):
            raise ValueError("Training visualization requires save_png or tensorboard")
        if settings["tensorboard"] and backend not in (None, "tensorboard"):
            raise ValueError("train_visualization.tensorboard=true requires --log-backend tensorboard")
    return settings


def capture_due(settings, step, rank=0):
    return rank == 0 and settings["enabled"] and (step == 1 or step % settings["interval"] == 0)


def _first_text(value):
    while isinstance(value, (list, tuple)) and value:
        value = value[0]
    return str(value)


def record_visualization(payload, *, step, output, settings, batch, logger):
    if not payload:
        raise RuntimeError("Training visualization was requested but the agent returned no image payload")
    if payload.get('chunk'):
        from .chunk_visualization import record
        return record(payload, step, output, settings, logger)
    from ..RLBench.training_visualization import record_training_visualization
    directory = Path(settings["output_dir"])
    if not directory.is_absolute():
        directory = Path(output) / directory
    saved = record_training_visualization(
        payload, step=step, output_dir=directory,
        task=_first_text(batch["tasks"]), language_goal=_first_text(batch["lang_goal"]),
        save_png=settings["save_png"], writer=logger.writer,
        write_tensorboard=settings["tensorboard"],
    )
    logger.flush()
    return saved
