"""Epoch/sample budgets using the same batch arithmetic as RLBench training."""
from ..RLBench.training_utils import build_batch_plan, optimizer_steps_per_epoch


def _positive(config, key):
    value = config.get(key)
    if type(value) is not int or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def validate_training_settings(config):
    _positive(config, "batch_size")
    target = config.get("global_batch_size", 0)
    if type(target) is not int or target < 0:
        raise ValueError("global_batch_size must be a nonnegative integer")
    if config.get("accumulation_steps") is not None or not target:
        _positive(config, "accumulation_steps")
    if config.get("sample_distribution_mode", "task_uniform") != "task_uniform":
        raise ValueError("OHT supports sample_distribution_mode=task_uniform")
    if config.get("use_scheduler", False):
        raise ValueError("OHT currently supports use_scheduler=false (constant learning rate)")
    if config.get("epochs") is None:
        _positive(config, "optimizer_steps")
        _positive(config, "checkpoint_interval")
    else:
        _positive(config, "epochs")
        if config.get("optimizer_steps") is not None or config.get("checkpoint_interval") is not None:
            raise ValueError("Epoch budgets require optimizer_steps=null and checkpoint_interval=null")
        if config.get("train_iter") is not None:
            _positive(config, "train_iter")
        cap = config.get("max_optimizer_steps", 0)
        if type(cap) is not int or cap < 0:
            raise ValueError("max_optimizer_steps must be a nonnegative integer")
        _positive(config, "checkpoint_every_epochs")


def resolve_training_plan(config, training_samples, world_size):
    validate_training_settings(config)
    if type(world_size) is not int or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    if type(training_samples) is not int or training_samples < 1:
        raise ValueError("training_samples must be a positive integer")
    batch = config["batch_size"]
    if config.get("global_batch_size", 0):
        batch_plan = build_batch_plan(batch, world_size, config["global_batch_size"])
        accumulation = batch_plan.gradient_accumulation_steps
        if config.get("accumulation_steps") not in (None, accumulation):
            raise ValueError("Explicit accumulation_steps conflicts with global_batch_size; use null for automatic accumulation")
        effective = batch_plan.target_global_batch_size
    else:
        accumulation = config["accumulation_steps"]
        effective = batch * world_size * accumulation
    epochs = config.get("epochs")
    epoch_samples, steps_per_epoch = None, None
    if epochs is None:
        steps, interval = config["optimizer_steps"], config["checkpoint_interval"]
    else:
        epoch_samples = config.get("train_iter") or training_samples
        steps_per_epoch = optimizer_steps_per_epoch(epoch_samples, effective)
        steps = epochs * steps_per_epoch
        if config.get("max_optimizer_steps", 0):
            steps = min(steps, config["max_optimizer_steps"])
        interval = config["checkpoint_every_epochs"] * steps_per_epoch
    return dict(optimizer_steps=steps, accumulation_steps=accumulation,
                effective_batch=effective, checkpoint_interval=interval,
                epochs=epochs, steps_per_epoch=steps_per_epoch, train_iter=epoch_samples,
                training_samples=training_samples, world_size=world_size)


def epoch_metrics(step, plan):
    width = plan["steps_per_epoch"]
    if width is None:
        return {}
    return dict(epoch=(step - 1) // width + 1, epoch_step=(step - 1) % width + 1,
                epoch_progress=step / width)
