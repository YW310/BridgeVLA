"""Small, dependency-free helpers for RLBench distributed training."""

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchPlan:
    per_device_batch_size: int
    world_size: int
    micro_global_batch_size: int
    target_global_batch_size: int
    gradient_accumulation_steps: int


def build_batch_plan(
    per_device_batch_size: int,
    world_size: int,
    target_global_batch_size: int = 0,
) -> BatchPlan:
    """Resolve an exact gradient-accumulation plan.

    A target of zero preserves the historical behavior: one optimizer update
    for every distributed micro-batch.
    """
    if per_device_batch_size <= 0:
        raise ValueError("per_device_batch_size must be > 0")
    if world_size <= 0:
        raise ValueError("world_size must be > 0")
    if target_global_batch_size < 0:
        raise ValueError("target_global_batch_size must be >= 0")

    micro_global_batch_size = per_device_batch_size * world_size
    target = target_global_batch_size or micro_global_batch_size
    if target < micro_global_batch_size:
        raise ValueError(
            "target_global_batch_size cannot be smaller than "
            f"bs * world_size ({micro_global_batch_size})"
        )
    if target % micro_global_batch_size != 0:
        raise ValueError(
            "target_global_batch_size must be divisible by bs * world_size: "
            f"{target} % {micro_global_batch_size} != 0"
        )

    return BatchPlan(
        per_device_batch_size=per_device_batch_size,
        world_size=world_size,
        micro_global_batch_size=micro_global_batch_size,
        target_global_batch_size=target,
        gradient_accumulation_steps=target // micro_global_batch_size,
    )


def optimizer_steps_per_epoch(train_samples: int, global_batch_size: int) -> int:
    """Return complete optimizer updates in an epoch-sized sample budget."""
    if train_samples <= 0:
        raise ValueError("train_samples must be > 0")
    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be > 0")
    steps = train_samples // global_batch_size
    if steps == 0:
        raise ValueError(
            "train_samples must contain at least one complete global batch"
        )
    return steps


def freeze_for_oracle_fusion(backbone) -> int:
    """Freeze all parameters except modules named oracle_prior_fusion."""
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    trainable = 0
    for name, parameter in backbone.named_parameters():
        if 'oracle_prior_fusion' in name:
            parameter.requires_grad = True
            trainable += parameter.numel()
    if trainable == 0:
        raise ValueError('No Oracle fusion parameters were created.')
    return trainable


def freeze_for_oracle_adaptation(backbone) -> int:
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    trainable = 0
    oracle_names = ('oracle_prior_fusion', 'oracle_prior_feature_adapter')
    for name, parameter in backbone.named_parameters():
        if any(module_name in name for module_name in oracle_names):
            parameter.requires_grad = True
            trainable += parameter.numel()
    if trainable == 0:
        raise ValueError('No Oracle adaptation parameters were created.')
    return trainable
