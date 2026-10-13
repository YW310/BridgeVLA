"""Resolved configurations are persisted and checked on checkpoint resume."""
from pathlib import Path
from .data.common import read_config
from .action_modes import resolve_action


def merge(first, second):
    result = dict(first)
    for key, value in second.items():
        result[key] = merge(result[key], value) if isinstance(result.get(key), dict) and isinstance(value, dict) else value
    return result


def load(path, visited=None):
    path = Path(path).resolve()
    visited = set() if visited is None else visited
    if path in visited:
        raise ValueError("Cyclic configuration inheritance")
    visited.add(path)
    config = read_config(path)
    parent = config.pop("extends", None)
    if parent:
        config = merge(load(path.parent / parent, visited), config)
    if config.get("mode") not in ("baseline", "role_queries", "predicted_external"):
        raise ValueError("Invalid OHT mode")
    action = resolve_action(config)
    config["action_mode"] = action["mode"]
    config["action"] = {key: value for key, value in action.items() if key != "mode"}
    if action["mode"] != "eef" and config.get("agent", {}).get("transform_augmentation"):
        raise ValueError("Vector actions require transform_augmentation=false: robot states cannot be independently augmented")
    for key in ("batch_size", "accumulation_steps", "optimizer_steps", "checkpoint_interval", "point_count"):
        if not isinstance(config.get(key), int) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config.get("agent", {}).get("collision_loss_weight") != 0:
        raise ValueError("OHT has no collision labels; collision_loss_weight must be 0")
    if config["mode"] == "role_queries" and (
        not config.get("mvt", {}).get("stage_two") or not config["mvt"].get("add_corr")
    ):
        raise ValueError("Role inheritance requires stage_two and XYZ correlation channels")
    return config
