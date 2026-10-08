"""Resolved configurations are persisted and checked on checkpoint resume."""
from pathlib import Path
from .data.common import read_config


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
