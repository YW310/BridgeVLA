"""Versioned manifests and explicit filesystem boundaries."""
import hashlib
import json
from pathlib import Path

TASKS = ("assemble_left", "assemble_right", "disassemble_left", "disassemble_right")
CAMERAS = ("global_left", "global_right", "local_left", "local_right", "wrist")
GOALS = {
    "assemble_left": "Assemble the left wheel onto the OHT axle from the workbench",
    "assemble_right": "Assemble the right wheel onto the OHT axle from the workbench",
    "disassemble_left": "Disassemble the left wheel from the OHT axle and place it on the plate",
    "disassemble_right": "Disassemble the right wheel from the OHT axle and place it on the plate",
}
SCHEMA = "oht_bridgevla_v2"


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_config(path):
    import yaml
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return config


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def inside(root, relative):
    root = Path(root).resolve()
    value = (root / relative).resolve()
    if not value.is_relative_to(root):
        raise ValueError(f"Path escapes dataset root: {relative}")
    return value


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
