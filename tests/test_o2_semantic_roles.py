import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "RLBench"))

from utils.o2_oracle_provider import (  # noqa: E402
    RLBenchGTOracleProvider,
    SemanticRoleMappingError,
    decode_handle_mask,
)


ROLE_CONFIG = ROOT / "finetune" / "RLBench" / "configs" / "rlbench_o2_semantic_roles.yaml"


class FakeObject:
    def __init__(self, name, handle, position=(0.0, 0.0, 0.0), children=()):
        self.name = name
        self.handle = handle
        self.position = position
        self.children = list(children)

    def get_name(self):
        return self.name

    def get_handle(self):
        return self.handle

    def get_position(self):
        return self.position

    def get_objects_in_tree(self, exclude_base=True, first_generation_only=False):
        output = [] if exclude_base else [self]
        for child in self.children:
            output.append(child)
            output.extend(child.get_objects_in_tree(exclude_base=True))
        return output


class FakeCondition:
    def __init__(self, met=False):
        self.met = met

    def condition_met(self):
        return self.met, False


class FakeTask:
    def __init__(self, objects):
        self.base = FakeObject("base", 1, children=objects)

    def get_base(self):
        return self.base

    def success(self):
        return False, False


def observation(handles, gripper_open=0.0):
    mask = np.asarray(handles, dtype=np.int64)
    rows, cols = np.indices(mask.shape)
    cloud = np.stack((cols, rows, np.ones_like(rows)), axis=-1).astype(np.float32)
    return SimpleNamespace(
        front_mask=mask,
        front_point_cloud=cloud,
        front_rgb=np.zeros((*mask.shape, 3), dtype=np.uint8),
        gripper_open=gripper_open,
    )


def provider(task_name, task, tmp_path=None, strict=True):
    value = RLBenchGTOracleProvider(
        ROLE_CONFIG,
        cameras=("front",),
        num_points=8,
        strict=strict,
        debug_root=tmp_path,
    )
    value.reset(SimpleNamespace(_task=task), task_name, 0, 0)
    return value


def test_role_config_covers_exact_bridgevla_18_tasks():
    with ROLE_CONFIG.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert config["schema_version"] == "rlbench_o2_semantic_roles_v1"
    assert len(config["tasks"]) == 18
    assert set(config["tasks"]) == {
        "close_jar", "insert_onto_square_peg", "light_bulb_in",
        "meat_off_grill", "open_drawer", "place_cups",
        "place_shape_in_shape_sorter", "place_wine_at_rack_location",
        "push_buttons", "put_groceries_in_cupboard", "put_item_in_drawer",
        "put_money_in_safe", "reach_and_drag",
        "slide_block_to_color_target", "stack_blocks", "stack_cups",
        "sweep_to_dustpan_of_size", "turn_tap",
    }


def test_decode_rgb_handle_mask_does_not_mutate_read_only_input():
    image = np.asarray([[[1, 2, 3]]], dtype=np.uint8)
    image.setflags(write=False)
    decoded = decode_handle_mask(image)
    assert int(decoded[0, 0]) == 1 + 256 * 2 + 65536 * 3


def test_close_jar_merges_lid_children_and_selects_variation_jar(tmp_path):
    lid_visual = FakeObject("jar_lid_visual", 12)
    lid = FakeObject("jar_lid0", 11, children=(lid_visual,))
    jar0 = FakeObject("jar0", 21)
    jar1 = FakeObject("jar1", 22)
    task = FakeTask([lid, jar0, jar1])
    task.lid = lid
    task.jars = [jar0, jar1]
    value = provider("close_jar", task, tmp_path)
    output = value.enrich(observation([[11, 12], [21, 0]]), {})
    assert output["oracle_target_object_valid"]
    assert output["oracle_reference_object_valid"]
    assert value._entries[0]["target"]["handles"] == [11, 12]
    assert value._entries[0]["reference"]["handles"] == [21]
    value.dump(tmp_path / "dump")
    manifest = json.loads(
        (tmp_path / "dump" / "semantic_role_manifests" / "close_jar" /
         "episode_0.json").read_text(encoding="utf-8")
    )
    assert manifest["entries"][0]["phase_id"] == "close_jar:0"


def test_place_cups_advances_only_after_condition_and_release():
    cups = [FakeObject(f"mug{i}", 10 + i) for i in range(3)]
    spokes = [FakeObject(f"spoke{i}", 20 + i) for i in range(3)]
    task = FakeTask(cups + spokes)
    task._cups = cups
    task._spokes = spokes
    task._index = 2
    task._on_peg_conditions = [FakeCondition(), FakeCondition(), FakeCondition()]
    value = provider("place_cups", task)
    value.enrich(observation([[10, 20], [11, 21]], gripper_open=1.0), {})
    assert value._entries[-1]["phase_id"] == "place_cups:0"
    task._on_peg_conditions[0].met = True
    value.enrich(observation([[10, 20], [11, 21]], gripper_open=0.0), {})
    assert value._entries[-1]["phase_id"] == "place_cups:0"
    value.enrich(observation([[10, 20], [11, 21]], gripper_open=1.0), {})
    assert value._entries[-1]["phase_id"] == "place_cups:1"
    assert value._entries[-1]["target"]["semantic_name"] == "mug1"


def test_open_drawer_has_no_reference_and_is_not_mapping_error():
    drawer = FakeObject("drawer_bottom", 31)
    task = FakeTask([drawer])
    value = provider("open_drawer", task)
    output = value.enrich(observation([[31, 0], [0, 0]]), {})
    assert output["oracle_target_object_valid"]
    assert not output["oracle_reference_object_valid"]
    assert value.stats["no_reference"] == 1
    assert value.stats["mapping_errors"] == 0


def test_strict_reset_rejects_missing_semantic_selector():
    task = FakeTask([])
    with pytest.raises(SemanticRoleMappingError):
        provider("open_drawer", task)
