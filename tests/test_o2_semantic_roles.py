import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "RLBench"))

from utils.o2_oracle_provider import (  # noqa: E402
    RLBenchGTOracleProvider,
    SceneObjectIndex,
    SemanticRoleMappingError,
    decode_handle_mask,
)
import utils.o2_oracle_provider as oracle_provider_module  # noqa: E402
from utils.rlbench_compat import rgb_handles_to_mask_safe  # noqa: E402


ROLE_CONFIG = ROOT / "finetune" / "RLBench" / "configs" / "rlbench_o2_semantic_roles.yaml"


class FakeObject:
    def __init__(
        self, name, handle, position=(0.0, 0.0, 0.0), children=(),
        bounding_box=None, matrix=None,
    ):
        self.name = name
        self.handle = handle
        self.position = position
        self.children = list(children)
        self.bounding_box = bounding_box
        self.matrix = matrix

    def get_name(self):
        return self.name

    def get_handle(self):
        return self.handle

    def get_position(self):
        return self.position

    def get_bounding_box(self):
        if self.bounding_box is None:
            raise NotImplementedError("bounding box unavailable")
        return self.bounding_box

    def get_matrix(self):
        if self.matrix is None:
            raise NotImplementedError("matrix unavailable")
        return self.matrix

    def get_objects_in_tree(self, exclude_base=True, first_generation_only=False):
        output = [] if exclude_base else [self]
        for child in self.children:
            output.append(child)
            output.extend(child.get_objects_in_tree(exclude_base=True))
        return output


class UnhashableFakeObject(FakeObject):
    __hash__ = None

    def __eq__(self, other):
        return (
            isinstance(other, UnhashableFakeObject)
            and self.handle == other.handle
        )


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


def observation(handles, gripper_open=0.0, gripper_pose=None):
    mask = np.asarray(handles, dtype=np.int64)
    rows, cols = np.indices(mask.shape)
    cloud = np.stack((cols, rows, np.ones_like(rows)), axis=-1).astype(np.float32)
    if gripper_pose is None:
        gripper_pose = np.asarray(
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32
        )
    return SimpleNamespace(
        front_mask=mask,
        front_point_cloud=cloud,
        front_rgb=np.zeros((*mask.shape, 3), dtype=np.uint8),
        gripper_open=gripper_open,
        gripper_pose=np.asarray(gripper_pose, dtype=np.float32),
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
    assert config["schema_version"] == "rlbench_o2_semantic_roles_v2"
    assert config["site_geometry_defaults"] == {
        "primitive": "box_volume",
        "fallback_extent_m": [0.02, 0.02, 0.02],
    }
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
    strategies = {
        task: spec["demo_phase"]["strategy"]
        for task, spec in config["tasks"].items()
    }
    assert set(strategies) == set(config["tasks"])
    assert {
        task for task, strategy in strategies.items()
        if strategy == "release_cycles"
    } == {"place_cups", "stack_blocks", "stack_cups"}
    assert strategies["push_buttons"] == "ordered_target_contact"
    assert sum(strategy == "single_success" for strategy in strategies.values()) == 14


def test_provider_rejects_legacy_v1_role_config(tmp_path):
    config_path = tmp_path / "roles.yaml"
    config_path.write_text(
        "schema_version: rlbench_o2_semantic_roles_v1\ntasks: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected.*v2"):
        RLBenchGTOracleProvider(config_path)


def test_online_site_sampling_uses_oriented_region_points():
    rotation = np.asarray([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [0.5, -0.25, 1.0]
    site = FakeObject(
        "site", 9,
        position=(0.5, -0.25, 1.0),
        bounding_box=[-0.1, 0.1, -0.2, 0.2, 0.0, 0.0],
        matrix=transform,
    )
    value = RLBenchGTOracleProvider(
        ROLE_CONFIG, cameras=("front",), num_points=16, strict=True)
    entity = value._entity_site("test_site", site)

    points, valid = value._sample_entity_points(entity, {}, {})

    assert valid
    assert entity.site_geometry.source == "object_bbox"
    assert len(np.unique(points, axis=0)) > 1
    local = (
        points - entity.site_geometry.center_world
    ) @ entity.site_geometry.rotation_world
    assert np.all(
        np.abs(local) <= entity.site_geometry.extent / 2.0 + 1e-6)


def test_site_role_can_override_only_the_fallback_extent():
    value = RLBenchGTOracleProvider(
        ROLE_CONFIG, cameras=("front",), num_points=8, strict=True)
    site = FakeObject("site", 9, position=(-0.5, 0.25, 1.0))

    entity = value._entity_site(
        "test_site", site,
        {"primitive": "box_volume", "fallback_extent_m": [0.04, 0.06, 0.08]},
    )

    assert entity.site_geometry.source == "fallback_box"
    np.testing.assert_allclose(
        entity.site_geometry.extent, [0.04, 0.06, 0.08])
    with pytest.raises(SemanticRoleMappingError, match="primitive"):
        value._entity_site(
            "test_site", site, {"primitive": "sphere_volume"})


def test_object_mask_sampling_path_is_unchanged():
    drawer = FakeObject("drawer_bottom", 31)
    value = provider("open_drawer", FakeTask([drawer]))

    output = value.enrich(observation([[31, 31], [0, 0]]), {})

    points = output["oracle_target_object_points"]
    expected = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    assert output["oracle_target_object_valid"]
    assert points.shape == (8, 3)
    assert all(np.any(np.all(point == expected, axis=1)) for point in points)


def test_decode_rgb_handle_mask_does_not_mutate_read_only_input():
    image = np.asarray([[[1, 2, 3]]], dtype=np.uint8)
    image.setflags(write=False)
    decoded = decode_handle_mask(image)
    assert int(decoded[0, 0]) == 1 + 256 * 2 + 65536 * 3


@pytest.mark.parametrize(
    "image",
    (
        np.asarray([[[1, 2, 3], [255, 0, 0]]], dtype=np.uint8),
        np.asarray(
            [[[1 / 255.0, 2 / 255.0, 3 / 255.0], [1.0, 0.0, 0.0]]],
            dtype=np.float32,
        ),
    ),
)
def test_safe_rlbench_mask_decoder_supports_numpy2_and_read_only_input(image):
    original = image.copy()
    image.setflags(write=False)
    decoded = rgb_handles_to_mask_safe(image)
    np.testing.assert_array_equal(
        decoded,
        np.asarray([[1 + 256 * 2 + 65536 * 3, 255]], dtype=np.int64),
    )
    np.testing.assert_array_equal(image, original)


def test_scene_object_index_deduplicates_unhashable_pyrep_shapes_by_handle():
    shape = UnhashableFakeObject("stack_blocks_target_plane", 17)
    index = SceneObjectIndex(FakeTask([shape]))
    assert index.find("stack_blocks_target_plane") == [shape]
    assert index.require_any(
        ("stack_blocks_target_plane", "stack_blocks_target_plane"),
        "stack target plane",
    ) == [shape]


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


@pytest.mark.parametrize("variation", range(4))
def test_slide_block_reference_comes_from_registered_success_detector(
    variation, tmp_path
):
    block = FakeObject("block", 10)
    sites = [
        FakeObject(f"success{index + 1}", 20 + index,
                   position=(float(index), 0.0, 0.0))
        for index in range(4)
    ]
    task = FakeTask([block, *sites])
    task.block = block
    task._success_conditions = [
        SimpleNamespace(_detector=sites[variation])
    ]
    with ROLE_CONFIG.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["tasks"]["slide_block_to_color_target"]["reference"][
        "site_geometry"
    ] = {"fallback_extent_m": [0.04, 0.06, 0.08]}
    config_path = tmp_path / "roles.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    value = RLBenchGTOracleProvider(
        config_path, cameras=("front",), num_points=8, strict=True)
    value.reset(
        SimpleNamespace(_task=task),
        "slide_block_to_color_target",
        variation,
        variation,
    )

    assignment = value._build_assignment()

    assert assignment.target.semantic_name == "block"
    assert assignment.target.handles == (10,)
    assert assignment.reference.semantic_name == "color_target"
    assert assignment.reference.kind == "site"
    assert assignment.reference.handles == ()
    np.testing.assert_allclose(
        assignment.reference.site_position, sites[variation].position)
    assert assignment.reference.site_geometry.source == "fallback_box"
    np.testing.assert_allclose(
        assignment.reference.site_geometry.extent, [0.04, 0.06, 0.08])


def test_retry_discards_failed_manifest_attempt_before_restarting(tmp_path):
    lid = FakeObject("jar_lid0", 11)
    jar0 = FakeObject("jar0", 21)
    jar1 = FakeObject("jar1", 22)
    task = FakeTask([lid, jar0, jar1])
    task.lid = lid
    task.jars = [jar0, jar1]
    value = provider("close_jar", task, tmp_path)
    value.set_expected_sample_frames([10])
    value.set_sample_frame(10)
    value.enrich(observation([[11, 21]]), {})

    value.reset(
        SimpleNamespace(_task=task),
        "close_jar",
        0,
        0,
        discard_current=True,
        generation_attempt=2,
    )

    assert ("close_jar", 0) not in value._manifests
    assert value._entries == []
    assert value.stats["discarded_attempts"] == 1

    value.set_expected_sample_frames([10])
    value.set_sample_frame(10)
    value.enrich(observation([[11, 21]]), {})
    value.dump(tmp_path / "retry_dump")
    manifest = json.loads(
        (tmp_path / "retry_dump" / "semantic_role_manifests" /
         "close_jar" / "episode_0.json").read_text(encoding="utf-8")
    )
    assert manifest["generation_attempt"] == 2
    assert len(manifest["entries"]) == 1


def test_failed_demo_event_attempt_is_not_dumped_as_partial_manifest(tmp_path):
    lid = FakeObject('jar_lid0', 11)
    jar0 = FakeObject('jar0', 21)
    jar1 = FakeObject('jar1', 22)
    task = FakeTask([lid, jar0, jar1])
    task.lid = lid
    task.jars = [jar0, jar1]
    value = provider('close_jar', task, tmp_path)
    value.set_expected_sample_frames([10])
    value.set_sample_frame(0)
    value.enrich(observation([[11, 21]]), {})

    value.discard_current_manifest()
    output = tmp_path / 'failed_dump'
    value.dump(output)

    manifest = (
        output / 'semantic_role_manifests' / 'close_jar' / 'episode_0.json')
    assert not manifest.exists()
    assert value.stats['discarded_attempts'] == 1


def test_place_cups_advances_when_detector_condition_is_met_without_release():
    cups = [FakeObject(f"mug{i}", 10 + i) for i in range(3)]
    spokes = [
        FakeObject(f"place_cups_holder_spoke{i}", 20 + i)
        for i in range(3)]
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
    assert value._entries[-1]["phase_id"] == "place_cups:1"
    assert value._entries[-1]["target"]["semantic_name"] == "mug1"


def test_place_cups_reference_does_not_absorb_descendant_spokes():
    cups = [FakeObject(f"mug{i}", 10 + i) for i in range(3)]
    spoke2 = FakeObject("place_cups_holder_spoke2", 22)
    spoke1 = FakeObject("place_cups_holder_spoke1", 21)
    spoke0 = FakeObject(
        "place_cups_holder_spoke0", 20, children=(spoke1, spoke2))
    task = FakeTask(cups + [spoke0])
    # Private task containers are deliberately reversed: semantic roles must
    # still follow the canonical names in the role configuration.
    task._cups = list(reversed(cups))
    task._spokes = [spoke2, spoke1, spoke0]
    task._index = 2
    task._on_peg_conditions = [
        FakeCondition(), FakeCondition(), FakeCondition()]
    value = provider("place_cups", task)

    phase0 = value._build_assignment()
    assert phase0.target.semantic_name == "mug0"
    assert phase0.target.handles == (10,)
    assert phase0.reference.semantic_name == "holder_spoke0"
    assert phase0.reference.handles == (20,)

    value._phase_index = 1
    phase1 = value._build_assignment()
    assert phase1.target.semantic_name == "mug1"
    assert phase1.target.handles == (11,)
    assert phase1.reference.semantic_name == "holder_spoke1"
    assert phase1.reference.handles == (21,)


def test_place_cups_demo_events_build_phase_manifest_without_sim_replay(tmp_path):
    cups = [FakeObject(f"mug{i}", 10 + i) for i in range(3)]
    spokes = [
        FakeObject(f"place_cups_holder_spoke{i}", 20 + i)
        for i in range(3)]
    task = FakeTask(cups + spokes)
    task._cups = cups
    task._spokes = spokes
    task._index = 1
    task._on_peg_conditions = [FakeCondition(), FakeCondition(), FakeCondition()]
    value = provider("place_cups", task, tmp_path)
    demo = [
        observation([[10, 20], [11, 21]], gripper_open=state)
        for state in (1.0, 0.0, 0.0, 1.0, 0.0, 1.0)
    ]
    value.set_sample_frame(0)
    value.enrich(demo[0], {})

    info = value.build_demo_event_manifest(demo, [1, 3, 5])

    assert info["release_frames"] == [3, 5]
    assert [entry["sample_frame"] for entry in value._entries] == [0, 1, 3, 5]
    assert [entry["phase_id"] for entry in value._entries] == [
        "place_cups:0", "place_cups:0", "place_cups:1", "place_cups:1"
    ]
    assert value._entries[2]["phase_advanced"] is True
    assert value._entries[-1]["completion_satisfied"] is True
    assert all(entry["phase_source"] == "demo_events" for entry in value._entries)

    value.dump(tmp_path / "demo_event_dump")
    manifest = json.loads(
        (tmp_path / "demo_event_dump" / "semantic_role_manifests" /
         "place_cups" / "episode_0.json").read_text(encoding="utf-8")
    )
    assert manifest["phase_source"] == "demo_events"
    assert manifest["source_alignment_validated"] is True
    assert manifest["phase_strategy"] == "release_cycles"
    assert manifest["phase_boundary_source"] == "gripper_close_to_open"
    assert manifest["phase_boundary_frames"] == [3, 5]


def test_place_cups_demo_events_recover_missing_final_release_from_keypoints():
    cups = [FakeObject(f"mug{i}", 10 + i) for i in range(3)]
    spokes = [
        FakeObject(f"place_cups_holder_spoke{i}", 20 + i)
        for i in range(3)]
    task = FakeTask(cups + spokes)
    task._cups = cups
    task._spokes = spokes
    task._index = 1
    task._on_peg_conditions = [FakeCondition(), FakeCondition(), FakeCondition()]
    value = provider("place_cups", task)
    masks = [[10, 11], [20, 21]]
    states = (1.0, 0.0, 1.0, 0.0)
    poses = (
        [5., 5., 1., 0., 0., 0., 1.],
        # This approach keypoint is closer than the later real release and must
        # not displace that stronger completion evidence.
        [0., 1., 1., 0., 0., 0., 1.],
        [0., 1., 1., 0., 0., 0., 1.],
        [1., 1., 1., 0., 0., 0., 1.],
    )
    demo = [
        observation(masks, gripper_open=state, gripper_pose=pose)
        for state, pose in zip(states, poses)]
    value.set_sample_frame(0)
    value.enrich(demo[0], {})

    info = value.build_demo_event_manifest(demo, [1, 2, 3])

    assert info["detected_release_frames"] == [2]
    assert info["release_frames"] == [2, 3]
    assert info["release_relation_distances"] == pytest.approx([0., 0.])
    assert info["phase_boundary_source"] == (
        "keypoints_recovered_by_ordered_reference_relation")


def test_place_cups_filters_extra_release_by_ordered_reference_relation():
    cups = [FakeObject(f"mug{i}", 10 + i) for i in range(3)]
    spokes = [
        FakeObject(f"place_cups_holder_spoke{i}", 20 + i)
        for i in range(3)]
    task = FakeTask(cups + spokes)
    task._cups = cups
    task._spokes = spokes
    task._index = 2
    task._on_peg_conditions = [FakeCondition() for _ in range(3)]
    value = provider("place_cups", task)
    masks = [[10, 11, 12], [20, 21, 22]]
    states = (1., 0., 1., 0., 1., 0., 1., 0., 1.)
    release_positions = {
        2: [0., 1., 1., 0., 0., 0., 1.],
        4: [9., 9., 1., 0., 0., 0., 1.],
        6: [1., 1., 1., 0., 0., 0., 1.],
        8: [2., 1., 1., 0., 0., 0., 1.],
    }
    demo = [
        observation(
            masks, gripper_open=state,
            gripper_pose=release_positions.get(
                frame, [0., 0., 1., 0., 0., 0., 1.]))
        for frame, state in enumerate(states)]
    # A placed cup can fully occlude its thin spoke at the release frame. The
    # fixed reference location must still be recoverable from frame 0.
    for frame in release_positions:
        demo[frame].front_mask[1, :] = 0
    value.set_sample_frame(0)
    value.enrich(demo[0], {})

    info = value.build_demo_event_manifest(demo, [2, 4, 6, 8])

    assert info["detected_release_frames"] == [2, 4, 6, 8]
    assert info["release_frames"] == [2, 6, 8]
    assert info["release_relation_distances"] == pytest.approx([0., 0., 0.])
    assert info["phase_boundary_source"] == (
        "keypoints_recovered_by_ordered_reference_relation")


def test_single_phase_demo_events_support_non_gripper_task():
    drawer = FakeObject("drawer_bottom", 31)
    task = FakeTask([drawer])
    value = provider("open_drawer", task)
    demo = [observation([[31]], gripper_open=1.0) for _ in range(3)]
    value.set_sample_frame(0)
    value.enrich(demo[0], {})

    info = value.build_demo_event_manifest(demo, [1, 2])

    assert info["phase_strategy"] == "single_success"
    assert info["phase_boundary_frames"] == [2]
    assert [entry["sample_frame"] for entry in value._entries] == [0, 1, 2]
    assert [entry["phase_id"] for entry in value._entries] == [
        "open_drawer:0", "open_drawer:0", "open_drawer:0"
    ]
    assert not value._entries[1]["completion_satisfied"]
    assert value._entries[2]["completion_satisfied"]


@pytest.mark.parametrize("failure,reason", [
    ("handles", "live_role_handles_absent_from_stored_masks"),
    ("mask", "stored_masks_missing"),
    ("cloud", "matching_mask_pixels_but_no_finite_point_cloud"),
    ("nan", "matching_mask_pixels_but_no_finite_point_cloud"),
])
def test_demo_initial_validation_distinguishes_source_failures(failure, reason):
    lid = FakeObject("jar_lid0", 11)
    jars = [FakeObject("jar0", 21), FakeObject("jar1", 22)]
    task = FakeTask([lid, *jars])
    task.lid, task.jars = lid, jars
    value = provider("close_jar", task)
    value.set_sample_frame(0)
    value.enrich(observation([[11, 21]]), {})
    initial = list(value._entries)
    stored = observation([[11, 21]])
    if failure == "handles":
        stored.front_mask = np.array([[111, 121]])
    elif failure == "mask":
        stored.front_mask = None
    elif failure == "cloud":
        stored.front_point_cloud = None
    else:
        stored.front_point_cloud[:] = np.nan
    with pytest.raises(SemanticRoleMappingError, match=reason) as error:
        value.build_demo_event_manifest([stored, stored], [1])
    assert '"live_handles": [11]' in str(error.value)
    assert '"front"' in str(error.value)
    assert value._entries == initial
    assert not value._source_alignment_validated


def test_stack_cups_demo_events_use_two_release_cycles():
    cups = [FakeObject(name, handle) for name, handle in (
        ("cup1", 11), ("cup2", 12), ("cup3", 13)
    )]
    task = FakeTask(cups)
    value = provider("stack_cups", task)
    demo = [
        observation([[11, 12], [13, 0]], gripper_open=state)
        for state in (1.0, 0.0, 1.0, 0.0, 1.0)
    ]
    value.set_sample_frame(0)
    value.enrich(demo[0], {})

    info = value.build_demo_event_manifest(demo, [1, 2, 3, 4])

    assert info["phase_strategy"] == "release_cycles"
    assert info["phase_boundary_frames"] == [2, 4]
    assert value._entries[2]["phase_id"] == "stack_cups:1"
    assert value._entries[2]["phase_advanced"] is True
    assert value._entries[-1]["completion_satisfied"] is True


def test_push_buttons_demo_events_locate_ordered_gt_target_contacts():
    plates = [
        FakeObject(
            f"target_button_topPlate{i}", 10 + i,
            position=(float(i), 0.0, 1.0),
        )
        for i in range(3)
    ]
    task = FakeTask(plates)
    task.target_topPlates = plates
    task.buttons_to_push = 2
    value = provider("push_buttons", task)
    poses = (
        [0.5, 0.5, 1.0, 0, 0, 0, 1],
        [0.0, 0.0, 1.0, 0, 0, 0, 1],
        [0.5, 0.5, 1.0, 0, 0, 0, 1],
        [1.0, 0.0, 1.0, 0, 0, 0, 1],
    )
    demo = [
        observation([[10, 11], [12, 0]], gripper_open=1.0, gripper_pose=pose)
        for pose in poses
    ]
    value.set_sample_frame(0)
    value.enrich(demo[0], {})

    assignment = value._build_assignment()
    assert assignment.target.kind == "site"
    assert assignment.target.semantic_name == "button0_contact_site"
    assert assignment.target.handles == ()
    np.testing.assert_allclose(assignment.target.site_position, [0.0, 0.0, 1.0])
    assert assignment.target.site_geometry.source == "fallback_box"

    info = value.build_demo_event_manifest(demo, [1, 3])

    assert info["phase_strategy"] == "ordered_target_contact"
    assert info["phase_boundary_frames"] == [1, 3]
    assert info["contact_distances"] == pytest.approx([0.0, 0.0])
    assert [entry["phase_id"] for entry in value._entries] == [
        "push_buttons:0", "push_buttons:1", "push_buttons:1"
    ]
    assert value._entries[1]["phase_advanced"] is True
    assert value._entries[-1]["completion_satisfied"] is True


def test_push_buttons_resolver_version_invalidates_legacy_manifests():
    assert RLBenchGTOracleProvider.task_resolver_version("push_buttons") == (
        "push_buttons_contact_site_v2")
    assert RLBenchGTOracleProvider.task_resolver_version("place_cups") == (
        "place_cups_detector_relation_v4")
    assert RLBenchGTOracleProvider.task_resolver_version("close_jar") is None


@pytest.mark.parametrize(
    "variation,option", tuple(enumerate(("bottom", "middle", "top")))
)
def test_put_item_in_drawer_reference_is_registered_success_site(
    variation, option
):
    item = FakeObject("item", 10)
    detector = FakeObject(
        f"success_{option}", 20, position=(0.1, float(variation), 0.75))
    task = FakeTask([item, detector])
    task._item = item
    task._success_conditions = [SimpleNamespace(_detector=detector)]
    value = RLBenchGTOracleProvider(
        ROLE_CONFIG, cameras=("front",), num_points=8, strict=True)
    value.reset(
        SimpleNamespace(_task=task), "put_item_in_drawer",
        variation, variation)

    assignment = value._build_assignment()

    assert assignment.target.semantic_name == "item"
    assert assignment.target.kind == "object"
    assert assignment.target.handles == (10,)
    assert assignment.reference.semantic_name == f"{option}_drawer_success_site"
    assert assignment.reference.kind == "site"
    assert assignment.reference.handles == ()
    np.testing.assert_allclose(
        assignment.reference.site_position, detector.position)
    assert RLBenchGTOracleProvider.task_resolver_version(
        "put_item_in_drawer") == "put_item_in_drawer_success_site_v2"


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


@pytest.mark.parametrize('mode', ['verified', 'mask_verified'])
def test_verified_demo_manifest_uses_saved_handles_and_reset_restores_live(tmp_path, mode):
    lid, jar0, jar1 = (FakeObject("jar_lid0", 87),
                       FakeObject("jar0", 88), FakeObject("jar1", 89))
    task = FakeTask([lid, jar0, jar1])
    task.lid, task.jars = lid, [jar0, jar1]
    value = RLBenchGTOracleProvider(
        ROLE_CONFIG, cameras=("front", "left_shoulder"), num_points=8,
        handle_alignment=mode, alignment_output_dir=tmp_path,
        manifest_output_dir=tmp_path / "output")
    value.reset(SimpleNamespace(_task=task), "close_jar", 0, 0)
    mask = np.full((8, 8), 87)
    mask[:, 4:] = 88
    live, stored = observation(mask), observation(mask)
    stored.front_mask = np.where(mask == 87, 99, 93)
    for obs in (live, stored):
        obs.left_shoulder_mask = obs.front_mask.copy()
        obs.left_shoulder_point_cloud = obs.front_point_cloud.copy()
        obs.misc = {
            f"{cam}_camera_{kind}": np.eye(size)
            for cam in value.cameras
            for kind, size in (("intrinsics", 3), ("extrinsics", 4))
        }
    if mode == 'mask_verified':
        for cam in value.cameras:
            getattr(stored, f'{cam}_point_cloud')[..., 2] += .016
    value.set_sample_frame(0)
    value.enrich(live, {})
    value.build_demo_event_manifest([stored, stored], [1])
    assert value._entries[0]["target"]["handles"] == [99]
    assert value._entries[0]["reference"]["handles"] == [93]
    assert value._entries[0]["target_valid"]
    assert value._demo_phase_metadata["handle_namespace"] == "stored"
    report = json.loads((tmp_path / "close_jar" / "episode_0.json").read_text())
    assert report["status"] == mode
    if mode == 'mask_verified':
        assert report['geometry_verified'] is False
    assert report["live_to_stored"] == {"87": 99, "88": 93}
    manifest_path = (tmp_path / "output" / "semantic_role_manifests"
                     / "close_jar" / "episode_0.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest['handle_alignment']['status'] == mode
    assert manifest["entries"][0]["target"]["handles"] == [99]
    assert set(manifest["source_frame0_masks"]) == set(value.cameras)
    value.reset(SimpleNamespace(_task=task), "close_jar", 0, 1)
    assert value._stored_handle_map is None
    assert value._build_assignment().target.handles == (87,)


def test_demo_manifest_uses_raw_png_masks_when_loaded_demo_masks_differ(tmp_path):
    lid = FakeObject('jar_lid0', 87)
    jar0 = FakeObject('jar0', 88)
    jar1 = FakeObject('jar1', 89)
    task = FakeTask([lid, jar0, jar1])
    task.lid, task.jars = lid, [jar0, jar1]

    raw_root = tmp_path / 'raw' / 'train'
    episode_dir = (
        raw_root / 'close_jar' / 'all_variations' / 'episodes' / 'episode0')
    raw_mask = np.full((8, 8), 99, dtype=np.int64)
    raw_mask[:, 4:] = 93
    encoded = np.stack((
        raw_mask & 255,
        (raw_mask >> 8) & 255,
        (raw_mask >> 16) & 255,
    ), axis=-1).astype(np.uint8)
    for camera in ('front', 'left_shoulder'):
        mask_dir = episode_dir / f'{camera}_mask'
        mask_dir.mkdir(parents=True)
        for frame in (0, 1):
            Image.fromarray(encoded).save(mask_dir / f'{frame}.png')

    value = RLBenchGTOracleProvider(
        ROLE_CONFIG,
        cameras=('front', 'left_shoulder'),
        num_points=8,
        handle_alignment='verified',
        raw_data_root=raw_root,
        alignment_output_dir=tmp_path / 'alignment',
        manifest_output_dir=tmp_path / 'output',
    )
    value.reset(SimpleNamespace(_task=task), 'close_jar', 0, 0)
    live_mask = np.full((8, 8), 87, dtype=np.int64)
    live_mask[:, 4:] = 88
    live = observation(live_mask)
    stored = observation(np.full((8, 8), 7, dtype=np.int64))
    for obs in (live, stored):
        obs.left_shoulder_mask = obs.front_mask.copy()
        obs.left_shoulder_point_cloud = obs.front_point_cloud.copy()
        obs.misc = {
            f'{camera}_camera_{kind}': np.eye(size)
            for camera in value.cameras
            for kind, size in (('intrinsics', 3), ('extrinsics', 4))
        }

    value.set_sample_frame(0)
    value.enrich(live, {})
    value.build_demo_event_manifest([stored, stored], [1])

    expected = value._mask_fingerprints({
        'front': raw_mask,
        'left_shoulder': raw_mask,
    })
    manifest_path = (
        tmp_path / 'output' / 'semantic_role_manifests'
        / 'close_jar' / 'episode_0.json')
    manifest = json.loads(manifest_path.read_text())
    assert manifest['source_frame0_masks'] == expected
    assert manifest['demo_frame0_masks'] != expected
    assert manifest['raw_demo_frame0_masks_match'] is False
    assert manifest['source_mask_origin'] == 'raw_png'
    assert Path(manifest['raw_mask_source']) == episode_dir.resolve()
    assert manifest['entries'][0]['target']['handles'] == [99]
    assert manifest['entries'][0]['reference']['handles'] == [93]
    assert manifest['entries'][0]['target_valid'] is True


def test_verified_mapping_checks_future_phase_before_generating_any_entries(tmp_path):
    cups = [FakeObject(name, h) for name, h in (("cup1", 11), ("cup2", 12), ("cup3", 13))]
    value = RLBenchGTOracleProvider(
        ROLE_CONFIG, cameras=("front", "left_shoulder"), handle_alignment="verified",
        alignment_output_dir=tmp_path)
    value.reset(SimpleNamespace(_task=FakeTask(cups)), "stack_cups", 0, 0)
    obs = observation(np.tile([11, 12], (20, 20)), gripper_open=1.)
    obs.left_shoulder_mask = obs.front_mask.copy()
    obs.left_shoulder_point_cloud = obs.front_point_cloud.copy()
    obs.misc = {
        f"{cam}_camera_{kind}": np.eye(size)
        for cam in value.cameras
        for kind, size in (("intrinsics", 3), ("extrinsics", 4))
    }
    value.set_sample_frame(0)
    value.enrich(obs, {})
    with pytest.raises(SemanticRoleMappingError, match="cup3"):
        value.build_demo_event_manifest([obs, obs], [1])
    assert len(value._entries) == 1
    assert not value._source_alignment_validated
    report = json.loads((tmp_path / "stack_cups" / "episode_0.json").read_text())
    assert report["status"] == "failed"


def test_mask_verified_fallback_keeps_individual_entity_certificates(
        monkeypatch, tmp_path):
    lid = FakeObject("jar_lid0", 87)
    jars = [FakeObject("jar0", 88), FakeObject("jar1", 89)]
    task = FakeTask([lid, *jars])
    task.lid, task.jars = lid, jars
    value = RLBenchGTOracleProvider(
        ROLE_CONFIG, cameras=("front", "left_shoulder"), num_points=8,
        handle_alignment="mask_verified", alignment_output_dir=tmp_path)
    value.reset(SimpleNamespace(_task=task), "close_jar", 0, 0)
    obs = observation(np.full((8, 8), 99))
    obs.left_shoulder_mask = obs.front_mask.copy()
    obs.left_shoulder_point_cloud = obs.front_point_cloud.copy()
    obs.misc = {
        f"{cam}_camera_{kind}": np.eye(size)
        for cam in value.cameras
        for kind, size in (("intrinsics", 3), ("extrinsics", 4))
    }

    def fake_align(_live, _stored, names, *_args, **_kwargs):
        handles = set(names)
        if handles == {87}:
            return {87: 199}, {"87": {"stored_handle": 199}}
        raise oracle_provider_module.HandleAlignmentError(
            "individual failure", {"attempted_handles": sorted(handles)})

    def fake_union(_live, _stored, handles, semantic_name):
        assert handles == {88}
        assert semantic_name == "target_jar"
        return (193,), {
            "source": "semantic_entity_union_mask_overlap",
            "semantic_name": semantic_name,
            "live_handles": [88],
            "stored_handles": [193],
            "views": {"front": {"passed": True},
                      "left_shoulder": {"passed": True}},
        }

    monkeypatch.setattr(oracle_provider_module, "align_handles", fake_align)
    monkeypatch.setattr(
        oracle_provider_module, "align_semantic_handle_group", fake_union)
    translated = value._prepare_stored_handles(obs, {
        "target": {"kind": "object", "handles": [87]},
        "reference": {"kind": "object", "handles": [88]},
    })

    assert translated["target"]["handles"] == [199]
    assert translated["reference"]["handles"] == [193]
    report = json.loads(
        (tmp_path / "close_jar" / "episode_0.json").read_text())
    assert report["alignment_scope"] == "mixed_entity_certificates"
    assert report["live_to_stored"] == {"87": 199}
    assert report["semantic_entity_to_stored"] == {
        "87": [199], "88": [193]}
    assert report["evidence"]["entity_certificates"]["87"][
        "source"] == "individual_handles"
    assert report["evidence"]["entity_certificates"]["88"][
        "source"] == "semantic_entity_union_mask_overlap"


def test_mask_verified_failure_keeps_individual_and_union_evidence(
        monkeypatch, tmp_path):
    lid = FakeObject("jar_lid0", 87)
    jars = [FakeObject("jar0", 88), FakeObject("jar1", 89)]
    task = FakeTask([lid, *jars])
    task.lid, task.jars = lid, jars
    value = RLBenchGTOracleProvider(
        ROLE_CONFIG, cameras=("front", "left_shoulder"), num_points=8,
        handle_alignment="mask_verified", alignment_output_dir=tmp_path)
    value.reset(SimpleNamespace(_task=task), "close_jar", 0, 0)
    obs = observation(np.full((8, 8), 99))
    obs.left_shoulder_mask = obs.front_mask.copy()
    obs.left_shoulder_point_cloud = obs.front_point_cloud.copy()
    obs.misc = {
        f"{cam}_camera_{kind}": np.eye(size)
        for cam in value.cameras
        for kind, size in (("intrinsics", 3), ("extrinsics", 4))
    }

    def fail_individual(_live, _stored, names, *_args, **_kwargs):
        raise oracle_provider_module.HandleAlignmentError(
            "individual failure", {"attempted_handles": sorted(names)})

    def fail_union(_live, _stored, handles, semantic_name):
        raise oracle_provider_module.HandleAlignmentError(
            "union failure", {
                "semantic_name": semantic_name,
                "live_handles": sorted(handles)})

    monkeypatch.setattr(
        oracle_provider_module, "align_handles", fail_individual)
    monkeypatch.setattr(
        oracle_provider_module, "align_semantic_handle_group", fail_union)
    with pytest.raises(SemanticRoleMappingError):
        value._prepare_stored_handles(obs, {
            "target": {"kind": "object", "handles": [87]},
            "reference": {"kind": "object", "handles": [88]},
        })

    report = json.loads(
        (tmp_path / "close_jar" / "episode_0.json").read_text())
    evidence = report["evidence"]["entities"]["jar_lid"]
    assert evidence["individual_alignment_error"] == "individual failure"
    assert evidence["individual_alignment_evidence"] == {
        "attempted_handles": [87]}
    assert evidence["semantic_name"] == "jar_lid"


def test_reach_and_drag_uses_color_target_site_without_mask_mapping(tmp_path):
    stick = FakeObject('stick', 101)
    target = FakeObject('target0', 102, position=(.2, -.1, .75))
    task = FakeTask([stick, target])
    task.stick, task.target = stick, target
    value = RLBenchGTOracleProvider(
        ROLE_CONFIG, cameras=('front', 'left_shoulder'), num_points=8,
        handle_alignment='mask_verified', alignment_output_dir=tmp_path,
        manifest_output_dir=tmp_path / 'output')
    value.reset(SimpleNamespace(_task=task), 'reach_and_drag', 13, 0)
    mask = np.full((8, 8), 101)
    live, stored = observation(mask), observation(mask)
    stored.front_mask = np.full_like(mask, 201)
    for obs in (live, stored):
        obs.left_shoulder_mask = obs.front_mask.copy()
        obs.left_shoulder_point_cloud = obs.front_point_cloud.copy()
        obs.misc = {
            f'{cam}_camera_{kind}': np.eye(size)
            for cam in value.cameras
            for kind, size in (('intrinsics', 3), ('extrinsics', 4))
        }
    value.set_sample_frame(0)
    value.enrich(live, {})
    value.build_demo_event_manifest([stored, stored], [1])
    entry = value._entries[0]
    assert entry['target']['handles'] == [201]
    assert entry['reference']['kind'] == 'site'
    assert entry['reference']['handles'] == []
    assert entry['reference']['site_position'] == pytest.approx([.2, -.1, .75])
    assert entry['reference']['site_geometry']['source'] == 'fallback_box'
    assert entry['reference']['site_geometry']['extent'] == pytest.approx(
        [.02, .02, .02])
    report = json.loads(
        (tmp_path / 'reach_and_drag' / 'episode_0.json').read_text())
    assert set(report['live_to_stored']) == {'101'}
    assert '102' not in report['evidence']
