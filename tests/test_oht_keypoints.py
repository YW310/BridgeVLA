"""OHT gripper-only and legacy action endpoints, without RLBench/GPU dependencies."""
import ast
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from finetune.OHT.data.actions import keypoint_options, keypoints
from finetune.OHT.data.common import read_jsonl
from finetune.OHT.data.observation import validate_data_config
from finetune.OHT.data.replay import build, load_contract
from tests.test_oht_migration import replay_fixture


def poses_from_x(values):
    poses = np.zeros((len(values), 7))
    poses[:, 0], poses[:, 6] = values, 1
    return poses


def events(poses, gripper=None, timestamps=None, instruction_ids=None, **options):
    n = len(poses)
    return keypoints(poses, np.ones(n) if gripper is None else gripper, instruction_ids,
                     timestamps=np.arange(n) * .1 if timestamps is None else timestamps,
                     method="bridgevla", **options)


@pytest.mark.parametrize("gripper,expected", [
    ([1, 1, 1, 1], []),
    ([0, 0, 0, 0], []),
    ([1, 1, 0, 0], [2]),  # keep penultimate event, do not add terminal.
    ([1, 1, 1, 0], [3]),  # terminal is a target only if the gripper changes.
    ([1, 0, 0, 1, 1, 1], [1, 3]),
    ([1, 1, 0, 1], [2, 3]),  # consecutive real changes must both survive.
    ([1, 0], [1]),
    ([0, 1], [1]),
    ([1, 1], []),
])
def test_gripper_mode_keeps_only_real_changes_without_terminal_padding(gripper, expected):
    assert keypoints(poses_from_x(np.zeros(len(gripper))), gripper, method="gripper") == expected
    assert keypoint_options(dict(method="gripper")) == dict(method="gripper")


def test_gripper_mode_ignores_motion_stops_rotation_and_instruction_changes():
    poses = poses_from_x([0, .1, .2, .2, .2, .2, 1., 2.])
    poses[:, 3:] = Rotation.from_euler("z", (np.arange(8) * 20)[:, None], degrees=True).as_quat()
    assert keypoints(poses, [1, 1, 0, 0, 0, 0, 1, 1], np.arange(8),
                     timestamps=np.arange(8) * .1, method="gripper") == [2, 6]


def test_continuous_motion_does_not_create_periodic_or_instruction_keypoints():
    poses = poses_from_x(np.arange(100) * .05)
    assert events(poses, instruction_ids=np.arange(100)) == [99]


def test_stops_gripper_changes_and_terminal_are_distinct_events():
    poses = poses_from_x([0, .1, .2, .3, .3, .3, .4, .5, .6, .7, .8, .9, 1., 1.1])
    gripper = [1] * 8 + [0] * 6
    assert events(poses, gripper) == [4, 8, 13]


def test_stopping_requires_rotation_to_stop_as_well_as_translation():
    poses = poses_from_x(np.zeros(10))
    poses[:, 3:] = Rotation.from_euler("z", (np.arange(10) * 10)[:, None], degrees=True).as_quat()
    assert events(poses) == [9]


def test_quaternion_sign_changes_are_not_rotation_motion():
    poses = poses_from_x([0, .1, .2, .3, .3, .3, .4, .5, .6])
    poses[::2, 3:] *= -1
    assert events(poses) == [4, 8]


def test_stopping_speed_uses_actual_irregular_dt_not_per_frame_displacement():
    poses = poses_from_x([0, .001, .002, .003, .004, .005])
    slow = events(poses, timestamps=[0, .2, .4, .6, .8, 1.])
    fast = events(poses, timestamps=[0, .001, .002, .003, .004, .005])
    assert slow == [2, 5]
    assert fast == [5]


def test_gripper_stability_window_prevents_neighbouring_false_stop():
    poses = poses_from_x(np.zeros(8))
    assert events(poses, [1, 1, 1, 0, 0, 0, 0, 0]) == [3, 5, 7]


def test_episode_start_does_not_read_terminal_gripper_for_stability():
    poses = poses_from_x(np.zeros(10))
    first = events(poses)
    gripper = np.ones(10)
    gripper[-1] = 0
    second = events(poses, gripper)
    assert first[0] == second[0] == 2
    assert 0 not in first and 0 not in second


def test_terminal_adjacent_event_is_pruned_like_original_bridgevla():
    assert events(poses_from_x([0, .1, .2, .3]), [1, 1, 0, 0]) == [3]
    assert events(poses_from_x([0, .1])) == [1]


def test_event_selection_matches_original_bridgevla_with_equivalent_stop_signal():
    # Load only the original pure functions, not their RLBench imports.
    path = Path(__file__).resolve().parents[1] / "finetune/bridgevla/libs/peract/helpers/demo_loading_utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("_is_stopped", "keypoint_discovery")]
    namespace = dict(np=np, List=list, Demo=list,
                     logging=SimpleNamespace(debug=lambda *args: None))
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    poses = poses_from_x([0, .1, .2, .3, .3, .3, .4, .5, .6, .7, .8, .9, 1., 1.1])
    gripper = [1] * 8 + [0] * 6
    demo = [SimpleNamespace(gripper_open=g, joint_velocities=np.zeros(6) if i == 4 else np.ones(6))
            for i, g in enumerate(gripper)]
    assert events(poses, gripper) == namespace["keypoint_discovery"](demo)


@pytest.mark.parametrize("timestamps", [None, [0, 1], [0, .1, .1], [0, .2, .1], [0, np.nan, .2]])
def test_bridgevla_requires_aligned_finite_increasing_timestamps(timestamps):
    with pytest.raises(ValueError, match="timestamps"):
        keypoints(poses_from_x([0, .1, .2]), [1, 1, 1], timestamps=timestamps, method="bridgevla")


@pytest.mark.parametrize("options", [
    dict(method="unknown"),
    dict(method="gripper", max_frames=30),
    dict(method="gripper", stopping_translation_speed=.01),
    dict(method="bridgevla", max_translation=.04),
    dict(method="bridgevla", stopping_translation_speed=0),
    dict(method="bridgevla", stopping_rotation_speed_degrees=np.nan),
    dict(method="bridgevla", stopping_rotation_speed_degrees=None),
    dict(method="geometric", max_frames=1.5),
    dict(method="geometric", max_frames=True),
])
def test_invalid_keypoint_config_is_rejected(options):
    with pytest.raises(ValueError, match="keypoints"):
        keypoint_options(options)


def test_legacy_geometric_configs_keep_their_original_boundaries():
    poses = poses_from_x(np.arange(10) * .02)
    gripper = [1] * 4 + [0] * 6
    instructions = [0] * 7 + [1] * 3
    legacy = dict(max_translation=.04, max_rotation_degrees=8, max_frames=30)
    actual = keypoints(poses, gripper, instructions, **legacy)
    assert actual == [2, 3, 4, 6, 7, 9]
    assert actual == keypoints(poses, gripper, instructions, method="geometric", **legacy)
    assert keypoint_options()["method"] == "geometric"


@pytest.mark.parametrize("gripper", [[1, .5, 1], [1, 1], [1, np.nan, 1]])
def test_binary_observed_gripper_is_required(gripper):
    with pytest.raises(ValueError, match="gripper"):
        events(poses_from_x([0, .1, .2]), gripper)


def test_invalid_keypoint_config_fails_before_output_creation(replay_fixture, tmp_path):
    f = replay_fixture
    config = deepcopy(f.config)
    config["keypoints"] = dict(method="bridgevla", stopping_translation_speed=-1)
    with pytest.raises(ValueError, match="stopping_translation_speed"):
        validate_data_config(config)
    output = tmp_path / "invalid-keypoints"
    with pytest.raises(ValueError, match="stopping_translation_speed"):
        build(f.root, f.manifest, config, output)
    assert not output.exists()


def test_replay_targets_next_event_and_keeps_world_absolute_gripper_labels(replay_fixture, tmp_path):
    f = replay_fixture
    config = deepcopy(f.config)
    config["keypoints"] = dict(method="bridgevla")
    output = tmp_path / "event-replay"
    build(f.root, f.manifest, config, output, sample_stride=2)
    rows = read_jsonl(output / "samples.jsonl")
    assert load_contract(output)["data_config"]["keypoints"]["method"] == "bridgevla"
    # Fixture motion never stops; observed gripper changes at frames 2 and 5.
    for row in rows:
        frame = row["frame"]
        expected_target = 2 if frame < 2 else 6  # frame 5 is adjacent to terminal.
        assert row["target_frame"] == expected_target > frame
        np.testing.assert_allclose(row["labels"]["gripper_pose"][:3],
                                   [.1 + row["episode_index"] * .01 + expected_target * .02, 0, .5])
        expected_gripper = int(expected_target == 6)
        assert row["labels"]["action"][-1] == expected_gripper
        assert row["labels"]["rot_grip_action_indicies"][-1] == expected_gripper


@pytest.mark.parametrize("stride,expected_frames", [(1, {0, 1, 2, 3, 4}), (2, {0, 2, 4}), (10, {0, 2})])
def test_gripper_replay_uses_next_change_and_excludes_tail(replay_fixture, tmp_path, stride, expected_frames):
    f = replay_fixture
    config = deepcopy(f.config)
    config["keypoints"] = dict(method="gripper")
    output = tmp_path / "gripper-replay"
    build(f.root, f.manifest, config, output, sample_stride=stride)
    rows = read_jsonl(output / "samples.jsonl")
    assert load_contract(output)["data_config"]["keypoints"] == dict(method="gripper")
    for record in f.report["episodes"]:
        episode_rows = [row for row in rows if row["task"] == record["task"] and
                        row["episode_index"] == record["episode_index"]]
        assert {row["frame"] for row in episode_rows} == expected_frames
    for row in rows:
        expected_target = 2 if row["frame"] < 2 else 5
        assert row["target_frame"] == expected_target > row["frame"]
        assert row["frame"] < 5  # no artificial frame-6 goal for the post-release tail.
        np.testing.assert_allclose(row["labels"]["gripper_pose"][:3],
                                   [.1 + row["episode_index"] * .01 + expected_target * .02, 0, .5])
        assert row["labels"]["action"][-1] == int(expected_target == 5)
        assert row["labels"]["rot_grip_action_indicies"][-1] == int(expected_target == 5)


@pytest.mark.parametrize("all_episodes", [False, True])
def test_gripper_replay_skips_no_event_episodes_without_decoding_or_terminal_fallback(
        replay_fixture, tmp_path, monkeypatch, all_episodes):
    from finetune.OHT.data import replay
    f = replay_fixture
    config = deepcopy(f.config)
    config["keypoints"] = dict(method="gripper")
    read_episode = replay.read_episode
    episode_videos = replay.EpisodeVideos
    skipped = []
    opened_videos = []

    def stationary_gripper(root, record):
        columns = read_episode(root, record)
        if all_episodes or record["episode_index"] == 0:
            for state in columns["observation.state"]:
                state[6] = config["gripper"]["open"]
            skipped.append((record["task"], record["episode_index"]))
        return columns

    def counted_videos(*args, **kwargs):
        opened_videos.append(args[0])
        return episode_videos(*args, **kwargs)

    monkeypatch.setattr(replay, "read_episode", stationary_gripper)
    monkeypatch.setattr(replay, "EpisodeVideos", counted_videos)
    output = tmp_path / "no-events-replay"
    if all_episodes:
        with pytest.raises(ValueError, match="No replay samples.*no keypoint events"):
            build(f.root, f.manifest, config, output)
        assert not (output / "complete.json").exists()
        assert not opened_videos
    else:
        with pytest.warns(UserWarning, match="Skipped.*episodes without keypoint events"):
            build(f.root, f.manifest, config, output)
        load_contract(output)
        rows = read_jsonl(output / "samples.jsonl")
        assert rows and all(row["episode_index"] != 0 for row in rows)
        complete = json.loads((output / "complete.json").read_text(encoding="utf-8"))
        assert {(item["task"], item["episode_index"]) for item in complete["skipped_episodes"]} == set(skipped)
        assert all(item["reason"] == "no_keypoints" for item in complete["skipped_episodes"])
        assert len(opened_videos) == len(f.report["episodes"]) - len(skipped)
