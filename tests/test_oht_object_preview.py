"""Object diagnostics use current audited poses and never enter policy inputs."""
from copy import deepcopy
import json
import shutil

import numpy as np
import pytest

from finetune.OHT.cli import main
from finetune.OHT.data.common import digest, file_digest, read_jsonl, write_json
from finetune.OHT.data.dataset import OHTDataset
from finetune.OHT.data.object_preview import capture_object_poses, load_source_objects, object_payload
from finetune.OHT.data.reader import read_episode, validate_episode
from finetune.OHT.data.replay import load_contract
from tests.test_oht_html_preview import payload_from_html
from tests.test_oht_migration import replay_fixture


def object_sample():
    return dict(id="task/000000/000010", frame=10, diagnostic_objects=dict(
        frame=10, positions=[[.2+i*.1,.3,.4] for i in range(4)],
        quaternions=[[2,0,0,0], [0,0,0,3], [1,0,0,1], [0,0,1,1]]))


def test_object_axes_require_independent_order_and_preserve_world_centres():
    sample = object_sample()
    before = deepcopy(sample)
    objects = object_payload(sample)
    assert [o["name"] for o in objects] == [f"object_{i}" for i in range(4)]
    assert [o["position"] for o in objects] == sample["diagnostic_objects"]["positions"]
    np.testing.assert_allclose(objects[0]["axes"]["wxyz"], np.eye(3))
    np.testing.assert_allclose(objects[0]["axes"]["xyzw"], np.diag([1,-1,-1]))
    # wxyz +90 degrees about Z: local X -> world Y (rows are local axis vectors).
    np.testing.assert_allclose(objects[2]["axes"]["wxyz"], [[0,1,0],[-1,0,0],[0,0,1]], atol=1e-15)
    assert sample == before
    assert object_payload(dict(frame=10)) == []  # Legacy cache has no invented objects.


@pytest.mark.parametrize("count", [0, 1, 4, 7])
def test_all_recorded_objects_survive_capture_and_html_payload(count):
    positions = [[i*.1,.3,.4] for i in range(count)]
    orientations = [[1,0,0,0] for _ in range(count)]
    columns = {"observation.objects_pos": [np.asarray(positions).reshape(-1).tolist()],
               "observation.objects_quat": [np.asarray(orientations).reshape(-1).tolist()]}
    metadata = capture_object_poses(columns, 0, dict(path="episode.parquet", parquet_sha256="checksum"))
    objects = object_payload(dict(frame=0, diagnostic_objects=metadata))
    assert len(objects) == count
    assert [o["position"] for o in objects] == positions
    assert [o["name"] for o in objects] == [f"object_{i}" for i in range(count)]


@pytest.mark.parametrize("positions,orientations", [([0,0],[1,0,0,0]),
                                                  ([0]*6,[1,0,0,0]),
                                                  ([0]*3,[0]*4)])
def test_capture_rejects_incomplete_or_unpaired_objects(positions, orientations):
    with pytest.raises(ValueError):
        capture_object_poses({"observation.objects_pos": [positions], "observation.objects_quat": [orientations]},
                             0, dict(path="episode.parquet", parquet_sha256="checksum"))


def test_episode_validation_supports_all_objects_but_rejects_inconsistent_frames(replay_fixture):
    f = replay_fixture
    record = f.report["episodes"][0]
    columns = read_episode(f.root, record)
    n = len(columns["frame_index"])
    columns["observation.objects_pos"] = [[i*.1,0,0]*7 for i in range(n)]
    columns["observation.objects_quat"] = [[1,0,0,0]*7 for _ in range(n)]
    assert validate_episode(columns, record)["frames"] == n
    columns["observation.objects_quat"][0] = [1,0,0,0]*6
    with pytest.raises(ValueError):
        validate_episode(columns, record)


@pytest.mark.parametrize("damage", ["frame", "positions", "quaternions", "zero", "nan"])
def test_reject_malformed_or_noncurrent_object_geometry(damage):
    sample = object_sample()
    objects = sample["diagnostic_objects"]
    if damage == "frame":
        objects["frame"] = 20
    elif damage in ("positions", "quaternions"):
        objects[damage].pop()
    elif damage == "zero":
        objects["quaternions"][0] = [0]*4
    else:
        objects["positions"][0][0] = float("nan")
    with pytest.raises(ValueError):
        object_payload(sample)


def test_build_keeps_objects_in_sample_metadata_only(replay_fixture):
    f = replay_fixture
    row = read_jsonl(f.replay/"samples.jsonl")[0]
    assert row["diagnostic_objects"]["frame"] == row["frame"]
    assert len(object_payload(row)) == 4
    with np.load(f.replay/row["observation"], allow_pickle=False) as observation:
        assert not any("object" in name for name in observation.files)
    dataset = OHTDataset(f.replay, "train")
    assert not any("object" in name for name in dataset[0])
    # Changing privileged display metadata in memory has no policy effect.
    first = dataset[0]
    dataset.samples[0]["diagnostic_objects"]["positions"][0] = [900,900,900]
    second = dataset[0]
    for key in first:
        if isinstance(first[key], np.ndarray):
            np.testing.assert_array_equal(first[key], second[key])
        else:
            assert first[key] == second[key]


def test_legacy_replay_backfills_audited_current_objects_without_cache_edits(replay_fixture, tmp_path):
    f = replay_fixture
    replay = tmp_path/"legacy"
    shutil.copytree(f.replay, replay)
    rows = read_jsonl(replay/"samples.jsonl")
    for row in rows:
        row.pop("diagnostic_objects")
    index = replay/"samples.jsonl"
    index.write_text("".join(json.dumps(row)+"\n" for row in rows), encoding="utf-8")
    contract = json.loads((replay/"contract.json").read_text(encoding="utf-8"))
    contract.pop("sha256")
    contract["index_sha256"] = file_digest(index)
    contract["sha256"] = digest(contract)
    write_json(replay/"contract.json", contract)
    write_json(replay/"complete.json", dict(contract_sha256=contract["sha256"], samples=len(rows),
                                          index_sha256=contract["index_sha256"]))
    inputs = [replay/"contract.json", index, replay/rows[0]["observation"]]
    before = [file_digest(path) for path in inputs]
    args = ["diagnose-geometry", "--replay", str(replay), "--sample-id", rows[0]["id"], "--html"]
    output = tmp_path/"objects"
    assert main([*args, "--output", str(output), "--object-root", str(f.root),
                 "--object-manifest", str(f.manifest)]) == 0
    payload = payload_from_html(output/"point-cloud-roi.html")
    assert len(payload["objects"]) == 4
    assert payload["object_source"]["frame"] == rows[0]["frame"]
    assert payload["objects"][0]["position"] == [0,0,0]
    assert [file_digest(path) for path in inputs] == before
    plain = tmp_path/"plain"
    assert main([*args, "--output", str(plain)]) == 0
    assert payload_from_html(plain/"point-cloud-roi.html")["objects"] == []


@pytest.mark.parametrize("damage", ["contract", "manifest", "parquet", "timestamp", "frame", "episode", "profile", "direct"])
def test_source_backfill_rejects_unbound_or_wrong_frame_data(replay_fixture, tmp_path, damage):
    f = replay_fixture
    contract = load_contract(f.replay)
    row = read_jsonl(f.replay/"samples.jsonl")[0]
    manifest = f.manifest
    root = f.root
    if damage == "contract":
        contract["audit_manifest_sha256"] = "wrong"
    elif damage == "manifest":
        value = json.loads(manifest.read_text(encoding="utf-8"))
        value["episodes"][0]["parquet_sha256"] = "wrong"
        manifest = tmp_path/"audit.json"
        write_json(manifest, value)
    elif damage == "parquet":
        root = tmp_path/"raw"
        record = next(r for r in json.loads(manifest.read_text(encoding="utf-8"))["episodes"]
                      if r["task"] == row["task"] and r["episode_index"] == row["episode_index"])
        path = root/record["path"]
        path.parent.mkdir(parents=True)
        path.write_bytes(b"changed parquet")
    elif damage == "timestamp":
        row["timestamp"] += 1/60
    elif damage == "frame":
        row["frame"] += 1
    elif damage == "episode":
        row["episode_index"] = 999
    elif damage == "profile":
        row["data_profile"] = "wrong"
    else:
        row["frame"] = None
    with pytest.raises(ValueError):
        load_source_objects(root, manifest, contract, row)


def test_object_source_requires_html_and_both_paths(tmp_path):
    for extra in (["--object-root", "raw", "--html"], ["--object-manifest", "audit", "--html"],
                  ["--object-root", "raw", "--object-manifest", "audit"]):
        output = tmp_path/"invalid"
        with pytest.raises(ValueError, match="together with --html"):
            main(["diagnose-geometry", "--replay", "unused", "--sample-id", "unused", "--output", str(output), *extra])
        assert not output.exists()
