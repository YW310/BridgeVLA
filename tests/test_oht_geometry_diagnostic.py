"""Incomplete replay inspection never relaxes training or invents labels."""
import base64
import json
import shutil

import numpy as np
import pytest
from PIL import Image

from finetune.OHT.cli import main
from finetune.OHT.data.common import digest, file_digest, read_jsonl, write_json
from finetune.OHT.data.dataset import OHTDataset
from finetune.OHT.data.geometry import backproject
from finetune.OHT.data.geometry_diagnostic import load_geometry_diagnostic
from finetune.OHT.data.html_preview import point_cloud_payload
from finetune.OHT.data.replay import load_contract
from tests.test_oht_migration import replay_fixture


@pytest.fixture
def partial(replay_fixture, tmp_path):
    """Only an initial contract and a finished NPZ, as during replay.build()."""
    root = tmp_path / "partial"
    root.mkdir()
    row = read_jsonl(replay_fixture.replay / "samples.jsonl")[0]
    contract = json.loads((replay_fixture.replay / "contract.json").read_text(encoding="utf-8"))
    contract.pop("index_sha256")
    _save_contract(root, contract)
    path = root / row["observation"]
    path.parent.mkdir(parents=True)
    shutil.copy2(replay_fixture.replay / row["observation"], path)
    return root, row, contract


def _save_contract(root, contract):
    contract.pop("sha256", None)
    contract["sha256"] = digest(contract)
    write_json(root / "contract.json", contract)


def _snapshot(root):
    return {path.relative_to(root).as_posix(): file_digest(path) for path in root.rglob("*") if path.is_file()}


def _direct(partial, **kwargs):
    root, row, _ = partial
    return load_geometry_diagnostic(root, observation_path=row["observation"],
                                    data_profile=row["data_profile"], allow_incomplete=True, **kwargs)


def test_incomplete_buffers_remain_rejected_by_default_and_training(partial, tmp_path):
    root, row, _ = partial
    for load in (lambda: load_contract(root), lambda: OHTDataset(root, "train"),
                 lambda: load_geometry_diagnostic(root, sample_id=row["id"])):
        with pytest.raises(FileNotFoundError, match="complete.json"):
            load()
    with pytest.raises(ValueError, match="requires --allow-incomplete"):
        load_geometry_diagnostic(root, observation_path=row["observation"])
    output = tmp_path / "strict-output"
    with pytest.raises(FileNotFoundError, match="complete.json"):
        main(["diagnose-geometry", "--replay", str(root), "--sample-id", row["id"], "--output", str(output)])
    assert not output.exists()


def test_missing_index_explains_direct_npz_mode(partial):
    root, row, _ = partial
    with pytest.raises(ValueError, match="No samples.jsonl.*--observation"):
        load_geometry_diagnostic(root, sample_id=row["id"], allow_incomplete=True)


@pytest.mark.parametrize("marker", [None, "{malformed completion marker"])
def test_indexed_diagnostic_ignores_only_completion_and_preserves_all_inputs(partial, marker):
    root, row, _ = partial
    (root / "samples.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    if marker is not None:
        (root / "complete.json").write_text(marker, encoding="utf-8")
    before = _snapshot(root)
    with pytest.warns(UserWarning, match="Diagnostic only"):
        _, _, sample, _, report = load_geometry_diagnostic(root, sample_id=row["id"], allow_incomplete=True)
    assert sample["labels"] == row["labels"] and sample["current_tcp"] == row["current_tcp"]
    assert not report["completion_checked"] and not report["index_checksum_verified"]
    assert report["observation_checksum_verified"]
    assert _snapshot(root) == before


def test_allow_incomplete_does_not_bypass_bound_index_or_npz_checksums(partial):
    root, row, contract = partial
    index = root / "samples.jsonl"
    index.write_text(json.dumps(row) + "\n", encoding="utf-8")
    contract["index_sha256"] = file_digest(index)
    _save_contract(root, contract)
    with pytest.warns(UserWarning):
        *_, report = load_geometry_diagnostic(root, sample_id=row["id"], allow_incomplete=True)
    assert report["index_checksum_verified"] and report["observation_checksum_verified"]
    old = index.read_bytes()
    index.write_bytes(old + b"\n")
    with pytest.raises(ValueError, match="sample index changed"):
        load_geometry_diagnostic(root, sample_id=row["id"], allow_incomplete=True)
    index.write_bytes(old)
    path = root / row["observation"]
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Observation changed"):
        load_geometry_diagnostic(root, sample_id=row["id"], allow_incomplete=True)


def test_direct_npz_without_index_exports_no_fake_tcp_goal_or_completion(partial, tmp_path):
    root, row, contract = partial
    # Make per-profile depth semantics differ from contract.data_config.
    contract["source_data_configs"][row["data_profile"]]["data_config"]["depth"]["kind"] = "ray"
    _save_contract(root, contract)
    before = _snapshot(root)
    output = tmp_path / "early-preview"
    with pytest.warns(UserWarning, match="Diagnostic only"):
        assert main(["diagnose-geometry", "--replay", str(root), "--observation", row["observation"],
                     "--data-profile", row["data_profile"], "--allow-incomplete", "--html",
                     "--html-source", "depth", "--output", str(output)]) == 0
    report = json.loads((output / "geometry.json").read_text(encoding="utf-8"))["validation"]
    assert report["metadata_source"] == "direct_npz" and report["data_profile"] == row["data_profile"]
    assert not any(report[key] for key in ("completion_checked", "index_checksum_verified", "observation_checksum_verified"))
    assert report["observation_sha256"] == row["observation_sha256"]  # recorded, not externally verified
    with pytest.warns(UserWarning):
        _, config, sample, observation, _ = _direct(partial)
    payload = point_cloud_payload(observation, config, sample, source="depth")
    assert payload["current_tcp"] is None and payload["goal"] is None
    assert payload["frame"] is None and payload["target_frame"] is None
    assert "仅诊断" in payload["diagnostic_note"]
    xyz = np.frombuffer(base64.b64decode(payload["xyz"]), "<f4").reshape(-1, 3)
    camera = next(iter(config["cameras"]))
    expected = backproject(observation[camera + "_depth"][0], observation[camera + "_camera_intrinsics"],
                           observation[camera + "_camera_extrinsics"], "ray").reshape(-1, 3)
    expected = expected[np.isfinite(expected).all(axis=1)]
    np.testing.assert_allclose(xyz[:len(expected)], expected)
    with Image.open(output / "fused_rgb.png") as image:
        assert image.size == (1024, 1170)  # global views only; no fabricated GT-centered row
    assert _snapshot(root) == before
    assert not (root / "complete.json").exists() and not (root / "samples.jsonl").exists()


def test_direct_npz_requires_unambiguous_source_profile(partial):
    root, row, contract = partial
    with pytest.raises(ValueError, match="Multiple source data profiles"):
        load_geometry_diagnostic(root, observation_path=row["observation"], allow_incomplete=True)
    with pytest.raises(ValueError, match="Unknown source data profile"):
        load_geometry_diagnostic(root, observation_path=row["observation"], data_profile="wrong", allow_incomplete=True)
    profile = row["data_profile"]
    contract["source_data_configs"] = {profile: contract["source_data_configs"][profile]}
    _save_contract(root, contract)
    with pytest.warns(UserWarning):
        *_, report = load_geometry_diagnostic(root, observation_path=row["observation"], allow_incomplete=True)
    assert report["data_profile"] == profile


def test_diagnostic_still_requires_authentic_contract(partial):
    root, _, contract = partial
    contract["data_config"]["scene_bounds"][0] -= 1
    write_json(root / "contract.json", contract)  # deliberately do not re-sign
    with pytest.raises(ValueError, match="contract checksum mismatch"):
        _direct(partial)


@pytest.mark.parametrize("damage", ["truncated", "missing_camera", "raw_depth", "bad_extrinsics", "wrong_resolution"])
def test_incomplete_or_malformed_npz_creates_no_preview(partial, tmp_path, damage):
    root, row, contract = partial
    path = root / row["observation"]
    with np.load(path, allow_pickle=False) as source:
        observation = {key: source[key] for key in source.files}
    camera = next(iter(contract["data_config"]["cameras"]))
    if damage == "truncated":
        path.write_bytes(path.read_bytes()[:30])
    else:
        if damage == "missing_camera":
            del observation[camera + "_camera_intrinsics"]
        elif damage == "raw_depth":
            observation[camera + "_depth"] = np.nan_to_num(observation[camera + "_depth"]).astype(np.uint16)
        elif damage == "bad_extrinsics":
            observation[camera + "_camera_extrinsics"][0, 0] = 2
        else:
            for suffix in ("rgb", "depth", "point_cloud"):
                observation[camera + "_" + suffix] = observation[camera + "_" + suffix][:, :2, :2]
        np.savez_compressed(path, **observation)
    output = tmp_path / "bad-preview"
    with pytest.raises(ValueError):
        main(["diagnose-geometry", "--replay", str(root), "--observation", row["observation"],
              "--data-profile", row["data_profile"], "--allow-incomplete", "--output", str(output)])
    assert not output.exists() and not (root / "complete.json").exists()


def test_direct_paths_cannot_escape_replay(partial, tmp_path):
    root, row, _ = partial
    outside = tmp_path / "outside.npz"
    shutil.copy2(root / row["observation"], outside)
    with pytest.raises(ValueError, match="Path escapes"):
        load_geometry_diagnostic(root, observation_path=outside, allow_incomplete=True)


def test_complete_replay_keeps_default_verification(replay_fixture):
    row = read_jsonl(replay_fixture.replay / "samples.jsonl")[0]
    *_, report = load_geometry_diagnostic(replay_fixture.replay, sample_id=row["id"])
    assert report["completion_checked"] and report["index_checksum_verified"] and report["observation_checksum_verified"]
