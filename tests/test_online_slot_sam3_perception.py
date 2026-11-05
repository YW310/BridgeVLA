"""Current-memory TASK interop; supplied Mock masks never authorize actions."""

import copy
from dataclasses import replace
import hashlib
import importlib
import os
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune"))
from bridgevla.online.slot_sam3_perception import OnlineTaskPerception


@pytest.fixture
def scene(tmp_path):
    repo = Path(os.environ.get("TASK_RELEVANT_OBJECT_REPO", ROOT.parent / "TASK_Relevant_object-main")).resolve()
    if not (repo / "slot_guidance_producers.py").is_file():
        pytest.skip("Set TASK_RELEVANT_OBJECT_REPO to the TASK checkout")
    client = OnlineTaskPerception(repo, config_path="configs/slot_guided_offline.json")
    api = client._load_api()
    adapter = importlib.import_module("slot_prompt_adapter")
    timestamp = "2026-11-05T08:00:00+08:00"
    meta = dict(schema_version=1, artifact_type="slot_rgbd_frame", episode_id="memory-episode",
                frame_id="sample-3", source_frame_index=150, observed_at=timestamp, cameras=[])
    cameras, masks = {}, {}
    intrinsics = np.array([[100.0, 0, 16], [0, 100.0, 16], [0, 0, 1]])
    for name, shift in (("front", 0), ("wrist", 2)):
        pose = np.eye(4)
        pose[:3, 3] = [0.4 + shift / 100, -0.2, 0.7]
        row = dict(name=name, rgb_path=f"{name}.not-read.png", depth_path=f"{name}.not-read.npy",
                   depth_encoding="meters_npy", source_frame_index=150, observed_at=timestamp,
                   calibration_source="current synthetic in-memory optical pose",
                   intrinsics=intrinsics.tolist(), extrinsics=pose.tolist())
        meta["cameras"].append(copy.deepcopy(row))
        rgb = np.full((32, 32, 3), [12, 64, 201], dtype=np.uint8)
        depth = np.ones((32, 32), dtype=np.float64)
        depth[0, 0] = np.nan
        camera = adapter.CameraFrame(name, meta["episode_id"], meta["frame_id"], 150,
                                      timestamp, intrinsics, pose, (32, 32), row["calibration_source"])
        cameras[name] = api.slot_guided_io.RGBDFrame(camera, rgb, depth, row)
        masks[name] = {}
        for role, left in (("target", 8 - shift), ("reference", 22 - shift)):
            mask = np.zeros((32, 32), dtype=bool)
            mask[8:14, left:left + 6] = True
            masks[name][role] = mask
    snapshot = api.slot_guided_io.FrameSnapshot(meta, cameras, tmp_path / "frame-that-does-not-exist.json")
    points = []
    for role in ("target", "reference"):
        frame = cameras["front"]
        cam = api.camera_geometry.backproject_mask(frame.depth_m, masks["front"][role], intrinsics, 0)
        points.append(api.camera_geometry.transform_points(cam, frame.camera.extrinsics))
    packet = dict(slot_output=dict(points=np.stack(points), confidence=np.array([0.9, 0.9]),
                                   role_token_valid=np.array([True, True]), valid=np.array([True, True]),
                                   reference_null_probability=np.array(0.02)),
                  point_valid=np.ones((2, len(points[0])), dtype=bool),
                  point_support_source="current independently known synthetic renderer coverage",
                  matrix_world_from_source=np.eye(4), source_frame="rlbench_world")
    backends = {name: api.slot_guided_sam3.MockBackend({
        role: (np.stack([masks[name][other], masks[name][role]]), np.array([0.99, 0.65]))
        for role, other in (("target", "reference"), ("reference", "target"))
    }) for name in cameras}
    checkpoint = tmp_path / "actual-test-checkpoint.bin"
    checkpoint.write_bytes(b"synthetic test provenance only; model inference not measured")
    client = OnlineTaskPerception(repo, "configs/slot_guided_offline.json", backend=backends)
    return dict(client=client, api=api, snapshot=snapshot, packet=packet,
                backends=backends, checkpoint=checkpoint, repo=repo)


def _run(scene, **kwargs):
    options = dict(instruction="move the left block next to the right block", task_name="memory",
                   checkpoint_path=scene["checkpoint"], bridge_git_sha="a" * 40, point_count=128)
    options.update(kwargs)
    return scene["client"].perceive(scene["snapshot"], scene["packet"], **options)


def _next_frame(snapshot, index):
    value = copy.deepcopy(snapshot)
    value.metadata.update(source_frame_index=index, frame_id=f"sample-{index}")
    for row in value.metadata["cameras"]:
        row["source_frame_index"] = index
    for frame in value.cameras.values():
        frame.metadata["source_frame_index"] = index
        frame.camera = replace(frame.camera, source_frame_index=index, frame_id=value.metadata["frame_id"])
    return value


def test_memory_chain_uses_real_task_fusion_and_no_frame_json_or_npz(scene, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Live current-frame perception must not round-trip frame artifacts")

    monkeypatch.setattr(scene["api"].slot_guided_io, "load_frame_snapshot", forbidden)
    monkeypatch.setattr(scene["api"].slot_guided_io, "run_offline_frame", forbidden)
    monkeypatch.setattr(scene["api"].slot_guidance, "load_guidance", forbidden)
    monkeypatch.setattr(scene["api"].slot_guidance, "save_guidance", forbidden)
    monkeypatch.setattr(np, "load", forbidden)
    monkeypatch.setattr(np, "savez_compressed", forbidden)
    result = _run(scene)
    assert result.backend_kind == "mock" and result.evidence_kind == "synthetic_mock"
    assert result.diagnostics is None
    assert not scene["snapshot"].path.exists()
    assert result.geometry.object_points.shape == (1, 2, 128, 3)
    assert result.geometry.object_role_valid.all()
    assert not result.geometry.object_points.requires_grad
    assert result.receipt["source_frame_index"] == 150
    assert result.receipt["max_source_frame_index_read"] == 150
    provenance = result.geometry.provenance[0]
    assert provenance["snapshot_input_sha256"] == result.receipt["snapshot_input_sha256"]
    assert provenance["source_checkpoint_sha256"] == hashlib.sha256(scene["checkpoint"].read_bytes()).hexdigest()
    assert provenance["backend_kind"] == "mock" and provenance["sam3_checkpoint_sha256"] is None
    for role in ("target", "reference"):
        row = result.fused_geometry["roles"][role]
        assert row["support_cameras"] == ["front", "wrist"]
        assert row["point_count"] > 0 and row["failure_reason"] is None
        for camera in result.inference["cameras"].values():
            candidates = camera["roles"][role]["candidates"]
            assert candidates[0]["accepted"] is False  # Higher-scoring wrong instance.
            assert candidates[1]["accepted"] is True
    for backend in scene["backends"].values():
        assert backend.set_image_calls == 1
        assert len({call["state_id"] for call in backend.predict_calls}) == 1


def test_next_frame_reencodes_without_reusing_masks_or_geometry(scene):
    first = _run(scene)
    scene["snapshot"] = _next_frame(scene["snapshot"], 151)
    second = _run(scene)
    assert second.geometry is not first.geometry
    assert second.receipt["source_frame_index"] == 151
    assert second.inference["cache_key"] != first.inference["cache_key"]
    for backend in scene["backends"].values():
        assert backend.set_image_calls == 2
        assert len(backend.predict_calls) == 4
    with pytest.raises(ValueError, match="strictly increasing"):
        _run(scene)
    scene["snapshot"] = _next_frame(scene["snapshot"], 149)
    with pytest.raises(ValueError, match="strictly increasing"):
        _run(scene)


def test_reset_keeps_backend_but_clears_episode_order(scene):
    _run(scene)
    backend = scene["client"]._backend
    scene["client"].reset()
    second = _run(scene)
    assert scene["client"]._backend is backend
    assert second.receipt["source_frame_index"] == 150
    assert all(item.set_image_calls == 2 for item in scene["backends"].values())


def test_bad_digest_or_transform_does_not_become_geometry(scene):
    with pytest.raises(ValueError, match="changed since"):
        _run(scene, expected_snapshot_digest="0" * 64)
    scene["packet"]["matrix_world_from_source"][2, 2] = 0
    with pytest.raises(ValueError, match="proper uniform-scale"):
        _run(scene)
    assert all(item.set_image_calls == 0 for item in scene["backends"].values())


def test_reference_null_and_unavailable_target_remain_explicit(scene):
    scene["packet"]["slot_output"]["reference_null_probability"] = np.array(0.99)
    result = _run(scene)
    assert result.geometry.statuses == (("VISIBLE", "NULL"),)
    assert not result.geometry.object_role_valid[0, 1]
    assert result.geometry.reference_null_probability[0] > 0.98
    scene["snapshot"] = _next_frame(scene["snapshot"], 151)
    scene["packet"]["slot_output"]["confidence"][0] = 0.1
    result = _run(scene)
    assert result.geometry.statuses == (("LOW_CONFIDENCE", "NULL"),)
    assert not result.geometry.object_role_valid.any()


def test_optional_diagnostics_are_write_only_and_separate_from_live_result(scene, tmp_path):
    directory = tmp_path / "diagnostics"
    result = _run(scene, diagnostics_dir=directory)
    assert Path(result.diagnostics["inference_path"]).is_file()
    assert Path(result.diagnostics["object_geometry_path"]).is_file()
    assert all("mask" in row for camera in result.inference["cameras"].values()
               for role in camera["roles"].values() for row in role["candidates"])
    assert list(directory.glob("*.diagnostic.png"))


def test_model_backend_constructed_once_and_frames_encode_fresh(scene, monkeypatch):
    constructed = []
    module = scene["api"].slot_guided_sam3

    def fake_builder(checkpoint, device="cpu"):
        constructed.append((checkpoint, device))
        return scene["backends"]["front"]

    monkeypatch.setattr(module, "Sam3Backend", fake_builder)
    scene["client"] = OnlineTaskPerception(scene["repo"], sam3_checkpoint=scene["checkpoint"])
    assert constructed == []
    assert scene["client"].prepare_backend(scene["snapshot"]) == "mock"
    _run(scene)
    scene["snapshot"] = _next_frame(scene["snapshot"], 151)
    _run(scene)
    assert len(constructed) == 1
    # A shared backend still encodes once for each of two cameras per frame.
    assert scene["backends"]["front"].set_image_calls == 4
