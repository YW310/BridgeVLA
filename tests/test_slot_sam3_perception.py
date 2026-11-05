"""Real TASK interoperability with supplied Mock masks, never model acceptance."""

import hashlib
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune"))
from bridgevla.offline.slot_sam3_perception import TaskPerception
from bridgevla.data.observations import preprocess_inputs


@pytest.fixture
def task_repo():
    path = Path(os.environ.get("TASK_RELEVANT_OBJECT_REPO", ROOT.parent / "TASK_Relevant_object-main")).resolve()
    if not (path / "slot_guidance_producers.py").is_file():
        pytest.skip("Set TASK_RELEVANT_OBJECT_REPO to the TASK checkout for interoperability tests")
    return path


@pytest.fixture
def scene(tmp_path, task_repo):
    timestamp = "2026-11-05T08:00:00+08:00"
    intrinsics = [[100.0, 0, 16.0], [0, 100.0, 16.0], [0, 0, 1.0]]
    cameras, masks = [], {}
    for name, shift in (("front", 0), ("wrist", 2)):
        pose = np.eye(4)
        pose[:3, 3] = [0.4 + shift / 100, -0.2, 0.7]
        rgb = np.full((32, 32, 3), [12, 64, 201], dtype=np.uint8)
        depth = np.ones((32, 32), dtype=np.float64)
        depth[0, 0] = np.nan
        Image.fromarray(rgb).save(tmp_path / f"{name}.png")
        np.save(tmp_path / f"{name}.npy", depth)
        masks[name] = {}
        for role, left in (("target", 8 - shift), ("reference", 22 - shift)):
            mask = np.zeros((32, 32), dtype=bool)
            mask[8:14, left:left + 6] = True
            masks[name][role] = mask
        cameras.append(dict(name=name, rgb_path=f"{name}.png", depth_path=f"{name}.npy",
                            depth_encoding="meters_npy", source_frame_index=150,
                            observed_at=timestamp, calibration_source="current synthetic optical pose",
                            intrinsics=intrinsics, extrinsics=pose.tolist()))
    frame_path = tmp_path / "frame.json"
    frame_path.write_text(json.dumps(dict(schema_version=1, artifact_type="slot_rgbd_frame",
                          episode_id="synthetic", frame_id="sample-3", source_frame_index=150,
                          observed_at=timestamp, cameras=cameras)), encoding="utf-8")
    client = TaskPerception(task_repo, config_path="configs/slot_guided_offline.json")
    snapshot = client.load_snapshot(frame_path)
    geometry = importlib.import_module("camera_geometry")
    guidance = importlib.import_module("slot_guidance")
    backend_module = importlib.import_module("slot_guided_sam3")
    points = []
    for role in ("target", "reference"):
        front = snapshot.cameras["front"]
        cloud = geometry.backproject_mask(front.depth_m, masks["front"][role], np.asarray(intrinsics), max_points=0)
        points.append(geometry.transform_points(cloud, np.asarray(front.camera.extrinsics)))
    export = dict(
        slot_output=dict(points=np.stack(points), confidence=np.array([0.9, 0.9]),
                         role_token_valid=np.array([True, True]), valid=np.array([True, True]),
                         reference_null_probability=np.array(0.02)),
        point_valid=np.ones((2, len(points[0])), dtype=bool),
        transform=guidance.identity_world_transform(source="synthetic detached world coordinates",
                                                    timestamp=timestamp, source_frame_index=150),
        point_support_source="known current synthetic measured samples",
    )
    backends = {}
    for name in masks:
        # Wrong same-category object has the higher SAM quality score.
        backends[name] = backend_module.MockBackend({
            role: (np.stack([masks[name][other], masks[name][role]]), np.array([0.99, 0.65]))
            for role, other in (("target", "reference"), ("reference", "target"))
        })
    checkpoint = tmp_path / "coarse-checkpoint.test"
    checkpoint.write_bytes(b"test fixture only; no checkpoint inference claimed")
    return dict(task_repo=task_repo, client=client, snapshot=snapshot, frame_path=frame_path,
                export=export, backends=backends, checkpoint=checkpoint, masks=masks, timestamp=timestamp)


def _perceive(scene, tmp_path, **kwargs):
    client = TaskPerception(scene["task_repo"], "configs/slot_guided_offline.json",
                            backend=scene["backends"])
    args = dict(snapshot_path=scene["frame_path"], coarse_export=scene["export"],
                instruction="move the left block next to the right block", task_name="synthetic",
                checkpoint_path=scene["checkpoint"], bridge_git_sha="a" * 40,
                output_dir=tmp_path / "perception", created_at=scene["timestamp"])
    args.update(kwargs)
    return client.perceive(**args)


def test_constructor_is_lazy_and_load_snapshot_never_constructs_sam3(task_repo, scene, monkeypatch):
    module = importlib.import_module("slot_guided_sam3")

    def forbidden(*args, **kwargs):
        raise AssertionError("SAM3 must not initialize while checking current inputs")

    monkeypatch.setattr(module, "Sam3Backend", forbidden)
    client = TaskPerception(task_repo, sam3_checkpoint="missing.pt")
    assert client._api is None and client._backend is None
    snapshot = client.load_snapshot(scene["frame_path"])
    assert snapshot.metadata["source_frame_index"] == 150
    assert client._backend is None


def test_mock_end_to_end_persists_bound_guidance_inference_and_geometry(scene, tmp_path):
    report = _perceive(scene, tmp_path)
    assert report["backend_kind"] == "mock"
    assert report["evidence_kind"] == "synthetic_mock"
    assert "real_offline_acceptance_passed" not in report
    assert report["source_checkpoint_sha256"] == hashlib.sha256(scene["checkpoint"].read_bytes()).hexdigest()
    assert report["sam3_checkpoint_sha256"] is None
    for key in ("object_geometry_path", "guidance_path", "inference_path", "report_path"):
        assert Path(report[key]).is_file()
    geometry = importlib.import_module("slot_geometry_fusion").load_geometry(report["object_geometry_path"])
    assert geometry["source_frame_index"] == 150
    assert geometry["frame_id"] == "sample-3"
    for role in ("target", "reference"):
        row = geometry["roles"][role]
        assert row["status"] == "VISIBLE" and row["point_count"] > 0
        assert row["support_cameras"] == ["front", "wrist"]
        assert row["failure_reason"] is None
    for backend in scene["backends"].values():
        assert backend.set_image_calls == 1
        assert len({call["state_id"] for call in backend.predict_calls}) == 1


def test_oracle_source_uses_current_explicit_masks(scene, tmp_path):
    report = _perceive(scene, tmp_path, source="oracle_gt", coarse_export=None,
                      checkpoint_path=None, oracle_role_masks=scene["masks"],
                      reference_is_null=False, oracle_instance_ids={"target": "left", "reference": "right"})
    guidance = importlib.import_module("slot_guidance").load_guidance(report["guidance_path"])
    assert guidance.metadata["guidance_source"] == "oracle_gt"
    assert guidance.metadata["target"]["instance_id"] == "left"
    assert report["backend_kind"] == "mock" and report["source_checkpoint_sha256"] is None


def test_snapshot_tensors_match_actual_bridge_preprocess_without_future_fields(scene):
    import torch
    observation = scene["client"].snapshot_to_observation(
        scene["snapshot"], [1.0, 0.02, 0.03, 0.5], "move the block", ["front", "wrist"], "cpu",
    )
    assert set(observation) == {"low_dim_state", "language_goal", "front_rgb", "front_point_cloud", "wrist_rgb", "wrist_point_cloud"}
    assert observation["low_dim_state"].shape == (1, 1, 4)
    assert observation["language_goal"] == ["move the block"]
    for camera in ("front", "wrist"):
        assert observation[f"{camera}_rgb"].shape == (1, 1, 3, 32, 32)
        assert observation[f"{camera}_rgb"].dtype == torch.uint8
        assert observation[f"{camera}_point_cloud"].dtype == torch.float32
        assert torch.isnan(observation[f"{camera}_point_cloud"][0, 0, :, 0, 0]).all()
    obs, clouds = preprocess_inputs(observation, ["front", "wrist"])
    assert len(obs) == len(clouds) == 2
    torch.testing.assert_close(obs[0][0][0, :, 5, 5], torch.tensor([12, 64, 201]) / 127.5 - 1)
    torch.testing.assert_close(clouds[0][0, :, 8, 8], torch.tensor([0.32, -0.28, 1.7]))
    # Wrist moves in world, so the same surface has a shifted source pixel.
    torch.testing.assert_close(clouds[1][0, :, 8, 6], clouds[0][0, :, 8, 8])


def test_missing_calibration_is_rejected_without_guessed_cloud(scene):
    frame = scene["snapshot"].cameras["wrist"]
    frame.camera = replace(frame.camera, intrinsics=None)
    with pytest.raises(ValueError, match="MISSING_CALIBRATION"):
        scene["client"].snapshot_to_observation(scene["snapshot"], [1, 0, 0], "move", ["wrist"])


def test_preflight_requires_policy_cameras_and_valid_calibration_without_backend(scene):
    client, snapshot = scene['client'], scene['snapshot']
    client.validate_policy_snapshot(snapshot, ['front', 'wrist'])
    assert client._backend is None
    with pytest.raises(ValueError, match='missing policy cameras'):
        client.validate_policy_snapshot(snapshot, ['front', 'left_shoulder'])
    frame = snapshot.cameras['wrist']
    frame.camera = replace(frame.camera, extrinsics=None)
    with pytest.raises(ValueError, match='MISSING_CALIBRATION'):
        client.validate_policy_snapshot(snapshot, ['front', 'wrist'])


@pytest.mark.parametrize("state", [[1, 2], [1, 2, float("nan")], {"action": [1, 2, 3]}])
def test_state_vector_validation_rejects_missing_or_future_structures(scene, state):
    with pytest.raises(ValueError, match="low_dim_state"):
        scene["client"].snapshot_to_observation(scene["snapshot"], state, "move", ["front"])


def test_coarse_checkpoint_and_transform_require_current_real_provenance(scene, tmp_path):
    with pytest.raises(FileNotFoundError, match="actual Coarse checkpoint"):
        _perceive(scene, tmp_path, checkpoint_path=tmp_path / "missing.pt")
    scene["export"]["transform"]["source_frame_index"] = 3
    with pytest.raises(ValueError, match="source_frame_index"):
        _perceive(scene, tmp_path)
    assert not (tmp_path / "perception").exists()


def test_receipt_rejects_rgbd_changes_between_coarse_and_perception(scene, tmp_path):
    receipt = scene["client"].snapshot_receipt(scene["snapshot"])
    assert receipt["max_source_frame_index_read"] == 150
    assert receipt["frame_id"] == "sample-3"
    assert len(receipt["calibration_sha256"]) == len(receipt["snapshot_input_sha256"]) == 64
    rgb_path = scene["frame_path"].parent / "front.png"
    with Image.open(rgb_path) as image:
        rgb = np.array(image)
    rgb[1, 1] = [1, 2, 3]
    Image.fromarray(rgb).save(rgb_path)
    with pytest.raises(ValueError, match="snapshot_input_sha256 mismatch"):
        _perceive(scene, tmp_path, expected_snapshot_digest=receipt["snapshot_input_sha256"])
    assert not (tmp_path / "perception").exists()


def test_mock_cannot_claim_sam3_and_existing_output_is_not_overwritten(scene, tmp_path):
    class ForgedBackend:
        def fingerprint(self):
            return {"kind": "sam3"}

    client = TaskPerception(scene["task_repo"], backend=ForgedBackend())
    with pytest.raises(ValueError, match="Mock evidence"):
        client.perceive(scene["frame_path"], scene["export"], "move", "synthetic", scene["checkpoint"],
                        "a" * 40, tmp_path / "forged", created_at=scene["timestamp"])
    report = _perceive(scene, tmp_path)
    before = Path(report["guidance_path"]).read_bytes()
    with pytest.raises(FileExistsError):
        _perceive(scene, tmp_path)
    assert Path(report["guidance_path"]).read_bytes() == before
