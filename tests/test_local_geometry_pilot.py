"""Cache provenance, episode isolation and the complete CPU pilot lifecycle."""
import json

import numpy as np
import pytest
import torch

from tools.train_local_geometry_refiner import (
    ensure_split_separation, load_cache, main,
)


@pytest.fixture
def caches(tmp_path, capsys):
    main(["synthetic", "--output-dir", str(tmp_path), "--train-samples", "6",
          "--validation-samples", "3", "--test-samples", "3"])
    capsys.readouterr()
    return tmp_path


def rewrite(input_path, destination, **updates):
    with np.load(input_path, allow_pickle=False) as archive:
        values = {key: archive[key] for key in archive.files}
    np.savez_compressed(destination, **{**values, **updates})


def test_synthetic_cache_has_explicit_conventions_and_independent_episodes(caches):
    training = load_cache(caches / "train.npz")
    validation = load_cache(caches / "validation.npz")
    testing = load_cache(caches / "test.npz")
    assert training["source"] == "synthetic"
    assert training["camera_model"] == "opencv_z_meters"
    assert training["candidate_source"] == "synthetic_perturbation"
    assert training["depth"].shape == (6, 1, 32, 32)
    assert not training["context"].any()  # Toy labels are not encoded in context.
    assert not torch.equal(training["candidate_pose"], training["target_pose"])
    ensure_split_separation(training, validation)
    ensure_split_separation(training, testing)
    ensure_split_separation(validation, testing)


def test_frame_disjoint_split_still_rejects_episode_leakage(caches):
    training, validation = (load_cache(caches / name) for name in ("train.npz", "validation.npz"))
    validation["episode_id"][0] = training["episode_id"][0]
    with pytest.raises(ValueError, match="overlap in episode_id"):
        ensure_split_separation(training, validation)


def test_two_caches_must_use_the_same_frozen_policy(caches):
    training, validation = (load_cache(caches / name) for name in ("train.npz", "validation.npz"))
    validation["baseline_id"] = "different_weights_or_preprocessing"
    with pytest.raises(ValueError, match="baseline_id"):
        ensure_split_separation(training, validation)


@pytest.mark.parametrize("updates,match", [
    ({"camera_model": np.array("radial_range")}, "camera model"),
    ({"source": np.array("rlbench")}, "policy_prediction"),
    ({"depth": np.ones((6, 1, 32, 32), dtype=np.uint16)}, "raw integer depth"),
    ({"context": np.full((6, 4), np.nan, dtype=np.float32)}, "context must be finite"),
    ({"target_pose": np.tile(np.eye(4, dtype=np.float32) * 2, (6, 1, 1))}, "rigid"),
    ({"sample_id": np.array(["same"] * 6)}, "must be unique"),
])
def test_malformed_or_ambiguous_inputs_are_rejected(caches, updates, match):
    changed = caches / "changed.npz"
    rewrite(caches / "train.npz", changed, **updates)
    with pytest.raises(ValueError, match=match):
        load_cache(changed)


def test_missing_depth_is_allowed_but_missing_calibration_is_not(caches):
    changed = caches / "changed.npz"
    rewrite(caches / "train.npz", changed,
            depth=np.full((6, 1, 32, 32), np.nan, dtype=np.float32))
    assert torch.isnan(load_cache(changed)["depth"]).all()
    rewrite(caches / "train.npz", changed,
            intrinsics=np.zeros((6, 1, 3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="intrinsics"):
        load_cache(changed)


def test_float16_pose_validation_is_supported(caches):
    changed = caches / "changed.npz"
    with np.load(caches / "train.npz", allow_pickle=False) as archive:
        target = archive["target_pose"].astype(np.float16)
    rewrite(caches / "train.npz", changed, target_pose=target)
    assert load_cache(changed)["target_pose"].dtype == torch.float32


def test_training_checkpoint_reload_and_held_out_evaluation(caches, capsys):
    checkpoint = caches / "refiner.pt"
    command = ["train", "--train-cache", str(caches / "train.npz"),
               "--validation-cache", str(caches / "validation.npz"),
               "--output", str(checkpoint), "--epochs", "2", "--resolution", "8",
               "--hidden-dim", "8", "--batch-size", "3", "--threads", "1"]
    assert main(command) == 0
    capsys.readouterr()
    stored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert stored["geometry"]["resolution"] == 8
    initial = stored["report"]["initial_validation"]
    assert initial["baseline"] == initial["refined"]  # Zero-init preserves the original policy.
    assert stored["report"]["epochs"] == 2
    assert main(["evaluate", "--cache", str(caches / "test.npz"),
                 "--checkpoint", str(checkpoint), "--threads", "1"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["source"] == "synthetic"
    assert report["scope"] == "offline_next_keyframe_pose_error"
    assert report["samples"] == 3
    assert np.isfinite(report["refined"]["translation_m"]["mean"])
    with pytest.raises(SystemExit):
        main(["evaluate", "--cache", str(caches / "validation.npz"),
              "--checkpoint", str(checkpoint), "--threads", "1"])
    assert "absent from training and validation" in capsys.readouterr().err
    with np.load(caches / "train.npz", allow_pickle=False) as archive:
        reused_ids = archive["sample_id"][:3]
    rewrite(caches / "test.npz", caches / "reused.npz", sample_id=reused_ids)
    with pytest.raises(SystemExit):
        main(["evaluate", "--cache", str(caches / "reused.npz"),
              "--checkpoint", str(checkpoint), "--threads", "1"])
    assert "sample IDs absent" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(command)
    assert "Refusing to overwrite" in capsys.readouterr().err


def test_synthetic_command_does_not_overwrite_existing_cache(caches, capsys):
    before = (caches / "train.npz").read_bytes()
    with pytest.raises(SystemExit):
        main(["synthetic", "--output-dir", str(caches)])
    assert "Refusing to overwrite" in capsys.readouterr().err
    assert (caches / "train.npz").read_bytes() == before
