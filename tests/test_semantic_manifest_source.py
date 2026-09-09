import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import rewrite_replay_with_semantic_roles as rewrite


def test_source_mask_fingerprint_rejects_wrong_raw_dataset(monkeypatch, tmp_path):
    mask = np.array([[99, 93]], dtype=np.int64)
    digest = hashlib.sha256(
        str(mask.shape).encode("ascii") + mask.astype("<i8").tobytes()).hexdigest()
    entries = [{"source_frame0_masks": {"front": digest}}]
    monkeypatch.setattr(rewrite, "load_frame_masks", lambda *args: {"front": mask})
    rewrite._validate_source_masks(tmp_path, entries)
    mask[0, 0] = 87
    with pytest.raises(ValueError, match="source mask mismatch"):
        rewrite._validate_source_masks(tmp_path, entries)


def test_stored_manifest_rejects_handles_outside_verified_map(tmp_path):
    folder = tmp_path / "close_jar"
    folder.mkdir()
    manifest = dict(
        schema_version=rewrite.SEMANTIC_ROLE_SCHEMA,
        phase_source="demo_events", source_alignment_validated=True,
        handle_namespace="stored", source_frame0_masks={"front": "digest"},
        handle_alignment={"status": "verified", "live_to_stored": {"87": 99}},
        expected_sample_frames=[0],
        entries=[dict(sample_frame=0, completion_satisfied=True,
                      target={"kind": "object", "handles": [87]}, reference=None)])
    path = folder / "episode_0.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Unverified stored handles"):
        rewrite._load_manifest(tmp_path, "close_jar", 0)
    manifest["entries"][0]["target"]["handles"] = [99]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    _, frames, entries = rewrite._load_manifest(tmp_path, "close_jar", 0)
    assert frames == [0]
    assert entries[0]["target"]["handles"] == [99]
