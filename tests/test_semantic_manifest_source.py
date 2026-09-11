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
    manifest['handle_alignment']['status'] = 'mask_verified'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='allow-mask-verified-handles'):
        rewrite._load_manifest(tmp_path, 'close_jar', 0)
    _, frames, _ = rewrite._load_manifest(tmp_path, 'close_jar', 0, allow_mask_verified=True)
    assert frames == [0]
    manifest['entries'][0]['target']['handles'] = [123]
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='Unverified stored handles'):
        rewrite._load_manifest(tmp_path, 'close_jar', 0, allow_mask_verified=True)


def test_stored_manifest_accepts_audited_semantic_entity_union_mapping(tmp_path):
    folder = tmp_path / 'meat_off_grill'
    folder.mkdir()
    entity_evidence = {
        'source': 'semantic_entity_union_mask_overlap',
        'stored_handles': [99],
        'views': {
            'front': {'passed': True},
            'left_shoulder': {'passed': True},
        },
    }
    manifest = dict(
        schema_version=rewrite.SEMANTIC_ROLE_SCHEMA,
        phase_source='demo_events', source_alignment_validated=True,
        handle_namespace='stored', source_frame0_masks={'front': 'digest'},
        handle_alignment={
            'status': 'mask_verified',
            'alignment_scope': 'semantic_entity_union',
            'live_to_stored': {},
            'semantic_entity_to_stored': {'82,83': [99]},
            'evidence': {'entities': {'chicken': entity_evidence}},
        },
        expected_sample_frames=[43],
        entries=[dict(
            sample_frame=43, completion_satisfied=True,
            target={'kind': 'object', 'handles': [99]}, reference={
                'kind': 'site', 'handles': [], 'site_position': [0., 0., 0.]})],
    )
    path = folder / 'episode_54.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')

    _, frames, entries = rewrite._load_manifest(
        tmp_path, 'meat_off_grill', 54, allow_mask_verified=True)

    assert frames == [43]
    assert entries[0]['target']['handles'] == [99]

    manifest['handle_alignment']['evidence']['entities']['chicken']['views'][
        'left_shoulder']['passed'] = False
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='Uncertified semantic entity handles'):
        rewrite._load_manifest(
            tmp_path, 'meat_off_grill', 54, allow_mask_verified=True)


    entity_evidence.update(
        source='semantic_entity_union_asymmetric_multiview_mask_overlap',
        views={
            'front': {
                'passed': False, 'identity_support': True,
                'strong_identity_support': False,
                'hard_mask_conflict': False,
            },
            'left_shoulder': {
                'passed': True, 'identity_support': True,
                'strong_identity_support': True,
                'hard_mask_conflict': False,
            },
        },
    )
    path.write_text(json.dumps(manifest), encoding='utf-8')
    _, frames, _ = rewrite._load_manifest(
        tmp_path, 'meat_off_grill', 54, allow_mask_verified=True)
    assert frames == [43]

    entity_evidence['views']['left_shoulder']['strong_identity_support'] = False
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='Uncertified semantic entity handles'):
        rewrite._load_manifest(
            tmp_path, 'meat_off_grill', 54, allow_mask_verified=True)

    entity_evidence.update(
        source='semantic_entity_union_thin_exact_multiview_mask_overlap',
        views={
            'front': {
                'thin_identity_support': True,
                'thin_strong_identity_support': False,
                'hard_mask_conflict': False,
            },
            'right_shoulder': {
                'thin_identity_support': True,
                'thin_strong_identity_support': True,
                'hard_mask_conflict': False,
            },
        },
    )
    path.write_text(json.dumps(manifest), encoding='utf-8')
    _, frames, _ = rewrite._load_manifest(
        tmp_path, 'meat_off_grill', 54, allow_mask_verified=True)
    assert frames == [43]

    entity_evidence['views']['front']['thin_identity_support'] = False
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='Uncertified semantic entity handles'):
        rewrite._load_manifest(
            tmp_path, 'meat_off_grill', 54, allow_mask_verified=True)


def test_stored_manifest_validates_each_mixed_entity_certificate(tmp_path):
    folder = tmp_path / 'close_jar'
    folder.mkdir()
    manifest = dict(
        schema_version=rewrite.SEMANTIC_ROLE_SCHEMA,
        phase_source='demo_events', source_alignment_validated=True,
        handle_namespace='stored', source_frame0_masks={'front': 'digest'},
        handle_alignment={
            'status': 'mask_verified',
            'alignment_scope': 'mixed_entity_certificates',
            'live_to_stored': {'87': 199},
            'semantic_entity_to_stored': {'87': [199], '88': [193]},
            'evidence': {'entity_certificates': {
                '87': {
                    'source': 'individual_handles', 'live_handles': [87],
                    'stored_handles': [199]},
                '88': {
                    'source': 'semantic_entity_union_mask_overlap',
                    'live_handles': [88], 'stored_handles': [193],
                    'views': {'front': {'passed': True},
                              'left_shoulder': {'passed': True}}},
            }},
        },
        expected_sample_frames=[0],
        entries=[dict(
            sample_frame=0, completion_satisfied=True,
            target={'kind': 'object', 'handles': [199]},
            reference={'kind': 'object', 'handles': [193]})],
    )
    path = folder / 'episode_0.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')
    _, frames, _ = rewrite._load_manifest(
        tmp_path, 'close_jar', 0, allow_mask_verified=True)
    assert frames == [0]

    manifest['handle_alignment']['evidence']['entity_certificates']['88'][
        'stored_handles'] = [199]
    path.write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='Uncertified semantic entity mapping'):
        rewrite._load_manifest(
            tmp_path, 'close_jar', 0, allow_mask_verified=True)
