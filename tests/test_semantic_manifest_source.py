import hashlib
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import rewrite_replay_with_semantic_roles as rewrite


def site_role(
    center=(0.0, 0.0, 0.0),
    extent=(0.02, 0.02, 0.02),
    source='fallback_box',
):
    return {
        'semantic_name': 'site',
        'kind': 'site',
        'handles': [],
        'site_position': list(center),
        'site_geometry': {
            'primitive': 'box_volume',
            'center_world': list(center),
            'rotation_world': np.eye(3).tolist(),
            'extent': list(extent),
            'source': source,
        },
    }


def test_rewrite_parser_supports_task_workers():
    args = rewrite.build_parser().parse_args([
        '--replay-dir', 'replay', '--raw-data-dir', 'raw',
        '--manifest-dir', 'manifests', '--output-dir', 'output',
        '--workers', '3',
    ])
    assert args.workers == 3


def test_rewrite_parser_supports_output_validation_and_visualization():
    args = rewrite.build_parser().parse_args([
        '--replay-dir', 'replay', '--raw-data-dir', 'raw',
        '--manifest-dir', 'manifests', '--output-dir', 'output',
        '--validate-output', '--validate-every', '100',
        '--visualize-every', '25',
        '--visualize-output-dir', 'visualizations',
        '--visualize-objects-only',
    ])
    assert args.validate_output
    assert args.validate_every == 100
    assert args.visualize_every == 25
    assert args.visualize_output_dir == Path('visualizations')
    assert args.visualize_objects_only


def test_rewrite_can_explicitly_fallback_invalid_manifest_to_raw(
        monkeypatch, tmp_path):
    source_dir = tmp_path / 'source'
    destination_dir = tmp_path / 'destination'
    source_dir.mkdir()
    with (source_dir / '0.replay').open('wb') as stream:
        pickle.dump({
            'terminal': np.asarray(0), 'episode_idx': np.asarray(7),
            'sample_frame': np.asarray(12), 'baseline': np.asarray([3.]),
        }, stream)
    monkeypatch.setattr(
        rewrite, '_load_manifest',
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError('source alignment failed')))
    monkeypatch.setattr(rewrite, '_copy_metadata', lambda *args, **kwargs: None)
    args = SimpleNamespace(
        resume=False, overwrite=False, manifest_dir=tmp_path / 'manifests',
        allow_mask_verified_handles=True,
        fallback_invalid_manifests_to_raw=True,
        raw_data_dir=tmp_path / 'raw', cameras=('front',),
        max_objects=4, num_points=8, cache_frames=0, cache_episodes=1, seed=0,
    )

    assert rewrite.process_task(
        args, 'place_shape_in_shape_sorter', source_dir, destination_dir) == 1

    with (destination_dir / '0.replay').open('rb') as stream:
        migrated = pickle.load(stream)
    assert migrated['baseline'].tolist() == [3.]
    assert not migrated['oracle_object_valid'].any()
    assert not bool(migrated['oracle_target_role_valid'])
    assert not bool(migrated['oracle_reference_role_valid'])
    assert migrated['oracle_phase_source'].tolist() == ['']
    stats = json.loads((
        destination_dir / 'semantic_role_rewrite_stats.json').read_text())
    assert stats['invalid_manifest_episodes'] == 1
    assert stats['invalid_manifest_transitions'] == 1
    assert stats['invalid_manifest_errors'] == {'7': 'source alignment failed'}


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
            target={'kind': 'object', 'handles': [99]},
            reference=site_role())],
    )
    path = folder / 'episode_54.json'
    path.write_text(json.dumps(manifest), encoding='utf-8')

    _, frames, entries = rewrite._load_manifest(
        tmp_path, 'meat_off_grill', 54, allow_mask_verified=True)

    assert frames == [43]
    assert entries[0]['target']['handles'] == [99]
    assert entries[0]['reference']['site_geometry']['source'] == 'fallback_box'

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


def test_v1_manifest_is_explicitly_rejected(tmp_path):
    folder = tmp_path / 'close_jar'
    folder.mkdir()
    (folder / 'episode_0.json').write_text(json.dumps({
        'schema_version': 'rlbench_o2_semantic_roles_v1',
    }), encoding='utf-8')

    with pytest.raises(ValueError, match='Unsupported semantic role schema'):
        rewrite._load_manifest(tmp_path, 'close_jar', 0)


def test_v2_manifest_rejects_site_without_geometry(tmp_path):
    folder = tmp_path / 'close_jar'
    folder.mkdir()
    (folder / 'episode_0.json').write_text(json.dumps({
        'schema_version': rewrite.SEMANTIC_ROLE_SCHEMA,
        'phase_source': 'sim_replay',
        'source_alignment_validated': True,
        'handle_namespace': 'stored',
        'source_frame0_masks': {'front': 'digest'},
        'handle_alignment': {
            'status': 'verified', 'live_to_stored': {},
            'semantic_entity_to_stored': {},
        },
        'expected_sample_frames': [0],
        'entries': [{
            'sample_frame': 0,
            'completion_satisfied': True,
            'target': {
                'semantic_name': 'legacy_site',
                'kind': 'site',
                'handles': [],
                'site_position': [0.0, 0.0, 0.0],
            },
            'reference': None,
        }],
    }), encoding='utf-8')

    with pytest.raises(ValueError, match='site_geometry'):
        rewrite._load_manifest(tmp_path, 'close_jar', 0)


def test_site_geometry_rewrite_uses_region_points_and_exact_descriptor_audit():
    role = site_role(
        center=(1.0, 2.0, 3.0),
        extent=(0.2, 0.4, 0.6),
        source='object_bbox',
    )
    raw = rewrite._role_points(role, {}, {}, 32)
    oracle = rewrite.empty_oracle_objects(4, 32)

    valid, count = rewrite._fill_slot(
        oracle, 0, rewrite.ORACLE_ROLE_TARGET, role, raw, 32,
        np.random.default_rng(0),
    )

    assert valid
    assert count == 32
    assert len(np.unique(oracle.points[0], axis=0)) > 1
    np.testing.assert_allclose(oracle.centers[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(oracle.sizes[0], [0.2, 0.4, 0.6])
    local = oracle.points[0] - oracle.centers[0]
    assert np.all(np.abs(local) <= oracle.sizes[0] / 2.0 + 1e-6)
    audit = rewrite._audit_fields(
        rewrite.SEMANTIC_ROLE_SCHEMA,
        {
            'phase_id': 'phase0',
            'target': role,
            'reference': None,
        },
        True,
        False,
        4,
    )
    assert audit['oracle_target_geometry_source'].tolist() == ['object_bbox']
    assert audit['oracle_reference_geometry_source'].tolist() == ['none']
    empty = rewrite._empty_audit(4)
    assert empty['oracle_target_geometry_source'].tolist() == ['none']


def _site_semantic_transition(max_objects=4, num_points=8):
    role = site_role()
    raw = rewrite._role_points(role, {}, {}, num_points)
    oracle = rewrite.empty_oracle_objects(max_objects, num_points)
    target_valid, _ = rewrite._fill_slot(
        oracle, 0, rewrite.ORACLE_ROLE_TARGET, role, raw, num_points,
        np.random.default_rng(0),
    )
    transition = {
        'terminal': np.asarray(0),
        'episode_idx': np.asarray(2),
        'sample_frame': np.asarray(7),
        'baseline': np.asarray([4.0], dtype=np.float32),
    }
    transition.update(oracle.as_replay_fields())
    transition.update(rewrite._audit_fields(
        rewrite.SEMANTIC_ROLE_SCHEMA,
        {
            'role_config_sha256': 'a' * 64,
            'phase_source': 'demo_events',
            'phase_id': 'phase0',
            'target': role,
            'reference': None,
        },
        target_valid,
        False,
        max_objects,
    ))
    return transition


def test_validate_task_output_checks_every_replay_and_site_geometry(tmp_path):
    source_dir = tmp_path / 'source'
    destination_dir = tmp_path / 'destination'
    source_dir.mkdir()
    destination_dir.mkdir()
    source = {
        'terminal': np.asarray(0),
        'episode_idx': np.asarray(2),
        'sample_frame': np.asarray(7),
        'baseline': np.asarray([4.0], dtype=np.float32),
    }
    with (source_dir / '0.replay').open('wb') as stream:
        pickle.dump(source, stream)
    transition = _site_semantic_transition()
    with (destination_dir / '0.replay').open('wb') as stream:
        pickle.dump(transition, stream)

    report = rewrite._validate_task_output(
        SimpleNamespace(max_objects=4, num_points=8),
        'reach_and_drag',
        source_dir,
        destination_dir,
    )

    assert report['valid']
    assert report['files'] == 1
    assert report['validated_files'] == 1
    assert report['validation_mode'] == 'full'
    assert report['validation_complete']
    assert report['counts_scope'] == 'all_files'
    assert report['site_roles'] == 1
    assert report['fallback_box_roles'] == 1
    assert report['raw_fallback_files'] == 0
    assert report['phase_sources'] == {'demo_events': 1}
    assert report['role_config_sha256'] == 'a' * 64
    assert report['num_points'] == 8
    assert report['max_objects'] == 4
    assert report['manifest_handle_namespace'] == 'stored'
    saved = json.loads((
        destination_dir / 'semantic_role_validation.json').read_text())
    assert saved == report

    transition['oracle_object_points'][0] = transition[
        'oracle_object_points'][0, 0]
    with pytest.raises(ValueError, match='repeated center point'):
        rewrite._validate_semantic_transition(transition, 4, 8)


def test_validate_task_output_supports_deterministic_sampling(
        monkeypatch, tmp_path):
    source_dir = tmp_path / 'source'
    destination_dir = tmp_path / 'destination'
    source_dir.mkdir()
    destination_dir.mkdir()
    for replay_index in range(8):
        payload = {'replay_index': replay_index}
        with (source_dir / f'{replay_index}.replay').open('wb') as stream:
            pickle.dump(payload, stream)
        with (destination_dir / f'{replay_index}.replay').open('wb') as stream:
            pickle.dump(payload, stream)

    checked = []

    def fake_validate(transition, max_objects, num_points, source=None):
        assert transition == source
        checked.append(transition['replay_index'])
        return SimpleNamespace(
            valid=np.zeros(max_objects, dtype=bool),
            roles=np.zeros(max_objects, dtype=np.int64),
        )

    monkeypatch.setattr(rewrite, '_validate_semantic_transition', fake_validate)
    report = rewrite._validate_task_output(
        SimpleNamespace(max_objects=4, num_points=8, validate_every=3),
        'reach_and_drag', source_dir, destination_dir)

    assert checked == [0, 3, 6, 7]
    assert report['files'] == 8
    assert report['validated_files'] == 4
    assert report['validation_mode'] == 'sampled'
    assert not report['validation_complete']
    assert report['counts_scope'] == 'validated_sample'
    assert report['validation_fraction'] == pytest.approx(4 / 8)


def test_semantic_visualization_reads_rewritten_oracle_fields(
        monkeypatch, tmp_path):
    destination_dir = tmp_path / 'semantic' / 'reach_and_drag'
    destination_dir.mkdir(parents=True)
    transition = _site_semantic_transition()
    with (destination_dir / '3.replay').open('wb') as stream:
        pickle.dump(transition, stream)
    captured = {}

    def fake_visualize(oracle, task, replay_index, output_dir, **kwargs):
        captured['points'] = oracle.points.copy()
        captured['task'] = task
        captured['replay_index'] = replay_index
        captured['group_by_id'] = kwargs['group_by_id']
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / 'reach_and_drag_replay_3.png'
        output.write_bytes(b'png')
        return output

    monkeypatch.setattr(rewrite, 'resolve_episode_dir', lambda *args: tmp_path)
    monkeypatch.setattr(rewrite, 'load_frame_rgb_images', lambda *args: {})
    monkeypatch.setattr(rewrite, 'load_frame_masks', lambda *args: {})
    monkeypatch.setattr(rewrite, 'visualize_oracle_objects', fake_visualize)
    args = SimpleNamespace(
        visualize_index=3,
        visualize_every=0,
        visualize_output_dir=tmp_path / 'visualizations',
        visualize_objects_only=True,
        max_objects=4,
        num_points=8,
        raw_data_dir=tmp_path / 'raw',
        cameras=('front',),
    )

    assert rewrite._visualize_task_output(
        args, 'reach_and_drag', destination_dir) == 1
    np.testing.assert_array_equal(
        captured['points'], transition['oracle_object_points'])
    assert captured['task'] == 'reach_and_drag'
    assert captured['replay_index'] == 3
    metadata = json.loads((
        tmp_path / 'visualizations' / 'reach_and_drag'
        / 'reach_and_drag_replay_3.json').read_text())
    assert metadata['phase_source'] == 'demo_events'
    assert metadata['target_kind'] == 'site'
