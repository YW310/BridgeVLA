import copy
import json
import shutil

import numpy as np
import pytest

from tests.test_oht_migration import replay_fixture  # noqa: F401
from finetune.OHT.data.common import CAMERAS, read_jsonl
from finetune.OHT.data.reader import discover
from finetune.OHT.data.seg_roles import export_roles, role_mask, role_spec


def test_masks_preserve_ids_and_exact_rgb_sampling():
    ids = np.array([[1, 255, 5, 255], [0, 1, 2, 3],
                    [4, 255, 6, 255], [0, 1, 2, 3]], np.uint8)
    np.testing.assert_array_equal(role_mask(ids, [1, 6], (2, 2)), [[True, False], [False, True]])
    with pytest.raises(ValueError, match='exact IDs'):
        role_mask(np.full((4, 4), 7, np.uint8), [1], (2, 2))
    with pytest.raises(ValueError, match='divide exactly'):
        role_mask(ids, [1], (3, 2))
    assert role_spec('unknown')[0] == dict(source='unknown', present=False, known=False)
    assert role_spec([])[0] == dict(source='none', present=False, known=True)
    with pytest.raises(ValueError):
        role_spec([255])


def test_seg_directory_has_canonical_task_identity_and_rejects_ambiguous_roots(tmp_path):
    base = tmp_path / 'disassemble_left-seg' / 'lerobot_dataset' / 'data' / 'chunk-000'
    base.mkdir(parents=True)
    (base / 'episode_000000.parquet').touch()
    row, = discover(tmp_path)
    assert row['task'] == 'disassemble_left'
    assert row['dataset'] == 'disassemble_left-seg/lerobot_dataset'
    assert row['metadata'] == 'disassemble_left-seg/episode_000000/metadata.json'
    (tmp_path / 'disassemble_left' / 'lerobot_dataset' / 'data').mkdir(parents=True)
    with pytest.raises(ValueError, match='Ambiguous'):
        discover(tmp_path)


def make_seg_source(f, tmp_path):
    import av
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = tmp_path / 'raw'
    shutil.copytree(f.root / 'disassemble_left', root / 'disassemble_left-seg')
    for record in discover(root):
        path = root / record['path']
        columns = pq.read_table(path).to_pydict()
        dataset = root / record['dataset']
        for camera in CAMERAS:
            relative = f'seg/chunk-000/observation.seg.{camera}/episode_{record["episode_index"]:06d}.mkv'
            movie = dataset / relative
            movie.parent.mkdir(parents=True, exist_ok=True)
            with av.open(str(movie), 'w') as container:
                stream = container.add_stream('ffv1', rate=60)
                stream.width, stream.height, stream.pix_fmt = 16, 16, 'gray'
                for frame in range(len(columns['timestamp'])):
                    ids = np.full((16, 16), 255, np.uint8)
                    ids[:8, :8] = 1
                    ids[8:, 8:] = 4
                    if frame == 2:
                        ids[:] = 255  # Complete occlusion is not semantic absence.
                    for packet in stream.encode(av.VideoFrame.from_ndarray(ids, format='gray')):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            columns[f'observation.seg.{camera}'] = [dict(Path=relative, Timestamp=[s]) for s in columns['timestamp']]
        pq.write_table(pa.table(columns), path)
    config = copy.deepcopy(f.config)
    config['depth']['path_pattern'] = '{task}-seg/depth/{episode:06d}/{frame:06d}.npy'
    config['keypoints'] = dict(method='gripper')
    return root, config


@pytest.mark.parametrize('version', [3, 4])
def test_real_seg_export_and_teacher_or_v4_annotations(replay_fixture, tmp_path, version, monkeypatch):
    from finetune.OHT.data.audit import audit
    from finetune.OHT.data.replay import build
    from finetune.OHT.data.role_teacher import build_teacher
    f = replay_fixture
    root, config = make_seg_source(f, tmp_path)
    manifest = tmp_path / 'audit.json'
    audit(root, manifest, fractions=(1, 0, 0), progress=False)
    replay = tmp_path / 'replay'
    if version == 4:
        from finetune.OHT.data.chunk_replay import build
        build(root, manifest, config, replay, sample_stride=2, chunk_config=dict(horizon=4), progress=False)
    else:
        build(root, manifest, config, replay, sample_stride=2)
    mapping = tmp_path / 'mapping.yaml'
    mapping.write_text('schema: oht_seg_roles_v1\ntasks:\n  disassemble_left:\n'
                       '    default: {target: [1], reference: unknown}\n'
                       '    phases:\n      2: {target: [1], reference: [4]}\n', encoding='utf-8')
    output = tmp_path / 'roles'
    result = export_roles(root, manifest, replay, mapping, output)
    rows = read_jsonl(replay / 'samples.jsonl')
    assert result['samples'] == len(rows)
    annotations = read_jsonl(output / 'annotations.jsonl')
    for annotation, row in zip(annotations, sorted(rows, key=lambda r: (r['episode_index'], r['frame']))):
        with np.load(output / annotation['target']['mask_path']) as masks:
            expected = np.zeros((8, 8), bool)
            if row['frame'] != 2:
                expected[:4, :4] = True
            np.testing.assert_array_equal(masks['wrist'], expected)
        assert annotation['target']['present'] and annotation['target']['known']
        assert annotation['reference']['source'] == ('visible_surface' if annotation['instruction_id'] == 2 else 'unknown')
    provenance = json.loads((output / 'provenance.json').read_text())
    assert len(provenance['video_sha256']) == 15
    if version == 3:
        teacher = build_teacher(replay, output / 'annotations.jsonl', tmp_path / 'teacher', point_count=16)
        assert len(teacher['samples']) == len(rows)
        occluded = next(r for r in rows if r['frame'] == 2)
        with np.load(tmp_path / 'teacher' / teacher['samples'][occluded['id']]['path']) as fields:
            assert fields['oracle_target_present']
            assert not fields['oracle_target_object_valid']
    # Reusing the old acquisition's replay is rejected before output is published.
    with pytest.raises(ValueError, match='does not match replay'):
        export_roles(root, f.manifest, replay, mapping, tmp_path / 'wrong-source')
    assert not (tmp_path / 'wrong-source').exists()
    from finetune.OHT.data.video import EpisodeVideos
    original_read = EpisodeVideos.read
    def wrong_rgb(self, reference, **kwargs):
        pixels = original_read(self, reference, **kwargs)
        if kwargs.get('format', 'rgb24') == 'rgb24':
            pixels = pixels.copy()
            pixels[0, 0] ^= 1
        return pixels
    monkeypatch.setattr(EpisodeVideos, 'read', wrong_rgb)
    bad_output = tmp_path / 'wrong-rgb'
    with pytest.raises(ValueError, match='Source RGB frame/sampling differs'):
        export_roles(root, manifest, replay, mapping, bad_output)
    assert not (bad_output / 'annotations.jsonl').exists()


@pytest.mark.parametrize('version', [3, 4])
def test_full_pipeline_starts_from_raw_only_without_any_contract(replay_fixture, tmp_path, version):
    import subprocess
    import sys
    from pathlib import Path
    import yaml
    root, config = make_seg_source(replay_fixture, tmp_path)
    assert not list(root.rglob('contract.json'))
    assert not list(root.rglob('audit.json'))
    config_path = tmp_path / 'dataset.yaml'
    config_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    output = tmp_path / 'generated'
    script = Path(__file__).parents[1] / 'tools/prepare_oht_seg_data.py'
    result = subprocess.run([sys.executable, '-u', str(script), '--root', str(root),
        '--config', str(config_path), '--output', str(output), '--replay-version', str(version),
        '--sample-stride', '2', '--chunk-horizon', '4', '--workers', '1', '--visualize-every', '0'],
        capture_output=True, text=True, encoding='utf-8', timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    status = json.loads((output / 'pipeline.json').read_text())
    assert status['status'] == 'complete'
    replay = output / f'replay-source-v{version}'
    assert (output / 'audit.json').is_file()
    assert (replay / 'contract.json').is_file() and (replay / 'complete.json').is_file()
    assert (output / 'seg-role-annotations/annotations.jsonl').is_file()
    assert (output / 'role-teacher/manifest.json').is_file() == (version == 3)
    assert not list(root.rglob('contract.json'))
