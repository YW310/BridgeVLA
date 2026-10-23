"""V4 rebuild/validation integration using actual Parquet, MP4 and depth."""
import copy
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from tests.test_oht_migration import replay_fixture  # noqa: F401
from finetune.OHT.cli import main
from finetune.OHT.data.audit import audit
from finetune.OHT.data.chunk_dataset import OHTChunkDataset, collate_chunk, load_chunk_contract
from finetune.OHT.data.chunk_replay import build
from finetune.OHT.data.common import digest, file_digest, read_jsonl, write_json
from finetune.OHT.data.replay import load_contract


@pytest.fixture(scope='module')
def chunk_fixture(replay_fixture, tmp_path_factory):
    f = replay_fixture
    output = tmp_path_factory.mktemp('v4') / 'replay-source-v4'
    config = copy.deepcopy(f.config)
    config['keypoints'] = dict(method='gripper')
    count = build(f.root, f.manifest, config, output, sample_stride=2,
                  chunk_config=dict(horizon=4), progress=False)
    return SimpleNamespace(source=f, root=output, config=config, count=count)


def rebind_index(root, rows):
    (root / 'samples.jsonl').write_text(
        ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    contract = json.loads((root / 'contract.json').read_text(encoding='utf-8'))
    complete = json.loads((root / 'complete.json').read_text(encoding='utf-8'))
    contract['index_sha256'] = file_digest(root / 'samples.jsonl')
    contract.pop('sha256')
    contract['sha256'] = digest(contract)
    complete.update(index_sha256=contract['index_sha256'], contract_sha256=contract['sha256'])
    write_json(root / 'contract.json', contract)
    write_json(root / 'complete.json', complete)


def test_v4_raw_rebuild_has_real_future_targets_and_retains_post_key_tail(chunk_fixture):
    f = chunk_fixture
    contract = load_chunk_contract(f.root)
    assert contract['schema'] == 'oht_bridgevla_v4'
    assert contract['replay_version'] == 4
    assert contract['chunk']['horizon'] == 4
    assert contract['chunk']['action_dt'] == pytest.approx(1 / 60)
    assert contract['audit_manifest_sha256'] == json.loads(f.source.manifest.read_text())['manifest_sha256']
    old_splits = {(row['task'], row['episode_index']): row['split']
                  for row in read_jsonl(f.source.replay / 'samples.jsonl')}
    for split in ('train', 'val', 'test'):
        data = OHTChunkDataset(f.root, split)
        assert data.validate_all(workers=2) == len(data)
        for index, row in enumerate(data.samples):
            sample = data[index]
            assert row['split'] == old_splits[row['task'], row['episode_index']]
            assert 'instruction_id' not in sample
            with np.load(f.root / row['observation'], allow_pickle=False) as observation:
                assert 'action_chunk' not in observation.files
            valid_frames = sample['action_frames'][sample['action_valid']]
            expected_x = .1 + row['episode_index'] * .01 + valid_frames * .02
            np.testing.assert_allclose(sample['action_chunk'][sample['action_valid'], 0], expected_x, atol=1e-7)
            assert not np.any(sample['action_chunk'] == 900)  # raw action commands are audit-only
        tail_index = next(i for i, row in enumerate(data.samples) if row['frame'] == 5)
        tail = data[tail_index]
        assert not tail['key_valid'] and tail['key_frame'] == -1
        assert tail['action_valid'].tolist() == [True, False, False, False]
        assert tail['action_frames'].tolist() == [6, -1, -1, -1]
    with pytest.raises(ValueError, match='single-keypoint trainer'):
        load_contract(f.root)


def test_v4_collate_preserves_history_and_future_axes_and_float_labels(chunk_fixture):
    data = OHTChunkDataset(chunk_fixture.root)
    batch = collate_chunk([data[0], data[-1]])
    assert tuple(batch['action_chunk'].shape) == (2, 1, 4, 8)
    assert tuple(batch['joint_chunk'].shape) == (2, 1, 4, 7)
    assert tuple(batch['key_action'].shape) == (2, 1, 8)
    assert tuple(batch['action_valid'].shape) == (2, 1, 4)
    assert batch['action_chunk'].is_floating_point()
    assert batch['action_timestamps'].dtype.__str__() == 'torch.float64'
    assert batch['action_valid'].dtype.__str__() == 'torch.bool'
    assert tuple(batch['gripper_open01'].shape) == (2, 1)


@pytest.mark.parametrize('workers', [1, 3])
def test_validate_cli_auto_detects_v4_and_checks_every_split(chunk_fixture, capsys, workers):
    assert main(['validate', '--replay', str(chunk_fixture.root), '--workers', str(workers), '--quiet']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['valid'] and sum(result['samples'].values()) == chunk_fixture.count
    assert result['contract'] == load_chunk_contract(chunk_fixture.root)['sha256']


@pytest.mark.parametrize('field', ['observation', 'targets'])
def test_v4_validation_checks_both_payload_hashes(chunk_fixture, tmp_path, field):
    root = tmp_path / 'copy'
    shutil.copytree(chunk_fixture.root, root)
    data = OHTChunkDataset(root)
    with (root / data.samples[-1][field]).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match=field + ' checksum mismatch'):
        data.validate_all(workers=3)


def test_v4_validation_rejects_bad_masks_even_with_updated_hashes(chunk_fixture, tmp_path):
    root = tmp_path / 'copy'
    shutil.copytree(chunk_fixture.root, root)
    rows = read_jsonl(root / 'samples.jsonl')
    row = next(row for row in rows if row['split'] == 'train' and row['frame'] == 5)
    path = root / row['targets']
    with np.load(path, allow_pickle=False) as source:
        targets = {key: source[key] for key in source.files}
    targets['action_valid'][1] = True
    np.savez_compressed(path, **targets)
    row['targets_sha256'] = file_digest(path)
    rebind_index(root, rows)
    with pytest.raises(ValueError, match='available future prefix'):
        OHTChunkDataset(root).validate_all()


def test_v4_rejects_group_leakage_even_with_rebound_manifest(chunk_fixture, tmp_path):
    root = tmp_path / 'copy'
    shutil.copytree(chunk_fixture.root, root)
    rows = read_jsonl(root / 'samples.jsonl')
    rows[0]['split'] = 'val' if rows[0]['split'] != 'val' else 'train'
    rebind_index(root, rows)
    with pytest.raises(ValueError, match='crosses dataset splits|episode metadata'):
        load_chunk_contract(root)


def test_v4_keeps_no_gripper_event_episodes_and_reports_gap_windows(replay_fixture, tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    raw = tmp_path / 'raw'
    shutil.copytree(replay_fixture.root, raw)
    selected = raw / 'assemble_left/lerobot_dataset/data/chunk-000/episode_000000.parquet'
    columns = pq.read_table(selected).to_pydict()
    for state in columns['observation.state']:
        state[-1] = -.006
    columns['observation.gripper_joints'] = [[-.003, -.003]] * len(columns['timestamp'])
    columns['timestamp'] = [stamp + (.1 if i >= 3 else 0) for i, stamp in enumerate(columns['timestamp'])]
    pq.write_table(pa.table(columns), selected)
    manifest = tmp_path / 'audit.json'
    audit(raw, manifest, fractions=(1 / 3, 1 / 3, 1 / 3))
    config = copy.deepcopy(replay_fixture.config)
    config['keypoints'] = dict(method='gripper')
    root = tmp_path / 'v4'
    build(raw, manifest, config, root, sample_stride=2, chunk_config=dict(horizon=4), progress=False)
    rows = [row for row in read_jsonl(root / 'samples.jsonl')
            if row['task'] == 'assemble_left' and row['episode_index'] == 0]
    assert rows and all(row['target_frame'] == -1 for row in rows)
    for split in {row['split'] for row in rows}:
        OHTChunkDataset(root, split).validate_all()
    complete = json.loads((root / 'complete.json').read_text())
    assert complete['skipped_window_counts']['timestamp_gap'] > 0


def test_build_cli_v4_is_full_rebuild_and_refuses_overwrite(replay_fixture, tmp_path, capsys):
    import yaml
    config = copy.deepcopy(replay_fixture.config)
    config['keypoints'] = dict(method='gripper')
    path = tmp_path / 'dataset.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    output = tmp_path / 'v4'
    args = ['build', '--root', str(replay_fixture.root), '--manifest', str(replay_fixture.manifest),
            '--config', str(path), '--output', str(output), '--replay-version', '4',
            '--chunk-horizon', '3', '--action-frame-stride', '1', '--sample-stride', '3', '--quiet']
    assert main(args) == 0
    assert load_chunk_contract(output)['chunk']['horizon'] == 3
    assert 'Built' in capsys.readouterr().out
    with pytest.raises(FileExistsError, match='new V4 replay directory'):
        main(args)


def test_chunk_cli_options_cannot_silently_affect_legacy_build():
    with pytest.raises(SystemExit) as error:
        main(['build', '--root', 'missing', '--manifest', 'missing', '--config', 'missing',
              '--output', 'missing', '--chunk-horizon', '16'])
    assert error.value.code == 2


@pytest.mark.parametrize('corruption', ['nonunit_tcp', 'gripper_features', 'key_chunk_conflict'])
def test_v4_semantic_checks_survive_rehashed_payloads(chunk_fixture, tmp_path, corruption):
    root = tmp_path / 'copy'
    shutil.copytree(chunk_fixture.root, root)
    rows = read_jsonl(root / 'samples.jsonl')
    row = next(row for row in rows if row['split'] == 'train' and row['frame'] == 0)
    field = 'targets' if corruption == 'key_chunk_conflict' else 'observation'
    path = root / row[field]
    with np.load(path, allow_pickle=False) as source:
        values = {key: source[key] for key in source.files}
    if corruption == 'nonunit_tcp':
        values['tcp_pose'][3:] *= 2
        row['current_tcp'] = values['tcp_pose'].tolist()
        error = 'unit norm'
    elif corruption == 'gripper_features':
        values['gripper_open01'] = np.asarray(.5, dtype=np.float32)
        error = 'opening features'
    else:
        values['key_action'][0] += .01
        error = 'same future frame'
    np.savez_compressed(path, **values)
    row[field + '_sha256'] = file_digest(path)
    rebind_index(root, rows)
    with pytest.raises(ValueError, match=error):
        OHTChunkDataset(root).validate_all()


def test_v4_optional_object_and_html_previews_bind_final_contract(replay_fixture, tmp_path):
    manifest = json.loads(replay_fixture.manifest.read_text())
    manifest['episodes'] = manifest['episodes'][:1]
    manifest.pop('manifest_sha256')
    manifest['manifest_sha256'] = digest(manifest)
    manifest_path = tmp_path / 'audit.json'
    write_json(manifest_path, manifest)
    root = tmp_path / 'v4'
    config = copy.deepcopy(replay_fixture.config)
    config['keypoints'] = dict(method='gripper')
    build(replay_fixture.root, manifest_path, config, root, sample_stride=2,
          chunk_config=dict(horizon=4), visualize_every=100,
          object_cloud_config=dict(source='depth', default_size=[.2, .2, .2]), progress=False)
    contract = load_chunk_contract(root)
    row = read_jsonl(root / 'samples.jsonl')[0]
    report = json.loads((root / row['object_clouds']).read_text(encoding='utf-8'))
    assert report['contract_sha256'] == contract['sha256']
    assert (root / 'preview/index.html').is_file()
    assert (root / 'preview' / row['id'] / 'preview.png').is_file()
