"""GT chunk previews preserve source labels and reuse the recorded camera geometry."""
import copy
import shutil

import numpy as np
import pytest
from PIL import Image

from tests.test_oht_chunk_replay import chunk_fixture, replay_fixture  # noqa: F401
from finetune.OHT.data.common import file_digest, read_jsonl
from finetune.OHT.data.chunk_replay import build
from finetune.OHT.data.replay_viewer import export_replay_viewer
from finetune.OHT.data.visualization import CHUNK, _chunk_data, _draw_chunk, save_preview


def scene():
    observation = dict(wrist_rgb=np.full((3, 32, 32), 60, np.uint8),
        wrist_depth=np.ones((1, 32, 32), np.float32),
        wrist_point_cloud=np.zeros((3, 32, 32), np.float32),
        wrist_camera_intrinsics=np.array([[16., 0, 16], [0, 16, 16], [0, 0, 1]]),
        wrist_camera_extrinsics=np.eye(4))
    config = dict(cameras={'wrist': {}}, scene_bounds=[-1, -1, .5, 1, 1, 1.5],
                  depth=dict(limits=[.001, 4.]))
    sample = dict(id='test/000000/000000', split='train', frame=0, target_frame=10, timestamp=0.,
                  goal='Inspect measured dense path', labels=dict(gripper_pose=[.6, .5, 1, 0, 0, 0, 1]))
    actions = np.tile([0., 0, 1, 0, 0, 0, 1, 1], (5, 1)).astype(np.float32)
    actions[:3, :2] = [[-.4, -.25], [-.05, .3], [.35, -.15]]
    actions[3:] = np.nan  # Padding cannot change bounds or appear at world zero.
    targets = dict(action_chunk=actions, action_valid=np.array([True, True, True, False, False]))
    current = np.array([-.6, -.35, 1, 0, 0, 0, 1])
    return observation, config, sample, current, targets


def has_chunk_color(image):
    return np.any(np.all(np.asarray(image) == CHUNK, axis=-1))


def test_preview_draws_masked_gt_in_camera_world_and_zoom_without_input_mutation(tmp_path):
    observation, config, sample, current, targets = scene()
    before = copy.deepcopy((observation, targets))
    base = save_preview(tmp_path / 'base.png', observation, config, sample, current_tcp=current)
    path = save_preview(tmp_path / 'chunk.png', observation, config, sample, current_tcp=current, chunk_targets=targets)
    with Image.open(path) as image, Image.open(base) as original:
        assert image.size == (original.width, original.height + 344)
        assert has_chunk_color(image.crop((0, 116, 256, 308)))  # Optical sensor projection.
        assert has_chunk_color(image.crop((0, 370, 340, 626)))  # World XY.
        assert has_chunk_color(image.crop((0, 970, 1024, 1310)))  # Future-only zoom.
    for old, new in zip(before, (observation, targets)):
        for name in old:
            np.testing.assert_array_equal(old[name], new[name])
    prepared = _chunk_data(targets, current)
    assert prepared['actions'].shape == (3, 8) and prepared['positions'].shape == (4, 3)
    np.testing.assert_array_equal(prepared['positions'][0], current[:3])


def test_no_future_key_and_single_valid_future_still_have_a_chunk_preview(tmp_path):
    observation, config, sample, current, targets = scene()
    sample['labels'] = {}
    sample['target_frame'] = -1
    targets['action_valid'][:] = [True, False, False, False, False]
    path = save_preview(tmp_path / 'tail.png', observation, config, sample, current_tcp=current, chunk_targets=targets)
    with Image.open(path) as image:
        assert has_chunk_color(image.crop((0, 116, 256, 308)))
        assert image.size == (1024, 1042)  # No synthetic key-centered view.


def test_chunk_projection_never_connects_across_hidden_or_outside_samples():
    class Draw:
        def __init__(self):
            self.lines, self.points = [], []
        def line(self, endpoints, **kwargs):
            self.lines.append(endpoints)
        def ellipse(self, bounds, **kwargs):
            self.points.append(bounds)
        def text(self, *args, **kwargs):
            pass
    draw = Draw()
    _draw_chunk(draw, np.array([[1., 1.], [np.nan, np.nan], [5., 1.], [6., 1.]]), [True, False, True, True])
    assert draw.lines == [[(5., 1.), (6., 1.)]]
    assert len(draw.points) == 3


def test_chunk_preview_rejects_discontinuous_or_nonfinite_valid_targets():
    _, _, _, current, targets = scene()
    targets['action_valid'][:] = [True, False, True, False, False]
    with pytest.raises(ValueError, match='valid prefix'):
        _chunk_data(targets, current)
    targets['action_valid'][:] = [True] * 5
    with pytest.raises(ValueError, match='nonfinite'):
        _chunk_data(targets, current)


def test_v4_build_previews_and_refresh_keep_observation_and_target_hashes(chunk_fixture, tmp_path):
    f = chunk_fixture
    rebuilt = tmp_path / 'with-preview'
    build(f.source.root, f.source.manifest, f.config, rebuilt, sample_stride=2,
          chunk_config=dict(horizon=4), visualize_every=100, progress=False)
    for name in ('contract.json', 'samples.jsonl'):
        assert (rebuilt / name).read_bytes() == (f.root / name).read_bytes()
    rows = read_jsonl(f.root / 'samples.jsonl')
    for row in rows:
        for name in ('observation', 'targets'):
            assert file_digest(rebuilt / row[name]) == file_digest(f.root / row[name])
    png = rebuilt / 'preview' / rows[0]['id'] / 'preview.png'
    with Image.open(png) as image:
        assert has_chunk_color(image)
    from tests.test_oht_replay_viewer import viewer_payload
    from tests.test_oht_html_preview import payload_from_html
    from finetune.OHT.data.chunk_dataset import load_chunk_contract
    checksum = load_chunk_contract(rebuilt)['sha256']
    viewer = viewer_payload(rebuilt / 'preview' / 'index.html')
    assert viewer['contract_sha256'] == checksum
    for entry in viewer['frames']:
        html = payload_from_html(rebuilt / 'preview' / entry['html'])
        assert html['contract_sha256'] == checksum
        assert html['preview_image'] == 'preview.png'
    assert not (f.root / 'preview').exists()  # Disabled previews produce no HTML/PNG.
    inputs = [f.root / name for name in ('contract.json', 'samples.jsonl', 'complete.json')]
    inputs += [f.root / row[name] for row in rows for name in ('observation', 'targets')]
    before = [file_digest(path) for path in inputs]
    output = tmp_path / 'refreshed'
    from finetune.OHT.cli import main
    assert main(['export-viewer', '--replay', str(f.root), '--output', str(output), '--every', '100',
                 '--max-points', '5']) == 0
    assert (output / 'index.html').is_file()
    with Image.open(output / rows[0]['id'] / 'preview.png') as image:
        assert has_chunk_color(image)
    assert [file_digest(path) for path in inputs] == before


def test_v4_basic_html_respects_external_preview_directory(replay_fixture, tmp_path):
    import json
    from finetune.OHT.data.common import digest, write_json
    from tests.test_oht_replay_viewer import viewer_payload
    manifest = json.loads(replay_fixture.manifest.read_text(encoding='utf-8'))
    manifest['episodes'] = manifest['episodes'][:1]
    manifest.pop('manifest_sha256')
    manifest['manifest_sha256'] = digest(manifest)
    audit = tmp_path / 'audit.json'
    write_json(audit, manifest)
    config = copy.deepcopy(replay_fixture.config)
    config['keypoints'] = dict(method='gripper')
    output, previews = tmp_path / 'v4', tmp_path / 'external-previews'
    build(replay_fixture.root, audit, config, output, chunk_config=dict(horizon=4),
          visualize_every=100, visualize_output_dir=previews, progress=False)
    frames = viewer_payload(previews / 'index.html')['frames']
    assert len(frames) == 1
    assert (previews / frames[0]['html']).is_file()
    assert (previews / frames[0]['image']).is_file()
    assert (output / 'complete.json').is_file()
    assert not (output / 'preview').exists()


def test_v4_refresh_checks_target_integrity_before_publishing(chunk_fixture, tmp_path):
    root = tmp_path / 'tampered'
    shutil.copytree(chunk_fixture.root, root)
    first = read_jsonl(root / 'samples.jsonl')[0]
    with (root / first['targets']).open('ab') as stream:
        stream.write(b'changed')
    output = tmp_path / 'viewer'
    with pytest.raises(ValueError, match='Targets changed'):
        export_replay_viewer(root, output, every=100, max_points=5)
    assert not output.exists()
