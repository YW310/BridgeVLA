"""Target-only workspace preflight matches V4 build without RGB-D decoding."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from finetune.OHT.data.common import read_jsonl
from finetune.OHT.data.chunk_replay import _target_bounds
from finetune.OHT.data.workspace import check_workspace
from tests.test_oht_chunk_replay import chunk_fixture, replay_fixture  # noqa: F401


def expected_targets(fixture):
    all_points = []
    for row in read_jsonl(fixture.root / 'samples.jsonl'):
        with np.load(fixture.root / row['targets'], allow_pickle=False) as target:
            all_points.extend(target['action_chunk'][target['action_valid'], :3])
            if target['key_valid']:
                all_points.append(target['key_action'][:3])
    return np.asarray(all_points)


def test_preflight_matches_built_key_and_dense_target_coverage_without_video(chunk_fixture, monkeypatch):
    from finetune.OHT.data import video
    def no_video(*args, **kwargs):
        raise AssertionError('Workspace checks must not decode videos')
    monkeypatch.setattr(video, 'EpisodeVideos', no_video)
    f = chunk_fixture
    report = check_workspace(f.source.root, f.source.manifest, f.config, 2, chunk_config=dict(horizon=4))
    expected = expected_targets(f)
    assert report['valid'] and report['outside_target_frames'] == 0
    assert report['valid_windows'] == f.count
    np.testing.assert_array_equal(report['target_min'], expected.min(0))
    np.testing.assert_array_equal(report['target_max'], expected.max(0))
    assert report['suggested_scene_bounds'] == f.config['scene_bounds']


def test_outside_targets_report_axes_and_suggest_only_expanding_workspace(chunk_fixture):
    f = chunk_fixture
    config = deepcopy(f.config)
    config['scene_bounds'][3] = .115
    original = deepcopy(config)
    report = check_workspace(f.source.root, f.source.manifest, config, 2, chunk_config=dict(horizon=4))
    assert config == original
    assert not report['valid'] and report['outside_target_frames'] > 0
    assert all(item['outside_axes'] == ['x'] for item in report['examples'])
    assert len(report['examples']) <= 20
    suggestion = np.array(report['suggested_scene_bounds'])
    assert (suggestion[:3] <= config['scene_bounds'][:3]).all()
    assert (suggestion[3:] >= config['scene_bounds'][3:]).all()
    assert suggestion[3] >= report['target_max'][0] + .02 - 1e-8


def test_keep_bounds_do_not_limit_legal_tcp_targets(chunk_fixture):
    f = chunk_fixture
    config = deepcopy(f.config)
    config['point_cloud_filter'] = dict(enabled=True, keep_bounds=[10., 10., 10., 11., 11., 11.], exclude_boxes=[])
    report = check_workspace(f.source.root, f.source.manifest, config, 2, chunk_config=dict(horizon=4))
    assert report['valid']  # This says nothing about surviving observed surfaces.


def test_target_error_identifies_sample_target_coordinates_and_upper_exclusive_axis(chunk_fixture):
    row = read_jsonl(chunk_fixture.root / 'samples.jsonl')[0]
    with np.load(chunk_fixture.root / row['targets'], allow_pickle=False) as source:
        targets = {key: source[key] for key in source.files}
    index = np.flatnonzero(targets['action_valid'])[0]
    targets['action_chunk'][index, 0] = 1.
    with pytest.raises(ValueError, match='outside scene_bounds') as exc:
        _target_bounds(targets, [-1., -1., -1., 1., 1., 1.], row['id'])
    message = str(exc.value)
    assert row['id'] in message and f'action_chunk[{index}]' in message
    assert "outside_axes=['x']" in message and 'valid_target_max=' in message


def test_workspace_cli_writes_report_on_failure_and_refuses_overwrite(chunk_fixture, tmp_path, capsys):
    spec = importlib.util.spec_from_file_location('workspace_cli', Path(__file__).parents[1] / 'tools/check_oht_workspace.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    f = chunk_fixture
    config = deepcopy(f.config)
    config['scene_bounds'][3] = .115
    path, output = tmp_path / 'data.yaml', tmp_path / 'coverage.json'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    args = ['--root', str(f.source.root), '--manifest', str(f.source.manifest), '--config', str(path),
            '--sample-stride', '2', '--chunk-horizon', '4', '--output', str(output)]
    assert cli.main(args) == 1
    assert not json.loads(output.read_text())['valid']
    assert json.loads(capsys.readouterr().out)['outside_target_frames'] > 0
    with pytest.raises(FileExistsError):
        cli.main(args)
