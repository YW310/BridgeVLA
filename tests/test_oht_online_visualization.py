"""Online images come from one supplied forward, with no labels or simulator dependencies."""
from copy import deepcopy
import json
import re
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image
import pytest

from finetune.OHT.runtime.online_visualization import OnlineVisualizer, viewer_html


def payload(two_stages=True):
    rgb = np.full((3, 8, 8, 3), .5, np.float32)
    maps = np.full((3, 8, 8), 1 / 64, np.float32)
    projection = np.zeros((3, 2, 4))
    projection[:, 0, 0] = projection[:, 1, 1] = 2
    projection[:, :, 3] = 3.5
    source = dict(rgb=rgb, heatmap=maps, world_to_pixel=projection,
                  target_prior=maps.copy(), reference_prior=maps.copy())
    return dict(stages={'mvt1': source, **({'mvt2': deepcopy(source)} if two_stages else {})},
                heatmap_waypoint_world=np.array([.2, -.1, .4]))


def request(step=0):
    return dict(request_id=f'request-{step}', episode_id='episode-one', step=step, timestamp=step * .02,
                instruction='<script>untrusted instruction</script>', model_id='model-one')


def submit(writer, data=None, step=0, mode='eef', **kwargs):
    action = (np.array([.2, -.1, .4, 0, 0, 0, 1, 1]) if mode == 'eef' else
              np.array([.1, 0, 0, 0, 0, 0, 1]))
    options = dict(mode=mode, **kwargs)
    observation = dict(tcp_pose=np.array([1., 2., 0., 0., 0., np.sqrt(.5), np.sqrt(.5)]),
                       gripper_pose=np.array([999.] * 7), action=np.array([999.] * 8), teacher_map=np.ones(3))
    return writer.submit(payload() if data is None else data, request(step), action, options, observation)


def wait_for(writer, *, entry_id=None, error=False):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = writer.latest()
        if error and state['error'] and state['writer_status'] == 'error':
            return state
        if (not error and state['latest'] and state['latest']['entry_id'] == entry_id
                and state['writer_status'] == 'idle'):
            return state
        time.sleep(.005)
    raise AssertionError(f'Writer did not produce the expected state: {writer.latest()}')


@pytest.fixture
def writer(tmp_path):
    instance = OnlineVisualizer(tmp_path / 'online', every=2, keep=2)
    yield instance
    instance.close(wait=True)


@pytest.mark.parametrize('mode', ['eef', 'relative_eef', 'joint', 'relative_joint'])
def test_label_free_images_probabilities_and_mode_specific_tcp_projection(writer, mode):
    result = submit(writer, mode=mode, **({'translation_frame': 'body'} if mode == 'relative_eef' else {}))
    assert result['status'] == 'queued' and re.fullmatch('[0-9a-f]{32}', result['entry_id'])
    record = wait_for(writer, entry_id=result['entry_id'])['latest']
    assert record['label_free'] and record['heatmap_is_auxiliary'] == (mode != 'eef')
    assert set(record['images']) == {'mvt1', 'mvt2'}
    if 'joint' in mode:
        assert record['policy_tcp_world'] is record['policy_pose_xyzw'] is None
    elif mode == 'relative_eef':
        np.testing.assert_allclose(record['policy_tcp_world'], [1., 2.1, 0], atol=1e-7)
    else:
        np.testing.assert_allclose(record['policy_tcp_world'], [.2, -.1, .4])
    folder = writer.root / result['entry_id']
    saved = json.loads((folder / 'prediction.json').read_text(encoding='utf-8'))
    assert saved == record
    assert 'gt_tcp_world' not in record and 'target_action' not in record and 'teacher_map' not in record
    for name in ('mvt1', 'mvt2'):
        with Image.open(folder / f'{name}.png') as image:
            image.verify()
        with np.load(folder / f'{name}.npz', allow_pickle=False) as arrays:
            assert 'gt_heatmap' not in arrays and 'gt_pixels' not in arrays
            assert ('policy_pixels' in arrays) == ('joint' not in mode)
            np.testing.assert_array_equal(arrays['heatmap'], payload()['stages'][name]['heatmap'])
            np.testing.assert_allclose(arrays['heatmap_waypoint_pixels'],
                                       arrays['world_to_pixel'] @ np.r_[record['heatmap_waypoint_world'], 1.])


def test_sampling_unique_run_directories_and_returned_state_are_independent(writer, tmp_path):
    assert writer.due(0) and writer.due(2) and not writer.due(1) and not writer.due(-1)
    assert writer.latest()['latest'] is None
    other = OnlineVisualizer(tmp_path / 'online')
    try:
        assert other.root != writer.root and other.root.parent == writer.root.parent
        result = submit(writer, payload(two_stages=False))
        record = wait_for(writer, entry_id=result['entry_id'])
        record['latest']['predicted_action'][0] = 900
        assert writer.latest()['latest']['predicted_action'][0] != 900
        assert set(record['latest']['images']) == {'mvt1'}
    finally:
        other.close(wait=True)


def test_pending_snapshots_do_not_alias_mutable_inputs_or_export_label_fields(writer, monkeypatch):
    started, release = threading.Event(), threading.Event()
    original = writer._write

    def blocked(*job):
        started.set()
        assert release.wait(3)
        return original(*job)

    monkeypatch.setattr(writer, '_write', blocked)
    data = payload()
    data['stages']['mvt1']['gt_heatmap'] = np.ones((3, 8, 8))
    data['stages']['mvt1']['teacher_target'] = np.ones((3, 8, 8))
    data['gt_tcp_world'] = np.full(3, 999.)
    try:
        result = submit(writer, data)
        assert started.wait(3)
        data['stages']['mvt1']['heatmap'][:] = 999
        release.set()
        wait_for(writer, entry_id=result['entry_id'])
    finally:
        release.set()
    with np.load(writer.root / result['entry_id'] / 'mvt1.npz', allow_pickle=False) as arrays:
        assert 'teacher_target' not in arrays and 'gt_heatmap' not in arrays
        assert arrays['heatmap'].max() < 1


def test_busy_writer_drops_excess_jobs_without_waiting_for_disk(writer, monkeypatch):
    started, release = threading.Event(), threading.Event()
    original = writer._write

    def blocked(*job):
        started.set()
        assert release.wait(3)
        return original(*job)

    monkeypatch.setattr(writer, '_write', blocked)
    try:
        first = submit(writer)
        assert first['status'] == 'queued' and started.wait(3)
        second = submit(writer, step=1)
        third = submit(writer, step=2)
        assert second['status'] == 'queued'
        assert third == dict(status='dropped', reason='writer_busy')
        assert writer.latest()['dropped'] == 1
        release.set()
        wait_for(writer, entry_id=second['entry_id'])
    finally:
        release.set()


def test_disk_retention_is_bounded_and_latest_snapshot_is_retained(writer):
    entries = []
    for step in range(4):
        result = submit(writer, step=step)
        entries.append(result['entry_id'])
        wait_for(writer, entry_id=entries[-1])
    assert {path.name for path in writer.root.iterdir()} == set(entries[-2:])
    assert writer.latest()['latest']['step'] == 3


def test_latest_publishes_the_complete_replacement_before_pruning_previous_files(writer, monkeypatch):
    writer.keep = 1
    first = submit(writer)
    wait_for(writer, entry_id=first['entry_id'])
    pruning, release = threading.Event(), threading.Event()
    original = writer._remove

    def blocked_remove(directory):
        current = writer.latest()['latest']
        assert current['entry_id'] != first['entry_id']
        assert (writer.root / current['entry_id'] / 'prediction.json').is_file()
        assert (writer.root / current['entry_id'] / 'mvt1.png').is_file()
        pruning.set()
        assert release.wait(3)
        return original(directory)

    monkeypatch.setattr(writer, '_remove', blocked_remove)
    try:
        second = submit(writer, step=1)
        assert pruning.wait(3)
        assert writer.latest()['latest']['entry_id'] == second['entry_id']
        assert (writer.root / first['entry_id']).is_dir()
        release.set()
        wait_for(writer, entry_id=second['entry_id'])
    finally:
        release.set()
    assert not (writer.root / first['entry_id']).exists()
    assert (writer.root / second['entry_id'] / 'prediction.json').is_file()


def test_export_failure_preserves_previous_images_and_next_success_recovers(writer):
    first = submit(writer)
    previous = wait_for(writer, entry_id=first['entry_id'])['latest']
    broken = payload()
    broken['stages']['mvt1']['heatmap'] = np.zeros((1, 1))
    assert submit(writer, broken, step=1)['status'] == 'queued'
    failed = wait_for(writer, error=True)
    assert failed['latest'] == previous and 'shapes do not match' in failed['error']
    assert (writer.root / first['entry_id'] / 'mvt1.png').is_file()
    assert not any(path.name.startswith('.pending-') for path in writer.root.iterdir())
    next_result = submit(writer, step=2)
    recovered = wait_for(writer, entry_id=next_result['entry_id'])
    assert recovered['error'] is None


def test_outside_policy_projection_is_reported_without_clipping(writer):
    action = np.array([100., 0, 0, 0, 0, 0, 1, 1])
    result = writer.submit(payload(), request(), action, {}, {})
    record = wait_for(writer, entry_id=result['entry_id'])['latest']
    assert record['stages']['mvt1']['policy_in_view'] == [False] * 3
    with np.load(writer.root / result['entry_id'] / 'mvt1.npz', allow_pickle=False) as arrays:
        assert (arrays['policy_pixels'][:, 0] > 8).all()


@pytest.mark.parametrize('field', ['target_prior', 'reference_prior', 'relation_anchor'])
def test_low_resolution_role_maps_are_resized_only_for_display_and_preserved_in_npz(writer, field):
    data = payload()
    small_map = np.arange(18, dtype=np.float32).reshape(3, 2, 3)
    data['stages']['mvt1'][field] = small_map
    result = submit(writer, data)
    wait_for(writer, entry_id=result['entry_id'])
    folder = writer.root / result['entry_id']
    with Image.open(folder / 'mvt1.png') as image:
        image.verify()
    with np.load(folder / 'mvt1.npz', allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays[field], small_map)
        assert arrays['rgb'].shape[1:3] == (8, 8)


def test_request_paths_never_select_files_and_stage_traversal_is_rejected(writer):
    malicious = request()
    malicious['request_id'] = '../../outside'
    action = np.array([.2, -.1, .4, 0, 0, 0, 1, 1])
    result = writer.submit(payload(), malicious, action, {}, {})
    record = wait_for(writer, entry_id=result['entry_id'])['latest']
    assert record['request_id'] == '../../outside'
    assert result['url'] == f'/visualization/files/{result["entry_id"]}/prediction.json'
    assert {p.name for p in writer.root.iterdir()} == {result['entry_id']}
    invalid = payload()
    invalid['stages']['../../outside'] = invalid['stages'].pop('mvt1')
    assert submit(writer, invalid)['status'] == 'error'
    assert writer.latest()['latest']['entry_id'] == result['entry_id']


def test_close_stops_accepting_work_without_requiring_simulator_cleanup(writer):
    writer.close(wait=True)
    assert submit(writer) == dict(status='dropped', reason='closed')
    assert not writer._thread.is_alive()


@pytest.mark.parametrize('kwargs', [dict(every=0), dict(every=True), dict(keep=0), dict(keep=1.5)])
def test_invalid_settings_fail_before_starting_thread(tmp_path, kwargs):
    with pytest.raises(ValueError, match='positive integers'):
        OnlineVisualizer(tmp_path, **kwargs)


def test_viewer_uses_safe_text_nodes_and_polls_online_metadata():
    page = viewer_html()
    assert "fetch('/visualization/latest'" in page and 'setInterval(refresh,1000)' in page
    assert 'textContent' in page and 'innerHTML' not in page
    assert 'Coarse stage' in page and 'Refine stage' in page
    assert 'state.capture_error' in page and 'state.reason' in page
    assert 'server:location.origin' in page and 'task:item.task' in page


def test_writer_import_does_not_load_torch_or_isaaclab():
    code = ('import sys; from finetune.OHT.runtime.online_visualization import OnlineVisualizer; '
            'assert "torch" not in sys.modules; '
            'assert not any(name.startswith("isaaclab") for name in sys.modules)')
    subprocess.run([sys.executable, '-X', 'utf8', '-c', code], check=True, capture_output=True, text=True)
