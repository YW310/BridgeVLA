"""Same-tick chunk plots, physical errors and inspectable training snapshots."""
import numpy as np
import pytest
import torch
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tests.test_oht_chunk_replay import chunk_fixture, replay_fixture  # noqa: F401
from tests.test_oht_chunk_pipeline import agent_data
from finetune.OHT import chunk_visualization as visualization
from finetune.OHT.data.chunk_dataset import collate_chunk
from finetune.OHT.training_log import TensorBoardLogger


def test_unsupervised_predicted_tail_cannot_change_plot_bounds_or_lines():
    predicted = np.array([[.1, .2, .3], [.11, .22, .3], [20., -20., 40.]])
    target = np.array([[.1, .2, .3], [.12, .22, .3], [0., 0., 0.]])
    valid = np.array([True, True, False])
    first = visualization.path_panel(predicted, target, valid)
    predicted[-1] = np.nan
    target[-1] = np.nan
    second = visualization.path_panel(predicted, target, valid)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
    with pytest.raises(ValueError, match='finite'):
        visualization.path_panel(predicted, target, np.ones(3, bool))


def test_world_translation_keeps_shared_plot_geometry_and_changes_only_axis_labels():
    target = np.array([[.1, .2, .3], [.12, .22, .34]])
    predicted = target + [.01, 0., 0.]
    shift = np.array([3., -2., 5.])
    first = visualization.path_panel(predicted, target)
    second = visualization.path_panel(predicted + shift, target + shift)
    # This area contains the equal-scale XY/XZ/YZ plots, not coordinate text.
    np.testing.assert_array_equal(np.asarray(first)[110:351], np.asarray(second)[110:351])


def test_plot_reports_metric_error_for_valid_ticks_and_current_pose(monkeypatch):
    texts = []
    original = visualization.ImageDraw.Draw

    class Draw:
        def __init__(self, image):
            self.actual = original(image)
        def text(self, xy, text, **kwargs):
            texts.append(text)
            return self.actual.text(xy, text, **kwargs)
        def __getattr__(self, key):
            return getattr(self.actual, key)

    monkeypatch.setattr(visualization.ImageDraw, 'Draw', Draw)
    target = np.array([[.1, 0., 0.], [.12, 0., 0.]])
    visualization.path_panel(target + [.01, 0., 0.], target,
                             current=[0., 0., 0.], sample_id='task/000001/000030', step=500,
                             pose_metrics=dict(chunk_rotation_error_degrees=30.,
                                               chunk_rotation_error_max_degrees=60., chunk_gripper_accuracy=.5))
    text = '\n'.join(texts)
    assert 'mean=10.00 mm' in text and 'max=10.00 mm' in text and 'last valid=10.00 mm' in text
    assert 'hold-current baseline=110.00 mm' in text
    assert 'task/000001/000030' in text and 'step=500' in text
    assert 'equal axis scale' in text and 'world XYZ (m)' in text
    assert 'mean=30.00 deg' in text and 'max=60.00 deg' in text and 'gripper accuracy=50.0%' in text


def test_training_snapshot_matches_same_sample_and_loss_forward(chunk_fixture, tmp_path):
    agent, data, _ = agent_data(chunk_fixture)
    tail = next(i for i in range(len(data)) if data[i]['action_valid'].sum() == 1)
    batch = collate_chunk([data[tail], data[0]])
    result = agent.update(batch, return_visualization=True)
    payload = result['train_visualization']
    assert payload['sample_id'] == data[tail]['sample_id']
    np.testing.assert_array_equal(payload['current'], data[tail]['tcp_pose'][:3])
    np.testing.assert_array_equal(payload['target'], data[tail]['action_chunk'][:, :3])
    np.testing.assert_array_equal(payload['valid'], data[tail]['action_valid'])
    expected_scene = visualization.scene_background(data[tail])
    for key in ('scene_points', 'scene_colors'):
        np.testing.assert_array_equal(payload[key], expected_scene[key])
    # Check exported coordinates, not a fresh post-optimizer prediction.
    expected = visualization.path_panel(payload['predicted'], payload['target'], payload['valid'],
        current=payload['current'], sample_id=payload['sample_id'], step=500, action_dt=payload['action_dt'],
        scene_points=payload['scene_points'], scene_colors=payload['scene_colors'], pose_metrics=payload['pose_metrics'])
    settings = dict(output_dir='images', save_png=True, tensorboard=True)
    with TensorBoardLogger(tmp_path / 'tensorboard') as logger:
        visualization.record(payload, 500, tmp_path, settings, logger)
    with Image.open(tmp_path / 'images/chunk_step_00000500.png') as saved:
        np.testing.assert_array_equal(np.asarray(saved), np.asarray(expected))
    with np.load(tmp_path / 'images/chunk_step_00000500.npz', allow_pickle=False) as arrays:
        for key in ('predicted', 'target', 'valid', 'current', 'scene_points', 'scene_colors',
                    'predicted_rotation_matrix', 'target_quaternion_xyzw', 'predicted_gripper_open01',
                    'target_gripper_open01', 'current_pose_xyzw'):
            np.testing.assert_array_equal(arrays[key], payload[key])
        for key, value in payload['pose_metrics'].items():
            assert arrays[key].item() == value
        assert arrays['sample_id'].item() == data[tail]['sample_id'] and arrays['step'] == 500
    events = EventAccumulator(str(tmp_path / 'tensorboard')).Reload()
    assert [e.step for e in events.Images('train_visualization/chunk')] == [500]


def test_meter_error_exposes_stationary_prediction_baseline(chunk_fixture):
    agent, data, _ = agent_data(chunk_fixture)
    with torch.no_grad():
        agent._net_mod.decoder.head.weight.zero_()
        agent._net_mod.decoder.head.bias[:3].zero_()
    batch = collate_chunk([data[0], data[-1]])
    result = agent.update(batch, step_optimizer=False)
    valid = batch['action_valid'][:, 0]
    target = batch['action_chunk'][:, 0, :, :3]
    expected = torch.linalg.vector_norm(batch['tcp_pose'][:, 0, None, :3] - target, dim=-1)[valid].mean().item()
    assert result['chunk_position_error_m'] == pytest.approx(expected)
    assert result['chunk_stationary_error_m'] == pytest.approx(expected)


@pytest.mark.parametrize('same_tcp', [False, True])
def test_training_heatmaps_project_first_action_and_key_with_each_stage_camera(tmp_path, same_tcp):
    from copy import deepcopy

    first = np.array([.1, .2, .3])
    waypoint = first.copy() if same_tcp else np.array([.7, .2, .3])
    projection = np.array([[[40., 0., 0., 10.], [0., 40., 0., 10.]]])
    refine = np.array([[[0., 0., -50., 40.], [50., 0., 0., 10.]]])
    stages = {name: dict(rgb=np.full((1, 64, 64, 3), .5),
                         heatmap=np.full((1, 64, 64), 1 / 4096), world_to_pixel=camera)
              for name, camera in (('mvt1', projection), ('mvt2', refine))}
    original = deepcopy(stages)
    payload = dict(predicted=np.array([first, first + .01]), target=np.array([first, first]),
                   valid=np.ones(2, bool), current=np.zeros(3),
                   policy=dict(stages=stages, heatmap_waypoint_world=waypoint))
    settings = dict(output_dir='images', save_png=True, tensorboard=True)
    with TensorBoardLogger(tmp_path / 'tensorboard') as logger:
        visualization.record(payload, 50, tmp_path, settings, logger)
    for name in stages:
        with np.load(tmp_path / f'images/{name}_step_00000050.npz') as arrays:
            camera = original[name]['world_to_pixel']
            np.testing.assert_allclose(arrays['policy_pixels'], camera @ np.r_[first, 1.])
            np.testing.assert_allclose(arrays['heatmap_waypoint_pixels'], camera @ np.r_[waypoint, 1.])
            np.testing.assert_array_equal(arrays['heatmap'], original[name]['heatmap'])
            np.testing.assert_array_equal(arrays['policy_tcp_world'], first)
            np.testing.assert_array_equal(arrays['heatmap_waypoint_world'], waypoint)
            np.testing.assert_array_equal(arrays['chunk_valid'], payload['valid'])
            for key, source in (('chunk_predicted', 'predicted'), ('chunk_target', 'target')):
                np.testing.assert_array_equal(arrays[key + '_world'], payload[source])
                expected = np.stack([camera @ np.r_[point, 1.] for point in payload[source]], axis=1)
                np.testing.assert_allclose(arrays[key + '_pixels'], expected)
            np.testing.assert_allclose(arrays['measured_tcp_pixels'], camera @ np.r_[payload['current'], 1.])
            assert arrays['step'] == 50
        with Image.open(tmp_path / f'images/{name}_step_00000050.png') as image:
            pixels = np.asarray(image)
        heatmap = pixels[96:160, 129:193]
        marked = pixels[96:160, 193:257]
        assert np.any(np.all(marked == [255, 0, 0], axis=-1))
        assert np.any(np.all(marked == [0, 255, 255], axis=-1))
        assert not np.array_equal(heatmap, marked)
        assert 'policy_pixels' not in stages[name]  # Never mutate the model's snapshot.
        assert 'chunk_valid' not in stages[name]
    events = EventAccumulator(str(tmp_path / 'tensorboard')).Reload()
    for name in stages:
        assert [event.step for event in events.Images('train_visualization/' + name)] == [50]


@pytest.mark.parametrize('gap', ['invalid', 'offscreen', 'nonfinite'])
def test_projected_paths_draw_on_rgb_and_tcp_tiles_without_bridging_gaps(gap):
    from finetune.OHT.runtime.online_visualization import _montage

    predicted = np.array([[[8., 16.], [16., 16.], [32., 16.], [48., 16.]]])
    target = predicted + [0., 32.]
    valid = np.ones(4, bool)
    if gap == 'invalid':
        valid[1] = False
    elif gap == 'offscreen':
        predicted[0, 1, 0] = target[0, 1, 0] = -10.
    else:
        predicted[0, 1] = target[0, 1] = np.nan
    stage = dict(rgb=np.full((1, 64, 64, 3), .5), heatmap=np.ones((1, 64, 64)),
                 chunk_predicted_pixels=predicted, chunk_target_pixels=target, chunk_valid=valid,
                 measured_tcp_pixels=np.array([[16., 32.]]))
    image = np.asarray(_montage(stage, 'mvt1', True))
    for column in (0, 2):
        tile = image[96:160, 65 + column * 64:65 + (column + 1) * 64]
        np.testing.assert_array_equal(tile[16, 40], [255, 96, 96])
        np.testing.assert_array_equal(tile[48, 40], [64, 219, 112])
        np.testing.assert_array_equal(tile[32, 16], [255, 255, 255])
        # A missing tick must not connect the visible endpoints on either side.
        for y in (16, 48):
            np.testing.assert_array_equal(tile[y, 24], tile[y, 58])
            np.testing.assert_array_equal(tile[y, 0], tile[y, 58])
    heatmap = image[96:160, 129:193]
    assert not np.any(np.all(heatmap == [255, 96, 96], axis=-1))
    assert not np.any(np.all(heatmap == [64, 219, 112], axis=-1))


def test_invalid_chunk_tail_cannot_change_mvt_montages_or_projected_valid_ticks(tmp_path):
    from copy import deepcopy
    from types import SimpleNamespace

    camera = np.array([[[40., 0., 0., 10.], [0., 40., 0., 10.]],
                       [[0., 40., 0., 10.], [0., 0., 40., 10.]]])
    stage = dict(rgb=np.full((2, 64, 64, 3), .5), heatmap=np.ones((2, 64, 64)), world_to_pixel=camera)
    payload = dict(predicted=np.array([[.1, .2, .3], [.4, .5, .6], [50., 50., 50.]]),
                   target=np.array([[.1, .4, .3], [.4, .7, .6], [50., 50., 50.]]),
                   valid=np.array([True, True, False]), current=np.zeros(3), sample_id='task/1/20', action_dt=1/60,
                   policy=dict(stages={'mvt1': stage}, heatmap_waypoint_world=np.array([.8, .2, .3])))
    frames = []
    logger = SimpleNamespace(writer=SimpleNamespace(add_image=lambda tag, image, *args, **kwargs:
                              frames.append(image.copy()) if tag == 'train_visualization/mvt1' else None),
                             flush=lambda: None)
    settings = dict(output_dir='images', save_png=True, tensorboard=True)
    visualization.record(payload, 20, tmp_path, settings, logger)
    changed = deepcopy(payload)
    changed['predicted'][-1] = changed['target'][-1] = np.nan
    visualization.record(changed, 21, tmp_path, settings, logger)
    np.testing.assert_array_equal(frames[0], frames[1])
    for step in (20, 21):
        with np.load(tmp_path / f'images/mvt1_step_{step:08d}.npz', allow_pickle=False) as arrays:
            assert arrays['chunk_predicted_pixels'].shape == (2, 3, 2)
            np.testing.assert_allclose(arrays['chunk_predicted_pixels'][0, :2], [[14., 18.], [26., 30.]])
            assert arrays['sample_id'].item() == 'task/1/20'
            assert arrays['action_dt'] == 1/60


def test_offscreen_tcp_marks_are_not_clamped_to_image_edges():
    from finetune.OHT.runtime.online_visualization import _montage

    stage = dict(rgb=np.full((1, 32, 32, 3), .5), heatmap=np.ones((1, 32, 32)),
                 policy_pixels=np.array([[-10., 16.]]),
                 heatmap_waypoint_pixels=np.array([[50., 16.]]))
    pixels = np.asarray(_montage(stage, 'mvt2', True))
    np.testing.assert_array_equal(pixels[78:110, 97:129], pixels[78:110, 129:161])
