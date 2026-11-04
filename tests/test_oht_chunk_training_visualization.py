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
                             current=[0., 0., 0.], sample_id='task/000001/000030', step=500)
    text = '\n'.join(texts)
    assert 'mean=10.00 mm' in text and 'max=10.00 mm' in text and 'last valid=10.00 mm' in text
    assert 'hold-current baseline=110.00 mm' in text
    assert 'task/000001/000030' in text and 'step=500' in text
    assert 'equal axis scale' in text and 'world XYZ (m)' in text


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
        scene_points=payload['scene_points'], scene_colors=payload['scene_colors'])
    settings = dict(output_dir='images', save_png=True, tensorboard=True)
    with TensorBoardLogger(tmp_path / 'tensorboard') as logger:
        visualization.record(payload, 500, tmp_path, settings, logger)
    with Image.open(tmp_path / 'images/chunk_step_00000500.png') as saved:
        np.testing.assert_array_equal(np.asarray(saved), np.asarray(expected))
    with np.load(tmp_path / 'images/chunk_step_00000500.npz', allow_pickle=False) as arrays:
        for key in ('predicted', 'target', 'valid', 'current', 'scene_points', 'scene_colors'):
            np.testing.assert_array_equal(arrays[key], payload[key])
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
        with Image.open(tmp_path / f'images/{name}_step_00000050.png') as image:
            pixels = np.asarray(image)
        heatmap = pixels[78:142, 129:193]
        marked = pixels[78:142, 193:257]
        assert np.any(np.all(marked == [255, 0, 0], axis=-1))
        assert np.any(np.all(marked == [0, 255, 255], axis=-1))
        assert not np.array_equal(heatmap, marked)
        assert 'policy_pixels' not in stages[name]  # Never mutate the model's snapshot.
    events = EventAccumulator(str(tmp_path / 'tensorboard')).Reload()
    for name in stages:
        assert [event.step for event in events.Images('train_visualization/' + name)] == [50]


def test_offscreen_tcp_marks_are_not_clamped_to_image_edges():
    from finetune.OHT.runtime.online_visualization import _montage

    stage = dict(rgb=np.full((1, 32, 32, 3), .5), heatmap=np.ones((1, 32, 32)),
                 policy_pixels=np.array([[-10., 16.]]),
                 heatmap_waypoint_pixels=np.array([[50., 16.]]))
    pixels = np.asarray(_montage(stage, 'mvt2', True))
    np.testing.assert_array_equal(pixels[78:110, 97:129], pixels[78:110, 129:161])
