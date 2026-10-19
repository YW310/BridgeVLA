"""Same-forward diagnostics, actual camera projections, and held-out label isolation."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tests.test_oht_migration import replay_fixture  # noqa: F401
from tests.test_oht_action_modes import TinyMVT, caches  # noqa: F401
from finetune.OHT import eval as evaluation
from finetune.OHT.action_modes import MODES, resolve_action
from finetune.OHT.config import load
from finetune.OHT.data.dataset import OHTDataset
from finetune.OHT.evaluation_visualization import EvaluationVisualizer
from finetune.OHT.model import build_agent
from finetune.OHT.runtime.policy import Policy
from finetune.OHT.runtime.predicted_wrapper import PredictedObjectWrapper
from bridgevla.models.inference_visualization import build_policy_visualization_payload

ROOT = Path(__file__).resolve().parents[1]


class ImageMVT(TinyMVT):
    def forward(self, pc, **kwargs):
        self.calls = getattr(self, 'calls', 0) + 1
        assert not any(key.startswith('teacher_') for key in kwargs)
        output = super().forward(pc, **kwargs)
        rendered = torch.zeros(len(pc), 3, 6, 8, 8)
        rendered[:, :, 3:6] = .5
        output['mvt1_ori_img'] = rendered
        if self.stage_two:
            output['mvt2_ori_img'] = rendered.clone()
            output['wpt_local1'] = torch.tensor([[.2, -.1, .15]])
        if kwargs.get('oracle_prior_points') is not None:
            # Stand in for the production external-point prior rasterizer.
            prior = kwargs['oracle_prior_points'].mean(dim=(2, 3)).sigmoid()
            for stage in (output, output.get('mvt2', output)):
                stage['oracle_target_prior'] = prior[:, 0, None, None, None].expand(-1, 3, 8, 8)
                stage['oracle_reference_prior'] = prior[:, 1, None, None, None].expand(-1, 3, 8, 8)
        return output

    def get_pt_loc_on_img(self, points, mvt1_or_mvt2=True, out=None, **kwargs):
        if not mvt1_or_mvt2:
            points = (points - out['wpt_local1'][:, None]) * 3
        return torch.stack((points[..., [0, 1]], points[..., [1, 2]], points[..., [2, 0]]), dim=2) * 2 + 3.5


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('object_mode', ['baseline', 'role_queries', 'predicted_external'])
def test_actual_policy_exports_without_changing_actions_or_repeating_forward(
        replay_fixture, caches, tmp_path, monkeypatch, mode, object_mode):
    from bridgevla.mvt import mvt
    monkeypatch.setattr(mvt, 'MVT', ImageMVT)
    config = load(ROOT / f'finetune/OHT/configs/{object_mode}.yaml')
    config['action_mode'] = mode
    config['point_count'] = 8
    config['action']['hidden_dim'] = 16
    options = resolve_action(config)
    data = OHTDataset(replay_fixture.replay, 'val', action_config=options)
    agent = build_agent(config, data.contract, torch.device('cpu'), training=False)
    agent.eval()
    wrapper = None
    if object_mode == 'predicted_external':
        def predictor(observation, goal):
            assert 'action' not in observation and 'gripper_pose' not in observation
            return caches[1]
        wrapper = PredictedObjectWrapper(predictor, data.contract['data_config']['cameras'], 8)
    policy = Policy(agent, data.contract, 'cpu', object_mode, wrapper)
    row = data[0]
    ordinary = policy.act(row, row['goal'], 0)
    calls, payloads = agent._net_mod.calls, []
    actual = policy.act(row, row['goal'], 0, visualization_callback=payloads.append)
    np.testing.assert_array_equal(actual, ordinary)
    assert agent._net_mod.calls == calls + 1 and len(payloads) == 1
    assert set(payloads[0]['stages']) == {'mvt1', 'mvt2'}
    for stage in payloads[0]['stages'].values():
        assert 'gt' not in stage and 'gt_heatmap' not in stage
        assert all(isinstance(value, np.ndarray) for value in stage.values())
        np.testing.assert_allclose(stage['heatmap'].sum(axis=(1, 2)), 1., atol=1e-6)
    visualizer = EvaluationVisualizer(tmp_path / 'visualizations', every=2)
    report = evaluation.open_loop(policy, replay_fixture.replay, limit=3, visualizer=visualizer)
    assert agent._net_mod.calls == calls + 4  # one captured call above + three evaluated samples
    assert report['all_samples']['samples'] == 3
    assert report['visualization']['samples'] == 2
    assert (visualizer.root / 'index.html').is_file()
    assert {p.name for p in visualizer.root.glob('sample_*')} == {'sample_000000', 'sample_000002'}
    metadata = json.loads((visualizer.root / 'sample_000000/prediction.json').read_text(encoding='utf-8'))
    assert metadata['heatmap_is_auxiliary'] == (mode != 'eef')
    assert (metadata['policy_tcp_world'] is None) == ('joint' in mode)
    for name in ('mvt1', 'mvt2'):
        with Image.open(visualizer.root / f'sample_000000/{name}.png') as image:
            image.verify()
        with np.load(visualizer.root / f'sample_000000/{name}.npz', allow_pickle=False) as arrays:
            assert arrays['gt_heatmap'].shape == arrays['heatmap'].shape == (3, 8, 8)
            assert arrays['world_to_pixel'].shape == (3, 2, 4)
            assert ('policy_pixels' in arrays) == ('joint' not in mode)
            if object_mode != 'baseline':
                assert 'target_prior' in arrays and 'reference_prior' in arrays


@pytest.mark.parametrize('two_stages', [False, True])
def test_projection_includes_rotated_views_world_cube_and_predicted_refine_crop(two_stages):
    class Cameras:
        def get_pt_loc_on_img(self, points, mvt1_or_mvt2, out=None, **kwargs):
            if not mvt1_or_mvt2:
                points = (points - out['wpt_local1'][:, None]) * 4
            matrix = torch.tensor([[[8., 2., 1.], [1., 6., -2.]],
                                   [[-3., 7., 2.], [5., 1., 4.]],
                                   [[4., -2., 6.], [-2., 5., 3.]]])
            return torch.einsum('bpi,vji->bpvj', points, matrix) + 16
    cameras = Cameras()
    output = dict(trans=torch.randn(1, 3, 32, 32), mvt1_ori_img=torch.ones(1, 3, 6, 32, 32))
    if two_stages:
        output.update(mvt2=dict(trans=torch.randn(1, 3, 32, 32)),
                      mvt2_ori_img=output['mvt1_ori_img'], wpt_local1=torch.tensor([[.2, -.1, .05]]))
    origin = torch.tensor([2., -1., .5])
    reverse = lambda points: points / 2.5 + origin
    local_point = torch.tensor([[[.16, -.08, .02]]])
    world_point = reverse(local_point)[0, 0]
    captured = build_policy_visualization_payload(cameras, output, reverse, world_point[None])
    assert len(captured['stages']) == (2 if two_stages else 1)
    for name, stage in captured['stages'].items():
        pixels = stage['world_to_pixel'] @ np.r_[world_point.numpy(), 1.]
        expected = cameras.get_pt_loc_on_img(local_point, name == 'mvt1', out=output)[0, 0]
        np.testing.assert_allclose(pixels, expected, atol=2e-5)


def test_relative_body_tcp_is_decoded_and_outside_gt_not_clipped(tmp_path):
    rgb = np.full((3, 224, 224, 3), .5, np.float32)
    matrix = np.zeros((3, 2, 4))
    matrix[:, 0, 0], matrix[:, 1, 1] = 20, 20
    matrix[:, :, 3] = 112
    payload = dict(stages={'mvt1': dict(rgb=rgb, heatmap=np.full((3, 224, 224), 1 / (224 * 224)),
                                       world_to_pixel=matrix)}, heatmap_waypoint_world=np.zeros(3))
    row = dict(goal='<script>instruction</script>', gripper_pose=np.array([100., 0, 0, 0, 0, 0, 1]),
               tcp_pose=np.array([1., 2., 0, 0, 0, np.sqrt(.5), np.sqrt(.5)]), action=np.zeros(7))
    predicted = np.array([.1, 0, 0, 0, 0, 0, 1.])
    meta = dict(id='test/000000/000000', task='test', episode_index=0, frame=0)
    writer = EvaluationVisualizer(tmp_path / 'diagnostics')
    writer.write(0, payload, row, meta, predicted, dict(mode='relative_eef', translation_frame='body'), {})
    writer.finish()
    record = json.loads((writer.root / 'sample_000000/prediction.json').read_text(encoding='utf-8'))
    np.testing.assert_allclose(record['policy_tcp_world'], [1., 2.1, 0], atol=1e-6)
    assert record['stages']['mvt1']['gt_in_view'] == [False] * 3
    with np.load(writer.root / 'sample_000000/mvt1.npz', allow_pickle=False) as arrays:
        assert not arrays['gt_heatmap'].any()
        assert (arrays['gt_pixels'][:, 0] > 224).all()
    page = (writer.root / 'index.html').read_text(encoding='utf-8')
    assert '<script>' not in page and '&lt;script&gt;' in page
    with pytest.raises(FileExistsError):
        EvaluationVisualizer(writer.root)


@pytest.mark.parametrize('flags', [['--visualize-every', '-1'],
                                  ['--visualize-every', '0', '--visualize-output-dir', 'unused']])
def test_cli_rejects_invalid_options_before_loading(tmp_path, monkeypatch, flags):
    from finetune.OHT.runtime import loading
    def unexpected(*args, **kwargs):
        pytest.fail('Invalid visualization settings must fail before model initialization')
    monkeypatch.setattr(loading, 'load_policy', unexpected)
    with pytest.raises(ValueError, match='--visualize'):
        evaluation.main(['open-loop', '--checkpoint', 'unused', '--replay', 'unused',
                         '--output', str(tmp_path / 'result.json'), *flags])


@pytest.mark.parametrize('custom_directory', [False, True])
@pytest.mark.parametrize('interval', [None, 1])
def test_cli_visualization_output(replay_fixture, tmp_path, monkeypatch, custom_directory, interval):
    from finetune.OHT.runtime import loading
    data = OHTDataset(replay_fixture.replay, 'val')
    captures = []
    class PolicyStub:
        contract = data.contract
        def reset(self):
            pass
        def act(self, row, goal, step, *, visualization_callback=None):
            stage = dict(rgb=np.ones((3, 8, 8, 3), np.float32),
                         heatmap=np.full((3, 8, 8), 1 / 64, np.float32),
                         world_to_pixel=np.zeros((3, 2, 4)))
            captures.append(visualization_callback is not None)
            if visualization_callback is not None:
                visualization_callback(dict(stages={'mvt1': stage}, heatmap_waypoint_world=np.zeros(3)))
            return row['action']
    monkeypatch.setattr(loading, 'load_policy', lambda *args, **kwargs: PolicyStub())
    checkpoint = tmp_path / 'weights.pth'
    checkpoint.write_bytes(b'fixture weights')
    output = tmp_path / 'evaluation.json'
    directory = tmp_path / ('custom' if custom_directory else 'evaluation_visualizations')
    extra = ['--visualize-output-dir', str(directory)] if custom_directory else []
    if interval is not None:
        extra.extend(['--visualize-every', str(interval)])
    assert evaluation.main(['open-loop', '--checkpoint', str(checkpoint), '--replay', str(replay_fixture.replay),
                            '--output', str(output), '--limit', '3', '--quiet', *extra]) == 0
    report = json.loads(output.read_text(encoding='utf-8'))
    assert report['visualization']['samples'] == (1 if interval is None else 3)
    assert report['visualization']['every'] == (50 if interval is None else interval)
    assert captures == ([True, False, False] if interval is None else [True, True, True])
    assert Path(report['visualization']['index']) == directory / 'index.html'
    assert (directory / 'sample_000000/mvt1.png').is_file()
