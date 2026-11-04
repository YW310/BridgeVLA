"""Measured-TCP crops preserve key predictions and world-aligned chunk context."""
import copy

import numpy as np
import pytest
import torch
from torch import nn

from finetune.OHT.chunk_config import chunk_settings
from finetune.OHT.config import load
from finetune.OHT.data.chunk_dataset import OHTChunkDataset
from finetune.OHT.model import build_agent, load_weights, model_state
from tests.test_oht_chunk_joint import JointMVT, mixed_batch
from tests.test_oht_chunk_key_guidance import guided_config
from tests.test_oht_chunk_replay import chunk_fixture, replay_fixture  # noqa: F401


def test_tcp_preset_and_coarse_ablation_only_change_local_context():
    old = chunk_settings(load('finetune/OHT/configs/oht_chunk_v4_delta_state_guided_8x40.yaml'))
    for suffix, mode in (('tcp', 'tcp'), ('coarse', 'coarse_only')):
        new = chunk_settings(load(f'finetune/OHT/configs/oht_chunk_v4_delta_state_{suffix}_guided_8x40.yaml'))
        assert {name for name in old if old[name] != new[name]} == {'local_context'}
        assert new['local_context'] == mode and new['spatial_key_token']


@pytest.mark.parametrize('patch', [dict(local_context='unknown'), dict(tcp_crop_size_m=0),
    dict(tcp_crop_size_m=float('nan')), dict(tcp_crop_size_m=True),
    dict(local_context='tcp', spatial_conditioning=False)])
def test_invalid_crop_settings_fail_early(patch):
    config = guided_config()
    config['chunk_policy'].update(patch)
    with pytest.raises(ValueError):
        chunk_settings(config)


@pytest.mark.parametrize('with_mean', [False, True])
def test_metric_crop_uses_exact_observed_cloud_normalization(with_mean):
    from bridgevla.mvt.chunk_geometry import tcp_crop_in_cube
    from bridgevla.mvt.utils import place_pc_in_cube
    points = torch.tensor([[1., -1., .2], [3., 2., 4.], [2., .5, 1.]])
    tcp, size, bounds = torch.tensor([1.8, .3, 1.4]), .4, [-2., -3., 0., 4., 3., 6.]
    center, scale = tcp_crop_in_cube(points, tcp, size, with_mean=with_mean, scene_bounds=bounds)
    normalized, reverse = place_pc_in_cube(points, app_pc=tcp[None],
        with_mean_or_bounds=with_mean, scene_bounds=None if with_mean else bounds)
    torch.testing.assert_close(center, normalized[0])
    # A crop edge is exactly half the metric cube width away from its center.
    torch.testing.assert_close(reverse(center + torch.tensor([1., 0., 0.]) / scale),
                               tcp + torch.tensor([size / 2, 0., 0.]))


def test_tcp_geometry_uses_its_own_per_sample_inverse_not_the_goal_crop():
    from bridgevla.mvt.chunk_geometry import chunk_geometry_to_world
    points = torch.tensor([[[.1, .2, .3]], [[-.4, .5, .6]]])
    centers = torch.tensor([[.3, -.1, .5], [-.2, .4, .6]])
    scales = torch.tensor([[2.], [4.]])
    stage = dict(oht_chunk_xyz=points, oht_chunk_valid=torch.ones(2, 1, dtype=torch.bool))
    output = dict(oht_chunk_tcp=stage,
                  oht_chunk_tcp_rev_trans=lambda x: x / scales + centers,
                  rev_trans=lambda x: x + 100)  # Unrelated goal crop must not enter TCP geometry.
    chunk_geometry_to_world(output, [lambda x: x * 3 + 2, lambda x: x * 2 - 1])
    expected = torch.stack(((points[0] / 2 + centers[0]) * 3 + 2,
                            (points[1] / 4 + centers[1]) * 2 - 1))
    torch.testing.assert_close(stage['oht_chunk_xyz_world'], expected)


def test_real_mvt_tcp_pass_preserves_key_branch_and_filters_all_axes():
    from bridgevla.mvt.mvt import MVT
    from bridgevla.mvt.chunk_geometry import chunk_geometry_to_world

    class Single(nn.Module):
        add_corr, norm_corr, add_pixel_loc = True, True, False
        oht_return_chunk_geometry = True
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.))
        def forward(self, img, wpt_local=None, **kwargs):
            assert wpt_local is None
            tokens = img[:, :, 3:6].permute(0, 1, 3, 4, 2).reshape(len(img), -1, 3)
            return dict(trans=img[:, :, 3] * self.weight,
                        oht_chunk_tokens=tokens * self.weight, oht_chunk_grid_shape=(3, 2, 2))
        def get_wpt(self, out, *args):
            return out['trans'].new_tensor([[.25, -.1, .3]]).expand(len(out['trans']), -1)

    model = MVT.__new__(MVT)
    nn.Module.__init__(model)
    model.mvt1 = Single()
    model.stage_two, model.st_sca, model.img_feat_dim = True, 3., 3
    model.object_slot_predictor1 = model.object_slot_predictor2 = model.refine_role_mask_head = None
    for name in ('object_conditioning_inherit_coarse_roles', 'object_slots_enabled',
                 'object_conditioning_shared_action_features', 'object_conditioning_use_context',
                 'object_conditioning_refine_mask_detach', 'oracle_adapter_translation_only',
                 'oracle_relation_gated_adapter'):
        setattr(model, name, False)
    model.object_slot_predictor_type = 'slots'
    rendered = []
    def renderer(points, features, **kwargs):
        rendered.append(points.clone())
        image = features.new_zeros(4, features.shape[-1])
        image[:len(features)] = features
        return image.reshape(1, 2, 2, -1).repeat(3, 1, 1, 1)
    model.renderer = renderer
    model.eval()
    points = torch.tensor([[1.5, .2, .3], [.2, .3, .4], [.1, .4, .3], [.1, .2, .3]])
    colors = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 12
    baseline = model([points], [colors])
    center, scale = torch.tensor([[.1, .2, .3]]), torch.tensor([4.])
    output = model([points], [colors], oht_chunk_tcp_local=center, oht_chunk_tcp_scale=scale)
    assert len(rendered) == 5  # Two key passes, then two key passes + TCP pass.
    assert len(rendered[-1]) == 3 and (rendered[-1].abs() <= 1).all()
    for stage in (None, 'mvt2'):
        before, after = (baseline, output) if stage is None else (baseline[stage], output[stage])
        torch.testing.assert_close(after['trans'], before['trans'])
        torch.testing.assert_close(after['oht_chunk_tokens'], before['oht_chunk_tokens'])
    reverse = lambda value: value / 2 + torch.tensor([1., 2., 3.])
    chunk_geometry_to_world(output, [reverse])
    local = output['oht_chunk_tcp']
    valid = local['oht_chunk_valid']
    torch.testing.assert_close(local['oht_chunk_xyz_world'][valid], reverse(points[1:]).repeat(3, 1))
    local['oht_chunk_tokens'].sum().backward()
    assert model.mvt1.weight.grad.abs() > 0
    # Empty crops stay explicitly invalid, without borrowing the goal crop.
    empty = model([points], [colors], oht_chunk_tcp_local=center + 10, oht_chunk_tcp_scale=scale)
    assert not empty['oht_chunk_tcp']['oht_chunk_valid'].any()


class TcpJointMVT(JointMVT):
    def forward(self, pc, **kwargs):
        output = super().forward(pc, **kwargs)
        if 'oht_chunk_tcp_local' in kwargs:
            center = kwargs['oht_chunk_tcp_local']
            scale = kwargs['oht_chunk_tcp_scale'][:, None]
            output['oht_chunk_tcp_rev_trans'] = lambda value: value / scale + center
            tokens = output['oht_chunk_tokens'] + .5
            output['oht_chunk_tcp'] = dict(oht_chunk_tokens=tokens,
                oht_chunk_xyz=center.new_zeros(len(pc), 6, 3),
                oht_chunk_valid=torch.ones(len(pc), 6, dtype=torch.bool))
        return output


@pytest.mark.parametrize('mode', ['tcp', 'coarse_only'])
def test_agent_train_serve_and_checkpoint_use_same_selected_context(chunk_fixture, monkeypatch, tmp_path, mode):
    from bridgevla.mvt import mvt
    from finetune.OHT.runtime.policy import Policy
    monkeypatch.setattr(mvt, 'MVT', TcpJointMVT)
    config = guided_config()
    config['chunk_policy'].update(local_context=mode)
    data = OHTChunkDataset(chunk_fixture.root)
    agent = build_agent(config, data.contract, torch.device('cpu'))
    sample = data[0]
    context = agent._context(sample)
    if mode == 'tcp':
        assert context[1].shape[1] == 6
        torch.testing.assert_close(context[5], torch.tensor(sample['tcp_pose'][:3])[None, None].expand(1, 6, 3))
        torch.testing.assert_close(context[1] - context[0], torch.full_like(context[1], .5))
    else:
        assert context[1].shape[1] == context[5].shape[1] == context[7].shape[1] == 0
    predicted = agent.predict(sample, sample['goal'])
    changed = copy.deepcopy(sample)
    changed['key_action'][:] = 999.
    changed['action_chunk'][:] = -999.
    np.testing.assert_array_equal(predicted['actions'], agent.predict(changed, sample['goal'])['actions'])
    policy = Policy(agent, data.contract, 'cpu', 'baseline')
    np.testing.assert_array_equal(predicted['actions'], policy.act(sample, sample['goal'], 0)['actions'])
    before = agent._net_mod.decoder.head.weight.detach().clone()
    agent.update(mixed_batch(data))
    assert not torch.equal(before, agent._net_mod.decoder.head.weight)
    checkpoint = dict(config=config, model_state=copy.deepcopy(model_state(agent)))
    other = build_agent(config, data.contract, torch.device('cpu'))
    load_weights(other, checkpoint)
    np.testing.assert_array_equal(agent.predict(sample, sample['goal'])['actions'],
                                  other.predict(sample, sample['goal'])['actions'])
    # Switching context is an explicit new-run initialization, never strict resume.
    config2 = copy.deepcopy(config)
    config2['chunk_policy']['local_context'] = 'key_refine'
    target = build_agent(config2, data.contract, torch.device('cpu'))
    with pytest.raises(ValueError, match='mismatch'):
        load_weights(target, checkpoint)
    load_weights(target, checkpoint, initialize=True)
    for name, value in checkpoint['model_state'].items():
        torch.testing.assert_close(model_state(target)[name], value)
    # Exercise the real HTTP validation/serialization boundary with this config.
    from fastapi.testclient import TestClient
    from finetune.OHT.vla_server import create_app
    from finetune.OHT.runtime.inference_api import OBSERVATION_FORMAT
    from finetune.OHT.runtime.predicted_wrapper import policy_observation
    from finetune.OHT.runtime.transport import pack_observation
    from tests.test_oht_vla_server import envelope
    observation = policy_observation(sample, data.contract['data_config']['cameras'])
    for name in ('tcp_pose', 'joint_positions', 'gripper_open01'):
        observation[name] = sample[name]
    path = tmp_path / 'model.pth'
    path.write_bytes(b'injected loader')
    restored = Policy(other, data.contract, 'cpu', 'baseline')
    app = create_app(path, policy_loader=lambda *args: restored,
                     output_dir=tmp_path / 'server', visualize_every=0)
    with TestClient(app) as client:
        info = client.get('/model_info').json()
        assert info['chunk_policy']['local_context'] == mode
        assert info['chunk_policy']['tcp_crop_size_m'] == .4
        response = client.post('/predict', json=dict(**envelope(info), step=0, timestamp=0.,
            instruction=sample['goal'], observation_format=OBSERVATION_FORMAT,
            prediction_schema=info['chunk_spec']['schema'], observation=pack_observation(observation)))
        assert response.status_code == 200, response.text
        np.testing.assert_allclose(response.json()['actions'],
                                  other.predict(sample, sample['goal'])['actions'], atol=1e-7)


def test_actual_single_tcp_encoding_skips_heatmap_heads_and_preserves_vlm_gradients(monkeypatch):
    from tests.test_object_conditioning_forward import ObjectConditioningForwardTest
    module, _, _, options, _ = ObjectConditioningForwardTest()._small_policy()
    options.update(object_slot_predictor=None, oracle_feature_adapter=None,
                   object_conditioning_use_context=False, object_conditioning_shared_action_features=False)
    module.eval()
    module.oht_return_chunk_tokens = True
    full = module(**options, forward_no_feat=True)
    def unexpected(*args, **kwargs):
        raise AssertionError('TCP token encoding must not compute a key heatmap')
    monkeypatch.setattr(module.up0, 'forward', unexpected)
    local = module(**options, forward_no_feat=True, chunk_tokens_only=True)
    assert set(local) == {'oht_chunk_tokens', 'oht_chunk_grid_shape'}
    torch.testing.assert_close(local['oht_chunk_tokens'], full['oht_chunk_tokens'])
    local['oht_chunk_tokens'].square().mean().backward()
    assert module.hidden.grad.abs().sum() > 0


def test_local_validity_diagnostics_report_empty_observed_crops():
    from finetune.OHT.chunk_model import ChunkNetwork
    from tests.test_oht_spatial_chunk import inputs
    decoder, context, target, valid = inputs('regression')
    context = list(context)
    context[7] = torch.ones_like(context[7])
    context[7][0] = False
    network = ChunkNetwork(nn.Identity(), decoder)
    prediction, _, _, extra = network(context_builder=lambda: context,
                                      chunk_target=target, chunk_valid=valid)
    assert all(torch.isfinite(value).all() for value in prediction)
    assert extra['chunk_local_valid_rate'] == .5
    assert extra['chunk_local_empty_rate'] == .5
