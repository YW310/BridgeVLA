"""3D render alignment, TCP-local trajectories and flow inference without GT."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from finetune.OHT import bootstrap  # noqa: F401
from bridgevla.mvt.chunk_geometry import pool_chunk_geometry, chunk_geometry_to_world
from finetune.OHT.chunk_config import chunk_settings
from finetune.OHT.chunk_model import ChunkAgent, chunk_loss, quaternion_matrix
from finetune.OHT.config import load
from finetune.OHT.data.chunk_dataset import OHTChunkDataset, collate_chunk
from finetune.OHT.model import build_agent, model_state, load_weights
from finetune.OHT.spatial_chunk import SpatialChunkDecoder, RelativeAttention
from tests import test_cross_scale_render as render_tests
from tests.test_oht_chunk_joint import JointMVT, mixed_batch
from tests.test_oht_chunk_replay import chunk_fixture, replay_fixture  # noqa: F401


def small_config(kind='flow_matching'):
    config = load('finetune/OHT/configs/oht_chunk_v4_flow_8x40.yaml')
    config['chunk_policy'].update(decoder_type=kind, horizon=4, execute_horizon=2,
                                  hidden_dim=16, layers=1, heads=2, flow_steps=2)
    config['freeze_gemma_prefix_layers'] = 2
    return config


def spatial_agent(root, kind='flow_matching', stage='joint', distributed=False):
    from bridgevla.mvt import mvt
    original, mvt.MVT = mvt.MVT, JointMVT
    try:
        data, config = OHTChunkDataset(root), small_config(kind)
        config['chunk_policy']['stage'] = stage
        agent = build_agent(config, data.contract, torch.device('cpu'), distributed=distributed)
        return agent, data, config
    finally:
        mvt.MVT = original


def inputs(kind='flow_matching'):
    torch.manual_seed(17)
    settings = chunk_settings(small_config(kind))
    decoder = SpatialChunkDecoder(4, settings, 3, 1 / 60)
    state = torch.zeros(2, 14)
    state[:, :3] = torch.tensor([[.5, .2, 1.2], [.4, -.2, .9]])
    state[:, 3:7] = torch.from_numpy(Rotation.from_euler('xyz', [[20, 10, 80], [-60, 25, 30]], degrees=True).as_quat())
    state[:, 7:13] = .1
    goal = torch.cat((state[:, :7].clone(), torch.ones(2, 1)), -1)
    goal[:, :3] += torch.tensor([.04, -.02, .03])
    xyz = state[:, None, :3] + torch.randn(2, 6, 3) * .03
    valid = torch.ones(2, 6, dtype=torch.bool)
    context = (torch.randn(2, 6, 4), torch.randn(2, 6, 4), state, goal, xyz, xyz + .01, valid, valid.clone())
    target = goal[:, None].repeat(1, 4, 1)
    target[:, :, :3] = state[:, None, :3] + torch.arange(1, 5)[None, :, None] * torch.tensor([-.001, 0., 0.])[None, None]
    return decoder, context, target, torch.ones(2, 4, dtype=torch.bool)


def test_new_configs_keep_training_budget_freezing_and_separate_decoder_comparison():
    parent = load('finetune/OHT/configs/oht_chunk_v4_8x40.yaml')
    for name, kind in (('spatial', 'regression'), ('flow', 'flow_matching')):
        config = load(f'finetune/OHT/configs/oht_chunk_v4_{name}_8x40.yaml')
        settings = chunk_settings(config)
        assert settings['spatial_conditioning'] and settings['decoder_type'] == kind
        assert settings['displacement_weight'] == 1.
        for key in ('epochs', 'global_batch_size', 'learning_rate', 'freeze_gemma_prefix_layers'):
            assert config[key] == parent[key]
        assert config['agent']['optimizer_type'] == parent['agent']['optimizer_type']
    assert not chunk_settings(parent)['spatial_conditioning']


@pytest.mark.parametrize('patch', [dict(spatial_conditioning=1), dict(decoder_type='diffusion'),
    dict(spatial_conditioning=False), dict(flow_weight=0), dict(flow_steps=0), dict(flow_steps=True),
    dict(flow_inference_seed=-1), dict(flow_inference_seed=True), dict(goal_conditioning='absolute')])
def test_invalid_spatial_flow_settings_rejected(patch):
    config = small_config()
    config['chunk_policy'].update(patch)
    with pytest.raises(ValueError):
        chunk_settings(config)


def test_render_adds_geometry_without_changing_rgb_normalized_correlations_or_depth():
    points = torch.tensor([[1.5, .2, .3], [.2, .3, .4], [.4, .5, .6], [0., 0., 0.]])
    colors = torch.full((4, 3), .3)
    def renderer(points, features, **kwargs):
        return torch.cat((features.reshape(1, 2, 2, -1), torch.ones(1, 2, 2, 1)), -1)
    stage = SimpleNamespace(add_corr=True, norm_corr=True, add_pixel_loc=False, oht_return_chunk_geometry=True)
    model = SimpleNamespace(mvt1=stage, renderer=renderer, object_conditioning_inherit_coarse_roles=False)
    render = render_tests.CrossScaleRenderTest._render_method()
    result = render(model, [points], [colors], 0., True, None)
    geometry = model._oht_chunk_render_geometry
    stage.oht_return_chunk_geometry = False
    torch.testing.assert_close(render(model, [points], [colors], 0., True, None), result)
    pooled, valid = pool_chunk_geometry(geometry[:, :, :3], geometry[:, :, 3:4], (1, 2, 2))
    torch.testing.assert_close(pooled[0], points)
    assert valid.all()  # Frame origin is a valid surface, not an empty pixel.
    torch.testing.assert_close(result[0, 0, :3].permute(1, 2, 0).reshape(4, 3), points / 1.5)


def test_masked_patch_pooling_handles_antialiasing_empty_and_nan_pixels():
    xyz = torch.zeros(1, 1, 3, 2, 4)
    coverage = torch.zeros(1, 1, 1, 2, 4)
    xyz[0, 0, :, 0, 0] = torch.tensor([.5, 1., 1.5])
    coverage[0, 0, 0, 0, 0] = .5
    xyz[0, 0, :, 1, 3] = float('nan')
    pooled, valid = pool_chunk_geometry(xyz, coverage, (1, 1, 2))
    torch.testing.assert_close(pooled, torch.tensor([[[1., 2., 3.], [0., 0., 0.]]]))
    assert valid.tolist() == [[True, False]]


def test_fine_geometry_undoes_predicted_crop_before_cube_to_world():
    points = torch.tensor([[[.2, -.4, .5], [.8, .6, -.2]]])
    center = torch.tensor([[.3, .1, -.1]])
    output = dict(oht_chunk_xyz=points, oht_chunk_valid=torch.tensor([[True, False]]),
        rev_trans=lambda value: value / 4 + center,
        mvt2=dict(oht_chunk_xyz=(points - center[:, None]) * 4, oht_chunk_valid=torch.tensor([[True, False]])))
    chunk_geometry_to_world(output, [lambda value: value / 2 + torch.tensor([1., 2., 3.])])
    expected = torch.tensor([[[1.1, 1.8, 3.25], [0., 0., 0.]]])
    for stage in (output, output['mvt2']):
        torch.testing.assert_close(stage['oht_chunk_xyz_world'], expected)


def test_crop_inverse_supports_multiple_samples_with_different_origins():
    points = torch.tensor([[[.1, .2, .3], [.4, .5, .6]], [[-.1, -.2, -.3], [-.4, -.5, -.6]]])
    center = torch.tensor([[.2, .4, .6], [-.3, .1, .5]])
    output = dict(rev_trans=lambda value: value / 3 + center,
                  mvt2=dict(oht_chunk_xyz=(points - center[:, None]) * 3,
                            oht_chunk_valid=torch.ones(2, 2, dtype=torch.bool)))
    chunk_geometry_to_world(output, [lambda x: x + 2, lambda x: x - 1])
    torch.testing.assert_close(output['mvt2']['oht_chunk_xyz_world'], torch.stack((points[0] + 2, points[1] - 1)))


def test_actual_mvt_forward_attaches_each_stage_geometry_before_render_cache_is_replaced():
    from bridgevla.mvt.mvt import MVT
    from torch import nn

    class Single(nn.Module):
        add_corr, norm_corr, add_pixel_loc = True, True, False
        oht_return_chunk_geometry = True

        def forward(self, img, wpt_local=None, **kwargs):
            assert wpt_local is None  # Both stages must follow prediction routing.
            tokens = img[:, :, 3:6].permute(0, 1, 3, 4, 2).reshape(len(img), -1, 3)
            return dict(trans=img.new_zeros(len(img), 3, 2, 2),
                        oht_chunk_tokens=tokens, oht_chunk_grid_shape=(3, 2, 2))

        def get_wpt(self, out, *args):
            return out['trans'].new_tensor([[.25, -.1, .3]]).expand(len(out['trans']), -1)

    model = MVT.__new__(MVT)
    nn.Module.__init__(model)
    model.mvt1 = Single()
    model.stage_two, model.st_sca, model.img_feat_dim = True, 3., 3
    model.object_slot_predictor1 = model.object_slot_predictor2 = None
    model.refine_role_mask_head = None
    for name in ('object_conditioning_inherit_coarse_roles', 'object_slots_enabled',
                 'object_conditioning_shared_action_features', 'object_conditioning_use_context',
                 'object_conditioning_refine_mask_detach', 'oracle_adapter_translation_only',
                 'oracle_relation_gated_adapter'):
        setattr(model, name, False)
    model.object_slot_predictor_type = 'slots'
    model.renderer = lambda points, features, **options: features.reshape(1, 2, 2, -1).repeat(3, 1, 1, 1)
    model.eval()
    points = torch.tensor([[1.5, .2, .3], [.2, .3, .4], [.4, .5, .6], [0., 0., 0.]])
    colors = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 12
    output = model([points], [colors])
    inverse = lambda value: value / 2 + torch.tensor([.1, .2, .3])
    chunk_geometry_to_world(output, [inverse])
    expected = inverse(points).repeat(3, 1)[None]
    for stage in (output, output['mvt2']):
        torch.testing.assert_close(stage['oht_chunk_xyz_world'], expected)
        assert stage['oht_chunk_valid'].all()
        torch.testing.assert_close(stage['oht_chunk_tokens'][0], colors.repeat(3, 1))
    model.mvt1.oht_return_chunk_geometry = False
    legacy = model([points], [colors])
    for name in ('mvt1_ori_img', 'mvt2_ori_img'):
        torch.testing.assert_close(output[name], legacy[name])
    assert 'oht_chunk_xyz' not in legacy and 'oht_chunk_xyz' not in legacy['mvt2']


def test_relative_attention_uses_geometry_and_is_translation_and_permutation_invariant():
    torch.manual_seed(9)
    attention = RelativeAttention(8, 2, 0.).eval()
    query, memory = torch.randn(1, 3, 8), torch.randn(1, 4, 8)
    qxyz, mxyz, valid = torch.randn(1, 3, 3), torch.randn(1, 4, 3), torch.tensor([[True, True, False, True]])
    output = attention(query, memory, qxyz, mxyz, valid)
    torch.testing.assert_close(output, attention(query, memory, qxyz + 2, mxyz + 2, valid))
    order = torch.tensor([3, 2, 1, 0])
    torch.testing.assert_close(output, attention(query, memory[:, order], qxyz, mxyz[:, order], valid[:, order]))
    assert not torch.allclose(output, attention(query, memory, qxyz, mxyz * 4, valid))


@pytest.mark.parametrize('kind', ['regression', 'flow_matching'])
def test_pose_roundtrip_and_prediction_equivariance_under_common_world_transform(kind):
    decoder, context, target, _ = inputs(kind)
    decoder.eval()
    state = context[2]
    position, rotation, _ = decoder.decode(decoder.encode_target(target, state), target[..., 7], state)
    torch.testing.assert_close(position, target[..., :3])
    torch.testing.assert_close(rotation, quaternion_matrix(target[..., 3:7]))
    first = decoder(*context)
    world = Rotation.from_euler('xyz', [20, -40, 65], degrees=True)
    r = torch.tensor(world.as_matrix(), dtype=torch.float32)
    offset = torch.tensor([2., -1., .5])
    changed = list(context)
    for index in (2, 3):
        changed[index] = context[index].clone()
        changed[index][:, :3] = context[index][:, :3] @ r.T + offset
        changed[index][:, 3:7] = torch.from_numpy((world * Rotation.from_quat(context[index][:, 3:7])).as_quat())
    for index in (4, 5):
        changed[index] = context[index] @ r.T + offset
    second = decoder(*changed)
    torch.testing.assert_close(second[0], first[0] @ r.T + offset, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(second[1], r @ first[1], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(second[2], first[2], atol=1e-6, rtol=1e-5)


def test_no_valid_visual_tokens_is_finite_and_invalid_content_cannot_change_output():
    decoder, context, _, _ = inputs()
    context = list(context)
    context[6], context[7] = torch.zeros_like(context[6]), torch.zeros_like(context[7])
    first = decoder(*context)
    for index in (0, 1, 4, 5):
        context[index] = torch.full_like(context[index], float('nan'))
    second = decoder(*context)
    for a, b in zip(first, second):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b)


def test_flow_euler_integrates_velocity_and_reads_gripper_at_clean_endpoint(monkeypatch):
    decoder, context, _, _ = inputs()
    memory = decoder.memory(*context)
    def constant_field(pose, tau, memory):
        return torch.full_like(pose, .2), tau[:, None].expand(pose.shape[:2])
    monkeypatch.setattr(decoder, 'field', constant_field)
    generator = torch.Generator().manual_seed(decoder.settings['flow_inference_seed'])
    initial = torch.randn(1, 4, 9, generator=generator).expand(2, -1, -1)
    expected = decoder.decode(initial + .2, torch.ones(2, 4), context[2])
    for a, b in zip(decoder.sample(memory, context[2]), expected):
        torch.testing.assert_close(a, b)


def test_masked_flow_training_has_finite_gradients_and_policy_sample_is_not_reconstruction():
    decoder, context, target, valid = inputs()
    valid[:, 2:] = False
    target[:, 2:] = float('nan')
    torch.manual_seed(11)
    prediction, extra = decoder.training_prediction(*context, target=target, valid=valid)
    loss, _ = chunk_loss(prediction, target, valid, decoder.settings, current_position=context[2][:, :3])
    (loss + extra['flow_loss']).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in decoder.parameters())
    for a, b in zip(extra['sampled_prediction'], decoder(*context)):
        torch.testing.assert_close(a, b)
    assert not torch.allclose(prediction[0], extra['sampled_prediction'][0])
    altered = target.clone()
    altered[:, 2:] = 9999.
    torch.manual_seed(11)
    other, extras = decoder.training_prediction(*context, target=altered, valid=valid)
    torch.testing.assert_close(extra['flow_loss'], extras['flow_loss'])
    for a, b in zip(prediction, other):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize('kind', ['regression', 'flow_matching'])
def test_real_joint_agent_trains_visual_features_and_head_and_keeps_future_labels_out_of_policy(chunk_fixture, kind, monkeypatch):
    from bridgevla.models import bridgevla_agent
    monkeypatch.setattr(bridgevla_agent, 'build_policy_visualization_payload', lambda *args, **kwargs: dict(stages={}))
    agent, data, config = spatial_agent(chunk_fixture.root, kind)
    sample, batch = data[0], mixed_batch(data)
    first = agent.predict(sample, sample['goal'])
    altered = copy.deepcopy(sample)
    altered['key_action'][:] = 999.
    altered['action_chunk'][:] = -999.
    np.testing.assert_array_equal(first['actions'], agent.predict(altered, sample['goal'])['actions'])
    before = copy.deepcopy(model_state(agent))
    metrics = agent.update(batch, return_visualization=True)
    np.testing.assert_allclose(metrics['train_visualization']['predicted'], first['actions'][:, :3], atol=1e-6)
    assert np.isfinite(metrics['total_loss'])
    assert not torch.equal(before['decoder.head.weight'], model_state(agent)['decoder.head.weight'])
    assert not torch.equal(before['encoder.trans'], model_state(agent)['encoder.trans'])
    assert not torch.equal(before['encoder.mvt1.model.language_model.layers.7.weight'],
                           model_state(agent)['encoder.mvt1.model.language_model.layers.7.weight'])
    assert torch.equal(before['encoder.mvt1.model.vision_tower.0.weight'],
                        model_state(agent)['encoder.mvt1.model.vision_tower.0.weight'])
    if kind == 'flow_matching':
        assert metrics['chunk_flow_weighted_loss'] > 0
    checkpoint = dict(config=config, model_state=copy.deepcopy(model_state(agent)))
    other, _, _ = spatial_agent(chunk_fixture.root, kind)
    load_weights(other, checkpoint)
    np.testing.assert_array_equal(agent.predict(sample, sample['goal'])['actions'], other.predict(sample, sample['goal'])['actions'])


def test_architecture_change_only_initializes_encoder_and_flow_loss_change_keeps_decoder(chunk_fixture):
    source, data, config = spatial_agent(chunk_fixture.root, 'regression')
    source.update(mixed_batch(data))
    checkpoint = dict(config=config, model_state=copy.deepcopy(model_state(source)))
    target, _, new_config = spatial_agent(chunk_fixture.root)
    initial = copy.deepcopy(target._net_mod.decoder.state_dict())
    with pytest.raises(ValueError, match='mismatch'):
        load_weights(target, checkpoint)
    load_weights(target, checkpoint, initialize=True)
    for key, value in source._net_mod.encoder.state_dict().items():
        torch.testing.assert_close(target._net_mod.encoder.state_dict()[key], value)
    for key, value in initial.items():
        torch.testing.assert_close(target._net_mod.decoder.state_dict()[key], value)
    checkpoint = dict(config=copy.deepcopy(new_config), model_state=copy.deepcopy(model_state(target)))
    target.settings['flow_weight'] = 2.
    load_weights(target, checkpoint, initialize=True)
    for key, value in checkpoint['model_state'].items():
        torch.testing.assert_close(model_state(target)[key], value)


@pytest.mark.parametrize('kind', ['regression', 'flow_matching'])
def test_new_decoder_serves_http_and_evaluates_held_out_replay(chunk_fixture, tmp_path, kind):
    from fastapi.testclient import TestClient
    from tests.test_oht_vla_server import envelope
    from finetune.OHT.runtime.policy import Policy
    from finetune.OHT.runtime.inference_api import OBSERVATION_FORMAT
    from finetune.OHT.runtime.transport import pack_observation
    from finetune.OHT.vla_server import create_app
    from finetune.OHT.runtime.chunk_protocol import validate_prediction
    from finetune.OHT.chunk_eval import open_loop_chunk
    agent, data, _ = spatial_agent(chunk_fixture.root, kind)
    agent._net_mod.eval()
    policy = Policy(agent, data.contract, 'cpu', 'baseline')
    sample = data[0]
    from finetune.OHT.runtime.predicted_wrapper import policy_observation
    observation = policy_observation(sample, data.contract['data_config']['cameras'])
    for name in ('tcp_pose', 'joint_positions', 'gripper_open01'):
        observation[name] = sample[name]
    checkpoint = tmp_path / 'injected.pth'
    checkpoint.write_bytes(b'test policy loader')
    app = create_app(checkpoint, policy_loader=lambda *args: policy, output_dir=tmp_path / 'server', visualize_every=0)
    with TestClient(app) as client:
        info = client.get('/model_info').json()
        assert info['chunk_policy']['decoder_type'] == kind
        body = dict(**envelope(info), step=0, timestamp=0., instruction=sample['goal'],
                    observation_format=OBSERVATION_FORMAT, observation=pack_observation(observation),
                    prediction_schema=info['chunk_spec']['schema'])
        response = client.post('/predict', json=body)
        assert response.status_code == 200, response.text
        result = validate_prediction(response.json(), info['chunk_spec'])
        np.testing.assert_array_equal(result['actions'], agent.predict(sample, sample['goal'])['actions'])
    report = open_loop_chunk(policy, chunk_fixture.root, 'val', limit=2)
    assert report['samples'] == 2 and report['all_actions']['samples'] > 0
    assert report['chunk_policy']['spatial_conditioning']


def test_fixed_flow_noise_is_independent_of_batching_and_does_not_advance_global_rng():
    decoder, context, _, _ = inputs()
    decoder.eval()
    rng = torch.random.get_rng_state().clone()
    batch = decoder(*context)
    assert torch.equal(rng, torch.random.get_rng_state())
    for index in range(2):
        single = decoder(*(value[index:index + 1] for value in context))
        for first, second in zip(single, batch):
            torch.testing.assert_close(first[0], second[index], atol=1e-6, rtol=1e-5)


def test_predict_temporarily_disables_dropout_and_restores_training_on_error(chunk_fixture, monkeypatch):
    agent, data, _ = spatial_agent(chunk_fixture.root)
    # Configure all real decoder dropout paths, including relative attention.
    for module in agent._net_mod.decoder.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = .5
        if isinstance(module, RelativeAttention):
            module.dropout = .5
    sample = data[0]
    assert agent._net_mod.decoder.training
    rng = torch.random.get_rng_state().clone()
    first = agent.predict(sample, sample['goal'])
    assert agent._net_mod.decoder.training
    np.testing.assert_array_equal(first['actions'], agent.predict(sample, sample['goal'])['actions'])
    assert torch.equal(rng, torch.random.get_rng_state())
    monkeypatch.setattr(agent, '_context', lambda *args, **kwargs: (_ for _ in ()).throw(ValueError('capture failed')))
    with pytest.raises(ValueError, match='capture failed'):
        agent.predict(sample, sample['goal'])
    assert agent._net_mod.decoder.training


def _spatial_ddp_worker(rank, root, directory, stage):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=(Path(directory) / 'rendezvous').as_uri(), rank=rank, world_size=2)
    try:
        torch.manual_seed(123 + rank)
        agent, data, _ = spatial_agent(root, stage=stage, distributed=True)
        for iteration in range(2):
            batch = mixed_batch(data)
            if rank != iteration:
                batch['key_valid'][:] = False
                batch['key_action'][:] = float('nan')
            with agent._network.no_sync():
                agent.update(batch, loss_scale=.5, step_optimizer=False)
            agent.update(batch, loss_scale=.5, reset_gradients=False)
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in agent._net_mod.parameters())
        torch.save(model_state(agent), Path(directory) / f'rank{rank}.pth')
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('stage', ['joint', 'frozen_key'])
@pytest.mark.skipif(not torch.distributed.is_available() or not torch.distributed.is_gloo_available(), reason='Gloo unavailable')
def test_flow_ddp_gradient_accumulation_and_empty_key_rank(chunk_fixture, tmp_path, stage):
    torch.multiprocessing.spawn(_spatial_ddp_worker, args=(str(chunk_fixture.root), str(tmp_path), stage), nprocs=2, join=True)
    first, second = [torch.load(tmp_path / f'rank{rank}.pth', weights_only=True) for rank in range(2)]
    for name in first:
        torch.testing.assert_close(first[name], second[name])
