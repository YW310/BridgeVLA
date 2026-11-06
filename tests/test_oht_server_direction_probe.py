"""Recorded-input direction diagnostics, with labels kept outside HTTP requests."""
import copy
import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from scipy.spatial.transform import Rotation

from tests.test_oht_migration import replay_fixture  # noqa: F401
from tests.test_oht_chunk_replay import chunk_fixture  # noqa: F401
from tests.test_oht_chunk_pipeline import agent_data
from finetune.OHT.data.source_config import sample_data_config
from finetune.OHT.runtime.policy import Policy
from finetune.OHT.vla_server import create_app
from tools.smoke_oht_vla_server import direction_report, raw_from_prepared, smoke


def test_opposite_direction_is_measured_from_current_pose_and_uses_executed_prefix():
    current = np.array([.4, -.2, 1.3, 0, 0, 0, 1])
    target = np.tile(np.r_[current[:3], 0, 0, 0, 1, 1], (4, 1))
    target[:, 0] += [.01, .02, .03, .04]
    predicted = target.copy()
    predicted[:2, 0] = current[0] - [.01, .02]
    target[3] = predicted[3] = np.nan
    report = direction_report(predicted, target, [True, True, True, False], current, 2)
    assert report['prefixes']['2']['opposite_directions'] == 2
    assert report['prefixes']['2']['direction_error_degrees_mean'] == pytest.approx(180)
    assert report['prefixes']['4']['compared_directions'] == 3
    assert report['offsets'][2]['direction_error_degrees'] == pytest.approx(0)
    assert report['offsets'][0]['predicted_displacement_world_m'] == pytest.approx([-.01, 0, 0])


def test_stationary_prediction_or_gt_has_no_defined_direction():
    current = np.zeros(7)
    target = np.zeros((3, 8))
    predicted = np.zeros_like(target)
    target[:, 0] = [0, .01, .02]
    predicted[:, 0] = [.01, 0, .00005]
    report = direction_report(predicted, target, [True] * 3, current, 2)
    assert all(row['direction_error_degrees'] is None for row in report['offsets'])
    assert report['prefixes']['2']['compared_directions'] == 0
    assert report['prefixes']['2']['direction_error_degrees_mean'] is None
    assert report['offsets'][2]['prediction_stationary']
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('kind', ['z', 'ray'])
def test_raw_roundtrip_preserves_nonidentity_tcp_and_camera_frames(chunk_fixture, kind):
    from types import SimpleNamespace
    from finetune.OHT.data.geometry import pose_matrix
    from finetune.OHT.runtime.raw_observation import decode_raw
    from finetune.OHT.runtime.transport import pack_observation
    agent, data, _ = agent_data(chunk_fixture, training=False)
    sample, row = data[0], data.samples[0]
    contract = copy.deepcopy(data.contract)
    config = sample_data_config(contract, row)
    config['link_to_tcp'] = pose_matrix([.03, -.07, .12], Rotation.from_euler('xyz', [10, 20, 30], degrees=True).as_quat()).tolist()
    config['depth'].update(kind=kind, round_to_mm=False)
    # Cache uses canonical optical extrinsics, independent of the raw sensor axes.
    sample['tcp_pose'][3:] = Rotation.from_euler('xyz', [-30, 10, 80], degrees=True).as_quat()
    for name in config['cameras']:
        sample[name + '_camera_extrinsics'] = pose_matrix([.2, -.1, 1.],
            Rotation.from_euler('xyz', [25, -10, 60], degrees=True).as_quat()).astype(np.float32)
    raw = raw_from_prepared(sample, config)
    policy = SimpleNamespace(contract=contract, chunk_settings=agent.settings, action_config=agent.oht_action_config)
    actual, _ = decode_raw(dict(task=row['task'], data_profile=row['data_profile'], tensors=pack_observation(raw)), policy)
    np.testing.assert_allclose(actual['tcp_pose'], sample['tcp_pose'], atol=1e-6)
    np.testing.assert_allclose(actual['joint_positions'], sample['joint_positions'])
    np.testing.assert_allclose(actual['gripper_open01'], sample['gripper_open01'], atol=1e-6)
    for name in config['cameras']:
        np.testing.assert_array_equal(actual[name + '_rgb'], sample[name + '_rgb'])
        np.testing.assert_allclose(actual[name + '_depth'], sample[name + '_depth'], atol=1e-6, equal_nan=True)
        np.testing.assert_allclose(actual[name + '_camera_extrinsics'], sample[name + '_camera_extrinsics'], atol=1e-6)


def test_smoke_compares_real_chunk_http_outputs_without_sending_gt(chunk_fixture, tmp_path, monkeypatch):
    agent, data, config = agent_data(chunk_fixture, training=False)
    policy = Policy(agent, data.contract, 'cpu', 'baseline')
    policy.training_config = config
    observations = []
    predict = policy.act
    def capture(observation, *args, **kwargs):
        observations.append(copy.deepcopy(observation))
        return predict(observation, *args, **kwargs)
    policy.act = capture
    checkpoint = tmp_path / 'model.pth'
    checkpoint.write_bytes(b'injected CPU loader')
    app = create_app(checkpoint, policy_loader=lambda *args: policy,
                     output_dir=tmp_path / 'logs', visualize_every=0)
    requests = []
    with TestClient(app) as client:
        class Opener:
            def open(self, request, timeout):
                if request.data:
                    requests.append(json.loads(request.data))
                response = client.request(request.get_method(), request.full_url.replace('http://local', ''),
                                          content=request.data, headers=dict(request.header_items()))
                assert response.status_code == 200, response.text
                return io.BytesIO(response.content)
        monkeypatch.setattr('urllib.request.build_opener', lambda *args: Opener())
        row = data.samples[0]
        result = smoke('http://local', data.root, split='all', task=row['task'],
                       episode_index=row['episode_index'], frame=row['frame'],
                       compare_targets=True, compare_raw=True)
        assert result['sample_id'] == row['id']
        assert result['checkpoint'] == str(checkpoint.resolve())
        assert result['direction_comparison']['prefixes']['2']['valid_ticks'] == 2
        assert result['raw_roundtrip']['position_difference_max_m'] < 1e-6
        assert len(observations) == 2
        for name in observations[0]:
            np.testing.assert_allclose(observations[0][name], observations[1][name], atol=1e-6, equal_nan=True,
                                       err_msg=f'Prepared/raw decoder input differs: {name}')
        predicts = [item for item in requests if 'observation' in item]
        assert len(predicts) == 2
        assert predicts[0]['episode_id'] != predicts[1]['episode_id']
        assert predicts[0]['observation_format'] == 'bridgevla_prepared_v1'
        assert predicts[1]['observation_format'] == 'bridgevla_rgbd_v1'
        for request in predicts:
            fields = request['observation'].get('tensors', request['observation'])
            assert not {'action_chunk', 'key_action', 'joint_chunk', 'action_valid'} & fields.keys()
        resets = len([item for item in requests if 'observation' not in item])
        with pytest.raises(ValueError, match='No matching sample'):
            smoke('http://local', data.root, frame=999999, compare_targets=True)
        assert len([item for item in requests if 'observation' not in item]) == resets
