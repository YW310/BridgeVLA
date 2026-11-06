"""Probe recorded current observations through /predict; never execute returned actions."""
import argparse
import json
from pathlib import Path
import sys
import urllib.request
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetune.OHT.data.common import TASKS, read_jsonl, inside, file_digest, write_json
from finetune.OHT.data.replay import load_contract
from finetune.OHT.runtime.inference_api import API_VERSION, OBSERVATION_FORMAT, RAW_OBSERVATION_FORMAT
from finetune.OHT.runtime.transport import pack_observation


def direction_report(actions, targets, valid, current, horizon, epsilon=1e-4):
    """Compare cumulative and per-tick world displacements, excluding stationary directions."""
    actions, targets, current = (np.asarray(value, dtype=float) for value in (actions, targets, current))
    valid = np.asarray(valid, dtype=bool)
    rows = []
    previous_pred, previous_gt = current[:3], current[:3]

    def angle(first, second):
        a, b = np.linalg.norm(first), np.linalg.norm(second)
        if min(a, b) <= epsilon:
            return None
        return float(np.rad2deg(np.arccos(np.clip(np.dot(first, second) / (a * b), -1, 1))))

    for tick in np.flatnonzero(valid[:len(actions)]):
        predicted, observed = actions[tick, :3], targets[tick, :3]
        delta, gt_delta = predicted - current[:3], observed - current[:3]
        rows.append(dict(offset=int(tick + 1), predicted_displacement_world_m=delta.tolist(),
                         observed_displacement_world_m=gt_delta.tolist(),
                         direction_error_degrees=angle(delta, gt_delta),
                         step_direction_error_degrees=angle(predicted - previous_pred, observed - previous_gt),
                         prediction_stationary=bool(np.linalg.norm(delta) <= epsilon),
                         observed_stationary=bool(np.linalg.norm(gt_delta) <= epsilon),
                         position_error_m=float(np.linalg.norm(predicted - observed))))
        previous_pred, previous_gt = predicted, observed

    def summary(length):
        selected = [row for row in rows if row['offset'] <= length]
        angles = [row['direction_error_degrees'] for row in selected if row['direction_error_degrees'] is not None]
        return dict(valid_ticks=len(selected), compared_directions=len(angles),
                    opposite_directions=sum(value > 90 for value in angles),
                    direction_error_degrees_mean=float(np.mean(angles)) if angles else None,
                    position_error_m_mean=float(np.mean([row['position_error_m'] for row in selected])) if selected else None)

    return dict(frame='world', stationary_epsilon_m=epsilon, offsets=rows,
                prefixes={str(length): summary(length) for length in sorted({1, horizon, len(actions)})})


def raw_from_prepared(observation, config):
    """Synthetic raw-wire roundtrip at cached resolution, NOT a live sensor observation."""
    from scipy.spatial.transform import Rotation
    from finetune.OHT.data.geometry import pose_matrix
    from finetune.OHT.runtime.raw_observation import OPENGL_TO_OPTICAL
    tcp = np.asarray(observation['tcp_pose'])
    link = pose_matrix(tcp[:3], tcp[3:]) @ np.linalg.inv(config['link_to_tcp'])
    q = Rotation.from_matrix(link[:3, :3]).as_quat()
    grip = config['gripper']
    result = dict(ee_link_pose=np.r_[link[:3, 3], q[[3, 0, 1, 2]]].astype(np.float32),
                  joint_positions=np.asarray(observation['joint_positions']).copy(),
                  gripper_position=np.asarray([grip['close'] - float(observation['gripper_open01']) *
                                               (grip['close'] - grip['open'])], np.float32))
    for name, calibration in config['cameras'].items():
        optical = np.asarray(observation[name + '_camera_extrinsics'])
        sensor = optical @ OPENGL_TO_OPTICAL
        q = Rotation.from_matrix(sensor[:3, :3]).as_quat()
        intrinsic = np.asarray(observation[name + '_camera_intrinsics'])
        depth = np.asarray(observation[name + '_depth'])[0].copy()
        if calibration.get('depth', config['depth'])['kind'] == 'ray':
            yy, xx = np.indices(depth.shape)
            rays = np.stack((xx, yy, np.ones_like(xx)), -1) @ np.linalg.inv(intrinsic).T
            depth /= np.linalg.norm(rays, axis=-1)
        result.update({name + '_rgb': np.asarray(observation[name + '_rgb']).transpose(1, 2, 0).copy(),
                       name + '_depth': depth.astype(np.float32), name + '_intrinsics': intrinsic.copy(),
                       name + '_pose': np.r_[sensor[:3, 3], q[[3, 0, 1, 2]]].astype(np.float32)})
    return result


def smoke(server, replay, *, split="val", index=0, timeout=60., episode_id=None,
          task=None, episode_index=None, frame=None, compare_targets=False, compare_raw=False):
    if index < 0:
        raise ValueError("index must be nonnegative")
    if any(value is not None and value < 0 for value in (episode_index, frame)):
        raise ValueError('episode-index and frame must be nonnegative')
    server = server.rstrip("/")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(path, payload=None):
        data = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        request = urllib.request.Request(server + path, data=data,
                                         headers={"Content-Type": "application/json"})
        # A control request is never automatically retried after a timeout.
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)

    info = call("/model_info")
    if info["api_version"] != API_VERSION or info["observation_format"] != OBSERVATION_FORMAT:
        raise ValueError("Server does not support the prepared BridgeVLA /predict API")
    replay = Path(replay)
    schema = json.loads((replay / 'contract.json').read_text(encoding='utf-8')).get('schema')
    if schema == 'oht_bridgevla_v4':
        from finetune.OHT.data.chunk_dataset import load_chunk_contract
        contract = load_chunk_contract(replay)
        if 'chunk_spec' not in info:
            raise ValueError('V4 smoke requires a chunk checkpoint/server')
    else:
        contract = load_contract(replay)
    if (compare_targets or compare_raw) and schema != 'oht_bridgevla_v4':
        raise ValueError('Chunk direction/raw comparisons require V4 replay')
    if compare_raw and RAW_OBSERVATION_FORMAT not in info.get('observation_formats', []):
        raise ValueError('Server does not advertise raw RGB-D observations')
    if info["contract_sha256"] != contract["sha256"]:
        raise ValueError("Server checkpoint and smoke replay contracts differ")
    rows = [row for row in read_jsonl(replay / "samples.jsonl") if (split == 'all' or row['split'] == split)
            and (task is None or row['task'] == task)
            and (episode_index is None or row['episode_index'] == episode_index)
            and (frame is None or row['frame'] == frame)]
    if index >= len(rows):
        raise ValueError(f'No matching sample at index {index}; {len(rows)} rows match split/task/episode/frame')
    row = rows[index]
    path = inside(replay, row["observation"])
    if file_digest(path) != row["observation_sha256"]:
        raise ValueError("Smoke observation checksum mismatch")
    with np.load(path, allow_pickle=False) as source:
        # Only declared current observations are sent. Comparison labels remain local.
        observation = {name: source[name] for name in info["observation_schema"]}
    episode_id = episode_id or "smoke-" + uuid.uuid4().hex
    identity = dict(api_version=API_VERSION, model_id=info["model_id"],
                    contract_sha256=info["contract_sha256"], episode_id=episode_id)
    reset = call("/reset", dict(**identity, request_id=episode_id + "-reset"))
    if reset.get("episode_id") != episode_id or reset.get("reset") is not True:
        raise ValueError("Unexpected reset response")
    request = dict(**identity, request_id=episode_id + "-0", step=0, timestamp=0.,
                   instruction=row["goal"], observation_format=OBSERVATION_FORMAT,
                   observation=pack_observation(observation))
    if 'chunk_spec' in info:
        request['prediction_schema'] = info['chunk_spec']['schema']
    body_size = len(json.dumps(request, allow_nan=False).encode("utf-8"))
    if body_size > info["max_request_bytes"]:
        raise ValueError("Smoke request exceeds server max_request_bytes")
    response = call("/predict", request)
    if any(response.get(key) != request[key] for key in (
            "api_version", "model_id", "contract_sha256", "request_id", "episode_id", "step", "timestamp")):
        raise ValueError("Stale/mismatched prediction response")
    if response.get("action_spec") != info["action_spec"]:
        raise ValueError("Prediction output description changed")
    if 'chunk_spec' in info:
        from finetune.OHT.runtime.chunk_protocol import validate_prediction
        if response.get('chunk_spec') != info['chunk_spec']:
            raise ValueError('Chunk output description changed')
        validate_prediction(response, info['chunk_spec'])
    result = dict(sample_id=row['id'], sample_frame=row['frame'], task=row['task'], instruction=row['goal'],
                  checkpoint=info.get('checkpoint'), chunk_policy=info.get('chunk_policy'), **response)
    if compare_targets or compare_raw:
        result['current_tcp_pose_xyzw'] = observation['tcp_pose'].tolist()
    if compare_raw:
        from finetune.OHT.data.source_config import sample_data_config
        from finetune.OHT.data.geometry import rotation_error
        config = sample_data_config(contract, row)
        profile = row['data_profile']
        if profile not in info['raw_observation']['data_profiles']:
            raise ValueError('Recorded sample data profile is unavailable on this server')
        raw_episode = episode_id + '-raw'
        raw_identity = dict(identity, episode_id=raw_episode)
        reset = call('/reset', dict(**raw_identity, request_id=raw_episode + '-reset'))
        if reset.get('reset') is not True or any(reset.get(key) != value for key, value in raw_identity.items()):
            raise ValueError('Unexpected raw-roundtrip reset response')
        raw_request = dict(request, **raw_identity, request_id=raw_episode + '-0',
                           observation_format=RAW_OBSERVATION_FORMAT,
                           observation=dict(task=row['task'], data_profile=profile,
                                            tensors=pack_observation(raw_from_prepared(observation, config))))
        if len(json.dumps(raw_request, allow_nan=False).encode('utf-8')) > info['max_request_bytes']:
            raise ValueError('Raw roundtrip exceeds server max_request_bytes')
        raw_response = call('/predict', raw_request)
        if any(raw_response.get(key) != raw_request[key] for key in (*raw_identity, 'request_id', 'step', 'timestamp')):
            raise ValueError('Stale/mismatched raw roundtrip response')
        if raw_response.get('chunk_spec') != info['chunk_spec'] or raw_response.get('action_spec') != info['action_spec']:
            raise ValueError('Raw roundtrip output description changed')
        raw_actions = validate_prediction(raw_response, info['chunk_spec'])['actions']
        prepared_actions = np.asarray(response['actions'])
        settings = info.get('chunk_policy') or {}
        result['raw_roundtrip'] = dict(actions=raw_actions.tolist(),
            position_difference_max_m=float(np.linalg.norm(raw_actions[:, :3] - prepared_actions[:, :3], axis=-1).max()),
            rotation_difference_max_degrees=float(np.rad2deg(rotation_error(raw_actions[:, 3:7], prepared_actions[:, 3:7])).max()),
            gripper_disagreements=int(np.sum(raw_actions[:, 7] != prepared_actions[:, 7])),
            flow_sampling_may_differ=settings.get('decoder_type') == 'flow_matching' and settings.get('flow_inference_seed') is None,
            note='Synthetic raw input at cached resolution; does not validate live cameras, rendering or scene alignment. '
                 'Fresh-session gripper hysteresis may differ from the recorded history at intermediate frames.')
    if compare_targets:
        from finetune.OHT.data.chunk_actions import validate_chunk_targets
        path = inside(replay, row['targets'])
        if file_digest(path) != row['targets_sha256']:
            raise ValueError('Comparison target checksum mismatch')
        with np.load(path, allow_pickle=False) as source:
            targets = {name: source[name] for name in source.files}
        validate_chunk_targets(targets, contract['chunk'], row)
        result['direction_comparison'] = direction_report(response['actions'], targets['action_chunk'],
            targets['action_valid'], observation['tcp_pose'], info['chunk_spec']['execute_horizon'])
        if compare_raw:
            result['raw_roundtrip']['direction_comparison'] = direction_report(raw_actions, targets['action_chunk'],
                targets['action_valid'], observation['tcp_pose'], info['chunk_spec']['execute_horizon'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", default="http://127.0.0.1:8010")
    parser.add_argument("--replay", required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="val")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=60.)
    parser.add_argument("--episode-id")
    parser.add_argument('--task', choices=TASKS)
    parser.add_argument('--episode-index', type=int)
    parser.add_argument('--frame', type=int)
    parser.add_argument('--compare-targets', action='store_true', help='Compare returned chunk with local GT after prediction; never send labels')
    parser.add_argument('--compare-raw', action='store_true', help='Also send a synthetic raw roundtrip of this cached observation in a fresh session')
    parser.add_argument('--output', help='Save the complete diagnostic JSON')
    args = parser.parse_args(argv)
    options = vars(args)
    output = options.pop('output')
    result = smoke(**options)
    if output:
        write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
