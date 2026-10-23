"""Strict V4 cache reader; kept separate from the legacy single-keypoint trainer."""
import json
from pathlib import Path

import numpy as np

from .chunk_actions import SCHEMA, chunk_options, validate_chunk_targets
from .common import TASKS, digest, file_digest, inside, read_jsonl
from .dataset import OHTDataset, collate
from .observation import validate_data_config, validate_observation
from .source_config import sample_data_config


def load_chunk_contract(root):
    root = Path(root)
    contract = json.loads((root / 'contract.json').read_text(encoding='utf-8'))
    complete = json.loads((root / 'complete.json').read_text(encoding='utf-8'))
    value = dict(contract)
    expected = value.pop('sha256', None)
    if expected != digest(value) or complete.get('contract_sha256') != expected:
        raise ValueError('V4 replay contract checksum mismatch')
    if (contract.get('schema') != SCHEMA or contract.get('replay_version') != 4 or
            contract.get('action_targets_version') != 2 or contract.get('joint_count') != 6 or
            contract.get('target') != 'key_goal_and_continuous_absolute_tcp' or
            contract.get('policy_frame') != 'world' or contract.get('quaternion_order') != 'xyzw' or
            contract.get('reference_point') != 'tcp' or contract.get('gripper_convention') != '1=open,0=close' or
            contract.get('label_storage') != 'separate_npz'):
        raise ValueError('Unsupported V4 replay schema or action coordinate contract')
    options = chunk_options(contract.get('chunk'))
    if options['action_dt'] is None or contract.get('chunk') != options:
        raise ValueError('V4 chunk contract requires explicit resolved action_dt and all chunk options')
    if type(contract.get('sample_stride')) is not int or contract['sample_stride'] < 1:
        raise ValueError('Invalid V4 observation sample_stride')
    validate_data_config(contract['data_config'], resolved=True)
    for profile in contract.get('source_data_configs', {}).values():
        validate_data_config(profile['data_config'], resolved=True)
    index_hash = file_digest(root / 'samples.jsonl')
    if index_hash != complete.get('index_sha256') or index_hash != contract.get('index_sha256'):
        raise ValueError('V4 replay sample index changed or not bound to contract')
    rows = read_jsonl(root / 'samples.jsonl')
    if not rows or type(complete.get('samples')) is not int or complete['samples'] != len(rows):
        raise ValueError('V4 replay sample count mismatch or empty index')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate V4 replay sample IDs')
    groups, episodes = {}, {}
    for row in rows:
        sample_data_config(contract, row)
        split = row.get('split')
        if split not in ('train', 'val', 'test') or row.get('task') not in TASKS:
            raise ValueError('Invalid V4 replay task or split')
        if any(type(row.get(name)) is not int for name in ('frame', 'target_frame', 'episode_index', 'episode_length')):
            raise ValueError('Invalid V4 frame/episode integer metadata')
        frame, target, length = row['frame'], row['target_frame'], row['episode_length']
        if (length < 2 or not 0 <= frame < length - 1 or row['episode_index'] < 0 or
                (target != -1 and not frame < target < length)):
            raise ValueError('Invalid V4 episode length or non-future key target')
        timestamp = row.get('timestamp')
        if isinstance(timestamp, bool) or not isinstance(timestamp, (float, int)) or not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError('Invalid V4 observation timestamp')
        if row['id'] != f"{row['task']}/{row['episode_index']:06d}/{frame:06d}":
            raise ValueError('V4 sample identity does not match task/episode/frame')
        if row.get('goal') is None or not isinstance(row['goal'], str) or not row['goal']:
            raise ValueError('V4 samples require a task instruction')
        for path_field in ('observation', 'targets'):
            if not isinstance(row.get(path_field), str) or not row[path_field]:
                raise ValueError(f'V4 sample requires {path_field} NPZ path')
            inside(root, row[path_field])
            checksum = row.get(path_field + '_sha256')
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise ValueError(f'V4 sample requires {path_field} checksum')
        group = row.get('group')
        if not isinstance(group, str) or not group:
            raise ValueError('V4 sample requires a scene/trajectory group')
        if groups.setdefault(group, split) != split:
            raise ValueError('Scene/trajectory group crosses dataset splits')
        identity = row['task'], row['episode_index']
        metadata = split, length, row['data_profile'], group
        if episodes.setdefault(identity, metadata) != metadata:
            raise ValueError('V4 episode metadata or split is inconsistent')
    return contract


class OHTChunkDataset:
    """Load float/bool chunk arrays without sending targets into policy observations."""
    validate_all = OHTDataset.validate_all

    def __init__(self, root, split='train'):
        if split not in ('train', 'val', 'test'):
            raise ValueError('Unknown dataset split')
        self.root = Path(root)
        self.role_cache = None
        self.contract = load_chunk_contract(root)
        self.samples = [row for row in read_jsonl(self.root / 'samples.jsonl') if row['split'] == split]
        if not self.samples:
            raise ValueError(f'No {split} samples in V4 replay')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        values = []
        for name in ('observation', 'targets'):
            path = inside(self.root, row[name])
            if file_digest(path) != row[name + '_sha256']:
                raise ValueError(f'V4 {name} checksum mismatch: {row["id"]}')
            with np.load(path, allow_pickle=False) as source:
                values.append({key: source[key] for key in source.files})
        observation, targets = values
        required = {'joint_positions', 'tcp_pose', 'gripper_open01'}
        if not required.issubset(observation):
            raise ValueError('V4 observation requires measured TCP, joints and gripper_open01')
        data = sample_data_config(self.contract, row)
        validate_observation(observation, data)
        validate_chunk_targets(targets, self.contract['chunk'], row)
        from .chunk_replay import _target_bounds
        _target_bounds(targets, data['scene_bounds'])
        if (observation['tcp_pose'].dtype != np.float32 or
                observation['joint_positions'].dtype != np.float32 or
                np.asarray(observation['gripper_open01']).dtype != np.float32):
            raise ValueError('V4 proprioception must be float32')
        if not np.isclose(np.linalg.norm(observation['tcp_pose'][3:]), 1., atol=1e-5, rtol=1e-5):
            raise ValueError('V4 measured TCP quaternion must have unit norm')
        opening_features = np.full(2, float(observation['gripper_open01']) * .04)
        if not np.allclose(observation['low_dim_state'][1:3], opening_features, atol=1e-6, rtol=1e-6):
            raise ValueError('V4 measured gripper differs from low_dim_state opening features')
        if not np.allclose(observation['tcp_pose'], row.get('current_tcp'), atol=1e-6, rtol=1e-6):
            raise ValueError('V4 observation TCP differs from sample index')
        return dict(**observation, **targets, goal=row['goal'], task=row['task'], sample_id=row['id'])


collate_chunk = collate
