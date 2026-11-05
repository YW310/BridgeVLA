"""IsaacLab metric Z-depth/OpenGL poses -> checkpoint-bound prepared observations."""
from copy import deepcopy
import logging
from pathlib import PurePosixPath
import numpy as np

from ..action_modes import state_fields
from ..data.actions import gripper_step, low_dim
from ..data.common import GOALS
from ..data.geometry import array, tcp_pose
from ..data.observation import camera_observation, validate_observation
from .inference_api import APIError, RAW_OBSERVATION_FORMAT
from .transport import unpack_observation

RAW_FORMAT = RAW_OBSERVATION_FORMAT
OPENGL_TO_OPTICAL = np.diag([1., -1., -1., 1.])
LOG = logging.getLogger(__name__)


def data_profiles(policy):
    return {name: profile['data_config'] for name, profile in policy.contract.get('source_data_configs', {}).items()} or {
        'default': policy.contract['data_config']}


def profile_task(name):
    # Match the original and segmentation directories accepted by reader.discover.
    tasks = {part.removesuffix('-seg') for part in PurePosixPath(name).parts} & set(GOALS)
    return next(iter(tasks)) if len(tasks) == 1 else None


def raw_schema(policy):
    return dict(format=RAW_FORMAT, depth='metric_z_metres', rgb='uint8 HWC',
                camera_pose='sensor OpenGL -> env-origin-relative world, xyz+wxyz',
                ee_link_pose='env-origin-relative world xyz+wxyz; server applies link_to_tcp',
                gripper_position='physical sum of measured gripper joints, not binary command',
                cameras=list(policy.contract['data_config']['cameras']),
                tasks=list(GOALS), data_profiles={
                    name: dict(task=profile_task(name), link_to_tcp=config['link_to_tcp'],
                               gripper={key: config['gripper'].get(key, default) for key, default in (
                                   ('open', None), ('close', None), ('open_threshold', .9),
                                   ('close_threshold', .1), ('diff_threshold', .05), ('initial_state', 1))},
                               scene_bounds=config['scene_bounds'],
                               image_size=config['image_size'], keypoints=deepcopy(config.get('keypoints', {})))
                    for name, config in data_profiles(policy).items()})


def decode_raw(packed, policy, previous=None, *, sensor_callback=None):
    """Return observation + pending causal gripper state; caller commits on success."""
    try:
        if not isinstance(packed, dict) or set(packed) != {'task', 'data_profile', 'tensors'}:
            raise ValueError('Raw observation requires exactly task, data_profile, tensors')
        task, profile = packed['task'], packed['data_profile']
        profiles = data_profiles(policy)
        if not isinstance(task, str) or task not in GOALS or not isinstance(profile, str) or profile not in profiles:
            raise ValueError('Unknown task or data_profile; select a profile advertised by /model_info')
        if profile_task(profile) not in (None, task):
            raise ValueError('Task does not match source data profile')
        config = deepcopy(profiles[profile])
        fields = {'joint_positions', 'ee_link_pose', 'gripper_position'}
        for camera in config['cameras']:
            fields.update(f'{camera}_{suffix}' for suffix in ('rgb', 'depth', 'intrinsics', 'pose'))
        tensors = packed['tensors']
        if not isinstance(tensors, dict) or set(tensors) != fields:
            raise ValueError('Raw tensor fields must match configured cameras and robot state; no teachers/labels')
        for name, spec in tensors.items():
            if not isinstance(spec, dict) or set(spec) != {'dtype', 'shape', 'data'}:
                raise ValueError(f'{name}: expected dtype, shape, data')
            allowed = ('uint8',) if name.endswith('_rgb') else ('float32', 'float64')
            if spec['dtype'] not in allowed or not isinstance(spec['data'], str):
                raise ValueError(f'{name}: invalid raw sensor dtype')
        values = unpack_observation(tensors)
        joints = array(values['joint_positions'], (6,), 'joint_positions')
        link = array(values['ee_link_pose'], (7,), 'ee_link_pose')
        tcp = tcp_pose(link[:3], link[3:][[1, 2, 3, 0]], config['link_to_tcp'])
        raw_gripper = float(array(values['gripper_position'], (1,), 'gripper_position')[0])
        grip = config['gripper']
        measured = float(np.clip((grip['close'] - raw_gripper) / (grip['close'] - grip['open']), 0, 1))
        last_measured, last_state = previous if previous is not None else (None, None)
        binary = gripper_step(measured, last_measured, last_state, grip)
        result = dict(low_dim_state=low_dim(measured, binary_state=binary))
        if getattr(policy, 'chunk_settings', None) is not None:
            result.update(tcp_pose=tcp.astype(np.float32), joint_positions=joints.astype(np.float32),
                          gripper_open01=np.asarray(measured, dtype=np.float32))
        for name in state_fields(policy.action_config):
            result[name] = (joints if name == 'joint_positions' else tcp).astype(np.float32)
        # Input axes are fixed by this wire format, independent of recorded-pose conventions.
        config['camera_quaternion_order'], config['camera_extrinsic_direction'] = 'wxyz', 'camera_to_world'
        for camera, calibration in config['cameras'].items():
            rgb, depth = values[f'{camera}_rgb'], values[f'{camera}_depth'].astype(np.float32)
            if depth.ndim != 2 or np.isinf(depth).any():
                raise ValueError(f'{camera}: metric Z-depth must be HxW; invalid values use NaN')
            calibration['intrinsics'] = array(values[f'{camera}_intrinsics'], (3, 3), 'intrinsics').tolist()
            calibration['optical_to_sensor'] = OPENGL_TO_OPTICAL.tolist()
            spec = calibration.get('depth', config['depth'])
            if spec['kind'] == 'ray':
                yy, xx = np.indices(depth.shape)
                rays = np.stack((xx, yy, np.ones_like(xx)), -1) @ np.linalg.inv(calibration['intrinsics']).T
                depth = (depth * np.linalg.norm(rays, axis=-1)).astype(np.float32)
            if spec.get('round_to_mm', False):
                depth = np.round(depth * 1000).astype(np.float32) / 1000
            result.update(camera_observation(camera, rgb, depth, values[f'{camera}_pose'], config))
        result = validate_observation(result, config)
        if sensor_callback is not None:
            # Reuse this validated decode before resize/rounding. Diagnostics
            # never feed back into the policy and must not reject an action.
            try:
                sensor_callback(values, config, tcp)
            except Exception:
                LOG.warning('Raw sensor visualization snapshot skipped', exc_info=True)
        return result, (measured, binary)
    except (ValueError, TypeError, KeyError, OverflowError, np.linalg.LinAlgError) as exc:
        raise APIError(422, 'invalid_observation', str(exc)) from exc
