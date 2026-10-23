"""A distinct prediction schema; single-action clients cannot execute a chunk as a key."""
import numpy as np
from ..action_modes import validate_action

SCHEMA = 'bridgevla_oht_eef_chunk_v1'


def chunk_spec(settings, contract):
    return dict(schema=SCHEMA, horizon=settings['horizon'], execute_horizon=settings['execute_horizon'],
                action_dt=contract['chunk']['action_dt'], frame='world', quaternion_order='xyzw',
                reference_point='tcp', gripper='1=open,0=close', action_size=8)


def validate_spec(spec):
    if not isinstance(spec, dict) or set(spec) != {
            'schema', 'horizon', 'execute_horizon', 'action_dt', 'frame', 'quaternion_order',
            'reference_point', 'gripper', 'action_size'}:
        raise ValueError('Invalid chunk prediction specification')
    if (spec['schema'] != SCHEMA or spec['frame'] != 'world' or spec['quaternion_order'] != 'xyzw' or
            spec['reference_point'] != 'tcp' or spec['gripper'] != '1=open,0=close' or spec['action_size'] != 8 or
            type(spec['horizon']) is not int or type(spec['execute_horizon']) is not int or
            not 1 <= spec['execute_horizon'] <= spec['horizon'] or
            type(spec['action_dt']) not in (float, int) or not np.isfinite(spec['action_dt']) or spec['action_dt'] <= 0):
        raise ValueError('Unsupported chunk coordinates, dimensions or timing')
    return spec


def validate_prediction(prediction, spec):
    validate_spec(spec)
    actions = np.asarray(prediction['actions'], dtype=np.float32)
    if actions.shape != (spec['horizon'], 8) or not np.isfinite(actions).all():
        raise ValueError('Chunk prediction must be finite [K,8]')
    if not np.allclose(np.linalg.norm(actions[:, 3:7], axis=1), 1., atol=1e-5, rtol=1e-5):
        raise ValueError('Chunk quaternions must be unit xyzw')
    actions = np.stack([validate_action(action, None) for action in actions])
    key = validate_action(prediction['key_action'], None)
    offsets = np.arange(1, spec['horizon'] + 1) * spec['action_dt']
    if 'time_offsets' in prediction and not np.array_equal(np.asarray(prediction['time_offsets']), offsets):
        raise ValueError('Chunk time offsets do not match advertised action_dt')
    return dict(key_action=key, actions=actions, time_offsets=offsets)
