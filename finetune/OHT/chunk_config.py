"""Explicit opt-in settings for continuous EEF policies."""
import math


DEFAULTS = dict(enabled=False, horizon=16, execute_horizon=4, hidden_dim=256,
                layers=2, heads=4, dropout=0., position_scale=.25,
                rotation_weight=1., gripper_weight=1., stage='frozen_key', key_loss_weight=1.)


def chunk_settings(config, contract=None):
    supplied = config.get('chunk_policy', {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError('Invalid chunk_policy configuration')
    result = {**DEFAULTS, **supplied}
    if type(result['enabled']) is not bool:
        raise ValueError('chunk_policy.enabled must be boolean')
    if not result['enabled']:
        return result
    for name in ('horizon', 'execute_horizon', 'hidden_dim', 'layers', 'heads'):
        if type(result[name]) is not int or result[name] < 1:
            raise ValueError(f'chunk_policy.{name} must be a positive integer')
    if result['execute_horizon'] > result['horizon'] or result['hidden_dim'] % result['heads']:
        raise ValueError('Chunk execution horizon or attention dimensions are invalid')
    for name in ('dropout', 'position_scale', 'rotation_weight', 'gripper_weight', 'key_loss_weight'):
        value = result[name]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError(f'chunk_policy.{name} must be finite and nonnegative')
    if result['dropout'] >= 1 or result['position_scale'] <= 0:
        raise ValueError('Chunk dropout must be <1 and position_scale >0')
    if result['stage'] not in ('frozen_key', 'joint'):
        raise ValueError('chunk_policy.stage must be frozen_key or joint')
    if result['stage'] == 'joint' and result['key_loss_weight'] <= 0:
        raise ValueError('Joint chunk training requires key_loss_weight >0')
    if config.get('mode') != 'baseline' or config.get('action_mode', 'eef') != 'eef':
        raise ValueError('Chunk baseline currently requires mode=baseline and action_mode=eef')
    if config.get('agent', {}).get('transform_augmentation'):
        raise ValueError('Chunk training requires transform_augmentation=false for measured proprioception')
    if contract is not None:
        if contract.get('schema') != 'oht_bridgevla_v4':
            raise ValueError('Chunk policy requires a V4 replay/checkpoint contract')
        if result['horizon'] > contract['chunk']['horizon']:
            raise ValueError('Model horizon exceeds V4 supervised horizon')
    return result
