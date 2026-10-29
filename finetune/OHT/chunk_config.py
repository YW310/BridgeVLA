"""Explicit opt-in settings for continuous EEF policies."""
import math


DEFAULTS = dict(enabled=False, horizon=16, execute_horizon=4, hidden_dim=256,
                layers=2, heads=4, dropout=0., position_scale=.25, goal_conditioning='absolute',
                position_weight=1., displacement_weight=0., displacement_scale=.01,
                rotation_weight=1., gripper_weight=1., stage='frozen_key', key_loss_weight=1.,
                spatial_conditioning=False, decoder_type='regression', flow_steps=8,
                flow_weight=1., flow_inference_seed=0,
                action_position_scale=None, geometry_scale=None, position_loss_scale=None,
                normalization_quantile=.99, normalization_min_scale=.01, normalization_max_samples=4096,
                flow_time_sampling='uniform', adaln=False, temporal_kernel_size=1,
                second_difference_weight=0., second_difference_scale=.001,
                state_conditioning=False, proprio_normalization='none',
                proprio_center=None, proprio_scale=None, proprio_min_scale=.1,
                key_condition_gate=False, key_condition_gate_init=.1,
                prediction_representation='relative_pose')


def chunk_scale(settings, name):
    value = settings.get(name)
    if value == 'train_quantile':
        raise ValueError('Resolve train_quantile with training replay before constructing the chunk model')
    return settings['position_scale'] if value is None else value


def chunk_settings(config, contract=None):
    supplied = config.get('chunk_policy', {})
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError('Invalid chunk_policy configuration')
    result = {**DEFAULTS, **supplied}
    if type(result['enabled']) is not bool:
        raise ValueError('chunk_policy.enabled must be boolean')
    if not result['enabled']:
        return result
    for name in ('horizon', 'execute_horizon', 'hidden_dim', 'layers', 'heads', 'flow_steps',
                 'normalization_max_samples', 'temporal_kernel_size'):
        if type(result[name]) is not int or result[name] < 1:
            raise ValueError(f'chunk_policy.{name} must be a positive integer')
    if result['execute_horizon'] > result['horizon'] or result['hidden_dim'] % result['heads']:
        raise ValueError('Chunk execution horizon or attention dimensions are invalid')
    for name in ('dropout', 'position_scale', 'position_weight', 'displacement_weight', 'displacement_scale',
                 'rotation_weight', 'gripper_weight', 'key_loss_weight', 'flow_weight',
                 'second_difference_weight', 'second_difference_scale', 'normalization_min_scale',
                 'normalization_quantile', 'proprio_min_scale', 'key_condition_gate_init'):
        value = result[name]
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError(f'chunk_policy.{name} must be finite and nonnegative')
    if result['dropout'] >= 1 or result['position_scale'] <= 0:
        raise ValueError('Chunk dropout must be <1 and position_scale >0')
    if result['displacement_scale'] <= 0:
        raise ValueError('chunk_policy.displacement_scale must be positive')
    for name in ('action_position_scale', 'geometry_scale', 'position_loss_scale'):
        value = result[name]
        if value is None or (name == 'action_position_scale' and value == 'train_quantile'):
            continue
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'chunk_policy.{name} must be positive or null'
                             + (' or train_quantile' if name == 'action_position_scale' else ''))
    if not 0 < result['normalization_quantile'] <= 1 or result['normalization_min_scale'] <= 0:
        raise ValueError('Invalid chunk action normalization quantile or minimum scale')
    if result['second_difference_scale'] <= 0:
        raise ValueError('chunk_policy.second_difference_scale must be positive')
    if type(result['adaln']) is not bool or result['temporal_kernel_size'] % 2 != 1:
        raise ValueError('Chunk adaln must be boolean and temporal_kernel_size must be odd')
    if result['flow_time_sampling'] not in ('uniform', 'beta_high_noise'):
        raise ValueError('chunk_policy.flow_time_sampling must be uniform or beta_high_noise')
    if (result['adaln'] or result['temporal_kernel_size'] > 1) and not result['spatial_conditioning']:
        raise ValueError('AdaLN and temporal convolution require spatial_conditioning=true')
    if result['goal_conditioning'] not in ('absolute', 'relative_tcp'):
        raise ValueError('chunk_policy.goal_conditioning must be absolute or relative_tcp')
    if type(result['spatial_conditioning']) is not bool:
        raise ValueError('chunk_policy.spatial_conditioning must be boolean')
    if result['decoder_type'] not in ('regression', 'flow_matching'):
        raise ValueError('chunk_policy.decoder_type must be regression or flow_matching')
    for name in ('state_conditioning', 'key_condition_gate'):
        if type(result[name]) is not bool:
            raise ValueError(f'chunk_policy.{name} must be boolean')
    if result['state_conditioning'] and not (result['spatial_conditioning'] and result['adaln']):
        raise ValueError('Direct state conditioning requires spatial_conditioning=true and adaln=true')
    if result['key_condition_gate'] and not result['state_conditioning']:
        raise ValueError('Key condition gate requires state_conditioning=true')
    if not 0 < result['key_condition_gate_init'] < 1:
        raise ValueError('chunk_policy.key_condition_gate_init must be between 0 and 1')
    if result['proprio_normalization'] not in ('none', 'train_quantile') or result['proprio_min_scale'] <= 0:
        raise ValueError('Invalid proprio normalization method or minimum scale')
    if result['proprio_normalization'] != 'none' and not result['spatial_conditioning']:
        raise ValueError('Proprio normalization requires spatial_conditioning=true')
    if result['proprio_normalization'] == 'train_quantile' and result['normalization_quantile'] <= .5:
        raise ValueError('Proprio normalization requires normalization_quantile >0.5')
    center, scale = result['proprio_center'], result['proprio_scale']
    if (center is None) != (scale is None):
        raise ValueError('Proprio normalization requires both center and scale')
    if center is not None:
        for name, values in (('proprio_center', center), ('proprio_scale', scale)):
            if (not isinstance(values, list) or len(values) != 7 or
                    any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
                raise ValueError(f'chunk_policy.{name} must be a finite seven-element list')
        if min(scale) <= 0 or result['proprio_normalization'] == 'none':
            raise ValueError('Proprio scales must be positive and require train_quantile normalization')
    if result['prediction_representation'] not in ('relative_pose', 'delta_pose'):
        raise ValueError('chunk_policy.prediction_representation must be relative_pose or delta_pose')
    if result['prediction_representation'] == 'delta_pose' and (
            not result['spatial_conditioning'] or result['decoder_type'] != 'regression'):
        raise ValueError('Delta pose representation currently requires spatial regression')
    if result['spatial_conditioning'] and result['goal_conditioning'] != 'relative_tcp':
        raise ValueError('Spatial chunk conditioning requires goal_conditioning=relative_tcp')
    if result['decoder_type'] == 'flow_matching' and (not result['spatial_conditioning'] or result['flow_weight'] <= 0):
        raise ValueError('Flow matching requires spatial_conditioning=true and flow_weight >0')
    if result['flow_inference_seed'] is not None and (type(result['flow_inference_seed']) is not int or result['flow_inference_seed'] < 0):
        raise ValueError('chunk_policy.flow_inference_seed must be a nonnegative integer or null')
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
