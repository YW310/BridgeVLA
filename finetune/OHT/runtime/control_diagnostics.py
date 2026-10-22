"""Inspect live control semantics without applying actions or stepping physics."""
import numpy as np

from .joint_executor import arm_ids, numpy_value


def serial(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): serial(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(item) for item in value]
    return numpy_value(value).tolist()


def gripper_report(env, torch, training=None):
    term = env.action_manager.get_term('gripper_action')
    cfg = getattr(term, 'cfg', None)
    result = dict(type=type(term).__name__, module=type(term).__module__,
                  open_command=1., close_command=-1., model_open01='1=open, 0=closed',
                  open_command_expr=serial(getattr(cfg, 'open_command_expr', None)),
                  close_command_expr=serial(getattr(cfg, 'close_command_expr', None)),
                  training_physical_endpoints=training, verified=None, calibration_direction_matches=None)
    ids = getattr(term, 'joint_ids', getattr(term, '_joint_ids', None))
    if ids is not None:
        if isinstance(ids, slice):
            ids = list(range(len(env.scene['robot'].joint_names)))[ids]
        ids = numpy_value(ids).astype(int).reshape(-1).tolist()
        names = getattr(env.scene['robot'], 'joint_names', None)
        result.update(joint_ids=ids, joint_names=[names[i] for i in ids] if names is not None else None)
    required = ('raw_actions', 'processed_actions', 'process_actions', '_open_command', '_close_command')
    if not all(hasattr(term, name) for name in required):
        result['note'] = 'Cannot verify custom gripper transform automatically; inspect its action implementation'
        return result
    previous = term.raw_actions.clone()
    try:
        targets = {}
        for name, command in (('open', 1.), ('close', -1.)):
            term.process_actions(torch.full_like(previous, command))
            target = numpy_value(term.processed_actions)[0].astype(float).copy()
            targets[name] = target
            result[f'{name}_joint_targets'] = target.tolist()
            result[f'{name}_physical_sum'] = float(target.sum())
        result['verified'] = bool(all(np.allclose(targets[name], numpy_value(getattr(term, f'_{name}_command')))
                                      for name in ('open', 'close')))
        if training is not None:
            result['endpoint_difference'] = {
                name: result[f'{name}_physical_sum'] - float(training[name]) for name in ('open', 'close')}
            live_span = result['open_physical_sum'] - result['close_physical_sum']
            training_span = float(training['open']) - float(training['close'])
            # Different magnitudes can reflect dataset coverage. Opposite
            # directions invert the server's measured open01 normalization.
            result['calibration_direction_matches'] = bool(
                np.isfinite([live_span, training_span]).all() and live_span * training_span > 0)
    finally:
        term.process_actions(previous)
    return result


def control_report(bridge):
    env, runtime = bridge.env, bridge.ref
    term = env.action_manager.get_term('arm_action')
    cfg = getattr(term, 'cfg', None)
    term_ids, canonical_ids = arm_ids(env, bridge.client.action_config)
    profile = bridge.client.info['raw_observation']['data_profiles'][bridge.profile]
    return dict(action_spec=bridge.client.info['action_spec'], training_scene_bounds=profile.get('scene_bounds'),
                scene=getattr(runtime, 'scene_info', None),
                ee_link=getattr(runtime, 'ee_link', None),
                ee_pose_source=getattr(runtime, 'ee_pose_source', 'body'),
                body_to_observation=serial(getattr(runtime, 'body_to_observation', np.eye(4))),
                link_to_tcp=bridge.link_to_tcp.tolist(), observation_to_control=bridge.control_offset.tolist(),
                world_from_base=bridge._base().tolist(), measured=bridge.measured_state(),
                arm=dict(type=type(term).__name__, module=type(term).__module__,
                         body_name=getattr(cfg, 'body_name', None),
                         target_types=serial(getattr(getattr(cfg, 'controller_cfg', None), 'target_types', None)),
                         position_scale=serial(getattr(cfg, 'position_scale', None)),
                         orientation_scale=serial(getattr(cfg, 'orientation_scale', None)),
                         term_joint_ids=term_ids, observation_joint_ids=canonical_ids,
                         observation_joint_names=[env.scene['robot'].joint_names[i] for i in canonical_ids]
                         if hasattr(env.scene['robot'], 'joint_names') else None),
                gripper=gripper_report(env, runtime.torch, profile.get('gripper')))
