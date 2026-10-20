"""Task-specific reset/scoring outside the BridgeVLA policy input and action path."""
import importlib
import numpy as np

from ..data.common import GOALS, read_config
from .joint_executor import numpy_value


def factory(value):
    module, separator, name = value.partition(':')
    if not separator:
        raise ValueError('Expected module:factory')
    return getattr(importlib.import_module(module), name)


def task_settings(path, task):
    if task not in GOALS:
        raise ValueError('Unknown BridgeVLA task')
    settings = dict(reset_assets=[], initializer_factory=None, evaluator_factory=None, success_term=None)
    if path is None:
        return settings
    config = read_config(path)
    if set(config) != {'version', 'tasks'} or config['version'] != 1 or not isinstance(config['tasks'], dict):
        raise ValueError('Task config requires version: 1 and a tasks mapping')
    for name, definition in config['tasks'].items():
        if name not in GOALS or not isinstance(definition, dict) or set(definition) - set(settings):
            raise ValueError(f'Invalid task settings for {name}')
        assets = definition.get('reset_assets', [])
        if not isinstance(assets, list) or any(not isinstance(v, str) or not v for v in assets) or len(set(assets)) != len(assets):
            raise ValueError('reset_assets must list distinct registered scene assets')
        for key in ('initializer_factory', 'evaluator_factory', 'success_term'):
            value = definition.get(key)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f'{key} must be a nonempty string or null')
    if task not in config['tasks']:
        raise ValueError(f'Task config does not define selected task: {task}')
    return {**settings, **config['tasks'][task]}


class EnvironmentSuccess:
    def __init__(self, env, term):
        manager = getattr(env, 'termination_manager', None)
        if manager is None or term not in manager.active_terms:
            raise ValueError(f'Environment does not provide configured success term: {term}')
        if manager.get_term_cfg(term).time_out:
            raise ValueError('A timeout term cannot be used as task success')
        self.manager, self.term, self.success = manager, term, False

    def reset(self, episode_index):
        self.success = False

    def update(self, *, step, executed_gripper_cmd):
        value = numpy_value(self.manager.get_term(self.term))
        if value.shape != (1,) or value.dtype != np.bool_:
            raise ValueError('Success term must return one explicit boolean for the single environment')
        self.success |= bool(value[0])
        return self.success


class TaskRuntime:
    def __init__(self, env, runtime, task, settings, *, score=False):
        self.env, self.task, self.settings, self.success = env, task, settings, False
        self.initializer, self.evaluator = None, None
        for key, attribute in (('initializer_factory', 'initializer'), ('evaluator_factory', 'evaluator')):
            if settings[key] is not None and (attribute != 'evaluator' or score):
                value = factory(settings[key])(env=env, runtime=runtime, task=task)
                if not callable(getattr(value, 'reset', None)):
                    raise ValueError(f'{key} must return an object with reset()')
                setattr(self, attribute, value)
        if score and self.evaluator is None:
            if settings['success_term'] is None:
                raise ValueError('Task scoring needs an explicit success_term or evaluator_factory')
            self.evaluator = EnvironmentSuccess(env, settings['success_term'])
        if self.evaluator is not None and not callable(getattr(self.evaluator, 'update', None)):
            raise ValueError('Task evaluator must provide update()')
        self.assets = []
        for name in settings['reset_assets']:
            asset = env.scene[name]
            if not all(hasattr(asset, key) for key in ('write_root_pose_to_sim', 'write_root_velocity_to_sim')):
                raise ValueError(f'{name}: reset_assets requires a registered rigid object')
            state = getattr(asset.data.root_state_w, 'torch', asset.data.root_state_w).clone()
            if tuple(state.shape) != (1, 13):
                raise ValueError(f'{name}: expected rigid root state [1,13]')
            self.assets.append((asset, state))

    def initialize(self, episode_index, seed):
        for asset, state in self.assets:
            asset.write_root_pose_to_sim(state[:, :7].clone())
            asset.write_root_velocity_to_sim(state[:, 7:].clone().zero_())
        if self.assets:
            self.env.scene.write_data_to_sim()
            self.env.scene.update(dt=0.)
        if self.initializer is not None:
            self.initializer.reset(episode_index=episode_index, seed=seed)

    def reset(self, episode_index):
        self.success = False
        if self.evaluator is not None:
            self.evaluator.reset(episode_index=episode_index)

    def update(self, *, step, executed_gripper_cmd):
        value = self.evaluator.update(step=step, executed_gripper_cmd=executed_gripper_cmd)
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError('Task evaluator update() must return an explicit boolean')
        self.success |= bool(value)
        return self.success
