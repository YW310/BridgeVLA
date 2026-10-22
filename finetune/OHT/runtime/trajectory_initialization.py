"""Restore one recorded OHT state before a normal, live-observation policy run."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from ..data.common import GOALS
from .raw_observation import profile_task
from .trajectory_replay import TrajectoryReplay


class InitialState:
    """Reuse the GT replay's state restoration without replaying its actions.

    The policy retains its own client, action mode, cameras and HTTP session.
    Only recorded robot joints and explicitly mapped rigid-object poses are
    restored; a separate initial TCP check validates the resulting live state.
    """

    def __init__(self, args, client, task, profile):
        if task not in GOALS:
            raise ValueError('Unknown task for recorded policy initialization')
        root = getattr(args, 'initial_state_root', None)
        if not root:
            raise ValueError('Recorded policy initialization requires --initial-state-root with original OHT data')
        episode = getattr(args, 'initial_state_episode', 0)
        frame = getattr(args, 'initial_state_frame', 0)
        if type(episode) is not int or episode < 0 or type(frame) is not int or frame < 0:
            raise ValueError('initial-state-episode and initial-state-frame must be nonnegative integers')
        data_config = getattr(args, 'initial_state_data_config', None)
        contract = getattr(args, 'initial_state_contract', None)
        if data_config is not None and contract is not None:
            raise ValueError('Choose initial-state-data-config or initial-state-contract, not both')
        requested_names = getattr(args, 'initial_state_joint_names', None)
        model_names = getattr(client, 'action_config', {}).get('joint_names')
        if requested_names is not None and model_names is not None and list(requested_names) != list(model_names):
            raise ValueError('Initial-state joint names/order differ from the model action configuration')
        joint_names = requested_names if requested_names is not None else model_names
        self.task, self.profile = task, profile
        self._tolerances = self._tolerance_args(args)
        try:
            advertised = client.info['raw_observation']['data_profiles'][profile]
        except (KeyError, TypeError) as exc:
            raise ValueError('Selected model data profile is unavailable for recorded initialization') from exc
        if advertised.get('task') not in (None, task) or profile_task(profile) not in (None, task):
            raise ValueError('Recorded initialization task does not match the model data profile')
        # Initial state restoration does not require a following frame or the
        # source trajectory's control rate. Replay itself keeps those checks.
        source_args = SimpleNamespace(
            task=task, replay_root=root, replay_episode=episode, replay_start_frame=0,
            replay_data_config=data_config, replay_contract=contract,
            replay_joint_names=joint_names,
            replay_gripper_joint_names=getattr(args, 'initial_state_gripper_joint_names', None),
            replay_object_map=getattr(args, 'initial_state_object_map', None), replay_cameras=False)
        self._source = TrajectoryReplay(source_args)
        if self._source.record['task'] != task:
            raise ValueError('Recorded initialization episode belongs to another task')
        recorded_profile = self._source.profile_name
        if recorded_profile != profile and 'default' not in (recorded_profile, profile):
            raise ValueError('Recorded dataset profile does not match the selected model data profile')
        if frame >= len(self._source.states):
            raise ValueError('initial-state-frame is outside the selected recorded episode')
        if contract is not None and self._source.info['contract_sha256'] != client.info.get('contract_sha256'):
            raise ValueError('Initial-state contract SHA256 does not match the policy checkpoint contract')
        self._check_calibration(self._source.config, advertised)
        self._source.start = frame
        # No call to validate_runtime() or action(): the source supplies state,
        # not policy targets, success labels, or a trajectory execution schedule.
        self.metadata = deepcopy(self._source.metadata)
        self.metadata.pop('available_actions', None)
        self.metadata.pop('action_config', None)
        self.metadata.update(source='recorded_initial_state', task=task, data_profile=profile,
                             recorded_data_profile=recorded_profile, start_frame=frame, source_frame=frame,
                             joint_names=source_args.replay_joint_names,
                             future_actions_used=False, server_queries=False,
                             calibration_matches_model=True,
                             calibration_fields=['link_to_tcp', 'gripper.open', 'gripper.close',
                                                 'scene_bounds', 'image_size'],
                             position_tolerance=self._tolerances.replay_initial_position_tolerance,
                             rotation_tolerance=self._tolerances.replay_initial_rotation_tolerance)

    @staticmethod
    def _tolerance_args(args):
        position = getattr(args, 'initial_state_position_tolerance', .005)
        rotation = getattr(args, 'initial_state_rotation_tolerance', .05)
        values = np.asarray([position, rotation], np.float64)
        if not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError('Initial-state TCP tolerances must be finite and positive')
        return SimpleNamespace(replay_initial_position_tolerance=float(position),
                               replay_initial_rotation_tolerance=float(rotation))

    @staticmethod
    def _check_calibration(recorded, advertised):
        for name, shape in (('link_to_tcp', (4, 4)), ('scene_bounds', (6,)), ('image_size', (2,))):
            try:
                actual, expected = np.asarray(recorded[name], np.float64), np.asarray(advertised[name], np.float64)
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'Missing or invalid model/initial-state calibration: {name}') from exc
            if (actual.shape != shape or expected.shape != shape or not np.isfinite(actual).all()
                    or not np.isfinite(expected).all() or not np.allclose(actual, expected, atol=1e-6, rtol=0)):
                raise ValueError(f'Initial-state calibration differs from model data profile: {name}')
        for name in ('open', 'close'):
            try:
                actual = float(recorded['gripper'][name])
                expected = float(advertised['gripper'][name])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f'Missing or invalid model/initial-state calibration: gripper.{name}') from exc
            if not np.isfinite([actual, expected]).all() or not np.isclose(actual, expected, atol=1e-8, rtol=0):
                raise ValueError(f'Initial-state calibration differs from model data profile: gripper.{name}')

    def _check_bridge(self, bridge):
        if getattr(bridge, 'task', self.task) != self.task or getattr(bridge, 'profile', self.profile) != self.profile:
            raise ValueError('Recorded initialization must use its selected policy task and data profile')

    def initialize(self, bridge):
        self._check_bridge(bridge)
        return self._source.initialize(bridge)

    def check_initial(self, bridge, args=None):
        self._check_bridge(bridge)
        return self._source.check_initial(bridge, self._tolerances if args is None else self._tolerance_args(args))
