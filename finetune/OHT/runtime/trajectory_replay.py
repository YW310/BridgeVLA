"""Observed OHT TCP trajectories through the online EEF executor, without a model."""
from pathlib import Path
import json

import numpy as np
from scipy.spatial.transform import Rotation

from ..action_modes import action_options, action_descriptor, validate_action
from ..data.actions import gripper_states, world_tcp_poses
from ..data.common import digest, file_digest, inside, read_config
from ..data.geometry import object_pose_arrays, quaternion
from ..data.reader import discover, read_episode, validate_episode
from ..data.source_config import resolve_dataset_config
from .joint_executor import arm_ids, numpy_value


class TrajectoryReplay:
    """Each source frame i commands the observed absolute TCP at frame i+1.

    This is neither raw OSC command playback nor replay-cache future-keypoint
    sampling. No model or HTTP session is created. Finger states are measured
    joint positions; policy gripper targets use the training hysteresis rule.
    """

    def __init__(self, args):
        if not args.replay_root:
            raise ValueError('--mode replay requires --replay-root containing the original OHT Parquet data')
        self.root = Path(args.replay_root).resolve()
        if (self.root / 'samples.jsonl').exists():
            raise ValueError('--replay-root requires raw OHT data, not the sampled replay-source-v3 buffer')
        matches = [r for r in discover(self.root)
                   if r['task'] == args.task and r['episode_index'] == args.replay_episode]
        if len(matches) != 1:
            raise ValueError(f'Expected one {args.task} episode {args.replay_episode} under '
                             f'{self.root}/<task>/lerobot_dataset/data/chunk-*/episode_*.parquet')
        self.record = matches[0]
        self.profile_name = self.record['dataset']
        columns = read_episode(self.root, self.record)
        validate_episode(columns, self.record)
        contract_sha = None
        if args.replay_contract:
            from ..data.replay import load_contract
            contract = load_contract(args.replay_contract)
            profiles = contract.get('source_data_configs', {})
            if profiles:
                if self.profile_name not in profiles:
                    raise ValueError('Raw task/dataset does not match a replay contract data profile')
                config = profiles[self.profile_name]['data_config']
            else:
                self.profile_name = 'default'
                config = contract['data_config']
            contract_sha = contract['sha256']
            calibration = dict(replay_contract=str(Path(args.replay_contract).resolve()), sha256=contract_sha)
        else:
            path = Path(args.replay_data_config or Path(__file__).resolve().parents[1] / 'configs/dataset.yaml')
            config, sources = resolve_dataset_config(inside(self.root, self.record['dataset']), read_config(path))
            calibration = dict(data_config=str(path), sha256=file_digest(path), metadata_sha256=sources)
        self.config = config
        self.poses = world_tcp_poses(columns, config['link_to_tcp'], config['ee_quaternion_order'])
        self.states = np.asarray(columns['observation.state'], np.float64)
        self.fingers = np.asarray(columns['observation.gripper_joints'], np.float64)
        if not np.allclose(self.fingers.sum(axis=1), self.states[:, 6], atol=1e-4, rtol=0):
            raise ValueError('Recorded gripper joint sum differs from observation.state physical gripper')
        _, grip = gripper_states(self.states, config['gripper'])
        self.targets = np.asarray([validate_action(np.r_[pose, g], {'mode': 'eef'})
                                   for pose, g in zip(self.poses, grip)])
        self.timestamps = np.asarray(columns['timestamp'], np.float64)
        self.positions, self.orientations = object_pose_arrays(columns['observation.objects_pos'],
                                                               columns['observation.objects_quat'])
        self.start = args.replay_start_frame
        if not 0 <= self.start < len(self.targets) - 1:
            raise ValueError('replay-start-frame must have at least one following recorded frame')
        self.length = len(self.targets) - 1 - self.start
        self.action_config = action_options({'mode': 'eef', 'joint_names': args.replay_joint_names})
        self.gripper_joint_names = args.replay_gripper_joint_names
        self.object_map = self._objects(args.replay_object_map)
        self.info = dict(model_id=None, contract_sha256=contract_sha,
                         action_spec=action_descriptor(self.action_config),
                         cameras=list(config['cameras']) if args.replay_cameras else [],
                         raw_observation={'data_profiles': {self.profile_name: dict(
                             task=args.task, link_to_tcp=config['link_to_tcp'], gripper=config['gripper'],
                             scene_bounds=config['scene_bounds'], image_size=config['image_size'])}})
        self.metadata = dict(source='observed_next_frame_tcp', root=str(self.root),
                             parquet=self.record['path'], parquet_sha256=file_digest(inside(self.root, self.record['path'])),
                             episode_index=args.replay_episode, start_frame=self.start,
                             available_actions=self.length, calibration=calibration,
                             action_config=self.action_config, data_config_sha256=digest(config),
                             object_map=self.object_map, gripper_joint_names=self.gripper_joint_names,
                             server_queries=False)

    def _objects(self, path):
        if path is None:
            return None
        mapping = json.loads(Path(path).read_text(encoding='utf-8'))
        if (not isinstance(mapping, dict) or set(mapping) != {'quaternion_order', 'assets'} or
                mapping['quaternion_order'] not in ('wxyz', 'xyzw') or
                not isinstance(mapping['assets'], dict) or not mapping['assets']):
            raise ValueError('replay-object-map requires quaternion_order (wxyz/xyzw) and assets {scene_key: index}')
        indices = list(mapping['assets'].values())
        if (any(not isinstance(k, str) or not k for k in mapping['assets']) or
                any(type(i) is not int or not 0 <= i < self.positions.shape[1] for i in indices) or
                len(set(indices)) != len(indices)):
            raise ValueError('replay-object-map must use distinct valid object indices and registered scene keys')
        return mapping

    def profile(self, task, name=None):
        if task != self.record['task'] or name not in (None, self.profile_name):
            raise ValueError('Replay task/data profile does not match the selected raw episode')
        return self.profile_name

    def validate_runtime(self, env, args):
        if args.max_substeps != 1:
            raise ValueError('Full-rate trajectory replay requires --max-substeps 1; do not repeat source actions')
        dt = float(env.step_dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Simulator step_dt must be finite and positive')
        count = min(args.max_steps, self.length)
        intervals = np.diff(self.timestamps)[self.start:self.start + count]
        if not np.allclose(intervals, dt, atol=1e-5, rtol=1e-4):
            raise ValueError(f'Recorded frame interval {intervals.min():.8g}..{intervals.max():.8g}s '
                             f'differs from simulator step_dt={dt:.8g}s; use the matching collection control rate')

    def action(self, step):
        frame = self.start + step
        if not 0 <= step < self.length:
            raise IndexError('Trajectory exhausted')
        return self.targets[frame + 1].copy(), dict(source_frame=frame, target_frame=frame + 1,
                    source_timestamp=float(self.timestamps[frame]), target_timestamp=float(self.timestamps[frame + 1]))

    def initialize(self, bridge):
        """Restore recorded joints once. No physics step or repeated teleport."""
        env, robot, torch = bridge.env, bridge.robot, bridge.ref.torch
        _, ids = arm_ids(env, self.action_config)
        gripper = env.action_manager.get_term('gripper_action')
        fingers = getattr(gripper, 'joint_ids', None)
        if fingers is None:
            fingers = getattr(gripper, '_joint_ids', None)
        if isinstance(fingers, slice):
            fingers = list(range(len(robot.joint_names)))[fingers]
        if fingers is None:
            raise ValueError('Replay initialization requires gripper joint IDs')
        fingers = numpy_value(fingers).astype(int).reshape(-1).tolist()
        if len(fingers) != 2 or len(set(fingers)) != 2 or set(ids) & set(fingers):
            raise ValueError('Replay initialization requires six arm and two distinct gripper joints')
        if self.gripper_joint_names is not None:
            canonical = [list(robot.joint_names).index(name) for name in self.gripper_joint_names]
            if len(set(canonical)) != 2 or set(canonical) != set(fingers):
                raise ValueError('Replay gripper joint names must match the environment gripper joints')
            fingers = canonical
        if not hasattr(robot.data, 'joint_vel') or not callable(getattr(robot, 'write_joint_state_to_sim', None)):
            raise ValueError('Replay initialization requires articulation joint-state writing')
        q = getattr(robot.data.joint_pos, 'torch', robot.data.joint_pos).clone()
        qd = getattr(robot.data.joint_vel, 'torch', robot.data.joint_vel).clone()
        q[0, ids] = torch.as_tensor(self.states[self.start, :6], device=env.device, dtype=q.dtype)
        q[0, fingers] = torch.as_tensor(self.fingers[self.start], device=env.device, dtype=q.dtype)
        qd[0, ids + fingers] = 0
        # Soft limits are policy safety margins, not the physical articulation
        # limits. Recorded collection poses may legitimately lie outside them.
        limits, limit_source = None, None
        for name in ('joint_pos_limits', 'joint_limits', 'default_joint_pos_limits', 'default_joint_limits'):
            value = getattr(robot.data, name, None)
            if value is not None:
                limits, limit_source = value, name
                break
        if limits is not None:
            bounds = numpy_value(limits)[0, ids + fingers]
            values = numpy_value(q)[0, ids + fingers]
            if np.isnan(bounds).any() or np.any(bounds[:, 0] > bounds[:, 1]):
                raise ValueError('Simulator hard joint limits are invalid')
            if np.any(values < bounds[:, 0] - 1e-4) or np.any(values > bounds[:, 1] + 1e-4):
                raise ValueError('Recorded initial joints are outside simulator hard joint limits')
        # Resolve every object before modifying any simulator state.
        objects = []
        if self.object_map:
            origin = numpy_value(env.scene.env_origins)[0]
            for key, index in self.object_map['assets'].items():
                obj = env.scene[key]
                if obj is robot or callable(getattr(obj, 'write_joint_state_to_sim', None)):
                    raise ValueError(f'{key}: replay object mapping requires a rigid object, not a robot/articulation')
                if any(obj is previous for previous, _ in objects):
                    raise ValueError(f'{key}: replay mapping selects the same scene object more than once')
                if not all(callable(getattr(obj, name, None)) for name in
                           ('write_root_pose_to_sim', 'write_root_velocity_to_sim')):
                    raise ValueError(f'{key}: replay object mapping requires a registered rigid object')
                quat = quaternion(self.orientations[self.start, index])
                if self.object_map['quaternion_order'] == 'xyzw':
                    quat = quat[[3, 0, 1, 2]]
                if getattr(bridge.ref, 'quaternion_order', 'wxyz') == 'xyzw':
                    quat = quat[[1, 2, 3, 0]]
                pose = np.r_[self.positions[self.start, index] + origin, quat]
                objects.append((obj, pose))
        robot.write_joint_state_to_sim(q, qd)
        if callable(getattr(robot, 'set_joint_position_target', None)):
            robot.set_joint_position_target(q)
        if callable(getattr(robot, 'set_joint_velocity_target', None)):
            robot.set_joint_velocity_target(qd)
        for obj, pose in objects:
            obj.write_root_pose_to_sim(torch.as_tensor(pose, device=env.device, dtype=q.dtype)[None])
            obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device, dtype=q.dtype))
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(dt=0.)
        actual = numpy_value(robot.data.joint_pos)[0]
        if not np.allclose(actual[ids + fingers], numpy_value(q)[0, ids + fingers], atol=1e-5, rtol=0):
            raise ValueError('Simulator did not restore the recorded initial joint state')
        return dict(source_frame=self.start, arm_joint_ids=ids, gripper_joint_ids=fingers, joint_limit_source=limit_source,
                    objects_restored=list(self.object_map['assets']) if self.object_map else [],
                    object_state='mapped_recorded_pose' if objects else 'environment_reset', velocities='zero')

    def check_initial(self, bridge, args):
        state = bridge.measured_state()
        pose, target = np.asarray(state['tcp_pose_xyzw']), self.poses[self.start]
        errors = dict(translation_m=float(np.linalg.norm(pose[:3] - target[:3])),
                      rotation_rad=float((Rotation.from_quat(target[3:]) * Rotation.from_quat(pose[3:]).inv()).magnitude()))
        if (errors['translation_m'] > args.replay_initial_position_tolerance or
                errors['rotation_rad'] > args.replay_initial_rotation_tolerance):
            raise ValueError(f'Replay initial TCP mismatch {errors}; check asset, base, joint order and ee-pose-source')
        return dict(measured=state, recorded_tcp_xyzw=target.tolist(), errors=errors)
