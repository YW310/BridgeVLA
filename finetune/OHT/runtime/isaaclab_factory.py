"""Built-in OHT scene launcher. Simulator imports occur only after AppLauncher."""
import argparse
import importlib
import json
from pathlib import Path
import random
import sys

import numpy as np

from .isaaclab_runtime import IsaacLabRuntime
from .isaaclab_scene import expected_asset, scene_configuration


def launch_application(extra):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description='IsaacLab application arguments')
    AppLauncher.add_app_launcher_args(parser)
    options = parser.parse_args(extra)
    options.enable_cameras = True
    launcher = AppLauncher(options)
    return launcher.app, options


def simulation_modules(extra_imports):
    # Importing these before the SimulationApp exists can break Isaac Sim startup.
    import gymnasium as gym
    import torch
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    for name in extra_imports:
        importlib.import_module(name)
    return gym, torch, parse_env_cfg


def camera_mapping(args, names):
    mapping = (json.loads(Path(args.camera_map).read_text(encoding='utf-8')) if args.camera_map else
               {name: name.upper() for name in names})
    if (not isinstance(mapping, dict) or set(mapping) != set(names) or
            any(not isinstance(value, str) or not value for value in mapping.values()) or
            len(set(mapping.values())) != len(mapping)):
        raise ValueError('camera-map must map all checkpoint cameras to distinct sensor names or USD paths')
    return mapping


def ee_link(env, requested):
    if requested:
        return requested
    name = getattr(getattr(env.action_manager.get_term('arm_action'), 'cfg', None), 'body_name', None)
    if not name:
        raise ValueError('Specify --ee-link for an environment without arm_action.cfg.body_name (including joint control)')
    return name


def check_controller(env, mode):
    term = env.action_manager.get_term('arm_action')
    if 'joint' in mode:
        if type(term).__name__ != 'JointPositionAction':
            raise ValueError('Joint checkpoint requires a JointPositionAction environment; select --env-id or --env-config')
    elif 'OperationalSpace' not in type(term).__name__:
        raise ValueError('EEF checkpoint requires an OperationalSpace action environment')


def build_cameras(env, app, args, model_info):
    names = model_info['cameras']
    mapping = camera_mapping(args, names)
    sensors = getattr(env.scene, 'sensors', {})
    bound = {}
    for name in names:
        key = mapping[name] if args.camera_map else (name if name in sensors else mapping[name])
        if key in sensors:
            bound[name] = sensors[key]
    if args.camera_source == 'sensors' or (args.camera_source == 'auto' and len(bound) == len(names)):
        if len(bound) != len(names):
            raise ValueError(f'Missing registered Camera sensors: {sorted(set(names) - set(bound))}')
        runtime = IsaacLabRuntime(env, bound, ee_link=ee_link(env, args.ee_link),
                                  quaternion_order=args.quaternion_order, simulation_app=app, owns_app=True,
                                  ee_pose_source=args.ee_pose_source)
        runtime.camera_source = 'sensors'
        return runtime
    from .usd_cameras import ReplicatorCameras, ReplicatorRuntime
    manager = ReplicatorCameras(env, mapping, args.camera_width, args.camera_height)
    try:
        runtime = ReplicatorRuntime(env, manager, ee_link=ee_link(env, args.ee_link),
                                    quaternion_order=args.quaternion_order, simulation_app=app, owns_app=True,
                                    ee_pose_source=args.ee_pose_source)
    except BaseException:
        manager.close()
        raise
    return runtime


def create_runtime(*, args, model_info):
    if min(args.camera_width, args.camera_height, args.camera_warmup_ticks) < 1:
        raise ValueError('Camera dimensions and warmup ticks must be positive')
    expected = expected_asset(args, model_info['action_spec']['mode'])
    app, env, runtime = None, None, None
    previous_argv = sys.argv
    try:
        # Keep BridgeVLA arguments out of simulator/task modules' argument parsers.
        sys.argv = [sys.argv[0]]
        app, options = launch_application(args.runtime_args)
        gym, torch, parse_env_cfg = simulation_modules(args.task_import)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cfg, scene_info = scene_configuration(args, parse_env_cfg, options.device, expected)
        env = gym.make(args.env_id, cfg=cfg).unwrapped
        if getattr(env, 'num_envs', 1) != 1:
            raise ValueError('BridgeVLA supports exactly one simulator environment')
        check_controller(env, model_info['action_spec']['mode'])
        env.reset()
        runtime = build_cameras(env, app, args, model_info)
        runtime.scene_info = scene_info
        for _ in range(args.camera_warmup_ticks):
            runtime._replicator_render_tick(env)
        # Fail on missing RGB/depth/calibration before starting an HTTP episode.
        for camera in runtime.cameras.values():
            runtime.camera_rgb(camera)
            runtime.camera_depth(camera)
            runtime.camera_intrinsic(camera)
            runtime.camera_pose_ros(camera, env, robot=env.scene['robot'])
        print(f'[BridgeVLA] native env={args.env_id} config={scene_info["config"]} '
              f'ee_link={runtime.ee_link} cameras={runtime.camera_source}', flush=True)
        return runtime
    except BaseException:
        if runtime is not None:
            runtime.close()
        else:
            try:
                if env is not None:
                    env.close()
            finally:
                if app is not None:
                    app.close()
        raise
    finally:
        sys.argv = previous_argv
