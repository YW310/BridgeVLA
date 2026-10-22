"""Explicit scene selection and asset provenance; never replace only the robot USD."""
import importlib
from pathlib import Path
import sys

from .task_runtime import factory

V4_MODULE = 'isaaclab_tasks.manager_based.manipulation.deploy.reach.config.ur_16e.all_scene_osc_env_cfg'
V4_CONFIG = V4_MODULE + ':UR16eDualEEAllSceneOSCReachEnvCfgV4'
V4_USD = Path('oht/assets_v4/ALL/Collected_all/all.usd')


def expected_asset(args, mode):
    expected = args.expected_robot_usd
    if args.scene_preset == 'oht-v4':
        if 'joint' in mode:
            raise ValueError('oht-v4 preset is an OSC scene; joint checkpoints need their own joint-position environment')
        if args.env_config not in (None, V4_CONFIG):
            raise ValueError('oht-v4 preset requires its V4 config; use scene-preset registry for a custom config')
        if not args.isaaclab_root:
            raise ValueError('oht-v4 requires --isaaclab-root or ISAACLAB_ROOT')
        required = (Path(args.isaaclab_root) / V4_USD).resolve()
        if expected is not None and Path(expected).resolve() != required:
            raise ValueError(f'oht-v4 requires the copied V4 asset {required}; expected-robot-usd disagrees')
        expected = str(required)
    if expected is not None and not Path(expected).is_file():
        raise FileNotFoundError(f'Required robot USD does not exist: {expected}; no fallback to another asset')
    return expected


def scene_configuration(args, parse_env_cfg, device, expected):
    config_name = V4_CONFIG if args.scene_preset == 'oht-v4' else args.env_config
    source = None
    if args.scene_preset == 'oht-v4':
        module = importlib.import_module(V4_MODULE)
        source = getattr(module, '__file__', None)
        if source is None or not Path(source).resolve().is_relative_to(Path(args.isaaclab_root).resolve()):
            raise ValueError(f'V4 task config imported from the wrong IsaacLab checkout: {source}')
        cls = getattr(module, V4_CONFIG.split(':')[1], None)
        if cls is None:
            raise ValueError(f'The IsaacLab task module does not provide {V4_CONFIG}')
        cfg = cls()
    elif config_name:
        cfg = factory(config_name)()
    else:
        cfg = parse_env_cfg(args.env_id, device=device, num_envs=1)
    cfg.scene.num_envs, cfg.sim.device = 1, device
    cfg.seed = args.seed
    cfg.env_name = args.env_id
    terminations = getattr(cfg, 'terminations', None)
    timeout_disabled = terminations is not None and hasattr(terminations, 'time_out')
    if timeout_disabled:
        # Match the reference rollout: the client owns the horizon. A short
        # training timeout otherwise resets the scene inside env.step().
        terminations.time_out = None
    if args.disable_fabric:
        cfg.sim.use_fabric = False
    robot_cfg = getattr(cfg.scene, 'robot', None)
    raw_usd = getattr(getattr(robot_cfg, 'spawn', None), 'usd_path', None)
    resolved = None
    if raw_usd:
        raw_usd = str(raw_usd)
        resolved = raw_usd if '://' in raw_usd else str(Path(raw_usd).expanduser().resolve())
    if expected is not None and resolved != str(Path(expected).resolve()):
        raise ValueError(f'Selected robot USD {resolved!r} differs from required asset {expected}; load the matching full environment config')
    if args.scene_preset == 'oht-v4':
        arm = cfg.actions.arm_action
        if arm.body_name != 'base_link_0':
            raise ValueError(f'V4 OSC body_name must be base_link_0, got {arm.body_name!r}')
        targets = arm.controller_cfg.target_types
        if ([targets] if isinstance(targets, str) else list(targets)) != ['pose_rel']:
            raise ValueError('V4 OSC controller must use only pose_rel targets')
    module = sys.modules.get(type(cfg).__module__)
    source = source or getattr(module, '__file__', None)
    info = dict(preset=args.scene_preset, config=config_name or 'registry',
                config_class=f'{type(cfg).__module__}:{type(cfg).__name__}',
                config_source=str(Path(source).resolve()) if source else None,
                robot_usd=raw_usd, resolved_robot_usd=resolved,
                expected_robot_usd=expected, asset_verified=expected is not None,
                version=str(getattr(cfg, 'version', 'unknown')), timeout_disabled=timeout_disabled)
    print(f'[BridgeVLA] scene preset={info["preset"]} config={info["config"]} source={info["config_source"]}', flush=True)
    print(f'[BridgeVLA] robot USD={resolved} expected={expected} verified={info["asset_verified"]}', flush=True)
    return cfg, info
