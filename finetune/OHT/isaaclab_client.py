"""Run a BridgeVLA policy or replay an observed OHT EEF trajectory in IsaacLab."""
import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback
from time import perf_counter
import numpy as np
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict

from .data.common import GOALS, write_json
from .runtime.http_client import BridgeVLAClient
from .runtime.isaaclab_bridge import IsaacLabBridge
from .runtime.joint_executor import JointPositionExecutor
from .runtime.task_runtime import TaskRuntime, task_settings
from .runtime.control_diagnostics import control_report, joint_report
from .runtime.policy_worker import PolicyWorker, refresh_while_waiting, keep_open_after_error
from .runtime.live_markers import LiveMarkers
from .runtime.waypoint_executor import WaypointExecutor, WaypointOptions


def load_reference(path, argv):
    """Import trusted runtime helpers without executing its __main__/JEPA loop."""
    path = Path(path).expanduser().resolve(strict=True)
    spec = importlib.util.spec_from_file_location('bridgevla_oht_reference', path)
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    original = sys.argv
    try:
        sys.argv = [str(path), *argv]
        spec.loader.exec_module(reference)
        if reference.simulation_app is None:
            raise ValueError('Reference must run with IsaacLab enabled, not offline/HDF mode')
        return reference
    except BaseException:
        app = getattr(reference, 'simulation_app', None)
        if app is not None:
            app.close()
        raise
    finally:
        sys.argv = original


def _factory(value):
    module, separator, name = value.partition(':')
    if not separator:
        raise ValueError('Expected module:factory')
    return getattr(importlib.import_module(module), name)


@contextmanager
def isaaclab_directory(value=None):
    """Use the selected simulator checkout for imports, relative assets and helpers."""
    previous_dir = Path.cwd()
    previous_root = os.environ.get('ISAACLAB_ROOT')
    selected = value or previous_root
    if selected is None and (previous_dir / 'isaaclab.sh').is_file():
        selected = previous_dir
    if selected is None:
        yield None
        return
    root = Path(selected).expanduser().resolve()
    if not root.is_dir() or not (root / 'isaaclab.sh').is_file():
        raise ValueError(f'ISAACLAB_ROOT must be an IsaacLab directory containing isaaclab.sh: {root}')
    previous_path = list(sys.path)
    try:
        os.environ['ISAACLAB_ROOT'] = str(root)
        os.chdir(root)
        sys.path.insert(0, str(root))
        print(f'[BridgeVLA] ISAACLAB_ROOT={root}', flush=True)
        yield str(root)
    finally:
        os.chdir(previous_dir)
        sys.path[:] = previous_path
        if previous_root is None:
            os.environ.pop('ISAACLAB_ROOT', None)
        else:
            os.environ['ISAACLAB_ROOT'] = previous_root


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__,
                                     epilog='Unrecognized arguments (e.g. --headless --device cuda:0) are passed in args.runtime_args to the runtime factory, or to the legacy runtime.')
    source = parser.add_mutually_exclusive_group()
    source.add_argument('--runtime-factory', help='module:factory(args=...,model_info=...) -> IsaacLabRuntime')
    source.add_argument('--reference-client', help='Optional compatibility source for legacy IsaacLab sensor helpers')
    parser.add_argument('--isaaclab-root', help='IsaacLab checkout containing isaaclab.sh; overrides ISAACLAB_ROOT and sets the runtime working directory')
    parser.add_argument('--server', default='http://127.0.0.1:8010')
    parser.add_argument('--mode', choices=('policy', 'replay'), default='policy')
    parser.add_argument('--replay-root', help='Original OHT root containing <task>/lerobot_dataset/data, not replay-source-v3')
    parser.add_argument('--replay-episode', type=int, default=0, help='Raw dataset episode ID; default: 0')
    parser.add_argument('--replay-start-frame', type=int, default=0)
    parser.add_argument('--replay-targets', choices=('frame', 'keypoints'), default='frame',
                        help='frame preserves dense GT timing; keypoints tests the trained future-waypoint contract')
    calibration = parser.add_mutually_exclusive_group()
    calibration.add_argument('--replay-data-config', help='OHT dataset YAML; default: repository configs/dataset.yaml')
    calibration.add_argument('--replay-contract', help='Optional existing replay buffer supplying the resolved training calibration')
    parser.add_argument('--replay-joint-names', nargs=6, help='Robot joint names in recorded observation.state[:6] order')
    parser.add_argument('--replay-gripper-joint-names', nargs=2, help='Gripper joint names in recorded observation.gripper_joints order')
    parser.add_argument('--replay-object-map', help='JSON with explicit quaternion_order and assets {scene_key: recorded_object_index}')
    parser.add_argument('--replay-cameras', action='store_true', help='Also bind live dataset cameras during GT replay; default: robot state only')
    parser.add_argument('--replay-initial-position-tolerance', type=float, default=.005)
    parser.add_argument('--replay-initial-rotation-tolerance', type=float, default=.05)
    parser.add_argument('--initial-state-root', help='Raw OHT dataset used only to restore policy initialization, not policy actions')
    parser.add_argument('--initial-state-episode', type=int, default=0)
    parser.add_argument('--initial-state-frame', type=int, default=0)
    initial_calibration = parser.add_mutually_exclusive_group()
    initial_calibration.add_argument('--initial-state-data-config')
    initial_calibration.add_argument('--initial-state-contract', help='Training replay contract used to verify initialization calibration')
    parser.add_argument('--initial-state-object-map')
    parser.add_argument('--initial-state-joint-names', nargs=6)
    parser.add_argument('--initial-state-gripper-joint-names', nargs=2)
    parser.add_argument('--initial-state-position-tolerance', type=float, default=.005)
    parser.add_argument('--initial-state-rotation-tolerance', type=float, default=.05)
    parser.add_argument('--task', choices=tuple(GOALS), default='disassemble_left')
    parser.add_argument('--env-id', default='Isaac-Deploy-Reach-UR16e-DualEE-AllScene-OSC-v2')
    parser.add_argument('--env-config', help='Optional module:ConfigClass to instantiate instead of the registered environment config')
    parser.add_argument('--scene-preset', choices=('registry', 'oht-v4'), default='registry',
                        help='oht-v4 loads and verifies the full reference V4 scene instead of the registry config')
    parser.add_argument('--expected-robot-usd', help='Require the selected full environment config to reference this local USD; does not override the asset')
    parser.add_argument('--task-import', action='append', default=[], help='Additional environment registration module (repeatable)')
    parser.add_argument('--ee-link', help='Robot EE link used by training link_to_tcp; defaults to the OSC body_name')
    parser.add_argument('--ee-pose-source', choices=('body', 'controller'), default='controller',
                        help='Native raw EE measurement: controller applies body_offset (OHT recorded TCP); body reads the bare link')
    parser.add_argument('--dry-run', action='store_true', help='Save control diagnostics and exit without policy predictions or env.step')
    parser.add_argument('--keep-open-on-error', action='store_true',
                        help='Keep an initialized GUI open after a recorded startup/rollout error, without more physics or policy requests')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Print first/every N action; 0 disables periodic logs. Failed waypoints always print diagnostics')
    display = parser.add_mutually_exclusive_group()
    display.add_argument('--visualize', dest='visualize', action='store_true', default=True,
                         help='Show measured TCP, resolved EEF target and trajectories in the GUI (default)')
    display.add_argument('--no-visualize', dest='visualize', action='store_false', help='Disable simulator TCP overlays')
    parser.add_argument('--visualize-history', type=int, default=200, help='Maximum number of trajectory points shown (default: 200)')
    parser.add_argument('--quaternion-order', choices=('wxyz', 'xyzw'), default='wxyz', help='Installed IsaacLab tensor/config quaternion order (default: IsaacLab 2.x wxyz)')
    parser.add_argument('--camera-source', choices=('auto', 'sensors', 'usd'), default='auto')
    parser.add_argument('--camera-width', type=int, default=640)
    parser.add_argument('--camera-height', type=int, default=480)
    parser.add_argument('--camera-warmup-ticks', type=int, default=8, help='Render-only warmup; does not advance physics')
    parser.add_argument('--disable-fabric', '--disable_fabric', action='store_true')
    parser.add_argument('--data-profile')
    parser.add_argument('--camera-map', help='JSON mapping checkpoint camera names to USD camera names')
    parser.add_argument('--episodes', type=int, help='Default: 20 policy episodes, or one replay of the selected trajectory')
    parser.add_argument('--max-steps', type=int, help='Default: 500 policy requests, or all remaining replay transitions')
    parser.add_argument('--max-substeps', type=int, default=1, help='Streaming control steps per target; waypoint uses waypoint-max-steps')
    parser.add_argument('--execution-mode', choices=('auto', 'waypoint', 'streaming', 'chunk'), default='auto',
                        help='auto: waypoint for policy/GT keypoints, streaming for dense GT replay')
    parser.add_argument('--chunk-execute-horizon', type=int, help='Override advertised H; requests still return K actions')
    parser.add_argument('--chunk-max-tracking-translation', type=float, default=.10)
    parser.add_argument('--chunk-max-tracking-rotation', type=float, default=.5)
    parser.add_argument('--waypoint-max-steps', type=int, default=600)
    parser.add_argument('--waypoint-position-tolerance', type=float, default=.005)
    parser.add_argument('--waypoint-rotation-tolerance', type=float, default=.05)
    parser.add_argument('--waypoint-arrival-steps', type=int, default=3)
    parser.add_argument('--waypoint-stall-steps', type=int, default=100)
    parser.add_argument('--gripper-min-steps', type=int, default=10)
    parser.add_argument('--gripper-stable-steps', type=int, default=5)
    parser.add_argument('--max-translation', type=float, default=.05)
    parser.add_argument('--max-rotation', type=float, default=.15)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--initialization', choices=('env-reset', 'task', 'collection', 'trajectory'),
                        help='Default: env-reset for policy; trajectory for replay or policy with initial-state-root')
    parser.add_argument('--score', choices=('none', 'task', 'disassemble-left'), default='none')
    parser.add_argument('--task-config', help='BridgeVLA YAML with per-task reset assets and initializer/evaluator factories')
    parser.add_argument('--success-term', help='Explicit environment success termination term for --score task')
    parser.add_argument('--joint-hold-steps', type=int, default=1)
    parser.add_argument('--joint-target-tolerance', type=float, default=.01)
    parser.add_argument('--joint-max-step', type=float, help='Optional per-control-step joint displacement bound [rad]')
    parser.add_argument('--joint-executor', help='module:factory(env,reference) -> callable(absolute_joint_target,after_step=...)')
    parser.add_argument('--output', required=True, help='New output directory for summary and action traces')
    args, reference_args = parser.parse_known_args(argv)
    replay_options = [option.split('=', 1)[0] for option in argv if option.startswith('--replay-')]
    if args.mode != 'replay' and replay_options:
        raise ValueError(f'{", ".join(replay_options)} require --mode replay')
    initial_options = [option.split('=', 1)[0] for option in argv if option.startswith('--initial-state-')]
    if initial_options and (args.mode != 'policy' or not args.initial_state_root):
        raise ValueError('initial-state options require --mode policy and --initial-state-root')
    args.execution_auto = args.execution_mode == 'auto'
    if args.execution_mode == 'auto':
        args.execution_mode = 'streaming' if args.mode == 'replay' and args.replay_targets == 'frame' else 'waypoint'
    if args.mode == 'replay' and args.execution_mode != ('waypoint' if args.replay_targets == 'keypoints' else 'streaming'):
        raise ValueError('Dense frame replay requires streaming execution; GT keypoint replay requires waypoint execution')
    args.runtime_args = reference_args
    if args.episodes is None:
        args.episodes = 1 if args.mode == 'replay' else 20
    if args.initialization is None:
        args.initialization = 'trajectory' if args.mode == 'replay' or args.initial_state_root else 'env-reset'
    if args.initial_state_root and args.initialization != 'trajectory':
        raise ValueError('initial-state-root requires --initialization trajectory')
    if args.max_steps is None and args.mode == 'policy':
        args.max_steps = 500
    # User-owned paths stay relative to the launch directory, even when the runtime changes directory.
    for name in ('reference_client', 'task_config', 'camera_map', 'output', 'expected_robot_usd',
                 'replay_root', 'replay_data_config', 'replay_contract', 'replay_object_map',
                 'initial_state_root', 'initial_state_data_config', 'initial_state_contract', 'initial_state_object_map'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, str(Path(value).expanduser().resolve()))
    with isaaclab_directory(args.isaaclab_root) as root:
        args.isaaclab_root = root
        return _run(args, reference_args)


def _run(args, reference_args):
    replay, initial_state = None, None
    if args.mode == 'replay':
        if args.reference_client:
            raise ValueError('Trajectory replay uses the native BridgeVLA executor, not --reference-client')
        if args.replay_object_map and args.initialization != 'trajectory':
            raise ValueError('replay-object-map requires --initialization trajectory')
        if not np.isfinite([args.replay_initial_position_tolerance, args.replay_initial_rotation_tolerance]).all() or min(
                args.replay_initial_position_tolerance, args.replay_initial_rotation_tolerance) <= 0:
            raise ValueError('Replay initial TCP tolerances must be finite and positive')
    elif args.initialization == 'trajectory' and not args.initial_state_root:
        raise ValueError('Policy trajectory initialization requires --initial-state-root')
    if args.initial_state_root and args.reference_client:
        raise ValueError('Recorded initial states require the native/runtime-factory executor, not --reference-client')
    if (args.scene_preset != 'registry' or args.expected_robot_usd is not None) and (args.reference_client or args.runtime_factory):
        raise ValueError('scene-preset/expected-robot-usd require the built-in native runtime')
    settings = task_settings(args.task_config, args.task)
    if args.success_term is not None:
        if args.score != 'task':
            raise ValueError('--success-term requires --score task')
        settings['success_term'] = args.success_term
    if args.initialization == 'task' and not (settings['reset_assets'] or settings['initializer_factory']):
        raise ValueError('Task initialization requires reset_assets or initializer_factory in --task-config')
    if min(args.episodes, args.max_substeps) < 1 or (args.max_steps is not None and args.max_steps < 1):
        raise ValueError('episodes, max-steps and max-substeps must be positive')
    if args.log_interval < 0:
        raise ValueError('log-interval must be nonnegative')
    if args.visualize_history < 1:
        raise ValueError('visualize-history must be positive')
    waypoint_options = dict(max_steps=args.waypoint_max_steps,
                            position_tolerance=args.waypoint_position_tolerance,
                            rotation_tolerance=args.waypoint_rotation_tolerance,
                            joint_tolerance=args.joint_target_tolerance,
                            arrival_steps=args.waypoint_arrival_steps, stall_steps=args.waypoint_stall_steps,
                            gripper_min_steps=args.gripper_min_steps, gripper_stable_steps=args.gripper_stable_steps)
    if args.execution_mode == 'waypoint':
        WaypointOptions(**waypoint_options)
    if not np.isfinite([args.timeout, args.max_translation, args.max_rotation]).all() or min(
            args.timeout, args.max_translation, args.max_rotation) <= 0:
        raise ValueError('timeout and physical clip limits must be finite and positive')
    if args.score == 'disassemble-left' and args.task != 'disassemble_left':
        raise ValueError('The reference success tracker supports only disassemble_left')
    if args.score == 'disassemble-left' and not args.reference_client:
        raise ValueError('Legacy disassemble-left scorer needs --reference-client; native runtimes use --score task')
    if not args.reference_client and args.initialization == 'collection':
        raise ValueError('Legacy collection initialization needs --reference-client; native clients use --initialization task')
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    if args.mode == 'replay':
        from .runtime.trajectory_replay import TrajectoryReplay
        if args.replay_targets == 'keypoints':
            from .runtime.keypoint_replay import KeypointReplay
            replay = KeypointReplay(args)
        else:
            replay = TrajectoryReplay(args)
        client = replay
        if args.max_steps is None:
            args.max_steps = replay.length
        print(f'[BridgeVLA] loaded replay={replay.record["path"]} start_frame={replay.start} '
              f'actions={min(args.max_steps, replay.length)} HTTP=disabled', flush=True)
    else:
        client = BridgeVLAClient(args.server, args.timeout)
        if getattr(client, 'chunk_spec', None) is not None:
            if not args.execution_auto and args.execution_mode != 'chunk':
                raise ValueError('Chunk checkpoint requires --execution-mode auto or chunk')
            args.execution_mode = 'chunk'
        elif args.execution_mode == 'chunk' or args.chunk_execute_horizon is not None:
            raise ValueError('Chunk execution options require a chunk checkpoint')
    profile = client.profile(args.task, args.data_profile)
    if args.initial_state_root:
        from .runtime.trajectory_initialization import InitialState
        initial_state = InitialState(args, client, args.task, profile)
    names = client.info['cameras']
    mapping = (json.loads(Path(args.camera_map).read_text(encoding='utf-8')) if args.camera_map
               else {name: name.upper() for name in names})
    if (not isinstance(mapping, dict) or set(mapping) != set(names) or
            any(not isinstance(name, str) or not name for name in mapping.values()) or
            len(set(mapping.values())) != len(mapping)):
        raise ValueError('camera-map must map every checkpoint camera to a distinct USD camera')
    runtime_factory = args.runtime_factory or (None if args.reference_client else
                                              'finetune.OHT.runtime.isaaclab_factory:create_runtime')
    reference, env, records, worker, markers = None, None, [], None, None
    output.mkdir(parents=True)
    startup_state = dict(status='starting', phase='runtime', data_profile=profile,
                         execution_mode=args.execution_mode, chunk_spec=getattr(client, 'chunk_spec', None))
    def report_startup(phase):
        startup_state['phase'] = phase
        write_json(output / 'startup.json', startup_state)
        print(f'[BridgeVLA] startup: {phase}', flush=True)
    try:
        report_startup('runtime')
        reference = (_factory(runtime_factory)(args=args, model_info=deepcopy(client.info)) if runtime_factory else
                     load_reference(args.reference_client,
                                    [*reference_args, '--task', args.env_id, '--num_envs', '1', '--seed', str(args.seed)]))
        report_startup('environment_reset')
        env = getattr(reference, 'env', None)
        if env is None:
            cfg = reference.parse_env_cfg(args.env_id, device=reference.args_cli.device, num_envs=1)
            cfg.env_name = args.env_id
            if hasattr(cfg, 'terminations') and hasattr(cfg.terminations, 'time_out'):
                cfg.terminations.time_out = None
            env = reference.gym.make(args.env_id, cfg=cfg).unwrapped
        startup_state['env_step_dt'] = float(env.step_dt)
        if args.execution_mode == 'chunk':
            print(f'[BridgeVLA] chunk timing: action_dt={client.chunk_spec["action_dt"]} '
                  f'env_step_dt={startup_state["env_step_dt"]} '
                  f'K={client.chunk_spec["horizon"]} H={client.chunk_spec["execute_horizon"]}', flush=True)
        robot = env.scene['robot']
        env.reset()
        if args.initialization == 'collection' and not callable(getattr(reference, 'randomize_robot_initial_pose_like_collection', None)):
            raise ValueError('Collection initialization needs legacy helpers; native runtimes use --initialization task')
        if args.reference_client:
            reference.args_cli.oht_collection_task = args.task
        report_startup('task_runtime')
        task_runtime = TaskRuntime(env, reference, args.task, settings, score=args.score == 'task')
        startup, tracker = None, None
        if args.score == 'disassemble-left':
            startup = reference.capture_startup_rigid_object_state(env, reference.args_cli.oht_wheel_prim_path)
            tracker = reference.DisassembleLeftSemanticSuccessTracker(env, robot, startup)
        elif args.score == 'task':
            tracker = task_runtime
        report_startup('camera_binding')
        # Extend the reference's three-camera discovery to the exact checkpoint camera set.
        bound_cameras = getattr(reference, 'cameras', None)
        if bound_cameras is None:
            reference.ONLINE_CAMERA_NAMES = tuple(mapping.values())
            cameras = reference.OnlineReplicatorCameraManager(env, width=640, height=480)
            bound_cameras = {name: cameras[usd] for name, usd in mapping.items()}
            cameras.warmup(env, num_steps=4)
        elif set(bound_cameras) != set(names):
            raise ValueError('Runtime cameras must match all checkpoint camera names exactly')
        # The reference dispatches live wrist pose lookup by this logical name.
        # Preserve that behavior even when --camera-map changes the USD prim name.
        if args.reference_client and 'wrist' in bound_cameras and hasattr(bound_cameras['wrist'], 'name'):
            bound_cameras['wrist'].name = 'WRIST'
        joint = _factory(args.joint_executor)(env, reference) if args.joint_executor else None
        if joint is None and 'joint' in client.action_config['mode']:
            joint = JointPositionExecutor(env, reference.torch, client.action_config,
                                          hold_steps=args.joint_hold_steps, tolerance=args.joint_target_tolerance,
                                          max_step=args.joint_max_step)
        report_startup('live_markers')
        markers = LiveMarkers(reference, env, enabled=args.visualize and not args.dry_run,
                              history=args.visualize_history)
        report_startup('control_bridge')
        bridge = IsaacLabBridge(reference, env, client, args.task, profile,
                                bound_cameras,
                                max_substeps=args.max_substeps, max_translation=args.max_translation,
                                max_rotation=args.max_rotation, joint_executor=joint, visualizer=markers)
        report_startup('executor')
        waypoint = WaypointExecutor(bridge, **waypoint_options) if args.execution_mode == 'waypoint' else None
        chunk_executor = None
        if args.execution_mode == 'chunk':
            from .runtime.chunk_executor import ChunkExecutor
            chunk_executor = ChunkExecutor(bridge, client.chunk_spec, args.chunk_execute_horizon,
                                            args.chunk_max_tracking_translation, args.chunk_max_tracking_rotation)
        if waypoint is not None and joint is not None and not callable(getattr(joint, 'step', None)):
            raise ValueError('Waypoint joint execution requires a single-step executor exposing step(target); '
                             'use the built-in JointPositionExecutor or select streaming explicitly')
        if replay is not None:
            replay.validate_runtime(env, args)
        report_startup('control_diagnostics')
        diagnostics = control_report(bridge)
        write_json(output / 'control_diagnostics.json', diagnostics)
        write_json(output / 'run.json', dict(task=args.task, instruction=GOALS[args.task], seed=args.seed,
                   mode=args.mode, replay=replay.metadata if replay is not None else None,
                   initial_state=initial_state.metadata if initial_state is not None else None,
                   execution_mode=args.execution_mode, waypoint_options=asdict(waypoint.options) if waypoint else None,
                   motion_path='direct_target_tracking_no_collision_planner',
                   model_id=client.info['model_id'], contract_sha256=client.info['contract_sha256'],
                   action_spec=client.info['action_spec'], data_profile=profile, task_settings=settings,
                   chunk_spec=getattr(client, 'chunk_spec', None),
                   chunk_execute_horizon=chunk_executor.horizon if chunk_executor else None,
                   initialization=args.initialization, score=args.score, runtime_factory=runtime_factory,
                   policy_wait='background_http_render_only' if replay is None else None,
                   keep_open_on_error=args.keep_open_on_error,
                   visualization=markers.info,
                   isaaclab_root=args.isaaclab_root,
                   env_id=args.env_id, env_config=args.env_config, ee_link=getattr(reference, 'ee_link', args.ee_link),
                   quaternion_order=args.quaternion_order, camera_source=getattr(reference, 'camera_source', args.camera_source),
                   ee_pose_source=diagnostics['ee_pose_source'],
                   scene=diagnostics['scene'],
                   reference_client=args.reference_client, joint_executor=args.joint_executor or
                   ('JointPositionExecutor' if joint is not None else None),
                   joint_hold_steps=args.joint_hold_steps, joint_target_tolerance=args.joint_target_tolerance,
                   joint_max_step=args.joint_max_step))
        print(f'[BridgeVLA] control diagnostics: {output / "control_diagnostics.json"} '
              f'ee_pose_source={diagnostics["ee_pose_source"]} gripper_verified={diagnostics["gripper"]["verified"]}', flush=True)
        grip = diagnostics['gripper']
        print(f'[BridgeVLA] gripper +1=open targets={grip.get("open_joint_targets")} '
              f'-1=close targets={grip.get("close_joint_targets")} '
              f'training_physical_endpoints={grip["training_physical_endpoints"]}', flush=True)
        if diagnostics['gripper']['verified'] is False:
            raise ValueError('Gripper action transform does not map +1 to configured open and -1 to close; inspect control_diagnostics.json')
        if diagnostics['gripper']['calibration_direction_matches'] is False:
            raise ValueError('Simulator gripper open/close physical direction disagrees with the training profile; check scene/asset and checkpoint calibration in control_diagnostics.json')
        startup_state['status'] = 'ready'
        report_startup('ready')
        if args.dry_run:
            print('[BridgeVLA] dry run complete; no predictions or control steps executed', flush=True)
            return 0
        rng = np.random.default_rng(args.seed)
        connection = f'replay={replay.record["path"]} HTTP=disabled' if replay is not None else f'HTTP={args.server}'
        print(f'[BridgeVLA] {connection} task={args.task} mode={client.action_config["mode"]} '
              f'profile={profile} cameras={mapping} initialization={args.initialization} score={args.score} '
              f'execution={args.execution_mode}', flush=True)
        print(f'[BridgeVLA] TCP overlay={markers.info}', flush=True)
        if replay is None and client.info.get('visualization', {}).get('enabled'):
            print(f'[BridgeVLA] live heatmaps: {args.server.rstrip("/")}/visualization', flush=True)
        if replay is None:
            worker = PolicyWorker(client, GOALS[args.task], profile)
        def refresh_wait():
            refresh_while_waiting(reference, env)
        for episode in range(args.episodes):
            row = dict(episode=episode, task=args.task, success=None if tracker is None else False,
                       stop_reason='max_steps', steps=0, actions=[])
            records.append(row)
            try:
                markers.reset()
                bridge.reset_execution_state()
                env.reset()
                if startup is not None:
                    reference.restore_rigid_object_to_startup_state(env, startup)
                if args.initialization == 'collection':
                    reference.randomize_robot_initial_pose_like_collection(env, robot, rng, episode)
                elif args.initialization == 'task':
                    task_runtime.initialize(episode_index=episode, seed=args.seed + episode)
                elif args.initialization == 'trajectory':
                    row['initialization'] = (initial_state if initial_state is not None else replay).initialize(bridge)
                for _ in range(4):
                    reference._replicator_render_tick(env)
                if replay is None:
                    if initial_state is not None:
                        row['initial_alignment'] = initial_state.check_initial(bridge)
                    worker.submit_reset()
                    worker.result(wait_callback=refresh_wait)
                else:
                    row['initial_alignment'] = replay.check_initial(bridge, args)
                simulator_steps = 0
                if tracker is not None:
                    tracker.reset(episode_index=episode)
                def after_step(gripper):
                    nonlocal simulator_steps
                    simulator_steps += 1
                    return tracker is not None and tracker.update(step=simulator_steps, executed_gripper_cmd=gripper)
                steps = args.max_steps if replay is None else min(args.max_steps, replay.length)
                for step in range(steps):
                    if not reference.simulation_app.is_running():
                        raise RuntimeError('Simulation application stopped')
                    step_started = perf_counter()
                    measured = bridge.capture()
                    camera_observation = {
                        name: dict(pose_wxyz=measured[f'{name}_pose'].tolist(),
                                   intrinsics=measured[f'{name}_intrinsics'].tolist(),
                                   image_size=list(measured[f'{name}_rgb'].shape[:2]),
                                   pose_source=getattr(camera, 'pose_source', 'runtime'),
                                   pose_diagnostics=deepcopy(getattr(camera, 'pose_diagnostics', None)))
                        for name, camera in bridge.cameras.items()}
                    markers.update(bridge.measured_state(measured), bridge.last_target, client.action_config['mode'])
                    captured = perf_counter()
                    timestamp = int(env.common_step_counter) * float(env.step_dt)
                    source_details = {}
                    if replay is None:
                        worker.submit(measured, args.task, timestamp)
                        action, response = worker.result(wait_callback=refresh_wait)
                    else:
                        action, source_details = replay.action(step)
                        response = {'timing_ms': {}}
                    predicted = perf_counter()
                    # Also guard a reply that completed just as the user closed the window.
                    if not reference.simulation_app.is_running():
                        raise RuntimeError('Simulation application stopped before executing policy action')
                    executor = chunk_executor if chunk_executor is not None else waypoint
                    commands = (executor.execute(action, measured, after_step) if executor is not None else
                                bridge.execute(action, measured, after_step))
                    execution = executor.last_result if executor is not None else None
                    executed = perf_counter()
                    before = bridge.measured_state(measured)
                    after = bridge.measured_state()
                    markers.update(after, bridge.last_target, client.action_config['mode'])
                    errors_before, errors_after = ((bridge.tracking_errors(before), bridge.tracking_errors(after))
                                                   if bridge.last_target is not None else ({}, {}))
                    client_timing = dict(response.get('client_timing_ms', {}))
                    client_timing.update(capture=(captured - step_started) * 1000,
                                         request=(predicted - captured) * 1000 if replay is None else 0.,
                                         execute=(executed - predicted) * 1000,
                                         total=(perf_counter() - step_started) * 1000)
                    row['actions'].append(dict(step=step, timestamp=timestamp, action=action.tolist(),
                                               target=bridge.last_target, commands=commands,
                                               measured_before=before, measured_after=after,
                                               camera_observation=camera_observation,
                                               errors_before=errors_before, errors_after=errors_after,
                                               action_source=('gt_keypoint' if args.replay_targets == 'keypoints' else 'gt_trajectory')
                                               if replay is not None else 'policy',
                                               **source_details,
                                               timing_ms=response['timing_ms'], client_timing_ms=client_timing,
                                               visualization=response.get('visualization')))
                    if execution is not None:
                        execution['joint_diagnostics'] = joint_report(bridge, after)
                        row['actions'][-1]['execution'] = execution
                    if args.log_interval and (step == 0 or (step + 1) % args.log_interval == 0):
                        print(f'[BridgeVLA] step={step} action={action.tolist()} target={bridge.last_target} '
                              f'error_before={errors_before} error_after={errors_after} '
                              f'gripper_physical={before["gripper_position"]:.6g}->{after["gripper_position"]:.6g} '
                              f'client_ms={client_timing} server_ms={response["timing_ms"]}', flush=True)
                    row['steps'] = step + 1
                    if tracker is not None and tracker.success:
                        row['success'] = True
                        row['stop_reason'] = 'success'
                        break
                    if bridge.last_done:
                        row['environment_done'] = True
                        row['stop_reason'] = 'environment_done'
                        break
                    if execution is not None and not execution['completed']:
                        row['stop_reason'] = ('chunk_' if chunk_executor else 'waypoint_') + execution['status']
                        row['error'] = execution.get('error') or f'Waypoint did not complete: {execution["status"]}'
                        print(f'[BridgeVLA] waypoint stopped: step={step} execution={execution}', flush=True)
                        break
                if replay is not None and row['stop_reason'] == 'max_steps' and row['steps'] == replay.length:
                    row['stop_reason'] = 'trajectory_end'
            except Exception as exc:
                row['error'] = f'{type(exc).__name__}: {exc}'
                row['stop_reason'] = 'error'
                # No timed-out inference is retried or followed by another control command.
                print(f'[BridgeVLA] stopped: {row["error"]}', flush=True)
                break
            finally:
                write_json(output / f'episode_{episode:04d}.json', row)
                successes = sum(item['success'] is True for item in records)
                write_json(output / 'summary.json', dict(
                    task=args.task, mode=args.mode, initialization=args.initialization, success_evaluator=args.score,
                    planned=args.episodes, attempted=len(records), unattempted=args.episodes - len(records),
                    successes=None if tracker is None else successes,
                    success_rate=None if tracker is None else successes / len(records),
                    episodes=[{key: value for key, value in item.items() if key != 'actions'} for item in records]))
            print(f'[BridgeVLA] episode={episode} steps={row["steps"]} success={row["success"]} '
                  f'stop_reason={row["stop_reason"]}', flush=True)
            if 'error' in row:
                break
        failed = any('error' in row for row in records)
        if failed and args.keep_open_on_error:
            if worker is not None:
                worker.close()
            keep_open_after_error(reference, env)
        return 1 if failed else 0
    except Exception as exc:
        if startup_state['status'] != 'ready':
            startup_state.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                                 traceback=traceback.format_exc())
            # Flush the original failure before closing the simulator: SDK shutdown
            # may terminate the process before Python can report an uncaught error.
            print(f'[BridgeVLA] startup failed: phase={startup_state["phase"]} '
                  f'{startup_state["error"]}', file=sys.stderr, flush=True)
            print(startup_state['traceback'], file=sys.stderr, flush=True)
            try:
                write_json(output / 'startup.json', startup_state)
            except Exception as log_error:
                print(f'[BridgeVLA] could not save startup.json: {log_error}', file=sys.stderr, flush=True)
            if args.keep_open_on_error and reference is not None and env is not None:
                keep_open_after_error(reference, env)
        raise
    finally:
        if worker is not None:
            worker.close()
        if markers is not None:
            markers.close()
        close = getattr(reference, 'close', None)
        if callable(close):
            close()
        elif reference is not None:
            try:
                if env is not None:
                    env.close()
            finally:
                reference.simulation_app.close()


if __name__ == '__main__':
    raise SystemExit(main())
