"""Run current-frame A/B perception and frozen action predictions beside C.

No simulator, controller, episode pickle, expert action or acceptance report is
read here. This command collects experimental outputs; it does not certify them.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
IDENTITY = ('episode_id', 'frame_id', 'source_frame_index', 'observed_at')


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'duplicate JSON key: {key}')
            result[key] = value
        return result
    result = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique)
    if not isinstance(result, dict):
        raise ValueError('expected a JSON object')
    return result


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def load_current_state(path, snapshot):
    state = read_json(path)
    required = {'schema_version', 'artifact_type', *IDENTITY,
                'low_dim_state', 'instruction', 'task_name'}
    if (set(state) != required or type(state['schema_version']) is not int
            or state['schema_version'] != 1 or state['artifact_type'] != 'slot_policy_state'):
        raise ValueError('state must be a slot_policy_state schema-1 current-frame object')
    for key in IDENTITY:
        if type(state[key]) is not type(snapshot.metadata[key]) or state[key] != snapshot.metadata[key]:
            raise ValueError(f'current policy state / RGB-D provenance mismatch: {key}')
    for key in ('instruction', 'task_name'):
        if not isinstance(state[key], str) or not state[key].strip():
            raise ValueError(f'{key} is required')
    vector = state['low_dim_state']
    if (not isinstance(vector, list) or len(vector) != 4
            or any(isinstance(x, bool) or not isinstance(x, (float, int)) for x in vector)
            or not np.isfinite(vector).all()):
        raise ValueError('RLBench low_dim_state must be the four finite current proprioceptive/time values')
    return state


def load_oracle_labels(path, snapshot):
    path = Path(path)
    labels = read_json(path)
    required = {*IDENTITY, 'reference_is_null', 'role_masks'}
    if set(labels) not in (required, required | {'instance_ids'}):
        raise ValueError('oracle labels need exact current identity, role masks and Reference NULL decision')
    for key in IDENTITY:
        if type(labels[key]) is not type(snapshot.metadata[key]) or labels[key] != snapshot.metadata[key]:
            raise ValueError(f'oracle labels / RGB-D provenance mismatch: {key}')
    if type(labels['reference_is_null']) is not bool or not isinstance(labels['role_masks'], dict):
        raise ValueError('oracle Reference NULL must be an explicit boolean')
    if set(labels['role_masks']) != set(snapshot.cameras):
        raise ValueError('oracle labels must annotate every current camera')
    masks = {}
    for camera, roles in labels['role_masks'].items():
        if not isinstance(roles, dict):
            raise ValueError('role_masks must map camera names to role NPY paths')
        masks[camera] = {}
        for role, name in roles.items():
            if role not in ('target', 'reference') or not isinstance(name, str) or Path(name).suffix != '.npy':
                raise ValueError('oracle masks must be role-specific boolean NPY files')
            value = np.load(path.parent / name, allow_pickle=False)
            if value.dtype != bool or value.shape != snapshot.cameras[camera].depth_m.shape:
                raise ValueError('oracle masks must be original-resolution boolean arrays')
            masks[camera][role] = value
    return dict(oracle_role_masks=masks, reference_is_null=labels['reference_is_null'],
                oracle_instance_ids=labels.get('instance_ids'))


def guidance_export(packet, snapshot):
    matrix = np.asarray(packet['matrix_world_from_source'])
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('coarse callback must supply its actual finite world inverse')
    return {
        'slot_output': packet['slot_output'], 'point_valid': packet['point_valid'],
        'point_support_source': packet['point_support_source'],
        'transform': dict(source_frame=packet['source_frame'], target_frame='rlbench_world',
                          matrix_world_from_source=matrix.tolist(),
                          source_frame_index=snapshot.metadata['source_frame_index'],
                          timestamp=snapshot.metadata['observed_at'],
                          source='RVTAgent.act current scene place_pc_in_cube inverse; coarse only'),
    }


def _action(result):
    action = np.asarray(result.action, dtype=np.float64)
    if action.shape != (9,) or not np.isfinite(action).all():
        raise ValueError('RLBench policy must return a finite 9D world action')
    return action


def runtime_digest():
    files = [ROOT / 'finetune/bridgevla/models/bridgevla_agent.py',
             ROOT / 'finetune/bridgevla/models/slot_guided_geometry.py',
             ROOT / 'finetune/bridgevla/models/coarse_slot_export.py',
             ROOT / 'finetune/bridgevla/models/oracle_prior.py',
             ROOT / 'finetune/bridgevla/models/cross_scale_roles.py',
             ROOT / 'finetune/bridgevla/data/observations.py',
             ROOT / 'finetune/bridgevla/mvt/mvt.py', ROOT / 'finetune/bridgevla/mvt/mvt_single.py']
    files += sorted((ROOT / 'finetune/bridgevla/offline').glob('*.py'))
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def run_single_frame(*, agent, perception, frame_path, state_path, checkpoint_path,
                     bridge_git_sha, output_dir, device='cpu', guidance_source='predicted_coarse_slot',
                     oracle_labels=None, point_count=128, fallback='baseline', model_config=None):
    import torch
    from bridgevla.models.slot_guided_geometry import load_pr3_geometry, sha256_file

    started_frame = time.perf_counter()
    cuda = torch.device(device).type == 'cuda'
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    if guidance_source not in ('predicted_coarse_slot', 'oracle_gt') or fallback not in ('baseline', 'fail'):
        raise ValueError('invalid offline source or fallback mode')
    if guidance_source == 'oracle_gt' and oracle_labels is None:
        raise ValueError('Oracle action experiment requires current-frame labels')
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('use a new empty output directory for each frame/mode/source experiment')
    snapshot = perception.load_snapshot(frame_path)
    state = load_current_state(state_path, snapshot)
    receipt = perception.snapshot_receipt(snapshot)
    oracle = load_oracle_labels(oracle_labels, snapshot) if guidance_source == 'oracle_gt' else {}
    observation = perception.snapshot_to_observation(
        snapshot, state['low_dim_state'], state['instruction'], agent.cameras, device=device)
    observation['slot_guided_current_frames'] = [{key: value for key, value in receipt.items()
                                                if key != 'snapshot_input_sha256'}]
    output.mkdir(parents=True, exist_ok=True)
    packets = []
    previous_flag = agent.enable_slot_guided_sam3
    checkpoint_sha = sha256_file(checkpoint_path)
    if model_config and model_config.get('bridge_checkpoint_sha256') != checkpoint_sha:
        raise ValueError('BridgeVLA checkpoint no longer matches the loaded model')
    report = dict(schema_version=1, artifact_type='slot_guided_offline_action_experiment',
                  execution_mode='offline_experiment', evidence_kind='offline_policy_prediction',
                  guidance_source=guidance_source, mode=perception.mode,
                  bridge_git_sha=bridge_git_sha, checkpoint_path=str(Path(checkpoint_path).resolve()),
                  bridge_checkpoint_sha256=checkpoint_sha,
                  model_config=model_config or {}, runtime_sha256=runtime_digest(),
                  current_frame=receipt, real_offline_acceptance_passed=False,
                  robot_task_success=None, action_translation_error_m=None,
                  timings_ms={'input_preparation': (time.perf_counter() - started_frame) * 1000},
                  peak_gpu_memory_bytes=None,
                  gpu_memory_scope={'policy_device': device, 'sam3_device': getattr(perception, 'device', device),
                                    'allocator': 'torch policy-device peak; separate SAM3 device excluded'})
    def synchronize():
        if cuda:
            torch.cuda.synchronize(device)
    def timed(name, function):
        synchronize()
        started = time.perf_counter()
        value = function()
        synchronize()
        report['timings_ms'][name] = (time.perf_counter() - started) * 1000
        return value
    try:
        with torch.inference_mode():
            # C and B share the same observation, checkpoint and scene transform.
            # Disabling the flag invokes the original action conditioning path.
            agent.enable_slot_guided_sam3 = False
            baseline = _action(timed('baseline_C', lambda: agent.act(
                0, observation, deterministic=True,
                coarse_slot_callback=packets.append if guidance_source == 'predicted_coarse_slot' else None)))
            report['baseline_C_action'] = baseline.tolist()
            if guidance_source == 'predicted_coarse_slot' and len(packets) != 1:
                raise ValueError('expected exactly one current-frame coarse export')
            export = guidance_export(packets[0], snapshot) if packets else None
            if packets:
                np.savez_compressed(output / 'coarse_slots.npz',
                                    **{key: value for key, value in packets[0]['slot_output'].items()
                                       if value is not None}, point_valid=packets[0]['point_valid'])
                write_json(output / 'coarse_transform.json', export['transform'])
            try:
                result = timed('perception', lambda: perception.perceive(
                    frame_path, export, state['instruction'], state['task_name'], checkpoint_path,
                    bridge_git_sha, output / 'perception', source=guidance_source,
                    expected_snapshot_digest=receipt['snapshot_input_sha256'], **oracle))
                if result['backend_kind'] != 'sam3':
                    raise ValueError('Mock evidence cannot produce the SAM3-conditioned action experiment')
                report['evidence_kind'] = 'real_offline_inference'
                geometry = load_pr3_geometry(result['object_geometry_path'], count=point_count)
                report['perception_report'] = result
                if (not geometry.object_role_valid[0, 0]
                        or geometry.object_geometry_confidence[0, 0] < .25):
                    raise ValueError('SAM3/fusion produced no reliable Target geometry')
                if sha256_file(checkpoint_path) != checkpoint_sha:
                    raise ValueError('BridgeVLA checkpoint file changed during the experiment')
                agent.enable_slot_guided_sam3 = True
                conditioned = _action(timed('conditioned_action', lambda: agent.act(
                    0, observation, deterministic=True, slot_guided_sam3=dict(
                        execution_mode='offline_experiment', geometry=geometry,
                        bridge_checkpoint_path=str(checkpoint_path), bridge_git_sha=bridge_git_sha))))
                report.update(status='completed', conditioned_action=conditioned.tolist(),
                              action_translation_change_from_C_m=float(np.linalg.norm(conditioned[:3] - baseline[:3])))
            except (ValueError, RuntimeError, OSError, ImportError) as error:
                report.update(status='baseline_fallback' if fallback == 'baseline' else 'failed',
                              failure_reason=f'{type(error).__name__}: {error}', conditioned_action=None)
                if fallback == 'baseline':
                    report['fallback_action'] = baseline.tolist()
                if fallback != 'baseline':
                    raise
    except Exception as error:
        report.update(status='failed', failure_reason=f'{type(error).__name__}: {error}')
        raise
    finally:
        agent.enable_slot_guided_sam3 = previous_flag
        synchronize()
        report['timings_ms']['total_frame'] = (time.perf_counter() - started_frame) * 1000
        if cuda:
            report['peak_gpu_memory_bytes'] = int(torch.cuda.max_memory_allocated(device))
        report['created_at'] = datetime.now(timezone.utc).isoformat()
        write_json(output / 'action_report.json', report)
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--config', type=Path, default=ROOT / 'finetune/RLBench/configs/rlbench_slot_guided_sam3_offline.yaml')
    result.add_argument('--check-config', action='store_true')
    result.add_argument('--check-inputs', action='store_true', help='validate files/config/provenance without loading model weights')
    for name in ('checkpoint', 'exp-cfg', 'mvt-cfg', 'task-repo', 'perception-config',
                 'frame', 'state', 'sam3-checkpoint', 'oracle-labels', 'output-dir'):
        result.add_argument('--' + name, type=Path)
    result.add_argument('--device', default='cuda:0')
    result.add_argument('--sam3-device', default=None)
    result.add_argument('--mode', choices=('box-only', 'point+box'), default='point+box')
    result.add_argument('--guidance-source', choices=('predicted_coarse_slot', 'oracle_gt'), default='predicted_coarse_slot')
    result.add_argument('--fallback', choices=('baseline', 'fail'), default='baseline')
    return result


def main(argv=None):
    from bridgevla.offline.slot_sam3_checkpoint import load_profile, load_saved_configs, load_frozen_agent
    from bridgevla.offline.slot_sam3_perception import TaskPerception
    args = parser().parse_args(argv)
    try:
        profile = load_profile(args.config)
        if args.check_config:
            print(json.dumps({'config': str(args.config), 'enabled': True,
                              'execution_mode': profile.rvt.slot_guided_sam3_execution_mode}))
            return 0
        for key in ('checkpoint', 'task_repo', 'frame', 'state', 'sam3_checkpoint'):
            value = getattr(args, key)
            if value is None or (not value.is_dir() if key == 'task_repo' else not value.is_file()):
                raise ValueError(f'--{key.replace("_", "-")} must name an existing local path')
        if not args.check_inputs and args.output_dir is None:
            raise ValueError('--output-dir is required')
        if args.guidance_source == 'oracle_gt' and args.oracle_labels is None:
            raise ValueError('--oracle-labels is required for Oracle guidance')
        perception = TaskPerception(args.task_repo, args.perception_config,
                                    args.sam3_checkpoint, args.sam3_device or args.device, args.mode)
        snapshot = perception.load_snapshot(args.frame)
        from bridgevla.data.observations import DEFAULT_CAMERAS
        perception.validate_policy_snapshot(snapshot, DEFAULT_CAMERAS)
        load_current_state(args.state, snapshot)
        if args.oracle_labels:
            load_oracle_labels(args.oracle_labels, snapshot)
        load_saved_configs(args.checkpoint, profile, args.exp_cfg, args.mvt_cfg)
        if args.check_inputs:
            print(json.dumps({'input_contract': 'valid', 'models_loaded': False,
                              'real_offline_acceptance_passed': False,
                              'current_frame': perception.snapshot_receipt(snapshot)}))
            return 0
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise FileExistsError('use a new empty --output-dir')
        if importlib.util.find_spec('sam3') is None:
            raise RuntimeError('SAM3 is not installed in this Python; install the TASK-audited SAM3 package before model inference')
        import torch
        if torch.device(args.device).type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('the selected policy device requires CUDA, which is unavailable in this Python')
        source_sha = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        model_started = time.perf_counter()
        agent, configs = load_frozen_agent(args.checkpoint, profile, device=args.device,
                                           exp_cfg_path=args.exp_cfg, mvt_cfg_path=args.mvt_cfg)
        if torch.device(args.device).type == 'cuda':
            torch.cuda.synchronize(args.device)
        configs['bridge_model_load_ms'] = (time.perf_counter() - model_started) * 1000
        sam3_started = time.perf_counter()
        perception.prepare_backend(snapshot)
        sam3_device = args.sam3_device or args.device
        if torch.device(sam3_device).type == 'cuda':
            torch.cuda.synchronize(sam3_device)
        configs['sam3_model_load_ms'] = (time.perf_counter() - sam3_started) * 1000
        report = run_single_frame(agent=agent, perception=perception, frame_path=args.frame,
            state_path=args.state, checkpoint_path=args.checkpoint, bridge_git_sha=source_sha,
            output_dir=args.output_dir, device=args.device, guidance_source=args.guidance_source,
            oracle_labels=args.oracle_labels, point_count=profile.object_slots.point_samples,
            fallback=args.fallback, model_config=configs)
        print(json.dumps({'status': report['status'], 'report': str(args.output_dir / 'action_report.json')}))
        return 0 if report['status'] == 'completed' else 2
    except (ValueError, RuntimeError, OSError, ImportError) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 2
