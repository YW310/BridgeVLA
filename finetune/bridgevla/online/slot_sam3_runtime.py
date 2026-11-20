"""Online runtime selection and launcher; no model imports during config checks."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / 'finetune/RLBench/configs/rlbench_slot_guided_sam3_online.yaml'
FIELDS = ('task_repo', 'sam3_checkpoint', 'perception_config', 'sam3_device',
          'prompt_mode', 'fallback', 'diagnostics_dir', 'stage', 'guidance_stage')
CLI_FIELDS = {
    'task_repo': '--slot-sam3-task-repo', 'sam3_checkpoint': '--slot-sam3-checkpoint',
    'perception_config': '--slot-sam3-perception-config', 'sam3_device': '--slot-sam3-device',
    'prompt_mode': '--slot-sam3-prompt-mode', 'fallback': '--slot-sam3-fallback',
    'diagnostics_dir': '--slot-sam3-diagnostics-dir',
    'stage': '--slot-sam3-stage',
    'guidance_stage': '--slot-sam3-guidance-stage',
}


def load_online_profile(path):
    from bridgevla.config import get_cfg_defaults
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(path))
    if not cfg.rvt.enable_slot_guided_sam3 or cfg.rvt.slot_guided_sam3_execution_mode != 'online_experiment':
        raise ValueError('the online launcher requires an enabled online_experiment profile')
    legacy = cfg.object_slots.predictor_type == 'slots'
    if (not cfg.object_slots.enabled or cfg.object_slots.predictor_type not in ('slots', 'role_queries')
            or cfg.object_slots.num_slots != 2 or cfg.oracle_relation_anchor_rank <= 0
            or not cfg.oracle_relation_gated_adapter
            or (not legacy and (not cfg.object_conditioning.shared_action_features
                               or not cfg.object_conditioning.preserve_role_tokens))):
        raise ValueError('online Slot/SAM3 requires two original slots or two role queries with an Anchor adapter')
    return cfg


def resolve_runtime(cfg, overrides=None, *, policy_device='cuda:0', validate_files=False):
    values = {key: getattr(cfg.slot_guided_sam3, key) for key in FIELDS}
    for key, value in (overrides or {}).items():
        if key not in FIELDS:
            raise ValueError(f'unknown online SAM3 setting: {key}')
        if value is not None:
            values[key] = value
    if values['prompt_mode'] not in ('box-only', 'point+box') or values['fallback'] not in ('baseline', 'fail'):
        raise ValueError('invalid online prompt or fallback mode')
    if values['stage'] not in ('action', 'prompts', 'segmentation'):
        raise ValueError('online stage must be action, prompts or segmentation')
    if values['guidance_stage'] not in ('coarse', 'refine'):
        raise ValueError('guidance_stage must be coarse or refine')
    for key in ('task_repo', 'sam3_checkpoint', 'diagnostics_dir'):
        if not values[key] and key == 'diagnostics_dir':
            values[key] = None
            continue
        path = Path(values[key]).expanduser()
        values[key] = str((ROOT / path).resolve() if not path.is_absolute() else path.resolve())
    values['sam3_device'] = values['sam3_device'] or policy_device
    if values['stage'] in ('prompts', 'segmentation') and values['diagnostics_dir'] is None:
        raise ValueError('prompts/segmentation stage requires --slot-sam3-diagnostics-dir')
    if cfg.object_slots.predictor_type == 'slots':
        if values['guidance_stage'] != 'coarse':
            raise ValueError('original slots require coarse guidance; independent Refine stays unchanged')
    if validate_files:
        repo = Path(values['task_repo'])
        for name in ('slot_guidance_producers.py', 'slot_guided_sam3.py', 'slot_geometry_fusion.py'):
            if not (repo / name).is_file():
                raise FileNotFoundError(f'TASK checkout is missing {name}: {repo}')
        if values['stage'] != 'prompts' and not Path(values['sam3_checkpoint']).is_file():
            raise FileNotFoundError('set --slot-sam3-checkpoint to the actual local SAM3 weights')
        config = Path(values['perception_config'])
        if not (config if config.is_absolute() else repo / config).is_file():
            raise FileNotFoundError('online TASK perception config is missing')
        if values['stage'] != 'prompts' and importlib.util.find_spec('sam3') is None:
            raise RuntimeError('install a SAM3 package supporting the TASK interactive API in the model environment')
    return values


def make_online_policy(agent, observer):
    """Called only by direct RLBench eval, after its mutable options are set."""
    from bridgevla.online.slot_sam3_perception import OnlineTaskPerception
    from bridgevla.online.slot_sam3_policy import OnlineSlotSam3Policy
    values = agent.slot_guided_sam3_runtime
    if values['stage'] == 'prompts':
        from bridgevla.online.slot_prompt_probe import SlotPromptProbe
        perception = SlotPromptProbe(values['task_repo'], config_path=values['perception_config'], mode=values['prompt_mode'])
        agent.slot_guided_sam3_model_load_ms = 0.
    else:
        perception = OnlineTaskPerception(
            values['task_repo'], values['perception_config'], values['sam3_checkpoint'],
            values['sam3_device'], values['prompt_mode'])
        # Startup fails before env.launch() if SAM3's runtime API/weights cannot load.
        agent.slot_guided_sam3_model_load_ms = perception.prepare_model()
    diagnostics_dir = values['diagnostics_dir']
    if diagnostics_dir is not None:
        from uuid import uuid4
        diagnostics_dir = Path(diagnostics_dir) / ('run_' + uuid4().hex)
    if values['stage'] == 'prompts':
        print(f'[SlotPromptProbe] SAM3 not run; action=baseline_C; outputs={diagnostics_dir}', flush=True)
    elif values['stage'] == 'segmentation':
        print(f'[SlotSAM3] SAM3 segmentation; action=baseline_C; outputs={diagnostics_dir}', flush=True)
    return OnlineSlotSam3Policy(
        agent, perception, observer, agent.slot_guided_sam3_checkpoint_path,
        agent.slot_guided_sam3_git_sha, enabled=True,
        point_count=agent.slot_guided_sam3_point_count,
        expected_checkpoint_sha256=agent.slot_guided_sam3_checkpoint_sha256,
        diagnostics_dir=diagnostics_dir, fallback=values['fallback'], stage=values['stage'],
        guidance_stage=values['guidance_stage'])


def main(argv=None):
    import json
    parser = argparse.ArgumentParser(description=(
        'Live RLBench Coarse Slot -> persistent SAM3 -> geometry -> action loop. '
        'Other options are forwarded to eval.py, including --model-folder, '
        '--model-name, --eval-datafolder, --tasks and --device.'))
    parser.add_argument('--config', type=Path, default=PROFILE)
    parser.add_argument('--check-config', action='store_true')
    for key, flag in CLI_FIELDS.items():
        choices = ('box-only', 'point+box') if key == 'prompt_mode' else (
            ('baseline', 'fail') if key == 'fallback' else (
                ('action', 'prompts', 'segmentation') if key == 'stage' else (
                    ('coarse', 'refine') if key == 'guidance_stage' else None)))
        parser.add_argument(flag, dest=key, default=None, choices=choices,
                            help=('prompts: inspect slots/prompts without SAM3; segmentation: run SAM3 '
                                  'and retain baseline action; both require a diagnostics directory')
                            if key == 'stage' else None)
    args, evaluation_args = parser.parse_known_args(argv)
    try:
        cfg = load_online_profile(args.config)
        overrides = {key: getattr(args, key) for key in FIELDS}
        settings = resolve_runtime(cfg, overrides)
        if args.check_config:
            print(json.dumps(dict(enabled=True, execution_mode='online_experiment',
                                  config=str(args.config.resolve()), settings=settings)))
            return 0
        if any(arg == '--exp_cfg_path' or arg.startswith('--exp_cfg_path=') for arg in evaluation_args):
            raise ValueError('select the online experiment through --config, not a second --exp_cfg_path')
        for key, value in overrides.items():
            if value is not None:
                evaluation_args += [CLI_FIELDS[key], value]
        eval_path = ROOT / 'finetune/RLBench/eval.py'
        if not any(arg == '--log-name' or arg.startswith('--log-name=') for arg in evaluation_args):
            from datetime import datetime, timezone
            from uuid import uuid4
            evaluation_args += ['--log-name', 'slot_sam3_online/'
                                + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
                                + '_' + uuid4().hex[:8]]
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f'{type(error).__name__}: {error}', file=sys.stderr)
        return 2

    # Keep concise errors for configuration above, but let evaluation failures
    # retain their full traceback and exception chain (e.g. a mixed-dtype op).
    previous = sys.argv
    sys.argv = [str(eval_path), '--exp_cfg_path', str(args.config.resolve()), *evaluation_args]
    try:
        runpy.run_path(str(eval_path), run_name='__main__')
    finally:
        sys.argv = previous
    return 0
