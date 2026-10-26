"""Run the OHT segmentation data pipeline without overwriting existing outputs."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, help='Raw root containing disassemble_left-seg/')
    p.add_argument('--output', help='New output directory; default: ROOT/bridgevla-seg-vN-TIMESTAMP')
    p.add_argument('--config', default=str(REPO_ROOT / 'finetune/OHT/configs/dataset.yaml'))
    p.add_argument('--mapping', default=str(REPO_ROOT / 'finetune/OHT/configs/seg_roles.yaml'))
    p.add_argument('--replay-version', type=int, choices=(3, 4), default=4,
                   help='4 (default): key goals + dense action chunks + masks; 3: single-key replay + masks + 3D teacher')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--sample-stride', type=int, default=10)
    p.add_argument('--visualize-every', type=int, default=100)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--point-count', type=int, default=512)
    p.add_argument('--chunk-horizon', type=int, default=16)
    p.add_argument('--action-frame-stride', type=int, default=1)
    p.add_argument('--dry-run', action='store_true', help='Print commands without creating files or decoding data')
    return p


def commands(args):
    out = Path(args.output)
    replay = out / f'replay-source-v{args.replay_version}'
    audit = out / 'audit.json'

    def command(tool, *values):
        return [sys.executable, '-u', str(REPO_ROOT / 'tools' / tool), *map(str, values)]

    stages = [('audit', command('audit_oht_dataset.py', '--root', args.root, '--output', audit,
                               '--seed', args.seed))]
    build = command('build_oht_replay.py', '--root', args.root, '--manifest', audit,
                    '--config', out / 'dataset.yaml', '--output', replay,
                    '--replay-version', args.replay_version, '--sample-stride', args.sample_stride,
                    '--visualize-every', args.visualize_every)
    if args.replay_version == 4:
        build += ['--chunk-horizon', str(args.chunk_horizon),
                  '--action-frame-stride', str(args.action_frame_stride)]
    stages += [('replay', build), ('validate', command('validate_oht_replay.py', '--replay', replay,
                                                     '--workers', args.workers)),
               ('seg-roles', command('export_oht_seg_roles.py', '--root', args.root,
                    '--manifest', audit, '--replay', replay, '--mapping', out / 'seg_roles.yaml',
                    '--output', out / 'seg-role-annotations'))]
    if args.replay_version == 3:
        stages.append(('teacher', command('build_oht_role_teacher.py', '--replay', replay,
            '--annotations', out / 'seg-role-annotations/annotations.jsonl',
            '--output', out / 'role-teacher', '--point-count', args.point_count,
            '--visualize-every', args.visualize_every)))
    elif args.visualize_every:
        stages.append(('role-preview', command('export_oht_replay_viewer.py', '--replay', replay,
            '--role-annotations', out / 'seg-role-annotations/annotations.jsonl',
            '--output', out / 'role-preview', '--every', args.visualize_every)))
    return stages


def write_status(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if not args.root.strip():
        p.error('--root is empty; set ROOT to the raw dataset path before running this command')
    if args.output is not None and not args.output.strip():
        p.error('--output is empty; specify a new output directory or omit the option')
    print(f'[OHT prepare] repo={REPO_ROOT} python={sys.executable}\n'
          f'[OHT prepare] raw_root={Path(args.root).resolve()}', flush=True)
    for name in ('sample_stride', 'workers', 'point_count', 'chunk_horizon', 'action_frame_stride'):
        if getattr(args, name) < 1:
            p.error(f'--{name.replace("_", "-")} must be positive')
    if args.visualize_every < 0:
        p.error('--visualize-every must be nonnegative')
    args.root = str(Path(args.root).resolve())
    for name in ('config', 'mapping'):
        value = Path(getattr(args, name)).resolve()
        if not value.is_file():
            p.error(f'Missing {name}: {value}')
        setattr(args, name, str(value))
    if not Path(args.root).is_dir():
        p.error(f'Missing raw root: {args.root}')
    if args.output is None:
        args.output = str(Path(args.root) / f'bridgevla-seg-v{args.replay_version}-{datetime.now():%Y%m%d-%H%M%S}')
    args.output = str(Path(args.output).resolve())
    output = Path(args.output)
    if output.exists():
        p.error(f'Output already exists; choose a new directory: {output}')
    stages = commands(args)
    print(f'OHT segmentation pipeline: replay v{args.replay_version}\nOutput: {output}', flush=True)
    if args.replay_version == 4:
        print(f'Dense targets: {args.chunk_horizon} future measured frames, '
              f'action frame stride={args.action_frame_stride}; independent of observation sample stride.', flush=True)
        print('V4 exports role masks; chunk training does not yet consume role caches.', flush=True)
    for name, command in stages:
        print(f'[{name}] {json.dumps(command, ensure_ascii=False)}', flush=True)
    if args.dry_run:
        print('[OHT prepare] dry-run only: no replay, masks or JSON files were generated.', flush=True)
        return 0
    output.mkdir(parents=True, exist_ok=False)
    status = dict(status='running', options=vars(args), completed=[], stages=dict(stages))
    status_path = output / 'pipeline.json'
    write_status(status_path, status)
    print(f'[OHT prepare] status={status_path}\n'
          f'[OHT prepare] replay={output / ("replay-source-v" + str(args.replay_version))}', flush=True)
    try:
        shutil.copyfile(args.config, output / 'dataset.yaml')
        shutil.copyfile(args.mapping, output / 'seg_roles.yaml')
    except OSError as exc:
        status.update(status='failed', failed_stage='setup', error=str(exc))
        write_status(status_path, status)
        print(f'[OHT prepare] stopped at setup: {exc}; see {status_path}', file=sys.stderr, flush=True)
        return 1
    for name, command in stages:
        print(f'\n[OHT prepare] starting {name}', flush=True)
        try:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
            if name == 'replay':
                replay = output / f'replay-source-v{args.replay_version}'
                missing = [str(replay / filename) for filename in ('contract.json', 'complete.json')
                           if not (replay / filename).is_file()]
                if missing:
                    raise FileNotFoundError('Replay build returned success without creating: '
                                            + ', '.join(missing)
                                            + '. Stopping before validation; inspect the build output above.')
        except (subprocess.CalledProcessError, OSError, KeyboardInterrupt) as exc:
            status.update(status='failed', failed_stage=name, error=str(exc))
            write_status(status_path, status)
            print(f'[OHT prepare] stopped at {name}; see {status_path}', file=sys.stderr, flush=True)
            return exc.returncode if isinstance(exc, subprocess.CalledProcessError) and exc.returncode > 0 else 1
        status['completed'].append(name)
        write_status(status_path, status)
    status['status'] = 'complete'
    write_status(status_path, status)
    print(f'\n[OHT prepare] complete: {output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
