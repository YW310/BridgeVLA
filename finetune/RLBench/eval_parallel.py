"""Evaluate RLBench tasks concurrently with at most one process per GPU."""

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


RLBENCH_TASKS = (
    'close_jar',
    'reach_and_drag',
    'insert_onto_square_peg',
    'meat_off_grill',
    'open_drawer',
    'place_cups',
    'place_wine_at_rack_location',
    'push_buttons',
    'put_groceries_in_cupboard',
    'put_item_in_drawer',
    'put_money_in_safe',
    'light_bulb_in',
    'slide_block_to_color_target',
    'place_shape_in_shape_sorter',
    'stack_blocks',
    'stack_cups',
    'sweep_to_dustpan_of_size',
    'turn_tap',
)


def parse_gpu_ids(value):
    ids = tuple(part.strip() for part in value.split(',') if part.strip())
    if not ids:
        raise argparse.ArgumentTypeError('at least one GPU id is required')
    if len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError('GPU ids must be unique')
    return ids


def read_task_result(csv_path, expected_task):
    with csv_path.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(
            f'{csv_path} must contain exactly one task row; found {len(rows)}'
        )
    row = rows[0]
    if row.get('task') != expected_task:
        raise RuntimeError(
            f'{csv_path} contains task {row.get("task")}, expected {expected_task}'
        )
    row['success rate'] = float(row['success rate'])
    return row


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-folder', type=Path, required=True)
    parser.add_argument('--eval-datafolder', type=Path, required=True)
    parser.add_argument('--model-name', default='model_80.pth')
    parser.add_argument('--gpus', type=parse_gpu_ids, default=parse_gpu_ids('0,1,2,3,4,5,6,7'))
    parser.add_argument('--tasks', nargs='+', default=list(RLBENCH_TASKS))
    parser.add_argument('--eval-episodes', type=int, default=25)
    parser.add_argument('--episode-length', type=int, default=25)
    parser.add_argument('--run-name', default=None)
    parser.add_argument('--no-xvfb', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if len(set(args.tasks)) != len(args.tasks):
        raise ValueError('tasks must not contain duplicates')
    unknown = sorted(set(args.tasks) - set(RLBENCH_TASKS))
    if unknown:
        raise ValueError(f'unknown RLBench tasks: {unknown}')
    if args.eval_episodes <= 0 or args.episode_length <= 0:
        raise ValueError('episode counts and lengths must be positive')

    script_dir = Path(__file__).resolve().parent
    run_name = args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S')
    run_root = args.model_folder / 'eval' / run_name
    run_root.mkdir(parents=True, exist_ok=False)
    gpu_pool = queue.Queue()
    for gpu in args.gpus:
        gpu_pool.put(gpu)

    def run_task(task):
        gpu = gpu_pool.get()
        try:
            task_log_name = f'{run_name}/{task}'
            command = [
                sys.executable,
                str(script_dir / 'eval.py'),
                '--model-folder',
                str(args.model_folder),
                '--eval-datafolder',
                str(args.eval_datafolder),
                '--tasks',
                task,
                '--eval-episodes',
                str(args.eval_episodes),
                '--episode-length',
                str(args.episode_length),
                '--log-name',
                task_log_name,
                '--device',
                '0',
                '--headless',
                '--model-name',
                args.model_name,
            ]
            if not args.no_xvfb:
                command = [
                    'xvfb-run',
                    '-a',
                    '-s',
                    '-screen 0 1280x1024x24 +extension GLX +render -noreset',
                    *command,
                ]
            environment = os.environ.copy()
            environment['CUDA_VISIBLE_DEVICES'] = gpu
            task_dir = run_root / task
            task_dir.mkdir(parents=True, exist_ok=False)
            log_path = task_dir / 'process.log'
            with log_path.open('w', encoding='utf-8') as log_handle:
                result = subprocess.run(
                    command,
                    cwd=script_dir,
                    env=environment,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode != 0:
                raise RuntimeError(
                    f'{task} failed on GPU {gpu}; see {log_path}'
                )
            model_stem = Path(args.model_name).stem
            result_path = task_dir / model_stem / 'eval_results.csv'
            return read_task_result(result_path, task)
        finally:
            gpu_pool.put(gpu)

    rows_by_task = {}
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        futures = {executor.submit(run_task, task): task for task in args.tasks}
        for future in as_completed(futures):
            task = futures[future]
            rows_by_task[task] = future.result()
            print(f'Finished {task}', flush=True)

    rows = [rows_by_task[task] for task in args.tasks]
    merged_path = run_root / 'merged_eval_results.csv'
    with merged_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    macro_success_rate = sum(row['success rate'] for row in rows) / len(rows)
    summary = {
        'run_name': run_name,
        'model_name': args.model_name,
        'tasks': len(rows),
        'episodes_per_task': args.eval_episodes,
        'macro_success_rate': macro_success_rate,
    }
    summary_path = run_root / 'summary.json'
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
