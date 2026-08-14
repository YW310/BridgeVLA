#!/usr/bin/env python3
'''Generate BridgeVLA/YARR replay files from stored RLBench raw episodes.

This is a standalone wrapper around the same create_replay() and fill_replay()
implementation used by finetune/RLBench/train.py. It generates the standard
BridgeVLA replay first; Oracle object fields can then be appended with
tools/augment_replay_with_oracle_objects.py.
'''

from __future__ import annotations

import argparse
import gc
import importlib
import os
import pickle
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RLBENCH_UTILS_DIR = REPOSITORY_ROOT / 'finetune' / 'RLBench' / 'utils'
EPISODE_PATTERN = re.compile(r'^episode([0-9]+)$')
REQUIRED_REPLAY_KEYS = (
    'terminal',
    'episode_idx',
    'sample_frame',
    'next_keypoint_frame',
)


@dataclass(frozen=True)
class TaskEpisodes:
    task: str
    episode_root: Path
    episode_indices: Tuple[int, ...]


@dataclass(frozen=True)
class ReplayBackend:
    torch: object
    clip: object
    create_replay: Callable[..., object]
    fill_replay: Callable[..., None]
    cameras: Tuple[str, ...]
    scene_bounds: Tuple[float, ...]
    voxel_sizes: Tuple[int, ...]
    episode_folder: str
    variation_descriptions_pkl: str
    rotation_resolution: int


def _parse_task_values(values: Sequence[str]) -> Optional[Tuple[str, ...]]:
    names: List[str] = []
    for value in values:
        names.extend(part.strip() for part in value.split(',') if part.strip())
    if not names or names == ['all']:
        return None
    if 'all' in names:
        raise ValueError('--task all cannot be combined with explicit tasks')
    return tuple(dict.fromkeys(names))


def _episode_indices(episode_root: Path) -> Tuple[int, ...]:
    indices = []
    for path in episode_root.iterdir():
        match = EPISODE_PATTERN.fullmatch(path.name)
        if path.is_dir() and match is not None:
            indices.append(int(match.group(1)))
    return tuple(sorted(indices))


def _candidate_episode_roots(
    raw_data_dir: Path,
    split: str,
    task: str,
) -> Tuple[Path, ...]:
    candidates = [
        raw_data_dir / split / task / 'all_variations' / 'episodes',
        raw_data_dir / task / 'all_variations' / 'episodes',
    ]
    if raw_data_dir.name == task:
        candidates.append(raw_data_dir / 'all_variations' / 'episodes')
    if raw_data_dir.name == 'episodes':
        candidates.append(raw_data_dir)
    unique: List[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return tuple(unique)


def resolve_task_episode_root(
    raw_data_dir: Path,
    split: str,
    task: str,
) -> Path:
    candidates = _candidate_episode_roots(raw_data_dir, split, task)
    for candidate in candidates:
        if candidate.is_dir() and _episode_indices(candidate):
            return candidate
    tried = '\n  '.join(str(path) for path in candidates)
    raise FileNotFoundError(
        f'Could not locate raw episodes for task {task!r}. Tried:\n  {tried}'
    )


def discover_task_names(raw_data_dir: Path, split: str) -> Tuple[str, ...]:
    discovered: Dict[str, Path] = {}
    for base in (raw_data_dir / split, raw_data_dir):
        if not base.is_dir():
            continue
        for child in base.iterdir():
            episode_root = child / 'all_variations' / 'episodes'
            if child.is_dir() and episode_root.is_dir():
                if _episode_indices(episode_root):
                    discovered.setdefault(child.name, episode_root)
    direct = raw_data_dir / 'all_variations' / 'episodes'
    if direct.is_dir() and _episode_indices(direct):
        discovered.setdefault(raw_data_dir.name, direct)
    if raw_data_dir.name == 'episodes' and _episode_indices(raw_data_dir):
        try:
            task_name = raw_data_dir.parent.parent.name
        except IndexError:
            task_name = ''
        if task_name:
            discovered.setdefault(task_name, raw_data_dir)
    if not discovered:
        raise FileNotFoundError(
            f'No RLBench task episode folders found under {raw_data_dir}'
        )
    return tuple(sorted(discovered))


def select_episode_indices(
    available: Sequence[int],
    start_episode: int,
    num_demos: Optional[int],
) -> Tuple[int, ...]:
    if start_episode < 0:
        raise ValueError('--start-episode must be non-negative')
    if num_demos is not None and num_demos <= 0:
        raise ValueError('--num-demos must be positive')
    available_set = {int(value) for value in available}
    if not available_set:
        raise ValueError('No raw episodes are available')
    if start_episode not in available_set:
        raise ValueError(
            f'Raw start episode {start_episode} is not available'
        )
    if num_demos is None:
        last_episode = max(available_set)
        requested = tuple(range(start_episode, last_episode + 1))
    else:
        requested = tuple(range(start_episode, start_episode + num_demos))
    missing = [index for index in requested if index not in available_set]
    if missing:
        preview = ', '.join(str(value) for value in missing[:10])
        suffix = ' ...' if len(missing) > 10 else ''
        raise ValueError(
            'fill_replay requires a contiguous episode range; missing raw '
            f'episode(s): {preview}{suffix}'
        )
    return requested


def discover_tasks(
    raw_data_dir: Path,
    split: str,
    task_values: Sequence[str],
    start_episode: int,
    num_demos: Optional[int],
) -> Tuple[TaskEpisodes, ...]:
    requested = _parse_task_values(task_values)
    task_names = (
        discover_task_names(raw_data_dir, split)
        if requested is None
        else requested
    )
    tasks = []
    for task in task_names:
        episode_root = resolve_task_episode_root(raw_data_dir, split, task)
        selected = select_episode_indices(
            _episode_indices(episode_root), start_episode, num_demos
        )
        tasks.append(TaskEpisodes(task, episode_root, selected))
    return tuple(tasks)


def _paths_overlap(first: Path, second: Path) -> bool:
    first = first.resolve()
    second = second.resolve()
    return (
        first == second
        or first in second.parents
        or second in first.parents
    )


def _safe_remove_task_directory(path: Path, output_root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = output_root.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ValueError(
            f'Refusing to remove path outside output root: {resolved_path}'
        )
    shutil.rmtree(resolved_path)


def validate_replay_directory(path: Path) -> int:
    files = []
    for replay_path in path.glob('*.replay'):
        try:
            index = int(replay_path.stem)
        except ValueError:
            continue
        files.append((index, replay_path))
    files.sort()
    if not files:
        raise ValueError(f'No numeric *.replay files generated in {path}')
    indices = [index for index, _ in files]
    if indices != list(range(len(files))):
        raise ValueError(
            f'Replay filenames are not contiguous from 0 in {path}'
        )

    replay_info_path = path / 'replay_info.npy'
    if not replay_info_path.is_file():
        raise ValueError(f'Missing replay_info.npy in {path}')
    replay_info = np.asarray(
        np.load(replay_info_path, allow_pickle=False)
    ).reshape(-1)
    if len(replay_info) != len(files):
        raise ValueError(
            f'replay_info.npy has {len(replay_info)} entries but '
            f'{len(files)} replay files exist'
        )
    ordinary = np.flatnonzero(replay_info != -1)
    sentinels = np.flatnonzero(replay_info == -1)
    if ordinary.size == 0 or sentinels.size == 0:
        raise ValueError(
            f'Generated replay lacks ordinary transitions or final sentinels: {path}'
        )
    for file_index in (int(ordinary[0]), int(sentinels[-1])):
        with files[file_index][1].open('rb') as stream:
            transition = pickle.load(stream)
        if not isinstance(transition, dict):
            raise ValueError(
                f'{files[file_index][1]} does not contain a transition dict'
            )
        terminal = int(np.asarray(transition.get('terminal', -2)).item())
        if terminal != int(replay_info[file_index]):
            raise ValueError(
                f'terminal mismatch at replay index {file_index}: '
                f'{terminal} != {int(replay_info[file_index])}'
            )
        if terminal != -1:
            missing = [
                key for key in REQUIRED_REPLAY_KEYS if key not in transition
            ]
            if missing:
                raise ValueError(
                    f'{files[file_index][1]} is missing fields: {missing}'
                )
    return len(files)


def _load_backend() -> ReplayBackend:
    if not RLBENCH_UTILS_DIR.is_dir():
        raise FileNotFoundError(
            f'BridgeVLA RLBench utilities not found: {RLBENCH_UTILS_DIR}'
        )
    sys.path.insert(0, str(RLBENCH_UTILS_DIR))
    try:
        torch = importlib.import_module('torch')
        clip = importlib.import_module('clip')
        dataset = importlib.import_module('dataset')
        constants = importlib.import_module('peract_utils_rlbench')
    except ImportError as exc:
        raise RuntimeError(
            'Raw replay generation requires the BridgeVLA RLBench environment. '
            'Activate the bridgevla Conda environment and install the editable '
            'RLBench, PyRep, YARR, peract_colab, and BridgeVLA packages.'
        ) from exc
    return ReplayBackend(
        torch=torch,
        clip=clip,
        create_replay=dataset.create_replay,
        fill_replay=dataset.fill_replay,
        cameras=tuple(constants.CAMERAS),
        scene_bounds=tuple(constants.SCENE_BOUNDS),
        voxel_sizes=tuple(constants.VOXEL_SIZES),
        episode_folder=str(constants.EPISODE_FOLDER),
        variation_descriptions_pkl=str(
            constants.VARIATION_DESCRIPTIONS_PKL
        ),
        rotation_resolution=int(constants.ROTATION_RESOLUTION),
    )


def _resolve_device(torch: object, requested: str) -> str:
    if requested == 'auto':
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if requested.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(
            f'CUDA device {requested!r} was requested but CUDA is unavailable'
        )
    return requested


def _prepare_task_destination(
    task_output: Path,
    temporary_output: Path,
    output_root: Path,
    *,
    overwrite: bool,
    skip_existing: bool,
) -> bool:
    if task_output.exists():
        if skip_existing:
            count = validate_replay_directory(task_output)
            print(
                f'{task_output.name}: existing replay validated '
                f'({count} files); skipping',
                flush=True,
            )
            return False
        if not overwrite:
            raise FileExistsError(
                f'Output already exists: {task_output}. Use --skip-existing '
                'or --overwrite.'
            )
        _safe_remove_task_directory(task_output, output_root)
    if temporary_output.exists():
        if not overwrite:
            raise FileExistsError(
                f'Incomplete temporary output exists: {temporary_output}. '
                'Inspect it, then rerun with --overwrite to rebuild.'
            )
        _safe_remove_task_directory(temporary_output, output_root)
    return True


def generate_task_replay(
    backend: ReplayBackend,
    task_input: TaskEpisodes,
    output_root: Path,
    clip_model: object,
    device: str,
    *,
    batch_size: int,
    replay_capacity: int,
    demo_augmentation: bool,
    demo_augmentation_every_n: int,
    overwrite: bool,
    skip_existing: bool,
) -> int:
    task_output = output_root / task_input.task
    temporary_output = output_root / (
        f'.{task_input.task}.raw_to_replay.tmp'
    )
    should_generate = _prepare_task_destination(
        task_output,
        temporary_output,
        output_root,
        overwrite=overwrite,
        skip_existing=skip_existing,
    )
    if not should_generate:
        return validate_replay_directory(task_output)

    replay = backend.create_replay(
        batch_size=batch_size,
        timesteps=1,
        disk_saving=True,
        cameras=list(backend.cameras),
        voxel_sizes=list(backend.voxel_sizes),
        replay_size=replay_capacity,
        use_oracle_objects=False,
    )
    first_episode = task_input.episode_indices[0]
    print(
        f'{task_input.task}: generating {len(task_input.episode_indices)} '
        f'episode(s) {first_episode}-{task_input.episode_indices[-1]} -> '
        f'{task_output}',
        flush=True,
    )
    try:
        backend.fill_replay(
            replay=replay,
            task=task_input.task,
            task_replay_storage_folder=str(temporary_output),
            start_idx=first_episode,
            num_demos=len(task_input.episode_indices),
            demo_augmentation=demo_augmentation,
            demo_augmentation_every_n=demo_augmentation_every_n,
            cameras=list(backend.cameras),
            rlbench_scene_bounds=list(backend.scene_bounds),
            voxel_sizes=list(backend.voxel_sizes),
            rotation_resolution=backend.rotation_resolution,
            crop_augmentation=False,
            data_path=str(task_input.episode_root),
            episode_folder=backend.episode_folder,
            variation_desriptions_pkl=backend.variation_descriptions_pkl,
            clip_model=clip_model,
            device=device,
        )
        replay_count = validate_replay_directory(temporary_output)
        os.replace(temporary_output, task_output)
    except BaseException:
        print(
            f'{task_input.task}: generation failed; partial files remain at '
            f'{temporary_output}',
            file=sys.stderr,
            flush=True,
        )
        raise
    finally:
        del replay
        gc.collect()
    print(
        f'{task_input.task}: complete ({replay_count} replay files)',
        flush=True,
    )
    return replay_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--raw-data-dir',
        type=Path,
        required=True,
        help=(
            'RLBench dataset root, split directory, task directory, or '
            'episodes directory'
        ),
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        required=True,
        help='output root; each task is written to OUTPUT_DIR/TASK',
    )
    parser.add_argument('--split', default='train')
    parser.add_argument(
        '--task',
        action='append',
        default=[],
        help='task name, comma-separated names, or all (default: all)',
    )
    parser.add_argument('--start-episode', type=int, default=0)
    parser.add_argument(
        '--num-demos',
        type=int,
        help='number of contiguous episodes; default: all from start episode',
    )
    parser.add_argument(
        '--demo-augmentation-every-n',
        type=int,
        default=10,
        help='create an augmented replay segment every N raw frames (default: 10)',
    )
    parser.add_argument(
        '--no-demo-augmentation',
        action='store_true',
        help='generate only the episode-start segment',
    )
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--replay-capacity', type=int, default=300000)
    parser.add_argument(
        '--device',
        default='auto',
        help='CLIP device, for example auto, cpu, cuda:0 (default: auto)',
    )
    parser.add_argument(
        '--clip-model',
        default='RN50',
        help='OpenAI CLIP model name or local checkpoint path (default: RN50)',
    )
    existing = parser.add_mutually_exclusive_group()
    existing.add_argument(
        '--overwrite',
        action='store_true',
        help='delete and rebuild existing task output directories',
    )
    existing.add_argument(
        '--skip-existing',
        action='store_true',
        help='validate and skip existing task output directories',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='discover and validate inputs without loading CLIP or writing files',
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    raw_data_dir = args.raw_data_dir.resolve()
    output_root = args.output_dir.resolve()
    if not raw_data_dir.is_dir():
        raise FileNotFoundError(f'Raw data directory not found: {raw_data_dir}')
    if _paths_overlap(raw_data_dir, output_root):
        raise ValueError(
            '--output-dir must be separate from --raw-data-dir and must not '
            'contain it'
        )
    if args.demo_augmentation_every_n <= 0:
        raise ValueError('--demo-augmentation-every-n must be positive')
    if args.batch_size <= 0:
        raise ValueError('--batch-size must be positive')
    if args.replay_capacity <= 0:
        raise ValueError('--replay-capacity must be positive')

    tasks = discover_tasks(
        raw_data_dir,
        args.split,
        args.task,
        args.start_episode,
        args.num_demos,
    )
    print(
        f'Discovered {len(tasks)} task(s) under {raw_data_dir}: '
        + ', '.join(task.task for task in tasks),
        flush=True,
    )
    for task in tasks:
        print(
            f'  {task.task}: raw={task.episode_root} episodes='
            f'{task.episode_indices[0]}-{task.episode_indices[-1]} '
            f'count={len(task.episode_indices)}',
            flush=True,
        )
    if args.dry_run:
        print('Dry run complete; no replay files were written.', flush=True)
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    backend = _load_backend()
    device = _resolve_device(backend.torch, args.device)
    print(
        f'Loading CLIP {args.clip_model!r} on {device} for language features ...',
        flush=True,
    )
    clip_model, _ = backend.clip.load(args.clip_model, device='cpu')
    clip_model = clip_model.to(device)
    clip_model.eval()

    total = 0
    try:
        for task in tasks:
            total += generate_task_replay(
                backend,
                task,
                output_root,
                clip_model,
                device,
                batch_size=args.batch_size,
                replay_capacity=args.replay_capacity,
                demo_augmentation=not args.no_demo_augmentation,
                demo_augmentation_every_n=args.demo_augmentation_every_n,
                overwrite=args.overwrite,
                skip_existing=args.skip_existing,
            )
    finally:
        del clip_model
        gc.collect()
        if device.startswith('cuda'):
            backend.torch.cuda.empty_cache()
    print(
        f'Done: {len(tasks)} task(s), {total} replay files in {output_root}',
        flush=True,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
