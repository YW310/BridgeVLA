import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.generate_rlbench_replay import (
    ReplayBackend,
    TaskEpisodes,
    _parse_task_values,
    _paths_overlap,
    build_parser,
    discover_task_names,
    discover_tasks,
    generate_task_replay,
    main,
    resolve_task_episode_root,
    select_episode_indices,
    validate_replay_directory,
)


def create_episode_layout(
    root: Path,
    task: str,
    indices,
    *,
    split: str = 'train',
) -> Path:
    episodes = root / split / task / 'all_variations' / 'episodes'
    episodes.mkdir(parents=True)
    for index in indices:
        (episodes / f'episode{index}').mkdir()
    return episodes


class GenerateRLBenchReplayTest(unittest.TestCase):
    def test_parser_defaults_match_bridgevla_generation(self):
        args = build_parser().parse_args(
            ['--raw-data-dir', 'raw', '--output-dir', 'replay']
        )
        self.assertEqual(args.split, 'train')
        self.assertEqual(args.task, [])
        self.assertEqual(args.start_episode, 0)
        self.assertIsNone(args.num_demos)
        self.assertEqual(args.demo_augmentation_every_n, 10)
        self.assertEqual(args.replay_capacity, 300000)
        self.assertEqual(args.clip_model, 'RN50')

    def test_task_values_support_repeated_and_comma_separated_names(self):
        self.assertIsNone(_parse_task_values([]))
        self.assertIsNone(_parse_task_values(['all']))
        self.assertEqual(
            _parse_task_values(['stack_blocks,open_drawer', 'stack_blocks']),
            ('stack_blocks', 'open_drawer'),
        )
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            _parse_task_values(['all', 'stack_blocks'])

    def test_discovers_tasks_from_dataset_root_and_split_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            stack = create_episode_layout(root, 'stack_blocks', range(3))
            create_episode_layout(root, 'open_drawer', range(2))
            self.assertEqual(
                discover_task_names(root, 'train'),
                ('open_drawer', 'stack_blocks'),
            )
            self.assertEqual(
                discover_task_names(root / 'train', 'train'),
                ('open_drawer', 'stack_blocks'),
            )
            self.assertEqual(
                resolve_task_episode_root(root, 'train', 'stack_blocks'),
                stack.resolve(),
            )
            self.assertEqual(
                resolve_task_episode_root(
                    root / 'train', 'train', 'stack_blocks'
                ),
                stack.resolve(),
            )

    def test_selects_only_contiguous_episode_ranges(self):
        self.assertEqual(
            select_episode_indices(range(5), 1, 3),
            (1, 2, 3),
        )
        self.assertEqual(
            select_episode_indices(range(5), 2, None),
            (2, 3, 4),
        )
        with self.assertRaisesRegex(ValueError, 'missing raw episode'):
            select_episode_indices((0, 2), 0, None)
        with self.assertRaisesRegex(ValueError, 'start episode'):
            select_episode_indices((1, 2), 0, None)

    def test_discovers_selected_task_episode_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episodes = create_episode_layout(root, 'stack_blocks', range(5))
            selected = discover_tasks(
                root,
                'train',
                ['stack_blocks'],
                start_episode=1,
                num_demos=2,
            )
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0].task, 'stack_blocks')
            self.assertEqual(selected[0].episode_root, episodes.resolve())
            self.assertEqual(selected[0].episode_indices, (1, 2))

    def test_input_and_output_paths_must_be_disjoint(self):
        root = Path('dataset').resolve()
        self.assertTrue(_paths_overlap(root, root))
        self.assertTrue(_paths_overlap(root, root / 'replay'))
        self.assertFalse(_paths_overlap(root, root.parent / 'replay'))

    def test_validates_minimal_numeric_replay_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ordinary = {
                'terminal': 0,
                'episode_idx': 0,
                'sample_frame': 0,
                'next_keypoint_frame': 10,
            }
            sentinel = {'terminal': -1}
            for index, transition in enumerate((ordinary, sentinel)):
                with (root / f'{index}.replay').open('wb') as stream:
                    pickle.dump(transition, stream)
            with (root / 'replay_info.npy').open('wb') as stream:
                np.save(stream, np.array([0, -1], dtype=np.int8))
            self.assertEqual(validate_replay_directory(root), 2)

    def test_generation_uses_temporary_directory_then_publishes_task(self):
        calls = {}

        def create_replay(**kwargs):
            calls['create'] = kwargs
            return object()

        def fill_replay(**kwargs):
            calls['fill'] = kwargs
            output = Path(kwargs['task_replay_storage_folder'])
            output.mkdir(parents=True)
            transitions = (
                {
                    'terminal': 1,
                    'episode_idx': 2,
                    'sample_frame': 0,
                    'next_keypoint_frame': 10,
                },
                {'terminal': -1},
            )
            for index, transition in enumerate(transitions):
                with (output / f'{index}.replay').open('wb') as stream:
                    pickle.dump(transition, stream)
            with (output / 'replay_info.npy').open('wb') as stream:
                np.save(stream, np.array([1, -1], dtype=np.int8))

        backend = ReplayBackend(
            torch=None,
            clip=None,
            create_replay=create_replay,
            fill_replay=fill_replay,
            cameras=('front', 'left_shoulder', 'right_shoulder', 'wrist'),
            scene_bounds=(-0.3, -0.5, 0.6, 0.7, 0.5, 1.6),
            voxel_sizes=(100,),
            episode_folder='episode%d',
            variation_descriptions_pkl='variation_descriptions.pkl',
            rotation_resolution=5,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / 'episodes'
            raw.mkdir()
            output = root / 'replay'
            output.mkdir()
            count = generate_task_replay(
                backend,
                TaskEpisodes('stack_blocks', raw, (2, 3)),
                output,
                clip_model=object(),
                device='cpu',
                batch_size=1,
                replay_capacity=100,
                demo_augmentation=True,
                demo_augmentation_every_n=10,
                overwrite=False,
                skip_existing=False,
            )
            task_output = output / 'stack_blocks'
            self.assertEqual(count, 2)
            self.assertTrue(task_output.is_dir())
            self.assertFalse(
                (output / '.stack_blocks.raw_to_replay.tmp').exists()
            )
            self.assertEqual(calls['fill']['start_idx'], 2)
            self.assertEqual(calls['fill']['num_demos'], 2)
            self.assertEqual(
                calls['fill']['task_replay_storage_folder'],
                str(output / '.stack_blocks.raw_to_replay.tmp'),
            )
            self.assertFalse(calls['create']['use_oracle_objects'])

    def test_dry_run_does_not_create_output_or_load_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            create_episode_layout(root, 'stack_blocks', range(2))
            output = root.parent / f'{root.name}_replay'
            self.assertEqual(
                main(
                    [
                        '--raw-data-dir',
                        str(root),
                        '--output-dir',
                        str(output),
                        '--task',
                        'stack_blocks',
                        '--num-demos',
                        '2',
                        '--dry-run',
                    ]
                ),
                0,
            )
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
