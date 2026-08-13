import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools.augment_replay_with_oracle_objects import (
    ORACLE_KEYS,
    OracleFrameCache,
    atomic_write_replay,
    augment_transition,
    decode_mask_image,
    extract_oracle_objects,
    build_parser,
    _bounded_thread_map,
    _select_dry_run_files,
    _select_visualization_files,
    _scene_points_for_visualization,
    _final_observation_oracle_for_visualization,
    _instance_color,
    _episode_detection_sources,
    _episode_ids_for_selected_files,
)


CAMERAS = ('front', 'left_shoulder', 'right_shoulder', 'wrist')


def point_cloud(offset):
    rows, columns = np.meshgrid(
        np.arange(2, dtype=np.float32),
        np.arange(3, dtype=np.float32),
        indexing='ij',
    )
    cloud = np.stack(
        (columns + offset, rows, np.ones_like(rows)), axis=-1
    )
    return np.moveaxis(cloud, -1, 0)


class OracleReplayAugmentationTest(unittest.TestCase):
    def test_visualization_output_directory_defaults_and_override(self):
        parser = build_parser()
        base = [
            '--replay-dir',
            'replay',
            '--raw-data-dir',
            'raw',
        ]
        args = parser.parse_args(base)
        self.assertEqual(
            args.visualize_output_dir, Path('oracle_visualizations')
        )
        args = parser.parse_args(
            base + ['--visualize-output-dir', 'custom_visualizations']
        )
        self.assertEqual(
            args.visualize_output_dir, Path('custom_visualizations')
        )
        args = parser.parse_args(base + ['--visualize-objects-only'])
        self.assertTrue(args.visualize_objects_only)
        self.assertFalse(args.task_prior_strict)
        args = parser.parse_args(base + ['--task-prior-strict'])
        self.assertTrue(args.task_prior_strict)
        args = parser.parse_args(
            base
            + [
                '--detect-robot-handles',
                '--robot-detection-frames',
                '6',
            ]
        )
        self.assertTrue(args.detect_robot_handles)
        self.assertEqual(args.robot_detection_frames, 6)

    def test_interval_visualization_selects_every_nth_sorted_file(self):
        files = [Path(f'{index}.replay') for index in range(10)]
        selected = _select_visualization_files(
            files,
            visualize_index=None,
            visualize_every=3,
        )
        self.assertEqual(
            [path.name for path in selected],
            ['0.replay', '3.replay', '6.replay', '9.replay'],
        )

    def test_robot_detection_uses_replay_info_and_requested_episodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transitions = [
                {'terminal': 0, 'episode_idx': 0, 'sample_frame': 0},
                {'terminal': 1, 'episode_idx': 0, 'sample_frame': 5},
                {'terminal': -1},
                {'terminal': 0, 'episode_idx': 1, 'sample_frame': 2},
                {'terminal': 1, 'episode_idx': 1, 'sample_frame': 8},
                {'terminal': -1},
            ]
            files = []
            for index, transition in enumerate(transitions):
                source = root / f'{index}.replay'
                with source.open('wb') as stream:
                    pickle.dump(transition, stream)
                files.append(source)
            with (root / 'replay_info.npy').open('wb') as stream:
                np.save(stream, np.array([0, 1, -1, 0, 1, -1], dtype=np.int8))

            selected = _episode_detection_sources(
                files,
                frames_per_episode=1,
                requested_episode_ids=(1,),
                show_progress=False,
            )
            self.assertEqual(list(selected), [1])
            self.assertEqual(selected[1][0][0], 2)

            cached = _episode_detection_sources(
                files,
                frames_per_episode=2,
                requested_episode_ids=(1,),
                cached_episode_ids=(1,),
                show_progress=False,
            )
            self.assertEqual(cached, {1: []})

    def test_dry_run_episode_ids_recover_final_sentinel_from_previous(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            previous = root / '8.replay'
            sentinel = root / '9.replay'
            with previous.open('wb') as stream:
                pickle.dump({'terminal': 1, 'episode_idx': 4}, stream)
            with sentinel.open('wb') as stream:
                pickle.dump({'terminal': -1}, stream)
            episode_ids = _episode_ids_for_selected_files(
                [sentinel], {9: previous}
            )
            self.assertEqual(episode_ids, (4,))

    def test_scene_visualization_points_keep_finite_nonzero_geometry(self):
        transition = {
            'front_point_cloud': point_cloud(0),
            'left_shoulder_point_cloud': np.zeros((3, 2, 3), dtype=np.float32),
        }
        transition['front_point_cloud'][:, 0, 0] = np.nan
        points = _scene_points_for_visualization(
            transition, ('front', 'left_shoulder'), max_points=3
        )
        self.assertEqual(points.shape, (3, 3))
        self.assertTrue(np.isfinite(points).all())
        self.assertTrue(np.any(points != 0, axis=1).all())

    def test_instance_color_is_stable_by_handle_id(self):
        self.assertEqual(_instance_color(42), _instance_color(42))
        self.assertNotEqual(_instance_color(42), _instance_color(43))

    def test_frame_cache_single_flight_reuses_one_result(self):
        cache = OracleFrameCache(capacity=2)
        calls = []
        expected = object()

        def compute():
            calls.append(True)
            return expected

        results = list(
            _bounded_thread_map(
                lambda _: cache.get_or_compute(
                    ('stack_blocks', 2, 4), compute
                ),
                [Path(str(index)) for index in range(20)],
                workers=4,
            )
        )
        self.assertEqual(len(calls), 1)
        self.assertTrue(all(result is expected for result in results))
        self.assertEqual(cache.stats(), (19, 1, 1))

    def test_bounded_thread_map_processes_every_item(self):
        items = [Path(str(index)) for index in range(20)]
        results = list(
            _bounded_thread_map(
                lambda item: int(item.name) ** 2,
                items,
                workers=4,
            )
        )
        self.assertEqual(
            sorted(results), [index ** 2 for index in range(20)]
        )

    def test_rgb_mask_decoder_receives_writable_normalized_input(self):
        encoded = np.array([[[0, 0, 0], [1, 0, 0]]], dtype=np.uint8)

        def decoder(rgb):
            self.assertTrue(rgb.flags.writeable)
            self.assertEqual(rgb.dtype, np.float32)
            self.assertGreaterEqual(float(rgb.min()), 0.0)
            self.assertLessEqual(float(rgb.max()), 1.0)
            rgb *= 255
            rgb = rgb.astype(np.int64)
            return (
                rgb[..., 0]
                + rgb[..., 1] * 256
                + rgb[..., 2] * 256 * 256
            )

        decoded = decode_mask_image(encoded, decoder=decoder)
        self.assertEqual(decoded.tolist(), [[0, 1]])

    def test_dry_run_stops_after_enough_regular_transitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            files = []
            for index in range(100):
                replay = Path(temporary) / f'{index}.replay'
                replay.touch()
                files.append(replay)
            with patch(
                'tools.augment_replay_with_oracle_objects.pickle.load',
                return_value={'terminal': np.int8(0)},
            ) as load:
                selected = _select_dry_run_files(
                    files, seed=4, sample_count=5, visualize_index=None
                )
            self.assertEqual(len(selected), 5)
            self.assertEqual(load.call_count, 5)

    def test_extract_merges_views_filters_invalid_and_pads(self):
        transition = {
            f'{camera}_point_cloud': point_cloud(index)
            for index, camera in enumerate(CAMERAS)
        }
        transition['front_point_cloud'][:, 0, 1] = np.nan
        masks = {
            'front': np.array([[0, 5, 5], [7, 7, 0]], dtype=np.int32),
            'left_shoulder': np.array(
                [[0, 5, 0], [0, 0, 0]], dtype=np.int32
            ),
            'right_shoulder': np.zeros((2, 3), dtype=np.int32),
            'wrist': np.zeros((2, 3), dtype=np.int32),
        }

        oracle = extract_oracle_objects(
            transition,
            masks,
            cameras=CAMERAS,
            max_objects=4,
            num_points=5,
            min_object_points=1,
            rng=np.random.default_rng(3),
        )

        self.assertEqual(oracle.points.shape, (4, 5, 3))
        self.assertEqual(oracle.centers.shape, (4, 3))
        self.assertEqual(oracle.sizes.shape, (4, 3))
        self.assertEqual(oracle.ids.tolist(), [5, 7, -1, -1])
        self.assertEqual(oracle.valid.tolist(), [True, True, False, False])
        self.assertEqual(oracle.raw_point_counts, (2, 2))
        self.assertEqual(oracle.excluded_object_ids, (0,))
        np.testing.assert_array_equal(
            oracle.sizes[2:], np.zeros((2, 3), dtype=np.float32)
        )
        self.assertTrue(np.isfinite(oracle.points).all())

    def test_filters_instances_below_minimum_fused_point_count(self):
        transition = {
            f'{camera}_point_cloud': point_cloud(index)
            for index, camera in enumerate(CAMERAS)
        }
        masks = {
            'front': np.array([[0, 5, 5], [7, 0, 0]], dtype=np.int32),
            'left_shoulder': np.zeros((2, 3), dtype=np.int32),
            'right_shoulder': np.zeros((2, 3), dtype=np.int32),
            'wrist': np.zeros((2, 3), dtype=np.int32),
        }
        oracle = extract_oracle_objects(
            transition,
            masks,
            cameras=CAMERAS,
            max_objects=4,
            num_points=5,
            min_object_points=2,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(oracle.ids.tolist(), [5, -1, -1, -1])
        self.assertEqual(oracle.filtered_objects, 1)
        self.assertEqual(oracle.small_object_ids, (7,))
        self.assertEqual(oracle.discovered_objects, 1)

    def test_reports_mask_instance_with_no_finite_point_cloud(self):
        cloud = point_cloud(0)
        cloud[:, 0, 1] = np.nan
        oracle = extract_oracle_objects(
            {'front_point_cloud': cloud},
            {'front': np.array([[0, 9, 0], [0, 0, 0]], dtype=np.int32)},
            cameras=('front',),
            max_objects=4,
            num_points=3,
            min_object_points=1,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(oracle.no_finite_point_object_ids, (9,))
        self.assertFalse(oracle.valid.any())

    def test_task_prior_filter_runs_during_oracle_extraction(self):
        transition = {'front_point_cloud': point_cloud(0)}
        masks = {
            'front': np.array([[0, 5, 5], [7, 7, 0]], dtype=np.int32)
        }
        oracle = extract_oracle_objects(
            transition,
            masks,
            cameras=('front',),
            max_objects=4,
            num_points=3,
            min_object_points=1,
            task_name='stack_blocks',
            task_prior_filter=True,
            action_position=np.array([1.5, 0.0, 1.0]),
            task_prior_radius=0.2,
            task_prior_strict=True,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(oracle.ids.tolist(), [5, -1, -1, -1])
        self.assertEqual(oracle.prior_filtered_objects, 1)
        self.assertEqual(oracle.prior_filtered_object_ids, (7,))

    def test_alignment_and_atomic_round_trip_preserve_original_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = (
                root
                / 'raw'
                / 'stack_blocks'
                / 'all_variations'
                / 'episodes'
                / 'episode2'
            )
            mask = np.array([[0, 11, 11], [0, 0, 0]], dtype=np.uint8)
            for camera in CAMERAS:
                folder = episode / f'{camera}_mask'
                folder.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask).save(folder / '4.png')

            original = {
                'terminal': np.int8(0),
                'episode_idx': 2,
                'sample_frame': 4,
                'keypoint_frame': 1,
                'lang_goal': np.array(['stack the blocks'], dtype=object),
            }
            original.update(
                {
                    f'{camera}_point_cloud': point_cloud(index)
                    for index, camera in enumerate(CAMERAS)
                }
            )

            migrated, oracle, episode_dir = augment_transition(
                original,
                root / 'raw',
                'stack_blocks',
                replay_index=9,
                cameras=CAMERAS,
                max_objects=3,
                num_points=4,
                excluded_ids=(0,),
                seed=8,
                min_object_points=1,
            )
            _, repeated_oracle, _ = augment_transition(
                original,
                root / 'raw',
                'stack_blocks',
                replay_index=999,
                cameras=CAMERAS,
                max_objects=3,
                num_points=4,
                excluded_ids=(0,),
                seed=8,
                min_object_points=1,
            )
            output = root / 'output' / '9.replay'
            atomic_write_replay(output, original, migrated, oracle)

            self.assertEqual(episode_dir, episode)
            np.testing.assert_array_equal(
                repeated_oracle.points, oracle.points
            )
            with output.open('rb') as stream:
                reloaded = pickle.load(stream)
            self.assertTrue(set(original).issubset(reloaded))
            self.assertTrue(set(ORACLE_KEYS).issubset(reloaded))
            np.testing.assert_array_equal(
                reloaded['lang_goal'], original['lang_goal']
            )
            self.assertEqual(
                reloaded['oracle_object_ids'].tolist(), [11, -1, -1]
            )
            self.assertEqual(
                reloaded['oracle_object_sizes'].shape, (3, 3)
            )
            self.assertFalse(Path(f'{output}.tmp').exists())

    def test_final_sentinel_skips_uninitialized_alignment_metadata(self):
        original = {
            'terminal': np.int8(-1),
            'episode_idx': np.empty((), dtype=int),
            'sample_frame': np.empty((), dtype=int),
        }
        migrated, oracle, episode_dir = augment_transition(
            original,
            Path('does-not-exist'),
            'stack_blocks',
            replay_index=10,
            cameras=CAMERAS,
            max_objects=2,
            num_points=3,
            excluded_ids=(0,),
            seed=0,
            min_object_points=1,
        )
        self.assertIsNone(episode_dir)
        self.assertFalse(oracle.valid.any())
        self.assertTrue(set(ORACLE_KEYS).issubset(migrated))

    def test_final_visualization_recovers_mask_from_previous_transition(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episode = (
                root
                / 'raw'
                / 'stack_blocks'
                / 'all_variations'
                / 'episodes'
                / 'episode2'
            )
            mask = np.array([[0, 11, 11], [0, 0, 0]], dtype=np.uint8)
            for camera in CAMERAS:
                folder = episode / f'{camera}_mask'
                folder.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask).save(folder / '5.png')
            previous_source = root / '9.replay'
            with previous_source.open('wb') as stream:
                pickle.dump(
                    {
                        'terminal': np.int8(1),
                        'episode_idx': 2,
                        'next_keypoint_frame': 5,
                    },
                    stream,
                )
            final_transition = {
                'terminal': np.int8(-1),
                **{
                    f'{camera}_point_cloud': point_cloud(index)
                    for index, camera in enumerate(CAMERAS)
                },
            }
            recovered = _final_observation_oracle_for_visualization(
                final_transition,
                previous_source,
                root / 'raw',
                'stack_blocks',
                CAMERAS,
                max_objects=3,
                num_points=4,
                excluded_ids=(0,),
                seed=8,
                min_object_points=1,
            )
            self.assertIsNotNone(recovered)
            oracle, episode_idx, sample_frame = recovered
            self.assertEqual(oracle.ids.tolist(), [11, -1, -1])
            self.assertEqual(episode_idx, 2)
            self.assertEqual(sample_frame, 5)


if __name__ == '__main__':
    unittest.main()
