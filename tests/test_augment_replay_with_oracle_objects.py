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
        self.assertEqual(oracle.discovered_objects, 1)

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


if __name__ == '__main__':
    unittest.main()
