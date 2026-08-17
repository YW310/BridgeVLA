import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools.augment_replay_with_oracle_objects import (
    ORACLE_KEYS,
    REPLAY_METADATA_CACHE_NAME,
    OracleFrameCache,
    atomic_write_replay,
    augment_transition,
    decode_depth_image,
    decode_mask_image,
    extract_oracle_objects,
    build_parser,
    load_frame_rgb_images,
    point_cloud_from_depth_and_camera_params,
    visualize_oracle_objects,
    _bounded_thread_map,
    _select_dry_run_files,
    _select_visualization_files,
    _scene_points_for_visualization,
    _final_observation_oracle_for_visualization,
    _instance_color,
    _instance_boxes_for_mask,
    _episode_detection_sources,
    _episode_ids_for_selected_files,
    _detect_task_relevant_handles,
    _load_current_gripper_states,
    _limit_episode_candidates,
    _open_gripper_prefix,
    _REPLAY_METADATA_MEMORY_CACHE,
    _resolve_task_cache_directory,
    _select_adaptive_robot_frames,
)
from tools.rlbench_task_handle_detector import TaskHandleDetection


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
    def test_cached_task_detection_returns_slots_roles_and_groups(self):
        detection = TaskHandleDetection(
            episode_idx=3,
            task_handles=(10, 11),
            observed_handles=(10, 11),
            interaction_handles=(10,),
            adjacent_handles=(11,),
            rejected_dynamic_handles=(),
            background_handles=(),
            sampled_frames=(0, 5),
            target_handles=(10, 11),
            reference_handles=(),
            object_groups=((10, 11),),
        )
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            (cache_dir / 'episode_0003.json').touch()
            with (
                patch(
                    'tools.augment_replay_with_oracle_objects.'
                    'load_task_handle_detection',
                    return_value=detection,
                ),
                patch(
                    'tools.augment_replay_with_oracle_objects.'
                    '_episode_detection_sources',
                    return_value={3: []},
                ),
            ):
                slots, roles, groups = _detect_task_relevant_handles(
                    'stack_blocks',
                    [],
                    Path('raw'),
                    CAMERAS,
                    (0,),
                    {},
                    16,
                    cache_dir,
                    False,
                    False,
                    None,
                    None,
                    0.60,
                    show_progress=False,
                )
        self.assertEqual(slots, {3: (10,)})
        self.assertEqual(roles, {3: {10: 1}})
        self.assertEqual(groups, {3: {10: 10, 11: 10}})

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
                '--filter-thin-planes',
                '--thin-plane-max-thickness',
                '0.004',
                '--thin-plane-min-extent',
                '0.10',
            ]
        )
        self.assertTrue(args.filter_thin_planes)
        self.assertAlmostEqual(args.thin_plane_max_thickness, 0.004)
        self.assertAlmostEqual(args.thin_plane_min_extent, 0.10)
        args = parser.parse_args(
            base
            + [
                '--temporal-task-filter',
                '--task-detection-frames',
                '24',
            ]
        )
        self.assertTrue(args.temporal_task_filter)
        self.assertEqual(args.task_detection_frames, 24)
        args = parser.parse_args(base + ['--temporal-id-matching'])
        self.assertTrue(args.temporal_task_filter)
        args = parser.parse_args(
            base
            + [
                '--detect-robot-handles',
                '--robot-detection-frames',
                '6',
                '--robot-detection-stride',
                '4',
                '--robot-detection-window',
                '80',
                '--robot-motion-threshold',
                '0.03',
                '--robot-link-motion-threshold',
                '0.002',
                '--robot-adjacency-distance',
                '0.08',
                '--refresh-replay-metadata-cache',
            ]
        )
        self.assertTrue(args.detect_robot_handles)
        self.assertEqual(args.robot_detection_frames, 6)
        self.assertEqual(args.robot_detection_stride, 4)
        self.assertEqual(args.robot_detection_window, 80)
        self.assertAlmostEqual(args.robot_motion_threshold, 0.03)
        self.assertAlmostEqual(args.robot_link_motion_threshold, 0.002)
        self.assertAlmostEqual(args.robot_adjacency_distance, 0.08)
        self.assertTrue(args.refresh_replay_metadata_cache)

    def test_robot_sampling_defaults_cover_raw_frames_zero_through_100(self):
        args = build_parser().parse_args(
            ['--replay-dir', 'replay', '--raw-data-dir', 'raw']
        )
        self.assertEqual(args.robot_detection_frames, 64)
        self.assertEqual(args.robot_detection_stride, 5)
        self.assertEqual(args.robot_detection_window, 100)
        self.assertAlmostEqual(args.robot_motion_threshold, 0.02)
        self.assertAlmostEqual(args.robot_link_motion_threshold, 0.008)
        self.assertAlmostEqual(args.robot_adjacency_distance, 0.05)

    def test_handle_cache_defaults_follow_output_and_explicit_path_wins(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / 'oracle_output'
            self.assertEqual(
                _resolve_task_cache_directory(
                    None, output_dir, 'stack_blocks', 'robot_handle_maps'
                ),
                (output_dir / 'robot_handle_maps').resolve(),
            )
            explicit = root / 'custom_robot_cache'
            self.assertEqual(
                _resolve_task_cache_directory(
                    explicit,
                    output_dir,
                    'stack_blocks',
                    'robot_handle_maps',
                ),
                (explicit / 'stack_blocks').resolve(),
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

    def test_robot_sampling_keeps_only_early_sequence(self):
        candidates = list(range(20))
        selected = _limit_episode_candidates(
            candidates, limit=8, strategy='early'
        )
        self.assertEqual(selected, list(range(8)))
        self.assertEqual(
            _limit_episode_candidates(
                candidates, limit=4, strategy='uniform'
            ),
            [0, 6, 12, 19],
        )

    def test_robot_sampling_stops_at_first_gripper_close(self):
        frame_sources = [
            (index, Path(f'{index}.replay')) for index in range(5)
        ]
        selected = _open_gripper_prefix(
            frame_sources,
            {0: 1.0, 1: 1.0, 2: 0.0, 3: 1.0, 4: 1.0},
        )
        self.assertEqual(
            [sample_frame for sample_frame, _ in selected], [0, 1]
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

    def test_load_current_gripper_states_keeps_frame_alignment(self):
        observations = [
            SimpleNamespace(
                gripper_pose=[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
                gripper_open=1.0,
            ),
            SimpleNamespace(
                gripper_pose=[0.4, 0.5, 0.6, 0.0, 0.0, 0.0, 1.0],
                gripper_open=0.0,
            ),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            episode_dir = Path(temporary)
            with (episode_dir / 'low_dim_obs.pkl').open('wb') as stream:
                pickle.dump(observations, stream)
            positions, openings = _load_current_gripper_states(
                episode_dir, [1, 0]
            )

        np.testing.assert_allclose(positions[0], [0.1, 0.2, 0.3])
        np.testing.assert_allclose(positions[1], [0.4, 0.5, 0.6])
        self.assertEqual(openings, {1: 0.0, 0: 1.0})

    def test_adaptive_robot_sampling_extends_static_initial_window(self):
        observations = [
            SimpleNamespace(
                gripper_pose=[
                    max(0, frame - 100) * 0.001,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ],
                gripper_open=1.0,
            )
            for frame in range(151)
        ]
        selected = _select_adaptive_robot_frames(
            observations,
            stride=5,
            initial_window=100,
            max_frames=64,
            motion_threshold=0.02,
        )
        self.assertEqual(selected.sample_frames[:3], (0, 5, 10))
        self.assertEqual(selected.sample_frames[-1], 120)
        self.assertTrue(selected.motion_sufficient)
        self.assertAlmostEqual(selected.max_gripper_displacement, 0.02)

    def test_adaptive_robot_sampling_never_crosses_first_close(self):
        observations = [
            SimpleNamespace(
                gripper_pose=[frame * 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                gripper_open=0.0 if frame >= 43 else 1.0,
            )
            for frame in range(100)
        ]
        selected = _select_adaptive_robot_frames(observations)
        self.assertEqual(selected.sample_frames, tuple(range(0, 43, 5)))
        self.assertTrue(selected.stopped_on_close)

    def test_adaptive_robot_sampling_reports_static_gripper(self):
        observations = [
            SimpleNamespace(
                gripper_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                gripper_open=1.0,
            )
            for _ in range(150)
        ]
        selected = _select_adaptive_robot_frames(
            observations, max_frames=25
        )
        self.assertEqual(len(selected.sample_frames), 25)
        self.assertEqual(selected.sample_frames[-1], 120)
        self.assertFalse(selected.motion_sufficient)
        self.assertEqual(selected.max_gripper_displacement, 0.0)

    def test_episode_sources_merge_augmented_segments_and_task_action_edges(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            transitions = [
                {
                    'terminal': 0,
                    'episode_idx': 3,
                    'sample_frame': 0,
                    'next_keypoint_frame': 5,
                },
                {
                    'terminal': 1,
                    'episode_idx': 3,
                    'sample_frame': 5,
                    'next_keypoint_frame': 10,
                },
                {'terminal': -1},
                {
                    'terminal': 0,
                    'episode_idx': 3,
                    'sample_frame': 2,
                    'next_keypoint_frame': 5,
                },
                {
                    'terminal': 1,
                    'episode_idx': 3,
                    'sample_frame': 5,
                    'next_keypoint_frame': 12,
                },
                {'terminal': -1},
            ]
            files = []
            for index, transition in enumerate(transitions):
                source = root / f'{index}.replay'
                with source.open('wb') as stream:
                    pickle.dump(transition, stream)
                files.append(source)
            with (root / 'replay_info.npy').open('wb') as stream:
                np.save(
                    stream,
                    np.array([0, 1, -1, 0, 1, -1], dtype=np.int8),
                )

            robot_sources = _episode_detection_sources(
                files,
                frames_per_episode=10,
                show_progress=False,
            )
            self.assertEqual(
                [sample_frame for sample_frame, _ in robot_sources[3]],
                [0, 2, 5],
            )

            task_sources = _episode_detection_sources(
                files,
                frames_per_episode=10,
                show_progress=False,
                preserve_action_edges=True,
            )
            self.assertEqual(
                [sample_frame for sample_frame, _ in task_sources[3]],
                [0, 2, 5, 5],
            )
            self.assertEqual(
                [int(source.stem) for _, source in task_sources[3]],
                [0, 3, 1, 4],
            )

    def test_fallback_scan_preserves_distinct_task_action_edges(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = []
            for index, next_frame in enumerate((10, 12)):
                source = root / f'{index}.replay'
                with source.open('wb') as stream:
                    pickle.dump(
                        {
                            'terminal': 0,
                            'episode_idx': 4,
                            'sample_frame': 5,
                            'next_keypoint_frame': next_frame,
                        },
                        stream,
                    )
                files.append(source)
            selected = _episode_detection_sources(
                files,
                frames_per_episode=10,
                show_progress=False,
                preserve_action_edges=True,
            )
            self.assertEqual(len(selected[4]), 2)

    def test_fallback_metadata_cache_hits_disk_and_invalidates_on_new_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = []
            for index in range(2):
                source = root / f'{index}.replay'
                with source.open('wb') as stream:
                    pickle.dump(
                        {
                            'terminal': index,
                            'episode_idx': 4,
                            'sample_frame': index,
                        },
                        stream,
                    )
                files.append(source)

            first = _episode_detection_sources(
                files, frames_per_episode=10, show_progress=False
            )
            self.assertEqual(
                [frame for frame, _ in first[4]], [0, 1]
            )
            self.assertTrue((root / REPLAY_METADATA_CACHE_NAME).is_file())

            _REPLAY_METADATA_MEMORY_CACHE.clear()
            with patch(
                'tools.augment_replay_with_oracle_objects.pickle.load',
                side_effect=AssertionError('disk cache should avoid pickle'),
            ):
                second = _episode_detection_sources(
                    files, frames_per_episode=10, show_progress=False
                )
            self.assertEqual(
                [frame for frame, _ in second[4]], [0, 1]
            )

            added = root / '2.replay'
            with added.open('wb') as stream:
                pickle.dump(
                    {
                        'terminal': 1,
                        'episode_idx': 4,
                        'sample_frame': 9,
                    },
                    stream,
                )
            files.append(added)
            _REPLAY_METADATA_MEMORY_CACHE.clear()
            rebuilt = _episode_detection_sources(
                files, frames_per_episode=10, show_progress=False
            )
            self.assertEqual(
                [frame for frame, _ in rebuilt[4]], [0, 1, 9]
            )

    def test_fallback_metadata_cache_can_live_in_output_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_dir = root / 'input'
            output_dir = root / 'output'
            source_dir.mkdir()
            source = source_dir / '0.replay'
            with source.open('wb') as stream:
                pickle.dump(
                    {
                        'terminal': 1,
                        'episode_idx': 2,
                        'sample_frame': 7,
                    },
                    stream,
                )

            selected = _episode_detection_sources(
                [source],
                frames_per_episode=10,
                show_progress=False,
                metadata_cache_dir=output_dir,
            )
            self.assertEqual(selected[2][0][0], 7)
            self.assertTrue(
                (output_dir / REPLAY_METADATA_CACHE_NAME).is_file()
            )
            self.assertFalse(
                (source_dir / REPLAY_METADATA_CACHE_NAME).exists()
            )

            _REPLAY_METADATA_MEMORY_CACHE.clear()
            with patch(
                'tools.augment_replay_with_oracle_objects.pickle.load',
                side_effect=AssertionError('output cache should avoid pickle'),
            ):
                cached = _episode_detection_sources(
                    [source],
                    frames_per_episode=10,
                    show_progress=False,
                    metadata_cache_dir=output_dir,
                )
            self.assertEqual(cached[2][0][0], 7)

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

    def test_decodes_rlbench_rgb_depth_fixed_point_values(self):
        encoded = np.array(
            [[[0, 0, 0], [128, 0, 0], [255, 255, 255]]],
            dtype=np.uint8,
        )
        decoded = decode_depth_image(encoded)
        np.testing.assert_allclose(
            decoded,
            [[0.0, 8388608.0 / 16777215.0, 1.0]],
            rtol=1e-7,
        )

    def test_reconstructs_world_point_cloud_from_depth(self):
        cloud = point_cloud_from_depth_and_camera_params(
            np.ones((2, 2), dtype=np.float32),
            np.eye(4, dtype=np.float32),
            np.eye(3, dtype=np.float32),
        )
        np.testing.assert_allclose(
            cloud,
            np.array(
                [
                    [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
                    [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                ],
                dtype=np.float32,
            ),
        )

    def test_loads_all_available_raw_rgb_camera_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            episode = Path(temporary)
            for index, camera in enumerate(CAMERAS):
                folder = episode / f'{camera}_rgb'
                folder.mkdir()
                pixels = np.full((4, 6, 3), index * 40, dtype=np.uint8)
                Image.fromarray(pixels).save(folder / '7.png')
            loaded = load_frame_rgb_images(episode, 7, CAMERAS)
        self.assertEqual(tuple(loaded), CAMERAS)
        self.assertTrue(all(image.shape == (4, 6, 3) for image in loaded.values()))

    def test_instance_boxes_use_retained_handle_pixels(self):
        mask = np.array(
            [
                [0, 5, 5, 0],
                [0, 5, 0, 7],
                [0, 0, 0, 7],
            ],
            dtype=np.int32,
        )
        boxes = _instance_boxes_for_mask(mask, (5, 8))
        self.assertEqual(boxes, {5: (1, 0, 2, 1)})
        grouped_boxes = _instance_boxes_for_mask(
            mask,
            (5,),
            group_by_id={5: 5, 7: 5},
        )
        self.assertEqual(grouped_boxes, {5: (1, 0, 3, 2)})

    def test_visualization_combines_camera_images_and_point_cloud_views(self):
        transition = {'front_point_cloud': point_cloud(0)}
        oracle = extract_oracle_objects(
            transition,
            {'front': np.array([[0, 5, 5], [0, 0, 0]], dtype=np.int32)},
            cameras=('front',),
            max_objects=2,
            num_points=3,
            min_object_points=1,
            rng=np.random.default_rng(0),
        )
        camera_images = {
            camera: np.full((16, 24, 3), index * 40, dtype=np.uint8)
            for index, camera in enumerate(CAMERAS)
        }
        camera_masks = {
            camera: np.pad(
                np.full((8, 12), 5, dtype=np.int32),
                ((4, 4), (6, 6)),
            )
            for camera in CAMERAS
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = visualize_oracle_objects(
                oracle,
                'stack_blocks',
                9,
                Path(temporary),
                episode_idx=2,
                sample_frame=7,
                camera_images=camera_images,
                camera_masks=camera_masks,
            )
            with Image.open(output) as image:
                width, height = image.size
        self.assertGreater(width, height)

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
            role_by_id={5: 1, 7: 2},
            rng=np.random.default_rng(3),
        )

        self.assertEqual(oracle.points.shape, (4, 5, 3))
        self.assertEqual(oracle.centers.shape, (4, 3))
        self.assertEqual(oracle.sizes.shape, (4, 3))
        self.assertEqual(oracle.ids.tolist(), [5, 7, -1, -1])
        self.assertEqual(oracle.valid.tolist(), [True, True, False, False])
        self.assertEqual(oracle.roles.tolist(), [1, 2, 0, 0])
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

    def test_filters_unknown_thin_plane_but_preserves_role_object(self):
        rows, columns = np.meshgrid(
            np.linspace(0.0, 0.1, 4, dtype=np.float32),
            np.linspace(0.0, 0.1, 4, dtype=np.float32),
            indexing='ij',
        )
        cloud = np.stack(
            (columns, rows, np.zeros_like(rows)), axis=-1
        )
        mask = np.full((4, 4), 5, dtype=np.int32)
        filtered = extract_oracle_objects(
            {'front_point_cloud': cloud},
            {'front': mask},
            cameras=('front',),
            max_objects=2,
            num_points=4,
            min_object_points=1,
            filter_thin_planes=True,
            rng=np.random.default_rng(0),
        )
        self.assertFalse(filtered.valid.any())
        self.assertEqual(filtered.thin_plane_objects, 1)
        self.assertEqual(filtered.thin_plane_object_ids, (5,))

        protected = extract_oracle_objects(
            {'front_point_cloud': cloud},
            {'front': mask},
            cameras=('front',),
            max_objects=2,
            num_points=4,
            min_object_points=1,
            filter_thin_planes=True,
            role_by_id={5: 1},
            rng=np.random.default_rng(0),
        )
        self.assertEqual(protected.ids.tolist(), [5, -1])
        self.assertEqual(protected.roles.tolist(), [1, 0])
        self.assertEqual(protected.thin_plane_object_ids, ())

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

    def test_episode_whitelist_reports_temporally_filtered_ids(self):
        transition = {'front_point_cloud': point_cloud(0)}
        oracle = extract_oracle_objects(
            transition,
            {'front': np.array([[0, 5, 5], [7, 7, 0]], dtype=np.int32)},
            cameras=('front',),
            max_objects=4,
            num_points=3,
            excluded_ids=(0,),
            included_ids=(5,),
            min_object_points=1,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(oracle.ids.tolist(), [5, -1, -1, -1])
        self.assertEqual(oracle.temporal_filtered_objects, 1)
        self.assertEqual(oracle.temporal_filtered_object_ids, (7,))

    def test_episode_slots_do_not_filter_and_keep_absent_slot_reserved(self):
        transition = {'front_point_cloud': point_cloud(0)}
        oracle = extract_oracle_objects(
            transition,
            {'front': np.array([[0, 7, 7], [9, 9, 0]], dtype=np.int32)},
            cameras=('front',),
            max_objects=4,
            num_points=3,
            excluded_ids=(0,),
            slot_ids=(5, 7),
            min_object_points=1,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(oracle.ids.tolist(), [-1, 7, 9, -1])
        self.assertEqual(oracle.valid.tolist(), [False, True, True, False])
        self.assertEqual(oracle.temporal_filtered_objects, 0)
        self.assertEqual(oracle.temporal_filtered_object_ids, ())

        next_oracle = extract_oracle_objects(
            transition,
            {'front': np.array([[5, 5, 0], [7, 7, 0]], dtype=np.int32)},
            cameras=('front',),
            max_objects=4,
            num_points=3,
            excluded_ids=(0,),
            slot_ids=(5, 7),
            min_object_points=1,
            rng=np.random.default_rng(1),
        )
        self.assertEqual(next_oracle.ids.tolist(), [5, 7, -1, -1])
        self.assertEqual(next_oracle.valid.tolist(), [True, True, False, False])

    def test_rigid_group_merges_multiple_mask_handles_into_one_slot(self):
        oracle = extract_oracle_objects(
            {'front_point_cloud': point_cloud(0)},
            {
                'front': np.array(
                    [[5, 5, 7], [7, 0, 0]], dtype=np.int32
                )
            },
            cameras=('front',),
            max_objects=4,
            num_points=4,
            excluded_ids=(0,),
            slot_ids=(5, 7),
            role_by_id={5: 1},
            group_by_id={5: 5, 7: 5},
            min_object_points=1,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(oracle.ids.tolist(), [5, -1, -1, -1])
        self.assertEqual(oracle.valid.tolist(), [True, False, False, False])
        self.assertEqual(oracle.roles.tolist(), [1, 0, 0, 0])
        self.assertEqual(oracle.raw_point_counts, (4,))
        np.testing.assert_allclose(
            oracle.centers[0],
            np.array([0.75, 0.25, 1.0], dtype=np.float32),
        )

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
            self.assertEqual(
                reloaded['oracle_object_roles'].tolist(), [0, 0, 0]
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
