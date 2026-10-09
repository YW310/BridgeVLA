"""Cross-camera rotations and raw-frame pairing, independent of action labels."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from finetune.OHT.data.geometry import camera_pose_matrix
from finetune.OHT.data.observation import camera_observation, validate_data_config
from finetune.OHT.data.video import EpisodeVideos, IndexedVideoReader, decode_depth
from tests.test_oht_depth_video import write_gray12_video


CONFIG = Path(__file__).resolve().parents[1] / "finetune/OHT/configs/dataset.yaml"


@pytest.mark.parametrize("order", ["wxyz", "xyzw"])
@pytest.mark.parametrize("direction", ["camera_to_world", "world_to_camera"])
@pytest.mark.parametrize("sensor_frame", ["opengl", "optical"])
@pytest.mark.parametrize("kind", ["z", "ray"])
def test_differently_rotated_cameras_reconstruct_the_same_world_plane(order, direction, sensor_frame, kind):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config.update(camera_quaternion_order=order, camera_extrinsic_direction=direction, image_size=[11, 13])
    config["point_cloud_filter"]["enabled"] = False
    config["depth"]["kind"] = kind
    K = np.array([[14., 0, 6], [0, 15., 5], [0, 0, 1.]])
    axis_flip = np.diag([1., -1., -1., 1.]) if sensor_frame == "opengl" else np.eye(4)
    normal = np.array([.25, -.15, 1.])
    normal /= np.linalg.norm(normal)
    v, u = np.indices((11, 13))
    rays = np.stack(((u - 6) / 14., (v - 5) / 15., np.ones_like(u)), axis=-1)
    if kind == "ray":
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    for index, camera in enumerate(config["cameras"]):
        # Independent optical->world ground truth, with noncommuting rotations.
        optical = np.eye(4)
        optical[:3, :3] = Rotation.from_euler("xyz", [171 + index * 3, -13 + index * 5, 19 + index * 11], degrees=True).as_matrix()
        optical[:3, 3] = [.1 + index * .07, -.2 + index * .04, 1.6]
        source = optical @ axis_flip
        if direction == "world_to_camera":
            source = np.linalg.inv(source)
        raw_q = Rotation.from_matrix(source[:3, :3]).as_quat()
        if order == "wxyz":
            raw_q = raw_q[[3, 0, 1, 2]]
        raw_pose = np.r_[source[:3, 3], raw_q]
        world_rays = rays @ optical[:3, :3].T
        depth = (.4 - optical[:3, 3] @ normal) / (world_rays @ normal)
        assert (depth > 0).all()
        config["cameras"][camera].update(intrinsics=K.tolist(), optical_to_sensor=axis_flip.tolist())
        observation = camera_observation(camera, np.zeros((11, 13, 3), np.uint8), depth, raw_pose, config)
        np.testing.assert_allclose(observation[f"{camera}_camera_extrinsics"], optical, atol=1e-6)
        cloud = observation[f"{camera}_point_cloud"].reshape(3, -1).T
        np.testing.assert_allclose(cloud @ normal, .4, atol=2e-7)
        _, _, vh = np.linalg.svd(cloud - cloud.mean(axis=0), full_matrices=False)
        assert abs(vh[-1] @ normal) > .999999  # Same surface orientation in every camera.


def test_reference_axis_flip_is_on_the_right_and_inverse_includes_translation():
    source = np.eye(4)
    source[:3, :3] = Rotation.from_euler("xyz", [23, -41, 67], degrees=True).as_matrix()
    source[:3, 3] = [.4, -.2, 1.6]
    flip = np.diag([1., -1., -1., 1.])
    raw_pose = np.r_[source[:3, 3], Rotation.from_matrix(source[:3, :3]).as_quat()[[3, 0, 1, 2]]]
    actual = camera_pose_matrix(raw_pose, "wxyz", "world_to_camera", flip)
    np.testing.assert_allclose(actual, np.linalg.inv(source) @ flip, atol=1e-12)
    assert not np.allclose(actual, flip @ np.linalg.inv(source))
    assert not np.allclose(actual[:3, 3], source[:3, 3])


def test_log_decoding_mm_rays_creates_cross_camera_surface_misalignment():
    # A depth decoder mismatch can look like a rotation error despite correct
    # extrinsics. This synthetic repro does not establish the server's encoding.
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["point_cloud_filter"]["enabled"] = False
    config["image_size"] = [11, 13]
    K = np.array([[14., 0, 6], [0, 15., 5], [0, 0, 1.]])
    normal = np.array([.25, -.15, 1.])
    normal /= np.linalg.norm(normal)
    v, u = np.indices((11, 13))
    rays = np.stack(((u - 6) / 14., (v - 5) / 15., np.ones_like(u)), axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    wrong_spec = dict(encoding="quantized", depth_min=.01, depth_max=10.,
                      shift=3.5, use_log=True, qmax=4095, invalid_values=[0],
                      kind="ray", limits=[.001, 10.])
    wrong_normals = []
    for index, camera in enumerate(config["cameras"]):
        optical = np.eye(4)
        optical[:3, :3] = Rotation.from_euler(
            "xyz", [171 + index * 3, -13 + index * 5, 19 + index * 11], degrees=True).as_matrix()
        optical[:3, 3] = [.1 + index * .07, -.2 + index * .04, 1.6]
        source = optical @ np.diag([1., -1., -1., 1.])
        raw_pose = np.r_[source[:3, 3], Rotation.from_matrix(source[:3, :3]).as_quat()[[3, 0, 1, 2]]]
        ray_distance = (.4 - optical[:3, 3] @ normal) / ((rays @ optical[:3, :3].T) @ normal)
        raw = np.rint(ray_distance * 1000).astype(np.uint16)
        assert ((raw > 0) & (raw < 4095)).all()
        config["cameras"][camera]["intrinsics"] = K.tolist()
        for spec, correct in ((config["depth"], True), (wrong_spec, False)):
            observation = camera_observation(camera, np.zeros((11, 13, 3), np.uint8),
                                             decode_depth(raw, spec), raw_pose, dict(config, depth=spec))
            np.testing.assert_allclose(observation[f"{camera}_camera_extrinsics"], optical, atol=1e-6)
            cloud = observation[f"{camera}_point_cloud"].reshape(3, -1).T
            if correct:
                np.testing.assert_allclose(cloud @ normal, .4, atol=.00051)
            else:
                _, _, vh = np.linalg.svd(cloud - cloud.mean(axis=0), full_matrices=False)
                wrong_normals.append(vh[-1])
    # The wrong depth inverse changes apparent surface orientation differently
    # across cameras; a single compensating world rotation cannot align them.
    angles = [np.degrees(np.arccos(np.clip(abs(first @ second), 0, 1)))
              for first in wrong_normals for second in wrong_normals]
    assert max(angles) > 1.


def test_invalid_direction_and_alignment_are_rejected():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for field in ("camera_extrinsic_direction", "video_alignment"):
        bad = deepcopy(config)
        bad[field] = "guess"
        with pytest.raises(ValueError, match=field):
            validate_data_config(bad)


def test_frame_index_selects_exact_raw_frame_not_shifted_reference_pts(tmp_path):
    path = tmp_path / "depth.mp4"
    samples = [np.full((64, 66), 900 + i * 83, np.uint16) for i in range(3)]
    write_gray12_video(path, samples)
    videos = EpisodeVideos(tmp_path, .009, alignment="frame_index", expected_frames=3)
    try:
        for index in (0, 2, 0, 1, 1):
            reference = dict(Path=path.name, Timestamp=[(index + 1) / 60])
            actual = videos.read(reference, frame=index, format=None, expected_format="gray12le")
            np.testing.assert_array_equal(actual, samples[index])
        videos.validate_lengths()
        videos.validate_lengths()  # Idempotent, even after a backward request.
        with pytest.raises(ValueError, match="frame_index"):
            videos.read(dict(Path=path.name, Timestamp=[0]), format=None)
    finally:
        videos.close()
    # Timestamp mode stays explicit and continues to pick the referenced PTS.
    legacy = EpisodeVideos(tmp_path, .009)
    try:
        np.testing.assert_array_equal(legacy.read(dict(Path=path.name, Timestamp=[1 / 60]), format=None), samples[1])
    finally:
        legacy.close()


@pytest.mark.parametrize("actual_count", [2, 4])
def test_indexed_reader_rejects_wrong_declared_stream_count(tmp_path, actual_count):
    path = tmp_path / "wrong-length.mp4"
    write_gray12_video(path, [np.ones((64, 64), np.uint16)] * actual_count)
    with pytest.raises(ValueError, match="frame count mismatch"):
        IndexedVideoReader(path, expected_frames=3)


@pytest.mark.parametrize("actual_count", [2, 4])
def test_unknown_stream_count_is_checked_by_decoding_the_unused_tail(monkeypatch, actual_count):
    class Container:
        streams = SimpleNamespace(video=[SimpleNamespace(frames=0)])

        def decode(self, stream):
            for _ in range(actual_count):
                yield av.VideoFrame.from_ndarray(np.zeros((2, 2, 3), np.uint8), format="rgb24")

        def close(self):
            pass

    monkeypatch.setattr(av, "open", lambda path: Container())
    reader = IndexedVideoReader("unused.mp4", expected_frames=3)
    try:
        assert reader.read(0).shape == (2, 2, 3)
        with pytest.raises(ValueError, match="frame count mismatch"):
            reader.validate_length()
    finally:
        reader.close()


@pytest.mark.parametrize("bad_count", [None, 0, -1, 3.5, True])
def test_frame_index_requires_an_explicit_episode_length(bad_count):
    with pytest.raises(ValueError, match="expected_frames"):
        EpisodeVideos(".", .009, alignment="frame_index", expected_frames=bad_count)
