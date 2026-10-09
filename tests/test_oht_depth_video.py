"""Real lossless HEVC gray12le round-trip and numeric-depth routing."""
from pathlib import Path

import av
import numpy as np
import pytest
import yaml

from finetune.OHT.data.video import VideoReader, EpisodeVideos, metric_depth
from finetune.OHT.data.observation import camera_observation


def write_gray12_video(path, samples):
    height, width = samples[0].shape
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("libx265", rate=60)
        stream.width, stream.height, stream.pix_fmt = width, height, "gray12le"
        stream.options = {"preset": "ultrafast", "x265-params": "lossless=1:log-level=error:pools=none:frame-threads=1"}
        for sample in samples:
            frame = av.VideoFrame(width, height, "gray12le")
            plane = frame.planes[0]
            padded = np.zeros((plane.height, plane.line_size // 2), dtype="<u2")
            padded[:height, :width] = sample
            plane.update(padded.tobytes())
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


@pytest.fixture(scope="module")
def depth_movie(tmp_path_factory):
    path = tmp_path_factory.mktemp("oht_gray12") / "depth.mp4"
    # Width 66 exercises FFmpeg row padding, not just tightly packed planes.
    height, width = 64, 66
    samples = [(np.arange(height * width).reshape(height, width) + i * 137).astype(np.uint16) % 4096
               for i in range(3)]
    write_gray12_video(path, samples)
    return path, samples


def test_native_gray12_roundtrip_forward_backward_and_pts(depth_movie):
    path, samples = depth_movie
    reader = VideoReader(path)
    try:
        for index in (0, 2, 0, 1, 2):
            raw = reader.read(index / 60, format=None, expected_format="gray12le")
            assert raw.dtype == np.uint16
            np.testing.assert_array_equal(raw, samples[index])
        with pytest.raises(ValueError, match="unmatched"):
            reader.read(5, format=None)
    finally:
        reader.close()


def test_native_gray12_rejects_wrong_source_format(depth_movie):
    reader = VideoReader(depth_movie[0])
    try:
        with pytest.raises(ValueError, match="Expected video pixel format gray16le, got gray12le"):
            reader.read(0, format=None, expected_format="gray16le")
    finally:
        reader.close()


def test_rgb_source_rejected_for_gray12_but_linear_channel_remains_supported(tmp_path):
    path = tmp_path / "rgb.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=60)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        pixels = np.full((64, 64, 3), [17, 83, 201], dtype=np.uint8)
        for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    reader = VideoReader(path)
    try:
        raw = reader.read(0)
        with pytest.raises(ValueError, match="Expected video pixel format gray12le"):
            reader.read(0, format=None, expected_format="gray12le")
    finally:
        reader.close()
    videos = EpisodeVideos(tmp_path, .01)
    columns = {"observation.depth.wrist": [{"Path": path.name, "Timestamp": [0]}]}
    try:
        depth = metric_depth(tmp_path, {}, columns, "wrist", 0, videos,
                             {"encoding": "linear_channel", "channel": 1, "scale": .005, "offset": .2})
    finally:
        videos.close()
    np.testing.assert_allclose(depth, raw[..., 1].astype(np.float64) * .005 + .2)


def test_parquet_reference_to_metric_ray_cloud(depth_movie):
    path, samples = depth_movie
    config_path = Path(__file__).resolve().parents[1] / "finetune/OHT/configs/dataset.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["depth"] = dict(encoding="scaled_integer", pixel_format="gray12le", scale=.001,
                           kind="ray", invalid_values=[0, 4095], limits=[.001, 4.094])
    videos = EpisodeVideos(path.parent, 1 / 120 + .0001)
    columns = {"observation.depth.wrist": [{"Path": path.name, "Timestamp": [i / 60]} for i in range(3)]}
    try:
        depth = metric_depth(path.parent, {}, columns, "wrist", 0, videos, config["depth"])
    finally:
        videos.close()
    expected = (samples[0].astype(np.float64) * .001).astype(np.float32)
    expected[(samples[0] == 0) | (samples[0] == 4095)] = np.nan
    np.testing.assert_allclose(depth, expected, equal_nan=True)
    assert np.isnan(depth).sum() == ((samples[0] == 0) | (samples[0] == 4095)).sum()
    config["image_size"] = [32, 33]
    config["cameras"]["wrist"]["intrinsics"] = [[32, 0, 32], [0, 32, 32], [0, 0, 1]]
    # A rotated world pose catches incorrect order of the optical-axis conversion.
    q = np.sqrt(.5)
    obs = camera_observation("wrist", np.zeros((64, 66, 3), np.uint8), depth,
                             [1, 2, 3, q, 0, 0, q], config)
    cloud = obs["wrist_point_cloud"].transpose(1, 2, 0)
    world_rays = cloud - np.array([1, 2, 3])
    np.testing.assert_allclose(np.linalg.norm(world_rays, axis=-1), depth[::2, ::2],
                               atol=1e-6, equal_nan=True)
    assert np.isfinite(cloud[samples[0][::2, ::2] == 4094]).all()
    # Camera centre ray [0,0,+depth] must point along USD -Z even after world Z rotation.
    np.testing.assert_allclose(cloud[16, 16], [1, 2, 3-depth[32, 32]], atol=1e-6)
