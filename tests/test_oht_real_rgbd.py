"""Native real OHT camera geometry and lossless capture-order regressions."""
import av
import numpy as np
import pytest

from finetune.OHT.data.real_rgbd import FixedCamera, NativeDepthReader


def _config(camera, size):
    return dict(image_size=size, camera_quaternion_order="xyzw",
                cameras={camera.name: dict(intrinsics=camera.intrinsics, optical_to_sensor=np.eye(4))},
                depth=dict(kind="z"))


def test_aligned_oht_depth_keeps_color_grid_when_axis_strides_differ():
    native = dict(height=6, width=8, fx=4, fy=6, ppx=2, ppy=3, coeffs=[0.] * 5)
    acquisition = dict(color=native, depth=native, depth_scale_m_per_unit=.001,
                       depth_aligned_to_color=True)
    camera = FixedCamera("camera_2", acquisition, np.eye(4), (2, 4))
    rgb = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    raw = np.full((6, 8), 700, dtype=np.uint16)
    raw[3, 2] = 0

    result = camera.read_observation(rgb, raw, _config(camera, (2, 4)))

    np.testing.assert_array_equal(result["camera_2_rgb"].transpose(1, 2, 0), rgb[::3, ::2])
    np.testing.assert_array_equal(result["camera_2_camera_intrinsics"],
                                  [[2, 0, 1], [0, 2, 1], [0, 0, 1]])
    np.testing.assert_allclose(result["camera_2_point_cloud"][:, 0, 0], [-.35, -.35, .7])
    assert np.isnan(result["camera_2_point_cloud"][:, 1, 1]).all()


def test_realsense_samples_distorted_color_and_keeps_native_depth_pose():
    depth = dict(height=8, width=8, fx=4, fy=4, ppx=4, ppy=4, coeffs=[0.] * 5)
    color = dict(height=12, width=12, fx=8, fy=8, ppx=4.08, ppy=4,
                 model="distortion.inverse_brown_conrady", coeffs=[-.42, 0., .09, -.07, 0.])
    # Rz(+90 degrees) stored in SDK column-major order.
    extrinsics = dict(rotation=[0, 1, 0, -1, 0, 0, 0, 0, 1], translation_m=[.25, 0., 0.])
    acquisition = dict(color=color, depth=depth, depth_scale_m_per_unit=.002,
                       depth_aligned_to_color=False, depth_to_color=extrinsics)
    base_from_color = np.eye(4)
    base_from_color[:3, 3] = [1., -.5, .125]
    camera = FixedCamera("camera_0", acquisition, base_from_color, (4, 4))
    rgb = np.arange(12 * 12 * 3, dtype=np.uint16).astype(np.uint8).reshape(12, 12, 3)
    raw = np.zeros((8, 8), dtype=np.uint16)
    raw[2, 6] = 500
    raw[0, 0] = 50  # This ray projects outside the color image.

    result = camera.read_observation(rgb, raw, _config(camera, (4, 4)))

    # Native point (.5,-.5,1) maps to (.75,.5,1) in color optical axes.
    # The SDK distortion/rounding puts it at column 8, row 7.
    np.testing.assert_array_equal(result["camera_0_rgb"][:, 1, 3], rgb[7, 8])
    np.testing.assert_allclose(result["camera_0_point_cloud"][:, 1, 3], [1.75, 0., 1.125], atol=1e-7)
    assert result["camera_0_depth"][0, 1, 3] == 1.
    assert np.isnan(result["camera_0_depth"][0, 0, 0])


def test_intra_ffv1_depth_skips_packets_without_losing_capture_index(tmp_path):
    path = tmp_path / "depth.mkv"
    height, width = 48, 70
    raw = np.arange(height * width, dtype=np.uint16).reshape(height, width)
    frames = [raw + offset * 8000 for offset in range(8)]
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("ffv1", rate=30)
        stream.height, stream.width, stream.pix_fmt = height, width, "gray16le"
        stream.codec_context.gop_size = 1
        for pixels in frames:
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="gray16le")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    reader = NativeDepthReader(path)
    try:
        with pytest.raises(ValueError, match="integer depth capture index"):
            reader.read(1.5)
        for index in (1, 1, 4, 7):
            np.testing.assert_array_equal(reader.read(index), frames[index])
        with pytest.raises(ValueError, match="nondecreasing"):
            reader.read(6)
        with pytest.raises(IndexError, match="no depth frame"):
            reader.read(8)
    finally:
        reader.close()
