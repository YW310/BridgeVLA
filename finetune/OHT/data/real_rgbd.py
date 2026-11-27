"""Native RGB-D decoding and fixed-camera geometry for the raw OHT export."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import transform_matrix, validate_intrinsics
from .observation import camera_observation


FIXED_CAMERAS = ("camera_0", "camera_2")


def _sdk_intrinsics(geometry):
    return validate_intrinsics(np.array([
        [geometry["fx"], 0., geometry["ppx"]],
        [0., geometry["fy"], geometry["ppy"]],
        [0., 0., 1.],
    ]))


def _project_realsense_color(points, K, coeffs):
    # The SDK's inverse Brown-Conrady color projection applies radial scaling
    # before the tangential terms. See rs2_project_point_to_pixel in librealsense.
    x = points[:, 0] / points[:, 2]
    y = points[:, 1] / points[:, 2]
    r2 = x * x + y * y
    k1, k2, p1, p2, k3 = coeffs
    radial = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    x, y = x * radial, y * radial
    u = x + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    v = y + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)
    return np.rint(u * K[0, 0] + K[0, 2]), np.rint(v * K[1, 1] + K[1, 2])


class FixedCamera:
    """A calibrated depth grid and its corresponding RGB colors in UR Base."""

    def __init__(self, camera, acquisition, base_from_color, image_size):
        self.name = camera
        depth = acquisition["depth"]
        color = acquisition["color"]
        self.depth_shape = depth["height"], depth["width"]
        self.color_shape = color["height"], color["width"]
        height, width = image_size
        dh, dw = self.depth_shape
        if height < 2 or width < 2 or dh % height or dw % width:
            raise ValueError(f"{camera}: image_size must divide native depth dimensions")
        self.sy, self.sx = dh // height, dw // width
        self.scale = float(acquisition["depth_scale_m_per_unit"])
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError(f"{camera}: invalid metric depth scale")
        if np.any(np.asarray(depth["coeffs"], dtype=float) != 0):
            raise ValueError(f"{camera}: distorted native depth is unsupported")
        if acquisition.get("depth_invalid_value", 0) != 0:
            raise ValueError(f"{camera}: expected zero as native invalid depth")
        K = _sdk_intrinsics(depth)
        K[0] /= self.sx
        K[1] /= self.sy
        self.intrinsics = K.tolist()
        color_K = _sdk_intrinsics(color)
        self.aligned = acquisition["depth_aligned_to_color"] is True
        if self.aligned:
            native_depth_K = _sdk_intrinsics(depth)
            if (self.depth_shape != self.color_shape or np.any(np.asarray(color["coeffs"]) != 0)
                    or not np.allclose(native_depth_K, color_K)):
                raise ValueError(f"{camera}: inconsistent aligned RGB-D intrinsics")
            base_from_depth = base_from_color
        else:
            if color["model"] != "distortion.inverse_brown_conrady":
                raise ValueError(f"{camera}: unsupported native color distortion model")
            extrinsics = acquisition["depth_to_color"]
            color_from_depth = np.eye(4)
            # rs2_extrinsics.rotation stores the SDK matrix in column-major order.
            color_from_depth[:3, :3] = np.asarray(extrinsics["rotation"]).reshape(3, 3, order="F")
            color_from_depth[:3, 3] = extrinsics["translation_m"]
            transform_matrix(color_from_depth, f"{camera} depth-to-color transform")
            base_from_depth = base_from_color @ color_from_depth
            v, u = np.indices((height, width))
            rays = np.stack(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1],
                             np.ones_like(u)), axis=-1)
            self.color_rays = rays @ color_from_depth[:3, :3].T
            self.color_offset = color_from_depth[:3, 3]
            self.color_intrinsics = color_K
            self.color_coeffs = np.asarray(color["coeffs"], dtype=float)
            if self.color_coeffs.shape != (5,) or not np.isfinite(self.color_coeffs).all():
                raise ValueError(f"{camera}: invalid RealSense color distortion")
        self.pose = np.r_[base_from_depth[:3, 3],
                          Rotation.from_matrix(base_from_depth[:3, :3]).as_quat()]

    def read_observation(self, rgb, raw_depth, config):
        """Sample native depth and put the associated colors on its output grid."""
        if (rgb.shape != (*self.color_shape, 3) or rgb.dtype != np.uint8
                or raw_depth.shape != self.depth_shape or raw_depth.dtype != np.uint16):
            raise ValueError(f"{self.name}: media format differs from acquisition calibration")
        sampled = raw_depth[::self.sy, ::self.sx]
        depth = sampled.astype(np.float32) * self.scale
        depth[sampled == 0] = np.nan
        if self.aligned:
            colors = rgb[::self.sy, ::self.sx]
        else:
            # Keep depth in its native pinhole grid, then project into the
            # distorted color image to fetch RGB. Mask points outside its FOV.
            rows, cols = np.nonzero(sampled)
            points = self.color_rays[rows, cols] * depth[rows, cols, None] + self.color_offset
            front = points[:, 2] > 0
            u, v = _project_realsense_color(points[front], self.color_intrinsics, self.color_coeffs)
            height, width = self.color_shape
            in_image = (u >= 0) & (u < width) & (v >= 0) & (v < height)
            rows, cols = rows[front][in_image], cols[front][in_image]
            colors = np.zeros((*depth.shape, 3), dtype=np.uint8)
            colors[rows, cols] = rgb[v[in_image].astype(np.intp), u[in_image].astype(np.intp)]
            registered = np.zeros(depth.shape, dtype=bool)
            registered[rows, cols] = True
            depth[~registered] = np.nan
        return camera_observation(self.name, colors, depth, self.pose, config)


def load_camera(dataset, episode_index, camera, image_size):
    """Build one fixed camera from its acquisition and mounting calibration."""
    if camera not in FIXED_CAMERAS:
        raise ValueError(f"{camera}: expected a calibrated fixed camera")
    dataset = Path(dataset)
    acquisition_path = dataset / "provenance" / f"episode_{episode_index:06d}" / "camera" / camera
    acquisition = json.loads((acquisition_path / "calibration.json").read_text(encoding="utf-8"))
    profile = json.loads((dataset / "calibration/current" / f"{camera}.json").read_text(encoding="utf-8"))
    extrinsics = profile["extrinsics"]
    if (profile["physical_camera"] != camera or extrinsics["target_frame"] != "UR Base"
            or extrinsics["units"] != "m" or extrinsics["source_frame"] != f"{profile['role']}_optical"
            or tuple(profile["resolution_wh"]) != (acquisition["color"]["width"], acquisition["color"]["height"])):
        raise ValueError(f"{camera}: expected a matching optical-to-UR Base mounting transform")
    base_from_color = transform_matrix(extrinsics["matrix"], f"{camera} to UR Base")
    return FixedCamera(camera, acquisition, base_from_color, image_size)


class NativeDepthReader:
    """Read increasing capture ordinals from the archive's 30 Hz intra FFV1."""

    def __init__(self, path):
        import av

        self.path = path
        self.container = av.open(str(path))
        try:
            self.stream, = self.container.streams.video
            if self.stream.codec_context.name != "ffv1" or self.stream.average_rate != 30:
                raise ValueError(f"{path}: expected FFV1 on the 30 Hz storage clock")
            self.stream.codec_context.thread_count = 1
            self.pts_to_index = float(self.stream.time_base) * 30
            self.packets = iter(self.container.demux(self.stream))
        except Exception:
            self.container.close()
            raise
        self.packet_index = -1
        self.index = -1
        self.last = None

    def read(self, index):
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise ValueError(f"{self.path}: expected an integer depth capture index")
        if index < 0 or index < self.index:
            raise ValueError(f"{self.path}: depth capture indices must be nonnegative and nondecreasing")
        if index == self.index:
            return self.last
        # FFV1 GOP=1 permits skipping packets without expanding unrequested depth.
        for packet in self.packets:
            if packet.pts is None:
                break
            ordinal = round(packet.pts * self.pts_to_index)
            if ordinal != self.packet_index + 1 or not packet.is_keyframe:
                raise ValueError(f"{self.path}: unexpected FFV1 capture clock or non-intra packet at {ordinal}")
            self.packet_index = ordinal
            if ordinal < index:
                continue
            frame, = packet.decode()
            if frame.pts != packet.pts or frame.format.name != "gray16le":
                raise ValueError(f"{self.path}: expected one native gray16le depth frame at {index}")
            data = frame.to_ndarray(format="gray16le")
            if data.ndim != 2 or data.dtype != np.uint16:
                raise ValueError(f"{self.path}: expected native two-dimensional uint16 depth")
            data.setflags(write=False)
            self.index, self.last = index, data
            return data
        raise IndexError(f"{self.path}: no depth frame at capture index {index}")

    def close(self):
        self.container.close()
        self.last = None
