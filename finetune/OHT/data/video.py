"""Timestamp-aligned decoding; metric depth always needs an explicit encoding."""
from pathlib import Path
import numpy as np
from .common import inside


class VideoReader:
    def __init__(self, path, tolerance=1 / 120 + 1e-4):
        import av
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.tolerance = tolerance
        self.iterator = iter(self.container.decode(self.stream))
        self.last = None
        self.before = None
        self.previous_request = -np.inf

    def read(self, timestamp, *, format="rgb24", expected_format=None):
        timestamp = float(timestamp)
        if not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Invalid video timestamp")
        if timestamp < self.previous_request:
            self.container.seek(0, stream=self.stream)
            self.iterator = iter(self.container.decode(self.stream))
            self.last = None
            self.before = None
        self.previous_request = timestamp
        best, best_error = None, float("inf")
        for candidate in (self.before, self.last):
            if candidate is not None:
                error = abs(float(candidate.pts * candidate.time_base) - timestamp)
                if error < best_error:
                    best, best_error = candidate, error
        while True:
            if self.last is not None and float(self.last.pts * self.last.time_base) >= timestamp:
                break
            try:
                current = next(self.iterator)
            except StopIteration:
                break
            if current.pts is None:
                raise ValueError("Video frame has no PTS")
            self.before, self.last = self.last, current
            error = abs(float(current.pts * current.time_base) - timestamp)
            if error < best_error:
                best, best_error = current, error
        if best is None or best_error > self.tolerance:
            raise ValueError(f"Video timestamp {timestamp} unmatched; error={best_error}")
        if expected_format is not None and best.format.name != expected_format:
            raise ValueError(
                f"Expected video pixel format {expected_format}, got {best.format.name}; "
                "do not convert numeric depth through RGB"
            )
        if format is None and best.format.name == "gray12le":
            # Older PyAV versions cannot export gray12le via to_ndarray().
            # FFmpeg stores each 12-bit sample in a little-endian uint16 word;
            # preserve those words and remove row padding without reformatting.
            plane = best.planes[0]
            words = np.frombuffer(plane, dtype="<u2").reshape(plane.height, plane.line_size // 2)
            return words[:best.height, :best.width].copy()
        return best.to_ndarray() if format is None else best.to_ndarray(format=format)

    def close(self):
        self.container.close()


class EpisodeVideos:
    def __init__(self, dataset, tolerance):
        self.dataset, self.tolerance = dataset, tolerance
        self.readers = {}

    def read(self, reference, *, format="rgb24", expected_format=None):
        stamp = np.asarray(reference["Timestamp"], dtype=float).reshape(-1)
        if stamp.size != 1:
            raise ValueError("Each video reference must contain one timestamp")
        path = inside(self.dataset, reference["Path"])
        if path not in self.readers:
            self.readers[path] = VideoReader(path, self.tolerance)
        return self.readers[path].read(stamp[0], format=format, expected_format=expected_format)

    def close(self):
        for reader in self.readers.values():
            reader.close()


def validate_quantization(config):
    """Validate the writer's fixed 12-bit linear/log quantization contract."""
    values = [float(config[key]) for key in ("depth_min", "depth_max", "shift", "qmax")]
    near, far, shift, qmax = values
    if not np.isfinite(values).all() or not (0 < near < far) or shift < 0:
        raise ValueError("Invalid depth quantization range/shift")
    if qmax != int(qmax) or not 0 < qmax <= 65535 or not isinstance(config.get("use_log"), bool):
        raise ValueError("Depth quantization requires integer qmax and boolean use_log")


def decode_depth(raw, config):
    encoding = config.get("encoding")
    raw = np.asarray(raw)
    if encoding == "metric":
        if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.floating):
            raise ValueError("metric depth requires a floating HxW array")
        depth = raw.astype(np.float32)
    elif encoding == "quantized":
        validate_quantization(config)
        if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("quantized depth requires integer HxW depth")
        qmax = int(config["qmax"])
        if np.any(raw < 0) or np.any(raw > qmax):
            raise ValueError(f"Quantized depth outside [0,{qmax}]")
        normalized = raw.astype(np.float64) / qmax
        near, far, shift = (float(config[key]) for key in ("depth_min", "depth_max", "shift"))
        if config["use_log"]:
            low, high = np.log(near + shift), np.log(far + shift)
            depth = np.exp(normalized * (high - low) + low) - shift
        else:
            depth = normalized * (far - near) + near
        # Keep metric floats; the reference TFDS converter additionally rounds
        # to uint16 millimetres for PNG storage, which our NPZ does not need.
        depth = depth.astype(np.float32)
        depth[raw == 0] = np.nan  # Reserved by the quantizing writer.
    elif encoding in ("scaled_integer", "linear_channel"):
        if "scale" not in config or float(config["scale"]) <= 0:
            raise ValueError("Depth scale must be explicit and positive")
        if encoding == "linear_channel":
            if raw.ndim != 3 or config.get("channel") not in (0, 1, 2):
                raise ValueError("linear_channel requires an explicit RGB channel")
            raw = raw[..., config["channel"]]
        elif raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("scaled_integer requires integer HxW depth")
        # Apply the exact configured scale before the final float32 rounding;
        # float32 multiplication can otherwise push valid 4094 mm above 4.094 m.
        depth = (raw.astype(np.float64) * float(config["scale"]) + float(config.get("offset", 0))).astype(np.float32)
    else:
        raise ValueError("Unknown depth encoding; inspect depth writer before building replay")
    for value in config.get("invalid_values", []):
        depth[raw == value] = np.nan
    return depth


def metric_depth(root, record, columns, camera, frame, videos, config):
    pattern = config.get("path_pattern")
    if pattern:
        path = inside(root, pattern.format(task=record["task"], episode=record["episode_index"],
                                           camera=camera, frame=frame))
        if path.suffix == ".npy":
            raw = np.load(path, allow_pickle=False)
        elif path.suffix.lower() in (".png", ".tif", ".tiff"):
            from PIL import Image
            with Image.open(path) as image:
                raw = np.asarray(image)
        else:
            raise ValueError("Depth sidecars must be .npy/.png/.tif/.tiff")
    else:
        reference = columns[f"observation.depth.{camera}"][frame]
        if config.get("encoding") in ("scaled_integer", "quantized"):
            raw = videos.read(reference, format=None, expected_format=config.get("pixel_format"))
        elif config.get("encoding") == "linear_channel":
            raw = videos.read(reference)
        else:
            raise ValueError(
                "Video depth requires native scaled_integer/quantized grayscale or documented "
                "linear_channel decoding; metric depth requires floating sidecars"
            )
    return decode_depth(raw, config)
