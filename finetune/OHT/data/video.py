"""Explicit frame-index/PTS alignment; numeric depth never goes through RGB."""
from pathlib import Path
import math
import numpy as np
from .common import inside


def video_alignment(config):
    mode = config.get("video_alignment", "timestamp")
    if mode not in ("timestamp", "frame_index"):
        raise ValueError("video_alignment must be timestamp or frame_index")
    return mode


def _frame_array(frame, format, expected_format):
    if expected_format is not None and frame.format.name != expected_format:
        raise ValueError(
            f"Expected video pixel format {expected_format}, got {frame.format.name}; "
            "do not convert numeric depth through RGB"
        )
    if format is None and frame.format.name == "gray12le":
        # Older PyAV cannot export gray12le via to_ndarray(). Preserve native
        # little-endian uint16 words, including removal of decoder row padding.
        plane = frame.planes[0]
        words = np.frombuffer(plane, dtype="<u2").reshape(plane.height, plane.line_size // 2)
        return words[:frame.height, :frame.width].copy()
    return frame.to_ndarray() if format is None else frame.to_ndarray(format=format)


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
        return _frame_array(best, format, expected_format)

    def close(self):
        self.container.close()


class IndexedVideoReader:
    """Stream the same decoded ordinal as the Parquet row, as the reference does.

    Never derive an index from FPS or silently fall back to a nearby PTS.
    Decode the unused tail at validation, so truncated/extra streams cannot
    produce a completed replay even if sampled frames were all present.
    """

    def __init__(self, path, expected_frames):
        if isinstance(expected_frames, (bool, np.bool_)) or not isinstance(expected_frames, (int, np.integer)) or expected_frames < 1:
            raise ValueError("frame_index alignment requires a positive expected_frames count")
        self.path, self.expected_frames = path, int(expected_frames)
        self._open()

    def _open(self):
        import av
        self.container = av.open(str(self.path))
        self.stream = self.container.streams.video[0]
        declared = self.stream.frames
        if declared and declared != self.expected_frames:
            self.container.close()
            raise ValueError(f"Video frame count mismatch {self.path}: video={declared}, parquet={self.expected_frames}")
        self.iterator = iter(self.container.decode(self.stream))
        self.index, self.last, self.exhausted = -1, None, False

    def _advance(self):
        if self.exhausted:
            return False
        try:
            self.last = next(self.iterator)
        except StopIteration:
            self.exhausted = True
            return False
        self.index += 1
        if self.index >= self.expected_frames:
            raise ValueError(f"Video frame count mismatch {self.path}: decoded>{self.expected_frames}, parquet={self.expected_frames}")
        return True

    def read(self, index, *, format="rgb24", expected_format=None):
        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)) or not 0 <= index < self.expected_frames:
            raise ValueError("Video frame_index must be an integer inside the Parquet episode")
        if index < self.index:
            self.close()
            self._open()
        while self.index < index:
            if not self._advance():
                raise ValueError(f"Video frame count mismatch {self.path}: decoded={self.index+1}, parquet={self.expected_frames}")
        return _frame_array(self.last, format, expected_format)

    def validate_length(self):
        while self._advance():
            pass
        if self.index + 1 != self.expected_frames:
            raise ValueError(f"Video frame count mismatch {self.path}: decoded={self.index+1}, parquet={self.expected_frames}")

    def close(self):
        self.container.close()


class EpisodeVideos:
    def __init__(self, dataset, tolerance, *, alignment="timestamp", expected_frames=None):
        self.dataset, self.tolerance = dataset, tolerance
        self.alignment = video_alignment({"video_alignment": alignment})
        if alignment == "frame_index" and (isinstance(expected_frames, (bool, np.bool_)) or
                not isinstance(expected_frames, (int, np.integer)) or expected_frames < 1):
            raise ValueError("frame_index alignment requires a positive expected_frames count")
        self.expected_frames = expected_frames
        self.readers = {}

    def read(self, reference, *, frame=None, format="rgb24", expected_format=None):
        stamp = np.asarray(reference["Timestamp"], dtype=float).reshape(-1)
        if stamp.size != 1 or not np.isfinite(stamp).all() or stamp[0] < 0:
            raise ValueError("Each video reference must contain one finite nonnegative timestamp")
        path = inside(self.dataset, reference["Path"])
        if path not in self.readers:
            self.readers[path] = (IndexedVideoReader(path, self.expected_frames) if self.alignment == "frame_index"
                                  else VideoReader(path, self.tolerance))
        selected = frame if self.alignment == "frame_index" else stamp[0]
        return self.readers[path].read(selected, format=format, expected_format=expected_format)

    def validate_lengths(self):
        if self.alignment == "frame_index":
            for reader in self.readers.values():
                reader.validate_length()

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
        round_to_mm = config.get("round_to_mm", False)
        if not isinstance(round_to_mm, bool):
            raise ValueError("depth.round_to_mm must be boolean")
        # The reference uses float32 before uint16-mm rounding. Retain that
        # arithmetic here so half-millimetre ties match its PNG values.
        normalized = raw.astype(np.float32 if round_to_mm else np.float64) / float(qmax)
        near, far, shift = (float(config[key]) for key in ("depth_min", "depth_max", "shift"))
        if config["use_log"]:
            low, high = math.log(near + shift), math.log(far + shift)
            depth = np.exp(normalized * (high - low) + low) - shift
        else:
            depth = normalized * (far - near) + near
        if round_to_mm:
            depth_mm = np.clip(np.rint(depth * 1000.0), 0, np.iinfo(np.uint16).max).astype(np.uint16)
            depth = depth_mm.astype(np.float32) * .001
        # Non-reference profiles can retain sub-millimetre metric floats.
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
            raw = videos.read(reference, frame=frame, format=None, expected_format=config.get("pixel_format"))
        elif config.get("encoding") == "linear_channel":
            raw = videos.read(reference, frame=frame)
        else:
            raise ValueError(
                "Video depth requires native scaled_integer/quantized grayscale or documented "
                "linear_channel decoding; metric depth requires floating sidecars"
            )
    return decode_depth(raw, config)
