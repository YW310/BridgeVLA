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

    def read(self, timestamp):
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
        return best.to_ndarray(format="rgb24")

    def close(self):
        self.container.close()


class EpisodeVideos:
    def __init__(self, dataset, tolerance):
        self.dataset, self.tolerance = dataset, tolerance
        self.readers = {}

    def read(self, reference):
        stamp = np.asarray(reference["Timestamp"], dtype=float).reshape(-1)
        if stamp.size != 1:
            raise ValueError("Each video reference must contain one timestamp")
        path = inside(self.dataset, reference["Path"])
        if path not in self.readers:
            self.readers[path] = VideoReader(path, self.tolerance)
        return self.readers[path].read(stamp[0])

    def close(self):
        for reader in self.readers.values():
            reader.close()


def decode_depth(raw, config):
    encoding = config.get("encoding")
    raw = np.asarray(raw)
    if encoding == "metric":
        if raw.ndim != 2 or not np.issubdtype(raw.dtype, np.floating):
            raise ValueError("metric depth requires a floating HxW array")
        depth = raw.astype(np.float32)
    elif encoding in ("scaled_integer", "linear_channel"):
        if "scale" not in config or float(config["scale"]) <= 0:
            raise ValueError("Depth scale must be explicit and positive")
        if encoding == "linear_channel":
            if raw.ndim != 3 or config.get("channel") not in (0, 1, 2):
                raise ValueError("linear_channel requires an explicit RGB channel")
            raw = raw[..., config["channel"]]
        elif raw.ndim != 2 or not np.issubdtype(raw.dtype, np.integer):
            raise ValueError("scaled_integer requires integer HxW depth")
        depth = raw.astype(np.float32) * float(config["scale"]) + float(config.get("offset", 0))
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
        if config.get("encoding") != "linear_channel":
            raise ValueError("MP4 depth requires documented linear_channel decoding or metric sidecars")
        raw = videos.read(columns[f"observation.depth.{camera}"][frame])
    return decode_depth(raw, config)
