"""Bounded, label-free CPU exports from an already completed policy forward."""
from collections import deque
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from queue import Empty, Queue
import shutil
from threading import Event, Lock, Semaphore, Thread
import uuid

import numpy as np
from PIL import Image, ImageDraw

from ..action_modes import action_descriptor, action_options, apply_eef_delta, validate_action
from ..run_directory import RUN_TIMEZONE


STAGES = ('mvt1', 'mvt2')
ARRAY_FIELDS = ('rgb', 'heatmap', 'world_to_pixel', 'target_prior', 'reference_prior', 'relation_anchor')


def _inside(pixel, width, height):
    return bool(np.isfinite(pixel).all() and 0 <= pixel[0] < width and 0 <= pixel[1] < height)


def _blend(rgb, heatmap):
    """Match the existing inference viewer's blue/yellow map without importing torch."""
    low, high = heatmap.min(), heatmap.max()
    value = (heatmap - low) / (high - low) if high > low else np.zeros_like(heatmap)
    if value.shape != rgb.shape[:2]:
        resampling = getattr(Image, 'Resampling', Image)
        value = np.asarray(Image.fromarray(np.asarray(value, np.float32)).resize(
            (rgb.shape[1], rgb.shape[0]), resampling.BILINEAR), np.float32)
    colors = np.stack((np.clip(1.5 * value, 0, 1), np.clip(1.5 * value - .35, 0, 1),
                       np.clip(1 - 1.5 * value, 0, 1)), axis=-1)
    return .3 * np.clip(rgb, 0, 1) + .7 * colors


def _montage(stage, name, auxiliary):
    rgb, maps = stage['rgb'], stage['heatmap']
    views, height, width, _ = rgb.shape
    extra = [key for key in ('target_prior', 'reference_prior', 'relation_anchor') if key in stage]
    columns = ['Rendered RGB', 'Prediction heatmap', 'Policy / heatmap TCP'] + extra
    left, top, footer = 65, 78, 18
    image = Image.new('RGB', (left + width * len(columns), top + views * (height + footer)), '#232323')
    draw = ImageDraw.Draw(image)
    draw.text((8, 6), f'{name}: online policy prediction', fill='white')
    draw.text((8, 24), 'Red: policy TCP   Cyan: heatmap TCP' + (' (auxiliary head)' if auxiliary else ''), fill='white')
    draw.text((8, 42), 'Per-view color normalization. Original probabilities are retained in NPZ.', fill='white')
    for column, title in enumerate(columns):
        draw.text((left + width * column + 2, top - 18), title, fill='white')
    for view in range(views):
        y = top + view * (height + footer)
        draw.text((4, y + 4), f'View {view}', fill='white')
        blend = _blend(rgb[view], maps[view])
        tiles = [rgb[view], blend, blend] + [_blend(rgb[view], stage[key][view]) for key in extra]
        for column, tile in enumerate(tiles):
            panel = Image.fromarray((np.clip(tile, 0, 1) * 255).astype(np.uint8))
            if column == 2:
                marks = ImageDraw.Draw(panel)
                for key, color in (('heatmap_waypoint_pixels', 'cyan'), ('policy_pixels', 'red')):
                    if key not in stage or not _inside(stage[key][view], width, height):
                        continue
                    x, py = map(float, stage[key][view])
                    marks.line((x - 4, py - 4, x + 4, py + 4), fill=color, width=2)
                    marks.line((x - 4, py + 4, x + 4, py - 4), fill=color, width=2)
            image.paste(panel, (left + column * width, y))
    return image


class OnlineVisualizer:
    """Keep at most two CPU export jobs, dropping excess work instead of delaying control."""

    def __init__(self, output_dir, every=10, keep=100):
        if type(every) is not int or every < 1 or type(keep) is not int or keep < 1:
            raise ValueError('Online visualization every and keep must be positive integers')
        self.every, self.keep = every, keep
        base = Path(output_dir).expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(RUN_TIMEZONE).strftime('%Y%m%dT%H%M%S%z')
        self.root = base / f'run-{stamp}-{uuid.uuid4().hex}'
        self.root.mkdir()
        self._queue, self._slots = Queue(maxsize=2), Semaphore(2)
        self._lock, self._closing = Lock(), Event()
        self._latest, self._error, self._dropped = None, None, 0
        self._writing, self._entries = False, deque()
        self._thread = Thread(target=self._run, name='OHTOnlineVisualization', daemon=True)
        self._thread.start()

    def due(self, step):
        return isinstance(step, (int, np.integer)) and step >= 0 and int(step) % self.every == 0

    def submit(self, payload, request, action, options, observation, *, sensor_observation=None):
        if self._closing.is_set():
            return dict(status='dropped', reason='closed')
        if not self._slots.acquire(blocking=False):
            with self._lock:
                self._dropped += 1
            return dict(status='dropped', reason='writer_busy')
        try:
            # Only inference-side arrays enter the writer. Never copy labels, teachers,
            # raw request tensors or caller-selected paths into files or metadata.
            stages = payload.get('stages', {})
            if not stages or any(name not in STAGES for name in stages):
                raise ValueError('Expected mvt1/mvt2 visualization stages')
            snapshot = dict(stages={name: {key: np.asarray(source[key]).copy()
                                          for key in ARRAY_FIELDS if key in source}
                                    for name, source in stages.items()},
                            heatmap_waypoint_world=np.asarray(payload['heatmap_waypoint_world']).copy())
            if 'chunk' in payload:
                from .chunk_protocol import validate_prediction
                chunk = payload['chunk']
                validated = validate_prediction(chunk, chunk['spec'])
                snapshot['chunk'] = dict(spec=dict(chunk['spec']), actions=validated['actions'].copy(),
                                         key_action=validated['key_action'].copy())
            identity = {key: request.get(key) for key in
                        ('request_id', 'episode_id', 'step', 'timestamp', 'instruction', 'model_id')}
            raw_observation = request.get('observation', {})
            identity['task'] = raw_observation.get('task') if isinstance(raw_observation, dict) else None
            if sensor_observation is None:
                from .sensor_visualization import prepared_sensor_snapshot
                sensor_observation = prepared_sensor_snapshot(observation)
            if sensor_observation is not None:
                # An export job owns only current sensor fields, never labels
                # or arbitrary metadata supplied alongside an observation.
                allowed = ('rgb', 'depth', 'intrinsics', 'world_from_optical', 'depth_kind', 'depth_limits')
                sensor_observation = dict(
                    source=sensor_observation['source'],
                    cameras={name: {key: deepcopy(camera[key]) for key in allowed if key in camera}
                             for name, camera in sensor_observation['cameras'].items()},
                    **{key: deepcopy(sensor_observation[key]) for key in ('tcp_pose_xyzw',)
                       if key in sensor_observation})
            current_tcp = observation.get('tcp_pose')
            job = (uuid.uuid4().hex, snapshot, identity, np.asarray(action).copy(), action_options(options),
                   None if current_tcp is None else np.asarray(current_tcp).copy(), sensor_observation)
            with self._lock:
                if self._closing.is_set():
                    self._slots.release()
                    return dict(status='dropped', reason='closed')
                self._queue.put_nowait(job)
            return dict(status='queued', entry_id=job[0],
                        url=f'/visualization/files/{job[0]}/prediction.json')
        except Exception as exc:
            self._slots.release()
            with self._lock:
                self._error = f'{type(exc).__name__}: {exc}'
            return dict(status='error', error=str(exc))

    def latest(self):
        with self._lock:
            return dict(enabled=True, every=self.every, keep=self.keep, latest=deepcopy(self._latest),
                        writer_status='error' if self._error else 'writing' if self._writing else 'idle',
                        dropped=self._dropped, error=self._error)

    def _remove(self, directory):
        # Every recursive delete is restricted to a generated direct child of .root.
        directory = directory.resolve()
        name = directory.name.removeprefix('.pending-')
        if directory.parent != self.root or len(name) != 32 or any(c not in '0123456789abcdef' for c in name):
            raise ValueError('Refusing to remove a path outside the visualization run')
        if directory.exists():
            shutil.rmtree(directory)

    def _write(self, entry_id, payload, request, action, options, current_tcp, sensor_observation=None):
        action = validate_action(action, options)
        mode = options['mode']
        policy_pose = (action[:7] if mode == 'eef' else
                       apply_eef_delta(current_tcp, action[:6], options) if mode == 'relative_eef' else None)
        policy_tcp = None if policy_pose is None else policy_pose[:3]
        waypoint = np.asarray(payload['heatmap_waypoint_world'], np.float64)
        if waypoint.shape != (3,) or not np.isfinite(waypoint).all():
            raise ValueError('Expected a finite heatmap waypoint in world coordinates')
        folder, temporary = self.root / entry_id, self.root / f'.pending-{entry_id}'
        temporary.mkdir()
        try:
            record = dict(entry_id=entry_id, **request, label_free=True, action_spec=action_descriptor(options),
                          predicted_action=action.tolist(), policy_tcp_world=None if policy_tcp is None else policy_tcp.tolist(),
                          policy_pose_xyzw=None if policy_pose is None else policy_pose.tolist(),
                          heatmap_waypoint_world=waypoint.tolist(), heatmap_is_auxiliary=mode != 'eef' or 'chunk' in payload,
                          prediction_url=f'/visualization/files/{entry_id}/prediction.json', images={}, stages={})
            if 'chunk' in payload:
                from ..chunk_visualization import path_panel
                chunk = payload['chunk']
                path_panel(chunk['actions']).save(temporary / 'chunk.png')
                np.savez_compressed(temporary / 'chunk.npz', actions=chunk['actions'], key_action=chunk['key_action'])
                record['chunk'] = dict(spec=chunk['spec'], actions=chunk['actions'].tolist(),
                    key_action=chunk['key_action'].tolist(), png=f'/visualization/files/{entry_id}/chunk.png',
                    arrays=f'/visualization/files/{entry_id}/chunk.npz')
                record['images']['chunk'] = record['chunk']['png']
            for name, source in payload['stages'].items():
                stage = dict(source)
                rgb, maps, projection = stage['rgb'], stage['heatmap'], stage['world_to_pixel']
                if rgb.ndim != 4 or rgb.shape[-1] != 3:
                    raise ValueError('Expected visualization RGB with shape [views,height,width,3]')
                views, height, width, _ = rgb.shape
                if (min(views, height, width) < 1 or maps.shape != (views, height, width)
                        or projection.shape != (views, 2, 4)):
                    raise ValueError('Visualization RGB, heatmap and projection shapes do not match')
                if any(not np.isfinite(value).all() for value in stage.values()):
                    raise ValueError('Visualization arrays must be finite')
                for key in ('target_prior', 'reference_prior', 'relation_anchor'):
                    if key in stage and (stage[key].ndim != 3 or stage[key].shape[0] != views
                                         or min(stage[key].shape[1:]) < 1):
                        raise ValueError(f'{key} must have one nonempty 2D map per visualization view')
                stage['heatmap_waypoint_pixels'] = projection @ np.r_[waypoint, 1.]
                if policy_tcp is not None:
                    stage['policy_pixels'] = projection @ np.r_[policy_tcp, 1.]
                _montage(stage, name, mode != 'eef' or 'chunk' in payload).save(temporary / f'{name}.png')
                np.savez_compressed(temporary / f'{name}.npz', **stage)
                png = f'/visualization/files/{entry_id}/{name}.png'
                record['images'][name] = png
                record['stages'][name] = dict(png=png, arrays=f'/visualization/files/{entry_id}/{name}.npz',
                    policy_in_view=None if policy_tcp is None else [_inside(p, width, height) for p in stage['policy_pixels']],
                    heatmap_in_view=[_inside(p, width, height) for p in stage['heatmap_waypoint_pixels']])
            record['sensors'], record['scene'] = None, None
            if sensor_observation is not None:
                try:
                    from .sensor_visualization import export_sensor_scene
                    record['sensors'], record['scene'] = export_sensor_scene(
                        temporary, entry_id, sensor_observation, policy_pose, waypoint)
                    if 'chunk' in payload:
                        scene_path = temporary / 'scene.json'
                        scene = json.loads(scene_path.read_text(encoding='utf-8'))
                        chunk = payload['chunk']
                        positions = chunk['actions'][:, :3].tolist()
                        origin = sensor_observation.get('tcp_pose_xyzw')
                        if origin is not None:
                            positions.insert(0, np.asarray(origin)[:3].tolist())
                        prefix = chunk['spec']['execute_horizon'] + (1 if origin is not None else 0)
                        scene['trajectories'] = [dict(positions=positions, color='#d4a8ff'),
                            dict(positions=positions[:prefix], color='#ff5965')]
                        scene_path.write_text(json.dumps(scene, allow_nan=False), encoding='utf-8')
                except Exception as exc:
                    # A sensor diagnostic failure must still publish the
                    # existing heatmaps and leave the predicted action intact.
                    record['sensors'] = dict(source=sensor_observation.get('source'),
                                             error=f'{type(exc).__name__}: {exc}')
                    for filename in ('sensors.png', 'sensors.npz', 'scene.json'):
                        (temporary / filename).unlink(missing_ok=True)
            (temporary / 'prediction.json').write_text(json.dumps(record, ensure_ascii=False, allow_nan=False), encoding='utf-8')
            temporary.rename(folder)
            self._entries.append(folder)
            return record
        except Exception:
            self._remove(temporary)
            raise

    def _run(self):
        while not self._closing.is_set() or not self._queue.empty():
            try:
                job = self._queue.get(timeout=.05)
            except Empty:
                continue
            with self._lock:
                self._writing = True
            try:
                record = self._write(*job)
                with self._lock:
                    self._latest, self._error = record, None
                # Publish the fully written replacement before removing its
                # predecessor; concurrent /latest readers must never receive a
                # pointer to a directory already removed by this writer.
                while len(self._entries) > self.keep:
                    self._remove(self._entries[0])
                    self._entries.popleft()
            except Exception as exc:
                with self._lock:
                    self._error = f'{type(exc).__name__}: {exc}'
            finally:
                with self._lock:
                    self._writing = False
                self._slots.release()
                self._queue.task_done()

    def close(self, *, wait=False, timeout=5.):
        with self._lock:
            self._closing.set()
        if wait:
            self._thread.join(timeout=timeout)


def viewer_html():
    """Serve the self-contained viewer without importing browser dependencies."""
    from .online_viewer import viewer_html as render_viewer
    return render_viewer()
