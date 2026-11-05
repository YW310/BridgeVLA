"""Bind online SAM3 to the exact current RLBench policy observation.

No demo, expert action, future observation or additional sensor render is read.
The ordinary environment does not import or instantiate this provider.

PyRep returns negative fx/fy. For TASK's positive-focal optical convention we
apply D=diag(-1,-1,1,1), K_opt=K_raw@D[:3,:3], T_world_opt=T_world_raw@D.
This is a proper 180-degree rotation around camera Z: RGB and world XYZ stay
unchanged. Original K/T bind policy tensors; optical K/T form TASK's calibration
digest. Metric Z-depth is recovered from the *current measured world cloud*,
with a per-pixel projection round trip, because baseline RLBench disables the
returned depth array while retaining the point cloud computed from that depth.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np


@dataclass(frozen=True)
class CurrentFramePacket:
    snapshot: Any
    receipt: dict
    task_name: str
    instruction: str


def simulation_clock():
    """Read the actual simulator clock, including action-execution microsteps.

    Scene.register_step_callback has a single callback slot in RLBench, so an
    extra callback would replace recording behavior. Reading this clock does
    not advance simulation or touch that callback.
    """
    from pyrep.backend import sim
    return float(sim.simGetSimulationTime()), float(sim.simGetSimulationTimeStep())


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _language(value):
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, str) or not value.strip():
        raise ValueError('current policy language_goal must contain one nonempty instruction')
    return value


def _readonly(value, dtype=None):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def optical_calibration(intrinsics, extrinsics):
    """Convert the proven paired-negative PyRep basis, never guess a missing K/T."""
    k, t = np.asarray(intrinsics, dtype=np.float64), np.asarray(extrinsics, dtype=np.float64)
    if k.shape != (3, 3) or t.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(t).all():
        raise ValueError('current observation requires finite 3x3 K and 4x4 T_world_camera')
    if (not np.allclose(k[[0, 1, 2, 2, 2], [1, 0, 0, 1, 2]], [0, 0, 0, 0, 1], rtol=0, atol=1e-10)
            or not np.allclose(t[3], [0, 0, 0, 1], rtol=0, atol=1e-10)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), rtol=0, atol=1e-7)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1., rtol=0, atol=1e-7)):
        raise ValueError('current K/T must describe a rigid zero-skew pinhole camera')
    d = np.eye(4)
    if k[0, 0] < 0 and k[1, 1] < 0:
        d[0, 0] = d[1, 1] = -1.
    elif not (k[0, 0] > 0 and k[1, 1] > 0):
        raise ValueError('camera focal signs must be paired; single-axis reflections are unsupported')
    return k @ d[:3, :3], t @ d, d


class SlotSam3ObservationProvider:
    """Opt-in cache populated by env.extract_obs, consumed by the online wrapper.

    ``time_source`` is a test seam returning (simulation_seconds, timestep_s).
    Production uses PyRep's actual simulation time, not policy-step/demo index.
    Calling the wrapper's reset must not clear this cache: YARR resets the env
    and captures its observation *before* resetting the agent.
    """
    def __init__(self, task_repo, cameras, *, time_source=None):
        if isinstance(cameras, (str, bytes)) or not cameras or len(set(cameras)) != len(cameras):
            raise ValueError('cameras must be an explicit nonempty unique camera sequence')
        if any(not isinstance(name, str) or not name.strip() for name in cameras):
            raise ValueError('camera names must be nonempty strings')
        self.task_repo = Path(task_repo).resolve()
        self.cameras = tuple(cameras)
        self._time_source = time_source or simulation_clock
        self._runtime_id = uuid4().hex
        self._generation = -1
        self._api = None
        self.begin_episode()

    def _load_api(self):
        if self._api is None:
            from bridgevla.offline.slot_sam3_perception import TaskPerception
            self._api = TaskPerception(self.task_repo)._load_api()
        return self._api

    def begin_episode(self, episode_id=None):
        self._generation += 1
        self._episode_label = episode_id
        label = '' if episode_id is None else str(episode_id) + ':'
        self._episode_id = f'live:{self._runtime_id}:{label}{self._generation}'
        self._origin_time = self._timestep = None
        self._last_time = None
        self._packet = None
        self._binding = {}
        self._failure = 'no current observation has been captured'

    def invalidate(self, error):
        self._packet = None
        self._binding = {}
        self._failure = f'{type(error).__name__}: {error}'

    def _clock(self):
        seconds, step = self._time_source()
        if (isinstance(seconds, (bool, np.bool_)) or isinstance(step, (bool, np.bool_))
                or not np.isfinite([seconds, step]).all() or seconds < 0 or step <= 0):
            raise ValueError('simulation clock requires nonnegative seconds and a positive timestep')
        return float(seconds), float(step)

    def capture_raw(self, obs, extracted, *, task_name, instruction, episode_id=None):
        """Capture only arrays from this raw Observation and its extracted dict.

        Errors invalidate the previous cache. The env hook records the failure
        and returns the original policy observation, allowing its baseline to
        execute and the online wrapper to apply the configured fallback.
        """
        self.invalidate(ValueError('current observation capture did not complete'))
        if episode_id is not None and episode_id != self._episode_label:
            # Explicit caller identities are labels only; never demo frame indices.
            raise ValueError('use begin_episode(episode_id) to change an episode identity')
        if not isinstance(task_name, str) or not task_name.strip():
            raise ValueError('current task_name is required')
        instruction = _language(instruction)
        seconds, step = self._clock()
        if self._origin_time is None:
            self._origin_time, self._timestep = seconds, step
        if not np.isclose(step, self._timestep, rtol=0, atol=1e-9):
            raise ValueError('simulation timestep changed within an episode')
        elapsed = (seconds - self._origin_time) / self._timestep
        index = int(round(elapsed))
        # PyRep's C binding returns a float; account for its float32 ULP at long
        # running simulator times without confusing half-steps with whole steps.
        clock_tolerance = min(.1, max(1e-3, 4 * float(np.spacing(np.float32(seconds))) / step))
        if index < 0 or abs(elapsed - index) > clock_tolerance or (self._last_time is not None and seconds < self._last_time):
            raise ValueError('simulation clock is not a monotonic integer microstep sequence')
        timestamp = datetime.now(timezone.utc).isoformat()
        frame_id = f'simstep-{index}'
        identity = dict(episode_id=self._episode_id, frame_id=frame_id,
                        source_frame_index=index, observed_at=timestamp)
        api = self._load_api()
        misc = getattr(obs, 'misc', None)
        if not isinstance(misc, dict):
            raise ValueError('current raw Observation.misc camera metadata is required')
        rows, frames, binding, bases = [], {}, {}, {}
        for name in self.cameras:
            rgb = np.asarray(getattr(obs, f'{name}_rgb', None))
            cloud = np.asarray(getattr(obs, f'{name}_point_cloud', None))
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3 or min(rgb.shape[:2]) < 2:
                raise ValueError(f'{name}: original-resolution uint8 HWC RGB is required')
            if cloud.shape != rgb.shape or cloud.dtype.kind not in 'fiu':
                raise ValueError(f'{name}: registered current world XYZ is required')
            k_raw, t_raw = misc.get(f'{name}_camera_intrinsics'), misc.get(f'{name}_camera_extrinsics')
            k, t, basis = optical_calibration(k_raw, t_raw)
            valid = np.isfinite(cloud).all(-1)
            camera_points = api.camera_geometry.transform_points(cloud[valid], np.linalg.inv(t))
            depth = np.full(rgb.shape[:2], np.nan, dtype=np.float64)
            positive = camera_points[:, 2] > 1e-6
            # Nonpositive/invalid measurements stay invalid, never world origin.
            positions = np.flatnonzero(valid)
            depth.flat[positions[positive]] = camera_points[positive, 2]
            measured = np.isfinite(depth) & (depth > 0)
            if measured.any():
                reconstructed = api.camera_geometry.transform_points(
                    api.camera_geometry.backproject_mask(depth, measured, k, max_points=0), t)
                if not np.allclose(reconstructed, cloud[measured], rtol=2e-5, atol=2e-6):
                    raise ValueError(f'{name}: current point cloud does not match camera calibration/pixel projection')
            calibration_source = ('current RLBench Observation.misc; optical K=K_raw@D3, T=T_raw@D; '
                                  'D=diag(-1,-1,1,1)' if basis[0, 0] < 0
                                  else 'current RLBench Observation.misc; already positive-focal optical basis')
            row = dict(name=name, rgb_path=f'memory://{name}.rgb', depth_path=f'memory://{name}.depth',
                       depth_encoding='meters_npy', source_frame_index=index, observed_at=timestamp,
                       calibration_source=calibration_source, intrinsics=k.tolist(), extrinsics=t.tolist())
            camera = api.slot_guided_io.CameraFrame(name=name, **identity, intrinsics=_readonly(k),
                       extrinsics=_readonly(t), image_size=(rgb.shape[1], rgb.shape[0]),
                       calibration_source=calibration_source)
            frames[name] = api.slot_guided_io.RGBDFrame(camera, _readonly(rgb), _readonly(depth), row.copy())
            rows.append(row)
            bases[name] = basis.tolist()
            expected = {f'{name}_rgb': np.moveaxis(rgb, -1, 0),
                        f'{name}_point_cloud': np.moveaxis(cloud, -1, 0),
                        f'{name}_camera_intrinsics': np.asarray(k_raw),
                        f'{name}_camera_extrinsics': np.asarray(t_raw)}
            for key, raw in expected.items():
                if key not in extracted:
                    raise ValueError(f'current extracted observation is missing {key}')
                value = _array(extracted[key])
                # Policy observation is channels-first in the supported RLBench runner.
                if value.shape != raw.shape or not np.allclose(value, raw, rtol=2e-6, atol=2e-7, equal_nan=True):
                    raise ValueError(f'{key}: raw and extracted policy observations disagree')
                binding[key] = _readonly(value, dtype=np.float32)
        low_dim = _array(extracted.get('low_dim_state'))
        if low_dim.ndim != 1 or not len(low_dim) or not np.isfinite(low_dim).all():
            raise ValueError('current extracted low_dim_state must be a finite vector')
        binding['low_dim_state'] = _readonly(low_dim, dtype=np.float32)
        snapshot = api.slot_guided_io.FrameSnapshot(
            dict(schema_version=1, artifact_type='slot_rgbd_frame', **identity, cameras=rows),
            frames, Path('.'))
        api.slot_guidance_producers._validate_snapshot(snapshot)
        receipt = {**identity, 'calibration_sha256': api.slot_guided_io.calibration_digest(snapshot),
                   'snapshot_input_sha256': api.slot_guided_io.snapshot_input_digest(snapshot),
                   'max_source_frame_index_read': index, 'source_frame_index_kind': 'live_simulation_microstep',
                   'simulation_time_s': seconds, 'simulation_timestep_s': step,
                   'episode_simulation_origin_s': self._origin_time,
                   'depth_source': 'current_observation_world_point_cloud_to_camera_z',
                   'depth_units': 'm', 'raw_camera_from_optical': bases}
        after, after_step = self._clock()
        if after != seconds or after_step != step:
            raise ValueError('simulation advanced during current observation capture')
        self._last_time = seconds
        self._binding = binding
        self._packet = CurrentFramePacket(snapshot, receipt, task_name, instruction)
        self._failure = None

    def capture(self, prepped_observation):
        """Return the cached current frame only after binding every latest T input."""
        if self._packet is None:
            raise ValueError(f'current online RGB-D unavailable: {self._failure}')
        seconds, step = self._clock()
        if seconds != self._packet.receipt['simulation_time_s'] or step != self._timestep:
            raise ValueError('simulation advanced after the cached policy observation')
        for key, expected in self._binding.items():
            if key not in prepped_observation:
                raise ValueError(f'current policy observation is missing {key}')
            value = _array(prepped_observation[key])
            if value.ndim != expected.ndim + 2 or value.shape[0] != 1 or value.shape[1] < 1:
                raise ValueError(f'{key}: expected one batch with an explicit current time axis')
            latest = value[0, -1]
            if (latest.shape != expected.shape or not np.allclose(latest, expected, rtol=0, atol=1e-7, equal_nan=True)):
                raise ValueError(f'{key}: policy tensors do not match the cached current frame')
        if _language(prepped_observation.get('language_goal')) != self._packet.instruction:
            raise ValueError('policy instruction does not match the current raw frame')
        api = self._load_api()
        if api.slot_guided_io.snapshot_input_digest(self._packet.snapshot) != self._packet.receipt['snapshot_input_sha256']:
            raise ValueError('cached current snapshot was modified after capture')
        return self._packet
