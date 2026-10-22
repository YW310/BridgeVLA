"""Bounded TCP debug overlays, kept separate from policy sensor captures.

Call ``suspend()`` before rendering/reading a policy observation and ``resume()``
afterwards. Debug-draw exclusion from sensor images varies across Isaac Sim
versions, so this explicit capture boundary is required even for viewport draws.
No USD prims, physics state, or simulation steps are created by this module.
"""
from collections import deque
import importlib
from threading import get_ident, main_thread
import warnings

import numpy as np
from scipy.spatial.transform import Rotation

from .joint_executor import numpy_value


GREEN = (0.1, 1.0, 0.2, 1.0)
RED = (1.0, 0.15, 0.1, 1.0)
AXIS_COLORS = ((1., .2, .2, .8), (.2, 1., .2, .8), (.2, .5, 1., .8))


def _debug_drawer():
    """Resolve an installed SDK only after the application has been launched."""
    def bindings(name):
        package = importlib.import_module(name)
        module = getattr(package, '_debug_draw', None)
        if module is None:
            try:
                module = importlib.import_module(name + '._debug_draw')
            except ImportError:
                module = importlib.import_module(name + '.bindings._debug_draw')
        return module

    errors = []
    for name in ('isaacsim.util.debug_draw', 'omni.isaac.debug_draw'):
        try:
            try:
                module = bindings(name)
            except ImportError:
                try:
                    extensions = importlib.import_module('isaacsim.core.utils.extensions')
                except ImportError:
                    extensions = importlib.import_module('omni.isaac.core.utils.extensions')
                extensions.enable_extension(name)
                module = bindings(name)
            drawer = module.acquire_debug_draw_interface()
            if drawer is None:
                raise RuntimeError('debug-draw interface is unavailable')
            return drawer, lambda: module.release_debug_draw_interface(drawer), name
        except Exception as exc:
            errors.append(f'{name}: {type(exc).__name__}: {exc}')
    raise RuntimeError('; '.join(errors))


class LiveMarkers:
    """Display actual TCP (green), EEF target (red), and bounded trajectories.

    ``measured_state['tcp_pose_xyzw']`` and the resolved EEF target are relative
    to environment zero's origin, as produced by IsaacLabBridge. Joint targets
    have no invented Cartesian position: only measured TCP is drawn for them.
    ``drawer`` is an optional debug-draw-compatible dependency for CPU tests.
    """

    def __init__(self, reference, env, *, enabled=True, history=200, drawer=None, axes=True):
        if isinstance(history, bool) or not isinstance(history, int) or history < 1:
            raise ValueError('Marker history must be a positive integer')
        self.reference, self.env = reference, env
        self.requested, self.enabled, self.available = bool(enabled), False, False
        self.history, self.axes = history, bool(axes)
        self.backend, self.reason = None, 'disabled'
        self._drawer, self._release = None, None
        self._closed = self._suspended = self._visible = self._warned = False
        self._unsafe_reason = None
        self._measured = self._target = None
        self._measured_history, self._target_history = deque(maxlen=history), deque(maxlen=history)
        if not self.requested:
            return
        self._check_thread()
        sim = getattr(env, 'sim', None)
        has_gui = getattr(sim, 'has_gui', None)
        headless = getattr(getattr(reference, 'args_cli', None), 'headless', False)
        if (callable(has_gui) and not has_gui()) or (not callable(has_gui) and headless):
            self.reason = 'headless'
            return
        try:
            if drawer is None:
                self._drawer, self._release, self.backend = _debug_drawer()
            else:
                self._drawer, self.backend = drawer, 'injected'
            self.enabled = self.available = True
            self.reason = None
        except Exception as exc:
            self._disable(exc)

    @property
    def info(self):
        return dict(requested=self.requested, enabled=self.enabled, available=self.available,
                    backend=self.backend, reason=self.reason, history=self.history,
                    isolation='clear-before-policy-capture',
                    measured_points=len(self._measured_history), target_points=len(self._target_history))

    @staticmethod
    def _check_thread():
        if get_ident() != main_thread().ident:
            raise RuntimeError('IsaacLab live markers must run on the main thread')

    def _origin(self):
        origins = getattr(self.env.scene, 'env_origins', None)
        value = np.zeros(3) if origins is None else numpy_value(origins)[0]
        origin = np.asarray(value, np.float64)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError('Expected one finite environment origin')
        return origin

    def _world_pose(self, value, *, target=False):
        pose = np.asarray(value, np.float64)
        valid_shape = pose.shape in ((7,), (8,)) if target else pose.shape == (7,)
        if not valid_shape or not np.isfinite(pose).all() or np.linalg.norm(pose[3:7]) == 0:
            raise ValueError('Expected a finite TCP xyz/xyzw pose')
        pose = pose[:7].copy()
        pose[:3] += self._origin()
        return pose

    @staticmethod
    def _append(history, value):
        if not history or not np.array_equal(history[-1], value):
            history.append(value.copy())

    def update(self, measured_state, target=None, action_mode='eef'):
        if not self.enabled or self._closed:
            return
        self._check_thread()
        try:
            self._measured = self._world_pose(measured_state['tcp_pose_xyzw'])
            self._append(self._measured_history, self._measured[:3])
            if 'joint' in action_mode:
                self._target = None
                self._target_history.clear()
            elif target is not None:
                self._target = self._world_pose(target, target=True)
                self._append(self._target_history, self._target[:3])
            else:
                self._target = None
            if not self._suspended:
                self._redraw()
        except Exception as exc:
            self._disable(exc)

    def _clear(self):
        if self._drawer is None or not self._visible:
            return
        errors = []
        for method in ('clear_points', 'clear_lines'):
            try:
                getattr(self._drawer, method)()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError('; '.join(f'{type(exc).__name__}: {exc}' for exc in errors))
        self._visible = False
        self._unsafe_reason = None

    def _redraw(self):
        try:
            self._clear()
            points, colors = [], []
            for pose, color in ((self._measured, GREEN), (self._target, RED)):
                if pose is not None:
                    points.append(tuple(pose[:3]))
                    colors.append(color)
            starts, ends, line_colors = [], [], []
            for history, color in ((self._measured_history, GREEN), (self._target_history, RED)):
                samples = list(history)
                starts.extend(tuple(point) for point in samples[:-1])
                ends.extend(tuple(point) for point in samples[1:])
                line_colors.extend([color] * max(0, len(samples) - 1))
            if self.axes:
                for pose in (self._measured, self._target):
                    if pose is not None:
                        basis = Rotation.from_quat(pose[3:7]).as_matrix()
                        starts.extend([tuple(pose[:3])] * 3)
                        ends.extend(tuple(pose[:3] + .04 * basis[:, axis]) for axis in range(3))
                        line_colors.extend(AXIS_COLORS)
            # An SDK draw can fail after creating only some geometry. Mark it as
            # potentially visible before invoking it so cleanup still clears it.
            self._visible = bool(points or starts)
            if points:
                self._drawer.draw_points(points, colors, [12.] * len(points))
            if starts:
                self._drawer.draw_lines(starts, ends, line_colors, [2.] * len(starts))
        except Exception as exc:
            self._disable(exc)

    def _disable(self, exc):
        self.enabled = self.available = False
        self.reason = f'{type(exc).__name__}: {exc}'
        try:
            self._clear()
        except Exception as clear_error:
            self._unsafe_reason = f'{type(clear_error).__name__}: {clear_error}'
        if not self._warned:
            warnings.warn(f'BridgeVLA live markers disabled: {self.reason}', RuntimeWarning, stacklevel=2)
            self._warned = True

    def suspend(self):
        """Remove overlays before the render/read that builds policy RGB-D."""
        self._check_thread()
        self._suspended = True
        try:
            self._clear()
        except Exception as exc:
            self._disable(exc)
        if self._unsafe_reason is not None:
            raise RuntimeError('Cannot clear live markers before policy sensor capture: ' + self._unsafe_reason)

    def resume(self):
        """Restore the last overlay after the independent snapshot is complete."""
        self._check_thread()
        self._suspended = False
        if self.enabled and not self._closed:
            self._redraw()

    def reset(self):
        self._check_thread()
        self._measured_history.clear()
        self._target_history.clear()
        self._measured = self._target = None
        try:
            self._clear()
        except Exception as exc:
            self._disable(exc)

    def close(self):
        self._check_thread()
        if self._closed:
            return
        try:
            self.reset()
        finally:
            self._closed = True
            self.enabled = False
            if self._release is not None:
                try:
                    self._release()
                except Exception as exc:
                    self._disable(exc)
