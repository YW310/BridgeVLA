"""TCP overlays are bounded, main-thread-only, and absent at capture boundaries."""
from copy import deepcopy
from types import SimpleNamespace
import threading
import warnings

import numpy as np
import pytest

from finetune.OHT.runtime import live_markers as markers


class Drawer:
    def __init__(self):
        self.points, self.lines = [], []
        self.point_colors, self.line_colors = [], []
        self.calls = []

    def clear_points(self):
        self.calls.append(('clear_points', threading.get_ident()))
        self.points, self.point_colors = [], []

    def clear_lines(self):
        self.calls.append(('clear_lines', threading.get_ident()))
        self.lines, self.line_colors = [], []

    def draw_points(self, points, colors, sizes):
        self.calls.append(('draw_points', threading.get_ident()))
        assert len(points) == len(colors) == len(sizes)
        self.points, self.point_colors = list(points), list(colors)

    def draw_lines(self, starts, ends, colors, widths):
        self.calls.append(('draw_lines', threading.get_ident()))
        assert len(starts) == len(ends) == len(colors) == len(widths)
        self.lines, self.line_colors = list(zip(starts, ends)), list(colors)


def runtime(*, origin=(0, 0, 0), headless=False):
    reference = SimpleNamespace(args_cli=SimpleNamespace(headless=headless))
    env = SimpleNamespace(scene=SimpleNamespace(env_origins=np.array([origin], np.float32)),
                          sim=SimpleNamespace(has_gui=lambda: not headless), physics_steps=0)
    return reference, env


def measured(x=0., *, quaternion=(0., 0., 0., 1.)):
    return {'tcp_pose_xyzw': [x, 2., 3., *quaternion]}


def test_world_positions_add_environment_origin_once_without_mutating_state():
    reference, env = runtime(origin=(10, 20, 30))
    drawer = Drawer()
    state, target = measured(1), np.array([4., 5., 6., 0, 0, 0, 1, 1])
    before_state, before_target = deepcopy(state), target.copy()
    overlay = markers.LiveMarkers(reference, env, drawer=drawer, axes=False)
    overlay.update(state, target)
    np.testing.assert_allclose(drawer.points, [[11, 22, 33], [14, 25, 36]])
    assert drawer.point_colors == [markers.GREEN, markers.RED]
    assert state == before_state
    np.testing.assert_array_equal(target, before_target)
    assert env.physics_steps == 0
    assert overlay.info['available'] and overlay.info['isolation'] == 'clear-before-policy-capture'
    overlay.close()


@pytest.mark.parametrize('mode', ['eef', 'relative_eef'])
def test_resolved_eef_target_draws_position_and_xyzw_orientation_axes(mode):
    reference, env = runtime()
    drawer = Drawer()
    overlay = markers.LiveMarkers(reference, env, drawer=drawer)
    rotation = (0., 0., np.sqrt(.5), np.sqrt(.5))
    overlay.update(measured(1, quaternion=rotation), [4., 5., 6., *rotation, 1.], mode)
    assert len(drawer.points) == 2 and len(drawer.lines) == 6
    # X of a +90-degree world-Z rotation points toward world Y, not world X.
    np.testing.assert_allclose(drawer.lines[0], [[1, 2, 3], [1, 2.04, 3]], atol=1e-10)
    np.testing.assert_allclose(drawer.lines[3], [[4, 5, 6], [4, 5.04, 6]], atol=1e-10)
    overlay.close()


@pytest.mark.parametrize('mode', ['joint', 'relative_joint'])
def test_joint_mode_draws_actual_tcp_only_and_never_treats_joints_as_xyz(mode):
    reference, env = runtime()
    drawer = Drawer()
    overlay = markers.LiveMarkers(reference, env, drawer=drawer, axes=False)
    overlay.update(measured(0), [100, 200, 300, 0, 0, 0, 1, 1], 'eef')
    overlay.update(measured(1), [99, -99, 88, -88, 77, -77, 1], mode)
    assert drawer.points == [(1., 2., 3.)]
    assert drawer.point_colors == [markers.GREEN]
    assert overlay.info['target_points'] == 0
    assert len(drawer.lines) == 1 and drawer.line_colors == [markers.GREEN]
    overlay.close()


def test_histories_are_bounded_and_consecutive_duplicate_positions_are_not_added():
    reference, env = runtime()
    drawer = Drawer()
    overlay = markers.LiveMarkers(reference, env, drawer=drawer, history=3, axes=False)
    for x in range(7):
        overlay.update(measured(x), [x + 1., 0, 0, 0, 0, 0, 1, 1])
    overlay.update(measured(6), [7., 0, 0, 0, 0, 0, 1, 1])
    assert overlay.info['measured_points'] == overlay.info['target_points'] == 3
    assert len(drawer.lines) == 4
    np.testing.assert_allclose(drawer.lines[0], [[4, 2, 3], [5, 2, 3]])
    assert drawer.line_colors == [markers.GREEN] * 2 + [markers.RED] * 2
    overlay.close()


def test_capture_boundary_clears_geometry_and_restores_it_without_physics_or_history_growth():
    reference, env = runtime(origin=(1, 2, 3))
    drawer = Drawer()
    overlay = markers.LiveMarkers(reference, env, drawer=drawer, axes=False)
    overlay.update(measured(1), [5., 6., 7., 0, 0, 0, 1, 1])
    displayed = list(drawer.points)
    overlay.suspend()
    assert not drawer.points and not drawer.lines
    # This is where bridge.capture() may render/read all policy RGB-D cameras.
    assert env.physics_steps == 0
    overlay.resume()
    assert drawer.points == displayed
    assert overlay.info['measured_points'] == overlay.info['target_points'] == 1
    overlay.suspend()
    overlay.update(measured(2))
    assert not drawer.points and not drawer.lines
    overlay.resume()
    assert drawer.points == [(3., 4., 6.)]
    overlay.close()


def test_episode_reset_clears_histories_and_close_is_idempotent():
    reference, env = runtime()
    drawer = Drawer()
    overlay = markers.LiveMarkers(reference, env, drawer=drawer)
    overlay.update(measured(1))
    overlay.reset()
    assert not drawer.points and not drawer.lines
    assert overlay.info['measured_points'] == overlay.info['target_points'] == 0
    overlay.update(measured(2))
    overlay.close()
    calls = len(drawer.calls)
    overlay.close()
    overlay.update(measured(3))
    assert not overlay.enabled and len(drawer.calls) == calls
    assert not drawer.points and not drawer.lines


@pytest.mark.parametrize('enabled,headless', [(False, False), (True, True)])
def test_disabled_and_headless_do_not_import_the_simulator(monkeypatch, enabled, headless):
    def forbidden():
        pytest.fail('Disabled/headless overlays must not import a simulator extension')
    monkeypatch.setattr(markers, '_debug_drawer', forbidden)
    reference, env = runtime(headless=headless)
    overlay = markers.LiveMarkers(reference, env, enabled=enabled)
    overlay.update(measured())
    overlay.suspend()
    overlay.resume()
    overlay.reset()
    overlay.close()
    assert not overlay.available and not overlay.enabled
    assert overlay.reason == ('headless' if enabled else 'disabled')


def test_missing_extension_warns_once_and_disables_without_control_failure(monkeypatch):
    calls = []
    def missing():
        calls.append(True)
        raise ImportError('debug-draw is not installed')
    monkeypatch.setattr(markers, '_debug_drawer', missing)
    reference, env = runtime()
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        overlay = markers.LiveMarkers(reference, env)
        overlay.update(measured())
        overlay.suspend()
        overlay.resume()
        overlay.close()
    assert len(calls) == len(emitted) == 1
    assert not overlay.available and 'debug-draw is not installed' in overlay.reason


def test_partial_draw_failure_clears_geometry_and_disables_overlay():
    class BrokenDrawer(Drawer):
        def draw_points(self, *args):
            super().draw_points(*args)
            raise RuntimeError('draw failed after creating points')
    reference, env = runtime()
    drawer = BrokenDrawer()
    overlay = markers.LiveMarkers(reference, env, drawer=drawer)
    with pytest.warns(RuntimeWarning, match='live markers disabled'):
        overlay.update(measured())
    assert not drawer.points and not drawer.lines and not overlay.enabled
    overlay.suspend()  # A failed, fully cleaned overlay cannot contaminate RGB-D.
    overlay.close()


def test_unclearable_visible_markers_refuse_sensor_capture():
    class BrokenClear(Drawer):
        def clear_points(self):
            raise RuntimeError('cannot clear points')
    reference, env = runtime()
    drawer = BrokenClear()
    overlay = markers.LiveMarkers(reference, env, drawer=drawer, axes=False)
    overlay.update(measured())
    with pytest.warns(RuntimeWarning, match='live markers disabled'):
        with pytest.raises(RuntimeError, match='before policy sensor capture'):
            overlay.suspend()
    assert drawer.points and env.physics_steps == 0
    with pytest.raises(RuntimeError, match='before policy sensor capture'):
        overlay.suspend()
    overlay.close()


def test_sdk_calls_reject_background_thread():
    reference, env = runtime()
    drawer = Drawer()
    overlay = markers.LiveMarkers(reference, env, drawer=drawer)
    errors = []
    def background():
        try:
            overlay.update(measured())
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=background)
    thread.start()
    thread.join(timeout=3)
    assert len(errors) == 1 and 'main thread' in str(errors[0])
    assert not drawer.calls
    overlay.update(measured())
    assert {thread_id for _, thread_id in drawer.calls} == {threading.get_ident()}
    overlay.close()


def test_modern_and_legacy_interfaces_import_lazily_and_release(monkeypatch):
    for selected in ('isaacsim.util.debug_draw', 'omni.isaac.debug_draw'):
        drawer, imports, releases = Drawer(), [], []
        api = SimpleNamespace(acquire_debug_draw_interface=lambda: drawer,
                              release_debug_draw_interface=lambda value: releases.append(value))
        def importer(name):
            imports.append(name)
            if name == selected:
                return SimpleNamespace(_debug_draw=api)
            raise ImportError(name)
        monkeypatch.setattr(markers.importlib, 'import_module', importer)
        reference, env = runtime()
        overlay = markers.LiveMarkers(reference, env)
        assert overlay.available and overlay.backend == selected
        overlay.close()
        assert releases == [drawer]
        assert selected in imports


@pytest.mark.parametrize('suffix', ['._debug_draw', '.bindings._debug_draw'])
def test_drawer_imports_bindings_not_exported_by_package(monkeypatch, suffix):
    drawer, imports, releases = Drawer(), [], []
    name = 'isaacsim.util.debug_draw'
    api = SimpleNamespace(acquire_debug_draw_interface=lambda: drawer,
                          release_debug_draw_interface=lambda value: releases.append(value))
    def importer(value):
        imports.append(value)
        if value == name:
            return SimpleNamespace()  # Plain package import does not bind _debug_draw.
        if value == name + suffix:
            return api
        raise ImportError(value)
    monkeypatch.setattr(markers.importlib, 'import_module', importer)
    reference, env = runtime()
    overlay = markers.LiveMarkers(reference, env)
    assert overlay.available and name + suffix in imports
    overlay.close()
    assert releases == [drawer]


def test_loader_enables_extension_when_bindings_require_it(monkeypatch):
    drawer, enabled = Drawer(), []
    name = 'isaacsim.util.debug_draw'
    api = SimpleNamespace(acquire_debug_draw_interface=lambda: drawer,
                          release_debug_draw_interface=lambda value: None)
    def importer(value):
        if value == 'isaacsim.core.utils.extensions':
            return SimpleNamespace(enable_extension=lambda selected: enabled.append(selected))
        if value == name and enabled:
            return SimpleNamespace(_debug_draw=api)
        raise ImportError(value)
    monkeypatch.setattr(markers.importlib, 'import_module', importer)
    reference, env = runtime()
    overlay = markers.LiveMarkers(reference, env)
    assert overlay.available and enabled == [name]
    overlay.close()


@pytest.mark.parametrize('invalid', ['missing_tcp', 'nonfinite_tcp', 'zero_quaternion', 'bad_target', 'bad_origin'])
def test_bad_overlay_geometry_disables_visuals_without_stopping_control(invalid):
    reference, env = runtime()
    drawer = Drawer()
    overlay = markers.LiveMarkers(reference, env, drawer=drawer)
    overlay.update(measured())
    state, target = measured(1), None
    if invalid == 'missing_tcp':
        state = {}
    elif invalid == 'nonfinite_tcp':
        state['tcp_pose_xyzw'][0] = np.nan
    elif invalid == 'zero_quaternion':
        state['tcp_pose_xyzw'][3:] = [0, 0, 0, 0]
    elif invalid == 'bad_target':
        target = [0., 1.]
    else:
        env.scene.env_origins[0, 0] = np.inf
    with pytest.warns(RuntimeWarning, match='live markers disabled'):
        overlay.update(state, target)
    assert not overlay.enabled and not drawer.points and not drawer.lines
    overlay.suspend()
    overlay.close()


@pytest.mark.parametrize('history', [0, -1, 1.5, True])
def test_invalid_history_is_rejected(history):
    reference, env = runtime()
    with pytest.raises(ValueError, match='positive integer'):
        markers.LiveMarkers(reference, env, history=history, drawer=Drawer())
