"""Keep failed rollouts inspectable without resuming control or HTTP requests."""
import json
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from finetune.OHT import isaaclab_client as runner
from finetune.OHT.runtime import isaaclab_factory as native
from finetune.OHT.runtime import policy_worker as workers
from finetune.OHT.runtime.http_client import BridgeVLAClient
from tests.test_oht_isaaclab_http import Opener
from tests.test_oht_migration import replay_fixture  # noqa: F401
from tests.test_oht_vla_server import fixture  # noqa: F401
from tests.test_oht_waypoint_client import configure_robot, install_policy


def forbidden(*args, **kwargs):
    raise AssertionError('Inspection must not advance physics or application updates')


def inspection_runtime(*, headless=False, gui=True, running=True):
    state = SimpleNamespace(running=running, renders=0, render_threads=[], sleeps=[])
    app = SimpleNamespace(is_running=lambda: state.running, update=forbidden)
    env = SimpleNamespace(step=forbidden, sim=SimpleNamespace(has_gui=lambda: gui))
    reference = SimpleNamespace(simulation_app=app, args_cli=SimpleNamespace(headless=headless))

    def render():
        state.renders += 1
        state.render_threads.append(threading.get_ident())
        if state.renders == 3:
            state.running = False

    env.sim.render = render
    return reference, env, state


def test_inspection_renders_only_on_calling_thread_until_window_closes(monkeypatch, capsys):
    reference, env, state = inspection_runtime()
    monkeypatch.setattr(workers, 'sleep', lambda seconds: state.sleeps.append(seconds))
    workers.keep_open_after_error(reference, env)
    assert state.renders == 3 and not state.running
    assert state.render_threads == [threading.get_ident()] * 3
    assert all(0 < seconds <= .05 for seconds in state.sleeps)
    assert 'keeping the window open' in capsys.readouterr().out


@pytest.mark.parametrize('headless,gui,running', [
    (True, True, True),
    (True, False, True),
    (False, False, True),
    (False, True, False),
])
def test_inspection_skips_headless_no_gui_and_already_closed(monkeypatch, headless, gui, running):
    reference, env, state = inspection_runtime(headless=headless, gui=gui, running=running)
    env.sim.render = forbidden
    monkeypatch.setattr(workers, 'sleep', forbidden)
    workers.keep_open_after_error(reference, env)
    assert state.renders == 0


def test_legacy_inspection_uses_render_only_hook_without_sim_object(monkeypatch):
    reference, env, state = inspection_runtime()
    del env.sim

    def render(selected):
        assert selected is env
        state.renders += 1
        if state.renders == 2:
            state.running = False

    reference._replicator_render_tick = render
    monkeypatch.setattr(workers, 'sleep', lambda seconds: None)
    workers.keep_open_after_error(reference, env)
    assert state.renders == 2


@pytest.mark.parametrize('error', [KeyboardInterrupt(), RuntimeError('render failed')])
def test_inspection_interrupt_or_render_error_does_not_replace_rollout_error(monkeypatch, capsys, error):
    reference, env, state = inspection_runtime()
    saved = {'error': 'Waypoint did not complete: stalled', 'stop_reason': 'waypoint_stalled'}
    original = saved.copy()

    def render():
        state.renders += 1
        raise error

    env.sim.render = render
    monkeypatch.setattr(workers, 'sleep', forbidden)
    workers.keep_open_after_error(reference, env)
    assert state.renders == 1 and saved == original
    output = capsys.readouterr().out
    if isinstance(error, RuntimeError):
        assert 'inspection ended: RuntimeError: render failed' in output


def run_failure(monkeypatch, fixture, raw, output, *, keep_open=False, headless=False):
    extra = ['--keep-open-on-error'] if keep_open else []
    if headless:
        extra.append('--headless')
    with TestClient(fixture.app(visualize_every=0)) as server:
        monkeypatch.setattr(runner, 'BridgeVLAClient',
                            lambda *args: BridgeVLAClient('http://testserver', opener=Opener(server)))
        return runner.main(['--env-id', 'OHT-test', '--ee-link', 'tool', '--task', raw['task'],
                            '--episodes', '2', '--max-steps', '2', '--no-visualize',
                            '--waypoint-stall-steps', '4', '--output', str(output), *extra])


@pytest.mark.parametrize('inspection_end', ['window_close', 'interrupt', 'render_error'])
def test_failed_policy_saves_and_closes_worker_before_inspection_then_cleans_up(
        fixture, replay_fixture, tmp_path, monkeypatch, inspection_end):
    env, raw, calibration, physical, native_state = configure_robot(
        monkeypatch, fixture, replay_fixture, jammed=True)
    predictions = install_policy(fixture, env, calibration, 'eef')
    output = tmp_path / 'inspect-failure'
    instances, events = [], []
    original_worker = runner.PolicyWorker

    class TrackedWorker(original_worker):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            instances.append(self)

        def close(self):
            events.append('worker_close')
            return super().close()

    def inspect(reference, selected_env):
        events.append('inspection')
        assert selected_env is env
        assert len(instances) == 1 and instances[0]._closed
        assert not env.closed and not native_state.app_closed
        assert len(predictions) == len(fixture.policy.calls) == 1
        assert len(physical) == env.common_step_counter == 4
        episode = json.loads((output / 'episode_0000.json').read_text(encoding='utf-8'))
        summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
        assert episode['stop_reason'] == 'waypoint_stalled'
        assert episode['error'] == 'Waypoint did not complete: stalled'
        assert len(episode['actions']) == 1 and len(episode['actions'][0]['commands']) == 4
        joints = episode['actions'][0]['execution']['joint_diagnostics']
        assert joints['hard'] is None and joints['soft']['source'] == 'soft_joint_pos_limits'
        assert joints['positions_rad'] == episode['actions'][0]['measured_after']['joint_positions']
        cameras = episode['actions'][0]['camera_observation']
        assert set(cameras) == set(env.scene.sensors)
        assert all(len(camera['pose_wxyz']) == 7 and len(camera['intrinsics']) == 3
                   for camera in cameras.values())
        assert summary['episodes'][0]['error'] == episode['error']
        assert summary['attempted'] == summary['unattempted'] == 1
        assert not (output / 'episode_0001.json').exists()
        state = SimpleNamespace(running=True, renders=0)
        reference.simulation_app.is_running = lambda: state.running
        reference.simulation_app.update = forbidden
        env.step = forbidden
        env.sim.has_gui = lambda: True

        def render():
            assert instances[0]._closed and not env.closed
            assert len(predictions) == 1 and env.common_step_counter == 4
            state.renders += 1
            if inspection_end == 'interrupt':
                raise KeyboardInterrupt()
            if inspection_end == 'render_error':
                raise RuntimeError('inspection renderer failed')
            if state.renders == 3:
                state.running = False

        env.sim.render = render
        workers.keep_open_after_error(reference, env)
        assert state.renders == (3 if inspection_end == 'window_close' else 1)
        assert json.loads((output / 'episode_0000.json').read_text(encoding='utf-8')) == episode
        events.append('inspection_end')

    monkeypatch.setattr(runner, 'PolicyWorker', TrackedWorker)
    monkeypatch.setattr(runner, 'keep_open_after_error', inspect)
    monkeypatch.setattr(workers, 'sleep', lambda seconds: None)
    assert run_failure(monkeypatch, fixture, raw, output, keep_open=True) == 1
    assert events[0:2] == ['worker_close', 'inspection']
    assert events[-2:] == ['inspection_end', 'worker_close']
    assert env.closed and native_state.app_closed
    assert len(predictions) == len(fixture.policy.calls) == 1 and env.common_step_counter == 4


def test_failed_policy_does_not_hold_window_by_default(fixture, replay_fixture, tmp_path, monkeypatch):
    env, raw, calibration, _, state = configure_robot(monkeypatch, fixture, replay_fixture, jammed=True)
    predictions = install_policy(fixture, env, calibration, 'eef')
    monkeypatch.setattr(runner, 'keep_open_after_error', forbidden)
    assert run_failure(monkeypatch, fixture, raw, tmp_path / 'default-failure') == 1
    assert env.closed and state.app_closed and len(predictions) == 1


def test_headless_failure_flag_never_waits_for_user(fixture, replay_fixture, tmp_path, monkeypatch):
    env, raw, calibration, _, state = configure_robot(monkeypatch, fixture, replay_fixture, jammed=True)
    predictions = install_policy(fixture, env, calibration, 'eef')
    launch = native.launch_application

    def headless_launch(extra):
        assert '--headless' in extra
        app, options = launch(extra)
        options.headless = True
        return app, options

    monkeypatch.setattr(native, 'launch_application', headless_launch)
    monkeypatch.setattr(workers, 'sleep', forbidden)
    env.sim.has_gui = lambda: False
    assert run_failure(monkeypatch, fixture, raw, tmp_path / 'headless-failure',
                       keep_open=True, headless=True) == 1
    assert env.closed and state.app_closed and len(predictions) == 1
    assert env.common_step_counter == 4
