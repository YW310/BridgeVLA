"""CPU-only checks for responsive HTTP waits without concurrent policy requests."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from finetune.OHT.runtime.policy_worker import PolicyWorker, refresh_while_waiting


class BlockingPolicy:
    """Hold prediction until the caller's main-thread render callback releases it."""

    def __init__(self, *, error=None):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.calls = []
        self.resets = []
        self.error = error

    def reset(self, episode=None):
        self.resets.append((threading.get_ident(), episode))
        return {'episode_id': episode or 'fresh-episode'}

    def predict(self, observation, instruction, timestamp, *, task, data_profile):
        self.calls.append((threading.get_ident(), observation, instruction, timestamp, task, data_profile))
        self.entered.set()
        try:
            if not self.release.wait(3):
                raise AssertionError('Prediction was not released by its test')
            if self.error is not None:
                raise self.error
            return np.array([.1, 0, .5, 0, 0, 0, 1, 1]), {'timing_ms': {'policy': 12.}}
        finally:
            self.finished.set()


def worker_for(policy):
    return PolicyWorker(policy, 'remove the left part', 'disassemble_left/lerobot_dataset')


def submit(worker, timestamp=0.):
    return worker.submit({'rgb': np.zeros((2, 3, 3), np.uint8)}, 'disassemble_left', timestamp)


def test_prediction_runs_off_main_thread_while_render_callback_stays_on_main_thread():
    policy = BlockingPolicy()
    main_thread = threading.get_ident()
    ui_threads = []
    simulation = SimpleNamespace(physics_steps=0, renders=0)

    def render_only():
        ui_threads.append(threading.get_ident())
        simulation.renders += 1
        if simulation.renders >= 2:
            policy.release.set()

    try:
        with worker_for(policy) as worker:
            submit(worker)
            assert policy.entered.wait(3)
            action, response = worker.result(wait_callback=render_only, poll_interval=.001)
    finally:
        policy.release.set()
    assert len(policy.calls) == 1
    assert policy.calls[0][0] != main_thread
    assert ui_threads and set(ui_threads) == {main_thread}
    assert simulation.renders >= 2 and simulation.physics_steps == 0
    np.testing.assert_array_equal(action, [.1, 0, .5, 0, 0, 0, 1, 1])
    assert response['timing_ms'] == {'policy': 12.}
    assert policy.calls[0][2:] == ('remove the left part', 0., 'disassemble_left', 'disassemble_left/lerobot_dataset')


def test_pending_request_cannot_be_overwritten_or_queued():
    policy = BlockingPolicy()
    try:
        with worker_for(policy) as worker:
            submit(worker)
            assert policy.entered.wait(3)
            with pytest.raises(RuntimeError):
                submit(worker, timestamp=1.)
            assert len(policy.calls) == 1
            policy.release.set()
            worker.result()
    finally:
        policy.release.set()


def test_completed_result_must_be_consumed_before_next_request():
    policy = BlockingPolicy()
    policy.release.set()
    with worker_for(policy) as worker:
        submit(worker)
        assert policy.finished.wait(3)
        with pytest.raises(RuntimeError):
            submit(worker, timestamp=1.)
        worker.result()
        submit(worker, timestamp=1.)
        worker.result()
    assert len(policy.calls) == 2
    assert [call[3] for call in policy.calls] == [0., 1.]


def test_http_exception_is_returned_without_automatic_retry():
    error = TimeoutError('request completion is unknown')
    policy = BlockingPolicy(error=error)
    policy.release.set()
    with worker_for(policy) as worker:
        submit(worker)
        with pytest.raises(TimeoutError) as raised:
            worker.result()
        with pytest.raises(RuntimeError):
            submit(worker, timestamp=1.)
    assert raised.value is error
    assert len(policy.calls) == 1


def test_missing_result_is_rejected():
    with worker_for(BlockingPolicy()) as worker:
        with pytest.raises(RuntimeError):
            worker.result()


def test_render_callback_error_stops_waiting_without_retry_or_simulator_calls_from_worker():
    policy = BlockingPolicy()
    main_thread = threading.get_ident()
    callbacks = []
    error = RuntimeError('simulation application stopped')

    def stopped():
        callbacks.append(threading.get_ident())
        raise error

    try:
        with worker_for(policy) as worker:
            submit(worker)
            assert policy.entered.wait(3)
            with pytest.raises(RuntimeError) as raised:
                worker.result(wait_callback=stopped, poll_interval=.001)
            assert raised.value is error
            with pytest.raises(RuntimeError):
                submit(worker, timestamp=1.)
    finally:
        policy.release.set()
        assert policy.finished.wait(3)
    assert callbacks == [main_thread]
    assert len(policy.calls) == 1


def test_reset_and_predict_share_one_persistent_worker_thread():
    policy = BlockingPolicy()
    policy.release.set()
    main_thread = threading.get_ident()
    with worker_for(policy) as worker:
        worker.submit_reset('episode-one')
        assert worker.result() == {'episode_id': 'episode-one'}
        submit(worker)
        worker.result()
        worker.submit_reset()
        assert worker.result() == {'episode_id': 'fresh-episode'}
    threads = [row[0] for row in policy.calls + policy.resets]
    assert len(set(threads)) == 1 and threads[0] != main_thread
    assert [row[1] for row in policy.resets] == ['episode-one', None]


def test_failed_reset_closes_worker_and_is_not_retried():
    class BrokenReset(BlockingPolicy):
        def reset(self, episode=None):
            self.resets.append((threading.get_ident(), episode))
            raise ConnectionError('server unavailable')

    policy = BrokenReset()
    with worker_for(policy) as worker:
        worker.submit_reset('episode-one')
        with pytest.raises(ConnectionError, match='server unavailable'):
            worker.result()
        with pytest.raises(RuntimeError):
            worker.submit_reset('episode-two')
    assert len(policy.resets) == 1 and not policy.calls


def test_close_does_not_wait_for_http_or_accept_more_requests():
    policy = BlockingPolicy()
    worker = worker_for(policy)
    try:
        submit(worker)
        assert policy.entered.wait(3)
        worker.close()
        assert not policy.finished.is_set()
        with pytest.raises(RuntimeError):
            submit(worker, timestamp=1.)
    finally:
        policy.release.set()
        assert policy.finished.wait(3)
        worker.close()
    assert len(policy.calls) == 1


def test_wait_refresh_renders_on_calling_thread_without_physics_or_app_update():
    calls = []
    main_thread = threading.get_ident()

    def forbidden():
        raise AssertionError('Waiting for inference must not advance physics')

    env = SimpleNamespace(sim=SimpleNamespace(has_gui=lambda: True,
                                             render=lambda: calls.append(threading.get_ident())), step=forbidden)
    reference = SimpleNamespace(simulation_app=SimpleNamespace(is_running=lambda: True, update=forbidden))
    refresh_while_waiting(reference, env)
    assert calls == [main_thread]


def test_headless_wait_checks_application_without_rendering_camera_frames():
    def forbidden():
        raise AssertionError('Headless HTTP waits must not rerender sensors')

    env = SimpleNamespace(sim=SimpleNamespace(has_gui=lambda: False, render=forbidden), step=forbidden)
    reference = SimpleNamespace(simulation_app=SimpleNamespace(is_running=lambda: True),
                                _replicator_render_tick=forbidden)
    refresh_while_waiting(reference, env)


def test_legacy_wait_refresh_uses_its_render_only_hook():
    env = SimpleNamespace()
    calls = []
    reference = SimpleNamespace(simulation_app=SimpleNamespace(is_running=lambda: True),
                                _replicator_render_tick=lambda selected: calls.append(selected))
    refresh_while_waiting(reference, env)
    assert calls == [env]


def test_legacy_headless_wait_does_not_require_a_render_hook():
    reference = SimpleNamespace(simulation_app=SimpleNamespace(is_running=lambda: True),
                                args_cli=SimpleNamespace(headless=True))
    refresh_while_waiting(reference, SimpleNamespace())


@pytest.mark.parametrize('close_during_render', [False, True])
def test_wait_refresh_detects_window_close_before_and_after_render(close_during_render):
    state = SimpleNamespace(running=close_during_render, renders=0)

    def render():
        state.renders += 1
        state.running = False

    env = SimpleNamespace(sim=SimpleNamespace(has_gui=lambda: True, render=render))
    reference = SimpleNamespace(simulation_app=SimpleNamespace(is_running=lambda: state.running))
    with pytest.raises(RuntimeError, match='Simulation application stopped'):
        refresh_while_waiting(reference, env)
    assert state.renders == int(close_during_render)

