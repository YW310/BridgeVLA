"""HTTP lifecycle, real prepared observations, concurrency and policy parity."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.test_oht_migration import replay_fixture, observation_only  # noqa: F401
from tests.test_oht_action_modes import patch_mvt, caches  # noqa: F401
from finetune.OHT.action_modes import action_options, resolve_action
from finetune.OHT.config import load
from finetune.OHT.data.dataset import OHTDataset
from finetune.OHT.data.replay import load_contract
from finetune.OHT.model import build_agent
from finetune.OHT.runtime.inference_api import API_VERSION, OBSERVATION_FORMAT
from finetune.OHT.runtime.inference_service import InferenceService, resolve_checkpoint
from finetune.OHT.runtime.policy import Policy
from finetune.OHT.runtime.predicted_wrapper import policy_observation, PredictedObjectWrapper
from finetune.OHT.runtime.transport import pack_observation
from finetune.OHT.vla_server import create_app, build_arg_parser

ROOT = Path(__file__).resolve().parents[1]


class FakePolicy:
    def __init__(self, contract):
        self.contract, self.mode = contract, "baseline"
        self.action_config = action_options()
        self.training_config = {"mvt": {"rend_three_views": True}}
        self.calls, self.threads, self.resets = [], [], 0
        self.entered, self.release = threading.Event(), threading.Event()
        self.release.set()
        self.fail = False

    def reset(self):
        self.threads.append(threading.get_ident())
        self.resets += 1

    def act(self, observation, instruction, step):
        self.threads.append(threading.get_ident())
        self.calls.append((observation, instruction, step))
        self.entered.set()
        if not self.release.wait(5):
            raise RuntimeError("Test release was not signalled")
        if self.fail:
            raise RuntimeError("private inference traceback")
        return np.array([.2, 0, .5, 0, 0, 0, 1, 1], dtype=np.float32)


@pytest.fixture
def fixture(replay_fixture, tmp_path):
    checkpoint = tmp_path / "run" / "model_last.pth"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"test checkpoint: model loading is injected")
    policy = FakePolicy(load_contract(replay_fixture.replay))
    raw, goal = observation_only(replay_fixture)
    observation = policy_observation(raw, policy.contract["data_config"]["cameras"], policy.action_config)
    loads = []

    def loader(*args):
        loads.append((args, threading.get_ident()))
        return policy

    def app(**kwargs):
        return create_app(checkpoint.parent, policy_loader=loader,
                          output_dir=tmp_path / "logs", **kwargs)

    return SimpleNamespace(policy=policy, checkpoint=checkpoint, observation=observation,
                           goal=goal, app=app, loads=loads, logs=tmp_path / "logs")


def envelope(info, episode="case-1", request_id="request-0"):
    return dict(api_version=API_VERSION, request_id=request_id, episode_id=episode,
                model_id=info["model_id"], contract_sha256=info["contract_sha256"])


def prediction(fixture, info, step=0, episode="case-1", request_id=None):
    return dict(**envelope(info, episode, request_id or f"{episode}-{step}"),
                step=step, timestamp=step*.1, instruction=fixture.goal,
                observation_format=OBSERVATION_FORMAT,
                observation=pack_observation(fixture.observation))


def test_lifespan_loads_once_http_session_and_json_logs(fixture):
    app = fixture.app()
    assert fixture.loads == [] and not fixture.logs.exists()
    with TestClient(app) as client:
        assert client.get("/health").json()["ready"]
        info = client.get("/model_info").json()
        assert len(fixture.loads) == 1
        assert info["observation_schema"]["global_left_rgb"]["shape"] == [3, 8, 8]
        assert "oracle_target_object_points" not in info["observation_schema"]
        body = prediction(fixture, info)
        reply = client.post("/predict", json=body)
        assert reply.status_code == 200, reply.text
        result = reply.json()
        np.testing.assert_allclose(result["action"], [.2, 0, .5, 0, 0, 0, 1, 1])
        assert all(result[key] == body[key] for key in ("request_id", "episode_id", "step", "timestamp"))
        assert set(result["timing_ms"]) == {"decode", "prepare", "policy", "total"}
        assert client.post("/predict", json=body).status_code == 409
        changed = prediction(fixture, info, 1)
        changed["instruction"] += " changed"
        assert client.post("/predict", json=changed).status_code == 409
        assert client.post("/predict", json=prediction(fixture, info, 1)).status_code == 200
        assert client.post("/predict", json=prediction(fixture, info, episode="case-2")).status_code == 409
        reset = envelope(info, "case-2", "reset-2")
        assert client.post("/reset", json=reset).json()["next_step"] == 0
        assert client.post("/predict", json=prediction(fixture, info, episode="case-2")).status_code == 200
        assert client.post("/reset", json=envelope(info, "case-1", "reset-old")).status_code == 409
        assert len(fixture.loads) == 1
    assert len(set(fixture.policy.threads + [fixture.loads[0][1]])) == 1
    assert threading.get_ident() not in fixture.policy.threads
    assert not app.state.service.health()["ready"]
    metadata = json.loads((fixture.logs / "info.json").read_text(encoding="utf-8"))
    assert metadata["render_config"]["rend_three_views"] is True
    records = [json.loads(row) for row in (fixture.logs / "requests.jsonl").read_text().splitlines()]
    assert any(row["status"] == 409 for row in records)
    assert any(row.get("action_summary", {}).get("length") == 8 for row in records)
    assert all("observation" not in row and "instruction" not in row for row in records)


@pytest.mark.parametrize("kind", ["teacher", "shape", "dtype", "base64", "extrinsic", "missing",
                                 "version", "model", "step", "timestamp", "format", "unknown"])
def test_bad_requests_do_not_consume_step_or_call_policy(fixture, kind):
    with TestClient(fixture.app()) as client:
        info = client.get("/model_info").json()
        bad = prediction(fixture, info)
        if kind == "teacher":
            bad["observation"]["oracle_target_object_points"] = bad["observation"]["low_dim_state"]
        elif kind == "shape":
            bad["observation"]["global_left_rgb"]["shape"] = [3, 999999999, 8]
        elif kind == "dtype":
            bad["observation"]["global_left_rgb"]["dtype"] = "object"
        elif kind == "base64":
            bad["observation"]["global_left_rgb"]["data"] = "!invalid!"
        elif kind == "extrinsic":
            bad["observation"].update(pack_observation({"global_left_camera_extrinsics": np.zeros((4, 4), np.float32)}))
        elif kind == "missing":
            bad["observation"].pop("global_left_depth")
        elif kind == "version":
            bad["api_version"] = "unknown"
        elif kind == "model":
            bad["model_id"] = "another model"
        elif kind == "step":
            bad["step"] = True
        elif kind == "timestamp":
            bad["timestamp"] = -1
        elif kind == "format":
            bad["observation_format"] = "json_numpy"
        elif kind == "unknown":
            bad["proprio"] = [1, 2, 3]
        assert client.post("/predict", json=bad).status_code in (409, 422)
        assert not fixture.policy.calls
        assert client.get("/health").json()["next_step"] == 0
        assert client.post("/predict", json=prediction(fixture, info)).status_code == 200


def test_actual_body_size_invalid_json_and_media_type(fixture):
    with TestClient(fixture.app(max_bytes=100)) as client:
        assert client.post("/predict", content=b"x"*101, headers={"Content-Type": "application/json"}).status_code == 413
        # No Content-Length: enforce bytes actually read, including chunked bodies.
        assert client.post("/predict", content=iter([b"x"*60, b"y"*60]),
                           headers={"Content-Type": "application/json"}).status_code == 413
        assert client.post("/predict", content=b"{", headers={"Content-Type": "application/json"}).status_code == 400
        assert client.post("/predict", content=b'{"timestamp":NaN}',
                           headers={"Content-Type": "application/json"}).status_code == 400
        assert client.post("/predict", content=b"{}").status_code == 415
        assert not fixture.policy.calls


def test_health_responds_during_inference_and_reset_cannot_interleave(fixture):
    with TestClient(fixture.app()) as client:
        info = client.get("/model_info").json()
        fixture.policy.release.clear()
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(client.post, "/predict", json=prediction(fixture, info))
            try:
                assert fixture.policy.entered.wait(3)
                health = client.get("/health")
                assert health.status_code == 200 and health.json()["busy"]
                assert client.get("/model_info").status_code == 200
                assert client.post("/predict", json=prediction(fixture, info, 1)).status_code == 429
                assert client.post("/reset", json=envelope(info, "case-2", "reset")).status_code == 429
            finally:
                fixture.policy.release.set()
            assert future.result(timeout=3).status_code == 200
        assert len(fixture.policy.calls) == 1


def test_cancelled_http_waiter_does_not_release_worker_or_replay_action(fixture):
    async def run():
        app = fixture.app()
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                info = (await client.get("/model_info")).json()
                fixture.policy.release.clear()
                body = prediction(fixture, info)
                task = asyncio.create_task(client.post("/predict", json=body))
                try:
                    assert await asyncio.to_thread(fixture.policy.entered.wait, 3)
                    task.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await task
                    assert (await client.get("/health")).json()["busy"]
                    assert (await client.post("/predict", json=body)).status_code == 429
                finally:
                    fixture.policy.release.set()
                for _ in range(100):
                    if not app.state.service.health()["busy"]:
                        break
                    await asyncio.sleep(.01)
                assert app.state.service.health()["next_step"] == 1
                assert (await client.post("/predict", json=body)).status_code == 409
                assert len(fixture.policy.calls) == 1
    asyncio.run(run())


def test_policy_failure_requires_reset_without_leaking_exception(fixture):
    with TestClient(fixture.app()) as client:
        info = client.get("/model_info").json()
        fixture.policy.fail = True
        reply = client.post("/predict", json=prediction(fixture, info))
        assert reply.status_code == 500
        assert "private" not in reply.text
        assert client.get("/health").json()["episode_failed"]
        assert client.post("/predict", json=prediction(fixture, info, request_id="retry")).status_code == 409
        fixture.policy.fail = False
        assert client.post("/reset", json=envelope(info, "recovered", "recover")).status_code == 200
        assert client.post("/predict", json=prediction(fixture, info, episode="recovered")).status_code == 200


def test_optional_warmup_reads_replay_but_never_passes_labels(fixture, replay_fixture):
    with TestClient(fixture.app(warmup_replay=replay_fixture.replay, warmup_index=0)) as client:
        assert client.get("/health").json()["episode_id"] is None
        assert len(fixture.policy.calls) == 1 and fixture.policy.resets == 1
        observation = fixture.policy.calls[0][0]
        assert set(observation) == set(fixture.observation)
        assert not any("oracle" in key or "action" in key or key == "goal" for key in observation)


@pytest.mark.parametrize("object_mode", ["baseline", "role_queries", "predicted_external"])
@pytest.mark.parametrize("action_mode", ["eef", "relative_eef", "joint", "relative_joint"])
def test_http_matches_actual_policy_for_all_supported_modes(fixture, replay_fixture, caches, patch_mvt,
                                                           object_mode, action_mode):
    import torch
    config = load(ROOT / f"finetune/OHT/configs/{object_mode}.yaml")
    config.update(action_mode=action_mode, point_count=8)
    config["action"]["hidden_dim"] = 16
    options = resolve_action(config)
    dataset = OHTDataset(replay_fixture.replay, action_config=options)
    sample = dataset[0]
    agent = build_agent(config, dataset.contract, torch.device("cpu"), training=False)
    wrapper = None
    if object_mode == "predicted_external":
        def predictor(observation, instruction):
            assert not any("oracle" in key or "action" in key for key in observation)
            return caches[1]
        wrapper = PredictedObjectWrapper(predictor, dataset.contract["data_config"]["cameras"], 8)
    policy = Policy(agent, dataset.contract, "cpu", object_mode, wrapper)
    fixture.observation = policy_observation(sample, dataset.contract["data_config"]["cameras"], options)
    fixture.goal = sample["goal"]
    expected = policy.act(fixture.observation, fixture.goal, 0)
    app = create_app(fixture.checkpoint, output_dir=fixture.logs, policy_loader=lambda *args: policy)
    with TestClient(app) as client:
        info = client.get("/model_info").json()
        actual = client.post("/predict", json=prediction(fixture, info))
        assert actual.status_code == 200, actual.text
        np.testing.assert_allclose(actual.json()["action"], expected, atol=1e-6)


def test_paths_cli_and_startup_failure_cleanup(fixture, tmp_path):
    assert resolve_checkpoint(fixture.checkpoint.parent) == fixture.checkpoint
    missing = tmp_path / "empty"
    missing.mkdir()
    with pytest.raises(ValueError, match="model_last.pth"):
        resolve_checkpoint(missing)
    parser = build_arg_parser()
    assert parser.parse_args(["--checkpoint", str(fixture.checkpoint)]).model_path == str(fixture.checkpoint)
    with pytest.raises(SystemExit):
        parser.parse_args(["--checkpoint", "a", "--model-path", "b"])
    def broken(*args):
        raise RuntimeError("load failed")
    app = create_app(fixture.checkpoint, output_dir=fixture.logs, policy_loader=broken)
    with pytest.raises(RuntimeError, match="load failed"):
        with TestClient(app):
            pass
    assert not app.state.service.health()["ready"]


def test_shutdown_waits_for_running_model_and_rejects_new_requests(fixture):
    async def run():
        service = InferenceService(fixture.checkpoint, output_dir=fixture.logs,
                                   policy_loader=lambda *args: fixture.policy)
        await service.start()
        fixture.policy.release.clear()
        request = json.dumps(prediction(fixture, service.model_info())).encode()
        work = asyncio.create_task(service.execute("predict", request))
        assert await asyncio.to_thread(fixture.policy.entered.wait, 3)
        shutdown = asyncio.create_task(service.close())
        try:
            await asyncio.sleep(.01)
            assert not shutdown.done() and not service.health()["ready"]
            from finetune.OHT.runtime.inference_api import APIError
            with pytest.raises(APIError) as caught:
                await service.execute("predict", request)
            assert caught.value.status == 503
        finally:
            fixture.policy.release.set()
        assert (await work)["step"] == 0
        await shutdown
        assert service.policy is None
    asyncio.run(run())


def test_rejected_requests_are_logged_and_logs_rotate(fixture):
    with TestClient(fixture.app(log_max_bytes=600, log_backups=1)) as client:
        for _ in range(12):
            assert client.post("/predict", content=b"{}").status_code == 415
    paths = sorted(fixture.logs.glob("requests.jsonl*"))
    assert len(paths) == 2
    for path in paths:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert all(row["status"] == 415 for row in rows)


def test_replay_smoke_tool_uses_a_real_uvicorn_http_connection(fixture, replay_fixture):
    import uvicorn
    from tools.smoke_oht_vla_server import smoke
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    address = f"http://127.0.0.1:{listener.getsockname()[1]}"
    server = uvicorn.Server(uvicorn.Config(fixture.app(), log_level="error", access_log=False, ws="none"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        for _ in range(300):
            if server.started:
                break
            if not thread.is_alive():
                pytest.fail("Uvicorn failed to start")
            time.sleep(.01)
        assert server.started
        result = smoke(address, replay_fixture.replay, timeout=5.)
        assert result["step"] == 0 and len(result["action"]) == 8
        assert fixture.policy.calls[-1][1] == result["instruction"]
        assert not any("oracle" in key or "action" in key for key in fixture.policy.calls[-1][0])
    finally:
        server.should_exit = True
        thread.join(timeout=5.)
        listener.close()
    assert not thread.is_alive()
