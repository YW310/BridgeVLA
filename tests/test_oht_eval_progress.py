"""Evaluation diagnostics stay responsive during loading and a slow first sample."""
import io
import json
import threading
from types import SimpleNamespace

import pytest

from tests.test_oht_migration import replay_fixture, FakeAgent  # noqa: F401
from finetune.OHT import eval as evaluation
from finetune.OHT.data.replay import load_contract
from finetune.OHT.runtime import loading
from finetune.OHT.runtime.policy import Policy
from finetune.OHT.runtime.progress import ProgressReporter


def test_heartbeat_reports_before_first_prediction_returns(replay_fixture):
    heartbeat = threading.Event()

    class Stream(io.StringIO):
        def write(self, text):
            result = super().write(text)
            if self.getvalue().count("stage=predict ") >= 2:
                heartbeat.set()
            return result

    class SlowAgent(FakeAgent):
        def act(self, *args, **kwargs):
            assert heartbeat.wait(3), "No status emitted while first prediction was blocked"
            return super().act(*args, **kwargs)

    stream = Stream()
    policy = Policy(SlowAgent(), load_contract(replay_fixture.replay), "cpu", "baseline")
    with ProgressReporter(.01, stream=stream) as progress:
        result = evaluation.open_loop(policy, replay_fixture.replay, limit=2, progress=progress)
    assert not progress._thread.is_alive()
    assert result["all_samples"]["samples"] == 2
    text = stream.getvalue()
    for stage in ("prepare_replay", "samples_ready", "read_sample", "predict", "sample_done", "summarize"):
        assert f"stage={stage} " in text
    assert "completed=2 total=2" in text and "eta_s=0.0" in text
    assert "read_s=" in text and "predict_s=" in text


def test_reporter_stops_on_failure_and_does_not_swallow_error():
    stream = io.StringIO()
    with pytest.raises(RuntimeError, match="original error"):
        with ProgressReporter(.01, stream=stream) as progress:
            progress.update("apply_checkpoint")
            raise RuntimeError("original error")
    assert not progress._thread.is_alive()
    assert "stage=failed" in stream.getvalue()
    assert "original error" in stream.getvalue()


@pytest.mark.parametrize("interval", [0, -1, float("nan"), float("inf")])
def test_invalid_progress_intervals(interval):
    with pytest.raises(ValueError, match="finite and positive"):
        ProgressReporter(interval)


def test_loader_reports_finetuned_weights_after_base_construction(replay_fixture, tmp_path, monkeypatch):
    import torch
    contract = load_contract(replay_fixture.replay)
    checkpoint = tmp_path / "checkpoint.pth"
    torch.save(dict(config={"mode": "baseline"}, oht_contract=contract), checkpoint)
    events = []
    agent = FakeAgent()
    def build(*args, **kwargs):
        assert kwargs["training"] is False
        events.append("constructed")
        return agent
    def weights(current, saved):
        assert current is agent and saved["oht_contract"] == contract
        events.append("weights_applied")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(loading, "build_agent", build)
    monkeypatch.setattr(loading, "load_weights", weights)
    progress = SimpleNamespace(update=lambda stage, **kwargs: events.append(stage))
    policy = loading.load_policy(checkpoint, progress=progress)
    assert policy.agent is agent
    assert events == ["read_checkpoint", "validate_contract", "build_agent", "constructed",
                      "apply_checkpoint", "weights_applied", "release_checkpoint", "model_ready"]


@pytest.mark.parametrize("quiet", [False, True])
def test_cli_writes_report_and_keeps_progress_on_stderr(replay_fixture, tmp_path, monkeypatch, capsys, quiet):
    checkpoint = tmp_path / "weights.pth"
    checkpoint.write_bytes(b"fixture weights")
    output = tmp_path / "evaluation.json"
    policy = Policy(FakeAgent(), load_contract(replay_fixture.replay), "cpu", "baseline")
    def load(*args, **kwargs):
        kwargs["progress"].update("model_ready")
        return policy
    monkeypatch.setattr(loading, "load_policy", load)
    args = ["open-loop", "--checkpoint", str(checkpoint), "--replay", str(replay_fixture.replay),
            "--output", str(output), "--limit", "2", "--progress-interval", ".01"]
    assert evaluation.main(args + (["--quiet"] if quiet else [])) == 0
    captured = capsys.readouterr()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert json.loads(captured.out)["all_samples"]["samples"] == 2
    assert len(report["predictions"]) == 2 and report["checkpoint_sha256"]
    if quiet:
        assert captured.err == ""
    else:
        assert "stage=hash_checkpoint" in captured.err
        assert "stage=write_report" in captured.err
        assert "stage=complete" in captured.err
