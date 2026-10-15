"""Read real TensorBoard events from live OHT logging, resume and JSONL export."""
import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tests.test_oht_migration import replay_fixture  # noqa: F401
from finetune.OHT import train, training_log
from finetune.OHT.config import load
from finetune.OHT.training_log import TensorBoardLogger
from tools.export_oht_tensorboard import export, main as export_main


def scalars(path, tag):
    events = EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()
    return [(event.step, event.value) for event in events.Scalars(tag)]


def test_logger_flushes_on_exception_and_records_learning_rates(tmp_path):
    output = tmp_path / "tensorboard"
    with pytest.raises(RuntimeError, match="interrupted"):
        with TensorBoardLogger(output, flush_secs=3600) as logger:
            logger.log(1, {"total_loss": 2., "description": "not a scalar"}, [.01, .02])
            raise RuntimeError("interrupted")
    assert scalars(output, "train/total_loss") == [(1, 2.)]
    assert scalars(output, "train/learning_rate")[0][1] == pytest.approx(.01)
    assert scalars(output, "train/learning_rate_group_1")[0][1] == pytest.approx(.02)


@pytest.mark.parametrize("settings", [{"rank": 1}, {"backend": "none"}])
def test_nonzero_rank_and_disabled_logging_create_no_events(tmp_path, monkeypatch, settings):
    def unexpected():
        raise AssertionError("Disabled writers must not import TensorBoard")
    monkeypatch.setattr(training_log, "summary_writer_class", unexpected)
    output = tmp_path / "tensorboard"
    with TensorBoardLogger(output, **settings) as logger:
        logger.log(1, {"total_loss": 2.})
        logger.flush()
    assert not output.exists()


def test_missing_tensorboard_has_actionable_error(monkeypatch):
    original = builtins.__import__
    def without_tensorboard(name, *args, **kwargs):
        if name == "torch.utils.tensorboard":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_tensorboard)
    with pytest.raises(RuntimeError, match="pip install tensorboard"):
        training_log.summary_writer_class()


@pytest.fixture
def cpu_training(monkeypatch):
    # Exercise the production CLI, sampling, accumulation, checkpoint and log
    # paths, replacing only CUDA transfers and the large model with a CPU agent.
    from torch.utils.data import DataLoader
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.utils.data, "DataLoader", lambda *args, **kwargs:
                        DataLoader(*args, **dict(kwargs, pin_memory=False)))
    monkeypatch.setattr(train, "to_device", lambda batch, device: batch)
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)

    def build(config, *args, **kwargs):
        network = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(network.parameters(), lr=config["learning_rate"])
        def update(batch, loss_scale, reset_gradients, step_optimizer):
            if reset_gradients:
                optimizer.zero_grad()
            loss = network.weight.square().mean()
            (loss * loss_scale).backward()
            if step_optimizer:
                optimizer.step()
            return {"total_loss": float(loss.detach()), "collision_loss": 0.}
        return SimpleNamespace(_net_mod=network, _optimizer=optimizer, update=update)
    monkeypatch.setattr(train, "build_agent", build)


def test_train_and_resume_write_matching_jsonl_and_events(replay_fixture, tmp_path, cpu_training):
    config = load(Path(__file__).resolve().parents[1] / "finetune/OHT/configs/smoke.yaml")
    config["accumulation_steps"] = 2
    path, output = tmp_path / "train.yaml", tmp_path / "run"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    args = ["--config", str(path), "--replay", str(replay_fixture.replay), "--output", str(output)]
    assert train.main(args) == 0
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in rows] == [1, 2]
    assert np.allclose([value for _, value in scalars(output / "tensorboard", "train/total_loss")],
                       [row["total_loss"] for row in rows])
    # Simulate events written after the last saved checkpoint, then resume it.
    with TensorBoardLogger(output / "tensorboard") as logger:
        logger.log(3, {"total_loss": 999.})
        logger.log(4, {"total_loss": 999.})
    config["optimizer_steps"] = 3
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert train.main([*args, "--resume", str(output / "model_last.pth"),
                       "--tensorboard-flush-secs", "1"]) == 0
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    events = scalars(output / "tensorboard", "train/total_loss")
    assert [step for step, _ in events] == [1, 2, 3]
    assert np.allclose([value for _, value in events], [row["total_loss"] for row in rows])
    assert [step for step, _ in scalars(output / "tensorboard", "train/learning_rate")] == [1, 2, 3]


def test_train_none_keeps_jsonl_without_events(replay_fixture, tmp_path, cpu_training):
    config = Path(__file__).resolve().parents[1] / "finetune/OHT/configs/smoke.yaml"
    output = tmp_path / "run"
    assert train.main(["--config", str(config), "--replay", str(replay_fixture.replay),
                       "--output", str(output), "--log-backend", "none"]) == 0
    assert len((output / "metrics.jsonl").read_text().splitlines()) == 2
    assert not (output / "tensorboard").exists()


def test_preflight_does_not_require_tensorboard(replay_fixture, tmp_path, monkeypatch):
    def unexpected():
        raise AssertionError("Data preflight must not require TensorBoard")
    monkeypatch.setattr(train, "summary_writer_class", unexpected)
    config = Path(__file__).resolve().parents[1] / "finetune/OHT/configs/smoke.yaml"
    output = tmp_path / "unused"
    assert train.main(["--config", str(config), "--replay", str(replay_fixture.replay),
                       "--output", str(output), "--validate-only"]) == 0
    assert not output.exists()


def test_export_replays_resume_branch_without_modifying_source(tmp_path, capsys):
    metrics = tmp_path / "metrics.jsonl"
    rows = [{"step": step, "total_loss": value} for step, value in
            [(1, 4.), (2, 3.), (3, 999.), (4, 999.), (2, 2.5), (3, 2.)]]
    metrics.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    before = metrics.read_bytes()
    assert export_main(["--run", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    output = tmp_path / "tensorboard-from-jsonl"
    assert report["steps"] == 3
    assert scalars(output, "train/total_loss") == [(1, 4.), (2, 2.5), (3, 2.)]
    assert metrics.read_bytes() == before
    with pytest.raises(FileExistsError):
        export(metrics, output)


@pytest.mark.parametrize("content", ["", '{"step": 0}\n', '{"step": 1.5}\n',
                                     '{"step": 1, "total_loss": NaN}\n', '{"step": 1'])
def test_export_rejects_invalid_metrics_before_creating_events(tmp_path, content):
    metrics, output = tmp_path / "metrics.jsonl", tmp_path / "events"
    metrics.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        export(metrics, output)
    assert not output.exists()
