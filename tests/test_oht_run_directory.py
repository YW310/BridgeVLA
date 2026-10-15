"""Timestamped OHT runs stay unique, rank-consistent and stable on resume."""
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tests.test_oht_migration import replay_fixture  # noqa: F401
from tests.test_oht_training_log import cpu_training  # noqa: F401
from finetune.OHT import train
from finetune.OHT.config import load
from finetune.OHT.run_directory import RUN_TIMEZONE, prepare_run_directory

NOW = datetime(2026, 10, 15, 14, 30, 12, tzinfo=RUN_TIMEZONE)
ROOT = Path(__file__).resolve().parents[1]


def arguments(tmp_path, **overrides):
    return SimpleNamespace(**dict(dict(config="oht_full_8x40.yaml", output=None,
                                      log_dir=str(tmp_path / "logs"), exp_id=None,
                                      exp_note="baseline_s0", resume=None), **overrides))


def test_timestamped_layout_and_complete_creation_metadata(tmp_path):
    args = arguments(tmp_path)
    output = prepare_run_directory(args, {}, now=NOW)
    assert output == tmp_path / "logs/train/oht_full_8x40/baseline_s0/10_15_14_30"
    metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
    assert metadata["created_at"] == "2026-10-15T14:30:12+08:00"
    assert metadata["timezone"] == "Asia/Shanghai"
    assert metadata["output"] == str(output)
    assert metadata["arguments"]["exp_note"] == "baseline_s0"


def test_same_minute_runs_are_unique_and_preserve_previous_data(tmp_path):
    args = arguments(tmp_path)
    first = prepare_run_directory(args, {}, now=NOW)
    before = (first / "run.json").read_bytes()
    (first / "model_last.pth").write_bytes(b"keep checkpoint")
    second = prepare_run_directory(args, {}, now=NOW)
    assert second.name == first.name + "_01"
    assert first.parent == second.parent
    assert (first / "run.json").read_bytes() == before
    assert (first / "model_last.pth").read_bytes() == b"keep checkpoint"


def test_default_root_config_name_and_experiment_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = arguments(tmp_path, log_dir=None, exp_note="")
    output = prepare_run_directory(args, {"exp_id": "configured"}, now=NOW)
    assert output == tmp_path / "exp/OHT/train/configured/10_15_14_30"
    args.exp_id = "overridden"
    second = prepare_run_directory(args, {"exp_id": "configured"}, now=NOW)
    assert second.parent.name == "overridden"


def test_explicit_output_keeps_legacy_exact_path_and_refuses_overwrite(tmp_path):
    output = tmp_path / "exact run"
    args = arguments(tmp_path, output=str(output))
    assert prepare_run_directory(args, {}, now=NOW) == output
    before = (output / "run.json").read_bytes()
    with pytest.raises(FileExistsError):
        prepare_run_directory(args, {}, now=NOW)
    assert (output / "run.json").read_bytes() == before


def test_resume_uses_checkpoint_parent_and_preserves_original_timestamp(tmp_path):
    args = arguments(tmp_path)
    output = prepare_run_directory(args, {}, now=NOW)
    checkpoint = output / "model_last.pth"
    checkpoint.write_bytes(b"checkpoint")
    before = (output / "run.json").read_bytes()
    args.resume = str(checkpoint)
    args.log_dir = str(tmp_path / "different root")
    args.exp_note = "different note"
    assert prepare_run_directory(args, {}, now=NOW) == output
    assert (output / "run.json").read_bytes() == before
    assert not Path(args.log_dir).exists()
    args.output = str(tmp_path / "wrong run")
    with pytest.raises(ValueError, match="match the resume checkpoint directory"):
        prepare_run_directory(args, {}, now=NOW)
    assert not Path(args.output).exists()


@pytest.mark.parametrize("overrides", [{"exp_id": "../outside"}, {"exp_note": "a/b"},
                                       {"exp_note": ".."}, {"exp_id": "a\\b"}])
def test_experiment_labels_are_directory_names(tmp_path, overrides):
    with pytest.raises(ValueError, match="directory name"):
        prepare_run_directory(arguments(tmp_path, **overrides), {}, now=NOW)
    assert not (tmp_path / "logs").exists()


class Broadcast:
    def __init__(self):
        self.message = None
    def broadcast_object_list(self, message, src):
        assert src == 0
        if message[0] is not None:
            self.message = message[0]
        else:
            message[0] = self.message


def test_ddp_ranks_share_rank_zero_path_and_create_only_one_run(tmp_path):
    args, distributed = arguments(tmp_path), Broadcast()
    first = prepare_run_directory(args, {}, rank=0, world_size=2, distributed=distributed, now=NOW)
    second = prepare_run_directory(args, {}, rank=1, world_size=2, distributed=distributed,
                                   now=NOW.replace(minute=31))
    assert first == second
    assert len(list((tmp_path / "logs").rglob("run.json"))) == 1


def test_ddp_directory_errors_are_propagated_to_every_rank(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    args, distributed = arguments(tmp_path, output=str(output)), Broadcast()
    for rank in (0, 1):
        with pytest.raises(FileExistsError):
            prepare_run_directory(args, {}, rank=rank, world_size=2, distributed=distributed, now=NOW)
    assert not list(output.iterdir())


def test_training_and_resume_keep_all_artifacts_in_one_timestamped_run(replay_fixture, tmp_path, cpu_training):
    config = load(ROOT / "finetune/OHT/configs/smoke.yaml")
    path, logs = tmp_path / "smoke.yaml", tmp_path / "logs"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    args = ["--config", str(path), "--replay", str(replay_fixture.replay),
            "--log_dir", str(logs), "--exp_note", "seed 0"]
    assert train.main(args) == 0
    runs = list((logs / "train/smoke/seed 0").iterdir())
    assert len(runs) == 1
    output = runs[0]
    for name in ("run.json", "config.json", "contract.json", "training_plan.json",
                 "metrics.jsonl", "model_last.pth"):
        assert (output / name).is_file()
    assert list((output / "tensorboard").glob("events.out.tfevents.*"))
    before = (output / "run.json").read_bytes()
    config["optimizer_steps"] = 3
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert train.main([*args, "--resume", str(output / "model_last.pth")]) == 0
    assert (output / "run.json").read_bytes() == before
    assert len(list((logs / "train/smoke/seed 0").iterdir())) == 1
    assert [json.loads(line)["step"] for line in (output / "metrics.jsonl").read_text().splitlines()] == [1, 2, 3]


def test_preflight_with_log_dir_creates_no_run(replay_fixture, tmp_path):
    logs = tmp_path / "logs"
    assert train.main(["--config", str(ROOT / "finetune/OHT/configs/smoke.yaml"),
                       "--replay", str(replay_fixture.replay), "--log-dir", str(logs), "--validate-only"]) == 0
    assert not logs.exists()
