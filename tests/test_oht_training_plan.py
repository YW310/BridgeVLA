"""The OHT full profile executes RLBench budgets, epochs and freeze settings."""
import copy
import inspect
import json
from pathlib import Path
from contextlib import contextmanager

import pytest
import torch
from torch import nn
import yaml

from tests.test_oht_migration import replay_fixture  # noqa: F401
from tests.test_oht_training_log import cpu_training, scalars  # noqa: F401
from finetune.OHT import train
from finetune.OHT.config import load
from finetune.OHT.model import build_agent, resume_fingerprint
from finetune.OHT.training_plan import resolve_training_plan, epoch_metrics
from finetune.OHT.data.common import read_config
from finetune.OHT.data.replay import load_contract

ROOT = Path(__file__).resolve().parents[1]


def full_config():
    return load(ROOT / "finetune/OHT/configs/oht_full_8x40.yaml")


def test_full_preset_matches_rlbench_recipe_and_mvt():
    config = full_config()
    reference = read_config(ROOT / "finetune/RLBench/configs/rlbench_full_8x40.yaml")
    for key in ("epochs", "train_iter", "global_batch_size", "max_optimizer_steps",
                "num_workers", "seed", "sample_distribution_mode", "use_scheduler",
                "checkpoint_every_epochs", "gradient_checkpointing", "efficient_paligemma_forward",
                "gpu_paligemma_preprocessing", "flash_attention_2", "freeze_gemma_prefix_layers",
                "freeze_multimodal_projector"):
        assert config[key] == reference[key], key
    assert config["batch_size"] == reference["bs"]
    # YACS decodes the reference's scientific-notation YAML string as a float.
    assert config["learning_rate"] == float(reference["peract"]["lr"])
    for key in ("optimizer_type", "transform_augmentation_xyz", "transform_augmentation_rpy"):
        assert config["agent"][key] == reference["peract"][key]
    for key, value in read_config(ROOT / "finetune/bridgevla/mvt/configs/rvt2.yaml").items():
        assert config["mvt"][key] == value, key
    assert config["agent"]["transform_augmentation"] is True
    assert config["agent"]["collision_loss_weight"] == 0


@pytest.mark.parametrize("world,accumulation", [(1, 48), (2, 24), (4, 12), (8, 6), (16, 3)])
def test_full_plan_preserves_global_batch_and_epoch_budget(world, accumulation):
    plan = resolve_training_plan(full_config(), 40634, world)
    assert plan["accumulation_steps"] == accumulation
    assert plan["effective_batch"] == 192
    assert plan["steps_per_epoch"] == 833
    assert plan["optimizer_steps"] == 83300
    assert plan["checkpoint_interval"] == 8330
    assert plan["epochs"] == 100 and plan["training_samples"] == 40634


def test_legacy_step_budget_is_unchanged():
    config = load(ROOT / "finetune/OHT/configs/baseline.yaml")
    plan = resolve_training_plan(config, 40634, 8)
    assert plan["optimizer_steps"] == 2000
    assert plan["accumulation_steps"] == 8
    assert plan["effective_batch"] == 128
    assert plan["checkpoint_interval"] == 250
    assert epoch_metrics(1, plan) == {}


def test_epoch_budget_can_use_replay_size_and_stop_mid_epoch():
    config = full_config()
    config.update(train_iter=None, max_optimizer_steps=500)
    plan = resolve_training_plan(config, 40634, 8)
    assert plan["steps_per_epoch"] == 211
    assert plan["optimizer_steps"] == 500
    assert epoch_metrics(500, plan)["epoch"] == 3
    assert epoch_metrics(500, plan)["epoch_step"] == 78


@pytest.mark.parametrize("step,epoch,epoch_step", [(1, 1, 1), (833, 1, 833),
                                                  (834, 2, 1), (83300, 100, 833)])
def test_epoch_progress_boundaries(step, epoch, epoch_step):
    plan = resolve_training_plan(full_config(), 40634, 8)
    fields = epoch_metrics(step, plan)
    assert fields["epoch"] == epoch and fields["epoch_step"] == epoch_step
    assert fields["epoch_progress"] == pytest.approx(step / 833)


@pytest.mark.parametrize("changes,world,error", [
    ({"epochs": 0}, 8, "epochs"),
    ({"optimizer_steps": 2000}, 8, "Epoch budgets"),
    ({"checkpoint_interval": 250}, 8, "Epoch budgets"),
    ({"accumulation_steps": 8}, 8, "conflicts"),
    ({"max_optimizer_steps": -1}, 8, "max_optimizer_steps"),
    ({"checkpoint_every_epochs": 0}, 8, "checkpoint_every_epochs"),
    ({"train_iter": 100}, 8, "complete global batch"),
    ({"global_batch_size": 193}, 8, "divisible"),
    ({"sample_distribution_mode": "transition_uniform"}, 8, "task_uniform"),
    ({"use_scheduler": True}, 8, "constant learning rate"),
])
def test_invalid_budget_settings_fail_early(changes, world, error):
    config = full_config()
    config.update(changes)
    with pytest.raises(ValueError, match=error):
        resolve_training_plan(config, 40634, world)


def test_epoch_resume_can_extend_budget_but_not_change_training_strategy():
    config, contract = full_config(), {"sha256": "a" * 64}
    original = resume_fingerprint(config, contract, None, 8)
    extended = copy.deepcopy(config)
    extended.update(epochs=200, max_optimizer_steps=166600, checkpoint_every_epochs=5, num_workers=2)
    assert resume_fingerprint(extended, contract, None, 8) == original
    for key, value in (("train_iter", 38400), ("global_batch_size", 384),
                       ("freeze_gemma_prefix_layers", 18), ("learning_rate", 4e-5)):
        changed = copy.deepcopy(config)
        changed[key] = value
        assert resume_fingerprint(changed, contract, None, 8) != original
    assert resume_fingerprint(config, contract, None, 4) != original


def test_epoch_preflight_resolves_budget_without_gpu(replay_fixture, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("WORLD_SIZE", "8")
    output = tmp_path / "unused"
    assert train.main(["--config", str(ROOT / "finetune/OHT/configs/oht_full_8x40.yaml"),
                       "--replay", str(replay_fixture.replay), "--output", str(output), "--validate-only"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["valid"] is True
    assert result["training_plan"]["accumulation_steps"] == 6
    assert result["training_plan"]["optimizer_steps"] == 83300
    assert not output.exists()


def test_epoch_training_saves_progress_and_resumes_at_next_sample(replay_fixture, tmp_path, cpu_training):
    config = load(ROOT / "finetune/OHT/configs/smoke.yaml")
    config.update(epochs=1, train_iter=4, global_batch_size=2, max_optimizer_steps=0,
                  checkpoint_every_epochs=1, optimizer_steps=None, accumulation_steps=None,
                  checkpoint_interval=None)
    path, output = tmp_path / "epoch.yaml", tmp_path / "run"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    args = ["--config", str(path), "--replay", str(replay_fixture.replay), "--output", str(output)]
    assert train.main(args) == 0
    first = torch.load(output / "model_last.pth", weights_only=False)
    assert first["optimizer_step"] == 2 and first["epochs_completed"] == 1
    assert (output / "model_epoch_1.pth").exists()
    assert train.main([*args, "--resume", str(output / "model_last.pth"), "--epochs", "2"]) == 0
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [(row["step"], row["epoch"], row["epoch_step"]) for row in rows] == [
        (1, 1, 1), (2, 1, 2), (3, 2, 1), (4, 2, 2)]
    assert [value for _, value in scalars(output / "tensorboard", "train/epoch_progress")] == [.5, 1., 1.5, 2.]
    last = torch.load(output / "model_last.pth", weights_only=False)
    assert last["optimizer_step"] == 4 and last["epochs_completed"] == 2
    assert (output / "model_epoch_2.pth").exists()
    assert json.loads((output / "training_plan.json").read_text())["optimizer_steps"] == 4


def test_ddp_accumulation_synchronizes_only_last_microbatch(replay_fixture, tmp_path, cpu_training, monkeypatch):
    import torch.distributed as dist
    original, suppressed, updates = train.build_agent, [], []
    class Network:
        @contextmanager
        def no_sync(self):
            suppressed.append(True)
            yield
    def build(*args, **kwargs):
        agent = original(*args, **kwargs)
        agent._network = Network()
        update = agent.update
        def record(*args, **kwargs):
            updates.append(kwargs["step_optimizer"])
            return update(*args, **kwargs)
        agent.update = record
        return agent
    monkeypatch.setattr(train, "build_agent", build)
    monkeypatch.setenv("WORLD_SIZE", "2")
    for name in ("init_process_group", "barrier", "destroy_process_group"):
        monkeypatch.setattr(dist, name, lambda *args, **kwargs: None)
    config = load(ROOT / "finetune/OHT/configs/smoke.yaml")
    config.update(global_batch_size=4, accumulation_steps=None)
    path = tmp_path / "ddp.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert train.main(["--config", str(path), "--replay", str(replay_fixture.replay),
                       "--output", str(tmp_path / "run"), "--log-backend", "none"]) == 0
    assert updates == [False, True, False, True]
    assert len(suppressed) == 2


def test_full_model_applies_freeze_optimizer_and_render_settings(replay_fixture, monkeypatch):
    from bridgevla.mvt import mvt
    captured = {}
    signature = inspect.signature(mvt.MVT.__init__)
    class SmallMVT(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            signature.bind(None, **kwargs)
            captured.update(kwargs)
            self.mvt1 = nn.Module()
            self.mvt1.model = nn.Module()
            self.mvt1.model.vision_tower = nn.Linear(1, 1)
            self.mvt1.model.multi_modal_projector = nn.Linear(1, 1)
            self.mvt1.model.language_model = nn.Module()
            language = self.mvt1.model.language_model
            language.model = nn.Module()
            language.model.layers = nn.ModuleList(nn.Linear(1, 1) for _ in range(8))
            language.embed_tokens = nn.Embedding(2, 1)
            language.lm_head = nn.Linear(1, 2)
            self.mvt1.enable_efficient_paligemma_forward = lambda: captured.update(efficient=True)
            self.mvt1.enable_gradient_checkpointing = lambda: captured.update(checkpointing=True)
    monkeypatch.setattr(mvt, "MVT", SmallMVT)
    config = full_config()
    agent = build_agent(config, load_contract(replay_fixture.replay), torch.device("cpu"))
    model = agent._net_mod.mvt1.model
    assert not any(p.requires_grad for p in model.vision_tower.parameters())
    assert not any(p.requires_grad for p in model.multi_modal_projector.parameters())
    assert not any(p.requires_grad for p in model.language_model.model.layers[:6].parameters())
    assert all(p.requires_grad for p in model.language_model.model.layers[6:].parameters())
    assert not any(p.requires_grad for p in model.language_model.embed_tokens.parameters())
    assert isinstance(agent._optimizer, torch.optim.AdamW)
    assert agent._optimizer.param_groups[0]["lr"] == 8e-5
    assert agent._optimizer.param_groups[0]["weight_decay"] == 1e-6
    assert captured["rot_ver"] == 1 and captured["cvx_up"] is True
    assert captured["rend_three_views"] is True and captured["inp_pre_con"] is False
    assert captured["efficient"] is True and "checkpointing" not in captured
    assert agent._transform_augmentation is True and agent.collision_loss_weight == 0
