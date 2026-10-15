"""OHT logs actual training heatmaps as TensorBoard images and optional PNGs."""
import copy
import io
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from tests.test_oht_migration import replay_fixture  # noqa: F401
from tests.test_oht_action_modes import TinyMVT, caches  # noqa: F401
from tests.test_oht_training_log import cpu_training  # noqa: F401
from finetune.OHT import train
from finetune.OHT.config import load
from finetune.OHT.model import build_agent, resume_fingerprint
from finetune.OHT.data.dataset import OHTDataset, collate
from finetune.OHT.training_log import TensorBoardLogger
from finetune.OHT.training_visualization import visualization_settings, capture_due, record_visualization

ROOT = Path(__file__).resolve().parents[1]


def image_payload():
    stage = dict(input=torch.full((3, 3, 8, 8), .5), gt=torch.zeros(3, 8, 8),
                 pred=torch.ones(3, 8, 8))
    return {"mvt1": stage, "mvt2": dict(stage)}


def test_capture_interval_is_optimizer_step_based_and_rank_zero_only():
    settings = visualization_settings({"train_visualization": {"enabled": True}})
    assert capture_due(settings, 1)
    assert not capture_due(settings, 499)
    assert capture_due(settings, 500)
    assert not capture_due(settings, 500, rank=1)
    settings["enabled"] = False
    assert not capture_due(settings, 1)


@pytest.mark.parametrize("settings,backend", [
    ({"enabled": True, "interval": 0}, "tensorboard"),
    ({"enabled": "true"}, "tensorboard"),
    ({"enabled": True, "save_png": False, "tensorboard": False}, "tensorboard"),
    ({"enabled": True}, "none"),
    ({"output_dir": ""}, "tensorboard"),
    ({"unexpected": True}, "tensorboard"),
])
def test_invalid_visualization_settings_fail_early(settings, backend):
    with pytest.raises(ValueError):
        visualization_settings({"train_visualization": settings}, backend)


def test_image_options_do_not_break_optimizer_continuous_resume():
    config = load(ROOT / "finetune/OHT/configs/oht_full_8x40.yaml")
    contract = {"sha256": "a" * 64}
    before = resume_fingerprint(config, contract, None, 8)
    changed = copy.deepcopy(config)
    changed["train_visualization"].update(enabled=False, interval=10, save_png=False)
    assert resume_fingerprint(changed, contract, None, 8) == before


@pytest.mark.parametrize("mode", ["eef", "relative_eef", "joint", "relative_joint"])
@pytest.mark.parametrize("object_mode", ["baseline", "role_queries", "predicted_external"])
def test_actual_agent_payload_is_logged_for_all_modes(replay_fixture, caches, tmp_path, monkeypatch, mode, object_mode):
    from bridgevla.mvt import mvt
    class ImageMVT(TinyMVT):
        def forward(self, pc, **kwargs):
            output = super().forward(pc, **kwargs)
            rendered = torch.zeros(len(pc), 3, 6, 8, 8)
            rendered[:, :, 3:6] = .5
            output["mvt1_ori_img"] = rendered
            output["mvt2_ori_img"] = rendered.clone()
            # Match the production teacher/prior grid to the rendered grid.
            for stage in (output, output["mvt2"]):
                for key in ("object_slot_prior", "object_slot_target_prior"):
                    if key in stage:
                        value = stage[key]
                        stage[key] = F.interpolate(value.reshape(-1, 2, 2, 2), size=(8, 8),
                                                   mode="nearest").reshape(len(pc), 3, 2, 8, 8)
            return output
    monkeypatch.setattr(mvt, "MVT", ImageMVT)
    config = load(ROOT / f"finetune/OHT/configs/{object_mode}.yaml")
    config["action_mode"] = mode
    config["point_count"] = 8
    config["action"]["hidden_dim"] = 16
    cache = None if object_mode == "baseline" else caches[0] / (
        "teacher" if object_mode == "role_queries" else "predicted")
    data = OHTDataset(replay_fixture.replay, mode=object_mode, role_cache=cache, point_count=8,
                      action_config={"mode": mode})
    agent = build_agent(config, data.contract, torch.device("cpu"))
    batch = collate([data[0], data[1]])
    result = agent.update(batch, return_visualization=True)
    payload = result["train_visualization"]
    assert set(payload) == {"mvt1", "mvt2"}
    for stage in payload.values():
        assert stage["input"].shape == (3, 3, 8, 8)
        assert stage["gt"].shape == stage["pred"].shape == (3, 8, 8)
        assert all(value.device.type == "cpu" and not value.requires_grad for value in stage.values())
    settings = visualization_settings({"train_visualization": {"enabled": True}})
    with TensorBoardLogger(tmp_path / "tensorboard") as logger:
        saved = record_visualization(payload, step=1, output=tmp_path,
                                     settings=settings, batch=batch, logger=logger)
    assert all(path.is_file() for path in saved.values())
    events = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    assert set(events.Tags()["images"]) == {"train_visualization/mvt1", "train_visualization/mvt2"}
    event = events.Images("train_visualization/mvt1")[0]
    assert event.step == 1
    with Image.open(io.BytesIO(event.encoded_image_string)) as image:
        assert image.size == (event.width, event.height)
        assert image.width <= 1600 and image.height <= 1000


def test_training_requests_images_only_on_capture_steps_and_last_microbatch(replay_fixture, tmp_path, cpu_training, monkeypatch):
    original, requested = train.build_agent, []
    def build(*args, **kwargs):
        agent = original(*args, **kwargs)
        update = agent.update
        def capture(*args, **kwargs):
            enabled = kwargs.pop("return_visualization", False)
            requested.append(enabled)
            result = update(*args, **kwargs)
            if enabled:
                result["train_visualization"] = image_payload()
            return result
        agent.update = capture
        return agent
    monkeypatch.setattr(train, "build_agent", build)
    config = load(ROOT / "finetune/OHT/configs/smoke.yaml")
    config.update(accumulation_steps=2, optimizer_steps=3)
    config["train_visualization"].update(enabled=True, interval=2)
    path, output = tmp_path / "images.yaml", tmp_path / "run"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert train.main(["--config", str(path), "--replay", str(replay_fixture.replay),
                       "--output", str(output)]) == 0
    assert requested == [False, True, False, True, False, False]
    events = EventAccumulator(str(output / "tensorboard")).Reload()
    assert [event.step for event in events.Images("train_visualization/mvt1")] == [1, 2]
    assert len(list((output / "train_visualizations").glob("*.png"))) == 4
    assert len((output / "metrics.jsonl").read_text().splitlines()) == 3
    assert all("train_visualization" not in json.loads(line)
               for line in (output / "metrics.jsonl").read_text().splitlines())


def test_png_only_visualization_needs_no_tensorboard_writer(tmp_path):
    settings = visualization_settings({"train_visualization": {"enabled": True, "tensorboard": False}}, "none")
    with TensorBoardLogger(tmp_path / "tensorboard", backend="none") as logger:
        saved = record_visualization(image_payload(), step=1, output=tmp_path, settings=settings,
                                     batch={"tasks": ["assemble_left"], "lang_goal": [[["assemble"]]]}, logger=logger)
    assert all(path.is_file() for path in saved.values())
    assert not (tmp_path / "tensorboard").exists()


def test_requested_but_missing_images_fail_clearly(tmp_path):
    with TensorBoardLogger(tmp_path / "tensorboard", backend="none") as logger:
        with pytest.raises(RuntimeError, match="returned no image payload"):
            record_visualization(None, step=1, output=tmp_path,
                                 settings=visualization_settings({}), batch={}, logger=logger)
