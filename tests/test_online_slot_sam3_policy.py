"""Current-frame online orchestration contracts; no robot performance evidence."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune"))
sys.path.insert(0, str(ROOT / "finetune/bridgevla/libs/YARR"))
from bridgevla.models.slot_guided_geometry import sample_geometry, sha256_file
from bridgevla.online.slot_sam3_policy import OnlineSlotSam3Policy
from yarr.agents.agent import ActResult


def receipt(index=0):
    return dict(episode_id="online-episode", frame_id=f"source-{index}",
                source_frame_index=index, observed_at="2026-11-05T00:00:00+00:00",
                calibration_sha256="a" * 64, snapshot_input_sha256="b" * 64,
                max_source_frame_index_read=index)


def geometry(index=0, *, empty=False, confidence=.9):
    cloud = np.empty((0, 3)) if empty else np.array([[.2, .1, .8], [.21, .1, .8]])
    value = sample_geometry([[cloud, np.empty((0, 3))]], [["VISIBLE", "NULL"]],
                            [[confidence, 0.]], [.2], [{**receipt(index), "source_checkpoint_sha256": "f" * 64}], count=4)
    # External tensors may require gradients; the action call must receive a
    # cloned, detached packet and preserve actual support rather than padding.
    return replace(value, object_points=value.object_points.requires_grad_(True))


class ProviderSeam:
    def __init__(self):
        self.calls = []
        self.error = None
        self.receipt_update = {}

    def capture(self, observation):
        self.calls.append(observation)
        if self.error:
            raise self.error
        return SimpleNamespace(snapshot=SimpleNamespace(index=observation["current_index"]),
                               receipt={**receipt(observation["current_index"]), **self.receipt_update},
                               instruction="pick the current cup", task_name="pick_cup")


class PerceptionSeam:
    """Controlled software seam; a kind string is not evidence of real SAM3."""
    def __init__(self):
        self.calls, self.resets = [], 0
        self.error, self.kind, self.empty, self.confidence = None, "sam3", False, .9
        self.receipt_update = {}
        self.checkpoint_sha = "f" * 64
        self.camera_peaks = {"front": None, "wrist": None}
        self.model_identity = object()

    def perceive(self, snapshot, packet, **kwargs):
        self.calls.append((snapshot, packet, kwargs))
        assert torch.is_inference_mode_enabled()
        if self.error:
            raise self.error
        self.last_geometry = geometry(snapshot.index, empty=self.empty, confidence=self.confidence)
        self.last_geometry = replace(self.last_geometry, provenance=({**self.last_geometry.provenance[0],
                                                                     "source_checkpoint_sha256": self.checkpoint_sha},))
        return SimpleNamespace(geometry=self.last_geometry, backend_kind=self.kind,
                               receipt={**receipt(snapshot.index), **self.receipt_update}, timings_ms={"segmentation": 5.},
                               inference={"cameras": {name: {"peak_vram_bytes": value} for name, value in self.camera_peaks.items()}})

    def reset(self):
        self.resets += 1


class AgentSeam:
    cameras = ["front", "wrist"]
    _training = False
    slot_guided_sam3_execution_mode = "online_experiment"
    enable_slot_guided_sam3 = True
    bridgevla_aligned_objects = False
    heatmap_action_anchor = False

    def __init__(self):
        self.calls, self.resets, self.builds = [], 0, []
        self.coarse_error, self.conditioned_error = None, None
        self.export_count = 1
        self.bad_conditioned_action = False

    def act(self, step, observation, deterministic=False, **kwargs):
        assert torch.is_inference_mode_enabled()
        self.calls.append((self.enable_slot_guided_sam3, step, observation, deterministic, kwargs))
        callback = kwargs.get("coarse_slot_callback")
        if callback is not None:
            if self.coarse_error:
                raise self.coarse_error
            for _ in range(self.export_count):
                callback(dict(source_frame="bridgevla_coarse", matrix_world_from_source=np.eye(4),
                              point_valid=np.ones((2, 2), dtype=bool),
                              point_support_source="controlled renderer coverage seam", slot_output={}))
        if self.enable_slot_guided_sam3:
            if self.conditioned_error:
                raise self.conditioned_error
            payload = kwargs["slot_guided_sam3"]
            assert payload["execution_mode"] == "online_experiment"
            assert "acceptance_report" not in payload
            assert observation["slot_guided_current_frames"] == [receipt(observation["current_index"])]
            data = payload["geometry"]
            assert not data.object_points.requires_grad
            assert data.object_point_valid[0, 0].tolist() == [True, True, False, False]
            assert data.object_point_valid[0, 1].sum() == 0
            assert data.statuses == (("VISIBLE", "NULL"),)
            assert float(data.reference_null_probability[0]) == pytest.approx(.2)
        action = np.array([step + .1 * self.enable_slot_guided_sam3, 0., 0., 0., 0., 0., 1., 1., 0.])
        if self.enable_slot_guided_sam3 and self.bad_conditioned_action:
            action[0] = np.nan
        result = ActResult(action, observation_elements={"original": 1}, replay_elements={"retained": 7},
                           info={"policy": "original"})
        self.last_result = result
        return result

    def reset(self):
        self.resets += 1

    def build(self, training=False, device=None):
        self.builds.append((training, device))

    def update(self, *args, **kwargs):
        return (args, kwargs)

    def train(self):
        return "original train"

    def act_summaries(self):
        return ["original summaries"]


def policy_fixture(**kwargs):
    agent, perception, provider = AgentSeam(), PerceptionSeam(), ProviderSeam()
    policy = OnlineSlotSam3Policy(agent, perception, provider, "actual-loaded-checkpoint.pt", "c" * 40,
                                  enabled=True, point_count=4, expected_checkpoint_sha256="f" * 64, **kwargs)
    return policy, agent, perception, provider


def test_default_disabled_is_exact_delegate_without_observer_or_perception_access():
    class Poison:
        def __getattribute__(self, name):
            raise AssertionError(f"disabled path touched dependency: {name}")

    calls = []
    expected = ActResult(np.arange(9.), info={"same": True})
    agent = SimpleNamespace(act=lambda *args, **kwargs: calls.append((args, kwargs)) or expected,
                            reset=lambda: calls.append("reset"))
    policy = OnlineSlotSam3Policy(agent, Poison(), Poison(), Poison(), Poison())
    observation = {"same": object()}
    assert policy.act(3, observation, deterministic=True, visualize=True) is expected
    assert calls == [((3, observation), {"deterministic": True, "visualize": True})]
    policy.reset()
    assert calls[-1] == "reset" and policy.last_diagnostics is None


def test_current_C_then_fresh_geometry_then_conditioned_action_preserves_YARR_fields():
    policy, agent, perception, provider = policy_fixture()
    observation = {"current_index": 0, "untouched": object()}
    result = policy.act(10, observation, deterministic=True, visualize=True)
    assert [call[0] for call in agent.calls] == [False, True]
    assert agent.calls[0][2] is observation and provider.calls == [observation]
    assert agent.calls[1][2] is not observation and "slot_guided_current_frames" not in observation
    assert agent.calls[0][4]["visualize"] is False and agent.calls[1][4]["visualize"] is True
    assert result.action[0] == pytest.approx(10.1)
    assert result.replay_elements == {"retained": 7, "slot_guided_sam3_used": True, "slot_guided_sam3_fallback": False}
    assert result.info["policy"] == "original" and result.info["slot_guided_sam3"]["status"] == "conditioned"
    assert result.observation_elements == {"original": 1}
    assert agent.last_result.info == {"policy": "original"}
    assert agent.enable_slot_guided_sam3 is True
    assert perception.calls[0][2]["expected_snapshot_digest"] == "b" * 64
    assert perception.calls[0][2]["diagnostics_dir"] is None
    assert perception.last_geometry.object_points.requires_grad
    assert agent.calls[1][4]["slot_guided_sam3"]["geometry"].object_points.data_ptr() != perception.last_geometry.object_points.data_ptr()


def test_next_frame_failure_uses_its_current_C_and_never_previous_geometry():
    policy, agent, perception, _ = policy_fixture()
    assert policy.act(10, {"current_index": 0}).action[0] == pytest.approx(10.1)
    perception.error = RuntimeError("current SAM3 empty prediction")
    result = policy.act(11, {"current_index": 1})
    assert result.action[0] == 11 and [call[0] for call in agent.calls] == [False, True, False]
    assert result.info["slot_guided_sam3"]["current_frame"]["source_frame_index"] == 1
    assert "current SAM3 empty prediction" in result.info["slot_guided_sam3"]["failure_reason"]
    assert result.replay_elements["slot_guided_sam3_fallback"] is True
    assert not hasattr(policy, "last_geometry")


def test_loaded_checkpoint_hash_is_pinned_and_changed_weights_fall_back_to_current_C():
    policy, agent, perception, _ = policy_fixture()
    perception.checkpoint_sha = "e" * 64
    result = policy.act(4, {"current_index": 0})
    assert result.action[0] == 4 and len(agent.calls) == 1
    assert "loaded policy checkpoint" in result.info["slot_guided_sam3"]["failure_reason"]


def test_constructor_pins_real_file_if_loader_has_not_supplied_expected_hash(tmp_path):
    checkpoint = tmp_path / "weights.pt"
    checkpoint.write_bytes(b"unit fixture only; not model evidence")
    agent = AgentSeam()
    policy = OnlineSlotSam3Policy(agent, PerceptionSeam(), ProviderSeam(), checkpoint, "c" * 40, enabled=True)
    assert policy.expected_checkpoint_sha256 == sha256_file(checkpoint)
    agent.slot_guided_sam3_checkpoint_sha256 = "a" * 64
    policy = OnlineSlotSam3Policy(agent, PerceptionSeam(), ProviderSeam(), "not-read-loader-pinned.pt", "c" * 40, enabled=True)
    assert policy.expected_checkpoint_sha256 == "a" * 64


def test_diagnostics_keep_capture_index_provenance_and_real_measurement_scope():
    policy, _, perception, provider = policy_fixture()
    provider.receipt_update = dict(index_kind="live_simulator_step", live_sim_seconds=1.25, live_sim_dt=.05,
                                   calibration_transform_source="explicit optical conversion")
    perception.camera_peaks = {"front": 1234, "wrist": 3456}
    result = policy.act(4, {"current_index": 0})
    diagnostics = result.info["slot_guided_sam3"]
    assert diagnostics["current_frame"]["index_kind"] == "live_simulator_step"
    assert diagnostics["current_frame"]["live_sim_seconds"] == 1.25
    assert diagnostics["perception_timings_ms"] == {"segmentation": 5.}
    assert diagnostics["peak_gpu_memory_bytes"] == 3456
    assert "policy forward excluded" in diagnostics["peak_gpu_memory_scope"]
    assert diagnostics["roles"]["target"]["point_count"] == 2
    assert diagnostics["roles"]["reference"]["status"] == "NULL"
    assert all("mask" not in str(row) for row in diagnostics.values())


@pytest.mark.parametrize("kind,empty,confidence", [("mock", False, .9), ("sam3", True, 0.), ("sam3", False, .1)])
def test_mock_empty_or_unreliable_geometry_returns_current_baseline(kind, empty, confidence):
    policy, agent, perception, _ = policy_fixture()
    perception.kind, perception.empty, perception.confidence = kind, empty, confidence
    result = policy.act(2, {"current_index": 0})
    assert result.action[0] == 2 and [call[0] for call in agent.calls] == [False]
    assert result.info["slot_guided_sam3"]["failure_reason"]
    assert result.info["slot_guided_sam3"]["peak_gpu_memory_bytes"] is None


@pytest.mark.parametrize("update", [{"source_frame_index": 1}, {"snapshot_input_sha256": "e" * 64},
                                    {"calibration_sha256": "d" * 64}, {"frame_id": 0}])
def test_result_receipt_must_match_actual_current_capture(update):
    policy, agent, perception, _ = policy_fixture()
    perception.receipt_update = update
    result = policy.act(4, {"current_index": 0})
    assert result.action[0] == 4 and len(agent.calls) == 1
    assert "mismatch" in result.info["slot_guided_sam3"]["failure_reason"] or "current source" in result.info["slot_guided_sam3"]["failure_reason"]


@pytest.mark.parametrize("update", [{"max_source_frame_index_read": 1}, {"source_frame_index": True},
                                    {"snapshot_input_sha256": "invalid"}])
def test_future_or_unbound_capture_receipt_does_not_reach_perception(update):
    policy, agent, perception, provider = policy_fixture()
    provider.receipt_update = update
    assert policy.act(4, {"current_index": 0}).action[0] == 4
    assert len(agent.calls) == 1 and not perception.calls


def test_capture_validation_error_returns_current_C_without_SAM():
    policy, agent, perception, provider = policy_fixture()
    provider.error = ValueError("policy RGB differs from raw capture")
    result = policy.act(5, {"current_index": 0})
    assert result.action[0] == 5 and len(agent.calls) == 1 and not perception.calls


def test_export_failure_recomputes_current_C_without_export_before_fallback():
    policy, agent, perception, provider = policy_fixture()
    agent.coarse_error = ValueError("renderer support unavailable")
    result = policy.act(6, {"current_index": 0})
    assert result.action[0] == 6 and [call[0] for call in agent.calls] == [False, False]
    assert "coarse_slot_callback" in agent.calls[0][4] and "coarse_slot_callback" not in agent.calls[1][4]
    assert not perception.calls and not provider.calls
    assert agent.enable_slot_guided_sam3 is True


@pytest.mark.parametrize("count", [0, 2])
def test_missing_or_multiple_export_packets_are_explicit_fallback(count):
    policy, agent, perception, _ = policy_fixture()
    agent.export_count = count
    result = policy.act(6, {"current_index": 0})
    assert result.action[0] == 6 and not perception.calls
    assert "exactly one" in result.info["slot_guided_sam3"]["failure_reason"]


@pytest.mark.parametrize("bad_action", [False, True])
def test_conditioned_exception_or_nonfinite_action_falls_back_and_restores_flag(bad_action):
    policy, agent, _, _ = policy_fixture()
    if bad_action:
        agent.bad_conditioned_action = True
    else:
        agent.conditioned_error = ValueError("online authorization rejected provenance")
    result = policy.act(7, {"current_index": 0})
    assert result.action[0] == 7 and [call[0] for call in agent.calls] == [False, True]
    assert agent.enable_slot_guided_sam3 is True and result.info["slot_guided_sam3"]["failure_reason"]


def test_fail_mode_propagates_current_failure_and_restores_original_flag():
    policy, agent, perception, _ = policy_fixture(fallback="fail")
    agent.enable_slot_guided_sam3 = False
    perception.error = RuntimeError("SAM3 unavailable")
    with pytest.raises(RuntimeError, match="SAM3 unavailable"):
        policy.act(8, {"current_index": 0})
    assert agent.enable_slot_guided_sam3 is False and policy.last_diagnostics["status"] == "failed"


def test_reset_keeps_loaded_models_and_env_capture_but_clears_episode_diagnostics():
    policy, agent, perception, provider = policy_fixture()
    model = perception.model_identity
    policy.act(9, {"current_index": 0})
    capture_count = len(provider.calls)
    policy.reset()
    assert agent.resets == perception.resets == 1 and perception.model_identity is model
    assert len(provider.calls) == capture_count and policy.last_diagnostics is None
    assert policy._frame_sequence == 0
    assert policy.act(0, {"current_index": 0}).action[0] == pytest.approx(.1)


def test_diagnostics_directory_is_optional_and_unique_without_default_frame_io(tmp_path):
    policy, _, perception, _ = policy_fixture(diagnostics_dir=tmp_path)
    policy.act(0, {"current_index": 0})
    first = perception.calls[-1][2]["diagnostics_dir"]
    policy.act(1, {"current_index": 1})
    second = perception.calls[-1][2]["diagnostics_dir"]
    policy.reset()
    policy.act(0, {"current_index": 0})
    third = perception.calls[-1][2]["diagnostics_dir"]
    assert len({first, second, third}) == 3 and not list(tmp_path.iterdir())


def test_YARR_metadata_and_methods_delegate_and_training_is_blocked_when_enabled():
    policy, agent, _, _ = policy_fixture()
    assert policy.cameras == agent.cameras and policy.act_summaries() == ["original summaries"]
    assert not agent.builds
    policy.build(False, "cpu")
    assert agent.builds == [(False, "cpu")]
    with pytest.raises(ValueError, match="inference-only"):
        policy.build(True)
    with pytest.raises(ValueError, match="weights"):
        policy.update(0, {})
    with pytest.raises(ValueError, match="eval mode"):
        policy.train()
    disabled = OnlineSlotSam3Policy(agent)
    assert disabled.update(1, sample=2) == ((1,), {"sample": 2})
    assert disabled.train() == "original train"


@pytest.mark.parametrize("field,value", [("_training", True), ("slot_guided_sam3_execution_mode", "offline_experiment"),
                                       ("bridgevla_aligned_objects", True), ("heatmap_action_anchor", True)])
def test_incompatible_or_stateful_agent_is_rejected_before_two_pass_inference(field, value):
    agent = AgentSeam()
    setattr(agent, field, value)
    with pytest.raises(ValueError):
        OnlineSlotSam3Policy(agent, PerceptionSeam(), ProviderSeam(), "loaded.pt", "c" * 40, enabled=True)
    assert not agent.calls


@pytest.mark.parametrize("kwargs", [{"slot_guided_sam3": {}}, {"coarse_slot_callback": lambda packet: None},
                                    {"context_with_grad": True}, {"return_gembench_action": True}])
def test_caller_cannot_supply_stale_geometry_or_gradients(kwargs):
    policy, agent, perception, provider = policy_fixture()
    with pytest.raises(ValueError):
        policy.act(0, {"current_index": 0}, **kwargs)
    assert not agent.calls and not perception.calls and not provider.calls
