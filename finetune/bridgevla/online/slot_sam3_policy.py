"""Two-pass current-observation Slot/SAM3 policy for the YARR agent interface.

The default-disabled wrapper delegates directly. Enabled execution first keeps
the current original policy result, then attempts fresh perception and a second
conditioned prediction. Geometry is local to one call and never retained as a
fallback. Model loading and environment stepping remain outside this wrapper.
"""

from __future__ import annotations

import copy
from pathlib import Path
import time
from typing import Any, Mapping

from bridgevla.utils.startup_progress import live_phase
from bridgevla.online.slot_sam3_diagnostics import (
    UnavailableTargetError, camera_role_diagnostics, guidance_role_diagnostics,
    print_target_diagnostics)


_RECEIPT_KEYS = (
    "episode_id", "frame_id", "source_frame_index", "observed_at",
    "calibration_sha256", "snapshot_input_sha256", "max_source_frame_index_read",
)


def _receipt(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not set(_RECEIPT_KEYS).issubset(value):
        raise ValueError("online capture requires an exact current RGB-D receipt")
    result = {key: value[key] for key in _RECEIPT_KEYS}
    index = result["source_frame_index"]
    if type(index) is not int or index < 0:
        raise ValueError("source_frame_index must be a nonnegative source index")
    if type(result["max_source_frame_index_read"]) is not int or result["max_source_frame_index_read"] != index:
        raise ValueError("online capture may only read the current source frame")
    for key in ("episode_id", "observed_at"):
        if not isinstance(result[key], str) or not result[key].strip():
            raise ValueError(f"online capture requires {key}")
    if type(result["frame_id"]) not in (str, int) or isinstance(result["frame_id"], str) and not result["frame_id"]:
        raise ValueError("online capture requires an explicit frame_id")
    for key in ("calibration_sha256", "snapshot_input_sha256"):
        digest = result[key]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"online capture requires {key}")
    return result


def _policy_result(result: Any) -> Any:
    import numpy as np

    action = np.asarray(result.action)
    if action.shape != (9,) or action.dtype.kind not in "fiu" or not np.isfinite(action).all():
        raise ValueError("online policy must return a finite 9D RLBench action")
    return result


class OnlineSlotSam3Policy:
    """Wrap an already loaded frozen RVTAgent and persistent online perception.

    ``snapshot_provider.capture(observation)`` must bind its immutable raw
    RGB-D/calibration snapshot to the actual current policy tensors. Perception
    returns detached ObjectGeometry and the same receipt. The RVTAgent performs
    its own online geometry authorization before the conditioned network pass.
    Neither a software test nor an online experiment certifies task success.
    """

    def __init__(self, agent: Any, perception: Any = None, snapshot_provider: Any = None,
                 checkpoint_path: str | Path | None = None, bridge_git_sha: str | None = None,
                 *, enabled: bool = False, point_count: int = 128,
                 diagnostics_dir: str | Path | None = None, fallback: str = "baseline",
                 expected_checkpoint_sha256: str | None = None, stage: str = 'action', guidance_stage: str = 'coarse',
                 guidance_source: str = 'predicted_coarse_slot'):
        self.agent, self.perception, self.snapshot_provider = agent, perception, snapshot_provider
        self.enabled = bool(enabled)
        self.checkpoint_path, self.bridge_git_sha = checkpoint_path, bridge_git_sha
        self.point_count, self.diagnostics_dir, self.fallback = point_count, diagnostics_dir, fallback
        self.expected_checkpoint_sha256 = expected_checkpoint_sha256
        self.stage = stage
        self.guidance_stage = guidance_stage
        self.guidance_source = guidance_source
        if guidance_source not in ('predicted_coarse_slot', 'oracle_gt'):
            raise ValueError('invalid online guidance source')
        self.execution_mode = 'oracle_online_experiment' if guidance_source == 'oracle_gt' else 'online_experiment'
        self.last_diagnostics: dict[str, Any] | None = None
        self._episode_number, self._frame_sequence = 0, 0
        # The disabled constructor does not inspect a provider, perception,
        # checkpoint or filesystem. Existing evaluation needs none of them.
        if self.enabled:
            if perception is None or snapshot_provider is None or checkpoint_path is None or bridge_git_sha is None:
                raise ValueError("online Slot/SAM3 requires perception, snapshot provider and loaded checkpoint provenance")
            if type(point_count) is not int or point_count < 1:
                raise ValueError("point_count must be a positive integer")
            if fallback not in ("baseline", "fail"):
                raise ValueError("fallback must be baseline or fail")
            if stage not in ('action', 'prompts', 'segmentation'):
                raise ValueError('stage must be action, prompts or segmentation')
            if guidance_stage not in ('coarse', 'refine'):
                raise ValueError('guidance_stage must be coarse or refine')
            if stage in ('prompts', 'segmentation') and diagnostics_dir is None:
                raise ValueError('prompts/segmentation stage requires a diagnostics directory')
            self._validate_agent()
            expected = (expected_checkpoint_sha256 if expected_checkpoint_sha256 is not None
                        else getattr(agent, "slot_guided_sam3_checkpoint_sha256", None))
            if expected is None:
                from bridgevla.models.slot_guided_geometry import sha256_file

                expected = sha256_file(checkpoint_path)
            if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
                raise ValueError("loaded BridgeVLA checkpoint needs a lowercase SHA256 digest")
            self.expected_checkpoint_sha256 = expected

    def __getattr__(self, name: str) -> Any:
        # YARR summaries, saved weights and public policy metadata retain the
        # original implementation. Evaluation sets mutable options before wrap.
        return getattr(object.__getattribute__(self, "agent"), name)

    def _validate_agent(self) -> None:
        if getattr(self.agent, "_training", False):
            raise ValueError("online Slot/SAM3 requires frozen inference")
        if getattr(self.agent, "slot_guided_sam3_execution_mode", None) != self.execution_mode:
            raise ValueError(f'online SAM3 requires the explicit {self.execution_mode} agent scope')
        if getattr(self.agent, "bridgevla_aligned_objects", False) or getattr(self.agent, "heatmap_action_anchor", False):
            raise ValueError("online Slot/SAM3 does not combine with privileged alignment/anchor state")
        if self.guidance_source == 'oracle_gt' and getattr(getattr(self.agent, '_net_mod', None), 'object_slots_enabled', True):
            raise ValueError('Oracle SAM3 action experiment requires object Slots disabled')

    def build(self, training: bool, device: Any = None) -> Any:
        if self.enabled and training:
            raise ValueError("online Slot/SAM3 is inference-only")
        return self.agent.build(training=training, device=device)

    def update(self, *args: Any, **kwargs: Any) -> Any:
        if self.enabled:
            raise ValueError("online Slot/SAM3 cannot update model weights")
        return self.agent.update(*args, **kwargs)

    def train(self) -> Any:
        if self.enabled:
            raise ValueError("online Slot/SAM3 must remain in eval mode")
        return self.agent.train()

    def reset(self) -> None:
        self.agent.reset()
        if self.enabled:
            # Environment reset already captured the new initial observation;
            # clearing its provider here would discard that valid capture.
            self.perception.reset()
            self.last_diagnostics = None
            self._episode_number += 1
            self._frame_sequence = 0

    def _finish(self, result: Any, diagnostics: dict[str, Any], started: float) -> Any:
        diagnostics["timings_ms"]["policy_total"] = (time.perf_counter() - started) * 1000
        self.last_diagnostics = diagnostics
        # Copy containers so fallback retains the current original action and
        # existing observation/replay fields without mutating the saved C result.
        result = copy.copy(result)
        result.replay_elements = dict(getattr(result, "replay_elements", {}) or {})
        result.replay_elements.update(
            slot_guided_sam3_used=diagnostics["status"] == "conditioned",
            slot_guided_sam3_fallback=diagnostics["status"] == "baseline_fallback",
        )
        result.info = dict(getattr(result, "info", {}) or {})
        result.info["slot_guided_sam3"] = diagnostics
        print(f'[SlotSAM3] RESULT call={self._frame_sequence}; status={diagnostics["status"]}; '
              f'policy_total={diagnostics["timings_ms"]["policy_total"] / 1000:.1f}s'
              + (f'; reason={diagnostics["failure_reason"]}' if diagnostics.get('failure_reason') else ''),
              flush=True)
        if diagnostics['status'] == 'baseline_fallback':
            print_target_diagnostics(diagnostics, call=self._frame_sequence)
        return result

    def act(self, step: int, observation: dict, deterministic: bool = False, **kwargs: Any) -> Any:
        if not self.enabled:
            return self.agent.act(step, observation, deterministic=deterministic, **kwargs)
        self._validate_agent()
        if self.guidance_source == 'oracle_gt' and any(key.startswith('oracle_') or key.endswith('_mask') for key in observation):
            raise ValueError('GT annotations must stay in the prompt side channel, outside policy observations')
        if "slot_guided_sam3" in kwargs or "coarse_slot_callback" in kwargs:
            raise ValueError("the online policy owns its current geometry and coarse callback")
        if kwargs.get("context_with_grad", False) or kwargs.get("return_gembench_action", False):
            raise ValueError("online Slot/SAM3 requires a detached YARR ActResult")

        import torch

        started = time.perf_counter()
        timings: dict[str, float] = {}
        diagnostics: dict[str, Any] = dict(status="started", failure_reason=None, timings_ms=timings,
                                           execution_mode=self.execution_mode)
        oracle = self.guidance_source == 'oracle_gt'
        if oracle:
            diagnostics.update(guidance_source='oracle_gt', oracle_conditioned=True, slot_prediction_used=False,
                               action_source='sam3_geometry_or_unconditioned_fallback')
        previous_flag = self.agent.enable_slot_guided_sam3
        packets: list[Any] = []
        original = None
        coarse_kwargs = dict(kwargs)
        # Avoid duplicate visualization writes/callbacks in the two-pass route.
        coarse_kwargs.update(visualize=False, visualization_callback=None)
        self._frame_sequence += 1
        diagnostics['policy_call_index'] = self._frame_sequence
        callback = packets.append
        if self.stage in ('prompts', 'segmentation') or self.guidance_stage == 'refine':
            def collect(packet):
                packets.append(packet)
            collect.include_visualization = self.stage in ('prompts', 'segmentation')
            collect.guidance_stage = self.guidance_stage
            callback = collect
        if self.stage == 'prompts':
            diagnostics.update(stage='prompts', action_source='baseline_C', sam3_executed=False)
        elif self.stage == 'segmentation':
            diagnostics.update(stage='segmentation', action_source=(
                'unconditioned_gt_trained_policy' if oracle else 'baseline_C'), sam3_executed=False)
        try:
            with torch.inference_mode():
                self.agent.enable_slot_guided_sam3 = False
                coarse_started = time.perf_counter()
                try:
                    with live_phase(f'call={self._frame_sequence} baseline policy', prefix='SlotSAM3'):
                        original = _policy_result(self.agent.act(step, observation, deterministic=deterministic,
                            **({} if oracle else dict(coarse_slot_callback=callback)), **coarse_kwargs))
                except Exception:
                    # Export can fail after the original network forward. A
                    # fresh original call without export is the current C action;
                    # a previous-frame action or geometry is never substituted.
                    with live_phase(f'call={self._frame_sequence} baseline retry without export', prefix='SlotSAM3'):
                        original = _policy_result(self.agent.act(
                            step, observation, deterministic=deterministic, **coarse_kwargs))
                    raise
                timings["baseline_C"] = (time.perf_counter() - coarse_started) * 1000
                if not oracle and len(packets) != 1:
                    raise ValueError("expected exactly one current-frame Coarse export")
                diagnostics['guidance_selection'] = dict(requested_source='oracle_gt') if oracle else copy.deepcopy(packets[0].get('guidance_selection',
                    dict(requested_stage=self.guidance_stage, roles={role: dict(selected_stage='coarse',
                        failure_reason='REFINE_SELECTION_UNAVAILABLE' if self.guidance_stage == 'refine' else None)
                                                        for role in ('target', 'reference')})))
                with live_phase(f'call={self._frame_sequence} current-frame validation', prefix='SlotSAM3'):
                    capture = self.snapshot_provider.capture(observation)
                    receipt = _receipt(capture.receipt)
                # Preserve live simulator index/optical-conversion provenance;
                # only the canonical fields above bind the action boundary.
                diagnostics["current_frame"] = copy.deepcopy(dict(capture.receipt))
                if oracle:
                    diagnostics['oracle_prompt_context'] = {
                        key: capture.oracle_labels.get(key)
                        for key in ('phase_id', 'phase_source', 'role_config_sha256', 'oracle_instance_ids')}
                diagnostic_path = None
                if self.diagnostics_dir is not None:
                    diagnostic_path = Path(self.diagnostics_dir) / f"episode_{self._episode_number:05d}" / f"frame_{self._frame_sequence:06d}"
                perceived_at = time.perf_counter()
                if self.stage == 'prompts':
                    with live_phase(f'call={self._frame_sequence} prompt inspection', prefix='SlotSAM3'):
                        report = self.perception.inspect(
                            capture.snapshot, packets[0], instruction=capture.instruction,
                            task_name=capture.task_name, checkpoint_path=self.checkpoint_path,
                            bridge_git_sha=self.bridge_git_sha, diagnostics_dir=diagnostic_path,
                            expected_snapshot_digest=receipt['snapshot_input_sha256'])
                    diagnostics.update(status='prompts_only', probe=report)
                    timings['prompt_probe'] = (time.perf_counter() - perceived_at) * 1000
                    return self._finish(original, diagnostics, started)
                with live_phase(f'call={self._frame_sequence} SAM3 perception', prefix='SlotSAM3'):
                    result = self.perception.perceive(
                        capture.snapshot, None if oracle else packets[0], instruction=capture.instruction,
                        task_name=capture.task_name, checkpoint_path=self.checkpoint_path,
                        bridge_git_sha=self.bridge_git_sha, point_count=self.point_count,
                        expected_snapshot_digest=receipt["snapshot_input_sha256"], diagnostics_dir=diagnostic_path,
                        **(dict(oracle_labels=capture.oracle_labels) if oracle else {}),
                        skip_unusable_target=self.stage == 'action',
                    )
                timings["perception"] = (time.perf_counter() - perceived_at) * 1000
                diagnostics["perception_timings_ms"] = dict(result.timings_ms)
                diagnostics["backend_kind"] = result.backend_kind
                if self.stage == 'segmentation':
                    diagnostics['sam3_executed'] = result.backend_kind == 'sam3'
                    diagnostics['artifacts'] = getattr(result, 'diagnostics', None)
                camera_results = getattr(result, "inference", {}).get("cameras", {})
                camera_peaks = {name: row.get("peak_vram_bytes") for name, row in camera_results.items()}
                measured_peaks = [value for value in camera_peaks.values() if type(value) is int and value >= 0]
                diagnostics["peak_gpu_memory_bytes"] = max(measured_peaks) if measured_peaks else None
                diagnostics["peak_gpu_memory_scope"] = "maximum SAM3 per-camera torch allocator peak; policy forward excluded"
                diagnostics["camera_peak_vram_bytes"] = camera_peaks
                diagnostics['camera_roles'] = camera_role_diagnostics(camera_results)
                if getattr(result, 'guidance', None) is not None:
                    diagnostics['guidance_roles'] = guidance_role_diagnostics(result.guidance)
                if result.backend_kind != "sam3":
                    raise ValueError("Mock perception cannot condition an online action")
                returned_receipt = _receipt(result.receipt)
                if any(type(returned_receipt[k]) is not type(receipt[k]) or returned_receipt[k] != receipt[k]
                       for k in _RECEIPT_KEYS):
                    raise ValueError("perception returned stale or mismatched current-frame geometry")
                geometry = result.geometry.detached()
                if geometry.object_points.shape[0] != 1:
                    raise ValueError("online policy supports exactly one current observation")
                diagnostics["roles"] = {
                    role: dict(status=geometry.statuses[0][index],
                               role_valid=bool(geometry.object_role_valid[0, index]),
                               point_count=int(geometry.object_point_valid[0, index].sum()),
                               geometry_confidence=float(geometry.object_geometry_confidence[0, index]))
                    for index, role in enumerate(("target", "reference"))
                }
                for role, row in getattr(result, 'fused_geometry', {}).get('roles', {}).items():
                    if role in diagnostics['roles']:
                        diagnostics['roles'][role].update(
                            failure_reason=row.get('failure_reason'),
                            role_confidence=row.get('role_confidence'),
                            support_cameras=row.get('support_cameras', []))
                if oracle:
                    from bridgevla.models.slot_guided_geometry import sha256_file
                    if (any(meta.get('guidance_source') != 'oracle_gt' for meta in geometry.provenance)
                            or sha256_file(self.checkpoint_path) != self.expected_checkpoint_sha256):
                        raise ValueError('Oracle SAM3 source or loaded policy checkpoint mismatch')
                elif any(meta.get("source_checkpoint_sha256") != self.expected_checkpoint_sha256
                         for meta in geometry.provenance):
                    raise ValueError("Coarse checkpoint changed or differs from the loaded policy checkpoint")
                if self.stage == 'segmentation':
                    diagnostics['status'] = 'segmentation_only'
                    return self._finish(original, diagnostics, started)
                if (not bool(geometry.object_role_valid[0, 0])
                        or float(geometry.object_geometry_confidence[0, 0]) < .25):
                    raise ValueError("SAM3/fusion produced no reliable current Target geometry")
                conditioned_observation = dict(observation)
                conditioned_observation["slot_guided_current_frames"] = [receipt]
                self.agent.enable_slot_guided_sam3 = True
                action_started = time.perf_counter()
                with live_phase(f'call={self._frame_sequence} conditioned policy', prefix='SlotSAM3'):
                    conditioned = _policy_result(self.agent.act(
                        step, conditioned_observation, deterministic=deterministic,
                        slot_guided_sam3=dict(execution_mode=self.execution_mode, geometry=geometry,
                                              bridge_checkpoint_path=str(self.checkpoint_path),
                                              bridge_git_sha=self.bridge_git_sha), **kwargs,
                    ))
                timings["conditioned_action"] = (time.perf_counter() - action_started) * 1000
                diagnostics["status"] = "conditioned"
                return self._finish(conditioned, diagnostics, started)
        except Exception as error:
            if isinstance(error, UnavailableTargetError):
                diagnostics.update(error.perception_diagnostics)
                timings['perception'] = error.perception_diagnostics['perception_timings_ms']['frame_total']
            diagnostics.update(status="baseline_fallback", failure_reason=f"{type(error).__name__}: {error}")
            if original is None or self.fallback == "fail":
                diagnostics["status"] = "failed"
                timings["policy_total"] = (time.perf_counter() - started) * 1000
                self.last_diagnostics = diagnostics
                print_target_diagnostics(diagnostics, call=self._frame_sequence)
                raise
            return self._finish(original, diagnostics, started)
        finally:
            self.agent.enable_slot_guided_sam3 = previous_flag
