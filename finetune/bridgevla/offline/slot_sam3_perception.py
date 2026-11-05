"""Current-frame BridgeVLA/TASK interop, without a simulator or control loop.

TASK remains the owner of Guidance, pixel prompts, SAM3 and RGB-D fusion. This
module only connects those public interfaces and converts one calibrated frame
to the existing BridgeVLA observation tensor layout. It grants no action
authorization and never reads episode observations or expert actions.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


_TASK_MODULES = (
    "camera_geometry", "slot_guidance", "slot_guidance_producers",
    "slot_guided_io", "slot_guided_sam3", "slot_guided_config", "slot_geometry_fusion",
)


def snapshot_to_observation(snapshot: Any, low_dim_state: Any, language_goal: str,
                            cameras: Sequence[str], device: str = "cpu") -> dict[str, Any]:
    """Build B=1,T=1 tensors consumed by data.observations.preprocess_inputs.

    First load the snapshot through TaskPerception, which establishes the TASK
    module location. RGB stays uint8 (the existing preprocessor normalizes it).
    XYZ is world metres. Invalid depth stays NaN rather than becoming invented
    world-origin points; the policy's existing move_pc_in_bound removes it.
    low_dim_state must already be the actual checkpoint-compatible current
    proprioceptive vector, including any required time component.
    """
    try:
        geometry = importlib.import_module("camera_geometry")
        producers = importlib.import_module("slot_guidance_producers")
    except ImportError as exc:
        raise RuntimeError("Load a TASK snapshot through TaskPerception first") from exc
    producers._validate_snapshot(snapshot)
    if not isinstance(language_goal, str) or not language_goal.strip():
        raise ValueError("language_goal must be a nonempty current instruction")
    if isinstance(cameras, (str, bytes)):
        raise ValueError("cameras must be an explicit sequence of camera names")
    names = list(cameras)
    if not names or len(set(names)) != len(names) or any(not isinstance(n, str) or n not in snapshot.cameras for n in names):
        raise ValueError("cameras must be unique calibrated cameras present in the snapshot")
    state = np.asarray(low_dim_state)
    if state.ndim != 1 or len(state) < 3 or state.dtype.kind not in "fiu" or not np.isfinite(state).all():
        raise ValueError("low_dim_state must be a finite current-state vector with at least three entries")
    import torch

    observation = {
        "low_dim_state": torch.as_tensor(np.array(state, dtype=np.float32, copy=True), device=device)[None, None],
        "language_goal": [language_goal],
    }
    for name in names:
        frame = snapshot.cameras[name]
        depth = np.asarray(frame.depth_m)
        valid = np.isfinite(depth) & (depth > 0)
        camera_points = geometry.backproject_mask(depth, valid, np.asarray(frame.camera.intrinsics), max_points=0)
        world_points = geometry.transform_points(camera_points, np.asarray(frame.camera.extrinsics))
        dense = np.full((*depth.shape, 3), np.nan, dtype=np.float32)
        dense[valid] = world_points
        rgb = np.ascontiguousarray(np.asarray(frame.rgb).transpose(2, 0, 1))
        cloud = np.ascontiguousarray(dense.transpose(2, 0, 1))
        observation[f"{name}_rgb"] = torch.as_tensor(rgb, device=device)[None, None]
        observation[f"{name}_point_cloud"] = torch.as_tensor(cloud, device=device)[None, None]
    return observation


class TaskPerception:
    """Lazily connect one selected TASK checkout to frozen offline perception.

    An injected backend is restricted to Mock evidence. The production SAM3
    backend is constructed internally and performs TASK's installed-source and
    checkpoint verification. Do not call one instance concurrently: SAM3's
    interactive predictor temporarily mutates its cached state.
    """

    def __init__(self, task_repo: str | Path, config_path: str | Path | None = None,
                 sam3_checkpoint: str | Path | None = None, device: str = "cpu",
                 mode: str = "point+box", backend: Any = None):
        if mode not in ("box-only", "point+box"):
            raise ValueError("mode must be box-only or point+box")
        self.task_repo = Path(task_repo).expanduser().resolve()
        self.config_path = Path(config_path) if config_path is not None else None
        self.sam3_checkpoint = sam3_checkpoint
        self.device, self.mode = device, mode
        self._backend, self._api, self._config = backend, None, None
        self._injected_backend = backend is not None

    def _load_api(self):
        if self._api is not None:
            return self._api
        for name in _TASK_MODULES:
            if not (self.task_repo / f"{name}.py").is_file():
                raise FileNotFoundError(f"TASK checkout is missing {name}.py: {self.task_repo}")
            loaded = sys.modules.get(name)
            filename = getattr(loaded, "__file__", None)
            if loaded is not None and (filename is None or Path(filename).resolve().parent != self.task_repo):
                raise RuntimeError(f"{name} was already imported from another checkout; use a fresh process")
        root = str(self.task_repo)
        if root not in sys.path:
            sys.path.insert(0, root)
        api = SimpleNamespace(**{name: importlib.import_module(name) for name in _TASK_MODULES})
        config_path = self.config_path
        if config_path is not None and not config_path.is_absolute():
            config_path = self.task_repo / config_path
        config = api.slot_guided_config.load_slot_guided_config(config_path)
        self._api, self._config = api, config
        return self._api

    def load_snapshot(self, frame_path: str | Path):
        return self._load_api().slot_guided_io.load_frame_snapshot(frame_path)

    def validate_policy_snapshot(self, snapshot: Any, cameras: Sequence[str]) -> None:
        """Validate policy camera availability and calibration without Torch/SAM3."""
        self._load_api().slot_guidance_producers._validate_snapshot(snapshot)
        missing = set(cameras) - set(snapshot.cameras)
        if missing:
            raise ValueError(f"Snapshot is missing policy cameras: {sorted(missing)}")

    def snapshot_receipt(self, snapshot: Any) -> dict[str, Any]:
        """Identify the exact current arrays used for Coarse and later SAM3."""
        api = self._load_api()
        api.slot_guidance_producers._validate_snapshot(snapshot)
        return {
            **{key: snapshot.metadata[key] for key in ("episode_id", "frame_id", "source_frame_index", "observed_at")},
            "calibration_sha256": api.slot_guided_io.calibration_digest(snapshot),
            "snapshot_input_sha256": api.slot_guided_io.snapshot_input_digest(snapshot),
            "max_source_frame_index_read": snapshot.metadata["source_frame_index"],
        }

    def snapshot_to_observation(self, snapshot: Any, low_dim_state: Any, language_goal: str,
                                cameras: Sequence[str], device: str | None = None) -> dict[str, Any]:
        self._load_api()
        return snapshot_to_observation(snapshot, low_dim_state, language_goal, cameras,
                                       self.device if device is None else device)

    def _active_backend(self, snapshot):
        api = self._load_api()
        if self._backend is None:
            if self.sam3_checkpoint is None:
                raise ValueError("An explicit local SAM3 checkpoint is required")
            self._backend = api.slot_guided_sam3.Sam3Backend(self.sam3_checkpoint, device=self.device)
        fingerprints = []
        for name in snapshot.cameras:
            item = self._backend[name] if isinstance(self._backend, Mapping) else self._backend
            fingerprint = item.fingerprint()
            kind = fingerprint.get("kind")
            if kind not in ("sam3", "mock") or (self._injected_backend and kind != "mock"):
                raise ValueError("Injected backends are permitted only as explicit Mock evidence")
            fingerprints.append(kind)
        if len(set(fingerprints)) != 1:
            raise ValueError("All cameras must use the same backend kind")
        return self._backend, fingerprints[0]

    def prepare_backend(self, snapshot: Any) -> str:
        """Load frozen SAM3 before per-frame timing; does not encode an image."""
        return self._active_backend(snapshot)[1]

    def perceive(self, snapshot_path: str | Path, coarse_export: Mapping[str, Any] | None,
                 instruction: str, task_name: str, checkpoint_path: str | Path | None,
                 bridge_git_sha: str, output_dir: str | Path,
                 source: str = "predicted_coarse_slot", *, oracle_role_masks: Any = None,
                 reference_is_null: bool | None = None, oracle_instance_ids: Mapping[str, str] | None = None,
                 created_at: str | None = None, expected_snapshot_digest: str | None = None) -> dict[str, Any]:
        """Export Guidance, segment current RGB and persist fused world geometry.

        coarse_export contains slot_output, point_valid, transform and
        point_support_source. Optional reference_exists/role_visibility preserve
        explicit semantic/visibility claims. Oracle mode requires separately
        supplied current GT masks and an explicit Reference-NULL decision.
        The result is inference evidence, never an acceptance/authorization.
        """
        if source not in ("predicted_coarse_slot", "oracle_gt"):
            raise ValueError("source must be predicted_coarse_slot or oracle_gt")
        api = self._load_api()
        snapshot = self.load_snapshot(snapshot_path)
        receipt = self.snapshot_receipt(snapshot)
        if expected_snapshot_digest is not None and receipt["snapshot_input_sha256"] != expected_snapshot_digest:
            raise ValueError("Current snapshot changed since Coarse observation: snapshot_input_sha256 mismatch")
        output = Path(output_dir).expanduser().resolve()
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError(f"Perception requires a new empty output directory: {output}")
        stamp = created_at or datetime.now(timezone.utc).isoformat()
        export = dict(coarse_export or {})
        if source == "predicted_coarse_slot":
            required = {"slot_output", "point_valid", "transform", "point_support_source"}
            allowed = required | {"reference_exists", "role_visibility"}
            if not required.issubset(export) or set(export) - allowed:
                raise ValueError("coarse_export must contain detached slots, support and the actual inverse transform")
            if checkpoint_path is None or not Path(checkpoint_path).is_file():
                raise FileNotFoundError("The actual Coarse checkpoint file is required for provenance")
            guidance = api.slot_guidance_producers.coarse_slot_guidance(
                snapshot, export["slot_output"], point_valid=export["point_valid"],
                transform=export["transform"], instruction=instruction, task_name=task_name,
                source_checkpoint=str(Path(checkpoint_path).resolve()), source_git_sha=bridge_git_sha,
                created_at=stamp, point_support_source=export["point_support_source"],
                **{key: export[key] for key in ("reference_exists", "role_visibility") if key in export},
            )
        else:
            masks = oracle_role_masks if oracle_role_masks is not None else export.get("role_masks")
            null = reference_is_null if reference_is_null is not None else export.get("reference_is_null")
            guidance = api.slot_guidance_producers.oracle_guidance(
                snapshot, masks, instruction=instruction, task_name=task_name,
                reference_is_null=null, source_git_sha=bridge_git_sha, created_at=stamp,
                role_instance_ids=oracle_instance_ids,
            )
        backend, kind = self._active_backend(snapshot)
        guidance_path = output / "guidance.json"
        # Keep the persisted metadata, including its real sidecar identity.
        guidance = api.slot_guidance.save_guidance(
            guidance_path, guidance.metadata, target_points_world=guidance.target_points_world,
            reference_points_world=guidance.reference_points_world,
        )
        inference = api.slot_guided_io.run_offline_frame(
            snapshot_path, guidance_path, backend, output / "sam3", mode=self.mode,
            adapter_config=self._config.adapter, selection_config=self._config.selection,
        )
        geometry = api.slot_geometry_fusion.fuse_geometry(snapshot, guidance, inference, config=self._config.fusion)
        geometry_path = output / "object_geometry.json"
        api.slot_geometry_fusion.save_geometry(geometry_path, geometry)
        report_path = output / "perception_report.json"
        report = {
            "schema_version": 1, "artifact_type": "bridgevla_offline_perception",
            "object_geometry_path": str(geometry_path), "guidance_path": str(guidance_path),
            "inference_path": inference["artifact_path"], "report_path": str(report_path),
            "backend_kind": kind, "evidence_kind": "synthetic_mock" if kind == "mock" else "real_offline_inference",
            "guidance_source": source, "mode": self.mode,
            "calibration_sha256": geometry["calibration_sha256"],
            "snapshot_input_sha256": receipt["snapshot_input_sha256"],
            "max_source_frame_index_read": receipt["max_source_frame_index_read"],
            "source_checkpoint_sha256": geometry["source_checkpoint_sha256"],
            "sam3_checkpoint_sha256": geometry["sam3_checkpoint_sha256"],
            **{key: snapshot.metadata[key] for key in ("episode_id", "frame_id", "source_frame_index", "observed_at")},
            "roles": {role: {key: row[key] for key in ("status", "point_count", "failure_reason")}
                      for role, row in geometry["roles"].items()},
        }
        api.slot_guided_io.atomic_json(report_path, report)
        return report
