"""Persistent, current-frame in-memory Slot/SAM3 perception.

No frame JSON/NPZ, episode pickle, ground-truth role mask or expert action is
read. TASK owns prompt construction, segmentation and instance fusion. This
module supplies their in-memory contract and samples detached policy geometry;
it does not grant control authorization or claim manipulation acceptance.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Mapping

import numpy as np

from bridgevla.offline.slot_sam3_perception import TaskPerception


@dataclass(frozen=True)
class OnlinePerceptionResult:
    geometry: Any
    receipt: dict[str, Any]
    guidance: Any
    inference: dict[str, Any]
    fused_geometry: dict[str, Any]
    backend_kind: str
    evidence_kind: str
    timings_ms: dict[str, float]
    diagnostics: dict[str, str] | None = None


class OnlineTaskPerception(TaskPerception):
    """Keep one frozen SAM3 backend; recompute image states for every new frame.

    Constructor and prepare_backend have the same interface as TaskPerception.
    Model state persists, but masks/geometry/image features are never reused
    across source observations. One instance processes frames sequentially.
    Per-episode source indices must increase strictly, including after a
    segmentation/fusion failure; the controller must fall back for that frame.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_source_indices: dict[str, int] = {}

    def reset(self) -> None:
        """Start fresh episode/frame bookkeeping while retaining model weights."""
        self._last_source_indices.clear()

    def prepare_model(self) -> float:
        """Load the real frozen backend once, before the simulator is launched."""
        started = time.perf_counter()
        if self._backend is None:
            api = self._load_api()
            if self.sam3_checkpoint is None:
                raise ValueError('an actual local SAM3 checkpoint is required')
            self._backend = api.slot_guided_sam3.Sam3Backend(self.sam3_checkpoint, device=self.device)
        return (time.perf_counter() - started) * 1000

    @staticmethod
    def _export(packet: Mapping[str, Any], snapshot: Any) -> dict[str, Any]:
        if not isinstance(packet, Mapping):
            raise ValueError("coarse_packet must be the detached current Coarse export")
        required = {"slot_output", "point_valid", "point_support_source"}
        if not required.issubset(packet):
            raise ValueError("coarse_packet lacks current points or independent renderer support")
        if "transform" in packet:
            transform = copy.deepcopy(dict(packet["transform"]))
        else:
            if "matrix_world_from_source" not in packet or "source_frame" not in packet:
                raise ValueError("coarse_packet must include the actual complete world inverse")
            matrix = np.asarray(packet["matrix_world_from_source"])
            if matrix.shape != (4, 4) or matrix.dtype.kind not in "fiu" or not np.isfinite(matrix).all():
                raise ValueError("coarse world inverse must be a finite numeric [4,4] matrix")
            transform = dict(
                source_frame=packet["source_frame"], target_frame="rlbench_world",
                matrix_world_from_source=matrix.tolist(),
                source_frame_index=snapshot.metadata["source_frame_index"],
                timestamp=snapshot.metadata["observed_at"],
                source="current policy Coarse renderer support and actual scene inverse",
            )
        return {**{key: packet[key] for key in required}, "transform": transform,
                **{key: packet[key] for key in ("reference_exists", "role_visibility") if key in packet}}

    def perceive(self, snapshot: Any, coarse_packet: Mapping[str, Any], *, instruction: str,
                 task_name: str, checkpoint_path: str | Path, bridge_git_sha: str,
                 point_count: int = 128, expected_snapshot_digest: str | None = None,
                 diagnostics_dir: str | Path | None = None) -> OnlinePerceptionResult:
        """Compute fresh geometry directly from the current in-memory snapshot.

        Checkpoint paths identify actual loaded weights. TASK verifies SAM3's
        source API and hashes the local weights; fusion records the real Coarse
        checkpoint digest. Mock remains explicitly marked and is never an
        authorization. Optional diagnostic writes occur only after inference.
        """
        api = self._load_api()
        receipt = self.snapshot_receipt(snapshot)
        if expected_snapshot_digest is not None and receipt["snapshot_input_sha256"] != expected_snapshot_digest:
            raise ValueError("Current RGB-D changed since the Coarse policy observation")
        episode, index = receipt["episode_id"], receipt["source_frame_index"]
        if index <= self._last_source_indices.get(episode, -1):
            raise ValueError("Online perception requires a fresh strictly increasing source frame; no stale geometry reuse")
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError("Actual loaded Coarse checkpoint is required")
        if isinstance(point_count, bool) or not isinstance(point_count, int) or point_count < 1:
            raise ValueError("point_count must be a positive integer")
        diagnostic_path = Path(diagnostics_dir).resolve() if diagnostics_dir is not None else None
        if diagnostic_path is not None and diagnostic_path.exists() and (not diagnostic_path.is_dir() or any(diagnostic_path.iterdir())):
            raise FileExistsError("Use a new empty diagnostics directory for this observation")
        export = self._export(coarse_packet, snapshot)
        # Model load can be moved out of frame timing by calling prepare_backend.
        model_started = time.perf_counter()
        backend, kind = self._active_backend(snapshot)
        model_ms = (time.perf_counter() - model_started) * 1000
        self._last_source_indices[episode] = index
        started = time.perf_counter()
        guidance = api.slot_guidance_producers.coarse_slot_guidance(
            snapshot, export["slot_output"], point_valid=export["point_valid"],
            transform=export["transform"], instruction=instruction, task_name=task_name,
            source_checkpoint=str(checkpoint), source_git_sha=bridge_git_sha,
            created_at=datetime.now(timezone.utc).isoformat(),
            point_support_source=export["point_support_source"],
            **{key: export[key] for key in ("reference_exists", "role_visibility") if key in export},
        )
        produced = time.perf_counter()
        fingerprints = {name: (backend[name] if isinstance(backend, Mapping) else backend).fingerprint()
                        for name in snapshot.cameras}
        cache_key = api.slot_guided_io.inference_cache_key(
            snapshot, guidance, fingerprints, mode=self.mode,
            adapter_config=self._config.adapter, selection_config=self._config.selection,
        )
        cameras = {}
        for name, frame in snapshot.cameras.items():
            active = backend[name] if isinstance(backend, Mapping) else backend
            cameras[name] = api.slot_guided_sam3.run_camera(
                guidance, frame.camera, frame.rgb, frame.depth_m, active,
                mode=self.mode, adapter_config=self._config.adapter,
                selection_config=self._config.selection,
            )
        segmented = time.perf_counter()
        if api.slot_guided_io.snapshot_input_digest(snapshot) != receipt["snapshot_input_sha256"]:
            raise ValueError("Current RGB-D snapshot was mutated during perception")
        inference = {
            "schema_version": 1, "artifact_type": "slot_sam3_frame", "cache_key": cache_key,
            **{key: receipt[key] for key in ("episode_id", "frame_id", "source_frame_index", "observed_at")},
            "coordinate_frame": "rlbench_world", "coordinate_units": "m", "mode": self.mode,
            "guidance_source": guidance.metadata["guidance_source"], "backend": fingerprints,
            "cameras": cameras, "guidance_metadata": guidance.metadata,
            "reference_null_probability": guidance.metadata["reference_null_probability"],
            "source_git_sha": guidance.metadata["source_git_sha"],
            "source_checkpoint": guidance.metadata["source_checkpoint"],
            "calibration_sha256": receipt["calibration_sha256"],
            "snapshot_input_sha256": receipt["snapshot_input_sha256"],
            "guidance_input_sha256": api.slot_guided_io.guidance_input_digest(guidance),
            "effective_config": {"adapter": asdict(self._config.adapter), "selection": asdict(self._config.selection)},
        }
        fused = api.slot_geometry_fusion.fuse_geometry(snapshot, guidance, inference, config=self._config.fusion)
        fused_at = time.perf_counter()
        # The sampler is the existing BridgeVLA fixed geometry interface. No
        # serialized PR3 geometry is read back into the action path.
        from bridgevla.models.slot_guided_geometry import sample_geometry

        provenance = {key: copy.deepcopy(value) for key, value in fused.items() if key != "points"}
        provenance.update(snapshot_input_sha256=receipt["snapshot_input_sha256"],
                          max_source_frame_index_read=index, execution_mode="online_experiment")
        geometry = sample_geometry(
            [[fused["points"][f"{role}_points_world"] for role in ("target", "reference")]],
            [[fused["roles"][role]["status"] for role in ("target", "reference")]],
            [[fused["roles"][role]["geometry_confidence"] for role in ("target", "reference")]],
            [fused["reference_null_probability"]], [provenance], count=point_count,
        ).detached()
        timings = {"model_load": model_ms, "guidance": (produced - started) * 1000,
                   "segmentation": (segmented - produced) * 1000,
                   "fusion": (fused_at - segmented) * 1000,
                   "frame_total": (time.perf_counter() - started) * 1000}
        diagnostics = None
        if diagnostic_path is not None:
            api.slot_guided_io.save_inference_artifact(diagnostic_path / "inference.json", inference, snapshot)
            api.slot_geometry_fusion.save_geometry(diagnostic_path / "object_geometry.json", fused)
            diagnostics = {"inference_path": str(diagnostic_path / "inference.json"),
                           "object_geometry_path": str(diagnostic_path / "object_geometry.json")}
        return OnlinePerceptionResult(geometry, receipt, guidance, inference, fused, kind,
                                      "synthetic_mock" if kind == "mock" else "real_online_inference",
                                      timings, diagnostics)
