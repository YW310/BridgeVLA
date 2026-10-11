"""Current-frame object pose metadata for diagnostics, never policy tensors."""
import json
from pathlib import Path

import numpy as np

from .common import digest, file_digest, inside
from .geometry import array, pose_matrix, quaternion


def capture_object_poses(columns, frame, record):
    positions = array(columns["observation.objects_pos"][frame], (12,), "objects_pos").reshape(4, 3)
    orientations = array(columns["observation.objects_quat"][frame], (16,), "objects_quat").reshape(4, 4)
    quaternion(orientations)  # Validate norms without assuming a component order.
    return dict(frame=int(frame), positions=positions.tolist(), quaternions=orientations.tolist(),
                source="raw_parquet", path=record["path"], parquet_sha256=record["parquet_sha256"])


def load_source_objects(root, manifest_path, contract, sample):
    """Recover poses for an indexed old cache using its original, bound audit."""
    if sample.get("frame") is None or sample.get("timestamp") is None:
        raise ValueError("Object source requires an indexed sample with frame/timestamp; direct NPZ is unsupported")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    value = dict(manifest)
    recorded = value.pop("manifest_sha256", None)
    if (recorded != digest(value) or recorded != contract.get("audit_manifest_sha256") or
            manifest.get("schema") != "oht_audit_v1"):
        raise ValueError("Object source audit does not match replay contract")
    records = [r for r in manifest["episodes"]
               if r["task"] == sample["task"] and r["episode_index"] == sample["episode_index"]
               and r["dataset"] == sample.get("data_profile")]
    if len(records) != 1:
        raise ValueError("Object source requires exactly one matching audited episode")
    record = records[0]
    path = inside(root, record["path"])
    if file_digest(path) != record["parquet_sha256"]:
        raise ValueError("Object source Parquet changed after audit")
    import pyarrow.parquet as pq
    columns = pq.read_table(path, columns=["frame_index", "episode_index", "timestamp",
                                         "observation.objects_pos", "observation.objects_quat"]).to_pydict()
    frame = sample["frame"]
    if (type(frame) is not int or not 0 <= frame < len(columns["frame_index"]) or
            columns["frame_index"][frame] != frame or
            columns["episode_index"][frame] != sample["episode_index"] or
            not np.isclose(columns["timestamp"][frame], sample["timestamp"], rtol=0, atol=1e-9)):
        raise ValueError("Object source frame/episode/timestamp does not match cached sample")
    return capture_object_poses(columns, frame, record)


def object_payload(sample):
    metadata = sample.get("diagnostic_objects")
    if metadata is None:
        return []
    if metadata.get("frame") != sample.get("frame") or sample.get("frame") is None:
        raise ValueError("Diagnostic object poses must belong to the current sample frame")
    positions = array(metadata["positions"], (4, 3), "diagnostic object positions")
    orientations = array(metadata["quaternions"], (4, 4), "diagnostic object quaternions")
    quaternion(orientations)
    objects = []
    for index, (position, orientation) in enumerate(zip(positions, orientations)):
        # The reference converter does not establish object quaternion order.
        # Supply both bases; the viewer requires an explicit user selection.
        axes = {order: pose_matrix(position, orientation if order == "xyzw" else orientation[[1, 2, 3, 0]])
                [:3, :3].T.tolist() for order in ("wxyz", "xyzw")}
        objects.append(dict(index=index, name=f"object_{index}", position=position.tolist(),
                            quaternion=orientation.tolist(), axes=axes))
    return objects
