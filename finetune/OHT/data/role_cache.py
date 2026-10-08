"""Teacher and prediction caches have different namespaces and provenance."""
import json
from pathlib import Path
import numpy as np
from .common import digest, file_digest, inside, write_json


def role_fields(kind, points, valid, present, known=None, confidence=None):
    points = np.asarray(points, dtype=np.float32)
    valid = np.asarray(valid)
    present = np.asarray(present)
    if points.ndim != 3 or points.shape[0] != 2 or points.shape[2] != 3 or not np.isfinite(points).all():
        raise ValueError("Role points must be finite [2,N,3]")
    for name, values in (("valid", valid), ("present", present)):
        if values.shape != (2,) or values.dtype != np.bool_:
            raise ValueError(f"{name} must be bool[2]")
    if np.any(valid & ~present):
        raise ValueError("Absent role cannot have valid geometry")
    result = {}
    prefix = "oracle" if kind == "teacher" else "predicted"
    if kind not in ("teacher", "predicted"):
        raise ValueError("Unknown role cache kind")
    for index, role in enumerate(("target", "reference")):
        result[f"{prefix}_{role}_object_points"] = points[index]
        result[f"{prefix}_{role}_object_valid"] = valid[index]
        result[f"{prefix}_{role}_present"] = present[index]
    if kind == "teacher":
        known = np.asarray(known)
        if known.shape != (2,) or known.dtype != np.bool_:
            raise ValueError("Teacher known must be bool[2]")
        if np.any(valid & ~known):
            raise ValueError("Geometry teacher requires known role assignment")
        result["oracle_role_present_known"] = known
    else:
        confidence = np.asarray(confidence, dtype=np.float32)
        if confidence.shape != (2,) or not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("Prediction confidence must be finite [2] within [0,1]")
        for index, role in enumerate(("target", "reference")):
            result[f"predicted_{role}_confidence"] = confidence[index]
    return result


def validate_fields(fields, kind, point_count):
    prefix = "oracle" if kind == "teacher" else "predicted"
    points, valid, present, confidence = [], [], [], []
    for role in ("target", "reference"):
        points.append(fields[f"{prefix}_{role}_object_points"])
        valid.append(fields[f"{prefix}_{role}_object_valid"])
        present.append(fields[f"{prefix}_{role}_present"])
        if kind == "predicted":
            confidence.append(fields[f"predicted_{role}_confidence"])
    result = role_fields(kind, np.asarray(points), np.asarray(valid), np.asarray(present),
                         fields.get("oracle_role_present_known"), confidence or None)
    if result[f"{prefix}_target_object_points"].shape != (point_count, 3):
        raise ValueError("Role point count does not match configuration")
    if set(fields) != set(result):
        raise ValueError("Unexpected or missing role cache fields")
    return result


def create_cache(output, kind, replay_contract, point_count, provenance, rows):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite cache {output}")
    if not provenance:
        raise ValueError("Role cache provenance is required")
    output.mkdir(parents=True)
    index = {}
    for sample_id, fields in rows:
        fields = validate_fields(fields, kind, point_count)
        if sample_id in index:
            raise ValueError(f"Duplicate cache ID: {sample_id}")
        relative = (Path("roles") / (sample_id + ".npz")).as_posix()
        path = inside(output, relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **fields)
        index[sample_id] = dict(path=relative, sha256=file_digest(path))
    manifest = dict(schema="oht_role_cache_v1", kind=kind, replay_contract_sha256=replay_contract,
                    point_count=point_count, provenance=provenance, samples=index)
    manifest["sha256"] = digest(manifest)
    write_json(output / "manifest.json", manifest)
    return manifest


class RoleCache:
    def __init__(self, root, kind, replay_contract, point_count):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        contents = dict(self.manifest)
        actual = contents.pop("sha256", None)
        if actual != digest(contents):
            raise ValueError("Role cache manifest checksum mismatch")
        for key, value in dict(schema="oht_role_cache_v1", kind=kind,
                               replay_contract_sha256=replay_contract,
                               point_count=point_count).items():
            if self.manifest.get(key) != value:
                raise ValueError(f"Role cache contract mismatch: {key}")
        self.kind, self.point_count = kind, point_count

    def read(self, sample_id):
        record = self.manifest["samples"].get(sample_id)
        if record is None:
            raise ValueError(f"Missing {self.kind} cache for {sample_id}")
        path = inside(self.root, record["path"])
        if file_digest(path) != record["sha256"]:
            raise ValueError(f"Role cache data changed: {sample_id}")
        with np.load(path, allow_pickle=False) as archive:
            fields = {key: archive[key] for key in archive.files}
        return validate_fields(fields, self.kind, self.point_count)
