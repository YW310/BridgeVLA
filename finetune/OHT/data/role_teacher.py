"""Build teachers from explicit per-frame masks or annotated site regions."""
import json
from pathlib import Path
from .source_config import sample_data_config
import numpy as np
from .common import read_jsonl, inside, file_digest
from .geometry import transform_matrix
from .replay import load_contract
from .role_cache import role_fields, create_cache
from .visualization import PreviewWriter


def sample_points(points, count, seed):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if not len(points):
        return np.zeros((count, 3), dtype=np.float32), False
    generator = np.random.default_rng(seed)
    choice = generator.choice(len(points), count, replace=len(points) < count)
    return points[choice], True


def build_teacher(replay, annotations_path, output, point_count=512, *,
                  visualize_every=0, visualize_output_dir=None):
    replay = Path(replay)
    contract = load_contract(replay)
    samples = {row["id"]: row for row in read_jsonl(replay / "samples.jsonl")}
    annotations = read_jsonl(annotations_path)
    if len({row["id"] for row in annotations}) != len(annotations):
        raise ValueError("Duplicate teacher annotation IDs")
    base = Path(annotations_path).resolve().parent
    source_hashes = {}
    preview = PreviewWriter(visualize_output_dir or Path(output) / "visualizations", visualize_every)
    def rows():
        for annotation in annotations:
            sample = samples.get(annotation["id"])
            if sample is None:
                raise ValueError(f"Teacher references unknown sample: {annotation['id']}")
            observation_path = inside(replay, sample["observation"])
            if file_digest(observation_path) != sample["observation_sha256"]:
                raise ValueError("Observation changed since replay construction")
            with np.load(observation_path, allow_pickle=False) as data:
                points, valid, present, known = [], [], [], []
                role_masks = {}
                for index, role in enumerate(("target", "reference")):
                    spec = annotation[role]
                    if type(spec.get("present")) is not bool or type(spec.get("known")) is not bool:
                        raise ValueError("Teacher present/known require explicit booleans")
                    present.append(spec["present"])
                    known.append(spec["known"])
                    source = spec.get("source")
                    cloud = np.empty((0, 3))
                    if source == "visible_surface":
                        if not spec["present"] or not spec["known"]:
                            raise ValueError("Visible surface requires a known present role")
                        chunks = []
                        if preview.every:
                            role_masks[role] = {}
                        mask_path = inside(base, spec["mask_path"])
                        source_hashes[str(mask_path)] = file_digest(mask_path)
                        with np.load(mask_path, allow_pickle=False) as masks:
                            for camera in contract["data_config"]["cameras"]:
                                mask = masks[camera]
                                cloud_camera = data[f"{camera}_point_cloud"].transpose(1, 2, 0)
                                if mask.dtype != np.bool_ or mask.shape != cloud_camera.shape[:2]:
                                    raise ValueError("Masks must be bool at cached RGB-D resolution")
                                chunks.append(cloud_camera[mask])
                                if preview.every:
                                    role_masks[role][camera] = mask
                        cloud = np.concatenate(chunks)
                    elif source == "site_region":
                        if not spec["present"] or not spec["known"]:
                            raise ValueError("Site requires a known present role")
                        transform = transform_matrix(spec["world_from_site"])
                        size = np.asarray(spec["size"], dtype=float)
                        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
                            raise ValueError("Site size must be positive [3]")
                        rng = np.random.default_rng(index)
                        local = rng.uniform(-.5, .5, (point_count, 3)) * size
                        cloud = local @ transform[:3, :3].T + transform[:3, 3]
                    elif source not in ("unknown", "none"):
                        raise ValueError("source must be visible_surface/site_region/unknown/none")
                    if source == "none" and spec["present"]:
                        raise ValueError("A present role is unknown, not semantic NULL")
                    sampled, usable = sample_points(cloud, point_count, index)
                    points.append(sampled)
                    valid.append(usable)
                fields = role_fields("teacher", points, np.asarray(valid, bool),
                                     np.asarray(present, bool), np.asarray(known, bool))
                if preview.every:
                    preview.write(data, sample_data_config(contract, sample), sample,
                                  current_tcp=sample.get("current_tcp"),
                                  role_specs=annotation, role_masks=role_masks,
                                  role_points=np.asarray(points), role_valid=np.asarray(valid, bool))
                yield annotation["id"], fields
    # create_cache writes its manifest after consuming rows; this shared map then
    # contains every source checksum without holding all point clouds in memory.
    return create_cache(output, "teacher", contract["sha256"], point_count,
                        dict(annotation_sha256=file_digest(annotations_path),
                             mask_sha256=source_hashes, geometry="typed_surface_or_site"), rows())
