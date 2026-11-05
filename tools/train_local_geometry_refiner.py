"""Standalone, observation-only geometry refinement experiment on explicit NPZ caches.

This pilot does not alter BridgeVLA's action path. Real caches must contain frozen
policy predictions, CV +Z metric depth, and episode-disjoint data partitions.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetune.bridgevla.geometry.numpy_ops import validate_intrinsics, validate_rigid_transforms
from finetune.bridgevla.models.local_geometry import build_local_evidence
from finetune.bridgevla.models.local_geometry_refiner import (
    LocalGeometryRefiner, pose_error_metrics, pose_refinement_loss,
)


SCHEMA = "local_geometry_refinement_v1"
ARRAY_KEYS = ("depth", "intrinsics", "camera_to_world", "candidate_pose", "target_pose", "context")
META_KEYS = ("schema", "source", "baseline_id", "camera_model", "candidate_source")


def _text(value, name):
    if value.shape != () or value.dtype.kind != "U" or not str(value.item()).strip():
        raise ValueError(f"{name} must be a nonempty scalar Unicode string")
    return str(value.item())


def _poses(value, name):
    try:
        validate_rigid_transforms(value, name, homogeneous_atol=1e-5, rotation_atol=1e-4)
    except ValueError as error:
        raise ValueError(f"{name} must contain finite rigid right-handed 4x4 poses") from error


def load_cache(path):
    """Load numeric arrays without pickle; reject ambiguous units and provenance."""
    with np.load(path, allow_pickle=False) as archive:
        required = set(ARRAY_KEYS + META_KEYS + ("episode_id", "sample_id"))
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Cache missing fields: {sorted(missing)}")
        metadata = {key: _text(archive[key], key) for key in META_KEYS}
        arrays = {key: np.array(archive[key], copy=True) for key in ARRAY_KEYS}
        identifiers = {key: np.array(archive[key], copy=True) for key in ("episode_id", "sample_id")}
    if metadata["schema"] != SCHEMA or metadata["camera_model"] != "opencv_z_meters":
        raise ValueError("Cache requires the current schema and opencv_z_meters camera model")
    if metadata["source"] not in ("synthetic", "rlbench", "oht"):
        raise ValueError("source must be synthetic, rlbench or oht")
    expected = "synthetic_perturbation" if metadata["source"] == "synthetic" else "policy_prediction"
    if metadata["candidate_source"] != expected:
        raise ValueError(f"{metadata['source']} candidates require candidate_source={expected}")
    depth = arrays["depth"]
    if depth.ndim != 4 or min(depth.shape) < 1:
        raise ValueError("depth must have nonempty shape [N,V,H,W]")
    count, views = depth.shape[:2]
    shapes = {"intrinsics": (count, views, 3, 3), "camera_to_world": (count, views, 4, 4),
              "candidate_pose": (count, 4, 4), "target_pose": (count, 4, 4)}
    for key, value in arrays.items():
        if value.dtype.kind != "f":
            raise ValueError(f"{key} must be floating point; raw integer depth is unsupported")
        if key in shapes and value.shape != shapes[key]:
            raise ValueError(f"{key} must have shape {shapes[key]}")
        if key != "depth" and not np.isfinite(value).all():
            raise ValueError(f"{key} must be finite")
    if arrays["context"].ndim != 2 or arrays["context"].shape[0] != count or arrays["context"].shape[1] < 1:
        raise ValueError("context must have shape [N,C] with C >= 1")
    for key in ("camera_to_world", "candidate_pose", "target_pose"):
        _poses(arrays[key], key)
    validate_intrinsics(arrays["intrinsics"], require_upper_triangular=True, atol=1e-6, rtol=0)
    for key, value in identifiers.items():
        if value.shape != (count,) or value.dtype.kind != "U" or any(not x.strip() for x in value.tolist()):
            raise ValueError(f"{key} must be a nonempty Unicode [N] array")
    if len(set(identifiers["sample_id"].tolist())) != count:
        raise ValueError("sample_id values must be unique")
    return {**metadata, **identifiers, **{key: torch.from_numpy(value.astype(np.float32))
                                      for key, value in arrays.items()}}


def ensure_split_separation(first, second):
    """Require the same frozen baseline and distinct episodes, not merely frames."""
    for key in META_KEYS:
        if first[key] != second[key]:
            raise ValueError(f"Cache provenance differs: {key}")
    if first["context"].shape[1] != second["context"].shape[1]:
        raise ValueError("Cache context dimensions differ")
    for key in ("episode_id", "sample_id"):
        overlap = set(first[key].tolist()) & set(second[key].tolist())
        if overlap:
            raise ValueError(f"Data partitions overlap in {key}: {sorted(overlap)[:3]}")


def _field(cache, indices, geometry):
    return build_local_evidence(cache["depth"][indices], cache["intrinsics"][indices],
                                cache["camera_to_world"][indices], cache["candidate_pose"][indices],
                                **geometry)["field"]


@torch.no_grad()
def evaluate_cache(model, cache, geometry, batch_size):
    model.eval()
    baseline, refined = {"translation_m": [], "rotation_rad": []}, {"translation_m": [], "rotation_rad": []}
    for start in range(0, len(cache["depth"]), batch_size):
        indices = slice(start, start + batch_size)
        candidate, target = cache["candidate_pose"][indices], cache["target_pose"][indices]
        output = model(_field(cache, indices, geometry), candidate, cache["context"][indices])["pose"]
        for destination, pose in ((baseline, candidate), (refined, output)):
            for key, values in pose_error_metrics(pose, target).items():
                destination[key].append(values.cpu())
    def summarize(values):
        result = {}
        for key, batches in values.items():
            errors = torch.cat(batches)
            result[key] = dict(mean=float(errors.mean()), median=float(errors.median()),
                               p90=float(torch.quantile(errors, .9)))
        return result
    return dict(source=cache["source"], samples=len(cache["depth"]),
                scope="offline_next_keyframe_pose_error", baseline=summarize(baseline), refined=summarize(refined))


def train(args):
    training, validation = load_cache(args.train_cache), load_cache(args.validation_cache)
    ensure_split_separation(training, validation)
    output = Path(args.output)
    if output.exists():
        raise ValueError(f"Refusing to overwrite checkpoint: {output}")
    torch.manual_seed(args.seed)
    model_options = dict(context_dim=training["context"].shape[1], hidden_dim=args.hidden_dim,
                         max_translation_m=args.max_translation_m, max_rotation_rad=args.max_rotation_rad)
    geometry = dict(size_m=args.size_m, resolution=args.resolution, truncation_m=args.truncation_m)
    model = LocalGeometryRefiner(**model_options)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    initial = evaluate_cache(model, validation, geometry, args.batch_size)
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = torch.randperm(len(training["depth"]))
        total = 0.
        for indices in order.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            result = model(_field(training, indices, geometry), training["candidate_pose"][indices],
                           training["context"][indices])
            loss = pose_refinement_loss(result["pose"], training["target_pose"][indices])
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(indices)
        record = dict(epoch=epoch + 1, loss=total / len(order))
        history.append(record)
        print(json.dumps(record), flush=True)
    report = dict(initial_validation=initial,
                  final_validation=evaluate_cache(model, validation, geometry, args.batch_size),
                  train_samples=len(training["depth"]), epochs=args.epochs, history=history)
    checkpoint = dict(schema=SCHEMA, model_options=model_options, geometry=geometry,
                      model_state=model.state_dict(), metadata={key: training[key] for key in META_KEYS},
                      train_episode_ids=sorted(set(training["episode_id"].tolist())),
                      validation_episode_ids=sorted(set(validation["episode_id"].tolist())),
                      train_sample_ids=training["sample_id"].tolist(),
                      validation_sample_ids=validation["sample_id"].tolist(),
                      run_config=dict(seed=args.seed, batch_size=args.batch_size,
                                      learning_rate=args.learning_rate, threads=args.threads),
                      report=report)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(json.dumps(report, indent=2))
    return report


def evaluate(args):
    cache = load_cache(args.cache)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["schema"] != SCHEMA:
        raise ValueError("Unsupported checkpoint schema")
    if any(checkpoint["metadata"][key] != cache[key] for key in META_KEYS):
        raise ValueError("Evaluation cache provenance differs from training")
    seen = set(checkpoint["train_episode_ids"]) | set(checkpoint["validation_episode_ids"])
    if seen & set(cache["episode_id"].tolist()):
        raise ValueError("Evaluation must use episodes absent from training and validation")
    seen_samples = set(checkpoint["train_sample_ids"]) | set(checkpoint["validation_sample_ids"])
    if seen_samples & set(cache["sample_id"].tolist()):
        raise ValueError("Evaluation must use sample IDs absent from training and validation")
    model = LocalGeometryRefiner(**checkpoint["model_options"])
    model.load_state_dict(checkpoint["model_state"])
    report = evaluate_cache(model, cache, checkpoint["geometry"], args.batch_size)
    print(json.dumps(report, indent=2))
    return report


def synthetic(args):
    """A recessed disk seen by one camera: a software smoke test, not a robot task."""
    directory = Path(args.output_dir)
    paths = [directory / f"{split}.npz" for split in ("train", "validation", "test")]
    if any(path.exists() for path in paths):
        raise ValueError("Refusing to overwrite synthetic caches")
    directory.mkdir(parents=True, exist_ok=True)
    for split, count, offset, path in zip(("train", "validation", "test"),
                                        (args.train_samples, args.validation_samples, args.test_samples),
                                        (0, 1, 2), paths):
        random = np.random.default_rng(args.seed + offset)
        height = width = 32
        calibration = np.array([[180., 0, 15.5], [0, 180., 15.5], [0, 0, 1]], dtype=np.float32)
        target_xy = random.uniform(-.012, .012, (count, 2)).astype(np.float32)
        target = np.tile(np.eye(4, dtype=np.float32), (count, 1, 1))
        target[:, :2, 3], target[:, 2, 3] = target_xy, .5
        candidate = target.copy()
        candidate[:, :2, 3] += random.uniform(-.009, .009, (count, 2))
        rows, columns = np.indices((height, width), dtype=np.float32)
        # Generate depths from surface coordinates, rather than writing labels into context.
        x, y = (columns - 15.5) * .5 / 180, (rows - 15.5) * .5 / 180
        disk = (x[None] - target_xy[:, 0, None, None]) ** 2 + (y[None] - target_xy[:, 1, None, None]) ** 2 < .012 ** 2
        depth = np.where(disk, .518, .5).astype(np.float32)[:, None]
        np.savez_compressed(path, schema=np.array(SCHEMA), source=np.array("synthetic"),
                            baseline_id=np.array("recessed_disk_synthetic_v1"),
                            camera_model=np.array("opencv_z_meters"), candidate_source=np.array("synthetic_perturbation"),
                            depth=depth, intrinsics=np.tile(calibration, (count, 1, 1, 1)),
                            camera_to_world=np.tile(np.eye(4, dtype=np.float32), (count, 1, 1, 1)),
                            candidate_pose=candidate, target_pose=target, context=np.zeros((count, 4), np.float32),
                            episode_id=np.array([f"synthetic/{split}/episode/{i}" for i in range(count)]),
                            sample_id=np.array([f"synthetic/{split}/sample/{i}" for i in range(count)]))
    print(json.dumps(dict(source="synthetic", scope="software_smoke_test_only", paths=[str(x) for x in paths]), indent=2))


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke = commands.add_parser("synthetic", help="Create toy train/validation/test caches")
    smoke.add_argument("--output-dir", required=True)
    for name, default in (("train-samples", 32), ("validation-samples", 12), ("test-samples", 12)):
        smoke.add_argument("--" + name, type=_positive_int, default=default)
    smoke.add_argument("--seed", type=int, default=0)
    smoke.set_defaults(run=synthetic)
    fitting = commands.add_parser("train", help="Train an independent CPU pilot on cached policy predictions")
    fitting.add_argument("--train-cache", required=True)
    fitting.add_argument("--validation-cache", required=True)
    fitting.add_argument("--output", required=True)
    fitting.add_argument("--epochs", type=_positive_int, default=10)
    fitting.add_argument("--batch-size", type=_positive_int, default=4)
    fitting.add_argument("--hidden-dim", type=_positive_int, default=16)
    fitting.add_argument("--learning-rate", type=_positive_float, default=1e-3)
    fitting.add_argument("--resolution", type=_positive_int, default=24)
    fitting.add_argument("--size-m", type=_positive_float, default=.10)
    fitting.add_argument("--truncation-m", type=_positive_float, default=.005)
    fitting.add_argument("--max-translation-m", type=_positive_float, default=.02)
    fitting.add_argument("--max-rotation-rad", type=_positive_float, default=.15)
    fitting.add_argument("--seed", type=int, default=0)
    fitting.add_argument("--threads", type=_positive_int, default=2)
    fitting.set_defaults(run=train)
    testing = commands.add_parser("evaluate", help="Measure pose errors on previously unused episodes")
    testing.add_argument("--cache", required=True)
    testing.add_argument("--checkpoint", required=True)
    testing.add_argument("--batch-size", type=_positive_int, default=4)
    testing.add_argument("--threads", type=_positive_int, default=2)
    testing.set_defaults(run=evaluate)
    args = parser.parse_args(argv)
    torch.set_num_threads(getattr(args, "threads", 2))
    try:
        args.run(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
