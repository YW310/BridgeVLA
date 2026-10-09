"""Data commands run on CPU and do not import a simulator or model."""
import argparse
import json
from pathlib import Path
import numpy as np
from .data.common import CAMERAS, read_config, read_jsonl, inside, file_digest, write_json
from .data.audit import audit
from .data.replay import build, load_contract
from .data.dataset import OHTDataset
from .data.role_cache import create_cache
from .data.role_teacher import build_teacher
from .data.source_config import sample_data_config
from .data.visualization import CAMERA_COLORS, save_preview
from .runtime.predicted_wrapper import load_predictor, PredictedObjectWrapper


def _visualization_arguments(parser):
    parser.add_argument("--visualize-every", type=int, default=0,
                        help="Save every N emitted samples per episode, including the first; 0 disables")
    parser.add_argument("--visualize-output-dir",
                        help="PNG directory (default: OUTPUT/visualizations); existing files are not overwritten")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("audit")
    p.add_argument("--root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fractions", nargs=3, type=float, default=[.8, .1, .1])
    p.add_argument("--groups", help="JSON mapping task/episode to a shared scene group")
    p.add_argument("--quiet", action="store_true", help="Hide episode progress on stderr; keep the JSON summary")
    p = sub.add_parser("build")
    for key in ("root", "manifest", "config", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--sample-stride", type=int, default=10)
    _visualization_arguments(p)
    p = sub.add_parser("validate")
    p.add_argument("--replay", required=True)
    p.add_argument("--mode", choices=("baseline", "role_queries", "predicted_external"), default="baseline")
    p.add_argument("--role-cache")
    p.add_argument("--point-count", type=int, default=512)
    p = sub.add_parser("diagnose-geometry", help="Export one cached frame per camera and camera-colored fusion; no cache edits")
    p.add_argument("--replay", required=True)
    p.add_argument("--sample-id", required=True, help="task/episode/frame, as recorded in samples.jsonl")
    p.add_argument("--output", required=True, help="New diagnostic directory; never overwrites existing files")
    p = sub.add_parser("teacher")
    for key in ("replay", "annotations", "output"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--point-count", type=int, default=512)
    _visualization_arguments(p)
    p = sub.add_parser("predict")
    for key in ("replay", "predictor", "output", "provenance"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--point-count", type=int, default=512)
    args = parser.parse_args(argv)
    if args.command == "audit":
        groups = json.loads(Path(args.groups).read_text(encoding="utf-8")) if args.groups else None
        result = audit(args.root, args.output, args.seed, args.fractions, groups=groups, progress=not args.quiet)
        print(json.dumps({k: result[k] for k in ("valid_episodes", "invalid_episodes", "split_counts", "missing_tasks")}))
        return 1 if result["invalid_episodes"] else 0
    if args.command == "build":
        count = build(args.root, args.manifest, read_config(args.config), args.output, args.sample_stride,
                      visualize_every=args.visualize_every, visualize_output_dir=args.visualize_output_dir)
        print(f"Built {count} OHT transitions")
    elif args.command == "validate":
        contract = load_contract(args.replay)
        counts = {}
        available = {row["split"] for row in read_jsonl(Path(args.replay) / "samples.jsonl")}
        for split in sorted(available):
            data = OHTDataset(args.replay, split, args.mode, args.role_cache, args.point_count)
            counts[split] = data.validate_all()
        print(json.dumps(dict(valid=True, samples=counts, contract=contract["sha256"])))
    elif args.command == "diagnose-geometry":
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(f"Use a new diagnostic directory: {output}")
        contract = load_contract(args.replay)
        with (Path(args.replay) / "samples.jsonl").open(encoding="utf-8") as stream:
            rows = (json.loads(line) for line in stream if line.strip())
            row = next((row for row in rows if row["id"] == args.sample_id), None)
        if row is None:
            raise ValueError(f"Unknown sample ID: {args.sample_id}")
        path = inside(args.replay, row["observation"])
        if file_digest(path) != row["observation_sha256"]:
            raise ValueError("Observation changed")
        with np.load(path, allow_pickle=False) as source:
            observation = {key: source[key] for key in source.files}
        config = sample_data_config(contract, row)
        output.mkdir(parents=True)
        save_preview(output / "fused_rgb.png", observation, config, row, current_tcp=row.get("current_tcp"))
        save_preview(output / "fused_camera_colors.png", observation, config, row,
                     current_tcp=row.get("current_tcp"), color_by_camera=True)
        for camera in config["cameras"]:
            selected = dict(config, cameras={camera: config["cameras"][camera]})
            save_preview(inside(output, camera + ".png"), observation, selected, row,
                         current_tcp=row.get("current_tcp"), color_by_camera=True)
        write_json(output / "geometry.json", dict(
            sample_id=row["id"], contract_sha256=contract["sha256"], data_config=config,
            video_alignment=config.get("video_alignment", "timestamp"),
            camera_extrinsic_direction=config.get("camera_extrinsic_direction", "camera_to_world"),
            note="Cached geometry only; colors identify cameras, not roles. No registration or auto-correction.",
            cameras={camera: dict(
                color=list(CAMERA_COLORS[camera]),
                world_from_optical=observation[f"{camera}_camera_extrinsics"].tolist(),
                intrinsics=observation[f"{camera}_camera_intrinsics"].tolist(),
                finite_points=int(np.isfinite(observation[f"{camera}_point_cloud"]).all(axis=0).sum()))
                for camera in config["cameras"]}))
        print(f"Saved geometry diagnostics to {output}")
    elif args.command == "teacher":
        result = build_teacher(args.replay, args.annotations, args.output, args.point_count,
                               visualize_every=args.visualize_every, visualize_output_dir=args.visualize_output_dir)
        print(f"Built {len(result['samples'])} teacher records")
    elif args.command == "predict":
        contract = load_contract(args.replay)
        wrapper = PredictedObjectWrapper(load_predictor(args.predictor),
                                          contract["data_config"]["cameras"], args.point_count)
        provenance = read_config(args.provenance)
        if not provenance.get("model_sha256") or not provenance.get("training_split"):
            raise ValueError("Prediction provenance requires model_sha256 and training_split")
        provenance["factory"] = args.predictor
        def predictions():
            episode = None
            for row in read_jsonl(Path(args.replay) / "samples.jsonl"):
                current = (row["task"], row["episode_index"])
                if current != episode:
                    wrapper.reset()
                    episode = current
                path = inside(args.replay, row["observation"])
                if file_digest(path) != row["observation_sha256"]:
                    raise ValueError("Observation changed")
                with np.load(path, allow_pickle=False) as source:
                    observation = {k: source[k] for k in source.files}
                yield row["id"], wrapper.predict(observation, row["goal"])
        result = create_cache(args.output, "predicted", contract["sha256"], args.point_count,
                              provenance, predictions())
        print(f"Built {len(result['samples'])} predicted records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
