"""Data commands run on CPU and do not import a simulator or model."""
import argparse
import json
import sys
from time import perf_counter
from pathlib import Path
import numpy as np
from .data.common import CAMERAS, read_config, read_jsonl, inside, file_digest, write_json
from .data.audit import audit
from .data.replay import build, load_contract
from .data.dataset import OHTDataset
from .data.role_cache import create_cache
from .data.role_teacher import build_teacher
from .data.visualization import CAMERA_COLORS, save_preview
from .runtime.predicted_wrapper import load_predictor, PredictedObjectWrapper


def _visualization_arguments(parser):
    parser.add_argument("--visualize-every", type=int, default=0,
                        help="Save every N emitted samples per episode, including the first; 0 disables")
    parser.add_argument("--visualize-output-dir",
                        help="Preview directory (default: OUTPUT/preview); PNG and build HTML share it")


def _validation_progress(split):
    started = last = perf_counter()

    def report(completed, total):
        nonlocal last
        now = perf_counter()
        if completed not in (0, total) and now - last < 5:
            return
        elapsed = now - started
        rate = completed / elapsed if elapsed > 0 else 0.
        eta = f"{(total - completed) / rate:.1f}" if rate > 0 else "unknown"
        print(f"[OHT validate] split={split} {completed}/{total} "
              f"elapsed_s={elapsed:.1f} samples_per_s={rate:.2f} eta_s={eta}",
              file=sys.stderr, flush=True)
        last = now

    return report


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
    p.add_argument("--object-cloud-config", help="Optional build-time object-cloud YAML (source/default_size/objects)")
    _visualization_arguments(p)
    p = sub.add_parser("validate")
    p.add_argument("--replay", required=True)
    p.add_argument("--mode", choices=("baseline", "role_queries", "predicted_external"), default="baseline")
    p.add_argument("--role-cache")
    p.add_argument("--point-count", type=int, default=512)
    p.add_argument("--workers", type=int, default=1,
                   help="Concurrent sample checks (default: 1); every sample still receives full validation")
    p.add_argument("--quiet", action="store_true", help="Hide validation progress on stderr; keep the JSON summary")
    p = sub.add_parser("diagnose-geometry", help="Export one cached frame per camera and camera-colored fusion; no cache edits")
    p.add_argument("--replay", required=True)
    selection = p.add_mutually_exclusive_group(required=True)
    selection.add_argument("--sample-id", help="task/episode/frame, as recorded in samples.jsonl")
    selection.add_argument("--observation", help="Direct NPZ path within replay; requires --allow-incomplete")
    p.add_argument("--allow-incomplete", action="store_true",
                   help="Diagnostic only: do not require/read complete.json; never enables incomplete training")
    p.add_argument("--data-profile", help="Direct NPZ source key from contract.json; required with multiple profiles")
    p.add_argument("--output", required=True, help="New diagnostic directory; never overwrites existing files")
    p.add_argument("--html", action="store_true", help="Also export an offline interactive world-XYZ ROI editor")
    p.add_argument("--max-points", type=int, default=40000, help="HTML display point budget, 1-60000; never changes cached XYZ")
    p.add_argument("--html-source", choices=("xyz", "depth"), default="xyz",
                   help="xyz: remaining cached points; depth: display-only backprojection before old ROI masking")
    p.add_argument("--object-root", help="Original raw dataset root; recover current object poses for an indexed old replay")
    p.add_argument("--object-manifest", help="Original audit manifest bound to replay; requires --object-root and --html")
    p = sub.add_parser("extract-object-clouds", help="Export approximate per-object box crops as full cached-resolution PLY/NPZ")
    p.add_argument("--replay", required=True)
    p.add_argument("--sample-id", required=True)
    p.add_argument("--boxes", required=True, help="oht_object_boxes_v1 JSON exported by the HTML viewer")
    p.add_argument("--output", required=True, help="New object cloud directory")
    p.add_argument("--allow-incomplete", action="store_true", help="Diagnostic-only incomplete replay inspection")
    p.add_argument("--object-root", help="Original raw root for an old replay without object poses")
    p.add_argument("--object-manifest", help="Original bound audit; requires --object-root")
    p = sub.add_parser("export-viewer", help="Export grouped PNG/HTML previews and an offline frame viewer from a complete replay")
    p.add_argument("--replay", required=True)
    p.add_argument("--output", required=True, help="New HTML directory; no replay edits")
    p.add_argument("--every", type=int, default=1, help="Every N emitted samples per episode, including the first")
    p.add_argument("--max-points", type=int, default=40000)
    p.add_argument("--source", choices=("xyz","depth"), help="Default: object's configured source, otherwise depth")
    p.add_argument("--object-cloud-config", help="Optional object-cloud YAML; default: replay diagnostic_object_clouds")
    p.add_argument("--object-root", help="Original raw root to recover legacy objects")
    p.add_argument("--object-manifest", help="Original bound audit; requires --object-root")
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
                      visualize_every=args.visualize_every, visualize_output_dir=args.visualize_output_dir,
                      object_cloud_config=read_config(args.object_cloud_config) if args.object_cloud_config else None)
        print(f"Built {count} OHT transitions")
    elif args.command == "validate":
        if args.workers < 1:
            parser.error("--workers must be a positive integer")
        if not args.quiet:
            print("[OHT validate] loading replay contract and sample index", file=sys.stderr, flush=True)
        contract = load_contract(args.replay)
        counts = {}
        available = {row["split"] for row in read_jsonl(Path(args.replay) / "samples.jsonl")}
        for split in sorted(available):
            data = OHTDataset(args.replay, split, args.mode, args.role_cache, args.point_count)
            counts[split] = data.validate_all(
                workers=args.workers, progress=None if args.quiet else _validation_progress(split))
        print(json.dumps(dict(valid=True, samples=counts, contract=contract["sha256"])))
    elif args.command == "diagnose-geometry":
        if bool(args.object_root) != bool(args.object_manifest) or (args.object_root and not args.html):
            raise ValueError("Use --object-root and --object-manifest together with --html")
        if args.html:
            from .data.html_preview import html_point_limit, save_point_cloud_html
            html_point_limit(args.max_points)
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(f"Use a new diagnostic directory: {output}")
        from .data.geometry_diagnostic import load_geometry_diagnostic
        contract, config, row, observation, validation = load_geometry_diagnostic(
            args.replay, sample_id=args.sample_id, observation_path=args.observation,
            data_profile=args.data_profile, allow_incomplete=args.allow_incomplete)
        if args.object_root:
            from .data.object_preview import load_source_objects
            row = dict(row, diagnostic_objects=load_source_objects(
                args.object_root, args.object_manifest, contract, row))
        row = dict(row, diagnostic_contract_sha256=contract["sha256"])
        # Validate the HTML payload before creating any output on invalid data.
        if args.html:
            from .data.html_preview import point_cloud_payload
            point_cloud_payload(observation, config, row, args.max_points, source=args.html_source)
        output.mkdir(parents=True)
        if args.html:
            save_point_cloud_html(output / "point-cloud-roi.html", observation, config, row, args.max_points,
                                  source=args.html_source)
        save_preview(output / "fused_rgb.png", observation, config, row, current_tcp=row.get("current_tcp"))
        save_preview(output / "fused_camera_colors.png", observation, config, row,
                     current_tcp=row.get("current_tcp"), color_by_camera=True)
        for camera in config["cameras"]:
            selected = dict(config, cameras={camera: config["cameras"][camera]})
            save_preview(inside(output, camera + ".png"), observation, selected, row,
                         current_tcp=row.get("current_tcp"), color_by_camera=True)
        write_json(output / "geometry.json", dict(
            sample_id=row["id"], contract_sha256=contract["sha256"], data_config=config,
            validation=validation,
            diagnostic_objects=row.get("diagnostic_objects"),
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
    elif args.command == "extract-object-clouds":
        if bool(args.object_root) != bool(args.object_manifest):
            raise ValueError("Use --object-root and --object-manifest together")
        from .data.geometry_diagnostic import load_geometry_diagnostic
        from .data.object_cloud import export_object_clouds
        output = Path(args.output)
        if output.exists():
            raise FileExistsError(f"Use a new object cloud directory: {output}")
        specification = json.loads(Path(args.boxes).read_text(encoding="utf-8"))
        contract, config, row, observation, validation = load_geometry_diagnostic(
            args.replay, sample_id=args.sample_id, allow_incomplete=args.allow_incomplete)
        if args.object_root:
            from .data.object_preview import load_source_objects
            row = dict(row, diagnostic_objects=load_source_objects(
                args.object_root, args.object_manifest, contract, row))
        report = export_object_clouds(output, observation, config, row, specification, contract["sha256"], validation)
        print(f"Saved {len(report['objects'])} approximate object clouds to {output}")
    elif args.command == "export-viewer":
        from .data.replay_viewer import export_replay_viewer
        count = export_replay_viewer(args.replay,args.output,every=args.every,max_points=args.max_points,source=args.source,
                                    object_cloud_config=read_config(args.object_cloud_config) if args.object_cloud_config else None,
                                    object_root=args.object_root,object_manifest=args.object_manifest)
        print(f"Saved {count} HTML frames; open {Path(args.output)/'index.html'}")
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
