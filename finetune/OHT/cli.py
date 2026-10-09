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
        result = audit(args.root, args.output, args.seed, args.fractions, groups=groups)
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
