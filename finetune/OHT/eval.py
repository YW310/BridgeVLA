"""Held-out open-loop pose errors, or simulator episodes through a client adapter."""
import argparse
import importlib
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from .data.common import read_jsonl, write_json, file_digest
from .data.dataset import OHTDataset
from .data.geometry import array, quaternion, rotation_error
from .data.replay import load_contract
from .runtime.transport import Client
from .runtime.progress import ProgressReporter
from .action_modes import action_options, action_size, resolve_action, apply_eef_delta, action_descriptor


def open_loop(policy, replay, split="val", limit=None, *, progress=None, visualizer=None):
    if getattr(policy, 'chunk_settings', None) is not None:
        from .chunk_eval import open_loop_chunk
        return open_loop_chunk(policy, replay, split, limit, progress=progress, visualizer=visualizer)
    def report(stage, **details):
        if progress is not None:
            progress.update(stage, **details)

    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    report("prepare_replay", replay=replay, split=split)
    options = action_options(getattr(policy, "action_config", None))
    data = OHTDataset(replay, split, "baseline", action_config=options)
    if policy.contract["sha256"] != data.contract["sha256"]:
        raise ValueError("Checkpoint/evaluation replay mismatch")
    total = len(data) if limit is None else min(len(data), limit)
    if not total:
        raise ValueError(f"Evaluation split is empty: {split}")
    report("samples_ready", completed=0, total=total, split=split)
    rows, episode = [], None
    for index in range(total):
        meta = data.samples[index]
        context = dict(total=total, sample=meta["id"], task=meta["task"])
        report("read_sample", emit=index == 0, completed=index, **context)
        started = time.perf_counter()
        row = data[index]
        read_seconds = time.perf_counter() - started
        report("predict", emit=index == 0, completed=index, read_s=round(read_seconds, 3), **context)
        current = (meta["task"], meta["episode_index"])
        if current != episode:
            policy.reset()
            episode = current
        captured = []
        extra = {"visualization_callback": captured.append} if visualizer is not None and visualizer.due(index) else {}
        started = time.perf_counter()
        predicted = array(policy.act(row, row["goal"], meta["frame"], **extra), (action_size(options),), "prediction")
        latency = time.perf_counter() - started
        target = row["action"]
        metrics = dict(gripper_correct=bool(predicted[-1] == target[-1]), latency_seconds=latency)
        if options["mode"] in ("joint", "relative_joint"):
            errors = predicted[:6] - target[:6]
            for joint in options["joint_wrap"]:
                errors[joint] = (errors[joint] + np.pi) % (2 * np.pi) - np.pi
            metrics.update(joint_mae_radians=float(np.abs(errors).mean()),
                           joint_rmse_radians=float(np.sqrt(np.square(errors).mean())))
        else:
            first, second = predicted, target
            if options["mode"] == "relative_eef":
                first = apply_eef_delta(row["tcp_pose"], predicted[:6], options)
                second = apply_eef_delta(row["tcp_pose"], target[:6], options)
            metrics.update(translation_m=float(np.linalg.norm(first[:3] - second[:3])),
                           rotation_degrees=float(np.rad2deg(rotation_error(first[3:7], second[3:7]))))
        rows.append(dict(id=meta["id"], task=meta["task"], **metrics))
        if extra:
            if len(captured) != 1:
                raise RuntimeError("Evaluation visualization requires exactly one payload from this inference")
            report("visualize", emit=index == 0, completed=index, **context)
            visualizer.write(index, captured[0], row, meta, predicted, options, metrics)
        report("sample_done", emit=index == 0 or index + 1 == total, completed=index + 1,
               read_s=round(read_seconds, 3), predict_s=round(latency, 3), **context)
    report("summarize", completed=total, total=total)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    def summarize(values):
        metrics = {key + "_mean": float(np.mean([v[key] for v in values]))
                   for key in values[0] if key not in ("id", "task", "gripper_correct")}
        return dict(samples=len(values), gripper_accuracy=float(np.mean([v["gripper_correct"] for v in values])), **metrics)
    per_task = {task: summarize(values) for task, values in grouped.items()}
    result = dict(kind="open_loop_errors", split=split, contract_sha256=data.contract["sha256"],
                action_spec=action_descriptor(options),
                per_task=per_task, all_samples=summarize(rows), predictions=rows,
                note="Pose errors do not measure closed-loop task success.")
    if visualizer is not None:
        report("write_visualization_index")
        result["visualization"] = visualizer.finish()
    return result


def closed_loop(client, environment, cases, max_steps=500):
    """Adapter owns execution (EEF or stage-wise IK), reset seeds, and scoring."""
    rows = []
    seen = set()
    for case in cases:
        if not isinstance(case.get("id"), str) or case["id"] in seen:
            raise ValueError("Evaluation cases need unique string IDs")
        seen.add(case["id"])
        episode = case["id"]
        start = time.perf_counter()
        steps, success, completed, error = 0, False, False, None
        try:
            current = environment.reset(case)
            for step in range(max_steps):
                action = client.act(current["observation"], current["goal"], episode, step,
                                    current["timestamp"])
                current = environment.step(action)
                steps = step + 1
                if type(current.get("done")) is not bool:
                    raise ValueError("Environment must return an explicit done boolean")
                if current["done"]:
                    if type(current.get("success")) is not bool:
                        raise ValueError("Completed episodes require an explicit success boolean")
                    success, completed = current["success"], True
                    break
        except Exception as exc:
            # Failed/time-out episodes remain in the success denominator.
            error = f"{type(exc).__name__}: {exc}"
        rows.append(dict(id=episode, task=case["task"], seed=case["seed"], steps=steps,
                         completed=completed, success=success, error=error,
                         elapsed_seconds=time.perf_counter() - start))
    if not rows:
        raise ValueError("Evaluation case list is empty")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    return dict(kind="closed_loop_success", cases=rows, attempted=len(rows),
                successes=sum(r["success"] for r in rows),
                success_rate=sum(r["success"] for r in rows) / len(rows),
                per_task={task: dict(attempted=len(values), successes=sum(v["success"] for v in values),
                                     success_rate=sum(v["success"] for v in values) / len(values))
                          for task, values in grouped.items()})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="kind", required=True)
    p = sub.add_parser("open-loop")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--replay", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--limit", type=int)
    p.add_argument("--pretrain-path")
    p.add_argument("--predictor")
    p.add_argument("--predictor-provenance")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--progress-interval", type=float, default=10,
                   help="Seconds between stderr status updates, including during loading/first inference")
    p.add_argument("--quiet", action="store_true", help="Disable OHT evaluation progress messages")
    p.add_argument("--visualize-every", type=int, default=50,
                   help="Capture sample indices 0,N,2N,... as PNG/NPZ (default: 50); 0 disables visualization")
    p.add_argument("--visualize-output-dir", help="New image directory; default: <output-stem>_visualizations")
    p.add_argument("--output", required=True)
    p = sub.add_parser("closed-loop")
    p.add_argument("--server", required=True)
    p.add_argument("--replay", required=True)
    p.add_argument("--environment", required=True, help="module:factory; factory(contract) returns reset/step adapter")
    p.add_argument("--cases", required=True, help="JSONL: id,task,seed; identical list for every configuration")
    p.add_argument("--executor-id", required=True, help="Audited executor implementation/version, shared across policies")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--config", help="Training YAML selecting action_mode/frames; absent means legacy EEF")
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        raise FileExistsError(f"Use a new evaluation report: {args.output}")
    if args.kind == "open-loop":
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive")
        if args.visualize_every < 0:
            raise ValueError("--visualize-every must be nonnegative")
        if args.visualize_output_dir and not args.visualize_every:
            raise ValueError("--visualize-output-dir requires --visualize-every > 0")
        visualizer = None
        if args.visualize_every:
            from .evaluation_visualization import EvaluationVisualizer
            directory = args.visualize_output_dir or str(Path(args.output).with_suffix('')) + '_visualizations'
            visualizer = EvaluationVisualizer(directory, args.visualize_every)
        from .runtime.loading import load_policy
        with ProgressReporter(args.progress_interval, enabled=not args.quiet) as progress:
            policy = load_policy(args.checkpoint, args.device, args.pretrain_path,
                                 args.predictor, args.predictor_provenance, progress=progress)
            result = open_loop(policy, args.replay, args.split, args.limit, progress=progress, visualizer=visualizer)
            progress.update("hash_checkpoint", checkpoint=args.checkpoint)
            result["checkpoint_sha256"] = file_digest(args.checkpoint)
            progress.update("write_report", output=args.output)
            write_json(args.output, result)
            progress.update("complete", output=args.output)
    else:
        if args.max_steps < 1:
            raise ValueError("--max-steps must be positive")
        contract = load_contract(args.replay)
        module, separator, factory = args.environment.partition(":")
        if not separator:
            raise ValueError("--environment must be module:factory")
        from .config import load
        options = resolve_action(load(args.config)) if args.config else action_options()
        runtime_contract = dict(contract, action_spec=action_descriptor(options))
        environment = getattr(importlib.import_module(module), factory)(runtime_contract)
        try:
            result = closed_loop(Client(args.server, contract["sha256"], args.timeout, options),
                                 environment, read_jsonl(args.cases), args.max_steps)
        finally:
            if hasattr(environment, "close"):
                environment.close()
        result.update(contract_sha256=contract["sha256"], cases_sha256=file_digest(args.cases),
                      executor_id=args.executor_id, environment_factory=args.environment,
                      action_spec=action_descriptor(options),
                      max_steps=args.max_steps)
        write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("predictions", "cases")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
