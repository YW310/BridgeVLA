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


def open_loop(policy, replay, split="val", limit=None):
    data = OHTDataset(replay, split, "baseline")
    if policy.contract["sha256"] != data.contract["sha256"]:
        raise ValueError("Checkpoint/evaluation replay mismatch")
    rows, episode = [], None
    for index in range(len(data) if limit is None else min(len(data), limit)):
        row, meta = data[index], data.samples[index]
        current = (meta["task"], meta["episode_index"])
        if current != episode:
            policy.reset()
            episode = current
        started = time.perf_counter()
        predicted = array(policy.act(row, row["goal"], meta["frame"]), (8,), "prediction")
        latency = time.perf_counter() - started
        target = row["action"]
        rows.append(dict(id=meta["id"], task=meta["task"],
                         translation_m=float(np.linalg.norm(predicted[:3] - target[:3])),
                         rotation_degrees=float(np.rad2deg(rotation_error(predicted[3:7], target[3:7]))),
                         gripper_correct=bool(predicted[7] == target[7]), latency_seconds=latency))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)
    def summarize(values):
        return dict(samples=len(values),
                    translation_m_mean=float(np.mean([v["translation_m"] for v in values])),
                    rotation_degrees_mean=float(np.mean([v["rotation_degrees"] for v in values])),
                    gripper_accuracy=float(np.mean([v["gripper_correct"] for v in values])),
                    latency_seconds_mean=float(np.mean([v["latency_seconds"] for v in values])))
    per_task = {task: summarize(values) for task, values in grouped.items()}
    return dict(kind="open_loop_errors", split=split, contract_sha256=data.contract["sha256"],
                per_task=per_task, all_samples=summarize(rows), predictions=rows,
                note="Pose errors do not measure closed-loop task success.")


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
    p.add_argument("--output", required=True)
    p = sub.add_parser("closed-loop")
    p.add_argument("--server", required=True)
    p.add_argument("--replay", required=True)
    p.add_argument("--environment", required=True, help="module:factory; factory(contract) returns reset/step adapter")
    p.add_argument("--cases", required=True, help="JSONL: id,task,seed; identical list for every configuration")
    p.add_argument("--executor-id", required=True, help="Audited executor implementation/version, shared across policies")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--timeout", type=float, default=30)
    p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if Path(args.output).exists():
        raise FileExistsError(f"Use a new evaluation report: {args.output}")
    if args.kind == "open-loop":
        if args.limit is not None and args.limit < 1:
            raise ValueError("--limit must be positive")
        from .runtime.loading import load_policy
        policy = load_policy(args.checkpoint, args.device, args.pretrain_path,
                             args.predictor, args.predictor_provenance)
        result = open_loop(policy, args.replay, args.split, args.limit)
        result["checkpoint_sha256"] = file_digest(args.checkpoint)
    else:
        if args.max_steps < 1:
            raise ValueError("--max-steps must be positive")
        contract = load_contract(args.replay)
        module, separator, factory = args.environment.partition(":")
        if not separator:
            raise ValueError("--environment must be module:factory")
        environment = getattr(importlib.import_module(module), factory)(contract)
        try:
            result = closed_loop(Client(args.server, contract["sha256"], args.timeout),
                                 environment, read_jsonl(args.cases), args.max_steps)
        finally:
            if hasattr(environment, "close"):
                environment.close()
        result.update(contract_sha256=contract["sha256"], cases_sha256=file_digest(args.cases),
                      executor_id=args.executor_id, environment_factory=args.environment,
                      max_steps=args.max_steps)
    write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("predictions", "cases")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
