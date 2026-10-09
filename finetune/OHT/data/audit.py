"""Full episode audit and reproducible group-preserving split manifests."""
import json
import sys
from time import perf_counter
from collections import defaultdict
from pathlib import Path
from .common import CAMERAS, TASKS, digest, inside, write_json
from .reader import discover, inspect_episode


def split_records(records, seed=0, fractions=(0.8, 0.1, 0.1), groups=None):
    if len(fractions) != 3 or min(fractions) < 0 or abs(sum(fractions) - 1) > 1e-8:
        raise ValueError("Split fractions must be three nonnegative values summing to one")
    groups = groups or {}
    members = defaultdict(list)
    # Union exact-duplicate trajectories and explicit scene groups.
    parent = {}
    def find(key):
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]
    for record in records:
        trajectory = "trajectory:" + record["trajectory_hash"]
        scene = groups.get(f'{record["task"]}/{record["episode_index"]}')
        if scene is not None:
            parent[find(trajectory)] = find("scene:" + str(scene))
    for record in records:
        members[find("trajectory:" + record["trajectory_hash"])].append(record)
    counts = {task: [0, 0, 0] for task in TASKS}
    totals = {task: sum(r["task"] == task for r in records) for task in TASKS}
    output = []
    for key in sorted(members, key=lambda key: digest([seed, key])):
        rows = members[key]
        involved = {row["task"] for row in rows}
        scores = [sum(totals[t] * fractions[s] - counts[t][s] for t in involved)
                  if fractions[s] > 0 else -float("inf") for s in range(3)]
        split = max(range(3), key=lambda s: scores[s])
        for row in rows:
            row = dict(row, split=("train", "val", "test")[split], group=key)
            counts[row["task"]][split] += 1
            output.append(row)
    return sorted(output, key=lambda r: (r["task"], r["episode_index"]))


def audit(root, output, seed=0, fractions=(.8, .1, .1), cameras=CAMERAS, groups=None, *, progress=False):
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    root = Path(root).resolve()
    records = discover(root)
    if not records:
        raise ValueError(f"No OHT Parquet episodes found in {root}")
    started = perf_counter()
    if progress:
        first = records[0]
        print(f"[OHT audit] 0/{len(records)} starting {first['task']}/episode_{first['episode_index']:06d}",
              file=sys.stderr, flush=True)
    valid, errors = [], []
    for index, record in enumerate(records, 1):
        episode_started = perf_counter()
        try:
            info = inspect_episode(root, record, cameras)
            valid.append(dict(record, **info))
            status = f"OK frames={info['frames']}"
        except (ValueError, KeyError, OSError) as exc:
            errors.append(dict(task=record["task"], episode_index=record["episode_index"],
                               error=str(exc)))
            status = "INVALID"
        if progress:
            now = perf_counter()
            print(f"[OHT audit] {index}/{len(records)} {record['task']}/episode_{record['episode_index']:06d} "
                  f"{status} episode_s={now-episode_started:.2f} elapsed_s={now-started:.2f} "
                  f"valid={len(valid)} invalid={len(errors)}", file=sys.stderr, flush=True)
    split = split_records(valid, seed, fractions, groups)
    result = dict(schema="oht_audit_v1", root=str(root), seed=seed,
                  fractions=list(fractions), valid_episodes=len(valid),
                  invalid_episodes=len(errors), errors=errors,
                  missing_tasks=sorted(set(TASKS) - {r["task"] for r in valid}),
                  split_counts={name: sum(r["split"] == name for r in split)
                                for name in ("train", "val", "test")},
                  episodes=split)
    result["manifest_sha256"] = digest(result)
    # Recheck in case another process created the output while auditing.
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite {output}")
    write_json(output, result)
    return result
