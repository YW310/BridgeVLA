"""Read-only GT gate: paired hierarchical bootstrap over seeds and episodes."""

import argparse
import json
import math
from pathlib import Path
import random
import statistics


def load_episodes(root):
    records = {}
    environment = None
    for path in sorted(Path(root).rglob('episode_*.json')):
        item = json.loads(path.read_text(encoding='utf-8'))
        if item.get('schema_version') != 'rlbench_eval_episode_v1':
            raise ValueError(f'Not a policy episode journal: {path}')
        key = (item.get('task'), item.get('episode_idx'))
        reward = item.get('reward')
        if (not isinstance(key[0], str) or not key[0]
                or type(key[1]) is not int or key[1] < 0
                or type(reward) not in (int, float) or not math.isfinite(reward)
                or reward not in (0, 100)):
            raise ValueError(f'Invalid sparse-reward policy episode: {path}')
        if key in records:
            raise ValueError(f'Duplicate task/episode: {key} under {root}')
        signature = item.get('run_signature', {})
        current = (signature.get('eval_datafolder'), signature.get('episode_length'),
                   signature.get('use_input_place_with_mean'))
        if current[0] is None or type(current[1]) is not int or current[1] <= 0:
            raise ValueError(f'Missing evaluation environment signature: {path}')
        if environment is not None and current != environment:
            raise ValueError(f'Mixed evaluation environments under {root}')
        environment = current
        records[key] = float(reward == 100)
    if not records:
        raise ValueError(f'No episode journals under {root}')
    return records, environment


def _quantile(values, probability):
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def compare(baseline, candidate, resamples=10000, bootstrap_seed=0):
    if set(baseline) != set(candidate) or len(baseline) < 3:
        raise ValueError('Matched baseline/candidate roots for at least three seeds are required')
    if resamples < 100:
        raise ValueError('Use at least 100 bootstrap resamples')
    pairs, environments, expected_keys = {}, set(), None
    base_rates, candidate_rates = [], []
    for seed in sorted(baseline):
        base, base_env = load_episodes(baseline[seed])
        new, new_env = load_episodes(candidate[seed])
        if set(base) != set(new) or (expected_keys is not None and set(base) != expected_keys):
            raise ValueError(f'Task/episode sets differ for seed {seed}; do not drop failures')
        expected_keys = set(base)
        environments.update((base_env, new_env))
        tasks = sorted({key[0] for key in base})
        pairs[seed] = [[new[key] - base[key] for key in sorted(base) if key[0] == task]
                       for task in tasks]
        base_rates.append(statistics.mean(statistics.mean(base[k] for k in base if k[0] == t)
                                          for t in tasks))
        candidate_rates.append(statistics.mean(statistics.mean(new[k] for k in new if k[0] == t)
                                               for t in tasks))
    if len(environments) != 1:
        raise ValueError('Evaluation dataset or episode length differs between runs')
    rng = random.Random(bootstrap_seed)
    seeds = sorted(pairs)
    samples = []
    for _ in range(resamples):
        seed_rates = []
        for seed in rng.choices(seeds, k=len(seeds)):
            seed_rates.append(statistics.mean(
                statistics.mean(rng.choices(task, k=len(task))) for task in pairs[seed]
            ))
        samples.append(statistics.mean(seed_rates))
    samples.sort()
    interval = [_quantile(samples, .025), _quantile(samples, .975)]
    return {
        'method': 'paired_seed_episode_hierarchical_bootstrap_task_macro',
        'seeds': len(seeds), 'episodes_per_seed': len(expected_keys),
        'baseline_success': statistics.mean(base_rates),
        'candidate_success': statistics.mean(candidate_rates),
        'success_difference': statistics.mean(candidate_rates) - statistics.mean(base_rates),
        'ci95': interval, 'gt_gate_passed': interval[0] > 0,
        'resamples': resamples, 'bootstrap_seed': bootstrap_seed,
    }


def _roots(values):
    roots = {}
    for value in values:
        seed, separator, path = value.partition('=')
        if not separator or not seed or not path or seed in roots:
            raise ValueError('Each root must be a unique SEED=EPISODE_RESULTS_DIR')
        roots[seed] = path
    return roots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', action='append', required=True, metavar='SEED=DIR')
    parser.add_argument('--candidate', action='append', required=True, metavar='SEED=DIR')
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--bootstrap-seed', type=int, default=0)
    args = parser.parse_args()
    try:
        result = compare(_roots(args.baseline), _roots(args.candidate),
                         args.resamples, args.bootstrap_seed)
    except (ValueError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
