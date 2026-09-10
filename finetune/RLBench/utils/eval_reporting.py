"""Reporting and resume helpers without simulator/model dependencies."""

import hashlib
import json
import math
import os
from numbers import Real
from pathlib import Path


MANIFEST_FIELDS = [
    'task', 'generated coverage', 'generated episodes',
    'requested episodes', 'logical transitions',
]

EVAL_FIELDS = ['task', 'success rate', 'length', 'total_transitions']


def manifest_result(task, generated, requested, logical_transitions):
    if requested <= 0 or not 0 <= generated <= requested:
        raise ValueError('Manifest reporting requires 0 <= generated <= requested and requested > 0')
    return dict(zip(MANIFEST_FIELDS, (
        task, 100.0 * generated / requested, generated, requested, logical_transitions)))


def numeric_task_scores(scores):
    # Missing metrics are not zero success and must not reach add_scalar.
    return {task: float(value) for task, value in scores.items()
            if isinstance(value, Real) and not isinstance(value, bool)
            and math.isfinite(float(value))}


def evaluation_result(task, rewards, lengths):
    """Aggregate a complete set of episode results using YARR's metric units."""
    if not rewards or len(rewards) != len(lengths):
        raise ValueError('Evaluation reporting requires one length per reward')
    if not all(isinstance(value, Real) and not isinstance(value, bool)
               and math.isfinite(float(value)) for value in rewards):
        raise ValueError('Evaluation rewards must be finite numbers')
    if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0
               for value in lengths):
        raise ValueError('Evaluation lengths must be positive integers')
    return dict(zip(EVAL_FIELDS, (
        task,
        sum(float(value) for value in rewards) / len(rewards),
        sum(lengths) / len(lengths),
        sum(lengths),
    )))


def _file_identity(path, hash_contents=False):
    if path is None:
        return None
    path = Path(path).resolve()
    identity = {'path': str(path)}
    try:
        stat = path.stat()
    except OSError:
        identity['missing'] = True
        return identity
    identity.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if hash_contents:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        identity['sha256'] = digest.hexdigest()
    return identity


def build_eval_run_signature(
        model_path, exp_cfg_path, mvt_cfg_path, eval_datafolder, *,
        episode_length, oracle_provider, oracle_role_config,
        oracle_num_points, oracle_strict, oracle_handle_alignment,
        use_input_place_with_mean, runtime_files=()):
    """Build a stable identity for results that may be reused across restarts."""
    signature = {
        'schema_version': 'rlbench_eval_run_v1',
        # Checkpoints can be several GB; size+mtime avoids rehashing at startup.
        'checkpoint': _file_identity(model_path, hash_contents=False),
        'experiment_config': _file_identity(exp_cfg_path, hash_contents=True),
        'mvt_config': _file_identity(mvt_cfg_path, hash_contents=True),
        'eval_datafolder': str(Path(eval_datafolder).resolve()),
        'episode_length': int(episode_length),
        'oracle_provider': oracle_provider,
        'oracle_role_config': (
            _file_identity(oracle_role_config, hash_contents=True)
            if oracle_provider == 'rlbench_gt' else None),
        'oracle_num_points': (
            int(oracle_num_points) if oracle_provider == 'rlbench_gt' else None),
        'oracle_strict': (
            bool(oracle_strict) if oracle_provider == 'rlbench_gt' else None),
        'oracle_handle_alignment': (
            oracle_handle_alignment if oracle_provider == 'rlbench_gt' else None),
        'use_input_place_with_mean': bool(use_input_place_with_mean),
        'runtime_files': [
            _file_identity(path, hash_contents=True) for path in runtime_files
        ],
    }
    encoded = json.dumps(
        signature, sort_keys=True, separators=(',', ':'), ensure_ascii=True
    ).encode('utf-8')
    return signature, hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path, value):
    """Replace a JSON file only after a complete, durable temporary write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def quarantine_file(path, directory):
    """Move an incompatible cache entry aside without overwriting prior evidence."""
    path, directory = Path(path), Path(directory)
    if not path.is_file():
        return None
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / path.name
    suffix = 1
    while target.exists():
        target = directory / f'{path.stem}.{suffix}{path.suffix}'
        suffix += 1
    os.replace(str(path), str(target))
    return target


def resumable_eval_episode(path, task, episode_idx, run_signature_sha256):
    """Return a verified standard-eval episode journal, else a rejection reason."""
    path = Path(path)
    if not path.is_file():
        return None, 'missing'
    try:
        result = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f'unreadable: {exc}'
    if result.get('schema_version') != 'rlbench_eval_episode_v1':
        return None, 'schema mismatch'
    if result.get('task') != task or result.get('episode_idx') != episode_idx:
        return None, 'task/episode mismatch'
    if result.get('run_signature_sha256') != run_signature_sha256:
        return None, 'run signature mismatch'
    reward = result.get('reward')
    length = result.get('length')
    attempts = result.get('attempts_used')
    if (not isinstance(reward, Real) or isinstance(reward, bool)
            or not math.isfinite(float(reward))):
        return None, 'reward is invalid'
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        return None, 'length is invalid'
    if not isinstance(attempts, int) or isinstance(attempts, bool) or attempts <= 0:
        return None, 'attempt count is invalid'
    return {
        'reward': float(reward), 'length': length, 'attempts_used': attempts,
    }, None


def resumable_manifest(path, task, episode_idx, alignment_mode,
                       role_config_sha256=None, resolver_version=None):
    """Return resume metadata for a complete manifest, else a rejection reason."""
    path = Path(path)
    if not path.is_file():
        return None, 'missing'
    try:
        manifest = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f'unreadable: {exc}'
    if manifest.get('schema_version') != 'rlbench_o2_semantic_roles_v1':
        return None, 'schema mismatch'
    if manifest.get('task') != task or manifest.get('episode_idx') != episode_idx:
        return None, 'task/episode mismatch'
    if manifest.get('phase_source') != 'demo_events':
        return None, 'phase_source is not demo_events'
    if (resolver_version is not None
            and manifest.get('resolver_version') != resolver_version):
        return None, 'task resolver version mismatch'
    if not manifest.get('source_alignment_validated'):
        return None, 'source alignment was not validated'
    if manifest.get('handle_namespace') != 'stored':
        return None, 'handle namespace is not stored'
    alignment = manifest.get('handle_alignment', {})
    if alignment.get('status') != alignment_mode:
        return None, 'handle alignment mode mismatch'
    if not manifest.get('source_frame0_masks'):
        return None, 'source mask fingerprints are missing'
    entries = manifest.get('entries')
    if not isinstance(entries, list) or not entries:
        return None, 'entries are missing'
    try:
        frames = [int(entry['sample_frame']) for entry in entries]
        expected = [int(frame) for frame in manifest['expected_sample_frames']]
    except (KeyError, TypeError, ValueError):
        return None, 'sample-frame metadata is invalid'
    if frames != sorted(set(frames)) or not expected or not set(expected) <= set(frames):
        return None, 'sample-frame coverage is incomplete'
    if not entries[-1].get('completion_satisfied'):
        return None, 'final completion is not satisfied'
    mapped = set(alignment.get('live_to_stored', {}).values())
    for entry in entries:
        for key in ('target', 'reference'):
            role = entry.get(key)
            if role is None:
                continue
            if role.get('kind') == 'object':
                handles = role.get('handles')
                if not handles or not set(handles) <= mapped:
                    return None, f'{key} object handles are not verified'
            elif role.get('kind') == 'site':
                position = role.get('site_position')
                if (not isinstance(position, list) or len(position) != 3
                        or not all(isinstance(v, Real) and math.isfinite(v)
                                   for v in position)):
                    return None, f'{key} site position is invalid'
            else:
                return None, f'{key} kind is invalid'
    saved_digest = manifest.get('role_config_sha256')
    if saved_digest and role_config_sha256 and saved_digest != role_config_sha256:
        return None, 'role config digest mismatch'
    return {
        'logical_transitions': len(entries),
        'legacy_config_digest': not bool(saved_digest),
    }, None
