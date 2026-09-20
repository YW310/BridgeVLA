'''Fail-closed semantic-GT train/eval contract.'''

import hashlib
from pathlib import Path
from typing import Mapping


SEMANTIC_SCHEMA = 'rlbench_o2_semantic_roles_v2'
REFERENCE_GEOMETRY = 'role_typed_point_set_v2'
MANIFEST_HANDLE_NAMESPACE = 'stored'
CONTRACT_VERSION = 2


def resolve_role_config(path):
    value = Path(path).expanduser()
    candidates = (value, Path(__file__).resolve().parents[1] / value)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f'Semantic role config does not exist: {path}')


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def build_semantic_contract(role_config, phase_source, num_points):
    if phase_source not in ('sim_replay', 'demo_events'):
        raise ValueError(f'Unsupported semantic phase source: {phase_source!r}')
    if int(num_points) <= 0:
        raise ValueError('Semantic contract num_points must be positive')
    return {
        'contract_version': CONTRACT_VERSION,
        'schema_version': SEMANTIC_SCHEMA,
        'phase_source': phase_source,
        'role_config_sha256': file_sha256(resolve_role_config(role_config)),
        'num_points': int(num_points),
        'reference_geometry': REFERENCE_GEOMETRY,
        'manifest_handle_namespace': MANIFEST_HANDLE_NAMESPACE,
    }


def validate_semantic_contract(stored, expected, source='checkpoint'):
    if not isinstance(stored, Mapping):
        raise RuntimeError(
            f'{source} has no verified semantic_contract; regenerate the '
            'sim_replay buffer and retrain.')
    differences = {
        key: (stored.get(key), expected.get(key))
        for key in expected if stored.get(key) != expected.get(key)
    }
    if differences:
        details = ', '.join(
            f'{key}={value[0]!r} (expected {value[1]!r})'
            for key, value in differences.items())
        raise RuntimeError(
            f'Semantic train/eval contract mismatch in {source}: {details}')


def validate_semantic_validation_report(report, expected, source):
    if not isinstance(report, Mapping):
        raise RuntimeError(f'Invalid semantic validation report: {source}')
    if not bool(report.get('valid', False)):
        raise RuntimeError(f'Semantic replay validation failed: {source}')
    if not bool(report.get('validation_complete', False)):
        raise RuntimeError(
            f'Semantic replay validation is sampled, not full: {source}')
    if int(report.get('raw_fallback_files', -1)) != 0:
        raise RuntimeError(
            f'Semantic replay contains raw fallback files: {source}')
    expected_phase = expected['phase_source']
    phase_sources = report.get('phase_sources', {})
    if (
        set(phase_sources) != {expected_phase}
        or int(phase_sources.get(expected_phase, 0)) <= 0
    ):
        raise RuntimeError(
            f'Semantic replay phase sources mismatch in {source}: '
            f'{phase_sources!r}')
    report_contract = {
        'schema_version': report.get('schema_version'),
        'role_config_sha256': report.get('role_config_sha256'),
        'num_points': report.get('num_points'),
        'manifest_handle_namespace': report.get('manifest_handle_namespace'),
    }
    expected_contract = {
        key: expected[key] for key in report_contract
    }
    validate_semantic_contract(
        report_contract, expected_contract, source=source)
