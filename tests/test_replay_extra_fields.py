import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "bridgevla" / "libs" / "YARR"))

from yarr.replay_buffer.uniform_replay_buffer import (  # noqa: E402
    _copy_required_disk_fields,
    _derive_role_presence,
)


def test_disk_replay_ignores_semantic_audit_only_fields():
    store = {"action": {}}
    transition = {
        "action": np.asarray([1.0], dtype=np.float32),
        "oracle_role_schema_version": np.asarray(
            ["rlbench_o2_semantic_roles_v2"], dtype=object
        ),
    }
    _copy_required_disk_fields(store, transition, 7, 3)
    np.testing.assert_array_equal(store["action"][7], transition["action"])
    assert "oracle_role_schema_version" not in store


def test_disk_replay_still_rejects_missing_required_fields():
    with pytest.raises(KeyError, match="missing required field 'action'"):
        _copy_required_disk_fields({"action": {}}, {}, 0, 5)


@pytest.mark.parametrize('kind, present', [('none', False), ('object', True), ('site', True)])
def test_presence_comes_from_semantic_kind_not_geometry(kind, present):
    transition = {
        'oracle_target_kind': np.asarray(['object'], dtype=object),
        'oracle_reference_kind': np.asarray([kind], dtype=object),
        'oracle_object_valid': np.zeros(32, dtype=bool),
    }
    before = set(transition)
    store = {key: {} for key in (
        'oracle_target_present', 'oracle_reference_present', 'oracle_role_present_known',
    )}
    _copy_required_disk_fields(store, transition, 0, 0)
    assert store['oracle_target_present'][0]
    assert bool(store['oracle_reference_present'][0]) is present
    assert store['oracle_role_present_known'][0].all()
    assert set(transition) == before  # no in-place migration


def test_legacy_and_terminal_placeholder_presence_labels_are_unknown():
    for transition in ({}, {'oracle_target_kind': [], 'oracle_reference_kind': []},
                       {'oracle_target_kind': ['none'], 'oracle_reference_kind': ['none']}):
        labels = _derive_role_presence(transition)
        assert not labels['oracle_role_present_known'].any()
