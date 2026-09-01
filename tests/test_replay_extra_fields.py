import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "bridgevla" / "libs" / "YARR"))

from yarr.replay_buffer.uniform_replay_buffer import (  # noqa: E402
    _copy_required_disk_fields,
)


def test_disk_replay_ignores_semantic_audit_only_fields():
    store = {"action": {}}
    transition = {
        "action": np.asarray([1.0], dtype=np.float32),
        "oracle_role_schema_version": np.asarray(
            ["rlbench_o2_semantic_roles_v1"], dtype=object
        ),
    }
    _copy_required_disk_fields(store, transition, 7, 3)
    np.testing.assert_array_equal(store["action"][7], transition["action"])
    assert "oracle_role_schema_version" not in store


def test_disk_replay_still_rejects_missing_required_fields():
    with pytest.raises(KeyError, match="missing required field 'action'"):
        _copy_required_disk_fields({"action": {}}, {}, 0, 5)
