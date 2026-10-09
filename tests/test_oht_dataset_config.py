"""Default OHT workspace covers the v423 description, not measured full data."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from finetune.OHT.data.geometry import check_bounds


CONFIG = Path(__file__).resolve().parents[1] / "finetune/OHT/configs/dataset.yaml"


@pytest.fixture
def bounds():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["scene_bounds"]


def test_default_workspace_contains_documented_v423_examples(bounds):
    # Data description sections 8.4 (objects) and 10.2 (waypoints).
    points = np.array([
        [.283, .480, .681], [.780, .183, 1.388],
        [.374, -.548, .936], [.779, -.182, 1.388],
        [.466, .334, 1.540], [.374, -.548, 1.039],
        [.374, -.548, 1.005], [.712, -.183, 1.388],
        [.507, -.183, 1.388],
    ])
    check_bounds(points, bounds)
    assert np.all(points > np.asarray(bounds[:3]))
    assert np.all(points < np.asarray(bounds[3:]))


def test_default_workspace_keeps_out_of_range_guard(bounds):
    with pytest.raises(ValueError, match="outside scene_bounds"):
        check_bounds(bounds[3:], bounds)
