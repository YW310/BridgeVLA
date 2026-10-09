"""Reference decoder compatibility and a depth-induced XY fusion regression."""
import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from finetune.OHT.data.geometry import backproject
from finetune.OHT.data.observation import validate_data_config
from finetune.OHT.data.video import decode_depth


CONFIG = Path(__file__).resolve().parents[1] / "finetune/OHT/configs/dataset.yaml"


@pytest.mark.parametrize("use_log,expected_sha256", [
    (True, "86e6654e050158c95262a2a23b97a4589e2ec5759bdc9aae120f09e143998396"),
    (False, "0bad74ad6d8e3dfdf1ba491a82811a4da52b05650137008f57912535b7442958"),
])
def test_all_codes_match_reference_uint16_mm_png(use_log, expected_sha256):
    # Golden digests from the supplied converter's _dequantize_depth_mm(),
    # evaluated on row-major codes 0..4095 with .01/10/3.5/4095 parameters.
    spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["depth"]
    spec["use_log"] = use_log
    raw = np.arange(4096, dtype=np.uint16).reshape(64, 64)
    depth = decode_depth(raw, spec)
    assert np.isnan(depth[0, 0])
    assert np.isfinite(depth.ravel()[1:]).all()
    assert depth[-1, -1] == 10.  # qmax is decoded; range filtering is separate.
    mm = np.rint(np.nan_to_num(depth) * 1000).astype("<u2")
    assert hashlib.sha256(mm.tobytes()).hexdigest() == expected_sha256


def test_reference_decoding_removes_xy_split_for_a_shared_static_point():
    spec = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["depth"]
    raw = np.full((2, 161), 1500, np.uint16)
    depth = decode_depth(raw, spec)
    # Independent reference golden: q=1500 -> 2249mm. Two cameras see the
    # same world point (0,0,.4) at opposite image columns, with a .8m baseline.
    z = 2.249
    K = np.array([[80 * z / .4, 0, 80.], [0, 450., 0], [0, 0, 1.]])
    millimetre_misread = decode_depth(raw, dict(encoding="scaled_integer", scale=.001))
    correct, wrong = [], []
    for x, u in ((-.4, 160), (.4, 0)):
        transform = np.diag([1., -1., -1., 1.])
        transform[:3, 3] = [x, 0, z + .4]
        correct.append(backproject(depth, K, transform, spec["kind"], spec["limits"])[0, u])
        wrong.append(backproject(millimetre_misread, K, transform, spec["kind"], spec["limits"])[0, u])
    np.testing.assert_allclose(correct, [[0, 0, .4], [0, 0, .4]], atol=2e-7)
    assert wrong[1][0] - wrong[0][0] > .26
    assert wrong[1][2] == wrong[0][2]  # Equal height does not establish correct depth.


@pytest.mark.parametrize("mode", ["guess", 1, None])
def test_invalid_depth_metadata_mode_is_rejected(mode):
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["depth"]["metadata"] = mode
    with pytest.raises(ValueError, match="depth.metadata"):
        validate_data_config(config)
