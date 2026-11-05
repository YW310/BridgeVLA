"""Shared NumPy contracts retain OHT behavior without importing training code."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from finetune.bridgevla.geometry import numpy_ops as geometry
from finetune.OHT.data import geometry as oht_geometry


def test_oht_legacy_geometry_names_are_the_shared_functions():
    for name in ("array", "quaternion", "pose_matrix", "transform_matrix", "backproject", "rotation_error"):
        assert getattr(oht_geometry, name) is getattr(geometry, name)


def test_numpy_package_and_backprojection_need_neither_torch_nor_scipy():
    script = """
import importlib.abc
import sys

class BlockTrainingAndScipy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'scipy'):
            raise AssertionError('Unexpected dependency: ' + fullname)

sys.meta_path.insert(0, BlockTrainingAndScipy())
from finetune.bridgevla.geometry import numpy_ops as geometry
import numpy as np
geometry.validate_rigid_transforms(np.eye(4)[None])
geometry.validate_intrinsics(np.eye(3))
geometry.quaternion([0, 0, 0, 2])
geometry.backproject(np.ones((2, 2)), np.eye(3), np.eye(4))
assert 'torch' not in sys.modules and 'scipy' not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script],
                            cwd=Path(__file__).resolve().parents[1],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_rigid_validation_accepts_batches_and_float16_without_linalg_failure():
    poses = np.tile(np.eye(4, dtype=np.float16), (2, 3, 1, 1))
    poses[1, 2, :3, 3] = [1, 2, 3]
    checked = geometry.validate_rigid_transforms(poses)
    np.testing.assert_array_equal(checked, poses)
    assert checked.dtype == np.float64


@pytest.mark.parametrize("change,match", [
    ((3, 0, .1), "homogeneous"),
    ((0, 0, -1), "rigid rotation"),
    ((1, 1, 2), "rigid rotation"),
    ((0, 3, np.nan), "finite"),
])
def test_one_invalid_transform_rejects_the_whole_batch(change, match):
    poses = np.tile(np.eye(4), (2, 4, 1, 1))
    row, column, value = change
    poses[1, 2, row, column] = value
    with pytest.raises(ValueError, match=match):
        geometry.validate_rigid_transforms(poses)


def test_single_oht_transform_retains_old_tolerances_and_shape_contract():
    pose = np.eye(4)
    pose[3, 0] = 5e-7
    geometry.transform_matrix(pose)
    with pytest.raises(ValueError, match="homogeneous"):
        geometry.validate_rigid_transforms(pose, homogeneous_atol=1e-8, rtol=0)
    with pytest.raises(ValueError, match="finite array"):
        geometry.transform_matrix(pose[None])


def test_general_oht_intrinsics_remain_supported_while_local_pinhole_is_opt_in():
    calibration = np.array([[4., .3, 1.], [.2, 5., 2.], [0., 0., 1.]])
    geometry.validate_intrinsics(calibration)
    points = oht_geometry.backproject(np.ones((3, 3)), calibration, np.eye(4))
    pixels = points @ calibration.T
    np.testing.assert_allclose(pixels[..., :2] / pixels[..., 2:],
                               np.stack(np.meshgrid(np.arange(3), np.arange(3)), axis=-1), atol=1e-6)
    with pytest.raises(ValueError, match="intrinsics"):
        geometry.validate_intrinsics(calibration, require_upper_triangular=True)
    calibration[1, 0] = 0
    geometry.validate_intrinsics(calibration, require_upper_triangular=True)


def test_metric_z_and_ray_range_preserve_distinct_geometry_and_invalid_samples():
    depth = np.array([[2., 2., np.nan], [0., 20., 2.]])
    calibration = np.diag([2., 2., 1.])
    transform = np.eye(4)
    transform[:3, 3] = [1., 2., 3.]
    z_points = geometry.backproject(depth, calibration, transform)
    ray_points = geometry.backproject(depth, calibration, transform, kind="ray")
    np.testing.assert_allclose(z_points[0, 1], [2., 2., 5.])
    assert np.linalg.norm(ray_points[0, 1] - transform[:3, 3]) == pytest.approx(2.)
    assert z_points[0, 1, 2] != ray_points[0, 1, 2]
    assert np.isnan(z_points[0, 2]).all()
    assert np.isnan(z_points[1, :2]).all()
    assert z_points.dtype == np.float32


def test_quaternion_normalization_and_strict_unit_validation_are_different_contracts():
    source = np.array([[0., 0., 0., -2.], [0., 0., 3., 0.]], dtype=np.float32)
    original = source.copy()
    with pytest.raises(ValueError, match="unit xyzw"):
        geometry.validate_unit_quaternions(source)
    np.testing.assert_array_equal(source, original)
    normalized = geometry.quaternion(source)
    np.testing.assert_array_equal(normalized, [[0., 0., 0., -1.], [0., 0., 1., 0.]])
    assert geometry.validate_unit_quaternions(normalized) is normalized
    with pytest.raises(ValueError, match="Zero quaternion"):
        geometry.quaternion(np.zeros(4))
    with pytest.raises(ValueError, match="finite"):
        geometry.quaternion([1e308, 0, 0, 1e308])


def test_refiner_cache_opts_into_strict_pinhole_calibration(tmp_path):
    from tools.train_local_geometry_refiner import SCHEMA, load_cache

    calibration = np.eye(3, dtype=np.float32)
    calibration[1, 0] = .1
    path = tmp_path / "cache.npz"
    np.savez(path, schema=np.array(SCHEMA), source=np.array("synthetic"),
             baseline_id=np.array("synthetic_test"), camera_model=np.array("opencv_z_meters"),
             candidate_source=np.array("synthetic_perturbation"),
             depth=np.ones((1, 1, 2, 2), np.float32), intrinsics=calibration[None, None],
             camera_to_world=np.eye(4, dtype=np.float32)[None, None],
             candidate_pose=np.eye(4, dtype=np.float32)[None], target_pose=np.eye(4, dtype=np.float32)[None],
             context=np.zeros((1, 1), np.float32), episode_id=np.array(["episode"]), sample_id=np.array(["sample"]))
    with pytest.raises(ValueError, match="intrinsics"):
        load_cache(path)
