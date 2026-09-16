import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "RLBench"))

from utils.site_geometry import (  # noqa: E402
    SiteGeometry,
    sample_site_geometry,
    site_geometry_from_object,
)


class GeometryObject:
    def __init__(self, bbox=None, matrix=None):
        self.bbox = bbox
        self.matrix = matrix

    def get_bounding_box(self):
        if self.bbox is None:
            raise NotImplementedError("bbox unavailable")
        return self.bbox

    def get_matrix(self):
        if self.matrix is None:
            raise NotImplementedError("matrix unavailable")
        return self.matrix


def matrix(rotation=None, translation=(0.0, 0.0, 0.0)):
    value = np.eye(4, dtype=np.float64)
    if rotation is not None:
        value[:3, :3] = rotation
    value[:3, 3] = translation
    return value


def test_axis_aligned_bbox_builds_world_obb_and_samples_inside():
    obj = GeometryObject(
        bbox=[-1.0, 1.0, -2.0, 2.0, -3.0, 3.0],
        matrix=matrix(translation=(4.0, 5.0, 6.0)),
    )

    geometry = site_geometry_from_object(obj, [4.0, 5.0, 6.0])
    points = sample_site_geometry(geometry, 32)

    assert geometry.source == "object_bbox"
    np.testing.assert_allclose(geometry.center_world, [4.0, 5.0, 6.0])
    np.testing.assert_allclose(geometry.extent, [2.0, 4.0, 6.0])
    local = (points - geometry.center_world) @ geometry.rotation_world
    assert np.all(np.abs(local) <= geometry.extent / 2.0 + 1e-6)
    assert len(np.unique(points, axis=0)) > 1


def test_rotated_offset_bbox_uses_local_center_and_object_rotation():
    rotation = np.asarray([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    obj = GeometryObject(
        bbox=[0.0, 2.0, -1.0, 1.0, 0.0, 4.0],
        matrix=matrix(rotation, (10.0, 20.0, 30.0)),
    )

    geometry = site_geometry_from_object(obj, [10.0, 20.0, 30.0])

    np.testing.assert_allclose(geometry.center_world, [10.0, 21.0, 32.0])
    np.testing.assert_allclose(geometry.rotation_world, rotation)
    np.testing.assert_allclose(geometry.extent, [2.0, 2.0, 4.0])


def test_planar_bbox_keeps_degenerate_axis():
    geometry = site_geometry_from_object(
        GeometryObject(
            bbox=[-1.0, 1.0, -2.0, 2.0, 0.0, 0.0],
            matrix=matrix(),
        ),
        [0.0, 0.0, 0.0],
    )
    points = sample_site_geometry(geometry, 16)

    assert geometry.source == "object_bbox"
    np.testing.assert_allclose(geometry.extent, [2.0, 4.0, 0.0])
    np.testing.assert_allclose(points[:, 2], 0.0)
    assert len(np.unique(points, axis=0)) > 1


def test_all_degenerate_or_missing_bbox_uses_explicit_fallback_box():
    for bbox in ([0.0] * 6, None):
        geometry = site_geometry_from_object(
            GeometryObject(bbox=bbox, matrix=matrix()),
            [0.5, -0.25, 1.0],
        )
        assert geometry.source == "fallback_box"
        np.testing.assert_allclose(geometry.center_world, [0.5, -0.25, 1.0])
        np.testing.assert_allclose(geometry.extent, [0.02, 0.02, 0.02])
        assert len(np.unique(sample_site_geometry(geometry, 8), axis=0)) > 1


@pytest.mark.parametrize(
    "bbox,transform,error",
    [
        ([0.0, np.nan, 0.0, 1.0, 0.0, 1.0], matrix(), "finite"),
        ([1.0, 0.0, 0.0, 1.0, 0.0, 1.0], matrix(), "negative extent"),
        ([0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
         matrix(np.diag([2.0, 1.0, 1.0])), "orthonormal"),
        (None, np.full((4, 4), np.nan), "finite"),
    ],
)
def test_invalid_bbox_or_matrix_is_rejected(bbox, transform, error):
    with pytest.raises(ValueError, match=error):
        site_geometry_from_object(
            GeometryObject(bbox=bbox, matrix=transform),
            [0.0, 0.0, 0.0],
        )


def test_attribute_error_inside_existing_geometry_interface_is_strict():
    class BrokenBBox(GeometryObject):
        def get_bounding_box(self):
            raise AttributeError("internal bbox failure")

    class BrokenMatrix(GeometryObject):
        def get_matrix(self):
            raise AttributeError("internal matrix failure")

    with pytest.raises(ValueError, match="failed to read object bounding box"):
        site_geometry_from_object(
            BrokenBBox(matrix=matrix()), [0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="failed to read object matrix"):
        site_geometry_from_object(
            BrokenMatrix(bbox=None), [0.0, 0.0, 0.0])


def test_descriptor_round_trip_and_sampling_are_deterministic():
    geometry = SiteGeometry(
        primitive="box_volume",
        center_world=[1.0, 2.0, 3.0],
        rotation_world=np.eye(3),
        extent=[0.1, 0.2, 0.3],
        source="object_bbox",
    )
    restored = SiteGeometry.from_mapping(geometry.audit_dict())

    np.testing.assert_array_equal(
        sample_site_geometry(geometry, 64),
        sample_site_geometry(restored, 64),
    )


def test_descriptor_rejects_all_zero_object_bbox_and_single_point_sampling():
    with pytest.raises(ValueError, match="positive extent"):
        SiteGeometry(
            primitive="box_volume",
            center_world=[0.0, 0.0, 0.0],
            rotation_world=np.eye(3),
            extent=[0.0, 0.0, 0.0],
            source="object_bbox",
        )
    geometry = site_geometry_from_object(
        GeometryObject(bbox=None, matrix=None),
        [0.0, 0.0, 0.0],
    )
    with pytest.raises(ValueError, match="at least two"):
        sample_site_geometry(geometry, 1)
