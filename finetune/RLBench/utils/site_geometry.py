"""Shared site-region geometry and deterministic point sampling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import numpy as np


SITE_GEOMETRY_PRIMITIVE = "box_volume"
SITE_GEOMETRY_SOURCES = frozenset(("object_bbox", "fallback_box"))
DEFAULT_FALLBACK_EXTENT_M = (0.02, 0.02, 0.02)
SEMANTIC_ROLE_SCHEMA = "rlbench_o2_semantic_roles_v2"


def _vector3(
    value: Any,
    field: str,
    *,
    non_negative: bool = False,
    allow_zero: bool = True,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,):
        raise ValueError(f"{field} must have shape [3], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field} must contain only finite values")
    if non_negative and (
        np.any(array < 0.0) or (not allow_zero and np.any(array <= 0.0))
    ):
        relation = "positive" if not allow_zero else "non-negative"
        raise ValueError(f"{field} must be {relation}")
    return array


def _rotation3(value: Any, field: str = "rotation_world") -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"{field} must have shape [3,3], got {rotation.shape}")
    if not np.all(np.isfinite(rotation)):
        raise ValueError(f"{field} must contain only finite values")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0.0):
        raise ValueError(f"{field} must be orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=1e-5, rtol=0.0):
        raise ValueError(f"{field} must be a proper rotation (determinant +1)")
    return rotation


@dataclass(frozen=True)
class SiteGeometry:
    primitive: str
    center_world: np.ndarray
    rotation_world: np.ndarray
    extent: np.ndarray
    source: str

    def __post_init__(self) -> None:
        if self.primitive != SITE_GEOMETRY_PRIMITIVE:
            raise ValueError(
                f"unsupported site primitive {self.primitive!r}; "
                f"expected {SITE_GEOMETRY_PRIMITIVE!r}"
            )
        if self.source not in SITE_GEOMETRY_SOURCES:
            raise ValueError(f"unsupported site geometry source {self.source!r}")
        center = _vector3(self.center_world, "center_world")
        rotation = _rotation3(self.rotation_world)
        extent = _vector3(self.extent, "extent", non_negative=True)
        if self.source == "fallback_box" and np.any(extent <= 0.0):
            raise ValueError("fallback_box extent must be positive on all axes")
        if self.source == "object_bbox" and not np.any(extent > 0.0):
            raise ValueError("object_bbox must have at least one positive extent")
        object.__setattr__(self, "center_world", center.copy())
        object.__setattr__(self, "rotation_world", rotation.copy())
        object.__setattr__(self, "extent", extent.copy())

    def audit_dict(self) -> dict[str, Any]:
        return {
            "primitive": self.primitive,
            "center_world": self.center_world.tolist(),
            "rotation_world": self.rotation_world.tolist(),
            "extent": self.extent.tolist(),
            "source": self.source,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SiteGeometry":
        if not isinstance(value, Mapping):
            raise ValueError("site_geometry must be a mapping")
        required = ("primitive", "center_world", "rotation_world", "extent", "source")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"site_geometry is missing fields: {', '.join(missing)}")
        return cls(
            primitive=str(value["primitive"]),
            center_world=value["center_world"],
            rotation_world=value["rotation_world"],
            extent=value["extent"],
            source=str(value["source"]),
        )


def validate_fallback_extent(value: Sequence[float]) -> np.ndarray:
    return _vector3(
        value, "fallback_extent_m", non_negative=True, allow_zero=False
    )


def _object_matrix(obj: Any) -> Optional[np.ndarray]:
    get_matrix = getattr(obj, "get_matrix", None)
    if not callable(get_matrix):
        return None
    try:
        matrix = np.asarray(get_matrix(), dtype=np.float64)
    except NotImplementedError:
        return None
    except Exception as exc:
        raise ValueError(f"failed to read object matrix: {exc}") from exc
    if matrix.shape != (4, 4):
        raise ValueError(f"object matrix must have shape [4,4], got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("object matrix must contain only finite values")
    _rotation3(matrix[:3, :3], "object matrix rotation")
    return matrix


def _object_bbox(obj: Any) -> Optional[np.ndarray]:
    get_bbox = getattr(obj, "get_bounding_box", None)
    if not callable(get_bbox):
        return None
    try:
        bbox = np.asarray(get_bbox(), dtype=np.float64)
    except NotImplementedError:
        return None
    except Exception as exc:
        raise ValueError(f"failed to read object bounding box: {exc}") from exc
    if bbox.shape != (6,):
        raise ValueError(f"object bounding box must have shape [6], got {bbox.shape}")
    if not np.all(np.isfinite(bbox)):
        raise ValueError("object bounding box must contain only finite values")
    minimum = bbox[[0, 2, 4]]
    maximum = bbox[[1, 3, 5]]
    if np.any(maximum < minimum):
        raise ValueError("object bounding box has a negative extent")
    return bbox


def site_geometry_from_object(
    obj: Any,
    site_position: Sequence[float],
    fallback_extent_m: Sequence[float] = DEFAULT_FALLBACK_EXTENT_M,
) -> SiteGeometry:
    """Build a world-space OBB, falling back only for missing or all-zero bbox."""

    site_position_array = _vector3(site_position, "site_position")
    fallback_extent = validate_fallback_extent(fallback_extent_m)
    bbox = _object_bbox(obj)
    matrix = _object_matrix(obj)

    if bbox is not None:
        minimum = bbox[[0, 2, 4]]
        maximum = bbox[[1, 3, 5]]
        extent = maximum - minimum
        if np.any(extent > 0.0):
            if matrix is None:
                raise ValueError("object matrix is required for a non-degenerate bounding box")
            local_center = (minimum + maximum) * 0.5
            center_world = matrix[:3, :3] @ local_center + matrix[:3, 3]
            return SiteGeometry(
                primitive=SITE_GEOMETRY_PRIMITIVE,
                center_world=center_world,
                rotation_world=matrix[:3, :3],
                extent=extent,
                source="object_bbox",
            )

    rotation = np.eye(3) if matrix is None else matrix[:3, :3]
    return SiteGeometry(
        primitive=SITE_GEOMETRY_PRIMITIVE,
        center_world=site_position_array,
        rotation_world=rotation,
        extent=fallback_extent,
        source="fallback_box",
    )


def _radical_inverse(index: int, base: int) -> float:
    value = 0.0
    scale = 1.0 / float(base)
    while index:
        index, digit = divmod(index, base)
        value += digit * scale
        scale /= float(base)
    return value


def sample_site_geometry(geometry: SiteGeometry, num_points: int) -> np.ndarray:
    """Return deterministic Halton samples inside the oriented box."""

    if num_points < 2:
        raise ValueError("site geometry requires at least two points")
    unit = np.asarray(
        [
            (
                _radical_inverse(index, 2),
                _radical_inverse(index, 3),
                _radical_inverse(index, 5),
            )
            for index in range(1, num_points + 1)
        ],
        dtype=np.float64,
    )
    local = (unit - 0.5) * geometry.extent[None, :]
    world = local @ geometry.rotation_world.T + geometry.center_world[None, :]
    return world.astype(np.float32)
