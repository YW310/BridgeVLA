"""NumPy geometry shared by data adapters and offline experiment caches.

Quaternions use xyzw. Optical cameras use +X right, +Y down, +Z forward;
metric Z-depth and radial range remain distinct, explicitly selected inputs.
This module has no torch or policy dependency.
"""
import numpy as np


def array(value, shape=None, name="value"):
    value = np.asarray(value, dtype=np.float64)
    if (shape is not None and value.shape != shape) or not np.isfinite(value).all():
        raise ValueError(f"{name}: expected finite array {shape}, got {value.shape}")
    return value


def quaternion(value):
    """Normalize nonzero xyzw quaternions, preserving their existing signs."""
    value = array(value, name="quaternion")
    if value.shape[-1:] != (4,):
        raise ValueError("Expected xyzw quaternion")
    with np.errstate(over="ignore"):
        norms = np.linalg.norm(value, axis=-1, keepdims=True)
    if np.any(norms < 1e-8):
        raise ValueError("Zero quaternion")
    if not np.isfinite(norms).all():
        raise ValueError("Expected nonzero finite xyzw quaternion")
    return value / norms


def validate_unit_quaternions(value, name="quaternions", *, atol=1e-5, rtol=1e-5):
    """Reject nonunit xyzw quaternions without normalizing or mutating them."""
    value = np.asarray(value)
    if (value.shape[-1:] != (4,) or not np.isfinite(value).all()
            or not np.allclose(np.linalg.norm(value, axis=-1), 1, atol=atol, rtol=rtol)):
        raise ValueError(f"{name} must be unit xyzw quaternions")
    return value


def pose_matrix(position, orientation):
    from scipy.spatial.transform import Rotation

    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quaternion(orientation)).as_matrix()
    result[:3, 3] = array(position, (3,), "position")
    return result


def validate_rigid_transforms(value, name="transform", *, homogeneous_atol=1e-6,
                              rotation_atol=1e-5, rtol=1e-5):
    """Validate finite [...,4,4] rigid transforms, including batched caches.

    Tolerances are explicit so existing data adapters retain their contracts.
    Validation uses float64, including for float16 cache arrays.
    """
    value = array(value, name=name)
    if value.shape[-2:] != (4, 4):
        raise ValueError(f"{name}: expected rigid transforms [...,4,4]")
    if not np.allclose(value[..., 3, :], [0, 0, 0, 1], atol=homogeneous_atol, rtol=rtol):
        raise ValueError(f"{name}: invalid homogeneous row")
    rotation = value[..., :3, :3]
    if (not np.allclose(rotation.swapaxes(-1, -2) @ rotation, np.eye(3),
                        atol=rotation_atol, rtol=rtol)
            or not np.allclose(np.linalg.det(rotation), 1, atol=rotation_atol, rtol=rtol)):
        raise ValueError(f"{name}: expected a rigid rotation")
    return value


def transform_matrix(value, name="transform"):
    """Validate one rigid transform with the original OHT tolerances."""
    return validate_rigid_transforms(array(value, (4, 4), name), name)


def validate_intrinsics(value, name="intrinsics", *, atol=1e-8, rtol=1e-5,
                        require_upper_triangular=False):
    """Validate [...,3,3] K while leaving stricter pinhole rules opt-in.

    Existing backprojection accepts nonzero K[1,0]. Projective local evidence
    requires standard upper-triangular pinhole intrinsics and opts into that
    additional condition. Upper-right skew K[0,1] is supported in both cases.
    """
    value = array(value, name=name)
    if (value.shape[-2:] != (3, 3)
            or np.any(value[..., 0, 0] <= 0) or np.any(value[..., 1, 1] <= 0)
            or not np.allclose(value[..., 2, :], [0, 0, 1], atol=atol, rtol=rtol)
            or (require_upper_triangular
                and not np.allclose(value[..., 1, 0], 0, atol=atol, rtol=rtol))):
        raise ValueError(f"{name}: invalid camera intrinsics")
    return value


def backproject(depth, intrinsics, world_from_optical, kind="z", limits=(0.001, 10.0)):
    """Unproject metric depth; only radial range normalizes K^-1[u,v,1]."""
    depth = np.asarray(depth, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("Metric depth must be HxW")
    intrinsics = validate_intrinsics(array(intrinsics, (3, 3), "intrinsics"))
    transform = transform_matrix(world_from_optical)
    v, u = np.indices(depth.shape)
    rays = np.stack((u, v, np.ones_like(u)), axis=-1) @ np.linalg.inv(intrinsics).T
    if kind == "ray":
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    elif kind != "z":
        raise ValueError("depth kind must be z or ray")
    valid = np.isfinite(depth) & (depth >= limits[0]) & (depth <= limits[1])
    points = (rays * depth[..., None]) @ transform[:3, :3].T + transform[:3, 3]
    points[~valid] = np.nan
    return points.astype(np.float32)


def rotation_error(first, second):
    """Geodesic error in radians for xyzw quaternions."""
    from scipy.spatial.transform import Rotation

    return (Rotation.from_quat(quaternion(first)).inv()
            * Rotation.from_quat(quaternion(second))).magnitude()
