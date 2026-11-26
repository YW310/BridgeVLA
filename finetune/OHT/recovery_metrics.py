"""CPU metrics for paired, physically consistent recovery observations.

These metrics describe intended model outputs, not motion executed by a robot.
In particular, a positive gain is a local output contraction, not task success.
Pose quaternions use xyzw order and rotation vectors use world axes.
"""

import numbers

import numpy as np


_QUATERNION_TOLERANCE = 1e-4


def _array(value, name, width, *, vector=False):
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    valid_shape = result.shape == (width,) if vector else (
        result.ndim == 2 and result.shape[0] > 0 and result.shape[1] == width
    )
    if not valid_shape:
        shape = f"[{width}]" if vector else f"[N, {width}] with N > 0"
        raise ValueError(f"{name} must have shape {shape}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    quaternions = result[3:7] if vector else result[:, 3:7]
    norms = np.linalg.norm(quaternions, axis=-1)
    if np.any(np.abs(norms - 1.0) > _QUATERNION_TOLERANCE):
        raise ValueError(f"{name} must contain unit xyzw quaternions")
    # Do not mutate caller arrays, including float64 inputs.
    result = result.copy()
    result[..., 3:7] /= np.expand_dims(norms, axis=-1)
    return result


def _floor(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a finite nonnegative number")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return value


def _rotation_difference(perturbed, base):
    """Log(R_perturbed R_base.T), with quaternion-sign invariance."""
    pv, pw = perturbed[:3], perturbed[3]
    bv, bw = base[:3], base[3]
    vector = -pw * bv + bw * pv - np.cross(pv, bv)
    scalar = pw * bw + np.dot(pv, bv)
    # The two signs represent the same rotation. At exactly pi use a
    # deterministic axis, where the shortest logarithm is otherwise ambiguous.
    if scalar < 0:
        vector, scalar = -vector, -scalar
    if abs(scalar) <= np.finfo(np.float64).eps:
        nonzero = np.flatnonzero(np.abs(vector) > np.finfo(np.float64).eps)
        if len(nonzero) and vector[nonzero[0]] < 0:
            vector = -vector
        scalar = abs(scalar)
    sine = float(np.linalg.norm(vector))
    if sine <= np.finfo(np.float64).eps:
        return 2.0 * vector
    return vector * (2.0 * np.arctan2(sine, scalar) / sine)


def _gain(error, difference, floor):
    magnitude = float(np.linalg.norm(error))
    if magnitude <= floor:
        return None
    # This form avoids squaring a very small magnitude.
    return float(-np.dot(error / magnitude, (difference - error) / magnitude))


def _path_distance(position, path):
    """Distance to the union of continuous line segments, including zero ones."""
    if len(path) == 1:
        return float(np.linalg.norm(position - path[0]))
    starts = path[:-1]
    segments = path[1:] - starts
    squared_lengths = np.einsum("ij,ij->i", segments, segments)
    fractions = np.zeros(len(segments), dtype=np.float64)
    np.divide(np.einsum("ij,ij->i", position - starts, segments),
              squared_lengths, out=fractions, where=squared_lengths > 0)
    closest = starts + np.clip(fractions, 0.0, 1.0)[:, None] * segments
    return float(np.linalg.norm(closest - position, axis=1).min())


def paired_recovery(base_pose, perturbed_pose, base_actions, perturbed_actions,
                    reference_path, *, execute_horizon=4,
                    position_floor_m=0.0005,
                    rotation_floor_rad=np.deg2rad(0.1)):
    """Compare two model forwards for a baseline and a perturbed state.

    ``base_pose`` and ``perturbed_pose`` have shape [7], ``*_actions`` have
    matching shape [K, 8], and ``reference_path`` has shape [N, 7]. All poses
    are world poses with unit xyzw quaternions. Actions are absolute poses
    followed by a gripper scalar. A one-point path is treated as a point.

    For each prefix endpoint, e is the initial perturbed-minus-base pose
    difference, D is its output counterpart, and delta_u = D - e. The gain is
    -e.dot(delta_u) / ||e||**2. A gain of zero means the intended output follows
    the perturbation; one means both forwards select the same endpoint.

    Position normal gain projects e and D onto the plane perpendicular to
    reference_path[-1].position - reference_path[0].position. Rotation gain
    uses differences of world-axis SO(3) logarithms and is a local diagnostic.
    Neither gain defines a unique correct action away from the reference.
    """
    if (isinstance(execute_horizon, (bool, np.bool_)) or
            not isinstance(execute_horizon, numbers.Integral) or execute_horizon <= 0):
        raise ValueError("execute_horizon must be a positive integer")
    horizon = int(execute_horizon)
    position_floor_m = _floor(position_floor_m, "position_floor_m")
    rotation_floor_rad = _floor(rotation_floor_rad, "rotation_floor_rad")
    base = _array(base_pose, "base_pose", 7, vector=True)
    perturbed = _array(perturbed_pose, "perturbed_pose", 7, vector=True)
    base_actions = _array(base_actions, "base_actions", 8)
    perturbed_actions = _array(perturbed_actions, "perturbed_actions", 8)
    reference = _array(reference_path, "reference_path", 7)
    if base_actions.shape != perturbed_actions.shape:
        raise ValueError("base_actions and perturbed_actions must have matching shapes")
    if horizon > len(base_actions):
        raise ValueError("execute_horizon cannot exceed the action chunk length")

    position_error = perturbed[:3] - base[:3]
    rotation_error = _rotation_difference(perturbed[3:7], base[3:7])
    tangent = reference[-1, :3] - reference[0, :3]
    tangent_magnitude = float(np.linalg.norm(tangent))
    tangent = tangent / tangent_magnitude if tangent_magnitude > position_floor_m else None
    projector = None if tangent is None else np.eye(3) - np.outer(tangent, tangent)
    normal_error = None if projector is None else projector @ position_error
    path = reference[:, :3]

    prefixes = []
    for index in range(horizon):
        position_difference = perturbed_actions[index, :3] - base_actions[index, :3]
        rotation_difference = _rotation_difference(
            perturbed_actions[index, 3:7], base_actions[index, 3:7])
        normal_difference = None if projector is None else projector @ position_difference
        prefixes.append({
            "prefix": index + 1,
            "position": {
                "e_world_m": position_error.tolist(),
                "D_world_m": position_difference.tolist(),
                "delta_u_world_m": (position_difference - position_error).tolist(),
                "gain": _gain(position_error, position_difference, position_floor_m),
                "normal_e_world_m": None if normal_error is None else normal_error.tolist(),
                "normal_D_world_m": None if normal_difference is None else normal_difference.tolist(),
                "normal_delta_u_world_m": None if normal_difference is None else (
                    normal_difference - normal_error).tolist(),
                "normal_gain": None if projector is None else _gain(
                    normal_error, normal_difference, position_floor_m),
            },
            "rotation": {
                "e_world_rad": rotation_error.tolist(),
                "D_world_rad": rotation_difference.tolist(),
                "delta_u_world_rad": (rotation_difference - rotation_error).tolist(),
                "gain": _gain(rotation_error, rotation_difference, rotation_floor_rad),
            },
            "path_distance_m": {
                "base": _path_distance(base_actions[index, :3], path),
                "perturbed": _path_distance(perturbed_actions[index, :3], path),
            },
        })

    return {
        "intended_output_only": True,
        "rotation_gain_is_local": True,
        "execute_horizon": horizon,
        "position_floor_m": position_floor_m,
        "rotation_floor_rad": rotation_floor_rad,
        "reference_tangent_world": None if tangent is None else tangent.tolist(),
        "initial": {
            "position_error_world_m": position_error.tolist(),
            "position_error_m": float(np.linalg.norm(position_error)),
            "rotation_error_world_rad": rotation_error.tolist(),
            "rotation_error_rad": float(np.linalg.norm(rotation_error)),
            "path_distance_m": {
                "base": _path_distance(base[:3], path),
                "perturbed": _path_distance(perturbed[:3], path),
            },
        },
        "prefixes": prefixes,
    }
