"""Prepared observations use the same layout as the checkpoint's replay."""
from ..action_modes import state_fields
from ..data.observation import validate_observation
from ..data.point_filter import filter_observation_points
from .transport import DTYPES, unpack_observation
from .inference_api import APIError


def observation_schema(policy):
    data = policy.contract["data_config"]
    height, width = map(int, data["image_size"])
    fields = {"low_dim_state": dict(dtype="float32", shape=[4])}
    if getattr(policy, 'chunk_settings', None) is not None:
        fields.update(tcp_pose=dict(dtype='float32', shape=[7]),
                      joint_positions=dict(dtype='float32', shape=[6]),
                      gripper_open01=dict(dtype='float32', shape=[]))
    widths = {"tcp_pose": 7, "joint_positions": 6}
    for name in state_fields(policy.action_config):
        fields[name] = dict(dtype="float32", shape=[widths[name]])
    for camera in data["cameras"]:
        for suffix, dtype, shape in (
            ("rgb", "uint8", [3, height, width]),
            ("depth", "float32", [1, height, width]),
            ("point_cloud", "float32", [3, height, width]),
            ("camera_intrinsics", "float32", [3, 3]),
            ("camera_extrinsics", "float32", [4, 4]),
        ):
            fields[f"{camera}_{suffix}"] = dict(dtype=dtype, shape=shape)
    return fields


def decode_prepared(packed, policy, schema):
    try:
        if not isinstance(packed, dict) or set(packed) != set(schema):
            raise ValueError("Observation fields must exactly match /model_info; teachers/labels are not accepted")
        for name, expected in schema.items():
            spec = packed[name]
            if not isinstance(spec, dict) or set(spec) != {"dtype", "shape", "data"}:
                raise ValueError(f"{name}: expected dtype, shape, data")
            dtype = spec["dtype"]
            allowed = ("uint8",) if expected["dtype"] == "uint8" else ("float32", "float64")
            if dtype not in allowed or spec["shape"] != expected["shape"] or not isinstance(spec["data"], str):
                raise ValueError(f"{name}: invalid dtype, shape or Base64 data")
        decoded = unpack_observation(packed)
        decoded = {name: value.astype(DTYPES[schema[name]["dtype"]], copy=False)
                   for name, value in decoded.items()}
        decoded = filter_observation_points(decoded, policy.contract["data_config"])
        return validate_observation(decoded, policy.contract["data_config"])
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        raise APIError(422, "invalid_observation", str(exc)) from exc
