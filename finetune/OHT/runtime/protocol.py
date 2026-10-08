"""Versioned local bridge protocol, deliberately distinct from unknown H-VLA APIs."""
import numpy as np
from ..data.geometry import array, quaternion

SCHEMA = "bridgevla_oht_absolute_tcp_v1"


def request_identity(request):
    if request.get("schema") != SCHEMA:
        raise ValueError("Unsupported OHT protocol schema")
    if not isinstance(request.get("episode"), str) or not request["episode"]:
        raise ValueError("episode must be a nonempty identifier")
    if type(request.get("step")) is not int or request["step"] < 0:
        raise ValueError("step must be a nonnegative integer")
    timestamp = float(request["timestamp"])
    if not np.isfinite(timestamp) or timestamp < 0:
        raise ValueError("Invalid measurement timestamp")
    return dict(schema=SCHEMA, episode=request["episode"], step=request["step"], timestamp=timestamp)


def action_response(request, action):
    identity = request_identity(request)
    action = array(action, (8,), "absolute TCP action").copy()
    if action[7] not in (0, 1):
        raise ValueError("gripper_open must be binary")
    action[3:7] = quaternion(action[3:7])
    return dict(**identity, frame="world", quaternion_order="xyzw", reference_point="tcp",
                action=action.tolist())


def validate_response(request, response):
    identity = request_identity(request)
    if any(response.get(key) != value for key, value in identity.items()):
        raise ValueError("Stale/mismatched episode, step or timestamp")
    if (response.get("frame"), response.get("quaternion_order"), response.get("reference_point")) != ("world", "xyzw", "tcp"):
        raise ValueError("Response action coordinate contract mismatch")
    return action_response(request, response["action"])["action"]
