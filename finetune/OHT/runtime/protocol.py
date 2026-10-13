"""Versioned action protocol with explicit EEF/joint units and frame negotiation."""
import numpy as np
from ..action_modes import action_options, action_descriptor, validate_action

SCHEMA = "bridgevla_oht_absolute_tcp_v1"
ACTION_SCHEMA = "bridgevla_oht_action_v2"


def protocol_schema(options=None):
    return SCHEMA if action_options(options)["mode"] == "eef" else ACTION_SCHEMA


def request_action_options(request):
    if request.get("schema") == SCHEMA:
        if "action_spec" in request and request["action_spec"] != action_descriptor():
            raise ValueError("Legacy EEF protocol cannot declare another action mode")
        return action_options()
    if request.get("schema") != ACTION_SCHEMA:
        raise ValueError("Unsupported OHT protocol schema")
    spec = request.get("action_spec")
    if not isinstance(spec, dict):
        raise ValueError("Vector protocol requires an explicit action_spec")
    options = action_options({key: spec[key] for key in (
        "mode", "translation_frame", "rotation_frame", "world_from_base", "joint_names", "joint_wrap")
        if key in spec})
    if options["mode"] == "eef" or spec != action_descriptor(options):
        raise ValueError("Invalid/missing action units, dimensions or coordinate contract")
    return options


def request_identity(request):
    request_action_options(request)
    if not isinstance(request.get("episode"), str) or not request["episode"]:
        raise ValueError("episode must be a nonempty identifier")
    if type(request.get("step")) is not int or request["step"] < 0:
        raise ValueError("step must be a nonnegative integer")
    timestamp = float(request["timestamp"])
    if not np.isfinite(timestamp) or timestamp < 0:
        raise ValueError("Invalid measurement timestamp")
    return dict(schema=request["schema"], episode=request["episode"], step=request["step"], timestamp=timestamp)


def action_response(request, action, action_config=None):
    identity = request_identity(request)
    options = request_action_options(request)
    if action_config is not None and action_descriptor(options) != action_descriptor(action_config):
        raise ValueError("Client/server action mode or coordinate contract mismatch")
    action = validate_action(action, options)
    if identity["schema"] == SCHEMA:
        return dict(**identity, frame="world", quaternion_order="xyzw", reference_point="tcp",
                    action=action.tolist())
    return dict(**identity, action_spec=action_descriptor(options), action=action.tolist())


def validate_response(request, response):
    identity = request_identity(request)
    if any(response.get(key) != value for key, value in identity.items()):
        raise ValueError("Stale/mismatched episode, step or timestamp")
    if identity["schema"] == SCHEMA:
        if (response.get("frame"), response.get("quaternion_order"), response.get("reference_point")) != ("world", "xyzw", "tcp"):
            raise ValueError("Response action coordinate contract mismatch")
    elif response.get("action_spec") != request["action_spec"]:
        raise ValueError("Response action mode/units/frame mismatch")
    return action_response(request, response["action"])["action"]
