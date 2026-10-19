"""Small, versioned HTTP envelopes independent of a particular action head."""
import json
import math

API_VERSION = "bridgevla_predict_v1"
OBSERVATION_FORMAT = "bridgevla_prepared_v1"


class APIError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.logged = False

    def payload(self):
        return dict(error=dict(code=self.code, message=self.message))


def text(value, name, limit=256):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise APIError(422, "invalid_request", f"{name} must be a nonempty string of at most {limit} characters")
    return value


def decode_request(raw, operation):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON number: {value}")
    try:
        value = json.loads(raw, parse_constant=reject_constant)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise APIError(400, "invalid_json", "Expected a valid JSON object with finite numeric fields") from exc
    common = {"api_version", "request_id", "episode_id", "model_id", "contract_sha256"}
    required = common | ({"step", "timestamp", "instruction", "observation_format", "observation"}
                         if operation == "predict" else set())
    if not isinstance(value, dict) or set(value) != required:
        raise APIError(422, "invalid_request", f"Expected exactly these fields: {sorted(required)}")
    if value["api_version"] != API_VERSION:
        raise APIError(422, "unsupported_version", f"Expected api_version={API_VERSION}")
    for key in common - {"api_version"}:
        text(value[key], key)
    if operation == "predict":
        text(value["instruction"], "instruction", 4096)
        if type(value["step"]) is not int or value["step"] < 0:
            raise APIError(422, "invalid_request", "step must be a nonnegative integer")
        timestamp = value["timestamp"]
        try:
            valid_time = type(timestamp) in (int, float) and math.isfinite(timestamp) and timestamp >= 0
        except OverflowError:
            valid_time = False
        if not valid_time:
            raise APIError(422, "invalid_request", "timestamp must be finite and nonnegative")
        if value["observation_format"] != OBSERVATION_FORMAT:
            raise APIError(422, "unsupported_observation", f"Expected observation_format={OBSERVATION_FORMAT}")
    return value
