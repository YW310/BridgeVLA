"""Compact typed tensors over the explicit OHT HTTP protocol."""
import base64
import json
import urllib.request
import numpy as np
from .protocol import SCHEMA, request_identity, validate_response

DTYPES = {"uint8": np.dtype("uint8"), "float32": np.dtype("<f4"), "float64": np.dtype("<f8")}
MAX_BYTES = 32 * 1024 * 1024


def pack_observation(observation):
    packed = {}
    for name, value in observation.items():
        value = np.asarray(value)
        dtype = DTYPES.get(value.dtype.name)
        if dtype is None:
            raise ValueError(f"Unsupported observation dtype {value.dtype}")
        value = np.ascontiguousarray(value, dtype=dtype)
        packed[name] = dict(dtype=value.dtype.name, shape=list(value.shape),
                            data=base64.b64encode(value.tobytes()).decode("ascii"))
    return packed


def unpack_observation(packed):
    if not isinstance(packed, dict) or len(packed) > 64:
        raise ValueError("Invalid observation envelope")
    result, total = {}, 0
    for name, spec in packed.items():
        dtype = DTYPES.get(spec.get("dtype"))
        shape = spec.get("shape")
        if dtype is None or not isinstance(shape, list) or not 1 <= len(shape) <= 4:
            raise ValueError("Invalid tensor dtype/shape")
        if any(type(v) is not int or v <= 0 for v in shape):
            raise ValueError("Tensor dimensions must be positive integers")
        count = 1
        for size in shape:
            count *= size
        expected = count * dtype.itemsize
        total += expected
        if total > MAX_BYTES:
            raise ValueError("Observation exceeds tensor size limit")
        raw = base64.b64decode(spec["data"], validate=True)
        if len(raw) != expected:
            raise ValueError("Tensor payload size does not match shape")
        result[name] = np.frombuffer(raw, dtype=dtype).reshape(shape).copy()
    return result


class Client:
    """One client per server; never retries a timed-out control action."""
    def __init__(self, url, contract_sha256, timeout=30):
        self.url = url.rstrip("/") + "/act"
        self.contract_sha256, self.timeout = contract_sha256, timeout
        # Control traffic is a direct simulator/server connection. Corporate
        # HTTP proxy environment variables must not route local robot requests.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def act(self, observation, goal, episode, step, timestamp):
        request = dict(schema=SCHEMA, episode=episode, step=step, timestamp=float(timestamp),
                       contract_sha256=self.contract_sha256, goal=goal,
                       observation=pack_observation(observation))
        request_identity(request)
        body = json.dumps(request, allow_nan=False).encode("utf-8")
        if len(body) > MAX_BYTES:
            raise ValueError("HTTP payload too large")
        message = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        with self.opener.open(message, timeout=self.timeout) as response:
            payload = json.load(response)
        if payload.get("contract_sha256") != self.contract_sha256:
            raise ValueError("Server response contract mismatch")
        return np.asarray(validate_response(request, payload), dtype=np.float32)
