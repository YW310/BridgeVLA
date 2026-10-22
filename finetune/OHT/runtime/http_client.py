"""IsaacLab HTTP policy client; control requests are never retried."""
import json
from time import perf_counter
import uuid
import urllib.request
import numpy as np

from ..action_modes import DEFAULTS, action_options, action_descriptor, validate_action
from .inference_api import API_VERSION
from .raw_observation import RAW_FORMAT, profile_task
from .transport import pack_observation


class BridgeVLAClient:
    def __init__(self, server, timeout=120, opener=None):
        self.server, self.timeout = server.rstrip('/'), timeout
        self.opener = opener or urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.info = self._call('/model_info')
        if self.info.get('api_version') != API_VERSION or RAW_FORMAT not in self.info.get('observation_formats', []):
            raise ValueError('Server must advertise bridgevla_rgbd_v1; update the BridgeVLA server')
        spec = self.info['action_spec']
        self.action_config = action_options({key: value for key, value in spec.items() if key in DEFAULTS})
        if spec != action_descriptor(self.action_config):
            raise ValueError('Unsupported action_spec: frame, quaternion order, units, size and gripper must match BridgeVLA semantics')
        self.episode, self.step, self.timestamp, self.instruction = None, 0, -1., None
        self.uncertain = False

    def _call(self, path, payload=None, *, timing=None):
        started = perf_counter()
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode('utf-8')
        if body is not None and len(body) > self.info['max_request_bytes']:
            raise ValueError('Raw request exceeds server max_request_bytes')
        request = urllib.request.Request(self.server + path, data=body, headers={'Content-Type': 'application/json'})
        serialized = perf_counter()
        with self.opener.open(request, timeout=self.timeout) as response:
            result = json.load(response)
        if timing is not None:
            timing.update(serialize=(serialized - started) * 1000,
                          http=(perf_counter() - serialized) * 1000)
        return result

    def _identity(self, episode):
        return dict(api_version=API_VERSION, model_id=self.info['model_id'],
                    contract_sha256=self.info['contract_sha256'], episode_id=episode,
                    request_id=uuid.uuid4().hex)

    def _check(self, request, response):
        for key in ('api_version', 'model_id', 'contract_sha256', 'episode_id', 'request_id'):
            if response.get(key) != request[key]:
                raise ValueError(f'Mismatched response {key}')

    def reset(self, episode=None):
        request = self._identity(episode or uuid.uuid4().hex)
        self.uncertain = True
        response = self._call('/reset', request)
        self._check(request, response)
        if response.get('reset') is not True or response.get('next_step') != 0:
            raise ValueError('Unexpected reset response')
        self.episode, self.step, self.timestamp, self.instruction = request['episode_id'], 0, -1., None
        self.uncertain = False
        return response

    def profile(self, task, name=None):
        profiles = self.info['raw_observation']['data_profiles']
        if name is None:
            matches = [key for key in profiles if profile_task(key) == task]
            if not matches and set(profiles) == {'default'}:
                matches = ['default']
            if len(matches) != 1:
                raise ValueError('Select --data-profile explicitly from /model_info')
            name = matches[0]
        if name not in profiles or profiles[name]['task'] not in (None, task):
            raise ValueError('Unknown/mismatched data profile')
        return name

    def predict(self, tensors, instruction, timestamp, *, task, data_profile):
        started = perf_counter()
        if self.episode is None or self.uncertain:
            raise RuntimeError('Reset with a fresh episode before requesting another action')
        if not np.isfinite(timestamp) or timestamp <= self.timestamp:
            raise ValueError('Measurement timestamp must increase')
        if self.instruction is not None and instruction != self.instruction:
            raise ValueError('Reset before changing instruction')
        request = dict(**self._identity(self.episode), step=self.step, timestamp=float(timestamp),
                       instruction=instruction, observation_format=RAW_FORMAT,
                       observation=dict(task=task, data_profile=data_profile, tensors=pack_observation(tensors)))
        timing = {'pack': (perf_counter() - started) * 1000}
        # Timeout, cancellation or a malformed reply leaves execution uncertain.
        self.uncertain = True
        response = self._call('/predict', request, timing=timing)
        self._check(request, response)
        if (response.get('step') != request['step'] or response.get('timestamp') != request['timestamp']
                or response.get('action_spec') != self.info['action_spec']):
            raise ValueError('Stale or mismatched policy response')
        action = validate_action(response['action'], self.action_config)
        self.step += 1
        self.timestamp, self.instruction, self.uncertain = timestamp, instruction, False
        # Local diagnostics; preserve the server's timing_ms unchanged.
        response['client_timing_ms'] = {**timing, 'predict': (perf_counter() - started) * 1000}
        return action, response
