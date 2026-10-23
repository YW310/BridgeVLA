"""One model worker, explicit admission and checkpoint-bound episode state."""
import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import copy
from datetime import datetime
import hashlib
import inspect
from importlib.metadata import PackageNotFoundError, version
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import threading
from time import perf_counter

from ..action_modes import action_descriptor, validate_action
from ..data.common import write_json
from ..run_directory import RUN_TIMEZONE
from .inference_api import API_VERSION, OBSERVATION_FORMAT, APIError, decode_request, text
from .observation_adapter import observation_schema, decode_prepared
from .raw_observation import RAW_FORMAT, data_profiles, decode_raw, raw_schema

LOG = logging.getLogger(__name__)


def resolve_checkpoint(model_path):
    path = Path(model_path).expanduser().resolve(strict=True)
    if path.is_dir():
        path = path / "model_last.pth"
    if not path.is_file():
        raise ValueError(f"Expected a checkpoint file or run directory containing model_last.pth: {path}")
    return path


class InferenceService:
    def __init__(self, model_path, *, device="cuda:0", pretrain_path=None, predictor=None,
                 predictor_provenance=None, output_dir="logs/oht_vla_server", model_id=None,
                 policy_loader=None, warmup_replay=None, warmup_split="val", warmup_index=0,
                 log_max_bytes=10*1024*1024, log_backups=3, visualize_every=10, visualize_keep=100):
        if type(visualize_every) is not int or visualize_every < 0:
            raise ValueError('visualize_every must be a nonnegative integer')
        if type(visualize_keep) is not int or visualize_keep < 1:
            raise ValueError('visualize_keep must be a positive integer')
        self.checkpoint = resolve_checkpoint(model_path)
        stat = self.checkpoint.stat()
        identity = f"{self.checkpoint}:{stat.st_size}:{stat.st_mtime_ns}"
        self.model_id = text(model_id or hashlib.sha256(identity.encode()).hexdigest()[:20], "model_id")
        self.device, self.pretrain_path = device, pretrain_path
        self.predictor, self.provenance = predictor, predictor_provenance
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.loader = policy_loader
        self.warmup = (warmup_replay, warmup_split, warmup_index)
        if type(warmup_index) is not int or warmup_index < 0:
            raise ValueError("warmup_index must be a nonnegative integer")
        self.log_max_bytes, self.log_backups = log_max_bytes, log_backups
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="oht-policy")
        self.lock = threading.Lock()
        self.policy, self.metadata, self.fields = None, {}, {}
        self.ready, self.busy, self.stopping = False, False, False
        self.error, self.failed = None, False
        self.episode, self.step, self.timestamp, self.instruction = None, 0, -1., None
        self.gripper_history, self.input_identity = None, None
        self.episodes, self.request_ids = set(), set()
        self.recent_ids = deque()
        self.logger, self.log_handler = None, None
        self.visualizer = None
        self.visualize_every, self.visualize_keep = visualize_every, visualize_keep
        self.visualization_reason, self.visualization_capture_error = 'disabled', None

    async def start(self):
        await self._wait(self.executor.submit(self._load))

    @staticmethod
    async def _wait(future):
        # Shield the actual model work when an HTTP waiter is cancelled. Retrieve
        # late exceptions as well, even when no request coroutine remains to await.
        wrapped = asyncio.wrap_future(future)
        wrapped.add_done_callback(lambda done: None if done.cancelled() else done.exception())
        return await asyncio.shield(wrapped)

    def _load(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"oht.requests.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        self.log_handler = RotatingFileHandler(self.output_dir / "requests.jsonl",
                                              maxBytes=self.log_max_bytes, backupCount=self.log_backups,
                                              encoding="utf-8")
        self.log_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(self.log_handler)
        loader = self.loader
        if loader is None:
            from .loading import load_policy
            loader = load_policy
        self.policy = loader(str(self.checkpoint), self.device, self.pretrain_path,
                             self.predictor, self.provenance)
        self.fields = observation_schema(self.policy)
        self._warmup()
        if self.visualize_every:
            try:
                parameters = inspect.signature(self.policy.act).parameters
                supported = ('visualization_callback' in parameters or
                             any(value.kind == inspect.Parameter.VAR_KEYWORD for value in parameters.values()))
            except (ValueError, TypeError):
                supported = False
            if supported:
                from .online_visualization import OnlineVisualizer
                try:
                    self.visualizer = OnlineVisualizer(self.output_dir / 'visualizations',
                                                        every=self.visualize_every, keep=self.visualize_keep)
                    self.visualization_reason = None
                except Exception as exc:
                    self.visualization_reason = f'Visualization initialization failed: {type(exc).__name__}'
                    LOG.warning('%s: %s', self.visualization_reason, exc)
            else:
                self.visualization_reason = 'Policy does not support same-forward visualization callbacks'
                LOG.warning(self.visualization_reason)
        config = getattr(self.policy, "training_config", {})
        versions = {}
        for name in ("fastapi", "uvicorn", "numpy", "torch", "transformers"):
            try:
                versions[name] = version(name)
            except PackageNotFoundError:
                pass
        self.metadata = dict(
            api_version=API_VERSION, model_id=self.model_id, model_family="bridgevla",
            checkpoint=str(self.checkpoint), device=str(self.device),
            contract_sha256=self.policy.contract["sha256"],
            mode=self.policy.mode, action_spec=action_descriptor(self.policy.action_config),
            cameras=list(self.policy.contract["data_config"]["cameras"]),
            observation_format=OBSERVATION_FORMAT, observation_schema=self.fields,
            observation_formats=[OBSERVATION_FORMAT, RAW_FORMAT], raw_observation=raw_schema(self.policy),
            conventions=dict(rgb="CHW uint8", depth="metric Z/depth per training contract",
                             point_cloud="world metres, organized CHW XYZ",
                             camera_extrinsics="T_world_from_optical"),
            render_config=config.get("mvt", {}),
            visualization=dict(enabled=self.visualizer is not None, every=self.visualize_every,
                               keep=self.visualize_keep, url='/visualization',
                               output_dir=str(self.visualizer.root) if self.visualizer is not None else None,
                               reason=self.visualization_reason),
            created_at=datetime.now(RUN_TIMEZONE).isoformat(), timezone="Asia/Shanghai",
            versions=versions, request_id_history=4096,
        )
        write_json(self.output_dir / "info.json", self.metadata)
        if getattr(self.policy, 'chunk_settings', None) is not None:
            from .chunk_protocol import chunk_spec
            self.metadata['chunk_spec'] = chunk_spec(self.policy.chunk_settings, self.policy.contract)
            write_json(self.output_dir / 'info.json', self.metadata)
        with self.lock:
            self.ready = True
        LOG.info("BridgeVLA ready: checkpoint=%s model_id=%s", self.checkpoint, self.model_id)

    def _warmup(self):
        replay, split, index = self.warmup
        if replay is None:
            return
        from ..data.dataset import OHTDataset
        from .predicted_wrapper import policy_observation
        if getattr(self.policy, 'chunk_settings', None) is not None:
            from ..data.chunk_dataset import OHTChunkDataset
            dataset = OHTChunkDataset(replay, split)
        else:
            dataset = OHTDataset(replay, split, action_config=self.policy.action_config)
        if dataset.contract["sha256"] != self.policy.contract["sha256"]:
            raise ValueError("Warmup replay differs from the checkpoint data contract")
        sample = dataset[index]
        current = policy_observation(sample, self.policy.contract["data_config"]["cameras"],
                                     self.policy.action_config)
        if getattr(self.policy, 'chunk_settings', None) is not None:
            current.update({name: sample[name] for name in ('tcp_pose', 'joint_positions', 'gripper_open01')})
        self.policy.act(current, sample["goal"], 0)
        self.policy.reset()

    def health(self):
        with self.lock:
            return dict(ready=self.ready and not self.stopping, busy=self.busy,
                        model_id=self.model_id, episode_id=self.episode, next_step=self.step,
                        episode_failed=self.failed, error=self.error)

    def model_info(self):
        if not self.health()["ready"]:
            raise APIError(503, "not_ready", "Model is not ready")
        return copy.deepcopy(self.metadata)

    def visualization_state(self):
        if self.visualizer is None:
            return dict(enabled=False, every=self.visualize_every, latest=None, writer_status='disabled',
                        reason=self.visualization_reason)
        result = self.visualizer.latest()
        result['capture_error'] = self.visualization_capture_error
        return result

    def check_available(self):
        with self.lock:
            if not self.ready or self.stopping:
                raise APIError(503, "not_ready", "Model is not ready")
            if self.busy:
                raise APIError(429, "busy", "Inference/reset is in progress; requests are not queued")

    def log_rejection(self, operation, error):
        if self.logger is not None and not self.stopping:
            self.logger.info(json.dumps(dict(created_at=datetime.now(RUN_TIMEZONE).isoformat(),
                                            operation=operation, model_id=self.model_id,
                                            status=error.status, code=error.code)))

    async def execute(self, operation, raw, started=None):
        with self.lock:
            if not self.ready or self.stopping:
                raise APIError(503, "not_ready", "Model is not ready")
            if self.busy:
                raise APIError(429, "busy", "Inference/reset is in progress; requests are not queued")
            self.busy = True
            try:
                future = self.executor.submit(self._run, operation, raw, started or perf_counter())
            except BaseException:
                self.busy = False
                raise
        return await self._wait(future)

    def _run(self, operation, raw, started):
        payload, response, status, code = {}, None, 500, "inference_failed"
        try:
            stamp = perf_counter()
            payload = decode_request(raw, operation)
            decode_ms = (perf_counter() - stamp)*1000
            if payload["model_id"] != self.model_id or payload["contract_sha256"] != self.policy.contract["sha256"]:
                raise APIError(409, "model_mismatch", "model_id/contract_sha256 do not match /model_info")
            if payload["request_id"] in self.request_ids:
                raise APIError(409, "duplicate_request", "request_id was already processed")
            response = self._predict(payload, started, decode_ms) if operation == "predict" else self._reset(payload)
            status, code = 200, "ok"
            return response
        except APIError as exc:
            status, code = exc.status, exc.code
            exc.logged = True
            raise
        except Exception as exc:
            with self.lock:
                self.failed, self.error = True, "inference_failed"
            LOG.exception("OHT %s failed", operation)
            error = APIError(500, "inference_failed", "Policy operation failed; use /reset with a new episode ID")
            error.logged = True
            raise error from exc
        finally:
            try:
                row = dict(created_at=datetime.now(RUN_TIMEZONE).isoformat(), operation=operation,
                           model_id=self.model_id, status=status, code=code,
                           **{key: payload.get(key) for key in ("request_id", "episode_id", "step")},
                           total_ms=(perf_counter() - started)*1000)
                if response is not None:
                    row["timing_ms"] = response.get("timing_ms", {})
                    if "action" in response:
                        row["action_summary"] = dict(length=len(response["action"]),
                                                     minimum=min(response["action"]), maximum=max(response["action"]))
                    if 'chunk_spec' in response:
                        row['chunk_summary'] = dict(horizon=len(response['actions']),
                                                    action_dt=response['chunk_spec']['action_dt'])
                    if 'visualization' in response:
                        row['visualization'] = response['visualization']
                self.logger.info(json.dumps(row, allow_nan=False))
            finally:
                # Only the worker releases admission, never the HTTP waiter.
                with self.lock:
                    self.busy = False

    def _remember(self, request_id):
        self.request_ids.add(request_id)
        self.recent_ids.append(request_id)
        if len(self.recent_ids) > 4096:
            self.request_ids.remove(self.recent_ids.popleft())

    def _predict(self, request, started, decode_ms):
        expected_schema = self.metadata.get('chunk_spec', {}).get('schema')
        if request.get('prediction_schema') != expected_schema:
            raise APIError(422, 'prediction_schema_mismatch', 'Acknowledge the advertised chunk schema before predicting')
        episode = request["episode_id"]
        if self.failed:
            raise APIError(409, "episode_failed", "Use /reset with a new episode ID after a failed inference")
        if self.episode is None:
            if request["step"] != 0 or episode in self.episodes:
                raise APIError(409, "episode_conflict", "A fresh episode must start at step zero")
        elif episode != self.episode:
            raise APIError(409, "episode_conflict", "Use /reset before switching episodes")
        if (request["step"] != self.step or request["timestamp"] <= self.timestamp
                or (self.instruction is not None and request["instruction"] != self.instruction)):
            raise APIError(409, "stale_request", "Expected the next step, a newer timestamp and the same instruction")
        stamp = perf_counter()
        raw = request['observation_format'] == RAW_FORMAT
        due = self.visualizer is not None and self.visualizer.due(request['step'])
        sensor_observation = None
        def capture_sensors(values, config, tcp):
            nonlocal sensor_observation
            from .sensor_visualization import raw_sensor_snapshot
            sensor_observation = raw_sensor_snapshot(values, config, tcp)
        pending = None
        if raw:
            observation, pending = decode_raw(request['observation'], self.policy, self.gripper_history,
                                              sensor_callback=capture_sensors if due else None)
            identity = (RAW_FORMAT, request['observation']['task'], request['observation']['data_profile'])
        else:
            observation = decode_prepared(request["observation"], self.policy, self.fields)
            identity = (OBSERVATION_FORMAT,)
        if due and sensor_observation is None:
            try:
                from .sensor_visualization import prepared_sensor_snapshot
                config = (data_profiles(self.policy)[request['observation']['data_profile']] if raw else
                          self.policy.contract['data_config'])
                sensor_observation = prepared_sensor_snapshot(observation, config)
            except Exception:
                LOG.warning('Prepared sensor visualization snapshot skipped', exc_info=True)
        if self.input_identity is not None and identity != self.input_identity:
            raise APIError(409, 'observation_conflict', 'Use /reset before changing format, task or data profile')
        prepare_ms = (perf_counter() - stamp)*1000
        self._remember(request["request_id"])
        stamp = perf_counter()
        if self.episode is None:
            self.policy.reset()
            with self.lock:
                self.episode = episode
                self.episodes.add(episode)
        captured, capture_errors = [], []
        def capture(payload):
            captured.append(payload)
        # Opt in to nonfatal diagnostic construction errors; no policy retry.
        capture.on_error = lambda exc: capture_errors.append(type(exc).__name__)
        extra = {'visualization_callback': capture} if due else {}
        prediction = self.policy.act(observation, request['instruction'], request['step'], **extra)
        chunk = self.metadata.get('chunk_spec')
        if chunk is not None:
            from .chunk_protocol import validate_prediction
            prediction = validate_prediction(prediction, chunk)
            action = prediction['actions'][0]
        else:
            action = validate_action(prediction, self.policy.action_config)
        policy_ms = (perf_counter() - stamp)*1000
        visualization = None
        if due:
            try:
                if capture_errors or len(captured) != 1:
                    raise ValueError('Diagnostic capture failed: ' + (', '.join(capture_errors) or 'expected one payload'))
                visualization = self.visualizer.submit(captured[0], request, action,
                                                        self.policy.action_config, observation,
                                                        sensor_observation=sensor_observation)
                self.visualization_capture_error = None
            except Exception as exc:
                LOG.warning('Online visualization skipped: %s', exc)
                self.visualization_capture_error = str(exc)
                visualization = dict(status='error', error='Diagnostic capture failed', url='/visualization')
        with self.lock:
            self.step += 1
            self.timestamp, self.instruction = request["timestamp"], request["instruction"]
            self.gripper_history, self.input_identity = pending, identity
        response = dict(api_version=API_VERSION, model_id=self.model_id,
                    contract_sha256=self.policy.contract["sha256"],
                    **{key: request[key] for key in ("request_id", "episode_id", "step", "timestamp")},
                    action=action.tolist(), action_spec=action_descriptor(self.policy.action_config),
                    timing_ms=dict(decode=decode_ms, prepare=prepare_ms, policy=policy_ms,
                                   total=(perf_counter() - started)*1000))
        if chunk is not None:
            response.pop('action')
            response.update(chunk_spec=chunk, key_action=prediction['key_action'].tolist(),
                            actions=prediction['actions'].tolist(), time_offsets=prediction['time_offsets'].tolist())
            if visualization is not None:
                # The first executed target is used by existing heatmap/sensor diagnostics.
                response['chunk_visualization'] = dict(key_action=response['key_action'], actions=response['actions'])
        if visualization is not None:
            response['visualization'] = visualization
        return response

    def _reset(self, request):
        episode = request["episode_id"]
        if episode in self.episodes:
            raise APIError(409, "episode_conflict", "Reset requires a fresh episode ID")
        self._remember(request["request_id"])
        self.policy.reset()
        with self.lock:
            self.episodes.add(episode)
            self.episode, self.step, self.timestamp, self.instruction = episode, 0, -1., None
            self.failed, self.error = False, None
            self.gripper_history, self.input_identity = None, None
        return dict(api_version=API_VERSION, model_id=self.model_id,
                    contract_sha256=self.policy.contract["sha256"],
                    request_id=request["request_id"], episode_id=episode, next_step=0, reset=True)

    async def close(self):
        with self.lock:
            self.stopping = True
        try:
            await self._wait(self.executor.submit(self._cleanup))
        finally:
            self.executor.shutdown(wait=False)

    def _cleanup(self):
        try:
            if self.policy is not None:
                self.policy.reset()
        finally:
            if self.visualizer is not None:
                self.visualizer.close()
            self.policy = None
            if self.log_handler is not None:
                self.logger.removeHandler(self.log_handler)
                self.log_handler.close()
            with self.lock:
                self.ready = False
