"""Current simulator observation binding, with no RLBench install or model weights."""
import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'finetune'))
MODULE_PATH = ROOT / 'finetune/RLBench/utils/slot_sam3_observation.py'
SPEC = importlib.util.spec_from_file_location('slot_sam3_observation_under_test', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
SlotSam3ObservationProvider = MODULE.SlotSam3ObservationProvider


@pytest.fixture
def task_repo():
    path = Path(os.environ.get('TASK_RELEVANT_OBJECT_REPO', ROOT.parent / 'TASK_Relevant_object-main')).resolve()
    if not (path / 'slot_guidance_producers.py').is_file():
        pytest.skip('set TASK_RELEVANT_OBJECT_REPO for current-frame TASK interoperability')
    return path


def raw_observation(wrist_translation=.2):
    obs = SimpleNamespace(misc={})
    extracted = {'low_dim_state': np.array([1., .03, .03, .95], dtype=np.float32)}
    for name, x in (('front', 0.), ('wrist', wrist_translation)):
        h, w = 8, 10
        k = np.array([[-80., 0., 4.5], [0., -90., 3.5], [0., 0., 1.]])
        angle = .3 if name == 'wrist' else 0.
        c, s = np.cos(angle), np.sin(angle)
        t = np.array([[c, -s, 0., x], [s, c, 0., -.1], [0., 0., 1., .6], [0., 0., 0., 1.]])
        yy, xx = np.indices((h, w))
        z = .8 + .02 * xx + .01 * yy
        camera_xyz = np.stack(((xx - k[0, 2]) * z / k[0, 0],
                               (yy - k[1, 2]) * z / k[1, 1], z), -1)
        world_xyz = camera_xyz @ t[:3, :3].T + t[:3, 3]
        world_xyz[0, 0] = np.nan
        rgb = np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)
        setattr(obs, f'{name}_rgb', rgb)
        setattr(obs, f'{name}_point_cloud', world_xyz)
        obs.misc[f'{name}_camera_intrinsics'] = k
        obs.misc[f'{name}_camera_extrinsics'] = t
        extracted[f'{name}_rgb'] = np.moveaxis(rgb, -1, 0)
        extracted[f'{name}_point_cloud'] = np.moveaxis(world_xyz.astype(np.float32), -1, 0)
        extracted[f'{name}_camera_intrinsics'] = k
        extracted[f'{name}_camera_extrinsics'] = t
    return obs, extracted


def prepped(extracted, instruction='move the red object'):
    # YARR casts float64 metadata to float32 and creates B,T axes.
    return {**{key: np.asarray(value, dtype=np.float32)[None, None] for key, value in extracted.items()},
            'language_goal': [[[instruction]]]}


@pytest.fixture
def current(task_repo):
    clock = [20.5, .05]
    provider = SlotSam3ObservationProvider(task_repo, ['front', 'wrist'], time_source=lambda: tuple(clock))
    raw, extracted = raw_observation()
    provider.capture_raw(raw, extracted, task_name='pick_up_cup', instruction='move the red object')
    return provider, clock, raw, extracted


def test_negative_pyrep_focals_convert_by_proper_rotation_preserving_rgb_and_world(current):
    provider, _, raw, extracted = current
    packet = provider.capture(prepped(extracted))
    api = provider._load_api()
    assert packet.receipt['source_frame_index'] == 0
    assert packet.receipt['source_frame_index_kind'] == 'live_simulation_microstep'
    assert packet.receipt['depth_source'] == 'current_observation_world_point_cloud_to_camera_z'
    assert packet.receipt['snapshot_input_sha256'] == api.slot_guided_io.snapshot_input_digest(packet.snapshot)
    for name, frame in packet.snapshot.cameras.items():
        np.testing.assert_array_equal(frame.rgb, getattr(raw, f'{name}_rgb'))
        assert frame.camera.intrinsics[0, 0] > 0 and frame.camera.intrinsics[1, 1] > 0
        assert np.linalg.det(frame.camera.extrinsics[:3, :3]) == pytest.approx(1.)
        d = np.asarray(packet.receipt['raw_camera_from_optical'][name])
        np.testing.assert_array_equal(d, np.diag([-1., -1., 1., 1.]))
        np.testing.assert_allclose(frame.camera.extrinsics, raw.misc[f'{name}_camera_extrinsics'] @ d)
        valid = np.isfinite(frame.depth_m) & (frame.depth_m > 0)
        restored = api.camera_geometry.transform_points(api.camera_geometry.backproject_mask(
            frame.depth_m, valid, frame.camera.intrinsics, max_points=0), frame.camera.extrinsics)
        np.testing.assert_allclose(restored, getattr(raw, f'{name}_point_cloud')[valid], rtol=1e-10, atol=1e-10)
        assert np.isnan(frame.depth_m[0, 0])


def test_dynamic_wrist_uses_current_microstep_pose_not_action_or_demo_index(current):
    provider, clock, _, _ = current
    first = provider._packet
    clock[0] += 17 * clock[1]  # One action can execute 17 simulator microsteps.
    raw, extracted = raw_observation(wrist_translation=.7)
    provider.capture_raw(raw, extracted, task_name='pick_up_cup', instruction='move the red object')
    packet = provider.capture(prepped(extracted))
    assert packet.receipt['source_frame_index'] == 17
    assert packet.receipt['max_source_frame_index_read'] == 17
    assert packet.snapshot.cameras['wrist'].camera.extrinsics[0, 3] == pytest.approx(.7)
    assert packet.receipt['calibration_sha256'] != first.receipt['calibration_sha256']
    assert packet.snapshot.metadata['episode_id'].startswith('live:')


@pytest.mark.parametrize('field', ['front_rgb', 'wrist_point_cloud', 'front_camera_intrinsics',
                                  'wrist_camera_extrinsics', 'low_dim_state'])
def test_stale_policy_field_rejected_even_when_all_other_fields_match(current, field):
    provider, _, _, extracted = current
    value = prepped(extracted)
    value[field] = value[field].copy()
    value[field].flat[-1] += .01
    with pytest.raises(ValueError, match='do not match'):
        provider.capture(value)


def test_latest_timestep_and_nested_language_are_bound(current):
    provider, _, _, extracted = current
    value = prepped(extracted)
    for key in extracted:
        old = np.full_like(value[key], -100.)
        value[key] = np.concatenate((old, value[key]), 1)
    assert provider.capture(value).instruction == 'move the red object'
    value['language_goal'] = [[['another instruction']]]
    with pytest.raises(ValueError, match='instruction'):
        provider.capture(value)


def test_simulation_advance_without_new_raw_capture_rejects_stale_cache(current):
    provider, clock, _, extracted = current
    clock[0] += .05
    with pytest.raises(ValueError, match='advanced'):
        provider.capture(prepped(extracted))


def test_episode_reset_uses_new_identity_and_restarts_only_microstep_origin(current):
    provider, clock, _, extracted = current
    first = provider.capture(prepped(extracted))
    provider.begin_episode('explicit-runtime-label')
    with pytest.raises(ValueError, match='no current observation'):
        provider.capture(prepped(extracted))
    clock[0] += 3.
    raw, extracted = raw_observation()
    provider.capture_raw(raw, extracted, task_name='pick_up_cup', instruction='move the red object',
                         episode_id='explicit-runtime-label')
    second = provider.capture(prepped(extracted))
    assert second.receipt['source_frame_index'] == 0
    assert second.receipt['episode_id'] != first.receipt['episode_id']


@pytest.mark.parametrize('kind', ['missing_K', 'reflected_K', 'wrong_T', 'singular_T'])
def test_bad_or_missing_calibration_invalidates_cache_and_never_guesses(current, kind):
    provider, _, raw, extracted = current
    if kind == 'missing_K':
        del raw.misc['wrist_camera_intrinsics']
    elif kind == 'reflected_K':
        raw.misc['wrist_camera_intrinsics'] = raw.misc['wrist_camera_intrinsics'].copy()
        raw.misc['wrist_camera_intrinsics'][0, 0] *= -1
    elif kind == 'wrong_T':
        raw.misc['wrist_camera_extrinsics'] = raw.misc['wrist_camera_extrinsics'].copy()
        raw.misc['wrist_camera_extrinsics'][0, 3] += .1
    else:
        raw.misc['wrist_camera_extrinsics'] = np.zeros((4, 4))
    with pytest.raises(ValueError):
        provider.capture_raw(raw, extracted, task_name='pick_up_cup', instruction='move the red object')
    with pytest.raises(ValueError, match='unavailable'):
        provider.capture(prepped(extracted))


def test_invalid_depth_and_behind_camera_points_stay_invalid(current):
    provider, _, raw, extracted = current
    t = raw.misc['front_camera_extrinsics']
    raw.front_point_cloud[1, 1] = np.array([0., 0., -1.]) @ t[:3, :3].T + t[:3, 3]
    extracted['front_point_cloud'] = np.moveaxis(raw.front_point_cloud.astype(np.float32), -1, 0)
    provider.capture_raw(raw, extracted, task_name='pick_up_cup', instruction='move the red object')
    frame = provider.capture(prepped(extracted)).snapshot.cameras['front']
    assert np.isnan(frame.depth_m[0, 0]) and np.isnan(frame.depth_m[1, 1])


def isolated_env(base):
    path = ROOT / 'finetune/RLBench/utils/custom_rlbench_env.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'CustomMultiTaskRLBenchEnv2')
    scope = {'CustomMultiTaskRLBenchEnv': base, 'np': np, 'change_case': lambda name: name}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), scope)
    return scope[cls.name]


class BaseEnv:
    def __init__(self, *args, **kwargs):
        self.output = {}
        self._episode_index = 0
        self.eval, self._record_every_n = False, 0
        self._lang_goal = 'current instruction'
        self._task = SimpleNamespace(_task=SimpleNamespace())

    def extract_obs(self, raw, *args, **kwargs):
        return self.output

    def reset(self):
        self._previous_obs_dict = self.extract_obs(object())


def test_disabled_env_preserves_exact_return_and_has_no_provider_import_or_copy():
    env = isolated_env(BaseEnv)()
    raw = object()  # No accessible arrays, misc, demo or clock.
    result = env.extract_obs(raw)
    assert result is env.output
    assert env.slot_sam3_observation_provider is None


def test_enabled_env_capture_failure_preserves_baseline_and_records_failure():
    events = []
    def fail(*args, **kwargs):
        events.append(('capture', args, kwargs))
        raise ValueError('bad current calibration')
    provider = SimpleNamespace(capture_raw=fail, invalidate=lambda error: events.append(('failure', str(error))))
    env = isolated_env(BaseEnv)(slot_sam3_observation_provider=provider)
    raw = object()
    assert env.extract_obs(raw) is env.output
    assert events[0][1] == (raw, env.output)
    assert events[-1] == ('failure', 'bad current calibration')


def test_env_begins_episode_before_capturing_reset_observation():
    events = []
    provider = SimpleNamespace(begin_episode=lambda: events.append('begin'),
                               capture_raw=lambda *args, **kwargs: events.append('capture'),
                               invalidate=lambda error: events.append('failure'))
    env = isolated_env(BaseEnv)(slot_sam3_observation_provider=provider)
    assert env.reset() is env.output
    assert events == ['begin', 'capture']


def test_float32_general_camera_rotations_keep_world_projection():
    # Actual PyRep returns rounded C float matrices, not ideal float64 rotations.
    rng = np.random.default_rng(17)
    k = np.array([[-100., 0., 16.], [0., -100., 16.], [0., 0., 1.]])
    probes = np.array([[.1, .2, 1.], [-.1, .05, 1.3]])
    for quaternion in rng.normal(size=(400, 4)):
        x, y, z, w = quaternion / np.linalg.norm(quaternion)
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                             [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                             [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], dtype=np.float32)
        raw = np.eye(4)
        raw[:3, :3], raw[:3, 3] = rotation, [.3, -.2, .7]
        positive_k, optical, basis = MODULE.optical_calibration(k, raw)
        assert np.allclose(optical[:3, :3].T @ optical[:3, :3], np.eye(3), rtol=0, atol=1e-7)
        world_raw = probes @ raw[:3, :3].T + raw[:3, 3]
        world_optical = (probes @ basis[:3, :3]) @ optical[:3, :3].T + optical[:3, 3]
        np.testing.assert_allclose(world_optical, world_raw, rtol=0, atol=1e-6)
        assert positive_k[0, 0] > 0 and positive_k[1, 1] > 0
