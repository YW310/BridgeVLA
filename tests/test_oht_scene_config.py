"""Select a complete V4 scene and verify its provenance instead of swapping USDs."""
import copy
from types import ModuleType, SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tests.test_oht_migration import replay_fixture  # noqa: F401
from tests.test_oht_vla_server import fixture  # noqa: F401
from tests.test_oht_isaaclab_http import raw_sample, Opener
from tests.test_oht_native_runtime import environment
from tests.test_oht_standalone_client import native_arguments, sensors_for, fake_simulator
from finetune.OHT import isaaclab_client as runner
from finetune.OHT.runtime import isaaclab_scene as scenes, isaaclab_factory as native
from finetune.OHT.runtime.http_client import BridgeVLAClient


@pytest.fixture
def v4(tmp_path, monkeypatch):
    root = tmp_path / 'IsaacLab'
    usd = root / scenes.V4_USD
    usd.parent.mkdir(parents=True)
    usd.touch()
    (root / 'isaaclab.sh').touch()
    source = root / 'source' / 'task_config.py'
    source.parent.mkdir()
    source.touch()
    prototype = SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(spawn=SimpleNamespace(usd_path=str(usd))), num_envs=9),
                                sim=SimpleNamespace(device='old', use_fabric=True), version='v4',
                                actions=SimpleNamespace(arm_action=SimpleNamespace(body_name='base_link_0',
                                                          controller_cfg=SimpleNamespace(target_types=['pose_rel'])),
                                                        gripper_action=SimpleNamespace(open_command_expr={'fingers': -.015},
                                                                                       close_command_expr={'fingers': .011})))
    class V4Config:
        def __init__(self):
            self.__dict__.update(copy.deepcopy(prototype.__dict__))
    V4Config.__module__ = scenes.V4_MODULE
    module = ModuleType(scenes.V4_MODULE)
    module.__file__ = str(source)
    module.UR16eDualEEAllSceneOSCReachEnvCfgV4 = V4Config
    original_import = scenes.importlib.import_module
    monkeypatch.setattr(scenes.importlib, 'import_module',
                        lambda name: module if name == scenes.V4_MODULE else original_import(name))
    args = native_arguments(scene_preset='oht-v4', isaaclab_root=str(root))
    return SimpleNamespace(root=root, usd=usd, module=module, prototype=prototype, args=args)


def test_v4_loads_full_config_and_verifies_asset(v4):
    v4.prototype.terminations = SimpleNamespace(time_out=object(), success=object())
    expected = scenes.expected_asset(v4.args, 'eef')
    cfg, info = scenes.scene_configuration(v4.args, lambda *a, **k: pytest.fail('Fell back to registry'), 'cpu', expected)
    assert info['asset_verified'] is True and info['preset'] == 'oht-v4'
    assert info['config'] == scenes.V4_CONFIG and info['resolved_robot_usd'] == str(v4.usd)
    assert info['config_source'] == v4.module.__file__
    assert cfg.scene.num_envs == 1 and cfg.sim.device == 'cpu'
    assert cfg.actions.gripper_action.open_command_expr == {'fingers': -.015}
    assert cfg.actions.gripper_action.close_command_expr == {'fingers': .011}
    assert cfg.terminations.time_out is None and cfg.terminations.success is not None
    assert info['timeout_disabled'] is True


def test_registry_scene_disables_only_default_timeout():
    timeout, success = object(), object()
    cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=1), sim=SimpleNamespace(device='cpu'),
                          terminations=SimpleNamespace(time_out=timeout, success=success))
    selected, info = scenes.scene_configuration(native_arguments(), lambda *a, **k: cfg, 'cpu', None)
    assert selected is cfg and cfg.terminations.time_out is None and cfg.terminations.success is success
    assert info['timeout_disabled'] is True


@pytest.mark.parametrize('failure', ['wrong_usd', 'wrong_source', 'missing_class', 'wrong_body', 'wrong_controller'])
def test_v4_rejects_mismatched_scene_without_fallback(v4, failure):
    if failure == 'wrong_usd':
        v4.prototype.scene.robot.spawn.usd_path = str(v4.root / 'assets_v2' / 'all.usd')
    elif failure == 'wrong_source':
        v4.module.__file__ = str(v4.root.parent / 'another_IsaacLab' / 'task_config.py')
    elif failure == 'missing_class':
        del v4.module.UR16eDualEEAllSceneOSCReachEnvCfgV4
    elif failure == 'wrong_body':
        v4.prototype.actions.arm_action.body_name = 'old_gripper_link'
    else:
        v4.prototype.actions.arm_action.controller_cfg.target_types = ['pose_abs']
    expected = scenes.expected_asset(v4.args, 'eef')
    with pytest.raises(ValueError):
        scenes.scene_configuration(v4.args, lambda *a, **k: pytest.fail('Silent registry fallback'), 'cpu', expected)


def test_missing_v4_asset_fails_before_starting_simulator(v4, monkeypatch):
    v4.usd.unlink()
    monkeypatch.setattr(native, 'launch_application', lambda *args: pytest.fail('Started Isaac Sim with missing asset'))
    with pytest.raises(FileNotFoundError, match='no fallback'):
        native.create_runtime(args=v4.args, model_info={'action_spec': {'mode': 'eef'}})


@pytest.mark.parametrize('conflict', ['joint', 'env_config', 'expected_usd'])
def test_v4_preset_has_explicit_controller_and_asset_contract(v4, conflict):
    mode = 'eef'
    if conflict == 'joint':
        mode = 'relative_joint'
    elif conflict == 'env_config':
        v4.args.env_config = 'old:Config'
    else:
        v4.args.expected_robot_usd = str(v4.root / 'other.usd')
    with pytest.raises(ValueError):
        scenes.expected_asset(v4.args, mode)


def test_custom_asset_assertion_does_not_rewrite_selected_usd(v4, monkeypatch):
    args = native_arguments(env_config='custom:Config', expected_robot_usd=str(v4.usd))
    cfg = copy.deepcopy(v4.prototype)
    cfg.scene.robot.spawn.usd_path = str(v4.root / 'old.usd')
    monkeypatch.setattr(scenes, 'factory', lambda value: lambda: cfg)
    with pytest.raises(ValueError, match='load the matching full environment config'):
        scenes.scene_configuration(args, None, 'cpu', scenes.expected_asset(args, 'eef'))
    assert cfg.scene.robot.spawn.usd_path.endswith('old.usd')


def test_v4_dry_run_records_actual_scene_in_reports(v4, fixture, replay_fixture, tmp_path, monkeypatch):
    import json
    tensors, _, raw = raw_sample(replay_fixture.replay, fixture.policy)
    env = environment('eef', raw['task'])
    env.scene['robot'].body_names = ['base_link_0']
    env.action_manager.get_term('arm_action').cfg.body_name = 'base_link_0'
    env.scene.sensors = sensors_for(tensors, fixture.policy.contract['data_config']['cameras'])
    state = fake_simulator(monkeypatch, env)
    output = tmp_path / 'v4-dry-run'
    with TestClient(fixture.app()) as server:
        monkeypatch.setattr(runner, 'BridgeVLAClient',
                            lambda *args: BridgeVLAClient('http://testserver', opener=Opener(server)))
        assert runner.main(['--env-id', 'OHT-test', '--scene-preset', 'oht-v4', '--isaaclab-root', str(v4.root),
                            '--task', raw['task'], '--dry-run', '--output', str(output), '--headless']) == 0
    metadata = json.loads((output / 'run.json').read_text(encoding='utf-8'))
    report = json.loads((output / 'control_diagnostics.json').read_text(encoding='utf-8'))
    assert metadata['scene'] == report['scene']
    assert metadata['scene']['resolved_robot_usd'] == str(v4.usd)
    assert metadata['scene']['asset_verified'] is True
    assert not state.parse_calls and not env.commands and not fixture.policy.calls
    assert env.closed and state.app_closed
