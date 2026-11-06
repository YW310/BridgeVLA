"""Actual online config, launcher, factory and direct-eval wiring contracts."""
import argparse
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'finetune'))
from bridgevla.config import get_cfg_defaults
from bridgevla.online import slot_sam3_runtime as runtime


def test_enabled_online_profile_preserves_original_architecture_and_default_off():
    profile = runtime.load_online_profile(runtime.PROFILE)
    assert profile.rvt.enable_slot_guided_sam3
    assert profile.rvt.slot_guided_sam3_execution_mode == 'online_experiment'
    assert not get_cfg_defaults().rvt.enable_slot_guided_sam3
    assert profile.slot_guided_sam3.prompt_mode == 'point+box'
    settings = runtime.resolve_runtime(profile)
    assert Path(settings['task_repo']) == ROOT.parent / 'TASK_Relevant_object-main'
    assert settings['diagnostics_dir'] is None
    with pytest.raises(ValueError, match='online launcher'):
        runtime.load_online_profile(runtime.PROFILE.with_name('rlbench_slot_guided_sam3_offline.yaml'))


def test_runtime_overrides_and_config_check_do_not_load_models(tmp_path, monkeypatch, capsys):
    cfg = runtime.load_online_profile(runtime.PROFILE)
    settings = runtime.resolve_runtime(cfg, {'sam3_checkpoint': str(tmp_path / 'sam3.pt'),
        'prompt_mode': 'box-only', 'fallback': 'fail'}, policy_device='cuda:3')
    assert settings['sam3_device'] == 'cuda:3'
    assert settings['sam3_checkpoint'] == str(tmp_path / 'sam3.pt')
    monkeypatch.setattr(runtime.runpy, 'run_path', lambda *a, **k: pytest.fail('must not start eval'))
    assert runtime.main(['--check-config', '--slot-sam3-prompt-mode', 'box-only',
                         '--slot-sam3-checkpoint', str(tmp_path / 'selected.pt')]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['execution_mode'] == 'online_experiment'
    assert report['settings']['prompt_mode'] == 'box-only'
    assert report['settings']['sam3_checkpoint'] == str(tmp_path / 'selected.pt')


def test_online_launcher_forwards_real_eval_options_and_restores_argv(monkeypatch):
    previous, observed = sys.argv, []
    monkeypatch.setattr(runtime.runpy, 'run_path', lambda path, **kwargs: observed.append((path, list(sys.argv), kwargs)))
    assert runtime.main(['--model-folder', 'weights', '--model-name', 'model.pth', '--tasks', 'open_drawer',
                         '--slot-sam3-checkpoint', 'sam3.pt']) == 0
    path, arguments, options = observed[0]
    assert path == str(ROOT / 'finetune/RLBench/eval.py') and options['run_name'] == '__main__'
    assert arguments[1:3] == ['--exp_cfg_path', str(runtime.PROFILE)]
    assert '--tasks' in arguments and '--slot-sam3-checkpoint' in arguments
    assert '--log-name' in arguments and 'slot_sam3_online/' in arguments[-1]
    assert sys.argv is previous


def test_second_profile_cannot_silently_disable_online(monkeypatch, capsys):
    monkeypatch.setattr(runtime.runpy, 'run_path', lambda *a, **k: pytest.fail('must reject second config'))
    assert runtime.main(['--exp_cfg_path', 'another.yaml']) == 2
    assert 'second --exp_cfg_path' in capsys.readouterr().err


def test_factory_preloads_one_backend_and_uses_same_observer(monkeypatch):
    import bridgevla.online.slot_sam3_perception as perception_module
    events = []
    class PreparedPerception:
        def __init__(self, *args):
            events.append(('init', args))
        def prepare_model(self):
            events.append(('prepare',))
            return 3.0
    monkeypatch.setattr(perception_module, 'OnlineTaskPerception', PreparedPerception)
    agent = SimpleNamespace(_training=False, slot_guided_sam3_execution_mode='online_experiment',
        slot_guided_sam3_runtime=runtime.resolve_runtime(runtime.load_online_profile(runtime.PROFILE)),
        slot_guided_sam3_checkpoint_path='actual-model.pth', slot_guided_sam3_git_sha='b' * 40,
        slot_guided_sam3_checkpoint_sha256='c' * 64, slot_guided_sam3_point_count=128)
    observer = object()
    wrapper = runtime.make_online_policy(agent, observer)
    assert wrapper.agent is agent and wrapper.snapshot_provider is observer
    assert len([e for e in events if e[0] == 'prepare']) == 1
    assert wrapper.expected_checkpoint_sha256 == 'c' * 64
    assert agent.slot_guided_sam3_model_load_ms == 3.0


def eval_tree():
    tree = ast.parse((ROOT / 'finetune/RLBench/eval.py').read_text(encoding='utf-8'))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'eval')


@pytest.mark.parametrize('extra', [{'replay_ground_truth': True}, {'oracle_provider_name': 'rlbench_gt'},
                                  {'bridgevla_aligned_objects': True}, {'eval_resume': True}])
def test_actual_eval_guards_run_before_environment_launch(extra):
    nodes = eval_tree().body[:3]
    code = compile(ast.Module(body=nodes, type_ignores=[]), '<actual_online_eval_guards>', 'exec')
    scope = dict(agent=SimpleNamespace(enable_slot_guided_sam3=True,
        slot_guided_sam3_execution_mode='online_experiment'), replay_ground_truth=False,
        oracle_provider_name='none', heatmap_action_anchor=False, bridgevla_aligned_objects=False,
        bridgevla_aligned_reference=False, eval_resume=False)
    scope.update(extra)
    with pytest.raises(ValueError):
        exec(code, scope)
    scope['agent'] = SimpleNamespace(enable_slot_guided_sam3=False)
    exec(code, scope)


def test_eval_injects_one_observer_into_the_environment_and_policy_before_launch(monkeypatch):
    import bridgevla.online.slot_sam3_runtime as runtime_module
    events, provider = [], object()
    fake_provider_module = SimpleNamespace(SlotSam3ObservationProvider=lambda *a: provider)
    monkeypatch.setitem(sys.modules, 'utils.slot_sam3_observation', fake_provider_module)
    monkeypatch.setattr(runtime_module, 'make_online_policy', lambda agent, obs: events.append(('wrap', agent, obs)) or agent)
    class Environment:
        def __init__(self, **kwargs):
            events.append(('env', kwargs))
    function = eval_tree()
    start = next(i for i, n in enumerate(function.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'online_observer' for t in n.targets))
    end = next(i for i, n in enumerate(function.body[start:], start) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'device' for t in n.targets))
    code = compile(ast.Module(body=function.body[start:end], type_ignores=[]), '<actual_online_env_wiring>', 'exec')
    agent = SimpleNamespace(slot_guided_sam3_runtime={'task_repo': 'TASK'}, cameras=['front'])
    scope = dict(online_slot_sam3=True, agent=agent, CustomMultiTaskRLBenchEnv=Environment,
        task_classes=[], obs_config=object(), action_mode=object(), eval_datafolder='data',
        episode_length=25, headless=True, eval_episodes=25, save_video=False,
        oracle_provider=None, logging=False, log_dir=None)
    exec(code, scope)
    assert events[0][0] == 'env' and events[0][1]['slot_sam3_observation_provider'] is provider
    assert events[1] == ('wrap', agent, provider)
    events.clear()
    scope.update(online_slot_sam3=False, agent=object())
    exec(code, scope)
    assert len(events) == 1 and 'slot_sam3_observation_provider' not in events[0][1]


def test_real_eval_parser_accepts_online_paths_without_enabling_other_modes():
    tree = ast.parse((ROOT / 'finetune/bridgevla/utils/rvt_utils.py').read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_eval_parser')
    scope = {'argparse': argparse}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<actual_eval_parser>', 'exec'), scope)
    args = scope['get_eval_parser']().parse_args(['--slot-sam3-task-repo', 'TASK',
                                                '--slot-sam3-prompt-mode', 'box-only'])
    assert args.slot_sam3_task_repo == 'TASK' and args.slot_sam3_prompt_mode == 'box-only'
    assert args.oracle_provider == 'none' and not args.ground_truth


def test_fail_mode_diagnostics_can_be_saved_without_a_transition(tmp_path):
    tree = ast.parse((ROOT / 'finetune/RLBench/eval.py').read_text(encoding='utf-8'))
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_append_slot_sam3_step')
    scope = {'json': json, 'Path': Path}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), '<actual_online_step_log>', 'exec'), scope)
    path = tmp_path / 'steps.jsonl'
    scope[helper.name](path, 'open_drawer', 3, dict(status='failed', failure_reason='current calibration mismatch'))
    saved = json.loads(path.read_text())
    assert saved['status'] == 'failed' and saved['episode_index'] == 3
    assert saved['failure_reason'] == 'current calibration mismatch'
    scope[helper.name](None, 'open_drawer', 3, object())  # disabled performs no payload parsing/I/O
