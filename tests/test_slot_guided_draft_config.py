"""CPU-only config/entrypoint checks; do not enable the external geometry path."""
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'finetune'))
from bridgevla.config import get_cfg_defaults
from bridgevla.mvt.config import get_cfg_defaults as get_mvt_defaults


def profile(name):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(ROOT / 'finetune/RLBench/configs' / name))
    return cfg


def test_new_profile_is_disabled_and_matches_existing_profile_except_name():
    cfg = profile('rlbench_slot_guided_sam3_draft.yaml')
    original = profile('rlbench_o2_role_queries.yaml')
    assert cfg.rvt.enable_slot_guided_sam3 is False
    assert get_cfg_defaults().rvt.enable_slot_guided_sam3 is False
    assert cfg.exp_id == 'rlbench_slot_guided_sam3_draft'
    cfg.exp_id = original.exp_id
    assert cfg == original


def test_new_profile_selects_supported_existing_anchor_and_two_stages():
    cfg = profile('rlbench_slot_guided_sam3_draft.yaml')
    mvt = get_mvt_defaults()
    mvt.merge_from_file(str(ROOT / 'finetune/bridgevla/mvt/configs/rvt2.yaml'))
    assert mvt.stage_two is True
    assert cfg.object_slots.enabled is True
    assert cfg.object_slots.predictor_type == 'role_queries'
    assert cfg.object_slots.num_slots == 2
    assert cfg.object_conditioning.shared_action_features is True
    assert cfg.object_conditioning.inherit_coarse_roles is True
    assert cfg.object_conditioning.preserve_role_tokens is True
    assert cfg.oracle_relation_gated_adapter is True
    assert cfg.oracle_prior_adapter_rank > 0 and cfg.oracle_relation_anchor_rank > 0
    assert cfg.rvt.object_prior_mode == 'o2_internal_slots'


def test_existing_eval_constructor_receives_same_model_args_for_new_profile():
    tree = ast.parse((ROOT / 'finetune/RLBench/eval.py').read_text(encoding='utf-8'))
    constructor = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Name) and node.func.id == 'MVT')
    kwargs = [kw for kw in constructor.keywords if kw.arg and 'exp_cfg' in ast.unparse(kw.value)]
    assert any(kw.arg == 'object_slot_predictor_type' for kw in kwargs)
    for kw in kwargs:
        expression = compile(ast.Expression(kw.value), '<actual_eval_model_argument>', 'eval')
        original = eval(expression, {'exp_cfg': profile('rlbench_o2_role_queries.yaml')})
        draft = eval(expression, {'exp_cfg': profile('rlbench_slot_guided_sam3_draft.yaml')})
        assert draft == original, kw.arg


def test_flag_reaches_existing_agent_through_rvt_kwargs():
    for entrypoint in ('train.py', 'eval.py'):
        tree = ast.parse((ROOT / 'finetune/RLBench' / entrypoint).read_text(encoding='utf-8'))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'RVTAgent']
        assert len(calls) == 1
        assert any(kw.arg is None and ast.unparse(kw.value) == 'exp_cfg.rvt'
                   for kw in calls[0].keywords)


def test_agent_constructor_default_is_false():
    tree = ast.parse((ROOT / 'finetune/bridgevla/models/bridgevla_agent.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RVTAgent')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    defaults = dict(zip((arg.arg for arg in method.args.args[-len(method.args.defaults):]),
                        method.args.defaults))
    assert ast.literal_eval(defaults['enable_slot_guided_sam3']) is False
