"""Exercise the real eval config branch without simulator/Torch imports."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]


class _Config:
    def __init__(self, **fields):
        self.__dict__.update(fields)
        self.frozen = False

    def merge_from_file(self, path):
        pass

    def freeze(self):
        self.frozen = True

    def defrost(self):
        self.frozen = False


def _isolated_loader_config_branch(exp_config, mvt_config):
    source = (ROOT / 'finetune/RLBench/eval.py').read_text(encoding='utf-8')
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'load_agent')
    prefix = []
    for node in function.body:
        if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Name)
                and node.value.func.id == 'MVT'):
            break
        prefix.append(node)
    else:
        raise AssertionError('Cannot isolate the eval config branch before MVT')
    function.body = prefix + [ast.Return(value=ast.Name(id='exp_cfg', ctx=ast.Load()))]
    ast.fix_missing_locations(function)
    namespace = {
        'os': os,
        'default_exp_cfg': SimpleNamespace(get_cfg_defaults=lambda: exp_config),
        'default_mvt_cfg': SimpleNamespace(get_cfg_defaults=lambda: mvt_config),
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<eval-config>', 'exec'), namespace)
    return namespace['load_agent']


class EvalPlaceWithMeanTest(unittest.TestCase):
    def test_rvt1_cli_override_and_rvt2_saved_config_for_all_flag_values(self):
        for stage_two in (False, True):
            for configured_mean in (False, True):
                for use_input_mean in (False, True):
                    with self.subTest(stage_two=stage_two,
                                      configured_mean=configured_mean,
                                      use_input_mean=use_input_mean):
                        exp_config = _Config(rvt=_Config(place_with_mean=configured_mean))
                        mvt_config = _Config(stage_two=stage_two)
                        load_config = _isolated_loader_config_branch(exp_config, mvt_config)
                        result = load_config(
                            model_path='unused/model.pth',
                            exp_cfg_path='unused/exp_cfg.yaml',
                            mvt_cfg_path='unused/mvt_cfg.yaml',
                            use_input_place_with_mean=use_input_mean,
                        )
                        expected = (configured_mean if stage_two or use_input_mean else True)
                        self.assertIs(result.rvt.place_with_mean, expected)
                        self.assertTrue(result.frozen)
                        self.assertTrue(mvt_config.frozen)


if __name__ == '__main__':
    unittest.main()
