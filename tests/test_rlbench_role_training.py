"""Exercise RLBench's real train loops without simulator/CUDA imports."""
import ast
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from finetune.RLBench.training_utils import format_training_metrics
from finetune.bridgevla.config import get_cfg_defaults


ROOT = Path(__file__).resolve().parents[1]


def test_small_nonzero_diversity_is_not_formatted_as_zero():
    text = format_training_metrics({'object_refine_diversity_loss': 1e-9,
                                    'object_refine_diversity_supervised_pixels': 2., 'lr': .01})
    assert 'object_refine_diversity_loss=1e-09' in text
    assert 'object_refine_diversity_supervised_pixels=2' in text
    assert 'lr=' not in text


class Progress:
    messages = []

    def __init__(self, values, **kwargs):
        self.values = values

    def __iter__(self):
        return iter(self.values)

    def set_postfix(self, *args, **kwargs):
        pass

    @classmethod
    def write(cls, message):
        cls.messages.append(message)


@pytest.mark.parametrize('accumulation', [1, 2])
def test_rlbench_loops_log_separation_support_and_weights(accumulation, monkeypatch):
    tree = ast.parse((ROOT / 'finetune/RLBench/train.py').read_text(encoding='utf-8'))
    names = {'train', 'train_with_accumulation', '_scalar_metrics', '_first_sample_text'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(torch=torch, defaultdict=defaultdict, nullcontext=nullcontext,
                 Path=Path, np=SimpleNamespace(ndarray=type('Array', (), {})),
                 DDP=type('DDP', (), {}), dist=SimpleNamespace(barrier=lambda: None),
                 tqdm=SimpleNamespace(tqdm=Progress), format_training_metrics=format_training_metrics,
                 visualization_due=lambda *args: False)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<rlbench-train>', 'exec'), scope)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda *args: 0)
    events, calls = [], []
    Progress.messages = []

    def update(**kwargs):
        calls.append(kwargs)
        assert kwargs['replay_sample']['observations'].device.type == 'cpu'
        return dict(total_loss=3., object_slot_diversity_loss=.2,
                    object_refine_diversity_loss=.3,
                    object_slot_diversity_loss_weight=.01,
                    object_refine_diversity_loss_weight=.01,
                    object_refine_diversity_supervised_pixels=float(len(calls) * 2),
                    object_refine_diversity_paired_views=1.,
                    object_refine_diversity_overlap_pixels=0.)

    agent = SimpleNamespace(_device='cpu', _network=object(),
                            _optimizer=SimpleNamespace(param_groups=[{'lr': 1e-4}]),
                            train=lambda: None, zero_grad=lambda: None,
                            optimizer_step=lambda: None, update=update)
    writer = SimpleNamespace(add_scalar=lambda *args: events.append(args))
    batch = dict(observations=torch.ones(1), tasks=['close_jar'], lang_goal=['close the jar'])
    if accumulation == 1:
        scope['train'](agent, [batch], 1, epoch=0, writer=writer, loss_print_interval=1)
        expected_pixels, expected_step = 2., 0
    else:
        step = scope['train_with_accumulation'](
            agent, [batch, batch], 1, epoch=0, gradient_accumulation_steps=2,
            global_step_offset=10, writer=writer, loss_print_interval=1)
        assert step == 11
        assert all(call['loss_scale'] == .5 and not call['step_optimizer'] for call in calls)
        expected_pixels, expected_step = 3., 10  # Mean across microbatches, not a global sum.
    logged = {key: (value, step) for key, value, step in events}
    assert logged['train/object_refine_diversity_supervised_pixels'] == (expected_pixels, expected_step)
    assert logged['train/object_refine_diversity_loss_weight'] == (.01, expected_step)
    assert 'object_refine_diversity_supervised_pixels=' in Progress.messages[-1]
    assert 'object_refine_diversity_paired_views=1' in Progress.messages[-1]
    assert 'object_refine_diversity_loss=0.3' in Progress.messages[-1]


@pytest.mark.parametrize('weight', [.01, 0.])
def test_real_rlbench_config_and_agent_call_forward_both_diversity_weights(weight):
    cfg = get_cfg_defaults()
    cfg.merge_from_file(str(ROOT / 'finetune/RLBench/configs/rlbench_o2_internal_slots_refine.yaml'))
    cfg.rvt.object_slot_diversity_loss_weight = weight
    cfg.rvt.object_refine_diversity_loss_weight = weight
    tree = ast.parse((ROOT / 'finetune/RLBench/train.py').read_text(encoding='utf-8'))
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and isinstance(n.value, ast.Call)
                      and ast.unparse(n.value.func) == 'bridgevla_agent.RVTAgent')
    scope = dict(backbone=object(), exp_cfg=cfg, IMAGE_SIZE=224, SCENE_BOUNDS=[], CAMERAS=[],
                 log_dir='run', mvt_cfg=SimpleNamespace(stage_two=True, rot_ver=0),
                 bridgevla_agent=SimpleNamespace(RVTAgent=lambda **kwargs: kwargs))
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), '<rlbench-agent>', 'exec'), scope)
    assert scope['agent']['object_slot_diversity_loss_weight'] == weight
    assert scope['agent']['object_refine_diversity_loss_weight'] == weight
