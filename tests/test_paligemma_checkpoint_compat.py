"""Strict policy checkpoint loading across PaliGemma namespace changes."""
import ast
import copy
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'finetune'))
from bridgevla.utils.checkpoint_compat import remap_paligemma_state_dict


def language_body():
    body = nn.Module()
    body.embed_tokens = nn.Embedding(8, 4)
    body.layers = nn.ModuleList([nn.Linear(4, 4, bias=False)])
    body.norm = nn.LayerNorm(4)
    return body


def paligemma(layout):
    result = nn.Module()
    vision, projector = nn.Linear(3, 4), nn.Linear(4, 4)
    body, head = language_body(), nn.Linear(4, 8, bias=False)
    if layout == 'legacy':
        result.vision_tower = vision
        result.multi_modal_projector = projector
        result.language_model = nn.Module()
        result.language_model.model = body
        result.language_model.lm_head = head
    else:
        result.model = nn.Module()
        result.model.vision_tower = vision
        result.model.multi_modal_projector = projector
        result.model.language_model = body
        result.lm_head = head
    return result


def policy(layout, stage='mvt1'):
    result = nn.Module()
    if stage == 'single':
        result.model = paligemma(layout)
    else:
        branch = nn.Module()
        branch.model = paligemma(layout)
        setattr(result, stage, branch)
    result.object_slot_predictor1 = nn.Linear(4, 2)
    result.oracle_prior_feature_adapter1 = nn.Linear(4, 4)
    result.action_head = nn.Linear(4, 3)
    return result


def prediction(model, layout, stage):
    base = model.model if stage == 'single' else getattr(model, stage).model
    if layout == 'legacy':
        core, body, head = base, base.language_model.model, base.language_model.lm_head
    else:
        core, body, head = base.model, base.model.language_model, base.lm_head
    x = core.multi_modal_projector(core.vision_tower(torch.tensor([[.1, .2, .3]])))
    x = body.norm(body.layers[0](x + body.embed_tokens(torch.tensor([1]))))
    return (head(x), model.action_head(model.oracle_prior_feature_adapter1(x)),
            model.object_slot_predictor1(x))


@pytest.mark.parametrize('source_layout,target_layout', [('legacy', 'current'), ('current', 'legacy')])
@pytest.mark.parametrize('stage', ['mvt1', 'mvt2', 'single'])
def test_strict_roundtrip_preserves_weights_and_predictions(source_layout, target_layout, stage):
    source, target = policy(source_layout, stage), policy(target_layout, stage)
    source_state = source.state_dict()
    original_keys = list(source_state)
    mapped, renamed = remap_paligemma_state_dict(source_state, target.state_dict())
    assert renamed and set(mapped) == set(target.state_dict())
    assert list(source_state) == original_keys
    for name, value in source_state.items():
        assert mapped[renamed.get(name, name)] is value
    assert not any('object_slot' in name or 'adapter' in name or 'action_head' in name
                   for name in renamed)
    result = target.load_state_dict(mapped, strict=True)
    assert not result.missing_keys and not result.unexpected_keys
    for expected, actual in zip(prediction(source, source_layout, stage),
                                prediction(target, target_layout, stage)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_same_namespace_is_noop_and_unrecognized_keys_are_preserved():
    target = policy('legacy').state_dict()
    target['unrelated.model.vision_tower.weight'] = torch.ones(2)
    mapped, renamed = remap_paligemma_state_dict(target, target)
    assert mapped is target and not renamed
    source = {'unknown.model.vision_tower.weight': torch.zeros(2)}
    mapped, renamed = remap_paligemma_state_dict(source, target)
    assert mapped is source and not renamed


@pytest.mark.parametrize('reverse_order', [False, True])
def test_mixed_duplicate_names_are_rejected_without_overwriting(reverse_order):
    source = policy('legacy').state_dict()
    destination = 'mvt1.model.model.vision_tower.weight'
    source[destination] = torch.full((4, 3), 123.)
    if reverse_order:
        source = dict(reversed(list(source.items())))
    snapshot = {k: v.clone() for k, v in source.items()}
    with pytest.raises(RuntimeError, match='Ambiguous PaliGemma'):
        remap_paligemma_state_dict(source, policy('current').state_dict())
    for key in source:
        torch.testing.assert_close(source[key], snapshot[key], rtol=0, atol=0)


def test_renamed_shape_mismatch_is_explicit():
    source = policy('legacy').state_dict()
    source['mvt1.model.vision_tower.weight'] = torch.zeros(5, 3)
    with pytest.raises(RuntimeError, match='shape mismatch.*vision_tower'):
        remap_paligemma_state_dict(source, policy('current').state_dict())


def actual_loader():
    """Execute the actual shared loader without importing simulator packages."""
    path = ROOT / 'finetune/bridgevla/utils/rvt_utils.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if (
        isinstance(n, ast.FunctionDef) and n.name in ('load_agent', 'strip_deprecated_oracle_fusion_state')
    ) or (isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == '_DEPRECATED_ORACLE_FUSION_KEYS' for t in n.targets))]
    scope = dict(torch=torch, DDP=torch.nn.parallel.DistributedDataParallel,
                 remap_paligemma_state_dict=remap_paligemma_state_dict)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope['load_agent']


def test_shared_loader_validates_metadata_and_loads_strictly_without_rewriting_file(tmp_path, capsys):
    source, target = policy('legacy'), policy('current')
    path = tmp_path / 'model_last.pth'
    torch.save(dict(epoch=12, object_predictor_type='role_queries', model_state=source.state_dict()), path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    validated = []
    def validator(checkpoint):
        validated.append(checkpoint['object_predictor_type'])
    epoch = actual_loader()(path, SimpleNamespace(_network=target), strict=True,
                            checkpoint_validator=validator)
    assert epoch == 12 and validated == ['role_queries']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert 'PaliGemma checkpoint compatibility: remapped' in capsys.readouterr().out
    for expected, actual in zip(prediction(source, 'legacy', 'mvt1'),
                                prediction(target, 'current', 'mvt1')):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_existing_same_layout_checkpoint_keeps_default_loader_predictions(tmp_path, capsys):
    source, target = policy('legacy'), policy('legacy')
    path = tmp_path / 'baseline.pth'
    torch.save(dict(epoch=3, model_state=source.state_dict()), path)
    assert actual_loader()(path, SimpleNamespace(_network=target)) == 3
    assert 'compatibility' not in capsys.readouterr().out
    for expected, actual in zip(prediction(source, 'legacy', 'mvt1'),
                                prediction(target, 'legacy', 'mvt1')):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize('error', ['missing_head', 'unexpected_head', 'same_name_bad_shape'])
def test_shared_strict_loader_still_rejects_actual_architecture_errors(tmp_path, error):
    source = policy('legacy').state_dict()
    if error == 'missing_head':
        del source['action_head.weight']
    elif error == 'unexpected_head':
        source['different_action_head.weight'] = torch.zeros(3, 4)
    else:
        source['action_head.weight'] = torch.zeros(7, 4)
    path = tmp_path / 'wrong.pth'
    torch.save(dict(epoch=0, model_state=source), path)
    with pytest.raises(RuntimeError, match='action_head'):
        actual_loader()(path, SimpleNamespace(_network=policy('current')), strict=True)


def test_metadata_rejection_precedes_weight_loading_and_epoch_only_is_unchanged(tmp_path):
    target = policy('current')
    before = copy.deepcopy(target.state_dict())
    path = tmp_path / 'model.pth'
    torch.save(dict(epoch=7, model_state=policy('legacy').state_dict()), path)
    def reject(checkpoint):
        raise ValueError('wrong role routing')
    with pytest.raises(ValueError, match='wrong role routing'):
        actual_loader()(path, SimpleNamespace(_network=target), strict=True,
                        checkpoint_validator=reject)
    for name, value in target.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert actual_loader()(path, only_epoch=True) == 7


def test_real_huggingface_paligemma_state_strictly_loads_across_namespaces():
    pytest.importorskip('transformers')
    from transformers import PaliGemmaConfig, PaliGemmaForConditionalGeneration
    config = PaliGemmaConfig(image_token_index=63, projection_dim=16, hidden_size=16, vocab_size=64,
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                           num_attention_heads=2, image_size=8, patch_size=4),
        text_config=dict(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8))
    original = PaliGemmaForConditionalGeneration(config)
    alternate = nn.Module()
    if hasattr(original, 'model'):
        alternate.vision_tower = copy.deepcopy(original.model.vision_tower)
        alternate.multi_modal_projector = copy.deepcopy(original.model.multi_modal_projector)
        alternate.language_model = nn.Module()
        alternate.language_model.model = copy.deepcopy(original.model.language_model)
        alternate.language_model.lm_head = copy.deepcopy(original.lm_head)
    else:
        alternate.model = nn.Module()
        alternate.model.vision_tower = copy.deepcopy(original.vision_tower)
        alternate.model.multi_modal_projector = copy.deepcopy(original.multi_modal_projector)
        alternate.model.language_model = copy.deepcopy(original.language_model.model)
        alternate.lm_head = copy.deepcopy(original.language_model.lm_head)
    source, destination = nn.Module(), nn.Module()
    source.model, destination.model = original, alternate
    mapped, renamed = remap_paligemma_state_dict(source.state_dict(), destination.state_dict())
    assert len(renamed) == len(source.state_dict())
    loaded = destination.load_state_dict(mapped, strict=True)
    assert not loaded.missing_keys and not loaded.unexpected_keys
    restored, _ = remap_paligemma_state_dict(destination.state_dict(), source.state_dict())
    for name, value in source.state_dict().items():
        torch.testing.assert_close(restored[name], value, rtol=0, atol=0)
