"""Gemma decoder access across wrapper/direct/nested PaliGemma layouts."""

import ast
import copy
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'finetune'))
from bridgevla.utils.paligemma_compat import gemma_decoder, paligemma_component
from test_paligemma_checkpoint_compat import paligemma, policy, prediction


def actual_function(path, name, scope=None):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8'))
    function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(__package__='bridgevla.mvt', **(scope or {}))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


@pytest.mark.parametrize('layout', ['legacy', 'direct', 'current'])
def test_freeze_uses_actual_decoder_and_projector_without_module_aliases(layout):
    network = policy(layout)
    model = network.mvt1.model
    decoder = gemma_decoder(model)
    decoder.layers.extend([nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False)])
    original = copy.deepcopy(network.state_dict())
    freeze = actual_function('finetune/RLBench/train.py', 'freeze_backbone_modules')
    count = freeze(network, True, freeze_gemma_prefix_layers=1, freeze_multimodal_projector=True)
    assert not any(p.requires_grad for p in decoder.layers[0].parameters())
    assert all(p.requires_grad for p in decoder.layers[1:].parameters())
    assert all(p.requires_grad for p in decoder.norm.parameters())
    assert not decoder.embed_tokens.weight.requires_grad
    assert not any(p.requires_grad for p in paligemma_component(model, 'vision_tower').parameters())
    assert not any(p.requires_grad for p in paligemma_component(model, 'multi_modal_projector').parameters())
    assert all(p.requires_grad for p in network.action_head.parameters())
    assert count == sum(p.numel() for p in network.parameters() if not p.requires_grad)
    assert set(network.state_dict()) == set(original)
    for name, value in network.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    with pytest.raises(ValueError, match='freeze_gemma_prefix_layers'):
        freeze(network, False, freeze_gemma_prefix_layers=4)


@pytest.mark.parametrize('layout', ['legacy', 'direct', 'current'])
def test_full_language_freeze_and_zero_prefix_remain_supported(layout):
    network = policy(layout)
    freeze = actual_function('finetune/RLBench/train.py', 'freeze_backbone_modules')
    freeze(network, False, freeze_language_model=True, freeze_gemma_prefix_layers=0)
    assert not any(p.requires_grad for p in gemma_decoder(network.mvt1.model).parameters())
    assert all(p.requires_grad for p in network.action_head.parameters())


def test_unrecognized_components_fail_explicitly():
    with pytest.raises(AttributeError, match='language_model'):
        gemma_decoder(nn.Module())
    model = nn.Module()
    model.language_model = nn.Linear(2, 2)
    with pytest.raises(AttributeError, match='expected layers/norm'):
        gemma_decoder(model)


def multimodal(layout):
    model = paligemma(layout)
    decoder = gemma_decoder(model)
    model.config = SimpleNamespace(image_token_index=7, image_seq_length=2)
    model.get_input_embeddings = lambda: decoder.embed_tokens
    model.get_image_features = lambda pixels: paligemma_component(model, 'multi_modal_projector')(
        paligemma_component(model, 'vision_tower')(pixels.mean((-1, -2))))[:, None].expand(-1, 2, -1) / 2.

    def forward(self, *, inputs_embeds, **kwargs):
        if getattr(self, 'fail', False):
            raise RuntimeError('controlled forward failure')
        hidden = decoder.norm(decoder.layers[0](inputs_embeds))
        return SimpleNamespace(hidden_states=(hidden,))
    model.forward = MethodType(forward, model)
    return model


@pytest.mark.parametrize('layout', ['legacy', 'direct', 'current'])
def test_actual_efficient_forward_norm_hook_preserves_values_gradients_and_is_removed(layout):
    from bridgevla.mvt.paligemma_inputs import merge_paligemma_images
    model = multimodal(layout)
    inputs = dict(input_ids=torch.tensor([[7, 7, 1, 2]]), attention_mask=torch.ones(1, 4, dtype=torch.long),
                  pixel_values=torch.rand(1, 3, 4, 4))
    expected = model(**merge_paligemma_images(model, inputs)).hidden_states[-1]
    probe = torch.randn_like(expected)
    (expected * probe).sum().backward()
    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    method = actual_function('finetune/bridgevla/mvt/mvt_single.py', '_forward_efficient_paligemma')
    actual = method(SimpleNamespace(model=model), inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    (actual * probe).sum().backward()
    for name, p in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(p.grad, gradients[name], rtol=0, atol=0)
        else:
            assert p.grad is None
    assert not gemma_decoder(model).norm._forward_hooks
    model.fail = True
    with pytest.raises(RuntimeError, match='controlled forward'):
        method(SimpleNamespace(model=model), inputs)
    assert not gemma_decoder(model).norm._forward_hooks


@pytest.mark.parametrize('source_layout,target_layout', [
    ('legacy', 'current'), ('legacy', 'direct'), ('direct', 'current'), ('current', 'legacy')])
def test_training_initialization_remaps_only_weights_and_keeps_new_refine_head(source_layout, target_layout):
    from test_role_feature_config import _train_namespace
    source, target = policy(source_layout), policy(target_layout)
    target.object_slot_predictor_type = 'role_queries'
    target.refine_role_mask_head = nn.Linear(4, 2)
    head = copy.deepcopy(target.refine_role_mask_head.state_dict())
    checkpoint = dict(model_state=source.state_dict(), object_predictor_type='role_queries',
                      optimizer_state={'must_not_load': True})
    original_keys = list(checkpoint['model_state'])
    namespace, _ = _train_namespace(checkpoint)
    namespace['load_initial_model_checkpoint'](SimpleNamespace(_network=target), 'test-only.pth')
    for expected, actual in zip(prediction(source, source_layout, 'mvt1'), prediction(target, target_layout, 'mvt1')):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for name, p in target.refine_role_mask_head.state_dict().items():
        torch.testing.assert_close(p, head[name], rtol=0, atol=0)
    assert list(checkpoint['model_state']) == original_keys


def test_training_initialization_does_not_hide_missing_backbone_or_bad_shapes():
    from test_role_feature_config import _train_namespace
    for failure in ('missing', 'shape'):
        source, target = policy('legacy'), policy('current')
        checkpoint = dict(model_state=source.state_dict())
        name = 'mvt1.model.language_model.model.layers.0.weight'
        if failure == 'missing':
            del checkpoint['model_state'][name]
        else:
            checkpoint['model_state'][name] = torch.zeros(5, 4)
        namespace, _ = _train_namespace(checkpoint)
        with pytest.raises(RuntimeError, match='missing=|shape mismatch'):
            namespace['load_initial_model_checkpoint'](SimpleNamespace(_network=target), 'test-only.pth')


def test_real_gemma_model_without_model_attribute_resolves_and_freezes():
    pytest.importorskip('transformers')
    from transformers import GemmaConfig, GemmaModel
    decoder = GemmaModel(GemmaConfig(vocab_size=32, hidden_size=8, intermediate_size=16,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=4))
    assert not hasattr(decoder, 'model')  # The exact failing access reported by the user.
    model = paligemma('current')
    model.model.language_model = decoder
    network = nn.Module()
    network.mvt1 = nn.Module()
    network.mvt1.model = model
    assert gemma_decoder(model) is decoder
    freeze = actual_function('finetune/RLBench/train.py', 'freeze_backbone_modules')
    freeze(network, True, freeze_gemma_prefix_layers=1, freeze_multimodal_projector=True)
    assert not any(p.requires_grad for p in decoder.layers[0].parameters())
    assert all(p.requires_grad for p in decoder.layers[1].parameters())
