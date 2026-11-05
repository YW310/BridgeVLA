"""Image/token alignment, original merge equivalence and optional real HF tests."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from finetune.bridgevla.mvt.paligemma_inputs import merge_paligemma_images


class SmallMultimodal(nn.Module):
    def __init__(self, image_id):
        super().__init__()
        self.embedding = nn.Embedding(32, 8)
        self.projector = nn.Linear(3, 8)
        self.config = SimpleNamespace(image_token_index=image_id, image_seq_length=2)
    def get_input_embeddings(self):
        return self.embedding
    def get_image_features(self, pixels):
        return self.projector(pixels.mean((-1, -2)))[:, None].expand(-1, 2, -1) / 8 ** .5


def batch(image_id, padding):
    rows = [[image_id] * 4 + [1, 2, 3], [image_id] * 4 + [1, 2]]
    rows[1] = ([0] + rows[1]) if padding == 'left' else (rows[1] + [0])
    ids = torch.tensor(rows)
    return dict(input_ids=ids, attention_mask=ids.ne(0).long(), pixel_values=torch.randn(4, 3, 4, 4))


def original_merge(model, inputs):
    ids = inputs['input_ids']
    mask = ids == model.config.image_token_index
    safe = ids.masked_fill(mask, 0) if model.config.image_token_index >= 32 else ids
    embeds = model.get_input_embeddings()(safe)
    features = model.get_image_features(inputs['pixel_values'])
    mask = mask.unsqueeze(-1).expand_as(embeds)
    assert embeds[mask].numel() == features.numel()
    return embeds.masked_scatter(mask, features.to(embeds.dtype))


@pytest.mark.parametrize('image_id', [16, 40])
@pytest.mark.parametrize('padding', ['left', 'right'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_checked_merge_preserves_values_and_gradients(image_id, padding, dtype):
    torch.manual_seed(9)
    model = SmallMultimodal(image_id).to(dtype)
    other = copy.deepcopy(model)
    data = batch(image_id, padding)
    data['pixel_values'] = data['pixel_values'].to(dtype)
    ids = data['input_ids'].clone()
    expected = original_merge(model, data)
    merged = merge_paligemma_images(other, data)
    torch.testing.assert_close(merged['inputs_embeds'], expected, rtol=0, atol=0)
    assert set(merged) == {'inputs_embeds', 'attention_mask'}
    assert merged['attention_mask'] is data['attention_mask']
    torch.testing.assert_close(data['input_ids'], ids)
    probe = torch.randn_like(expected)
    (expected * probe).sum().backward()
    (merged['inputs_embeds'] * probe).sum().backward()
    for (_, old), (_, new) in zip(model.named_parameters(), other.named_parameters()):
        torch.testing.assert_close(new.grad, old.grad, rtol=.03 if dtype == torch.bfloat16 else 1e-6, atol=1e-6)


@pytest.mark.parametrize('mutation,message', [
    ('negative', 'vocabulary'), ('oov', 'vocabulary'), ('per_row', 'counts per sample'),
    ('masked_image', 'padding'), ('nan_pixels', 'NaN/Inf'), ('image_batch', 'equal positive'),
])
def test_bad_inputs_fail_before_embedding_or_vision(mutation, message, monkeypatch):
    model, data = SmallMultimodal(40), batch(40, 'left')
    if mutation == 'negative':
        data['input_ids'][0, -1] = -1
    elif mutation == 'oov':
        data['input_ids'][0, -1] = 32
    elif mutation == 'per_row':
        # Total token count is unchanged, but images would leak across samples.
        data['input_ids'][0, 0] = 1
        data['input_ids'][1, -1] = 40
    elif mutation == 'masked_image':
        data['attention_mask'][0, 0] = 0
    elif mutation == 'nan_pixels':
        data['pixel_values'][0, 0, 0, 0] = float('nan')
    elif mutation == 'image_batch':
        data['pixel_values'] = data['pixel_values'][:3]
    def unexpected(*args, **kwargs):
        raise AssertionError('Bad input must be rejected before GPU embedding/vision')
    monkeypatch.setattr(model.embedding, 'forward', unexpected)
    monkeypatch.setattr(model, 'get_image_features', unexpected)
    with pytest.raises(ValueError, match=message):
        merge_paligemma_images(model, data)


def test_counts_follow_real_paligemma_text_config_field():
    model, data = SmallMultimodal(40), batch(40, 'left')
    model.config = SimpleNamespace(image_token_index=40, text_config=SimpleNamespace(num_image_tokens=2))
    data['input_ids'][0, 0] = 1
    with pytest.raises(ValueError, match='counts per sample'):
        merge_paligemma_images(model, data)


@pytest.mark.parametrize('image_id,checkpointing', [(63, False), (70, False), (63, True)])
def test_real_paligemma_hidden_states_and_gradients_match_original(image_id, checkpointing):
    pytest.importorskip('transformers')
    from transformers import PaliGemmaConfig, PaliGemmaForConditionalGeneration
    from finetune.OHT import bootstrap  # noqa: F401; expose first-party bridgevla imports
    from finetune.bridgevla.mvt.mvt_single import MVT
    torch.manual_seed(23)
    config = PaliGemmaConfig(image_token_index=image_id, projection_dim=16, hidden_size=16, vocab_size=64,
        vision_config=dict(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                           num_attention_heads=2, image_size=8, patch_size=4),
        text_config=dict(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8,
                         max_position_embeddings=64, pad_token_id=0, bos_token_id=1, eos_token_id=2))
    model = PaliGemmaForConditionalGeneration(config)
    other = copy.deepcopy(model)
    if checkpointing:
        for value in (model, other):
            value.train()
            value.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    else:
        model.eval()
        other.eval()
    ids = torch.tensor([[image_id] * 12 + [1, 3, 4, 2], [0] + [image_id] * 12 + [1, 3, 2]])
    inputs = dict(input_ids=ids, attention_mask=ids.ne(0).long(), pixel_values=torch.randn(6, 3, 8, 8))
    expected = model(**inputs, output_hidden_states=True, use_cache=False,
                     logits_to_keep=1, return_dict=True).hidden_states[-1]
    wrapper = MVT.__new__(MVT)
    nn.Module.__init__(wrapper)
    wrapper.model = other
    actual = wrapper._forward_efficient_paligemma(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    probe = torch.randn_like(actual)
    (expected * probe).sum().backward()
    (actual * probe).sum().backward()
    for (name, old), (_, new) in zip(model.named_parameters(), other.named_parameters()):
        if old.grad is None:
            assert new.grad is None, name
        else:
            torch.testing.assert_close(new.grad, old.grad, rtol=1e-4, atol=1e-5, msg=name)
    assert not other.language_model.model.norm._forward_hooks
