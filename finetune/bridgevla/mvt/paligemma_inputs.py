"""Checked multi-image token merging without a full expanded boolean gather."""
import torch


def merge_paligemma_images(model, model_inputs):
    """Preserve PaliGemma's image features/scaling and language-model routing.

    Validate small integer tensors on CPU before embedding lookup. Flattened
    image positions then replace whole embedding rows, avoiding the upstream
    [B,L,D] boolean-index count check and its large temporary index tensors.
    No token IDs or image counts are clipped, silently padded or discarded.
    """
    ids, pixels = model_inputs['input_ids'], model_inputs['pixel_values']
    if ids.ndim != 2 or ids.dtype not in (torch.int32, torch.int64) or min(ids.shape) < 1:
        raise ValueError('PaliGemma input_ids must be a nonempty integer [batch,sequence] tensor')
    batch, length = ids.shape
    if pixels.ndim != 4 or pixels.shape[1] != 3 or pixels.shape[0] < batch or pixels.shape[0] % batch:
        raise ValueError('PaliGemma requires an equal positive number of RGB images per sample')
    if 'inputs_embeds' in model_inputs or model_inputs.get('past_key_values') is not None:
        raise ValueError('Checked PaliGemma image merging requires an uncached input_ids/pixel_values forward')
    embedding = model.get_input_embeddings()
    vocab, hidden = embedding.weight.shape
    image_id = model.config.image_token_index
    cpu_ids = ids.detach().cpu()
    image_mask = cpu_ids == image_id
    invalid = ((cpu_ids < 0) | (cpu_ids >= vocab)) & ~image_mask
    if invalid.any():
        locations = invalid.nonzero()[:8]
        bad = [(int(row), int(column), int(cpu_ids[row, column])) for row, column in locations]
        raise ValueError(f'PaliGemma token IDs outside embedding vocabulary [0,{vocab}): row/column/id={bad}')
    counts = image_mask.sum(-1)
    images_per_sample = pixels.shape[0] // batch
    expected = getattr(model.config, 'image_seq_length', None)
    if expected is None:
        expected = getattr(getattr(model.config, 'text_config', None), 'num_image_tokens', None)
    if expected is not None and not (counts == images_per_sample * expected).all():
        raise ValueError(f'PaliGemma image-token counts per sample={counts.tolist()}, '
                         f'expected={images_per_sample * expected} for {images_per_sample} images')
    attention = model_inputs.get('attention_mask')
    if attention is not None:
        if attention.shape != ids.shape:
            raise ValueError('PaliGemma attention_mask and input_ids shapes differ')
        if (image_mask & (attention.detach().cpu() != 1)).any():
            raise ValueError('PaliGemma image tokens must not be masked as padding')
    if not torch.isfinite(pixels).all().item():
        raise ValueError('PaliGemma pixel_values contain NaN/Inf')
    # Image IDs may intentionally be outside Gemma's embedding vocabulary.
    # Their embedding rows are fully replaced, so embed a valid placeholder.
    safe_ids = cpu_ids.masked_fill(image_mask, 0).to(device=embedding.weight.device, dtype=ids.dtype)
    embeds = embedding(safe_ids)
    features = model.get_image_features(pixels)
    if (features.ndim != 3 or features.shape[0] != pixels.shape[0] or features.shape[-1] != hidden
            or not (counts == images_per_sample * features.shape[1]).all()):
        raise ValueError(f'PaliGemma image features {tuple(features.shape)} do not match '
                         f'token counts {counts.tolist()} and embedding width {hidden}')
    positions = image_mask.flatten().nonzero(as_tuple=True)[0].to(embeds.device)
    merged = embeds.reshape(batch * length, hidden).index_copy(
        0, positions, features.to(device=embeds.device, dtype=embeds.dtype).reshape(-1, hidden))
    result = {key: value for key, value in model_inputs.items() if key not in ('input_ids', 'pixel_values')}
    result['inputs_embeds'] = merged.reshape(batch, length, hidden)
    return result
