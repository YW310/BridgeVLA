"""Resolve PaliGemma modules without changing their registered namespaces.

Legacy PaliGemma exposes a GemmaForCausalLM wrapper at language_model;
newer layouts expose the GemmaModel decoder directly, under model or via
top-level accessors. Return the actual modules; never install alias modules
that would duplicate state-dict keys or optimizer parameters.
"""


def paligemma_component(model, name):
    core = getattr(model, 'model', None)
    for owner in (core, model):
        value = getattr(owner, name, None) if owner is not None else None
        if value is not None:
            return value
    raise AttributeError(f'{type(model).__name__} has no supported PaliGemma {name} module')


def gemma_decoder(model):
    language = paligemma_component(model, 'language_model')
    for body in (language, getattr(language, 'model', None)):
        if body is not None and hasattr(body, 'layers') and hasattr(body, 'norm'):
            return body
    raise AttributeError(
        f'Unsupported Gemma decoder {type(language).__name__}: expected layers/norm '
        'directly or inside its causal-LM model wrapper')
