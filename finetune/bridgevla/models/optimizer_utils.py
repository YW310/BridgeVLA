"""Pure helpers for BridgeVLA optimizer parameter grouping."""


def parameter_learning_rate(
    parameter_name,
    base_lr,
    gemma_lr,
    gemma_layer_lr_decay,
    num_gemma_layers,
):
    """Return the configured LR for one named parameter.

    Gemma's last decoder layer receives ``gemma_lr``. Earlier decoder layers
    receive geometrically decayed rates. Other Gemma parameters, such as the
    final norm, receive ``gemma_lr``; non-Gemma parameters use ``base_lr``.
    A non-positive ``gemma_lr`` preserves the historical single-LR behavior.
    """
    if gemma_lr <= 0 or 'language_model.' not in parameter_name:
        return base_lr
    if not 0 < gemma_layer_lr_decay <= 1:
        raise ValueError('gemma_layer_lr_decay must be in (0, 1]')
    if num_gemma_layers <= 0:
        raise ValueError('num_gemma_layers must be positive')

    marker = 'language_model.model.layers.'
    if marker not in parameter_name:
        return gemma_lr

    suffix = parameter_name.split(marker, 1)[1]
    layer_index = int(suffix.split('.', 1)[0])
    if not 0 <= layer_index < num_gemma_layers:
        raise ValueError(
            f'Gemma layer index {layer_index} is outside '
            f'[0, {num_gemma_layers})'
        )
    distance_from_top = num_gemma_layers - 1 - layer_index
    return gemma_lr * (gemma_layer_lr_decay ** distance_from_top)
