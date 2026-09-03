"""Runtime compatibility fixes for the pinned RLBench dependency."""

import numpy as np


def rgb_handles_to_mask_safe(rgb_coded_handles):
    """Decode RGB handle pixels without uint8 overflow or input mutation."""
    image = np.asarray(rgb_coded_handles)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(
            "RGB handle mask must have shape [H, W, >=3]; "
            f"got {image.shape}"
        )

    rgb = image[..., :3]
    if np.issubdtype(rgb.dtype, np.integer):
        encoded = rgb.astype(np.int64, copy=False)
    else:
        values = np.array(rgb, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(values)):
            raise ValueError("RGB handle mask contains non-finite values")
        if values.size and np.max(values) <= 1.0:
            values *= 255.0
        encoded = np.rint(values).astype(np.int64)

    return (
        encoded[..., 0]
        + 256 * encoded[..., 1]
        + 65536 * encoded[..., 2]
    )


def install_rlbench_mask_decoder_compat():
    """Patch both RLBench references used while loading stored demos."""
    from rlbench.backend import utils as backend_utils
    from rlbench import utils as rlbench_utils

    backend_utils.rgb_handles_to_mask = rgb_handles_to_mask_safe
    # rlbench.utils imports the decoder into its own module namespace.
    rlbench_utils.rgb_handles_to_mask = rgb_handles_to_mask_safe
