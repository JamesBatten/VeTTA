# --- FILE: vetta/model/norm.py ---
"""
A factory function for creating PyTorch normalization layers.

This module provides a simple, extensible way to instantiate normalization layers
like GroupNorm or LayerNorm based on a string identifier.
"""
import torch.nn as nn

# A set of supported normalization layer keys for efficient lookup and clear error messages.
_SUPPORTED_NORMS = {"groupnorm", "layernorm"}


def make_norm(
    out_dims: int,
    norm: str = 'groupnorm',
    numgroups: int = 8
) -> nn.Module:
    """
    Creates an instance of a normalization module from a string identifier.

    This factory is case-insensitive. It preserves the original function signature
    while providing more robust and maintainable internal logic.

    Args:
        out_dims: The number of dimensions or channels for the normalization layer.
                  Corresponds to `num_channels` for GroupNorm and
                  `normalized_shape` for LayerNorm.
        norm: The name of the normalization layer. Supported values are
              'groupnorm' and 'layernorm'. Defaults to 'groupnorm'.
        numgroups: The number of groups for GroupNorm. This argument is
                   ignored if `norm` is not 'groupnorm'. Defaults to 8.

    Returns:
        An instance of the requested torch.nn.Module normalization layer.

    Raises:
        ValueError: If the `norm` argument is None or an unsupported type.
    """
    if norm is None:
        # The original code would raise an exception. This is a clearer version.
        # The calling code in mlp2.py prevents this, but a robust utility should check.
        raise ValueError("The 'norm' argument cannot be None.")

    key = norm.lower()

    if key == 'groupnorm':
        return nn.GroupNorm(num_groups=numgroups, num_channels=out_dims)
    elif key == 'layernorm':
        return nn.LayerNorm(normalized_shape=out_dims)
    else:
        # Replaced generic Exception with a more informative ValueError.
        raise ValueError(
            f"Unknown normalization layer '{norm}'. "
            f"Available options are: {list(_SUPPORTED_NORMS)}"
        )