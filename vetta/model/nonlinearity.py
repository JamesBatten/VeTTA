# --- FILE: vetta/model/nonlinearity.py ---
"""
A factory function for creating PyTorch nonlinearity activation layers.

This module provides a simple, extensible way to instantiate activation functions
based on a string identifier. It replaces the need for if/elif chains or custom
wrapper classes for standard PyTorch nonlinearities.
"""
import torch.nn as nn
from typing import Dict, Type

# A mapping from string identifiers to their corresponding nn.Module classes.
# This makes the factory function clean and easily extensible.
_NONLINEARITIES: Dict[str, Type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
}


def make_nlrity(nonlinearity: str = 'relu') -> nn.Module:
    """
    Creates an instance of a nonlinearity module from a string identifier.

    This factory function is case-insensitive.

    Args:
        nonlinearity: The name of the nonlinearity to create.
                      Defaults to 'relu'.

    Returns:
        An instance of the requested torch.nn.Module nonlinearity.

    Raises:
        ValueError: If the requested nonlinearity is not supported.
    """
    key = nonlinearity.lower()
    if key in _NONLINEARITIES:
        return _NONLINEARITIES[key]()

    raise ValueError(
        f"Unknown nonlinearity '{nonlinearity}'. "
        f"Available options are: {list(_NONLINEARITIES.keys())}"
    )