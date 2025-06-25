# --- FILE: vetta/model/weight_init.py ---
"""
Provides a factory function for creating weight initialization routines for PyTorch models.

The main entry point is `make_weights_init`, which returns a function suitable for
use with `torch.nn.Module.apply()`. This allows for consistent and configurable
initialization of different layer types across a model.
"""
import torch
import torch.nn as nn
from typing import Callable


def make_weights_init(
    nonlinearity: str = 'relu',
    initialisation: str = 'xavier'
) -> Callable[[nn.Module], None]:
    """
    Creates a weight initialization function closure to be used with `model.apply()`.

    This factory allows for configuring the initialization strategy (e.g., 'xavier')
    and the activation function in use, which can affect how biases are initialized.

    Args:
        nonlinearity: The name of the activation function that follows the layers.
                      Currently, this is used to apply a special bias initialization
                      for 'relu' to prevent dead neurons. Defaults to 'relu'.
        initialisation: The name of the weight initialization strategy.
                        Currently, only 'xavier' is supported. Defaults to 'xavier'.

    Returns:
        A function that takes an `nn.Module` and initializes its weights and biases
        in-place according to the specified strategy.

    Raises:
        ValueError: If an unknown `initialisation` strategy is provided.
    """
    if initialisation != 'xavier':
        # This can be extended later with a dictionary of init functions.
        raise ValueError(f"Unknown initialisation strategy '{initialisation}'. Only 'xavier' is supported.")

    def weights_init(module: nn.Module) -> None:
        """
        Applies initialization to Linear, Conv, and Norm layers.

        - Linear and Conv layers: Uses Xavier uniform for weights. Bias is set to
          a small constant (0.1) for ReLU activations and zero otherwise.
        - GroupNorm and LayerNorm layers: Initializes weight to 1 and bias to 0,
          making them identity transformations at the start of training.
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Initialize weights using the Xavier uniform distribution.
            nn.init.xavier_uniform_(module.weight)
            
            # Initialize biases.
            if module.bias is not None:
                # A small positive bias for ReLU is a common heuristic to prevent
                # "dead neurons" at the beginning of training.
                if nonlinearity == 'relu':
                    nn.init.constant_(module.bias, 0.1)
                else:
                    # For other activations, a zero bias is a standard, safe default.
                    nn.init.constant_(module.bias, 0.0)

        elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
            # For normalization layers, the standard initialization is to make them
            # an identity function, so they don't affect the network at the start.
            nn.init.constant_(module.weight, 1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    return weights_init