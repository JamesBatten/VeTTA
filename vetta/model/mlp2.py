# --- FILE: vetta/model/mlp2.py ---
"""
Defines various Multi-Layer Perceptron (MLP) architectures for use in PyTorch models.

This module provides:
- MLP2: A standard two-layer MLP with fully connected layers.
- ConvMLP2: A two-layer MLP using 1x1 convolutions, suitable for operating on image-like data.
- CMLP2: A composite two-layer MLP with shared weights that can be applied either as a
  standard MLP or as a 1x1 convolutional MLP.
"""
import torch
import torch.nn as nn
from typing import Optional

from .nonlinearity import make_nlrity
from .norm import make_norm
from .weight_init import make_weights_init


class MLP2(nn.Module):
    """
    A standard two-layer MLP with a configurable hidden layer.

    Architecture: Linear -> Non-linearity -> [Normalization] -> Linear
    """

    def __init__(
        self,
        in_dims: int,
        hidden_dims: int,
        out_dims: int,
        nonlinearity: str = 'relu',
        bias_1: bool = True,
        bias_2: bool = True,
        norm_1: Optional[str] = None,
        numgroups: int = 8
    ) -> None:
        """
        Args:
            in_dims: Number of input features.
            hidden_dims: Number of features in the hidden layer.
            out_dims: Number of output features.
            nonlinearity: The non-linearity to use in the hidden layer (e.g., 'relu', 'gelu').
            bias_1: If True, adds a learnable bias to the first linear layer.
            bias_2: If True, adds a learnable bias to the second linear layer.
            norm_1: The type of normalization to apply after the non-linearity.
                    (e.g., 'groupnorm', 'layernorm'). If None, no normalization is added.
            numgroups: Number of groups for GroupNorm. Ignored if `norm_1` is not 'groupnorm'.
        """
        super().__init__()

        layers = [
            nn.Linear(in_dims, hidden_dims, bias=bias_1),
            make_nlrity(nonlinearity)
        ]

        if norm_1 is not None:
            layers.append(make_norm(hidden_dims, norm=norm_1, numgroups=numgroups))

        layers.append(nn.Linear(hidden_dims, out_dims, bias=bias_2))

        self.net = nn.Sequential(*layers)
        self.net.apply(make_weights_init(nonlinearity))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the input through the MLP."""
        return self.net(x)


class ConvMLP2(nn.Module):
    """
    A two-layer MLP implemented with 1x1 convolutions.

    This is useful for applying MLP-style transformations to each "pixel" of a
    feature map in a translation-equivariant manner.

    Architecture: Conv2d(1x1) -> Non-linearity -> Conv2d(1x1)
    """

    def __init__(
        self,
        in_dims: int,
        hidden_dims: int,
        out_dims: int,
        nonlinearity: str = 'relu',
        bias_1: bool = True,
        bias_2: bool = True,
        stride_1: int = 1
    ) -> None:
        """
        Args:
            in_dims: Number of input channels.
            hidden_dims: Number of hidden channels.
            out_dims: Number of output channels.
            nonlinearity: The non-linearity to use in the hidden layer.
            bias_1: If True, adds a learnable bias to the first convolutional layer.
            bias_2: If True, adds a learnable bias to the second convolutional layer.
            stride_1: Stride for the first convolution.
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_dims, hidden_dims, kernel_size=1, stride=stride_1, bias=bias_1),
            make_nlrity(nonlinearity),
            nn.Conv2d(hidden_dims, out_dims, kernel_size=1, stride=1, bias=bias_2)
        )
        self.net.apply(make_weights_init(nonlinearity))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes the input tensor through the 1x1 convolutional MLP."""
        return self.net(x)


class CMLP2(nn.Module):
    """
    A composite two-layer MLP whose weights can be applied as either a standard
    fully-connected MLP or as a 1x1 convolutional MLP.

    This module maintains a single set of weights (via `nn.Linear` layers) and provides
    two forward methods (`forward_mlp` and `forward_conv`) to apply them differently.
    """

    def __init__(
        self,
        in_dims: int,
        hidden_dims: int,
        out_dims: int,
        nonlinearity_1: str = 'relu',
        nonlinearity_2: Optional[str] = None,
        bias_1: bool = True,
        bias_2: bool = True,
        stride_1: int = 1,
        norm_1: Optional[str] = None,
        norm_2: Optional[str] = None,
        numgroups: int = 8
    ) -> None:
        super().__init__()
        self.stride_1 = stride_1

        # --- First Block: Linear -> NLRity -> Norm ---
        self.linear_1 = nn.Linear(in_dims, hidden_dims, bias=bias_1)
        
        block_1_layers = [
            self.linear_1,
            make_nlrity(nonlinearity_1)
        ]
        if norm_1:
            block_1_layers.append(make_norm(hidden_dims, norm=norm_1, numgroups=numgroups))
        self.block_1 = nn.Sequential(*block_1_layers)

        # --- Second Block: Linear -> NLRity -> Norm ---
        self.linear_2 = nn.Linear(hidden_dims, out_dims, bias=bias_2)

        block_2_layers = [self.linear_2]
        if nonlinearity_2:
            block_2_layers.append(make_nlrity(nonlinearity_2))
        if norm_2:
            block_2_layers.append(make_norm(out_dims, norm=norm_2, numgroups=numgroups))
        self.block_2 = nn.Sequential(*block_2_layers)

        # Apply consistent weight initialization
        # We use nonlinearity_1 as the reference for bias initialization heuristics.
        self.apply(make_weights_init(nonlinearity_1))

    def forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the MLP weights as 1x1 convolutions.

        Args:
            x: Input tensor of shape (B, C_in, H, W).

        Returns:
            Output tensor of shape (B, C_out, H', W').
        """
        # Manually apply first block as a convolution
        w1 = self.linear_1.weight.view(self.linear_1.out_features, self.linear_1.in_features, 1, 1)
        x = nn.functional.conv2d(x, w1, self.linear_1.bias, stride=self.stride_1)
        
        # Sequentially apply the rest of block_1 (nlrity, norm)
        # Note: We skip the first element of block_1, which is the linear layer itself.
        for layer in self.block_1[1:]:
            x = layer(x)

        # Manually apply second block as a convolution
        w2 = self.linear_2.weight.view(self.linear_2.out_features, self.linear_2.in_features, 1, 1)
        x = nn.functional.conv2d(x, w2, self.linear_2.bias)
        
        # Sequentially apply the rest of block_2
        for layer in self.block_2[1:]:
            x = layer(x)
            
        return x

    def forward_mlp(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the MLP as a standard sequence of fully-connected layers.

        Args:
            x: Input tensor of shape (*, in_dims).

        Returns:
            Output tensor of shape (*, out_dims).
        """
        x = self.block_1(x)
        x = self.block_2(x)
        return x