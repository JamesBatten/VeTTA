"""
Common utility functions for handling and transforming NumPy arrays and PyTorch tensors.
Includes functions for type conversion, data manipulation, and feature engineering.
"""
import torch
import numpy as np
import numpy.typing as npt
from typing import List, Dict, Union, Any, Callable, TypeVar, Tuple

# --- Type Aliases for Clarity ---

# Represents a NumPy array or a PyTorch tensor. TypeVar allows functions to indicate
# that their output type matches their input type.
TensorOrArray = TypeVar('TensorOrArray', torch.Tensor, npt.NDArray)

# Represents a nested structure of data that the conversion functions can handle.
NestedData = Union[
    TensorOrArray, Dict[str, 'NestedData'], List['NestedData'], int, float, np.number, None
]
NestedCollection = Union[Dict[str, 'NestedData'], List['NestedData']]


# --- Core Conversion Functions ---

def to_tensor(
    x: NestedData,
    to_cuda: bool = False
) -> Union[torch.Tensor, Dict[str, Any], None]:
    """
    Recursively converts a NumPy array, list, dict, or number to a PyTorch Tensor.

    Args:
        x: Input data. Can be a NumPy array, list, dict, number, or None.
        to_cuda: If True, moves the resulting tensor to the default CUDA device.

    Returns:
        The converted data as a PyTorch Tensor or a dictionary of Tensors.
        Returns None if the input is None or a NumPy array of objects containing None.
    """
    if x is None:
        return None

    if isinstance(x, torch.Tensor):
        return x.cuda() if to_cuda else x

    if isinstance(x, dict):
        return {k: to_tensor(v, to_cuda=to_cuda) for k, v in x.items()}

    if isinstance(x, list):
        # Convert list to array first, as_tensor is efficient.
        tensor = torch.as_tensor(np.array(x))
        return tensor.cuda() if to_cuda else tensor

    if isinstance(x, np.ndarray):
        # Special case for object arrays which may contain None, which torch can't handle.
        if x.dtype == np.object_ and np.any(x == None):
            return None
        # as_tensor avoids a data copy if the NumPy array is already compatible.
        tensor = torch.as_tensor(x)
        return tensor.cuda() if to_cuda else tensor

    if isinstance(x, (int, float, np.number)):
        return torch.tensor([x])

    raise TypeError(f"Unhandled type for tensor conversion: {type(x)}")


def to_array(x: NestedData) -> Union[npt.NDArray, NestedCollection, None]:
    """
    Recursively converts a PyTorch Tensor, list, or dict to a NumPy array.

    Args:
        x: Input data. Can be a PyTorch Tensor, list, dict, number, or None.

    Returns:
        The converted data as a NumPy array or a nested structure of arrays.
        Returns None if input is None.
    """
    if x is None:
        return None

    if isinstance(x, np.ndarray):
        return x

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()

    if isinstance(x, dict):
        return {k: to_array(v) for k, v in x.items()}

    if isinstance(x, list):
        # BUG FIX: The original implementation iterated but didn't update the list.
        # This comprehension correctly creates a new list of converted items.
        return [to_array(item) for item in x]

    if isinstance(x, (int, float, np.number)):
        return np.array([x])

    raise TypeError(f"Unhandled type for array conversion: {type(x)}")


# --- Data Manipulation and Feature Engineering ---

def take_along_axis(
    x: TensorOrArray,
    index: Union[int, List[int], torch.Tensor, npt.NDArray],
    axis: int
) -> TensorOrArray:
    """
    Selects values from an array or tensor along a given axis using an index.
    A framework-agnostic wrapper for np.take and torch.index_select that provides
    consistent behavior.

    Args:
        x: The source tensor or array.
        index: The index or indices to select.
        axis: The axis along which to select.

    Returns:
        The selected slice from the input. If a single index was provided,
        the selection dimension is squeezed out.
    """
    if isinstance(x, np.ndarray):
        # Squeeze to match the original PyTorch implementation's behavior.
        return np.take(x, index, axis=axis).squeeze(axis)

    if isinstance(x, torch.Tensor):
        if isinstance(index, int):
            # Convert single int to a 1-element tensor for index_select.
            index_tensor = torch.tensor([index], device=x.device, dtype=torch.long)
        else:
            # Ensure the index is a LongTensor on the correct device.
            index_tensor = torch.as_tensor(index, device=x.device, dtype=torch.long)

        # Ensure index is a 1D tensor for index_select.
        if index_tensor.dim() == 0:
            index_tensor = index_tensor.unsqueeze(0)

        return torch.index_select(x, dim=axis, index=index_tensor).squeeze(axis)

    raise TypeError(f"Unhandled type for take_along_axis: {type(x)}")


# A mapping of lifter names to their NumPy and PyTorch implementations.
# This replaces the old liftsin/liftcos functions and is easily extensible.
LIFTER_MAP: Dict[str, Tuple[Callable, Callable]] = {
    "sin": (np.sin, torch.sin),
    "cos": (np.cos, torch.cos),
}

def lift(x: TensorOrArray, octave: float, lifter: str) -> TensorOrArray:
    """
    Applies a sinusoidal lifting function to a tensor or array.

    Args:
        x: The input tensor or array.
        octave: The frequency multiplier for the sinusoidal function.
        lifter: The lifting function to apply, either 'sin' or 'cos'.

    Returns:
        The transformed tensor or array, cast to a 32-bit float type.
    """
    if lifter not in LIFTER_MAP:
        raise ValueError(f"Unknown lifter '{lifter}'. Available: {list(LIFTER_MAP.keys())}")

    func_np, func_torch = LIFTER_MAP[lifter]
    scaled_x = 2 * np.pi * octave * x

    if isinstance(x, np.ndarray):
        return func_np(scaled_x).astype(np.float32)
    if isinstance(x, torch.Tensor):
        return func_torch(scaled_x).float()

    raise TypeError(f"Unhandled type for lift: {type(x)}")


def add_octaves(
    x: TensorOrArray,
    octaves: List[int],
    dim: int,
    channels: List[int] = None,
    ret_type: str = None,
    lifters: List[str] = ['sin', 'cos'],
    mask: TensorOrArray = None,
    include_base: bool = True
) -> TensorOrArray:
    """
    Applies positional encoding by adding sinusoidal features (octaves).

    Example:
        - Input: x tensor of shape (b, 3, h, w), octaves=[1, 2, 4], dim=1,
          channels=[1, 2], lifters=['sin', 'cos'], include_base=True
        - Output: x tensor of shape (b, 15, h, w)
        - Explanation: The original 3 channels are kept. Channel 1 (e.g., X) and
          channel 2 (e.g., Y) are each lifted into 3 sin channels and 3 cos
          channels, adding 6+6=12 new channels. Total channels = 3 + 12 = 15.

    Args:
        x: Input tensor or NumPy array.
        octaves: List of frequency multipliers.
        dim: The dimension over which to compute and concatenate the octaves.
        channels: List of channel indices within `dim` to lift. If None, all are used.
        ret_type: Desired output type ('numpy' or 'torch'). If None, type of `x` is kept.
        lifters: List of lifting functions to apply ('sin', 'cos').
        mask: (For validation only). A mask that must have the same shape as x,
              excluding the `dim` dimension.
        include_base: If True, concatenates new features to the original input `x`.
                      Otherwise, returns only the new features.

    Returns:
        The transformed tensor or array with added octave features.
    """
    if mask is not None:
        expected_mask_shape = list(x.shape)
        expected_mask_shape.pop(dim)
        if list(mask.shape) != expected_mask_shape:
            raise ValueError(
                f"Mask shape {mask.shape} is invalid for input shape {x.shape} "
                f"and dim {dim}. Expected mask shape {expected_mask_shape}."
            )

    # Set up framework-specific functions to avoid code duplication.
    is_torch = isinstance(x, torch.Tensor)
    stack_fn = torch.stack if is_torch else np.stack
    cat_fn = torch.cat if is_torch else np.concatenate

    if channels is None:
        channels = list(range(x.shape[dim]))

    new_features = []
    for c_idx in channels:
        if not (0 <= c_idx < x.shape[dim]):
            raise IndexError(
                f"Channel index {c_idx} is out of range for dimension {dim} "
                f"with size {x.shape[dim]}."
            )
        # Squeeze is used to match the original function's behavior.
        channel_slice = take_along_axis(x, c_idx, dim)
        for octave in octaves:
            for lifter_name in lifters:
                lifted = lift(channel_slice, octave, lifter_name)
                new_features.append(lifted)

    if not new_features:
        if include_base:
            result = x
        else:
            # Create an empty tensor/array with the correct shape.
            empty_shape = list(x.shape)
            empty_shape[dim] = 0
            result = torch.empty(empty_shape) if is_torch else np.empty(empty_shape, dtype=np.float32)
    else:
        # Stack all new features along the specified dimension.
        stacked_features = stack_fn(new_features, dim=dim)
        result = cat_fn([x, stacked_features], dim=dim) if include_base else stacked_features

    # Handle final type conversion if explicitly requested.
    if ret_type == "numpy" and is_torch:
        return to_array(result).astype(np.float32)
    if ret_type == "torch" and not is_torch:
        return to_tensor(result).float()

    return result