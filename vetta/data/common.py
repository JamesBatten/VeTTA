from __future__ import annotations

from vetta.utils import require_optional_dependency

try:
    import torch
    import torch.utils.data
    from torch.utils.data import IterableDataset
except ModuleNotFoundError:
    torch = None

    class IterableDataset:
        pass

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

try:
    import zmq
except ModuleNotFoundError:
    zmq = None


def _require_optional_dependency(dependency, package_name: str) -> None:
    require_optional_dependency(dependency, package_name, "this data path")
