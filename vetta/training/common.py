from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


def str_to_bool(val: str) -> bool:
    value = val.lower().strip()
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"unknown boolean string {val!r}")


def normalize_domain(
    pos: torch.Tensor | np.ndarray,
    rad: torch.Tensor | np.ndarray | None,
    domain: Sequence[float],
) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray | None]:
    pos_out = (pos - domain[0]) / (domain[1] - domain[0])
    rad_out = None
    if rad is not None:
        rad_out = rad / (domain[1] - domain[0])
    return pos_out, rad_out
