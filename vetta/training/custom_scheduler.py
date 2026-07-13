"""Warmup + exponential-decay learning-rate schedule."""
from __future__ import annotations

import numpy as np
import torch


def make_custom_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int = 10000,
    gamma_factor: float = 0.1,
    gamma_period: int = 100000,
    min_mult: float | None = None,
) -> torch.optim.lr_scheduler.LambdaLR:
    a = -np.log(gamma_factor) / float(gamma_period)

    def lambda_fn(x: float) -> float:
        if x <= warmup_steps:
            ret = float(x) / float(warmup_steps)
        else:
            ret = float(np.exp(-a * (x - warmup_steps)))
        if min_mult is not None:
            ret = max(min_mult, ret)
        return ret

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_fn)
