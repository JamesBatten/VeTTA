from __future__ import annotations

import numpy as np
import scipy.optimize
import torch

from vetta.utils import to_array


def _as_numpy_float32(x: torch.Tensor | np.ndarray) -> np.ndarray:
    return np.asarray(to_array(x), dtype=np.float32)


def custom_matching_fast(
    cost_mat: np.ndarray,
    target_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    s, t = cost_mat.shape
    active_tgt_idxs = np.argwhere(target_mask > 0.5).flatten()
    active_cost_mat = cost_mat[:, active_tgt_idxs]
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(active_cost_mat)
    aux_preds = np.setdiff1d(np.arange(s), row_ind)

    matching_lhs = np.zeros((s, t), dtype=np.float32)
    matching_rhs = np.zeros((s, t), dtype=np.float32)
    matching_lhs[row_ind, active_tgt_idxs[col_ind]] = 1.0
    matching_rhs[row_ind, active_tgt_idxs[col_ind]] = 1.0

    if aux_preds.size > 0:
        aux_targets = active_tgt_idxs[np.argmin(active_cost_mat[aux_preds, :], axis=1)]
        matching_lhs[aux_preds, aux_targets] = 1.0

    return matching_lhs, matching_rhs


def top_k_matching_fast(
    cost_mat: np.ndarray,
    target_mask: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_tgts = np.sum(target_mask).astype(int)
    assert k * n_tgts <= cost_mat.shape[0]

    s, t = cost_mat.shape
    mval = np.max(cost_mat) + 1.0
    matching_lhs_acc = np.zeros((s, t), dtype=np.float32)
    matching_rhs_acc = np.zeros((s, t), dtype=np.float32)
    cost_mat_cpy = np.copy(cost_mat)

    for _ in range(k):
        _, matching_rhs = custom_matching_fast(cost_mat_cpy, target_mask)
        matching_lhs_acc += matching_rhs
        matching_rhs_acc += matching_rhs
        pidxs = np.argwhere(matching_rhs > 0.5)[:, 0]
        cost_mat_cpy[pidxs, :] = mval

    pidxs = np.argwhere(np.sum(matching_lhs_acc, axis=1) < 0.5)[:, 0]
    if pidxs.size > 0:
        cost_mat_cpy[pidxs, :] += mval * (1.0 - target_mask)[None, :]
        tidxs = np.argmin(cost_mat_cpy[pidxs, :], axis=1)
        matching_lhs_acc[pidxs, tidxs] = 1.0

    return matching_lhs_acc, matching_rhs_acc


def batch_topk_matching_fast(
    cost_mat: torch.Tensor | np.ndarray,
    target_mask: torch.Tensor | np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    cost_mat_np = _as_numpy_float32(cost_mat)
    target_mask_np = _as_numpy_float32(target_mask)
    b, s, t = cost_mat_np.shape

    matching_lhs = np.zeros((b, s, t), dtype=np.float32)
    matching_rhs = np.zeros((b, s, t), dtype=np.float32)
    for b_idx in range(b):
        lhs, rhs = top_k_matching_fast(
            cost_mat_np[b_idx, :, :],
            target_mask_np[b_idx, :],
            k,
        )
        matching_lhs[b_idx, :, :] = lhs
        matching_rhs[b_idx, :, :] = rhs
    return matching_lhs, matching_rhs
