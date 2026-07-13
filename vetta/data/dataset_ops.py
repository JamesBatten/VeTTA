from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from vetta.utils import randn_like


@dataclass(frozen=True)
class NormParams:
    extent: float
    center: npt.NDArray[np.float32]


def compute_norm_params(
    pos: npt.NDArray[np.float32],
    node_mask: npt.NDArray[np.float32],
) -> NormParams:
    active_idxs = np.argwhere(node_mask > 0.5)[:, 0]
    pos_vals = pos[active_idxs, :]
    min_v = np.min(pos_vals, axis=0)
    max_v = np.max(pos_vals, axis=0)
    e_v = max_v - min_v
    em = np.max(e_v)
    center = min_v + e_v / 2.0
    assert em > 0.0
    return NormParams(extent=em, center=center)


def normalise_aux(
    pos: npt.NDArray[np.float32],
    radius: npt.NDArray[np.float32] | None,
    node_mask: npt.NDArray[np.float32],
    em: float,
    center: npt.NDArray[np.float32],
    norm_domain: tuple,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32] | None]:
    pos = np.copy(pos)
    if radius is not None:
        radius = np.copy(radius)
    active_idxs = np.argwhere(node_mask > 0.5)[:, 0]
    da = norm_domain[0]
    db = norm_domain[1]
    if active_idxs.shape[0] > 0:
        pos_vals = pos[active_idxs, :]
        pos_vals = (db - da) * (0.5 + (pos_vals - center.reshape(1, -1)) / em) + da
        pos[active_idxs, :] = pos_vals
        if radius is not None:
            rad_vals = radius[active_idxs]
            rad_vals = rad_vals * (db - da) / em
            radius[active_idxs] = rad_vals
    inactive_idxs = np.argwhere(node_mask < 0.5)[:, 0]
    if inactive_idxs.shape[0] > 0:
        pos[inactive_idxs, :] = 0.0
        if radius is not None:
            radius[inactive_idxs] = 0.0
    return pos, radius


def denormalise_aux(
    pos: npt.NDArray[np.float32],
    node_mask: npt.NDArray[np.float32],
    em: float,
    center: npt.NDArray[np.float32],
    norm_domain: tuple,
    radius: npt.NDArray[np.float32] | None,
) -> npt.NDArray[np.float32]:
    pos = np.copy(pos)
    if radius is not None:
        radius = np.copy(radius)
    da = norm_domain[0]
    db = norm_domain[1]
    active_idxs = np.argwhere(node_mask > 0.5)[:, 0]
    pos_vals = pos[active_idxs, :]
    pos_vals = ((pos_vals - da) / (db - da) - 0.5) * em + center.reshape(1, -1)
    pos[active_idxs, :] = pos_vals
    if radius is not None:
        rad_vals = radius[active_idxs]
        rad_vals = rad_vals * em / (db - da)
        radius[active_idxs] = rad_vals
        return pos, radius
    return pos


def remove_batch_dim(full_tree_dict: dict):
    for key in full_tree_dict.keys():
        if full_tree_dict[key] is not None:
            full_tree_dict[key] = full_tree_dict[key][0, :]
    return full_tree_dict


def add_batch_dim(full_tree_dict: dict):
    for key in full_tree_dict.keys():
        if full_tree_dict[key] is not None:
            shape = list(full_tree_dict[key].shape)
            new_shape = [1] + shape
            full_tree_dict[key] = full_tree_dict[key].reshape(new_shape)
    return full_tree_dict


def make_qeidx(query_idx, edges_lhs):
    if query_idx == -1:
        return -1
    if query_idx == 0:
        return 0
    return np.argwhere(edges_lhs[:, 1] == query_idx)[0, 0]


def make_semi(edges, edges_mask, edges_lhs, edges_mask_lhs, query_idx):
    semi_edge = np.array([0, 0]).astype(np.int32)
    semi_edge = semi_edge.reshape(1, 2)

    edges = np.concatenate([semi_edge, edges], axis=0)
    edges_mask = np.concatenate(
        [np.array([1.0]).astype(np.float32), edges_mask],
        axis=0,
    )

    if query_idx == -1:
        edges_lhs = np.concatenate(
            [-np.ones_like(semi_edge), edges_lhs], axis=0
        )
        edges_mask_lhs = np.concatenate(
            [np.array([0.0]).astype(np.float32), edges_mask_lhs],
            axis=0,
        )
    else:
        edges_lhs = np.concatenate(
            [semi_edge, edges_lhs], axis=0
        )
        edges_mask_lhs = np.concatenate(
            [np.array([1.0]).astype(np.float32), edges_mask_lhs],
            axis=0,
        )

    return {
        'edges': edges,
        'edges_mask': edges_mask,
        'edges_lhs': edges_lhs,
        'edges_mask_lhs': edges_mask_lhs,
    }


def filter_non_proximal_aux(query_idx, node_mask_lhs, parents, edges_lhs, edges_mask_lhs):
    if query_idx != -1:
        fnode_mask_lhs = np.zeros_like(node_mask_lhs)
        fnode_mask_lhs[0] = 1.0
        current_idx = query_idx
        while current_idx != -1 and current_idx != 0:
            fnode_mask_lhs[current_idx] = 1.0
            current_idx = np.argmax(parents[current_idx, :])

        fedges_mask_lhs = np.zeros_like(edges_mask_lhs)
        fedges_lhs = -np.ones_like(edges_lhs)
        for eidx in range(0, edges_lhs.shape[0]):
            if edges_mask_lhs[eidx] > 0.5:
                accept = True
                if fnode_mask_lhs[edges_lhs[eidx, 0]] < 0.5:
                    accept = False
                if fnode_mask_lhs[edges_lhs[eidx, 1]] < 0.5:
                    accept = False
                if accept:
                    fedges_mask_lhs[eidx] = 1.0
                    fedges_lhs[eidx, :] = edges_lhs[eidx, :]

        return {
            'node_mask_lhs': fnode_mask_lhs,
            'edges_mask_lhs': fedges_mask_lhs,
            'edges_lhs': fedges_lhs,
        }

    return {
        'node_mask_lhs': node_mask_lhs,
        'edges_mask_lhs': edges_mask_lhs,
        'edges_lhs': edges_lhs,
    }


def do_jitter(pos, mask, config):
    aconfig = config['aug_config']
    probs = aconfig['probs']
    njscale = aconfig['pos_noise']

    pos_cpy = np.copy(pos)
    mask_cpy = np.copy(mask)

    if len(mask_cpy.shape) == 1:
        mask_cpy = mask_cpy.reshape(-1, 1)

    jmask = np.random.uniform(0.0, 1.0, (mask_cpy.shape[0]))
    jmask = (jmask < probs['jitter']).astype(np.float32).reshape(-1, 1)
    noise_j = randn_like(pos_cpy) * njscale * jmask
    pos_cpy += noise_j * mask_cpy

    return pos_cpy
