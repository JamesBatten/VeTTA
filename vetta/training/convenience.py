from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Sequence

import numpy as np
import scipy.optimize
import torch

from vetta.data import Tree
from vetta.inference import compute_norm_params
from vetta.inference import filter_non_proximal_aux
from vetta.inference import make_qeidx
from vetta.inference import make_semi
from vetta.inference import normalise_aux
from vetta.model import AutoencoderBatch
from vetta.model import add_octaves
from vetta.utils import to_array
from vetta.utils import to_tensor


DEFAULT_NORM_DOMAIN = (-0.25, 0.25)
DEFAULT_COST_FACTORS = {"pos": 0.9, "topology": 1.0, "rad": 0.1}


@contextmanager
def temporary_seed(seed: int):
    """Run a block under a fixed numpy RNG seed, restoring state afterwards."""
    state = np.random.get_state()
    np.random.seed(int(seed) % (2**32))
    try:
        yield
    finally:
        np.random.set_state(state)


def log_safe(x):
    if torch.is_tensor(x):
        return torch.log(x + 1e-10)
    return np.log(x + 1e-10)


def normalize_domain(pos, rad, domain):
    pos_out = (pos - domain[0]) / (domain[1] - domain[0])
    rad_out = None if rad is None else rad / (domain[1] - domain[0])
    return pos_out, rad_out


# ---------------------------------------------------------------------------
# Random decoding splits (full tree -> partial/lhs tree with a query node)
# ---------------------------------------------------------------------------


def random_decoding_order(tree: Tree):
    """Random node order preserving proximal/distal + sibling-adjacency, plus
    the candidate pivots whose query nodes have children and complete siblings.
    """
    children = tree.children
    parents = tree.parents
    dec_set = [[0]]
    order: list[int] = []
    while len(dec_set) > 0:
        dec_set_idx = np.random.choice(len(dec_set), 1, replace=False)[0]
        node_idxs = dec_set[dec_set_idx]
        dec_set.remove(node_idxs)
        for node_idx in node_idxs:
            order.append(node_idx)
            child_idxs = []
            for k in range(0, children.shape[1]):
                child_idx = children[node_idx, k]
                if child_idx > 0:
                    child_idxs.append(child_idx)
            np.random.shuffle(child_idxs)
            if len(child_idxs) > 0:
                dec_set.append(child_idxs)

    pivots = [0]
    for order_idx in range(0, len(order)):
        node_idx = order[order_idx]
        n_children = 0
        for k in range(0, children.shape[1]):
            if children[node_idx, k] != -1:
                n_children += 1
        accept_node = True
        if node_idx != 0:
            pidx = np.argmax(parents[node_idx, :])
            sidxs = []
            for k in range(0, children.shape[-1]):
                c = children[pidx, k]
                if c != -1 and c != node_idx:
                    sidxs.append(c)
            for sidx in sidxs:
                soidx = order.index(sidx)
                if soidx > order_idx:
                    accept_node = False
        if accept_node and n_children > 0:
            pivots.append(order_idx + 1)

    return order, pivots


def _get_bounds(order, minpos=None, maxpos=None):
    ret_minpos = 0
    ret_maxpos = len(order) - 1
    if maxpos is not None:
        ret_maxpos = min(ret_maxpos, maxpos)
    if minpos is not None:
        ret_minpos = max(ret_minpos, minpos)
    return ret_minpos, ret_maxpos


def _filter_pivots(order, pivots, minpos=None, maxpos=None):
    if isinstance(order, list):
        order = np.array(order)
    minpos, maxpos = _get_bounds(order, minpos, maxpos)
    pivot_range = np.arange(minpos, maxpos + 1)
    return [pivot for pivot in pivots if pivot in pivot_range]


def random_pivot(order, pivots, minpos=None, maxpos=None):
    if maxpos == 0:
        return 0
    filtered_pivots = _filter_pivots(order, pivots, minpos, maxpos)
    return np.random.choice(filtered_pivots)


def get_split_matrices(tree: Tree, order, pivot_idx: int) -> dict[str, Any]:
    """Split the tree at ``pivot_idx`` into the left-hand (partial) tree arrays
    plus the query node and its children mask. Verbatim numpy port.
    """
    node_mask = tree.node_mask
    mask = np.zeros_like(node_mask)
    if isinstance(order, list):
        order = np.array(order)

    order_lhs = []
    if pivot_idx > 0:
        order_lhs = order[0:pivot_idx]
        mask[order_lhs] = 1.0

    branches_isin = np.isin(tree.branches, order_lhs).astype(np.float32)
    branches_lhs = (branches_isin * tree.branches + (1.0 - branches_isin) * -1.0).astype(np.int32)
    branches_lhs = np.unique(branches_lhs, axis=0)
    negnull_branch_idxs = np.where(branches_lhs[:, 1] <= 0)
    if len(negnull_branch_idxs[0]) > 0:
        branches_lhs = np.delete(branches_lhs, negnull_branch_idxs, 0)

    non_leaf_idxs: set = set()
    terminate_idxs = []
    for branch_idx in range(branches_lhs.shape[0]):
        col_idxs = np.argwhere(branches_lhs[branch_idx, :] >= 0)[:, 0].tolist()
        if len(col_idxs) > 1:
            branch_non_leaves = branches_lhs[branch_idx, col_idxs[0:-1]].tolist()
            non_leaf_idxs.update(branch_non_leaves)
        if len(col_idxs) >= 1:
            terminate_idxs.append(branches_lhs[branch_idx, col_idxs[-1]])
    non_leaf_idxs = list(non_leaf_idxs)
    terminate_non_leaf_idxs = np.where(np.isin(terminate_idxs, non_leaf_idxs))
    if len(terminate_non_leaf_idxs[0]) > 0:
        branches_lhs = np.delete(branches_lhs, terminate_non_leaf_idxs, 0)

    pad_size = tree.branches.shape[0] - branches_lhs.shape[0]
    pad_lhs = -np.ones((pad_size, branches_lhs.shape[1]), dtype=np.int32)
    branches_lhs = np.concatenate([branches_lhs, pad_lhs], axis=0)
    branch_mask_lhs = (branches_lhs[:, 0] >= 0).astype(np.float32)

    mask = np.expand_dims(mask, axis=-1)
    pos = tree.pos * mask
    mask = mask[:, 0]

    radius = None
    if tree.radius is not None:
        radius = tree.radius * mask

    depth = tree.depth * mask - (1.0 - mask) * np.ones_like(tree.depth)
    depth = depth.astype(np.int32)

    query_idx = -1
    if pivot_idx > 0:
        query_idx = order_lhs[-1]

    query = np.zeros_like(node_mask)
    if pivot_idx > 0:
        query[query_idx] = 1.0

    query_children = np.zeros_like(node_mask)
    if pivot_idx == 0:
        query_children[0] = 1.0
    else:
        for k in range(0, tree.children.shape[1]):
            child_idx = tree.children[query_idx, k]
            if child_idx >= 0:
                query_children[child_idx] = 1.0

    n_children_lhs = (
        tree.n_children * mask - (1.0 - mask) * np.ones_like(tree.n_children)
    ).astype(np.int32)

    mask_r = np.expand_dims(mask, axis=-1)
    topology_lhs = (
        tree.topology * mask_r - (1.0 - mask_r) * np.ones_like(tree.topology)
    ).astype(np.float32)

    edges_lhs = None
    edges_mask_lhs = None
    if tree.edges is not None and tree.edges_mask is not None:
        active_node_idxs = np.argwhere(mask > 0.5)[:, 0]
        emask_a = np.isin(tree.edges[:, 0], active_node_idxs).astype(np.float32)
        emask_b = np.isin(tree.edges[:, 1], active_node_idxs).astype(np.float32)
        edges_mask_lhs = tree.edges_mask * emask_a * emask_b
        edges_lhs = (
            tree.edges * edges_mask_lhs.reshape(-1, 1)
            - np.ones_like(tree.edges) * (1.0 - edges_mask_lhs.reshape(-1, 1))
        ).astype(np.int32)

    ret = {
        "pivot_idx": pivot_idx,
        "query_idx": query_idx,
        "node_mask_lhs": mask,
        "branches_lhs": branches_lhs,
        "branch_mask_lhs": branch_mask_lhs,
        "pos_lhs": pos,
        "depth_lhs": depth,
        "n_children_lhs": n_children_lhs,
        "topology_lhs": topology_lhs,
        "query": query,
        "query_children": query_children,
        "edges_lhs": edges_lhs,
        "edges_mask_lhs": edges_mask_lhs,
    }
    if radius is not None:
        ret["radius_lhs"] = radius
    return ret


def random_decoding_split(tree: Tree, *, minpos=None, maxpos=None, order_seed=None) -> dict[str, Any]:
    local_seed = np.random.randint(2**32 - 1, dtype=np.uint32)
    if order_seed is not None:
        local_seed = order_seed
    with temporary_seed(local_seed):
        order, pivots = random_decoding_order(tree)
    order = np.array(order)
    pivot_idx = random_pivot(order, pivots, minpos, maxpos)
    return get_split_matrices(tree, order, pivot_idx)


# ---------------------------------------------------------------------------
# Matching (Hungarian top-k assignment) -- scipy backed
# ---------------------------------------------------------------------------


def custom_matching_fast(cost_mat: np.ndarray, target_mask: np.ndarray):
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


def top_k_matching_fast(cost_mat: np.ndarray, target_mask: np.ndarray, k: int):
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


def batch_topk_matching_fast(cost_mat, target_mask, k: int):
    cost_mat_np = np.asarray(to_array(cost_mat), dtype=np.float32)
    target_mask_np = np.asarray(to_array(target_mask), dtype=np.float32)
    b, s, t = cost_mat_np.shape

    matching_lhs = np.zeros((b, s, t), dtype=np.float32)
    matching_rhs = np.zeros((b, s, t), dtype=np.float32)
    for b_idx in range(b):
        lhs, rhs = top_k_matching_fast(cost_mat_np[b_idx], target_mask_np[b_idx], k)
        matching_lhs[b_idx] = lhs
        matching_rhs[b_idx] = rhs
    return matching_lhs, matching_rhs


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Matching:
    """Left/right soft-assignment masks from the topk matcher."""

    lhs: torch.Tensor
    rhs: torch.Tensor


@dataclass(frozen=True)
class CostMatrices:
    """Per-component cost matrices plus their weighted total."""

    total: torch.Tensor
    components: dict[str, torch.Tensor]


def compute_loss_kl(z_mu, z_logvar, config=None):
    del config
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    kl_loss /= z_mu.size(0)
    return kl_loss


def compute_unweighted_loss_aux(cm, ml, mr):
    loss_l = torch.sum(cm * ml) / torch.sum(ml)
    loss_r = torch.sum(cm * mr) / torch.sum(mr)
    return loss_l + loss_r


def _compute_cost_matrices(pred_dict, bdict, include_rad: bool, config) -> CostMatrices:
    """Build the per-component cost matrices and the weighted total.

    Position and topology always; log-radius only when radius is enabled and the
    model emits ``lograd``. Mirrors the private
    ``autoencoder._compute_cost_matrices`` decomposition.
    """
    cf = config["cost_factors"]
    model_config = config["model_config"]

    pred_pos = pred_dict["pos"].unsqueeze(2)
    pred_top = pred_dict["topology"].unsqueeze(2)
    tgt_top = bdict["topology"].unsqueeze(1)
    tgt_pos = bdict["global_pos"].unsqueeze(1)

    if model_config["out_mode"] == "octaves":
        tgt_pos, _ = normalize_domain(tgt_pos, None, model_config["domain"])
        channels = list(range(model_config["posdims"]))
        tgt_pos = add_octaves(
            tgt_pos,
            octaves=model_config["pos_octaves"],
            dim=-1,
            channels=channels,
            ret_type="torch",
            include_base=False,
        )

    pos_cost_mat = torch.sum((pred_pos - tgt_pos) ** 2.0, dim=-1) / float(pred_pos.shape[-1])
    top_cost_mat = torch.sum((pred_top - tgt_top) ** 2.0, dim=-1) / float(pred_top.shape[-1])
    cost_mat = cf["pos"] * pos_cost_mat + cf["topology"] * top_cost_mat

    components: dict[str, torch.Tensor] = {
        "pos": pos_cost_mat,
        "topology": top_cost_mat,
    }
    if include_rad and pred_dict.get("lograd") is not None:
        tgt_lograd = log_safe(bdict["radius"]).unsqueeze(1)
        lograd_cost_mat = (pred_dict["lograd"] - tgt_lograd) ** 2.0
        cost_mat = cost_mat + cf["rad"] * lograd_cost_mat
        components["lograd"] = lograd_cost_mat

    return CostMatrices(total=cost_mat, components=components)


def _compute_matching(cost_total, bdict, config) -> Matching:
    mlhs_np, mrhs_np = batch_topk_matching_fast(
        cost_total, bdict["query_children"], config["matching_k"]
    )
    mlhs = to_tensor(mlhs_np).to(cost_total.device)
    mrhs = to_tensor(mrhs_np).to(cost_total.device)
    return Matching(lhs=mlhs, rhs=mrhs)


def _add_vae_losses(ret, pred_dict, unweighted_reconst_loss, config):
    """Populate VAE loss keys and return the final ``loss`` value.

    When VAE is disabled the loss is just the unweighted reconstruction term and
    no extra keys are added (matching the private trainer behaviour).
    """
    if not config["use_vae"]:
        return unweighted_reconst_loss
    unweighted_kl_loss = compute_loss_kl(pred_dict["z_mu"], pred_dict["z_logvar"], config)
    reconst_loss = config["weight_reconst"] * unweighted_reconst_loss
    kl_loss = config["weight_kl"] * unweighted_kl_loss
    ret["reconst_loss"] = reconst_loss
    ret["kl_loss"] = kl_loss
    return reconst_loss + kl_loss


def compute_loss(pred_dict, bdict, config) -> dict[str, torch.Tensor]:
    """Matching reconstruction (+ optional VAE KL) loss.

    Public metric contract (a deliberate subset of the private trainer's richer
    dict): always ``pos_loss`` / ``top_loss`` / ``loss``, plus ``lograd_loss``
    when radius is enabled and ``reconst_loss`` / ``kl_loss`` in VAE mode. The
    private trainer additionally returns weighted/unweighted split metrics and
    inactive metric keys that the public release intentionally omits.
    """
    include_rad = bool(config["include_rad"])
    cost_matrices = _compute_cost_matrices(pred_dict, bdict, include_rad, config)
    matching = _compute_matching(cost_matrices.total, bdict, config)

    unweighted_reconst_loss = compute_unweighted_loss_aux(
        cost_matrices.total, matching.lhs, matching.rhs
    )
    pos_loss = compute_unweighted_loss_aux(
        cost_matrices.components["pos"], matching.lhs, matching.rhs
    )
    top_loss = compute_unweighted_loss_aux(
        cost_matrices.components["topology"], matching.lhs, matching.rhs
    )

    ret: dict[str, torch.Tensor] = {
        "pos_loss": pos_loss,
        "top_loss": top_loss,
    }
    lograd_cost_mat = cost_matrices.components.get("lograd")
    if lograd_cost_mat is not None:
        ret["lograd_loss"] = compute_unweighted_loss_aux(
            lograd_cost_mat, matching.lhs, matching.rhs
        )

    ret["loss"] = _add_vae_losses(ret, pred_dict, unweighted_reconst_loss, config)
    return ret


# ---------------------------------------------------------------------------
# Batch construction from in-memory Trees
# ---------------------------------------------------------------------------


def _semi_edges(edges, edges_mask, query_idx):
    if int(query_idx) == -1:
        out = make_semi(edges, edges_mask, edges, edges_mask, -1)
    else:
        out = make_semi(edges, edges_mask, edges, edges_mask, int(query_idx))
    return out


def build_training_element(
    tree: Tree,
    *,
    norm_domain,
    include_rad: bool,
    min_pivot_pos=None,
    max_pivot_pos=None,
    order_seed=None,
    filter_non_proximal: bool = True,
) -> dict[str, np.ndarray]:
    """Turn one ``Tree`` into a single training element (numpy arrays).

    Builds the full-tree encoder inputs + reconstruction targets and the
    partial (lhs) decoder inputs from a random decoding split, normalising
    positions with a shared extent/center, prepending semi-edges, and filtering
    the lhs edges to the proximal subtree.
    """
    split = random_decoding_split(
        tree, minpos=min_pivot_pos, maxpos=max_pivot_pos, order_seed=order_seed
    )
    query_idx = int(split["query_idx"])

    full_pos = np.array(tree.pos, copy=True)
    node_mask = np.array(tree.node_mask, copy=True)
    em, center = compute_norm_params(full_pos, node_mask)

    full_radius = None
    if include_rad:
        full_radius = (
            np.array(tree.radius, copy=True)
            if tree.radius is not None
            else np.zeros(node_mask.shape, dtype=np.float32)
        )
    full_pos_norm, full_radius_norm = normalise_aux(
        full_pos, full_radius, node_mask, em, center, norm_domain
    )

    lhs_radius = split.get("radius_lhs")
    lhs_pos_norm, lhs_radius_norm = normalise_aux(
        np.array(split["pos_lhs"], copy=True),
        None if lhs_radius is None else np.array(lhs_radius, copy=True),
        np.array(split["node_mask_lhs"], copy=True),
        em,
        center,
        norm_domain,
    )

    # Full-tree edges: semi edge prepended, query node 0.
    full_semi = make_semi(
        np.array(tree.edges, copy=True),
        np.array(tree.edges_mask, copy=True),
        np.array(tree.edges, copy=True),
        np.array(tree.edges_mask, copy=True),
        0,
    )
    # Partial-tree edges: semi prepended at the query node, then proximity filter.
    lhs_semi = _semi_edges(
        np.array(split["edges_lhs"], copy=True),
        np.array(split["edges_mask_lhs"], copy=True),
        query_idx,
    )
    qeidx = make_qeidx(query_idx, lhs_semi["edges_lhs"])
    if filter_non_proximal:
        filtered = filter_non_proximal_aux(
            query_idx,
            np.array(split["node_mask_lhs"], copy=True),
            np.array(tree.parents, copy=True),
            lhs_semi["edges_lhs"],
            lhs_semi["edges_mask_lhs"],
        )
        lhs_semi["edges_lhs"] = filtered["edges_lhs"]
        lhs_semi["edges_mask_lhs"] = filtered["edges_mask_lhs"]

    element = {
        "global_pos": full_pos_norm.astype(np.float32),
        "depth": np.array(tree.depth, copy=True).astype(np.int32),
        "edges": full_semi["edges"].astype(np.int32),
        "edges_mask": full_semi["edges_mask"].astype(np.float32),
        "topology": np.array(tree.topology, copy=True).astype(np.float32),
        "query_children": np.array(split["query_children"], copy=True).astype(np.float32),
        "global_pos_lhs": lhs_pos_norm.astype(np.float32),
        "depth_lhs": np.array(split["depth_lhs"], copy=True).astype(np.int32),
        "edges_lhs": lhs_semi["edges_lhs"].astype(np.int32),
        "edges_mask_lhs": lhs_semi["edges_mask_lhs"].astype(np.float32),
        "topology_lhs": np.array(split["topology_lhs"], copy=True).astype(np.float32),
        "query_idx": np.array([query_idx], dtype=np.int32),
        "qeidx": np.array([int(qeidx)], dtype=np.int32),
    }
    if include_rad:
        element["radius"] = full_radius_norm.astype(np.float32)
        element["radius_lhs"] = (
            np.zeros(node_mask.shape, dtype=np.float32)
            if lhs_radius_norm is None
            else lhs_radius_norm.astype(np.float32)
        )
    return element


_STACK_KEYS = (
    "global_pos",
    "depth",
    "edges",
    "edges_mask",
    "topology",
    "query_children",
    "global_pos_lhs",
    "depth_lhs",
    "edges_lhs",
    "edges_mask_lhs",
    "topology_lhs",
    "query_idx",
    "qeidx",
)


def collate_elements(elements: Sequence[dict[str, np.ndarray]], *, include_rad: bool, device):
    """Stack per-sample numpy elements into batched tensors on ``device``."""
    keys = list(_STACK_KEYS)
    if include_rad:
        keys += ["radius", "radius_lhs"]
    batch = {}
    for key in keys:
        stacked = np.stack([element[key] for element in elements], axis=0)
        batch[key] = to_tensor(stacked).to(device)
    return batch


def make_autoencoder_batch(bdict: dict[str, Any], *, include_rad: bool) -> AutoencoderBatch:
    """Build an ``AutoencoderBatch`` from a collated in-memory batch dict.

    Radius (``lograd`` / ``lograd_lhs``) is the only dataset-dependent part: it is
    added only when ``include_rad`` is set, centralising the "radius means
    log-radius fields" rule. Mirrors the private
    ``autoencoder.make_autoencoder_batch``.
    """
    kwargs: dict[str, Any] = {
        "pos": bdict["global_pos"],
        "depth": bdict["depth"],
        "edges": bdict["edges"],
        "edges_mask": bdict["edges_mask"],
        "pos_lhs": bdict["global_pos_lhs"],
        "depth_lhs": bdict["depth_lhs"],
        "edges_lhs": bdict["edges_lhs"],
        "edges_mask_lhs": bdict["edges_mask_lhs"],
        "topology": bdict["topology"],
        "topology_lhs": bdict["topology_lhs"],
        "query_idx": bdict["query_idx"],
        "qeidx": bdict["qeidx"],
    }
    if include_rad:
        kwargs["lograd"] = log_safe(bdict["radius"])
        kwargs["lograd_lhs"] = log_safe(bdict["radius_lhs"])
    return AutoencoderBatch.from_tree_tensors(**kwargs)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def make_optimizer(model, *, lr: float, weight_decay: float):
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def train(
    model,
    trees: Sequence[Tree],
    *,
    n_steps: int = 1000,
    batch_size: int = 8,
    lr: float = 1e-5,
    weight_decay: float = 0.1,
    max_grad_norm: float = 300.0,
    matching_k: int = 3,
    cost_factors: dict[str, float] | None = None,
    norm_domain=DEFAULT_NORM_DOMAIN,
    weight_reconst: float = 0.999,
    weight_kl: float = 0.001,
    device: str = "cpu",
    min_pivot_pos=None,
    max_pivot_pos=None,
    model_config: dict[str, Any] | None = None,
    optimizer=None,
    callback: Callable[[int, float], None] | None = None,
) -> list[float]:
    """Train ``model`` on an in-memory list of ``Tree`` objects.

    Runs ``n_steps`` optimisation steps, each sampling ``batch_size`` trees (with
    replacement) and a fresh random decoding split per tree. Returns the per-step
    total-loss history. ``out_mode`` / ``use_vae`` / ``posdims`` / ``pos_octaves``
    / ``include_rad`` are read from ``model.config`` unless overridden via
    ``model_config``. ``matching_k * max_children`` must not exceed the model's
    ``n_slots``.
    """
    if len(trees) == 0:
        raise ValueError("train requires at least one Tree")

    mc = dict(model.config)
    if model_config:
        mc.update(model_config)
    out_mode = mc["out_mode"]
    use_vae = bool(mc.get("use_vae", False))
    include_rad = bool(mc.get("include_rad", False))
    if out_mode == "octaves" and mc.get("domain") is None:
        raise ValueError(
            "out_mode='octaves' training requires model_config['domain'] (the "
            "octave-target position domain, e.g. (-3.0, 3.0)). It defaults from "
            "the model config; pass model_config={'domain': ...} to override."
        )

    config = {
        "include_rad": include_rad,
        "use_vae": use_vae,
        "matching_k": matching_k,
        "cost_factors": cost_factors or DEFAULT_COST_FACTORS,
        "weight_reconst": weight_reconst,
        "weight_kl": weight_kl,
        "model_config": {
            "out_mode": out_mode,
            "posdims": mc["posdims"],
            "pos_octaves": list(mc["pos_octaves"]),
            "domain": mc.get("domain"),
        },
    }

    torch_device = torch.device(device)
    model.to(torch_device)
    model.train()
    use_cuda = torch_device.type == "cuda"
    if optimizer is None:
        optimizer = make_optimizer(model, lr=lr, weight_decay=weight_decay)

    loss_history: list[float] = []
    for step in range(n_steps):
        chosen = np.random.randint(0, len(trees), size=batch_size)
        elements = [
            build_training_element(
                trees[int(idx)],
                norm_domain=norm_domain,
                include_rad=include_rad,
                min_pivot_pos=min_pivot_pos,
                max_pivot_pos=max_pivot_pos,
            )
            for idx in chosen
        ]
        bdict = collate_elements(elements, include_rad=include_rad, device=torch_device)

        batch = make_autoencoder_batch(bdict, include_rad=include_rad)
        pred_dict = model.forward_batch(batch, to_cuda=use_cuda)
        loss_dict = compute_loss(pred_dict, bdict, config)
        loss = loss_dict["loss"]

        optimizer.zero_grad()
        loss.backward()
        if max_grad_norm is not None and max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        loss_value = float(loss.detach().cpu())
        loss_history.append(loss_value)
        if callback is not None:
            callback(step, loss_value)

    return loss_history


__all__ = [
    "DEFAULT_COST_FACTORS",
    "DEFAULT_NORM_DOMAIN",
    "CostMatrices",
    "Matching",
    "build_training_element",
    "collate_elements",
    "compute_loss",
    "compute_loss_kl",
    "make_autoencoder_batch",
    "make_optimizer",
    "random_decoding_split",
    "train",
]
