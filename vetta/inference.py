from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np
import numpy.typing as npt

from vetta.config import VettaSettings
from vetta.data import Tree
from vetta.model import EdgeBatch
from vetta.model import add_octaves
from vetta.utils import to_array
from vetta.utils import to_tensor


DEFAULT_NORM_DOMAIN = (-0.25, 0.25)
POSITION_LOOKUP_RESOLUTION = 100_000


def make_parents_single(children_i: np.ndarray, node_mask_i: np.ndarray) -> np.ndarray:
    parents_i = np.zeros((children_i.shape[0], children_i.shape[0]), dtype=np.int32)
    for parent_idx in range(children_i.shape[0]):
        if node_mask_i[parent_idx] <= 0.5:
            continue
        for child_idx in children_i[parent_idx]:
            if child_idx < 0 or child_idx >= children_i.shape[0]:
                continue
            if node_mask_i[child_idx] <= 0.5:
                continue
            parents_i[child_idx, parent_idx] = 1
    return parents_i


def make_parents_batch(children_np: np.ndarray, node_mask_np: np.ndarray) -> np.ndarray:
    n_trees = int(children_np.shape[0])
    n_nodes = int(children_np.shape[1])
    parents_np = np.zeros((n_trees, n_nodes, n_nodes), dtype=np.int32)
    for tree_idx in range(n_trees):
        parents_np[tree_idx] = make_parents_single(
            children_np[tree_idx],
            node_mask_np[tree_idx],
        )
    return parents_np


def make_branches_single(
    children_i: np.ndarray,
    node_mask_i: np.ndarray,
    root_mask_i: np.ndarray,
    max_branches: int,
    max_branch_len: int,
):
    root_candidates = np.argwhere(root_mask_i > 0.5)[:, 0]
    if root_candidates.size == 0:
        active = np.argwhere(node_mask_i > 0.5)[:, 0]
        root_idx = int(active[0]) if active.size > 0 else 0
    else:
        root_idx = int(root_candidates[0])

    branches_list: list[list[int]] = []

    def _walk(node_idx, path):
        valid_children = []
        for child_idx in children_i[node_idx]:
            if child_idx < 0 or child_idx >= children_i.shape[0]:
                continue
            if node_mask_i[child_idx] <= 0.5:
                continue
            valid_children.append(int(child_idx))

        if not valid_children:
            branches_list.append(list(path))
            return

        for child_idx in valid_children:
            _walk(child_idx, path + [child_idx])

    if node_mask_i[root_idx] > 0.5:
        _walk(root_idx, [root_idx])

    if len(branches_list) > max_branches:
        raise ValueError(
            "number of reconstructed branches "
            + str(len(branches_list))
            + " exceeds max_branches "
            + str(max_branches)
        )

    branches_i = -np.ones((max_branches, max_branch_len), dtype=np.int32)
    branch_mask_i = np.zeros((max_branches,), dtype=np.float32)
    for branch_idx, branch in enumerate(branches_list):
        if len(branch) > max_branch_len:
            raise ValueError(
                "branch length "
                + str(len(branch))
                + " exceeds max_branch_len "
                + str(max_branch_len)
            )
        branches_i[branch_idx, : len(branch)] = np.array(branch, dtype=np.int32)
        branch_mask_i[branch_idx] = 1.0
    return branches_i, branch_mask_i


def make_branches_batch(
    children_np: np.ndarray,
    node_mask_np: np.ndarray,
    root_mask_np: np.ndarray,
    max_branches: int,
    max_branch_len: int,
):
    n_trees = int(children_np.shape[0])
    branches_np = -np.ones((n_trees, max_branches, max_branch_len), dtype=np.int32)
    branch_mask_np = np.zeros((n_trees, max_branches), dtype=np.float32)
    for tree_idx in range(n_trees):
        branches_i, branch_mask_i = make_branches_single(
            children_np[tree_idx],
            node_mask_np[tree_idx],
            root_mask_np[tree_idx],
            max_branches=max_branches,
            max_branch_len=max_branch_len,
        )
        branches_np[tree_idx] = branches_i
        branch_mask_np[tree_idx] = branch_mask_i
    return branches_np, branch_mask_np


# ---------------------------------------------------------------------------
# Position normalisation + partial-tree edge helpers
# ---------------------------------------------------------------------------


def compute_norm_params(
    pos: npt.NDArray[np.float32],
    node_mask: npt.NDArray[np.float32],
) -> tuple[float, npt.NDArray[np.float32]]:
    active_idxs = np.argwhere(node_mask > 0.5)[:, 0]
    pos_vals = pos[active_idxs, :]
    min_v = np.min(pos_vals, axis=0)
    max_v = np.max(pos_vals, axis=0)
    e_v = max_v - min_v
    em = np.max(e_v)
    center = min_v + e_v / 2.0
    assert em > 0.0
    return em, center


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
        edges_lhs = np.concatenate([-np.ones_like(semi_edge), edges_lhs], axis=0)
        edges_mask_lhs = np.concatenate(
            [np.array([0.0]).astype(np.float32), edges_mask_lhs],
            axis=0,
        )
    else:
        edges_lhs = np.concatenate([semi_edge, edges_lhs], axis=0)
        edges_mask_lhs = np.concatenate(
            [np.array([1.0]).astype(np.float32), edges_mask_lhs],
            axis=0,
        )

    return {
        "edges": edges,
        "edges_mask": edges_mask,
        "edges_lhs": edges_lhs,
        "edges_mask_lhs": edges_mask_lhs,
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
            "node_mask_lhs": fnode_mask_lhs,
            "edges_mask_lhs": fedges_mask_lhs,
            "edges_lhs": fedges_lhs,
        }

    return {
        "node_mask_lhs": node_mask_lhs,
        "edges_mask_lhs": edges_mask_lhs,
        "edges_lhs": edges_lhs,
    }


# ---------------------------------------------------------------------------
# Tree geometry -> segmentation mask
# ---------------------------------------------------------------------------


class SegmentationConfig(VettaSettings):
    """Rendering options for converting a vessel tree geometry into a mask."""

    size: int
    n_interpolate: int


@dataclass(frozen=True)
class TreeGeometry:
    """Array payload describing the vessel tree edges and per-node geometry."""

    edges: npt.NDArray[np.int64]
    edges_mask: npt.NDArray[np.float32]
    pos: npt.NDArray[np.float32]
    radius: npt.NDArray[np.float32]


def tree_geometry_to_segmentation(
    geometry: TreeGeometry,
    config: SegmentationConfig,
) -> npt.NDArray[np.uint8]:
    segmentation = np.zeros((config.size, config.size, 3), dtype=np.uint8)
    row_grid, col_grid = np.indices((config.size, config.size), dtype=np.float32)
    for eidx in range(0, geometry.edges.shape[0]):
        if geometry.edges_mask[eidx] <= 0.5:
            continue
        aidx, bidx = geometry.edges[eidx]
        pos_a = geometry.pos[aidx]
        pos_b = geometry.pos[bidx]
        rad_a = float(geometry.radius[aidx]) * config.size
        rad_b = float(geometry.radius[bidx]) * config.size
        for step in range(0, config.n_interpolate):
            t = step / float(max(config.n_interpolate - 1, 1))
            center = pos_a + (pos_b - pos_a) * t
            rad = rad_a + (rad_b - rad_a) * t
            px = center[0] * config.size
            py = center[1] * config.size
            segmentation[(col_grid - px) ** 2 + (row_grid - py) ** 2 <= rad**2] = 255
    return segmentation


def tree_to_segmentation(
    seg_size: int,
    n_interpolate: int,
    edges: npt.NDArray[np.int64],
    edges_mask: npt.NDArray[np.float32],
    pos: npt.NDArray[np.float32],
    radius: npt.NDArray[np.float32],
) -> npt.NDArray[np.uint8]:
    return tree_geometry_to_segmentation(
        TreeGeometry(edges=edges, edges_mask=edges_mask, pos=pos, radius=radius),
        SegmentationConfig(size=seg_size, n_interpolate=n_interpolate),
    )


# ---------------------------------------------------------------------------
# Slot decoding + clustering
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictedCluster:
    position: np.ndarray
    radius: float | None
    slot_indices: tuple[int, ...]
    topology_logits: tuple[float, ...]
    topology_child_count: int

    def to_summary(self) -> dict[str, Any]:
        return {
            "position": self.position.tolist(),
            "radius": self.radius,
            "slot_indices": list(self.slot_indices),
            "topology_logits": list(self.topology_logits),
            "topology_child_count": self.topology_child_count,
        }


def coerce_pos_octaves(model_config: dict[str, Any]) -> list[int]:
    return [int(value) for value in model_config["pos_octaves"]]


def resolve_position_decode_domain(
    *,
    infer_node_config: dict[str, Any] | None = None,
    model_config: dict[str, Any] | None = None,
) -> tuple[float, float]:
    # ``domain`` is the authoritative octave-decode range (e.g. the checkpoints'
    # [-3, 3]); ``wrap_domain`` is the encoder's position-validation range and is
    # only a fallback. An explicit infer_node_config override wins over the model
    # config either way.
    for source in (infer_node_config, model_config):
        if source is None:
            continue
        for key in ("domain", "wrap_domain"):
            if key in source:
                return tuple(float(value) for value in source[key])
    return DEFAULT_NORM_DOMAIN


def resolve_position_lookup_resolution(
    *,
    infer_node_config: dict[str, Any] | None = None,
) -> int:
    if infer_node_config is not None and "grid_size" in infer_node_config:
        return int(infer_node_config["grid_size"])
    return POSITION_LOOKUP_RESOLUTION


def make_position_lookup(
    model_config: dict[str, Any],
    *,
    infer_node_config: dict[str, Any] | None = None,
) -> np.ndarray:
    grid = np.linspace(
        0.0,
        1.0,
        resolve_position_lookup_resolution(infer_node_config=infer_node_config),
        dtype=np.float32,
    ).reshape(-1, 1)
    return add_octaves(
        grid,
        octaves=coerce_pos_octaves(model_config),
        dim=-1,
        channels=[0],
        ret_type="numpy",
        include_base=False,
    )


def decode_slot_positions(
    slot_pos_features: np.ndarray,
    *,
    model_config: dict[str, Any],
    position_lookup: np.ndarray,
    infer_node_config: dict[str, Any] | None = None,
) -> np.ndarray:
    domain = resolve_position_decode_domain(
        infer_node_config=infer_node_config,
        model_config=model_config,
    )
    posdims = int(model_config["posdims"])
    octaves = coerce_pos_octaves(model_config)
    positions = np.zeros((slot_pos_features.shape[0], posdims), dtype=np.float32)
    for slot_idx in range(slot_pos_features.shape[0]):
        for pos_idx in range(posdims):
            start = pos_idx * 2 * len(octaves)
            end = (pos_idx + 1) * 2 * len(octaves)
            dists = np.linalg.norm(
                position_lookup - slot_pos_features[slot_idx, start:end].reshape(1, -1),
                axis=-1,
            )
            lookup_idx = int(np.argmin(dists))
            positions[slot_idx, pos_idx] = domain[0] + (
                float(lookup_idx) / float(position_lookup.shape[0] - 1)
            ) * (domain[1] - domain[0])
    return positions


def average_linkage_labels(cost_matrix: np.ndarray, *, n_clusters: int) -> np.ndarray:
    n_points = int(cost_matrix.shape[0])
    if n_points == 0:
        return np.zeros((0,), dtype=np.int32)
    if n_clusters >= n_points:
        return np.arange(n_points, dtype=np.int32)

    clusters: list[list[int]] = [[index] for index in range(n_points)]
    while len(clusters) > n_clusters:
        best_pair: tuple[int, int] | None = None
        best_cost: float | None = None
        for left_idx in range(len(clusters) - 1):
            left_cluster = clusters[left_idx]
            for right_idx in range(left_idx + 1, len(clusters)):
                right_cluster = clusters[right_idx]
                pair_cost = float(
                    np.mean(cost_matrix[np.ix_(left_cluster, right_cluster)])
                )
                if (
                    best_cost is None
                    or pair_cost < best_cost
                    or (
                        pair_cost == best_cost
                        and best_pair is not None
                        and (left_idx, right_idx) < best_pair
                    )
                ):
                    best_cost = pair_cost
                    best_pair = (left_idx, right_idx)
        if best_pair is None:
            break
        left_idx, right_idx = best_pair
        merged_cluster = clusters[left_idx] + clusters[right_idx]
        clusters[left_idx] = merged_cluster
        clusters.pop(right_idx)

    labels = np.zeros((n_points,), dtype=np.int32)
    for cluster_idx, cluster in enumerate(clusters):
        labels[np.array(cluster, dtype=np.int32)] = cluster_idx
    return labels


def cluster_slots(
    slot_position_features: np.ndarray,
    topology_logits: np.ndarray,
    *,
    n_clusters: int,
    pos_weight: float = 0.9,
    topology_weight: float = 0.1,
) -> tuple[tuple[int, ...], ...]:
    if n_clusters <= 0:
        return ()
    if n_clusters == 1:
        return (tuple(range(slot_position_features.shape[0])),)

    pos_cost = np.linalg.norm(
        slot_position_features[:, None, :] - slot_position_features[None, :, :],
        axis=-1,
    )
    top_cost = np.linalg.norm(
        topology_logits[:, None, :] - topology_logits[None, :, :],
        axis=-1,
    )
    total_cost = pos_weight * pos_cost + topology_weight * top_cost
    labels = average_linkage_labels(total_cost, n_clusters=n_clusters)
    unique_labels = np.unique(labels)
    clusters: list[tuple[int, ...]] = []
    for label in unique_labels.tolist():
        member_indices = np.argwhere(labels == label)[:, 0]
        clusters.append(tuple(int(index) for index in member_indices.tolist()))
    return tuple(clusters)


def decode_clusters(
    pred_dict: dict[str, Any],
    *,
    model_config: dict[str, Any],
    n_target_children: int,
    position_lookup: np.ndarray,
    infer_node_config: dict[str, Any] | None = None,
) -> tuple[PredictedCluster, ...]:
    if n_target_children <= 0:
        return ()

    slot_pos_features = to_array(pred_dict["pos"])[0]
    topology_logits = to_array(pred_dict["topology"])[0]

    slot_radius_norm = None
    if pred_dict.get("lograd") is not None:
        slot_radius_norm = np.exp(to_array(pred_dict["lograd"])[0, :, 0]) - 1e-10

    clusters = cluster_slots(
        slot_pos_features,
        topology_logits,
        n_clusters=n_target_children,
    )
    decoded_clusters: list[PredictedCluster] = []
    for slot_indices in clusters:
        if not slot_indices:
            continue
        slot_indices_np = np.array(slot_indices, dtype=np.int32)
        mean_position_features = np.mean(slot_pos_features[slot_indices_np], axis=0, keepdims=True)
        mean_position_norm = decode_slot_positions(
            mean_position_features,
            model_config=model_config,
            position_lookup=position_lookup,
            infer_node_config=infer_node_config,
        )
        mean_radius_norm = None
        if slot_radius_norm is not None:
            mean_radius_norm = np.mean(slot_radius_norm[slot_indices_np], axis=0).reshape(1)
        radius = None
        if mean_radius_norm is not None:
            radius = float(mean_radius_norm[0])
        mean_topology_logits = np.mean(topology_logits[slot_indices_np], axis=0)
        decoded_clusters.append(
            PredictedCluster(
                position=np.array(mean_position_norm[0], copy=True),
                radius=radius,
                slot_indices=tuple(int(index) for index in slot_indices_np.tolist()),
                topology_logits=tuple(float(value) for value in mean_topology_logits.tolist()),
                topology_child_count=int(np.argmax(mean_topology_logits)),
            )
        )
    return tuple(decoded_clusters)


# ---------------------------------------------------------------------------
# Decoded-tree accumulator state
# ---------------------------------------------------------------------------


@dataclass
class RecursiveDecodeResult:
    tree: Tree
    root_clusters: tuple[PredictedCluster, ...]


@dataclass
class DecodedNode:
    position: np.ndarray
    depth: int
    topology_child_count: int
    node_idx: int
    radius: float | None = None
    child_nodes: list["DecodedNode"] = field(default_factory=list)
    edge_indices: list[int] = field(default_factory=list)

    def count_nodes(self) -> int:
        total = 1
        for child_node in self.child_nodes:
            total += child_node.count_nodes()
        return total

    def count_edges(self) -> int:
        total = len(self.child_nodes)
        for child_node in self.child_nodes:
            total += child_node.count_edges()
        return total

    def get_node(self, node_idx: int) -> "DecodedNode | None":
        if self.node_idx == node_idx:
            return self
        for child_node in self.child_nodes:
            found = child_node.get_node(node_idx)
            if found is not None:
                return found
        return None


@dataclass(frozen=True)
class ShapeField:
    name: str
    tree_attr: str
    axis: int


SHAPE_FIELDS = (
    ShapeField("max_nodes", "pos", 0),
    ShapeField("max_children", "children", 1),
    ShapeField("max_edges", "edges", 0),
    ShapeField("max_branches", "branches", 0),
    ShapeField("max_branch_len", "branches", 1),
    ShapeField("topology_size", "topology", 1),
)


@dataclass(frozen=True)
class DecodedTreeShape:
    """Fixed array dimensions used while accumulating a decoded vessel tree."""

    max_nodes: int
    max_children: int
    max_edges: int
    max_branches: int
    max_branch_len: int
    topology_size: int

    @classmethod
    def from_tree(cls, tree: Tree) -> "DecodedTreeShape":
        return cls(
            **{
                field_spec.name: int(getattr(tree, field_spec.tree_attr).shape[field_spec.axis])
                for field_spec in SHAPE_FIELDS
            }
        )


@dataclass(frozen=True)
class ArraySpec:
    name: str
    dims: tuple
    dtype: Any
    fill: Any = 0
    include_when_radius: bool = False

    def shape_for(self, shape: DecodedTreeShape) -> tuple[int, ...]:
        return tuple(dim if isinstance(dim, int) else int(getattr(shape, dim)) for dim in self.dims)

    def make(self, shape: DecodedTreeShape) -> np.ndarray:
        return np.full(self.shape_for(shape), self.fill, dtype=self.dtype)


DECODED_ARRAY_SPECS = (
    ArraySpec("pos", ("max_nodes", 2), np.float32, 0),
    ArraySpec("node_mask", ("max_nodes",), np.float32, 0),
    ArraySpec("depth", ("max_nodes",), np.int32, -1),
    ArraySpec("edges", ("max_edges", 2), np.int32, -1),
    ArraySpec("edges_mask", ("max_edges",), np.float32, 0),
    ArraySpec("topology", ("max_nodes", "topology_size"), np.int32, -1),
    ArraySpec("parents", ("max_nodes", "max_nodes"), np.int32, 0),
    ArraySpec("n_children", ("max_nodes",), np.int32, -1),
    ArraySpec("children", ("max_nodes", "max_children"), np.int32, -1),
    ArraySpec("root_mask", ("max_nodes",), np.float32, 0),
    ArraySpec("branches", ("max_branches", "max_branch_len"), np.int32, -1),
    ArraySpec("branch_mask", ("max_branches",), np.float32, 0),
    ArraySpec("rad", ("max_nodes",), np.float32, 0, include_when_radius=True),
)


def empty_decoded_arrays(
    shape: DecodedTreeShape,
    *,
    include_radius: bool,
) -> dict[str, np.ndarray]:
    return {
        spec.name: spec.make(shape)
        for spec in DECODED_ARRAY_SPECS
        if not spec.include_when_radius or include_radius
    }


TREE_ARRAY_FIELDS = (
    "branch_mask",
    "branches",
    "children",
    "depth",
    "n_children",
    "node_mask",
    "parents",
    "pos",
    "root_mask",
    "topology",
    "edges",
    "edges_mask",
)


def tree_array_kwargs(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: arrays[name] for name in TREE_ARRAY_FIELDS}


def one_hot_child_count(count: int, size: int) -> np.ndarray:
    one_hot = np.zeros((size,), dtype=np.int32)
    one_hot[min(max(count, 0), size - 1)] = 1
    return one_hot


def write_node_arrays(
    arrays: dict[str, np.ndarray],
    node: DecodedNode,
    *,
    topology_size: int,
    include_radius: bool,
) -> None:
    node_idx = node.node_idx
    arrays["pos"][node_idx, :] = node.position
    arrays["node_mask"][node_idx] = 1.0
    arrays["depth"][node_idx] = node.depth
    arrays["n_children"][node_idx] = len(node.child_nodes)
    arrays["topology"][node_idx, :] = one_hot_child_count(
        node.topology_child_count,
        topology_size,
    )
    if include_radius and node.radius is not None:
        arrays["rad"][node_idx] = float(node.radius)


def write_child_edge(
    arrays: dict[str, np.ndarray],
    *,
    parent_idx: int,
    child_slot: int,
    child_idx: int,
    edge_idx: int,
) -> None:
    arrays["children"][parent_idx, child_slot] = child_idx
    arrays["edges"][edge_idx, :] = np.array([parent_idx, child_idx], dtype=np.int32)
    arrays["edges_mask"][edge_idx] = 1.0


def add_tree_derived_arrays(
    arrays: dict[str, np.ndarray],
    shape: DecodedTreeShape,
) -> None:
    arrays["parents"] = make_parents_batch(
        arrays["children"][np.newaxis, ...],
        arrays["node_mask"][np.newaxis, ...],
    )[0]
    branches, branch_mask = make_branches_batch(
        arrays["children"][np.newaxis, ...],
        arrays["node_mask"][np.newaxis, ...],
        arrays["root_mask"][np.newaxis, ...],
        max_branches=shape.max_branches,
        max_branch_len=shape.max_branch_len,
    )
    arrays["branches"] = branches[0]
    arrays["branch_mask"] = branch_mask[0]


@dataclass(init=False)
class DecodedTree:
    shape: DecodedTreeShape
    include_radius: bool
    root_node: DecodedNode | None

    def __init__(
        self,
        shape: DecodedTreeShape | None = None,
        include_radius: bool = False,
        root_node: DecodedNode | None = None,
    ) -> None:
        if shape is None:
            raise TypeError("DecodedTree requires a shape")
        self.shape = shape
        self.include_radius = include_radius
        self.root_node = root_node

    @property
    def max_nodes(self) -> int:
        return self.shape.max_nodes

    @property
    def max_children(self) -> int:
        return self.shape.max_children

    @property
    def max_edges(self) -> int:
        return self.shape.max_edges

    @property
    def max_branches(self) -> int:
        return self.shape.max_branches

    @property
    def max_branch_len(self) -> int:
        return self.shape.max_branch_len

    @property
    def topology_size(self) -> int:
        return self.shape.topology_size

    def count_nodes(self) -> int:
        if self.root_node is None:
            return 0
        return self.root_node.count_nodes()

    def count_edges(self) -> int:
        if self.root_node is None:
            return 0
        return self.root_node.count_edges()

    def get_node(self, node_idx: int) -> DecodedNode | None:
        if self.root_node is None:
            return None
        return self.root_node.get_node(node_idx)

    def incomplete_node_indices(self) -> list[int]:
        if self.root_node is None:
            return []
        queue = [self.root_node]
        candidates: list[int] = []
        while queue:
            node = queue.pop(0)
            if node.topology_child_count > 0 and len(node.child_nodes) == 0:
                candidates.append(node.node_idx)
            queue.extend(node.child_nodes)
        return candidates

    def select_incomplete_node_idx(
        self, random_state: np.random.RandomState | None = None
    ) -> int | None:
        candidates = self.incomplete_node_indices()
        if not candidates:
            return None
        if random_state is None:
            return candidates[0]
        return int(random_state.choice(np.array(candidates, dtype=np.int32)))

    def make_arrays(self) -> dict[str, np.ndarray]:
        arrays = empty_decoded_arrays(self.shape, include_radius=self.include_radius)
        if self.root_node is None:
            return arrays

        arrays["root_mask"][self.root_node.node_idx] = 1.0
        self._fill_arrays(arrays, self.root_node)
        add_tree_derived_arrays(arrays, self.shape)
        return arrays

    def _fill_arrays(self, arrays: dict[str, np.ndarray], node: DecodedNode) -> None:
        node_idx = node.node_idx
        if node_idx >= self.max_nodes:
            raise ValueError(f"decoded node index {node_idx} exceeds max_nodes={self.max_nodes}")
        write_node_arrays(
            arrays,
            node,
            topology_size=self.topology_size,
            include_radius=self.include_radius,
        )
        if len(node.child_nodes) > self.max_children:
            raise ValueError(
                f"decoded node {node_idx} has {len(node.child_nodes)} children but "
                f"max_children={self.max_children}"
            )
        for child_slot, child_node in enumerate(node.child_nodes):
            edge_idx = node.edge_indices[child_slot]
            if edge_idx >= self.max_edges:
                raise ValueError(f"decoded edge index {edge_idx} exceeds max_edges={self.max_edges}")
            write_child_edge(
                arrays,
                parent_idx=node_idx,
                child_slot=child_slot,
                child_idx=child_node.node_idx,
                edge_idx=edge_idx,
            )
            self._fill_arrays(arrays, child_node)

    def to_tree(
        self,
        template_tree: Tree,
        *,
        metadata: dict[str, Any],
    ) -> Tree:
        arrays = self.make_arrays()
        return Tree(
            **tree_array_kwargs(arrays),
            metadata=metadata,
            world_to_image=copy.deepcopy(template_tree.world_to_image),
            segmentation=None,
            radius=arrays.get("rad"),
            config=copy.deepcopy(template_tree.config),
        )


# ---------------------------------------------------------------------------
# Tensor batch builders (drive the public VesselTreeAutoencoder)
# ---------------------------------------------------------------------------


FULL_TREE_BATCH_FIELDS = (
    ("pos", "provided", "full_pos_norm"),
    ("depth", "tree", "depth"),
    ("edges", "semi", "edges"),
    ("edges_mask", "semi", "edges_mask"),
    ("topology", "tree", "topology"),
)

DECODER_BATCH_FIELDS = (
    ("pos_lhs", "lhs", "pos"),
    ("depth_lhs", "lhs", "depth"),
    ("edges_lhs", "semi", "edges_lhs"),
    ("edges_mask_lhs", "semi", "edges_mask_lhs"),
    ("topology_lhs", "lhs", "topology"),
)


def copy_array(value: np.ndarray) -> np.ndarray:
    return np.array(value, copy=True)


def batch_tensor(value: Any, *, use_cuda: bool):
    tensor = to_tensor(value)
    if use_cuda:
        tensor = tensor.cuda()
    return tensor.unsqueeze(0)


def tree_array(tree: Tree, name: str) -> np.ndarray:
    return copy_array(getattr(tree, name))


def lhs_array(lhs_arrays: Mapping[str, np.ndarray], name: str) -> np.ndarray:
    return copy_array(lhs_arrays[name])


def make_tree_radius(tree: Tree, *, force_present: bool) -> np.ndarray | None:
    if tree.radius is not None:
        return tree_array(tree, "radius")
    if force_present:
        return np.zeros(tree.node_mask.shape, dtype=np.float32)
    return None


def make_tensor_fields(
    fields: tuple,
    resolve: Any,
    *,
    use_cuda: bool,
) -> dict[str, Any]:
    return {
        out_key: batch_tensor(resolve(source, source_key), use_cuda=use_cuda)
        for out_key, source, source_key in fields
    }


def make_lograd(
    radius_norm: np.ndarray | None,
    *,
    include_rad: bool,
    use_cuda: bool,
    missing_message: str,
) -> Any:
    if not include_rad:
        return None
    if radius_norm is None:
        raise ValueError(missing_message)
    return batch_tensor(np.log(radius_norm + 1e-10), use_cuda=use_cuda)


def prepare_full_tree_batch(
    full_tree: Tree,
    *,
    full_pos_norm: np.ndarray,
    full_radius_norm: np.ndarray | None,
    include_rad: bool,
    use_cuda: bool,
) -> dict[str, Any]:
    semi = make_semi(
        tree_array(full_tree, "edges"),
        tree_array(full_tree, "edges_mask"),
        tree_array(full_tree, "edges"),
        tree_array(full_tree, "edges_mask"),
        0,
    )
    provided = {"full_pos_norm": copy_array(full_pos_norm)}

    def resolve(source: str, key: str) -> np.ndarray:
        if source == "provided":
            return provided[key]
        if source == "tree":
            return tree_array(full_tree, key)
        if source == "semi":
            return copy_array(semi[key])
        raise ValueError(f"unknown batch source {source!r}")

    batch = make_tensor_fields(FULL_TREE_BATCH_FIELDS, resolve, use_cuda=use_cuda)
    batch["lograd"] = make_lograd(
        full_radius_norm,
        include_rad=include_rad,
        use_cuda=use_cuda,
        missing_message="checkpoint expects radius inputs but none were prepared",
    )
    return batch


def make_lhs_semi(
    lhs_arrays: Mapping[str, np.ndarray],
    *,
    query_idx: int,
) -> dict[str, np.ndarray]:
    edges = lhs_array(lhs_arrays, "edges")
    edges_mask = lhs_array(lhs_arrays, "edges_mask")
    if int(query_idx) == -1:
        return {
            "edges_lhs": edges,
            "edges_mask_lhs": edges_mask,
        }
    return make_semi(
        edges,
        edges_mask,
        edges,
        edges_mask,
        int(query_idx),
    )


def maybe_filter_lhs_semi(
    lhs_arrays: Mapping[str, np.ndarray],
    semi: Mapping[str, np.ndarray],
    *,
    query_idx: int,
    enabled: bool,
) -> dict[str, np.ndarray]:
    if not enabled:
        return dict(semi)
    filtered = filter_non_proximal_aux(
        int(query_idx),
        lhs_array(lhs_arrays, "node_mask"),
        lhs_array(lhs_arrays, "parents"),
        semi["edges_lhs"],
        semi["edges_mask_lhs"],
    )
    return {
        **dict(semi),
        "edges_lhs": filtered["edges_lhs"],
        "edges_mask_lhs": filtered["edges_mask_lhs"],
    }


@dataclass(frozen=True)
class DecoderBatchContext:
    lhs_arrays: Mapping[str, np.ndarray]
    semi: Mapping[str, np.ndarray]
    query_idx: int
    qeidx: int

    def resolve(self, source: str, key: str) -> np.ndarray:
        if source == "lhs":
            return lhs_array(self.lhs_arrays, key)
        if source == "semi":
            return copy_array(self.semi[key])
        raise ValueError(f"unknown batch source {source!r}")

    def scalar(self, value: int) -> np.ndarray:
        return np.array(int(value), dtype=np.int32)


def prepare_decoder_batch(
    lhs_arrays: Mapping[str, np.ndarray],
    *,
    query_idx: int,
    include_rad: bool,
    filter_non_proximal: bool,
    use_cuda: bool,
) -> dict[str, Any]:
    query_idx = int(query_idx)
    lhs_radius_norm = None if lhs_arrays.get("rad") is None else lhs_array(lhs_arrays, "rad")
    semi = make_lhs_semi(lhs_arrays, query_idx=query_idx)
    qeidx = make_qeidx(query_idx, semi["edges_lhs"])
    semi = maybe_filter_lhs_semi(
        lhs_arrays,
        semi,
        query_idx=query_idx,
        enabled=filter_non_proximal,
    )

    ctx = DecoderBatchContext(
        lhs_arrays=lhs_arrays,
        semi=semi,
        query_idx=query_idx,
        qeidx=int(qeidx),
    )

    batch = make_tensor_fields(DECODER_BATCH_FIELDS, ctx.resolve, use_cuda=use_cuda)
    batch["query_idx"] = batch_tensor(ctx.scalar(ctx.query_idx), use_cuda=use_cuda)
    batch["qeidx"] = batch_tensor(ctx.scalar(ctx.qeidx), use_cuda=use_cuda)
    batch["lograd_lhs"] = make_lograd(
        lhs_radius_norm,
        include_rad=include_rad,
        use_cuda=use_cuda,
        missing_message="checkpoint expects radius inputs but lhs radius was not prepared",
    )
    return batch


FULL_EDGE_BATCH_KEYS = ("pos", "lograd", "depth", "edges", "edges_mask", "topology")


def edge_batch_from_mapping(
    batch: Mapping[str, Any],
    keys: tuple[str, ...] = FULL_EDGE_BATCH_KEYS,
) -> EdgeBatch:
    return EdgeBatch(**{key: batch.get(key) for key in keys})


# Maps EdgeBatch fields to the partial (lhs) decoder-batch keys produced by
# prepare_decoder_batch. The SSA decoder always queries with no
# segmentation/radius-vector channels on the partial tree, so edges_enc_v and
# skip_vessel_mask stay None.
PARTIAL_EDGE_BATCH_KEY_MAP = {
    "pos": "pos_lhs",
    "depth": "depth_lhs",
    "topology": "topology_lhs",
    "edges": "edges_lhs",
    "edges_mask": "edges_mask_lhs",
    "lograd": "lograd_lhs",
    "query_idx": "query_idx",
    "qeidx": "qeidx",
}


def partial_edge_batch_from_decoder_batch(decoder_batch: Mapping[str, Any]) -> EdgeBatch:
    return EdgeBatch(
        **{
            target: decoder_batch.get(source)
            for target, source in PARTIAL_EDGE_BATCH_KEY_MAP.items()
        },
        edges_enc_v=None,
        skip_vessel_mask=None,
    )


def encode_full_tree_latent(
    model: Any,
    *,
    full_batch: Mapping[str, Any],
    use_cuda: bool,
    use_vae: bool,
):
    batch = edge_batch_from_mapping(full_batch)
    out = model.edges_encoder_full.encode_batch(batch, to_cuda=use_cuda)
    if use_vae:
        z = out.mu
    else:
        z = out.features
    return z.unsqueeze(0)


def denormalise_decoded_tree(
    tree: Tree,
    *,
    em: float,
    center: np.ndarray,
) -> Tree:
    denorm = denormalise_aux(
        tree_array(tree, "pos"),
        tree_array(tree, "node_mask"),
        em,
        center,
        DEFAULT_NORM_DOMAIN,
        None if tree.radius is None else tree_array(tree, "radius"),
    )
    if tree.radius is None:
        tree.pos = denorm
    else:
        tree.pos, tree.radius = denorm
    return tree


# ---------------------------------------------------------------------------
# Recursive decoder + public entry point
# ---------------------------------------------------------------------------


def build_decode_metadata(
    *,
    full_tree: Tree,
    source_chunk: str,
    source_tree_index: int,
    checkpoint_path: str,
    decoded_tree: DecodedTree,
    root_expansion_clusters: tuple[PredictedCluster, ...],
) -> dict[str, Any]:
    return {
        "source_metadata": full_tree.metadata,
        "source_chunk": source_chunk,
        "source_tree_index": source_tree_index,
        "checkpoint_path": checkpoint_path,
        "predicted_child_count": len(root_expansion_clusters),
        "decoded_node_count": decoded_tree.count_nodes(),
        "root_clusters": [cluster.to_summary() for cluster in root_expansion_clusters],
    }


def decode_full_ssa_tree(
    full_tree: Tree,
    *,
    model: Any,
    checkpoint_path: str,
    checkpoint_config: dict[str, Any],
    source_chunk: str,
    source_tree_index: int,
    device: str,
    position_lookup: np.ndarray,
    infer_node_config: dict[str, Any] | None = None,
    order_seed: int | None = None,
) -> RecursiveDecodeResult:
    model_config = checkpoint_config["model_config"]
    include_rad = bool(model_config.get("include_rad", False))
    use_vae = bool(model_config.get("use_vae", False))
    filter_non_proximal = True
    use_cuda = str(device).startswith("cuda")
    random_state = None if order_seed is None else np.random.RandomState(int(order_seed))
    full_radius = make_tree_radius(full_tree, force_present=include_rad)
    em, center = compute_norm_params(full_tree.pos, full_tree.node_mask)
    full_pos_norm, full_radius_norm = normalise_aux(
        np.array(full_tree.pos, copy=True),
        full_radius,
        np.array(full_tree.node_mask, copy=True),
        em,
        center,
        DEFAULT_NORM_DOMAIN,
    )

    decoded_tree = DecodedTree(
        shape=DecodedTreeShape.from_tree(full_tree),
        include_radius=include_rad,
    )

    full_batch = prepare_full_tree_batch(
        full_tree,
        full_pos_norm=full_pos_norm,
        full_radius_norm=full_radius_norm,
        include_rad=include_rad,
        use_cuda=use_cuda,
    )
    z = encode_full_tree_latent(
        model,
        full_batch=full_batch,
        use_cuda=use_cuda,
        use_vae=use_vae,
    )
    root_decoder_batch = prepare_decoder_batch(
        decoded_tree.make_arrays(),
        query_idx=-1,
        include_rad=include_rad,
        filter_non_proximal=filter_non_proximal,
        use_cuda=use_cuda,
    )
    root_pred_dict = model.decode_from_partial(
        z, partial_edge_batch_from_decoder_batch(root_decoder_batch), to_cuda=use_cuda
    )
    root_clusters = decode_clusters(
        root_pred_dict,
        model_config=model_config,
        n_target_children=1,
        position_lookup=position_lookup,
        infer_node_config=infer_node_config,
    )
    if len(root_clusters) != 1:
        raise ValueError(f"root decode returned {len(root_clusters)} clusters instead of 1")

    root_cluster = root_clusters[0]
    decoded_tree.root_node = DecodedNode(
        position=np.array(root_cluster.position, copy=True),
        depth=0,
        topology_child_count=root_cluster.topology_child_count,
        node_idx=0,
        radius=root_cluster.radius,
    )

    root_expansion_clusters: tuple[PredictedCluster, ...] = ()
    query_idx = decoded_tree.select_incomplete_node_idx(random_state=random_state)
    while query_idx is not None:
        query_node = decoded_tree.get_node(query_idx)
        if query_node is None:
            raise ValueError(f"decoded tree is missing query node {query_idx}")
        lhs_arrays = decoded_tree.make_arrays()
        decoder_batch = prepare_decoder_batch(
            lhs_arrays,
            query_idx=query_idx,
            include_rad=include_rad,
            filter_non_proximal=filter_non_proximal,
            use_cuda=use_cuda,
        )
        pred_dict = model.decode_from_partial(
            z, partial_edge_batch_from_decoder_batch(decoder_batch), to_cuda=use_cuda
        )
        clusters = decode_clusters(
            pred_dict,
            model_config=model_config,
            n_target_children=query_node.topology_child_count,
            position_lookup=position_lookup,
            infer_node_config=infer_node_config,
        )
        if query_idx == 0:
            root_expansion_clusters = clusters
        for cluster in clusters:
            node_idx = decoded_tree.count_nodes()
            if node_idx >= decoded_tree.max_nodes:
                break
            query_node.edge_indices.append(decoded_tree.count_edges())
            query_node.child_nodes.append(
                DecodedNode(
                    position=np.array(cluster.position, copy=True),
                    depth=query_node.depth + 1,
                    topology_child_count=cluster.topology_child_count,
                    node_idx=node_idx,
                    radius=cluster.radius,
                )
            )
        if decoded_tree.count_nodes() >= decoded_tree.max_nodes:
            break
        query_idx = decoded_tree.select_incomplete_node_idx(random_state=random_state)

    metadata = build_decode_metadata(
        full_tree=full_tree,
        source_chunk=source_chunk,
        source_tree_index=source_tree_index,
        checkpoint_path=checkpoint_path,
        decoded_tree=decoded_tree,
        root_expansion_clusters=root_expansion_clusters,
    )
    output_tree = decoded_tree.to_tree(full_tree, metadata=metadata)
    return RecursiveDecodeResult(
        tree=denormalise_decoded_tree(output_tree, em=em, center=center),
        root_clusters=root_expansion_clusters,
    )


def infer_tree(
    model: Any,
    full_tree: Tree,
    *,
    checkpoint_config: dict[str, Any],
    device: str = "cpu",
    source_chunk: str = "",
    source_tree_index: int = 0,
    checkpoint_path: str = "",
    infer_node_config: dict[str, Any] | None = None,
    order_seed: int | None = None,
) -> Tree:
    """Recursively decode a predicted vessel ``Tree`` from ``full_tree``.

    ``model`` is a trained ``vetta.VesselTreeAutoencoder`` (already on ``device``
    and in ``eval`` mode); ``checkpoint_config`` must contain a ``"model_config"``
    dict carrying at least ``pos_octaves`` / ``posdims`` and the optional
    ``use_vae`` / ``include_rad`` / ``wrap_domain`` keys. Returns the decoded
    output tree (positions denormalised back into the input frame).
    """

    model_config = checkpoint_config["model_config"]
    position_lookup = make_position_lookup(model_config, infer_node_config=infer_node_config)
    result = decode_full_ssa_tree(
        full_tree,
        model=model,
        checkpoint_path=checkpoint_path,
        checkpoint_config=checkpoint_config,
        source_chunk=source_chunk,
        source_tree_index=source_tree_index,
        device=device,
        position_lookup=position_lookup,
        infer_node_config=infer_node_config,
        order_seed=order_seed,
    )
    return result.tree


__all__ = [
    "DEFAULT_NORM_DOMAIN",
    "POSITION_LOOKUP_RESOLUTION",
    "DecodedTree",
    "DecodedTreeShape",
    "PredictedCluster",
    "RecursiveDecodeResult",
    "SegmentationConfig",
    "Tree",
    "TreeGeometry",
    "cluster_slots",
    "decode_clusters",
    "decode_full_ssa_tree",
    "decode_slot_positions",
    "infer_tree",
    "make_position_lookup",
    "make_tree_radius",
    "tree_geometry_to_segmentation",
    "tree_to_segmentation",
]
