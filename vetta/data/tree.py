from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .common import _require_optional_dependency
from .element import Element
from .field_specs import TREE_ARRAY_SPECS
from .field_specs import TREE_METADATA_SPECS
from .field_specs import TREE_PAYLOAD_SPECS
from .field_specs import TREE_REQUIRED_ARRAY_SPECS
from .field_specs import collect_present_fields
from .field_specs import field_names
from .field_specs import path_map
from .field_specs import rename_fields
from .field_specs import resolve_existing_path
from .field_specs import select_known_fields
from vetta.utils import load_array
from vetta.utils import load_image
from vetta.utils import load_json
from vetta.utils import save_array
from vetta.utils import save_image
from vetta.utils import save_json
from vetta.utils import temporary_seed

try:
    import zmq
except ModuleNotFoundError:
    zmq = None


def gaussian_blob_single(height, width, xi, yi, sigma):
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)
    return np.exp(
        -((x - xi) ** 2 + (y - yi) ** 2) /
        (2 * sigma ** 2)
    ).astype(np.float32)


# --- begin public data: tree.py ---
TREE_ARRAY_KEYS = (
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

TREE_REQUIRED_ARRAY_KEYS = (
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
)

TREE_OPTIONAL_ARRAY_KEYS = (
    "edges",
    "edges_mask",
)

TREE_REQUIRED_VIEW_SHAPES = {
    "branch_mask": ("max_branches",),
    "branches": ("max_branches", "max_branch_len"),
    "children": ("max_nodes", "max_children"),
    "depth": ("max_nodes",),
    "n_children": ("max_nodes",),
    "node_mask": ("max_nodes",),
    "parents": ("max_nodes", "max_nodes"),
    "pos": ("max_nodes", 2),
    "root_mask": ("max_nodes",),
    "topology": ("max_nodes", "topology_width"),
}

TREE_OPTIONAL_VIEW_SHAPES = {
    "edges": ("max_edges", 2),
    "edges_mask": ("max_edges",),
}

TREE_ARRAY_DTYPES = {
    "branch_mask": np.float32,
    "branches": np.int64,
    "children": np.int64,
    "depth": np.float32,
    "n_children": np.int64,
    "node_mask": np.float32,
    "parents": np.int64,
    "pos": np.float32,
    "root_mask": np.float32,
    "topology": np.int64,
    "edges": np.int64,
    "edges_mask": np.float32,
    "segmentation": np.uint8,
    "radius": np.float32,
}

TREE_OPTIONAL_ZMQ_KEYS = ("segmentation", "radius")


@dataclass
class TreeArrays:
    branch_mask: Any
    branches: Any
    children: Any
    depth: Any
    n_children: Any
    node_mask: Any
    parents: Any
    pos: Any
    root_mask: Any
    topology: Any
    edges: Any | None = None
    edges_mask: Any | None = None

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> TreeArrays:
        fields = select_known_fields(data, TREE_ARRAY_SPECS)
        missing = sorted(
            name for name in field_names(TREE_REQUIRED_ARRAY_SPECS)
            if name not in fields
        )
        if missing:
            raise KeyError(f"missing required tree array fields: {missing}")

        _validate_optional_edge_pair(fields)
        return cls(**fields)

    def to_mapping(self, *, include_none: bool = True) -> dict[str, Any]:
        return collect_present_fields(
            self,
            TREE_ARRAY_SPECS,
            include_none=include_none,
        )


def tree_array_path_map(path: str | Path) -> dict[str, Path]:
    return path_map(path, TREE_ARRAY_SPECS)


def tree_payload_path_map(path: str | Path) -> dict[str, Path]:
    return path_map(path, TREE_PAYLOAD_SPECS)


def tree_metadata_path_map(path: str | Path) -> dict[str, Path]:
    return path_map(path, TREE_METADATA_SPECS)


def tree_path_map(path: str | Path) -> dict[str, Path]:
    return {
        **tree_array_path_map(path),
        **tree_payload_path_map(path),
        **tree_metadata_path_map(path),
    }


def resolve_tree_array_path(path: str | Path) -> Path:
    path = Path(path)
    if path.suffix == ".npy":
        return resolve_existing_path(path, (path.with_suffix(".npz"),))
    return resolve_existing_path(path)


def _require_path(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _shape_from_config(shape_spec, config):
    config_with_derived = dict(config)
    config_with_derived["topology_width"] = config["max_children"] + 1
    return [
        config_with_derived[item] if isinstance(item, str) else item
        for item in shape_spec
    ]


def _validate_tree_array_shapes(arrays, config, optional_arrays=None):
    optional_arrays = optional_arrays or {}
    for key_str, shape_spec in TREE_REQUIRED_VIEW_SHAPES.items():
        if key_str not in arrays:
            raise KeyError(f"missing required tree array {key_str!r}")
        expected = _shape_from_config(shape_spec, config)
        actual = list(arrays[key_str].shape)
        if actual != expected:
            raise ValueError(
                f"Tree array {key_str!r} has shape {actual}; expected {expected}"
            )

    _validate_optional_edge_pair(optional_arrays)
    for key_str, shape_spec in TREE_OPTIONAL_VIEW_SHAPES.items():
        array = optional_arrays.get(key_str)
        if array is None:
            continue
        expected = _shape_from_config(shape_spec, config)
        actual = list(array.shape)
        if actual != expected:
            raise ValueError(
                f"Tree array {key_str!r} has shape {actual}; expected {expected}"
            )


def _validate_optional_edge_pair(arrays) -> None:
    has_edges = arrays.get("edges") is not None
    has_edges_mask = arrays.get("edges_mask") is not None
    if has_edges != has_edges_mask:
        raise ValueError("optional tree arrays 'edges' and 'edges_mask' must be present together")


def _load_required_arrays(paths):
    return {
        key_str: load_array(paths[key_str])
        for key_str in TREE_REQUIRED_ARRAY_KEYS
    }


def _load_optional_array(paths, key_str):
    if os.path.exists(paths[key_str]):
        return load_array(paths[key_str])
    return None


def _load_optional_arrays(paths):
    optional_arrays = {
        key_str: _load_optional_array(paths, key_str)
        for key_str in TREE_OPTIONAL_ARRAY_KEYS
    }
    _validate_optional_edge_pair(optional_arrays)
    return optional_arrays


def _load_segmentation_image(path):
    if not os.path.exists(path):
        return None
    segmentation = load_image(path)
    if segmentation.ndim == 2:
        return np.expand_dims(segmentation, axis=-1)
    if segmentation.ndim == 3:
        return segmentation[:, :, 0:1]
    raise ValueError(f"tree segmentation image has unsupported shape {list(segmentation.shape)}: {path}")


def _segmentation_for_save(segmentation):
    if segmentation.ndim == 2:
        return np.expand_dims(segmentation, axis=-1)
    if segmentation.ndim == 3 and segmentation.shape[-1] in (1, 3):
        return segmentation
    raise ValueError(f"tree segmentation has unsupported shape {list(segmentation.shape)}")


def _save_tree_array(path, array) -> None:
    if array is not None:
        save_array(path, array)


def _zmq_payload_keys(arr_shapes):
    missing = [
        key_str
        for key_str in TREE_REQUIRED_ARRAY_KEYS
        if key_str not in arr_shapes
    ]
    if missing:
        raise ValueError("missing required tree ZMQ frame shape(s): " + ", ".join(missing))

    send_keys = [key_str for key_str in TREE_ARRAY_KEYS if key_str in arr_shapes]
    for key_str in TREE_OPTIONAL_ZMQ_KEYS:
        if key_str in arr_shapes:
            send_keys.append(key_str)
    return send_keys


def _as_zmq_array(key_str, array):
    array = np.asarray(array)
    expected_dtype = TREE_ARRAY_DTYPES[key_str]
    if array.dtype != expected_dtype:
        array = array.astype(expected_dtype)
    return array


def _decode_zmq_array(message, key_str, shape):
    buf = memoryview(message)
    array = np.frombuffer(buf, dtype=TREE_ARRAY_DTYPES[key_str])
    return array.reshape(shape)


def _add_missing_optional_tree_payloads(tree_dict):
    for key_str in (*TREE_OPTIONAL_ARRAY_KEYS, *TREE_OPTIONAL_ZMQ_KEYS):
        tree_dict.setdefault(key_str, None)
    return tree_dict


class Tree:


    def __init__(self,
                 branch_mask,
                 branches,
                 children,
                 depth,
                 n_children,
                 node_mask,
                 parents,
                 pos,
                 root_mask,
                 topology,
                 edges=None,
                 edges_mask=None,
                 metadata=None,
                 world_to_image=None,
                 segmentation=None,
                 radius=None,
                 config=None):

        if config is None:
            config = Tree.default_config()

        self.branch_mask = branch_mask
        self.branches = branches
        self.children = children
        self.depth = depth
        self.n_children = n_children
        self.node_mask = node_mask
        self.parents = parents
        self.pos = pos
        self.root_mask = root_mask
        self.topology = topology
        self.edges = edges
        self.edges_mask = edges_mask
        self.metadata = metadata
        self.world_to_image = world_to_image
        self.segmentation = segmentation
        self.radius = radius
        self.config = config


    @classmethod
    def default_config(cls):
        return {
            'max_nodes': 40,
            'max_branches': 25,
            'max_branch_len': 15,
            'max_children': 3,
            'max_edges': 50,
            'sigma': 3.5
        }


    @classmethod
    def from_arrays(
        cls,
        arrays: TreeArrays | Mapping[str, object],
        **kwargs,
    ) -> Tree:
        tree_arrays = (
            arrays
            if isinstance(arrays, TreeArrays)
            else TreeArrays.from_mapping(arrays)
        )
        return cls(**tree_arrays.to_mapping(), **kwargs)


    @classmethod
    def load_view(cls, path, config=None):
        if config is None:
            config = Tree.default_config()
        _require_path(path, "tree view path")

        sa_dir = os.path.join(path, "structure_arrays")
        paths = Tree.get_paths(sa_dir)
        arrays = _load_required_arrays(paths)
        optional_arrays = _load_optional_arrays(paths)
        segmentation = _load_segmentation_image(os.path.join(path, "segmentation.png"))
        radius = _load_optional_array(paths, "radius")

        metadata = load_json(os.path.join(path, "metadata.json"))

        metadata['view_path'] = path

        _validate_tree_array_shapes(
            arrays,
            config,
            optional_arrays=optional_arrays,
        )

        return Tree.from_arrays(
            {**arrays, **optional_arrays},
            metadata=metadata,
            segmentation=segmentation,
            radius=radius,
            config=config,
        )


    @classmethod
    def get_paths(cls, path):
        paths = tree_path_map(path)
        keys = (
            *TREE_ARRAY_KEYS,
            "segmentation",
            "radius",
            "metadata",
        )
        return {
            key: str(paths[key])
            for key in keys
        }


    @classmethod
    def load(cls, path, config=None):
        if config is None:
            config = Tree.default_config()

        paths = Tree.get_paths(path)
        arrays = _load_required_arrays(paths)
        optional_arrays = _load_optional_arrays(paths)
        segmentation = _load_segmentation_image(paths["segmentation"])
        radius = _load_optional_array(paths, "radius")
        _validate_tree_array_shapes(
            arrays,
            config,
            optional_arrays=optional_arrays,
        )

        return Tree.from_arrays(
            {**arrays, **optional_arrays},
            metadata=load_json(paths['metadata']),
            segmentation=segmentation,
            radius=radius,
            config=config,
        )


    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        paths = Tree.get_paths(path)
        _validate_optional_edge_pair({"edges": self.edges, "edges_mask": self.edges_mask})

        arrays = TreeArrays(
            branch_mask=self.branch_mask,
            branches=self.branches,
            children=self.children,
            depth=self.depth,
            n_children=self.n_children,
            node_mask=self.node_mask,
            parents=self.parents,
            pos=self.pos,
            root_mask=self.root_mask,
            topology=self.topology,
            edges=self.edges,
            edges_mask=self.edges_mask,
        ).to_mapping()

        for key_str in TREE_REQUIRED_ARRAY_KEYS:
            save_array(paths[key_str], arrays[key_str])
        for key_str in TREE_OPTIONAL_ARRAY_KEYS:
            _save_tree_array(paths[key_str], arrays[key_str])
        save_json(paths['metadata'], self.metadata)

        if self.segmentation is not None:
            save_image(paths['segmentation'], _segmentation_for_save(self.segmentation))

        if self.radius is not None:
            save_array(paths['radius'], self.radius)


    def to_element(self) -> Element:
        fields = rename_fields(
            TreeArrays(
                branch_mask=self.branch_mask,
                branches=self.branches,
                children=self.children,
                depth=self.depth,
                n_children=self.n_children,
                node_mask=self.node_mask,
                parents=self.parents,
                pos=self.pos,
                root_mask=self.root_mask,
                topology=self.topology,
                edges=self.edges,
                edges_mask=self.edges_mask,
            ).to_mapping(),
            {"pos": "global_pos"},
        )
        fields.update(
            {
                "segmentation": self.segmentation,
                "radius": self.radius,
            }
        )
        return Element(**fields)


    @classmethod
    def from_element(cls, el: Element) -> Tree:
        return Tree.from_arrays(
            {
                **el.to_dict(include_none=True),
                "pos": getattr(el, "pos", el.global_pos),
            },
            segmentation=el.segmentation,
            radius=el.radius,
            metadata=None,
        )


    def to_dict(self) -> dict:
        arrays = TreeArrays(
            branch_mask=self.branch_mask,
            branches=self.branches,
            children=self.children,
            depth=self.depth,
            n_children=self.n_children,
            node_mask=self.node_mask,
            parents=self.parents,
            pos=self.pos,
            root_mask=self.root_mask,
            topology=self.topology,
            edges=self.edges,
            edges_mask=self.edges_mask,
        ).to_mapping()
        return {
            **arrays,
            "segmentation": self.segmentation,
            "radius": self.radius,
            "metadata": self.metadata,
        }


    @classmethod
    def from_dict(cls, tree_dict, config=None) -> Tree:
        if config is None:
            config = Tree.default_config()

        payload = select_known_fields(tree_dict, TREE_PAYLOAD_SPECS)
        return Tree.from_arrays(
            TreeArrays.from_mapping(tree_dict),
            segmentation=payload.get("segmentation"),
            radius=payload.get("radius"),
            metadata=tree_dict.get("metadata"),
            config=config,
        )


    @classmethod
    def array_keys(cls) -> list[str]:
        return list(TREE_ARRAY_KEYS)


    @classmethod
    def array_dtypes(cls) -> dict:
        return dict(TREE_ARRAY_DTYPES)


    @classmethod
    def send_keys(cls) -> list:
        return list(TREE_ARRAY_KEYS)


    def array_shapes(self) -> dict:
        arr_dict = self.to_dict()
        arr_keys = Tree.array_keys()
        ret = {}
        for key_str in arr_keys:
            if key_str in arr_dict.keys() and arr_dict[key_str] is not None:
                ret[key_str] = list(arr_dict[key_str].shape)
        if self.segmentation is not None:
            ret['segmentation'] = list(self.segmentation.shape)
        if self.radius is not None:
            ret['radius'] = list(self.radius.shape)
        return ret


    def send_zmq(self, socket, flags=0, copy=True, track=False) -> None:
        _require_optional_dependency(zmq, "pyzmq")
        arr_dict = self.to_dict()
        arr_shapes = self.array_shapes()
        payload_keys = _zmq_payload_keys(arr_shapes)
        socket.send_json(arr_shapes, flags | zmq.SNDMORE)
        socket.send_json(self.metadata, flags | zmq.SNDMORE)
        for i in range(0, len(payload_keys)):
            key_str = payload_keys[i]
            f = flags
            if i < len(payload_keys) - 1:
                f = f | zmq.SNDMORE
            socket.send(
                _as_zmq_array(key_str, arr_dict[key_str]),
                f,
                copy=copy,
                track=track,
            )


    @classmethod
    def recv_zmq(cls, socket, flags=0, copy=True, track=False, recv_segmentation=False,
                 recv_radius=False) -> Tree:
        _require_optional_dependency(zmq, "pyzmq")
        arr_shapes = socket.recv_json(flags=flags)
        metadata = socket.recv_json(flags=flags)
        payload_keys = _zmq_payload_keys(arr_shapes)
        tree_dict = {
            "metadata": metadata
        }
        for key_str in payload_keys:
            msg = socket.recv(flags=flags, copy=copy, track=track)
            if key_str == 'segmentation' and not recv_segmentation:
                continue
            if key_str == 'radius' and not recv_radius:
                continue
            tree_dict[key_str] = _decode_zmq_array(
                msg,
                key_str,
                arr_shapes[key_str],
            )
        return Tree.from_dict(_add_missing_optional_tree_payloads(tree_dict))


    def leaf_mask(self):
        return self.node_mask * (self.n_children == 0).astype(np.float32)


    def random_decoding_order(self):
        '''
            Returns:

                order   :   list of node indices in random order
                pivots  :   list of candidate pivots

            Notes

                This method returns a random node order which guarantees that
                nodes preserve their proximal/distal relationships. Furthermore,
                sibling nodes are guaranteed to be adjacent to one another in this
                order.

                This method also returns a list of candidate pivots which exclude
                those which would result in the query's sibling not being present
                in the lhs tree.

                Finally, this method ensures that the query nodes specified by the
                pivots all have child nodes
        '''
        dec_set = [[0]]
        order = []
        while len(dec_set) > 0:
            dec_set_idx = np.random.choice(len(dec_set), 1, replace=False)[0]
            node_idxs = dec_set[dec_set_idx]
            dec_set.remove(node_idxs)
            for node_idx in node_idxs:
                order.append(node_idx)
                child_idxs = []
                for k in range(0, self.children.shape[1]):
                    child_idx = self.children[node_idx, k]
                    if child_idx > 0:
                        child_idxs.append(child_idx)
                np.random.shuffle(child_idxs) # randomise order of child_idxs
                if len(child_idxs) > 0:
                    dec_set.append(child_idxs)

        pivots = [0]
        for order_idx in range(0, len(order)):
            node_idx = order[order_idx]
            n_children = 0
            for k in range(0, self.children.shape[1]):
                if self.children[node_idx, k] != -1:
                    n_children += 1
            accept_node = True
            if node_idx != 0:
                pidx = np.argmax(self.parents[node_idx, :])
                sidxs = [] # get sibling indices
                for k in range(0, self.children.shape[-1]):
                    c = self.children[pidx, k]
                    if c != -1 and c != node_idx:
                        sidxs.append(c)
                for sidx in sidxs:
                    soidx = order.index(sidx)
                    if soidx > order_idx:
                        accept_node = False
            if accept_node and n_children > 0:
                pivots.append(order_idx + 1)

        return order, pivots


    def get_bounds(self, order, minpos=None, maxpos=None):
        ret_minpos = 0
        ret_maxpos = len(order) - 1
        if maxpos is not None:
            ret_maxpos = min(ret_maxpos, maxpos)
        if minpos is not None:
            ret_minpos = max(ret_minpos, minpos)
        return ret_minpos, ret_maxpos


    def filter_pivots(self, order, pivots, minpos=None, maxpos=None):
        '''
            Arguments

                order               :   list (or array) of node indices in
                                        random order
                pivots              :   list (or array) of pivots
                minpos              :   (inclusive) minimum order pos of
                                        pivot nodes
                maxpos              :   (inclusive) maximum order pos of
                                        pivot nodes

            Returns

                filtered_pivots     :   array of pivot indices within the
                                        specified pos range (referring to
                                        positions in the order sequence)

        '''
        if isinstance(order, list):
            order = np.array(order)

        minpos, maxpos = self.get_bounds(order, minpos, maxpos)
        pivot_range = np.arange(minpos, maxpos + 1)
        filtered_pivots = []
        for pivot in pivots:
            if pivot in pivot_range:
                filtered_pivots.append(pivot)

        return filtered_pivots


    def random_pivot(self, order, pivots, minpos=None, maxpos=None):
        '''
            Arguments

                order           :   list (or array) of node indices in random order
                minpos          :   list (or array) of node indices in random order
                maxpos          :   list (or array) of node indices in random order

            Returns

                pivot_idx       :   int

        '''

        if maxpos == 0:
            return 0

        filtered_pivots = self.filter_pivots(
            order, pivots, minpos, maxpos
        )
        return np.random.choice(filtered_pivots)


    def get_split_matrices(self, order, pivot_idx):
        '''
            Arguments:

                order               :   list (or array) of node indices in random order
                pivot_idx           :   int

            Returns

                node_mask_lhs       :   binary mask indicating which nodes
                                        are in the first half of the decoding
                                        order
                pivot_idx           :   index of the node which is the pivot
                branches_lhs        :   branches matrix for the first half of
                                        the tree
                branch_mask_lhs     :   binary mask indicating which branches
                                        are in the first half of the tree
                n_children_lhs      :   number of children for the first half
                                        of the tree
                radius_lhs          :   radius matrix for the first half of the
                                        tree

            Notes

                The n_children_lhs term does not reflect the topology of the lhs
                tree. It is simply for every node in the first half of the tree,
                the number of children in the full tree. This is a static property
                invariant to the decoding order, much like the X/Y coordinates of
                the nodes.
        '''

        mask = np.zeros_like(self.node_mask)
        if isinstance(order, list):
            order = np.array(order)

        order_lhs = []
        if pivot_idx > 0:
            order_lhs = order[0:pivot_idx]
            mask[order_lhs] = 1.0

        # split the branches matrix
        branches_isin = np.isin(
            self.branches, order_lhs
        ).astype(np.float32)
        branches_lhs = (
            branches_isin * self.branches +
            (1.0 - branches_isin) * -1.0
        ).astype(np.int32)

        # retain only unique non-negative non-null (only root) branches
        branches_lhs = np.unique(branches_lhs, axis=0)
        negnull_branch_idxs = np.where(branches_lhs[:, 1] <= 0)
        if len(negnull_branch_idxs[0]) > 0:
            branches_lhs = np.delete(branches_lhs, negnull_branch_idxs, 0)

        # filter out branches which terminate in a non-leaf node
        # (with respect to the lhs tree topology)
        non_leaf_idxs = set()
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

        pad_size = self.branches.shape[0] - branches_lhs.shape[0]
        pad_lhs = -np.ones((pad_size, branches_lhs.shape[1]), dtype=np.int32)
        branches_lhs = np.concatenate([branches_lhs, pad_lhs], axis=0)
        branch_mask_lhs = (branches_lhs[:, 0] >= 0).astype(np.float32)

        mask = np.expand_dims(mask, axis=-1)

        # build the pos matrix
        pos = self.pos * mask

        mask = mask[:, 0]

        # build the radius matrix
        radius = None
        if self.radius is not None:
            radius = self.radius * mask

        # build the depth matrix
        depth = self.depth * mask - (1.0 - mask) * np.ones_like(self.depth)
        depth = depth.astype(np.int32)

        query_idx = -1
        if pivot_idx > 0:
            query_idx = order_lhs[-1]

        # build the query matrix
        query = np.zeros_like(self.node_mask)
        if pivot_idx > 0:
            query[query_idx] = 1.0

        # build the query_children mask
        query_children = np.zeros_like(self.node_mask)
        if pivot_idx == 0:
            query_children[0] = 1.0
        else:
            for k in range(0, self.children.shape[1]):
                child_idx = self.children[query_idx, k]
                if child_idx >= 0:
                    query_children[child_idx] = 1.0

        # build the n_children_lhs matrix
        n_children_lhs = (
            self.n_children * mask -
            (1.0 - mask) * np.ones_like(self.n_children)
        ).astype(np.int32)

        # build the topology_lhs matrix
        mask_r = np.expand_dims(mask, axis=-1)
        topology_lhs = (
            self.topology * mask_r -
            (1.0 - mask_r) * np.ones_like(self.topology)
        ).astype(np.float32)

        edges_lhs = None
        edges_mask_lhs = None
        if self.edges is not None and self.edges_mask is not None:
            # Retain only edges whose endpoints are active in the LHS tree.
            active_node_idxs = np.argwhere(mask > 0.5)[:, 0]
            emask_a = np.isin(self.edges[:, 0], active_node_idxs).astype(np.float32)
            emask_b = np.isin(self.edges[:, 1], active_node_idxs).astype(np.float32)
            edges_mask_lhs = self.edges_mask * emask_a * emask_b
            edges_lhs = (
                self.edges * edges_mask_lhs.reshape(-1, 1) -
                np.ones_like(self.edges) * (1.0 - edges_mask_lhs.reshape(-1, 1))
            ).astype(np.int32)

        ret = {
            'pivot_idx': pivot_idx,
            'query_idx': query_idx,
            'node_mask_lhs': mask,
            'branches_lhs': branches_lhs,
            'branch_mask_lhs': branch_mask_lhs,
            'pos_lhs': pos,
            'depth_lhs': depth,
            'n_children_lhs': n_children_lhs,
            'topology_lhs': topology_lhs,
            'query': query,
            'query_children': query_children,
            'edges_lhs': edges_lhs,
            'edges_mask_lhs': edges_mask_lhs
        }

        if radius is not None:
            ret['radius_lhs'] = radius

        return ret


    def random_decoding_split(self, minpos=None, maxpos=None, order_seed=None):
        '''
            Arguments:

                minpos              :   (inclusive) minimum depth of pivot nodes
                maxpos              :   (inclusive) maximum depth of pivot nodes

            Returns:

                node_mask_lhs       :   binary mask indicating which nodes
                                        are in the first half of the decoding
                                        order
                pivot_idx           :   index of the node which is the pivot
                branches_lhs        :   branches matrix for the first half of
                                        the tree
                branch_mask_lhs     :   binary mask indicating which branches
                                        are in the first half of the tree


            Notes

                This method generates a binary mask which splits
                the tree into two halves. This split is guaranteed
                to preserve the proximal/distal relationships of
                the nodes. For example, if a node is included in this
                mask, then all proximal nodes of that node are also
                included in this mask.

                Note that the pivot is selected in the range [0, n_nodes]
                If the pivot is 0, then the left half of the split is empty,
                and the right half is the full tree.
                If the pivot is n_nodes - 1, then the left half of the split
                is the full tree, and the right half is empty.
        '''
        local_seed = np.random.randint(2**32-1, dtype=np.uint32)
        if order_seed is not None:
            local_seed = order_seed
        with temporary_seed(local_seed):
            order, pivots = self.random_decoding_order()
        order = np.array(order)
        pivot_idx = self.random_pivot(order, pivots, minpos, maxpos)
        split_matrices = self.get_split_matrices(order, pivot_idx)
        return split_matrices


    @classmethod
    def get_query_gaussian(cls, height, width, query_idx, pos_np,
                           world_to_image, sigma=10.0):
        gaussian = np.zeros((height, width), dtype=np.float32)
        if query_idx >= 0:
            pos = pos_np[query_idx, :]
            mag = world_to_image['magnification']
            x_offset = world_to_image['x_offset']
            y_offset = world_to_image['y_offset']
            xi = pos[0] * mag + x_offset
            yi = pos[1] * mag + y_offset
            gaussian = gaussian_blob_single(
                width, height, xi, yi, sigma
            )
        return gaussian


    def sample_points_aux(self, node_idx: int, ne: int = 100) -> npt.NDArray[np.float32]:
        points = self.pos[node_idx].reshape(1, -1)
        for c in range(0, self.children.shape[1]):
            cidx = self.children[node_idx, c]
            if cidx != -1:
                delta = self.pos[cidx] - self.pos[node_idx]
                for k in range(0, ne - 1):
                    f = float(k + 1) / float(ne)
                    pk = self.pos[node_idx, :] + f * delta
                    points = np.concatenate(
                        [points, pk.reshape(1, -1)], axis=0
                    )
                cpoints = self.sample_points_aux(cidx)
                points = np.concatenate([points, cpoints], axis=0)
        return points


    def sample_points(self, ne: int = 100) -> npt.NDArray[np.float32]:
        if np.sum(self.node_mask) <= 0.5:
            raise ValueError("cannot sample points from a tree with no active nodes")
        return self.sample_points_aux(0)

# --- end public data: tree.py ---
