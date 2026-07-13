from __future__ import annotations

import math
import os
import shutil
from dataclasses import dataclass

from .tree import Tree
from .field_specs import TREE_OPTIONAL_ARRAY_SPECS
from .field_specs import TREE_REQUIRED_ARRAY_SPECS
from .field_specs import field_names
import numpy as np

from vetta.utils import load_array
from vetta.utils import load_image
from vetta.utils import load_json
from vetta.utils import mkdir_custom
from vetta.utils import save_array
from vetta.utils import save_image
from vetta.utils import save_json

# --- begin public data: chunk.py ---
def _require_path(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _tree_attr_present(tree: Tree, attr_name: str) -> bool:
    return getattr(tree, attr_name) is not None


def _all_trees_have(trees: list[Tree], attr_name: str) -> bool:
    return all(_tree_attr_present(tree, attr_name) for tree in trees)


def _any_tree_has(trees: list[Tree], attr_name: str) -> bool:
    return any(_tree_attr_present(tree, attr_name) for tree in trees)


def _require_consistent_optional(trees: list[Tree], attr_name: str) -> bool:
    has_any = _any_tree_has(trees, attr_name)
    has_all = _all_trees_have(trees, attr_name)
    if has_any and not has_all:
        raise ValueError(f"optional tree field {attr_name!r} must be present for every tree or no trees")
    return has_all


def _require_consistent_edge_arrays(trees: list[Tree]) -> bool:
    has_edges = _require_consistent_optional(trees, "edges")
    has_edges_mask = _require_consistent_optional(trees, "edges_mask")
    if has_edges != has_edges_mask:
        raise ValueError("optional tree fields 'edges' and 'edges_mask' must be present together")
    return has_edges


def _stack_tree_attr(trees: list[Tree], attr_name: str):
    return np.stack([getattr(tree, attr_name) for tree in trees], axis=0)


@dataclass
class ChunkArrays:
    arrays: dict[str, np.ndarray]

    @classmethod
    def from_trees(cls, trees: list[Tree]) -> "ChunkArrays":
        arrays = {
            attr_name: _stack_tree_attr(trees, attr_name)
            for attr_name in field_names(TREE_REQUIRED_ARRAY_SPECS)
        }
        if _require_consistent_edge_arrays(trees):
            arrays.update(
                {
                    attr_name: _stack_tree_attr(trees, attr_name)
                    for attr_name in field_names(TREE_OPTIONAL_ARRAY_SPECS)
                }
            )
        return cls(arrays)

    def save(self, paths: dict[str, str]) -> None:
        for attr_name, array in self.arrays.items():
            save_array(paths[attr_name], array)


def _segmentation_for_chunk(segmentation):
    if len(segmentation.shape) == 2:
        return np.expand_dims(segmentation, axis=-1)
    if len(segmentation.shape) == 3 and segmentation.shape[-1] in (1, 3):
        return segmentation
    raise ValueError(f"tree segmentation has unsupported shape {list(segmentation.shape)}")


class TreeChunk:

    def __init__(self, trees):
        self.trees = trees


    def save(self, path):
        if len(self.trees) == 0:
            raise ValueError("cannot save an empty TreeChunk")
        if os.path.exists(path):
            shutil.rmtree(path)
        mkdir_custom(path)

        paths = TreeChunk.paths(path)
        n_trees = len(self.trees)

        save_segmentation = _require_consistent_optional(self.trees, "segmentation")
        save_world_to_image = _require_consistent_optional(self.trees, "world_to_image")
        save_radius = _require_consistent_optional(self.trees, "radius")
        chunk_arrays = ChunkArrays.from_trees(self.trees)

        height = None
        width = None
        n_horiz = None
        n_vert = None
        if save_segmentation:
            height = self.trees[0].segmentation.shape[0]
            width = self.trees[0].segmentation.shape[1]
            n_horiz = int(np.ceil(np.sqrt(n_trees)))
            n_vert = int(np.ceil(float(n_trees) / float(n_horiz)))

        x_offset_list = []
        y_offset_list = []
        magnification_list = []

        for i in range(0, n_trees):
            if save_world_to_image:
                x_offset_list.append(self.trees[i].world_to_image['x_offset'])
                y_offset_list.append(self.trees[i].world_to_image['y_offset'])
                magnification_list.append(self.trees[i].world_to_image['magnification'])

        x_offset_np = np.array(x_offset_list)
        y_offset_np = np.array(y_offset_list)
        magnification_np = np.array(magnification_list)
        segmentation_np = None
        slice_coords_json = None
        radius_np = None

        if save_segmentation:
            segmentation_np = np.zeros((height * n_vert, width * n_horiz, 3), dtype=np.uint8)
            slice_coords_json = []
            for i in range(0, n_trees):
                a0 = (i % n_horiz) * width
                b0 = (i // n_horiz) * height
                slice_coords_json.append([a0, b0, a0 + width, b0 + height])
                segmentation = _segmentation_for_chunk(self.trees[i].segmentation)
                if segmentation.shape[-1] == 1:
                    segmentation = np.tile(segmentation, (1, 1, 3))
                segmentation_np[b0:b0+height, a0:a0+width, :] = segmentation

        if save_radius:
            radius_np = _stack_tree_attr(self.trees, "radius")

        chunk_arrays.save(paths)
        if save_world_to_image:
            save_array(paths['x_offset'], x_offset_np)
            save_array(paths['y_offset'], y_offset_np)
            save_array(paths['magnification'], magnification_np)

        if save_segmentation:
            save_image(paths['segmentation'], segmentation_np)
            save_json(paths['slice_coords'], slice_coords_json)

        if save_radius:
            save_array(paths['radius'], radius_np)

        metadata = []
        for k in range(0, n_trees):
            metadata.append(self.trees[k].metadata)

        save_json(paths['metadata'], metadata)


    @classmethod
    def load(cls, path, load_segmentation=False) -> TreeChunk:
        paths = cls.resolve_paths(path)
        arrays = cls._load_raw_arrays(paths)
        structure_config = cls._infer_structure_config(
            path=path,
            pos_np=arrays['pos'],
            node_mask_np=arrays['node_mask'],
            children_np=arrays['children'],
            depth_np=arrays['depth'],
        )
        arrays.update(cls._load_or_make_structure_arrays(paths, arrays, structure_config))
        arrays.update(cls._load_or_make_edge_arrays(paths, arrays, structure_config))
        payload = cls._load_payload(paths, load_segmentation=load_segmentation)
        trees = cls._build_trees(arrays, payload, structure_config, load_segmentation)
        return cls(trees)


    @classmethod
    def _load_raw_arrays(cls, paths):
        return {
            'pos': load_array(paths['pos']),
            'node_mask': load_array(paths['node_mask']),
            'children': cls._normalise_children_array(load_array(paths['children'])),
            'root_mask': load_array(paths['root_mask']),
            'topology': cls._normalise_signed_array(load_array(paths['topology'])),
            'depth': cls._normalise_signed_array(load_array(paths['depth'])),
            'n_children': cls._normalise_signed_array(load_array(paths['n_children'])),
        }


    @classmethod
    def _load_or_make_structure_arrays(cls, paths, arrays, structure_config):
        parents_np = cls._load_optional_array(paths['parents'])
        if parents_np is None:
            parents_np = cls._make_parents_batch(arrays['children'], arrays['node_mask'])

        branches_np = cls._load_optional_array(paths['branches'])
        branch_mask_np = cls._load_optional_array(paths['branch_mask'])
        if branches_np is None:
            branches_np, branch_mask_np = cls._make_branches_batch(
                children_np=arrays['children'],
                node_mask_np=arrays['node_mask'],
                root_mask_np=arrays['root_mask'],
                max_branches=structure_config['max_branches'],
                max_branch_len=structure_config['max_branch_len'],
            )
        elif branch_mask_np is None:
            branch_mask_np = (branches_np[:, :, 0] >= 0).astype(np.float32)

        return {
            'parents': parents_np,
            'branches': branches_np,
            'branch_mask': branch_mask_np,
        }


    @classmethod
    def _load_or_make_edge_arrays(cls, paths, arrays, structure_config):
        edges_np = cls._load_optional_array(paths['edges'])
        edges_mask_np = cls._load_optional_array(paths['edges_mask'])
        if edges_np is None or edges_mask_np is None:
            edges_np, edges_mask_np = cls._make_edges_batch(
                children_np=arrays['children'],
                node_mask_np=arrays['node_mask'],
                max_edges=structure_config['max_edges'],
            )
        return {
            'edges': edges_np,
            'edges_mask': edges_mask_np,
        }


    @classmethod
    def _load_payload(cls, paths, *, load_segmentation):
        payload = {
            'x_offset': cls._load_optional_array(paths['x_offset']),
            'y_offset': cls._load_optional_array(paths['y_offset']),
            'magnification': cls._load_optional_array(paths['magnification']),
            'metadata': load_json(paths['metadata']) if os.path.exists(paths['metadata']) else None,
            'radius': load_array(paths['radius']) if os.path.exists(paths['radius']) else None,
            'segmentation': None,
            'slice_coords': None,
        }
        if load_segmentation:
            _require_path(paths['segmentation'], "chunk segmentation image")
            _require_path(paths['slice_coords'], "chunk segmentation slice coordinates")
            payload['segmentation'] = load_image(paths['segmentation'])[:, :, 0:1]
            payload['slice_coords'] = load_json(paths['slice_coords'])
        return payload


    @classmethod
    def _build_trees(cls, arrays, payload, structure_config, load_segmentation):
        pos_np = arrays['pos']
        children_np = arrays['children']

        trees = []
        for tree_idx in range(0, pos_np.shape[0]):
            segmentation_i = cls._segmentation_at(payload, tree_idx, load_segmentation)
            radius_i = None if payload['radius'] is None else payload['radius'][tree_idx, :]
            metadata_i = None if payload['metadata'] is None else payload['metadata'][tree_idx]
            tree_config_i = {
                **Tree.default_config(),
                **structure_config,
                'max_nodes': int(pos_np.shape[1]),
                'max_children': int(children_np.shape[2]),
            }
            tree_arrays = {
                attr_name: batch[tree_idx]
                for attr_name, batch in arrays.items()
            }
            trees.append(Tree.from_arrays(
                tree_arrays,
                segmentation=segmentation_i,
                radius=radius_i,
                metadata=metadata_i,
                world_to_image=cls._world_to_image_at(payload, tree_idx),
                config=tree_config_i,
            ))
        return trees


    @classmethod
    def _segmentation_at(cls, payload, tree_idx, load_segmentation):
        if not load_segmentation:
            return None
        slice_coords = payload['slice_coords']
        a0 = int(slice_coords[tree_idx][0])
        b0 = int(slice_coords[tree_idx][1])
        a1 = int(slice_coords[tree_idx][2])
        b1 = int(slice_coords[tree_idx][3])
        return payload['segmentation'][b0:b1, a0:a1, :]


    @classmethod
    def _world_to_image_at(cls, payload, tree_idx):
        x_offset_np = payload['x_offset']
        y_offset_np = payload['y_offset']
        magnification_np = payload['magnification']
        make_default = (
            x_offset_np is None
            or y_offset_np is None
            or magnification_np is None
            or x_offset_np.shape[0] <= tree_idx
            or y_offset_np.shape[0] <= tree_idx
            or magnification_np.shape[0] <= tree_idx
        )
        if make_default:
            return {
                'x_offset': 0.0,
                'y_offset': 0.0,
                'magnification': 1.0,
            }
        return {
            'x_offset': x_offset_np[tree_idx],
            'y_offset': y_offset_np[tree_idx],
            'magnification': magnification_np[tree_idx],
        }


    @classmethod
    def _load_optional_array(cls, path):
        if not os.path.exists(path):
            return None
        return load_array(path)


    @classmethod
    def _normalise_children_array(cls, children_np):
        return cls._normalise_signed_array(children_np)


    @classmethod
    def _normalise_signed_array(cls, array_np):
        if np.issubdtype(array_np.dtype, np.unsignedinteger):
            max_val = np.iinfo(array_np.dtype).max
            array_np = array_np.astype(np.int64, copy=True)
            array_np[array_np == max_val] = -1
            return array_np.astype(np.int32)
        return array_np


    @classmethod
    def _infer_structure_config(cls, path, pos_np, node_mask_np, children_np, depth_np):
        tree_config_path = cls.resolve_paths(path)['tree_config']
        config = {}
        if os.path.exists(tree_config_path):
            config = load_json(tree_config_path)

        max_nodes = int(pos_np.shape[1])
        max_children = int(children_np.shape[-1]) if children_np.ndim >= 3 else 2

        max_branch_len = config.get('max_branch_len')
        if max_branch_len is None:
            max_branch_len = config.get('branch_len')
        if max_branch_len is None:
            active_depth = depth_np[node_mask_np > 0.5]
            inferred_branch_len = int(active_depth.max()) + 1 if active_depth.size > 0 else 1
            if max_nodes == 40 and max_children == 2:
                max_branch_len = max(15, inferred_branch_len)
            else:
                max_branch_len = inferred_branch_len

        max_branches = config.get('max_branches')
        if max_branches is None:
            leaf_counts = np.sum((children_np < 0).all(axis=-1) & (node_mask_np > 0.5), axis=1)
            inferred_max_branches = int(leaf_counts.max()) if leaf_counts.size > 0 else 1
            if max_nodes == 40 and max_children == 2:
                max_branches = max(20, inferred_max_branches)
            else:
                max_branches = max(1, inferred_max_branches, int(math.ceil(max_nodes / max(max_children, 1))))

        max_edges = config.get('max_edges')
        if max_edges is None:
            inferred_max_edges = int(np.sum(children_np >= 0, axis=(1, 2)).max()) if children_np.size > 0 else 1
            if max_nodes == 40 and max_children == 2:
                max_edges = max(50, inferred_max_edges)
            else:
                max_edges = max(1, inferred_max_edges)

        return {
            'max_branch_len': int(max_branch_len),
            'max_branches': int(max_branches),
            'max_edges': int(max_edges),
        }


    @classmethod
    def _make_parents_batch(cls, children_np, node_mask_np):
        n_trees = int(children_np.shape[0])
        n_nodes = int(children_np.shape[1])
        parents_np = np.zeros((n_trees, n_nodes, n_nodes), dtype=np.int32)
        for tree_idx in range(n_trees):
            parents_np[tree_idx] = cls._make_parents_single(
                children_np[tree_idx],
                node_mask_np[tree_idx],
            )
        return parents_np


    @classmethod
    def _make_parents_single(cls, children_i, node_mask_i):
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


    @classmethod
    def _make_branches_batch(cls, children_np, node_mask_np, root_mask_np, max_branches, max_branch_len):
        n_trees = int(children_np.shape[0])
        branches_np = -np.ones((n_trees, max_branches, max_branch_len), dtype=np.int32)
        branch_mask_np = np.zeros((n_trees, max_branches), dtype=np.float32)
        for tree_idx in range(n_trees):
            branches_i, branch_mask_i = cls._make_branches_single(
                children_np[tree_idx],
                node_mask_np[tree_idx],
                root_mask_np[tree_idx],
                max_branches=max_branches,
                max_branch_len=max_branch_len,
            )
            branches_np[tree_idx] = branches_i
            branch_mask_np[tree_idx] = branch_mask_i
        return branches_np, branch_mask_np


    @classmethod
    def _make_branches_single(cls, children_i, node_mask_i, root_mask_i, max_branches, max_branch_len):
        root_candidates = np.argwhere(root_mask_i > 0.5)[:, 0]
        if root_candidates.size == 0:
            active = np.argwhere(node_mask_i > 0.5)[:, 0]
            root_idx = int(active[0]) if active.size > 0 else 0
        else:
            root_idx = int(root_candidates[0])

        branches_list = []

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
            branches_i[branch_idx, :len(branch)] = np.array(branch, dtype=np.int32)
            branch_mask_i[branch_idx] = 1.0
        return branches_i, branch_mask_i


    @classmethod
    def _make_edges_batch(cls, children_np, node_mask_np, max_edges):
        n_trees = int(children_np.shape[0])
        edges_np = -np.ones((n_trees, max_edges, 2), dtype=np.int32)
        edges_mask_np = np.zeros((n_trees, max_edges), dtype=np.float32)
        for tree_idx in range(n_trees):
            edges_i, edges_mask_i = cls._make_edges_single(
                children_np[tree_idx],
                node_mask_np[tree_idx],
                max_edges=max_edges,
            )
            edges_np[tree_idx] = edges_i
            edges_mask_np[tree_idx] = edges_mask_i
        return edges_np, edges_mask_np


    @classmethod
    def _make_edges_single(cls, children_i, node_mask_i, max_edges):
        edges_list: list[list[int]] = []
        for parent_idx in range(children_i.shape[0]):
            if node_mask_i[parent_idx] <= 0.5:
                continue
            for child_idx in children_i[parent_idx]:
                if child_idx < 0 or child_idx >= children_i.shape[0]:
                    continue
                if node_mask_i[child_idx] <= 0.5:
                    continue
                edges_list.append([int(parent_idx), int(child_idx)])
        if len(edges_list) > max_edges:
            raise ValueError(
                "number of reconstructed edges "
                + str(len(edges_list))
                + " exceeds max_edges "
                + str(max_edges)
            )
        edges_i = -np.ones((max_edges, 2), dtype=np.int32)
        edges_mask_i = np.zeros((max_edges,), dtype=np.float32)
        for edge_idx, edge in enumerate(edges_list):
            edges_i[edge_idx] = np.array(edge, dtype=np.int32)
            edges_mask_i[edge_idx] = 1.0
        return edges_i, edges_mask_i


    @classmethod
    def paths(cls, path):
        return {
            'segmentation': os.path.join(path, 'segmentation.png'),
            'pos': os.path.join(path, "pos.npy"),
            'node_mask': os.path.join(path, "node_mask.npy"),
            'branch_mask': os.path.join(path, "branch_mask.npy"),
            'branches': os.path.join(path, "branches.npy"),
            'parents': os.path.join(path, "parents.npy"),
            'children': os.path.join(path, "children.npy"),
            'root_mask': os.path.join(path, "root_mask.npy"),
            'topology': os.path.join(path, "topology.npy"),
            'depth': os.path.join(path, "depth.npy"),
            'n_children': os.path.join(path, "n_children.npy"),
            'edges': os.path.join(path, "edges.npy"),
            'edges_mask': os.path.join(path, "edges_mask.npy"),
            'x_offset': os.path.join(path, "x_offset.npy"),
            'y_offset': os.path.join(path, "y_offset.npy"),
            'magnification': os.path.join(path, "magnification.npy"),
            'radius': os.path.join(path, "radius.npy"),
            'slice_coords': os.path.join(path, 'slice_coords.json'),
            'metadata': os.path.join(path, "metadata.json"),
            'tree_config': os.path.join(path, "tree_config.json")
        }


    @classmethod
    def resolve_paths(cls, path):
        paths = cls.paths(path)

        def _resolve(base_path, *alternatives):
            candidates = (base_path,) + alternatives
            for candidate in candidates:
                if os.path.exists(candidate):
                    return candidate
            return base_path

        resolved = dict(paths)
        for key in (
            'pos',
            'node_mask',
            'branch_mask',
            'branches',
            'parents',
            'children',
            'root_mask',
            'topology',
            'depth',
            'n_children',
            'edges',
            'edges_mask',
            'x_offset',
            'y_offset',
            'magnification',
            'radius',
        ):
            resolved[key] = _resolve(paths[key], paths[key].replace('.npy', '.npz'))

        resolved['segmentation'] = _resolve(
            paths['segmentation'],
            paths['segmentation'].replace('.png', '.jpg'),
        )
        return resolved

# --- end public data: chunk.py ---
