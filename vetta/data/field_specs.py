from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FieldSpec:
    name: str
    np_dtype: Any
    torch_dtype_name: str | None = None
    required: bool = True
    shape: tuple[str | int, ...] | None = None
    filename: str | None = None
    group: str = "core"
    aliases: tuple[str, ...] = ()


TREE_REQUIRED_ARRAY_SPECS = (
    FieldSpec("branch_mask", np.float32, "float32", True, ("max_branches",), "branch_mask.npy", "tree_array"),
    FieldSpec("branches", np.int64, "int64", True, ("max_branches", "max_branch_len"), "branches.npy", "tree_array"),
    FieldSpec("children", np.int64, "int64", True, ("max_nodes", "max_children"), "children.npy", "tree_array"),
    FieldSpec("depth", np.float32, "float32", True, ("max_nodes",), "depth.npy", "tree_array"),
    FieldSpec("n_children", np.int64, "int64", True, ("max_nodes",), "n_children.npy", "tree_array"),
    FieldSpec("node_mask", np.float32, "float32", True, ("max_nodes",), "node_mask.npy", "tree_array"),
    FieldSpec("parents", np.int64, "int64", True, ("max_nodes", "max_nodes"), "parents.npy", "tree_array"),
    FieldSpec("pos", np.float32, "float32", True, ("max_nodes", 2), "pos.npy", "tree_array"),
    FieldSpec("root_mask", np.float32, "float32", True, ("max_nodes",), "root_mask.npy", "tree_array"),
    FieldSpec("topology", np.int64, "int64", True, ("max_nodes", "topology_width"), "topology.npy", "tree_array"),
)

TREE_OPTIONAL_ARRAY_SPECS = (
    FieldSpec("edges", np.int64, "int64", False, ("max_edges", 2), "edges.npy", "tree_array"),
    FieldSpec("edges_mask", np.float32, "float32", False, ("max_edges",), "edges_mask.npy", "tree_array"),
)

TREE_ARRAY_SPECS = (*TREE_REQUIRED_ARRAY_SPECS, *TREE_OPTIONAL_ARRAY_SPECS)

TREE_PAYLOAD_SPECS = (
    FieldSpec("segmentation", np.uint8, "uint8", False, None, "segmentation.png", "tree_payload"),
    FieldSpec("radius", np.float32, "float32", False, ("max_nodes",), "radius.npy", "tree_payload", ("rad",)),
)

TREE_METADATA_SPECS = (
    FieldSpec("metadata", object, None, False, None, "metadata.json", "metadata"),
    FieldSpec("slice_coords", object, None, False, None, "slice_coords.json", "metadata"),
    FieldSpec("tree_config", object, None, False, None, "tree_config.json", "metadata"),
    FieldSpec("x_offset", np.float32, "float32", False, None, "x_offset.npy", "world_to_image"),
    FieldSpec("y_offset", np.float32, "float32", False, None, "y_offset.npy", "world_to_image"),
    FieldSpec("magnification", np.float32, "float32", False, None, "magnification.npy", "world_to_image"),
)

ELEMENT_TREE_SPECS = (
    FieldSpec("branch_mask", np.float32, "float32", True, ("max_branches",), group="element_tree"),
    FieldSpec("branches", np.int32, "int32", True, ("max_branches", "max_branch_len"), group="element_tree"),
    FieldSpec("children", np.int32, "int32", True, ("max_nodes", "max_children"), group="element_tree"),
    FieldSpec("depth", np.int32, "int32", True, ("max_nodes",), group="element_tree"),
    FieldSpec("n_children", np.int32, "int32", True, ("max_nodes",), group="element_tree"),
    FieldSpec("node_mask", np.float32, "float32", True, ("max_nodes",), group="element_tree"),
    FieldSpec("parents", np.float32, "float32", True, ("max_nodes", "max_nodes"), group="element_tree"),
    FieldSpec("global_pos", np.float32, "float32", True, ("max_nodes", 2), group="element_tree"),
    FieldSpec("root_mask", np.float32, "float32", True, ("max_nodes",), group="element_tree"),
    FieldSpec("topology", np.float32, "float32", True, ("max_nodes", "topology_width"), group="element_tree"),
    FieldSpec("edges", np.int32, "int32", True, ("max_edges", 2), group="element_tree"),
    FieldSpec("edges_mask", np.float32, "float32", True, ("max_edges",), group="element_tree"),
)

ELEMENT_OPTIONAL_PAYLOAD_SPECS = (
    FieldSpec("local_pos", np.float32, "float32", False, group="element_payload"),
    FieldSpec("chunk_index", np.int32, "int32", False, group="element_payload"),
    FieldSpec("chunk_epoch", np.int32, "int32", False, group="element_payload"),
    FieldSpec("pivot_idx", np.int32, "int32", False, group="element_payload"),
    FieldSpec("query_idx", np.int32, "int32", False, group="element_payload"),
    FieldSpec("qeidx", np.int32, "int32", False, group="element_payload"),
    FieldSpec("query", np.float32, "float32", False, group="element_payload"),
    FieldSpec("query_children", np.int32, "int32", False, group="element_payload"),
    FieldSpec("segmentation", np.uint8, "uint8", False, group="element_payload"),
    FieldSpec("radius", np.float32, "float32", False, group="element_payload", aliases=("rad",)),
)

ELEMENT_SPLIT_SUPERVISION_SPECS = (
    FieldSpec("local_pos_lhs", np.float32, "float32", False, group="element_split"),
    FieldSpec("branch_mask_lhs", np.float32, "float32", False, group="element_split"),
    FieldSpec("branches_lhs", np.int32, "int32", False, group="element_split"),
    FieldSpec("depth_lhs", np.int32, "int32", False, group="element_split"),
    FieldSpec("n_children_lhs", np.int32, "int32", False, group="element_split"),
    FieldSpec("node_mask_lhs", np.float32, "float32", False, group="element_split"),
    FieldSpec("topology_lhs", np.float32, "float32", False, group="element_split"),
    FieldSpec("edges_lhs", np.int32, "int32", False, group="element_split"),
    FieldSpec("edges_mask_lhs", np.float32, "float32", False, group="element_split"),
    FieldSpec("global_pos_lhs", np.float32, "float32", False, group="element_split"),
    FieldSpec("radius_lhs", np.float32, "float32", False, group="element_split"),
)

ELEMENT_CANDIDATE_SPECS = (
    FieldSpec("cand_tgt_sel", np.float32, "float32", False, group="element_candidate"),
    FieldSpec("cand_global_pos", np.float32, "float32", False, group="element_candidate"),
    FieldSpec("cand_local_pos", np.float32, "float32", False, group="element_candidate"),
    FieldSpec("cand_time", np.float32, "float32", False, group="element_candidate"),
    FieldSpec("cand_bnd_global_pos", np.float32, "float32", False, group="element_candidate"),
    FieldSpec("cand_sel", np.float32, "float32", False, group="element_candidate"),
    FieldSpec("cand_time_1hot", np.float32, "float32", False, group="element_candidate"),
    FieldSpec("cand_nidx_map", np.int32, "int32", False, group="element_candidate"),
)

ELEMENT_FIELD_SPECS = (
    *ELEMENT_TREE_SPECS,
    *ELEMENT_OPTIONAL_PAYLOAD_SPECS,
    *ELEMENT_SPLIT_SUPERVISION_SPECS,
    *ELEMENT_CANDIDATE_SPECS,
)


def field_names(specs: Iterable[FieldSpec]) -> tuple[str, ...]:
    return tuple(spec.name for spec in specs)


def spec_by_name(specs: Iterable[FieldSpec]) -> dict[str, FieldSpec]:
    return {spec.name: spec for spec in specs}


def required_field_names(specs: Iterable[FieldSpec]) -> tuple[str, ...]:
    return tuple(spec.name for spec in specs if spec.required)


def optional_field_names(specs: Iterable[FieldSpec]) -> tuple[str, ...]:
    return tuple(spec.name for spec in specs if not spec.required)


def dtype_map(specs: Iterable[FieldSpec]) -> dict[str, Any]:
    return {spec.name: spec.np_dtype for spec in specs}


def torch_dtype_name_map(specs: Iterable[FieldSpec]) -> dict[str, str]:
    return {
        spec.name: spec.torch_dtype_name
        for spec in specs
        if spec.torch_dtype_name is not None
    }


def filename_map(specs: Iterable[FieldSpec]) -> dict[str, str]:
    return {
        spec.name: spec.filename
        for spec in specs
        if spec.filename is not None
    }


def path_map(base_path: str | Path, specs: Iterable[FieldSpec]) -> dict[str, Path]:
    base_path = Path(base_path)
    return {
        spec.name: base_path / spec.filename
        for spec in specs
        if spec.filename is not None
    }


def select_known_fields(
    data: Mapping[str, Any],
    specs_or_names: Iterable[FieldSpec | str],
    *,
    include_aliases: bool = True,
) -> dict[str, Any]:
    specs = _coerce_specs(specs_or_names)
    selected = {}
    for spec in specs:
        if spec.name in data:
            selected[spec.name] = data[spec.name]
            continue
        if include_aliases:
            for alias in spec.aliases:
                if alias in data:
                    selected[spec.name] = data[alias]
                    break
    return selected


def collect_present_fields(
    source: Mapping[str, Any] | object,
    specs_or_names: Iterable[FieldSpec | str],
    *,
    include_none: bool = False,
) -> dict[str, Any]:
    specs = _coerce_specs(specs_or_names)
    collected = {}
    for spec in specs:
        present, value = _get_field(source, spec.name)
        if present and (include_none or value is not None):
            collected[spec.name] = value
    return collected


def rename_fields(data: Mapping[str, Any], rename_map: Mapping[str, str]) -> dict[str, Any]:
    return {
        rename_map.get(name, name): value
        for name, value in data.items()
    }


def resolve_existing_path(
    path: str | Path,
    alternatives: Iterable[str | Path] = (),
) -> Path:
    candidates = (Path(path), *(Path(candidate) for candidate in alternatives))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _coerce_specs(specs_or_names: Iterable[FieldSpec | str]) -> tuple[FieldSpec, ...]:
    specs = []
    for item in specs_or_names:
        if isinstance(item, FieldSpec):
            specs.append(item)
        else:
            specs.append(FieldSpec(str(item), object))
    return tuple(specs)


def _get_field(source: Mapping[str, Any] | object, name: str) -> tuple[bool, Any]:
    if isinstance(source, Mapping):
        if name in source:
            return True, source[name]
        return False, None
    if hasattr(source, name):
        return True, getattr(source, name)
    return False, None
