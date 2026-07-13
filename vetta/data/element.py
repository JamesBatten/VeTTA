from __future__ import annotations

import os
from collections.abc import Callable
from collections.abc import Mapping
from typing import Protocol

import numpy as np

from .field_specs import ELEMENT_CANDIDATE_SPECS
from .field_specs import ELEMENT_FIELD_SPECS
from .field_specs import ELEMENT_OPTIONAL_PAYLOAD_SPECS
from .field_specs import ELEMENT_SPLIT_SUPERVISION_SPECS
from .field_specs import ELEMENT_TREE_SPECS
from .field_specs import collect_present_fields
from .field_specs import dtype_map
from .field_specs import field_names
from .field_specs import select_known_fields
from .field_specs import torch_dtype_name_map
from vetta.utils import to_array
from vetta.utils import to_tensor
from vetta.utils import check_dtype
from vetta.utils import save_array_txt
from vetta.utils import save_string

try:
    import torch
except ModuleNotFoundError:
    torch = None

# --- begin public data: element.py ---
TREE_ARRAY_FIELDS = field_names(ELEMENT_TREE_SPECS)
OPTIONAL_PAYLOAD_FIELDS = field_names(ELEMENT_OPTIONAL_PAYLOAD_SPECS)
SPLIT_SUPERVISION_FIELDS = field_names(ELEMENT_SPLIT_SUPERVISION_SPECS)
CANDIDATE_FIELDS = field_names(ELEMENT_CANDIDATE_SPECS)

OPTIONAL_FIELDS = (
    *OPTIONAL_PAYLOAD_FIELDS,
    *SPLIT_SUPERVISION_FIELDS,
    *CANDIDATE_FIELDS,
)

ELEMENT_FIELDS = (*TREE_ARRAY_FIELDS, *OPTIONAL_FIELDS)
TEXT_ARRAY_FIELDS = tuple(
    field for field in ELEMENT_FIELDS if field != "segmentation"
)

NP_DTYPES = dtype_map(ELEMENT_FIELD_SPECS)


def _torch_dtypes() -> dict:
    return {
        field: getattr(torch, dtype_name)
        for field, dtype_name in torch_dtype_name_map(ELEMENT_FIELD_SPECS).items()
    }


def _present_fields(element: "Element", fields: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(field for field in fields if getattr(element, field) is not None)


def _set_fields(element: "Element", values: Mapping[str, object], fields: tuple[str, ...]) -> None:
    for field in fields:
        setattr(element, field, values.get(field))


def _validate_fields(values: Mapping[str, object]) -> None:
    unknown = sorted(set(values) - set(ELEMENT_FIELDS))
    if unknown:
        raise TypeError(f"Unknown Element fields: {unknown}")

    missing = sorted(field for field in TREE_ARRAY_FIELDS if field not in values)
    if missing:
        raise TypeError(f"Missing required Element fields: {missing}")


def _field_text(
    element: "Element",
    fields: tuple[str, ...],
    suffix: str,
    value_fn: Callable[[object], object],
) -> str:
    lines = []
    for field in fields:
        value = getattr(element, field)
        if value is not None:
            lines.append(f"{field}.{suffix}: {value_fn(value)}")
    return "\n".join(lines) + ("\n" if lines else "")


class SupportsToTensor(Protocol):
    def to_tensor(self) -> object:
        ...


class Element:

    __annotations__ = {
        **dict.fromkeys(TREE_ARRAY_FIELDS, "np.ndarray | torch.Tensor"),
        **dict.fromkeys(OPTIONAL_FIELDS, "np.ndarray | torch.Tensor | int | None"),
    }

    def __init__(self, **fields: np.ndarray | torch.Tensor | int | None) -> None:
        _validate_fields(fields)
        _set_fields(self, fields, ELEMENT_FIELDS)


    def to_tensor(self) -> Element:
        return Element.from_dict(self.tensor_dict())


    def tensor_dict(self) -> dict[str, torch.Tensor]:
        return {
            field: self.get_tensor(field)
            for field in collect_present_fields(
                self,
                (*TREE_ARRAY_FIELDS, *OPTIONAL_FIELDS),
            )
        }


    def to_dict(self, *, include_none: bool = False) -> dict[str, object]:
        return collect_present_fields(
            self,
            ELEMENT_FIELDS,
            include_none=include_none,
        )


    def update_fields(self, values: Mapping[str, object]) -> Element:
        for field, value in values.items():
            if field not in ELEMENT_FIELDS:
                raise KeyError(f"unknown Element field {field!r}")
            setattr(self, field, value)
        return self


    @classmethod
    def from_dict(cls, in_dict) -> Element:
        fields = select_known_fields(in_dict, ELEMENT_FIELD_SPECS)
        for field in TREE_ARRAY_FIELDS:
            if field in fields:
                fields[field] = to_tensor(fields[field])
        return cls(**fields)


    @classmethod
    def get_dtypes_np(cls) -> dict:
        return dict(NP_DTYPES)


    @classmethod
    def get_dtypes_pt(cls) -> dict:
        return _torch_dtypes()


    def check_dtypes(self, fail_on_cast=False):
        dtypes_np = Element.get_dtypes_np()
        dtypes_pt = Element.get_dtypes_pt()
        for key_str in dtypes_np.keys():
            arr = getattr(self, key_str)
            arr = check_dtype(
                arr, key_str, dtypes_np, dtypes_pt,
                fail_on_cast=fail_on_cast
            )
            setattr(self, key_str, arr)


    def check_arrays(self):
        # fix n_children array
        nlocs = np.argwhere(self.node_mask < 0.5)[:, 0]
        self.n_children[nlocs] = -1

        if self.node_mask_lhs is not None:
            nlocs_lhs = np.argwhere(self.node_mask_lhs < 0.5)[:, 0]
            # fix n_children_lhs array
            if self.n_children_lhs is not None:
                self.n_children_lhs[nlocs_lhs] = -1

            # fix topology_lhs array
            if self.topology_lhs is not None:
                self.topology_lhs[nlocs_lhs] = 0.0


    def get_float(self, name) -> float:
        f = getattr(self, name)
        if isinstance(f, float):
            return f
        return float(f[0])


    def get_array(self, name) -> np.ndarray:
        a = getattr(self, name)
        if isinstance(a, np.ndarray):
            return a
        return to_array(a)


    def get_tensor(self, name) -> torch.Tensor:
        t = getattr(self, name)
        if isinstance(t, torch.Tensor):
            return t
        return to_tensor(t)


    def save_arrays_txt(self, output_dir):
        for field in _present_fields(self, TEXT_ARRAY_FIELDS):
            save_array_txt(
                os.path.join(output_dir, f"{field}.txt"),
                to_array(getattr(self, field)),
            )


    def save_array_shapes(self, path):
        save_string(
            path,
            _field_text(
                self,
                ELEMENT_FIELDS,
                "shape",
                lambda value: to_array(value).shape,
            ),
        )


    def save_array_dtypes(self, path):
        save_string(
            path,
            _field_text(
                self,
                ELEMENT_FIELDS,
                "dtype",
                lambda value: to_array(value).dtype,
            ),
        )

# --- end public data: element.py ---


class ToTensor:
    """Transform an element-like object by calling its `to_tensor()` method."""

    def __call__(self, element: SupportsToTensor | None) -> object | None:
        if element is None:
            return None
        return element.to_tensor()
