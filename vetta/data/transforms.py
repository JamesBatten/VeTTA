from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar
import warnings

import numpy as np
from scipy import ndimage

from .element import Element
from .element import ToTensor
from vetta.settings.data import TreeRotateSettings
from vetta.settings.data import TreeTranslateSettings
from vetta.settings.data import TreeZoomSettings
from vetta.utils import to_array
from vetta.utils import to_tensor

try:
    import torch
except ModuleNotFoundError:
    torch = None

T = TypeVar("T")


def active_node_indices(mask: np.ndarray | torch.Tensor) -> np.ndarray:
    return np.argwhere(to_array(mask).reshape(-1) > 0.5)[:, 0]


def active_positions_in_bounds(
    pos: np.ndarray | torch.Tensor,
    mask: np.ndarray | torch.Tensor,
    wrap_domain: tuple,
    *,
    ignore_aug_bounds: bool = False,
) -> bool:
    if ignore_aug_bounds:
        return True
    active_indices = active_node_indices(mask)
    if active_indices.size == 0:
        return True
    active_pos = to_array(pos)[active_indices, :]
    return bool(active_pos.min() >= wrap_domain[0] and active_pos.max() <= wrap_domain[1])


def warn_random_operation_failed(operation_name: str, attempts: int) -> None:
    warnings.warn(
        "{} failed after {} attempts".format(operation_name, attempts),
        RuntimeWarning,
        stacklevel=3,
    )


def retry_bounded_random_operation(
    *,
    operation_name: str,
    make_candidate: Callable[[], T],
    get_candidate_pos: Callable[[T], np.ndarray | torch.Tensor],
    mask: np.ndarray | torch.Tensor,
    wrap_domain: tuple,
    max_attempts: int = 10,
    ignore_aug_bounds: bool = False,
) -> T | None:
    for _attempt in range(max_attempts):
        candidate = make_candidate()
        candidate_pos = get_candidate_pos(candidate)
        if active_positions_in_bounds(
            candidate_pos,
            mask,
            wrap_domain,
            ignore_aug_bounds=ignore_aug_bounds,
        ):
            return candidate
    warn_random_operation_failed(operation_name, max_attempts)
    return None


def apply_lhs_position_transform(element: Element, transform: Callable) -> None:
    if element.global_pos_lhs is None:
        return
    if element.node_mask_lhs is None:
        raise ValueError("node_mask_lhs is required when global_pos_lhs is present")
    element.global_pos_lhs = transform(element.global_pos_lhs, element.node_mask_lhs)


def apply_segmentation_transform(element: Element, transform: Callable) -> None:
    if element.segmentation is not None:
        element.segmentation = transform(element.segmentation)


def mask_column_like(mask: np.ndarray | torch.Tensor, reference: np.ndarray | torch.Tensor):
    if torch is not None and isinstance(reference, torch.Tensor):
        return torch.as_tensor(mask, dtype=reference.dtype, device=reference.device).reshape(-1, 1)
    return to_array(mask).reshape(-1, 1)


@dataclass
class PosRotData:
    rotpos: np.ndarray | torch.Tensor
    valrads: float


class TreeRotate:
    """
    Randomly rotate tree positions, optional LHS positions, and segmentation in-place.
    """

    def __init__(
        self,
        max_degrees: float = 45.0,
        rot_center: tuple = (0.0, 0.0),
        fill: float = 0.0,
        prob: float = 0.5,
        wrap_domain: tuple = (-0.5, 0.5),
        ignore_aug_bounds: bool = False,
    ) -> None:
        self.maxrads = float(np.deg2rad(max_degrees))
        self.rot_center = rot_center
        self.fill = fill
        self.prob = prob
        self.wrap_domain = wrap_domain
        self.ignore_aug_bounds = ignore_aug_bounds

    @classmethod
    def from_settings(cls, settings=None) -> TreeRotate:
        if settings is None:
            settings = TreeRotateSettings()
        elif not isinstance(settings, TreeRotateSettings):
            raise TypeError(
                "TreeRotate.from_settings expected a TreeRotateSettings or None, "
                f"got {type(settings).__name__}"
            )
        return cls(
            max_degrees=settings.max_degrees,
            rot_center=settings.rot_center,
            fill=settings.fill,
            prob=settings.prob,
            wrap_domain=settings.wrap_domain,
            ignore_aug_bounds=settings.ignore_aug_bounds,
        )

    def __call__(self, el: Element) -> Element | None:
        if el is None:
            return None
        if np.random.uniform() > self.prob:
            return el

        wix = self.rot_center[0]
        wiy = self.rot_center[1]

        posrdat = TreeRotate.random_rotate(
            el.global_pos,
            el.node_mask,
            wix,
            wiy,
            self.maxrads,
            self.wrap_domain,
            ignore_aug_bounds=self.ignore_aug_bounds,
        )
        if posrdat is None:
            return None

        apply_lhs_position_transform(
            el,
            lambda pos, mask: TreeRotate.rotate_pos(pos, mask, wix, wiy, posrdat.valrads),
        )

        el.global_pos = posrdat.rotpos
        apply_segmentation_transform(
            el,
            lambda seg: TreeRotate.rotate_seg(seg, wix, wiy, posrdat.valrads),
        )

        return el

    @classmethod
    def random_rotate(
        cls,
        pos: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor,
        wix: float,
        wiy: float,
        maxrads: float,
        wrap_domain: tuple,
        max_attempts: int = 10,
        ignore_aug_bounds: bool = False,
    ) -> PosRotData | None:
        def make_candidate() -> PosRotData:
            valrads = np.random.uniform(-maxrads, maxrads)
            pos_rot = TreeRotate.rotate_pos(
                np.copy(pos),
                mask,
                wix,
                wiy,
                valrads,
            )
            return PosRotData(pos_rot, valrads)

        return retry_bounded_random_operation(
            operation_name="random_rotate",
            make_candidate=make_candidate,
            get_candidate_pos=lambda candidate: candidate.rotpos,
            mask=mask,
            wrap_domain=wrap_domain,
            max_attempts=max_attempts,
            ignore_aug_bounds=ignore_aug_bounds,
        )

    @classmethod
    def rotate_pos(
        cls,
        pos: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor,
        wix: float,
        wiy: float,
        valrads: float,
    ) -> np.ndarray | torch.Tensor:
        rot_mat = torch.tensor(
            np.array(
                [
                    [np.cos(valrads), -np.sin(valrads)],
                    [np.sin(valrads), np.cos(valrads)],
                ]
            )
        ).float()
        off_mat = torch.tensor(np.array([[wix, wiy]])).float()
        pos_pt = to_tensor(pos)
        pos_pt = (pos_pt - off_mat).float()
        mask_pt = to_tensor(mask)
        n = int(pos_pt.shape[0])
        rot_mat = rot_mat.unsqueeze(0).repeat(n, 1, 1)
        pos_pt = pos_pt.unsqueeze(-1)
        pos_pt = torch.matmul(rot_mat, pos_pt)
        pos_pt = pos_pt[:, :, 0] + off_mat
        pos_pt *= mask_pt.unsqueeze(-1)
        return to_array(pos_pt)

    @classmethod
    def rotate_seg(
        cls,
        seg: np.ndarray | torch.Tensor,
        wix: float,
        wiy: float,
        valrads: float,
    ) -> np.ndarray | torch.Tensor:
        ret_pt = isinstance(seg, torch.Tensor)
        assert wix == 0.5
        assert wiy == 0.5
        deg = -np.rad2deg(valrads)
        rot_seg = ndimage.rotate(to_array(seg), deg, reshape=False)
        if ret_pt:
            return to_tensor(rot_seg)
        return rot_seg


@dataclass
class PosTransData:
    transpos: np.ndarray
    delta_w: np.ndarray


class TreeTranslate:
    def __init__(
        self,
        max_delta: float,
        prob: float = 0.5,
        wrap_domain: tuple = (-0.5, 0.5),
        ignore_aug_bounds: bool = False,
    ) -> None:
        self.max_delta = max_delta
        self.prob = prob
        self.wrap_domain = wrap_domain
        self.ignore_aug_bounds = ignore_aug_bounds

    @classmethod
    def from_settings(cls, settings=None) -> TreeTranslate:
        if settings is None:
            settings = TreeTranslateSettings()
        elif not isinstance(settings, TreeTranslateSettings):
            raise TypeError(
                "TreeTranslate.from_settings expected a TreeTranslateSettings or None, "
                f"got {type(settings).__name__}"
            )
        return cls(
            max_delta=settings.max_delta,
            prob=settings.prob,
            wrap_domain=settings.wrap_domain,
            ignore_aug_bounds=settings.ignore_aug_bounds,
        )

    def __call__(self, el: Element) -> Element | None:
        if el is None:
            return None
        if np.random.uniform() > self.prob:
            return el
        ptdat = TreeTranslate.random_delta(
            el.global_pos,
            el.node_mask,
            self.max_delta,
            self.wrap_domain,
            ignore_aug_bounds=self.ignore_aug_bounds,
        )
        if ptdat is None:
            return None
        apply_lhs_position_transform(
            el,
            lambda pos, mask: TreeTranslate.translate_pos(pos, ptdat.delta_w, mask),
        )
        el.global_pos = TreeTranslate.translate_pos(el.global_pos, ptdat.delta_w, el.node_mask)

        return el

    @classmethod
    def random_delta(
        cls,
        pos: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor,
        max_delta: float,
        wrap_domain: tuple,
        max_attempts: int = 10,
        ignore_aug_bounds: bool = False,
    ) -> PosTransData | None:
        def make_candidate() -> PosTransData:
            delta_w = np.random.normal(0.0, 1.0, 2)
            norm = np.linalg.norm(delta_w)
            if norm == 0.0:
                norm = 1.0
            d = np.random.uniform(0.0, max_delta)
            delta_w = d * delta_w / norm
            transpos = TreeTranslate.translate_pos(pos, delta_w, mask)
            return PosTransData(transpos, delta_w)

        return retry_bounded_random_operation(
            operation_name="random_translate",
            make_candidate=make_candidate,
            get_candidate_pos=lambda candidate: candidate.transpos,
            mask=mask,
            wrap_domain=wrap_domain,
            max_attempts=max_attempts,
            ignore_aug_bounds=ignore_aug_bounds,
        )

    @classmethod
    def translate_pos(
        cls,
        pos: np.ndarray | torch.Tensor,
        delta_w: np.ndarray,
        node_mask: np.ndarray | torch.Tensor,
    ) -> np.ndarray | torch.Tensor:
        pos_cpy = np.copy(pos)
        delta_w = delta_w.reshape(1, 2)
        node_mask = to_array(node_mask).reshape(-1, 1)
        pos_cpy += node_mask * delta_w
        return pos_cpy


@dataclass
class PosZoomData:
    zoompos: np.ndarray
    valzoom: float


class TreeZoom:
    def __init__(
        self,
        min_zoom: float,
        max_zoom: float,
        fill: float,
        zoom_center: tuple = (0.5, 0.5),
        prob: float = 0.5,
        wrap_domain: tuple = (-0.5, 0.5),
        ignore_aug_bounds: bool = False,
    ) -> None:
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.fill = fill
        self.zoom_center = zoom_center
        self.prob = prob
        self.wrap_domain = wrap_domain
        self.ignore_aug_bounds = ignore_aug_bounds

    @classmethod
    def from_settings(cls, settings=None) -> TreeZoom:
        if settings is None:
            settings = TreeZoomSettings()
        elif not isinstance(settings, TreeZoomSettings):
            raise TypeError(
                "TreeZoom.from_settings expected a TreeZoomSettings or None, "
                f"got {type(settings).__name__}"
            )
        return cls(
            min_zoom=settings.min_zoom,
            max_zoom=settings.max_zoom,
            fill=settings.fill,
            zoom_center=settings.zoom_center,
            prob=settings.prob,
            wrap_domain=settings.wrap_domain,
            ignore_aug_bounds=settings.ignore_aug_bounds,
        )

    def __call__(self, el: Element) -> Element | None:
        if el is None:
            return None
        if np.random.uniform() >= self.prob:
            return el

        wix = self.zoom_center[0]
        wiy = self.zoom_center[1]

        pzdat = TreeZoom.random_zoom(
            el.global_pos,
            el.node_mask,
            wix,
            wiy,
            self.min_zoom,
            self.max_zoom,
            self.wrap_domain,
            ignore_aug_bounds=self.ignore_aug_bounds,
        )
        if pzdat is None:
            return None
        el.global_pos = pzdat.zoompos

        if el.radius is not None:
            el.radius *= pzdat.valzoom

        apply_lhs_position_transform(
            el,
            lambda pos, mask: TreeZoom.zoom_pos(pos, mask, wix, wiy, pzdat.valzoom),
        )

        if el.radius_lhs is not None:
            el.radius_lhs *= pzdat.valzoom

        return el

    @classmethod
    def random_zoom(
        cls,
        pos: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor,
        wix: float,
        wiy: float,
        min_zoom: float,
        max_zoom: float,
        wrap_domain: tuple,
        max_attempts: int = 10,
        ignore_aug_bounds: bool = False,
    ) -> PosZoomData | None:
        def make_candidate() -> PosZoomData:
            valzoom = np.random.uniform(min_zoom, max_zoom)
            poszoom = TreeZoom.zoom_pos(
                to_tensor(pos).clone(),
                mask,
                wix,
                wiy,
                valzoom,
            )
            return PosZoomData(poszoom, valzoom)

        return retry_bounded_random_operation(
            operation_name="random_zoom",
            make_candidate=make_candidate,
            get_candidate_pos=lambda candidate: candidate.zoompos,
            mask=mask,
            wrap_domain=wrap_domain,
            max_attempts=max_attempts,
            ignore_aug_bounds=ignore_aug_bounds,
        )

    @classmethod
    def zoom_pos(
        cls,
        pos: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor,
        wix: float,
        wiy: float,
        valzoom: float,
    ) -> np.ndarray | torch.Tensor:
        dx = pos[:, 0] - wix
        dy = pos[:, 1] - wiy
        pos[:, 0] = wix + dx * valzoom
        pos[:, 1] = wiy + dy * valzoom
        return pos * mask_column_like(mask, pos)


class _Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, value):
        for transform in self.transforms:
            value = transform(value)
        return value


def make_transforms(config):
    transform_list = []
    aconfig = config["aug_config"]
    probs = aconfig["probs"]
    if config["mode"] == "train" and config["allow_augmentation"]:
        if config["allow_zoom"]:
            transform_list.append(
                TreeZoom(
                    min_zoom=aconfig["min_zoom"],
                    max_zoom=aconfig["max_zoom"],
                    fill=config["pad_val"],
                    prob=probs["zoom"],
                    wrap_domain=config["wrap_domain"],
                    ignore_aug_bounds=config["ignore_aug_bounds"],
                )
            )
        if config["allow_rotate"]:
            wrap_domain = config["wrap_domain"]
            rot_center = (0.0, 0.0)
            if config["load_segmentation"]:
                wrap_domain = [0.0, 1.0]
                rot_center = (0.5, 0.5)
            transform_list.append(
                TreeRotate(
                    max_degrees=aconfig["max_degrees"],
                    fill=config["pad_val"],
                    prob=probs["rotate"],
                    wrap_domain=wrap_domain,
                    rot_center=rot_center,
                    ignore_aug_bounds=config["ignore_aug_bounds"],
                )
            )
        if config["allow_translate"]:
            transform_list.append(
                TreeTranslate(
                    max_delta=aconfig["max_delta"],
                    prob=probs["translate"],
                    wrap_domain=config["wrap_domain"],
                    ignore_aug_bounds=config["ignore_aug_bounds"],
                )
            )
    transform_list.append(ToTensor())
    return _Compose(transform_list)


__all__ = [
    "PosRotData",
    "PosTransData",
    "PosZoomData",
    "TreeRotate",
    "TreeTranslate",
    "TreeZoom",
    "_Compose",
    "active_node_indices",
    "active_positions_in_bounds",
    "apply_lhs_position_transform",
    "apply_segmentation_transform",
    "make_transforms",
    "mask_column_like",
    "retry_bounded_random_operation",
    "warn_random_operation_failed",
]
