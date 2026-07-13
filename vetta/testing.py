from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from statistics import mean
from typing import Any

import numpy as np

from vetta.inference import tree_to_segmentation


def segmentation_mask(segmentation: np.ndarray) -> np.ndarray:
    """Threshold a segmentation array into a boolean foreground mask.

    A 3-channel image uses its first channel; values above 127 are foreground
    (matching how the inference segmentations are written, 0/255 per channel).
    """
    segmentation = np.asarray(segmentation)
    if segmentation.ndim == 3:
        mask = segmentation[..., 0] > 127
    else:
        mask = segmentation > 127
    return mask.astype(bool)


def dice_score(input_mask: np.ndarray, output_mask: np.ndarray) -> float:
    """2D Dice overlap between two boolean-able masks (1.0 when both empty)."""
    input_bool = np.asarray(input_mask).astype(bool)
    output_bool = np.asarray(output_mask).astype(bool)
    denom = int(input_bool.sum()) + int(output_bool.sum())
    if denom == 0:
        return 1.0
    intersection = int(np.logical_and(input_bool, output_bool).sum())
    return float(2.0 * intersection / denom)


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def dice_statistics(scores: Sequence[float]) -> dict[str, float | None]:
    """Mean + 5th/50th/95th percentile of a sequence of Dice scores."""
    values = [float(score) for score in scores if score is not None]
    return {
        "mean": None if not values else float(mean(values)),
        "p5": _percentile(values, 5),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
    }


def evaluate(
    pairs: Sequence[tuple[np.ndarray, np.ndarray]] | Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    """Score many ``(ground_truth, prediction)`` segmentation pairs.

    ``pairs`` is either a sequence of ``(gt, pred)`` arrays or a mapping of
    ``name -> (gt, pred)``. Returns ``{"scores": {...}, "dice": {stats}}`` where
    each pair is reduced to a boolean mask via :func:`segmentation_mask` before
    scoring.
    """
    items = pairs.items() if isinstance(pairs, Mapping) else enumerate(pairs)
    scores: dict[str, float] = {}
    for key, (target, prediction) in items:
        scores[str(key)] = dice_score(
            segmentation_mask(target),
            segmentation_mask(prediction),
        )
    return {"scores": scores, "dice": dice_statistics(list(scores.values()))}


def _tree_segmentation(tree: Any, *, seg_size: int, n_interpolate: int, radius: np.ndarray | None):
    tree_radius = tree.radius if getattr(tree, "radius", None) is not None else radius
    if tree_radius is None:
        raise ValueError("tree_dice requires a per-node radius (tree.radius or the radius argument)")
    return tree_to_segmentation(
        seg_size=seg_size,
        n_interpolate=n_interpolate,
        edges=np.asarray(tree.edges),
        edges_mask=np.asarray(tree.edges_mask),
        pos=np.asarray(tree.pos),
        radius=np.asarray(tree_radius),
    )


def tree_dice(
    tree_a: Any,
    tree_b: Any,
    *,
    seg_size: int = 250,
    n_interpolate: int = 100,
    radius: np.ndarray | None = None,
) -> float:
    """Rasterise two public ``Tree``s and return their 2D Dice overlap.

    Each tree's node positions must be in fractional image coordinates (``[0, 1]``)
    and carry a per-node radius (``tree.radius`` or the shared ``radius``
    argument), matching how SSA inference segmentations are rendered.
    """
    seg_a = _tree_segmentation(tree_a, seg_size=seg_size, n_interpolate=n_interpolate, radius=radius)
    seg_b = _tree_segmentation(tree_b, seg_size=seg_size, n_interpolate=n_interpolate, radius=radius)
    return dice_score(segmentation_mask(seg_a), segmentation_mask(seg_b))


__all__ = [
    "dice_score",
    "dice_statistics",
    "evaluate",
    "segmentation_mask",
    "tree_dice",
]
