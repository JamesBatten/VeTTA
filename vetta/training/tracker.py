"""Scalar trackers and their plotting helpers for public training."""

from __future__ import annotations

import copy
import multiprocessing
import os
from dataclasses import dataclass
from typing import Any
from typing import Iterable
from typing import Mapping

import numpy as np

from vetta.utils import listdir_fullpath
from vetta.utils import load_json
from vetta.utils import mkdir_custom
from vetta.utils import require_optional_dependency
from vetta.utils import save_json
from vetta.utils import torch

try:
    import matplotlib
except ModuleNotFoundError:
    matplotlib = None


# Values that mark a missing point. ``"None"`` appears because ``Tracker.save``
# stringifies values through ``save_json``; a reloaded tracker therefore carries
# the literal string ``"None"`` where ``None`` was recorded.
MISSING_VALUES: frozenset[Any] = frozenset({None, "None"})


def as_optional_float(value: Any) -> float | None:
    """Return ``float(value)`` unless it is a missing marker (``None``/``"None"``)."""
    if value in MISSING_VALUES:
        return None
    if torch is not None and torch.is_tensor(value):
        value = value.detach().cpu()
    return float(value)


def tracker_points(
    tracker: "Tracker",
    *,
    log_y: bool = False,
    eps: float = 1e-10,
) -> tuple[list[float], list[float]]:
    """Filter and convert a tracker's points into plottable ``(xs, ys)`` lists.

    Drops any point whose x or y is a missing marker, optionally applies
    ``log10`` (floored at ``eps``) to y. Matches the legacy ``plot_single``
    point-selection behaviour exactly.
    """
    xs: list[float] = []
    ys: list[float] = []
    for raw_x, raw_y in zip(tracker.x_values, tracker.y_values):
        x = as_optional_float(raw_x)
        y = as_optional_float(raw_y)
        if x is None or y is None:
            continue
        if log_y:
            y = float(np.log10(max(y, eps)))
        xs.append(x)
        ys.append(y)
    return xs, ys


@dataclass(frozen=True)
class PlotOptions:
    """Rendering options for tracker plots."""

    figsize: tuple[int, int] = (10, 10)
    dpi: int = 100
    log_y: bool = False
    eps: float = 1e-10


@dataclass(frozen=True)
class GradNorms:
    """Per-iteration gradient-norm summary."""

    min: float
    max: float
    mean: float


@dataclass(frozen=True)
class MetricSnapshot:
    """One iteration's metrics, flattened for tracker recording."""

    iteration: int
    loss: Mapping[str, Any]
    pred: Mapping[str, Any]
    learning_rate: Any
    grad_norms: GradNorms

    def values(self) -> dict[str, Any]:
        # Precedence matches the legacy ``update_trackers`` if/elif chain:
        # loss-dict entries win over pred-dict entries, which win over the
        # special learning-rate / grad-norm keys.
        return {
            "learning_rate": self.learning_rate,
            "min_grad_norm": self.grad_norms.min,
            "max_grad_norm": self.grad_norms.max,
            "mean_grad_norm": self.grad_norms.mean,
            **self.pred,
            **self.loss,
        }


class Tracker:
    def __init__(self) -> None:
        self.x_values: list[Any] = []
        self.y_values: list[Any] = []

    def add(self, x: Any, y: Any) -> None:
        self.x_values.append(x)
        self.y_values.append(y)

    def save(self, path: str) -> None:
        save_json(path, {"x_values": self.x_values, "y_values": self.y_values})

    @classmethod
    def load(cls, path: str) -> "Tracker":
        json_data = load_json(path)
        tracker = Tracker()
        tracker.x_values = json_data["x_values"]
        tracker.y_values = json_data["y_values"]
        return tracker

    @classmethod
    def log_vals(cls, vals: list[Any]) -> np.ndarray:
        ret = []
        for v in vals:
            v = max(np.power(10.0, -10), float(v))
            ret.append(np.log10(v))
        return np.array(ret)

    def get_xvals_np(self, use_log: bool = False) -> np.ndarray:
        if use_log:
            return Tracker.log_vals(self.x_values)
        return np.array([float(x) for x in self.x_values])

    def get_yvals_np(self, use_log: bool = False) -> np.ndarray:
        if use_log:
            return Tracker.log_vals(self.y_values)
        return np.array([float(y) for y in self.y_values])

    @classmethod
    def copy(cls, rhs: "Tracker") -> "Tracker":
        tracker = Tracker()
        tracker.x_values = copy.copy(rhs.x_values)
        tracker.y_values = copy.copy(rhs.y_values)
        return tracker

    def plot(
        self,
        path: str,
        figsize: tuple[int, int] = (10, 10),
        dpi: int = 100,
        log_y: bool = False,
        eps: float = 1e-10,
    ) -> None:
        require_optional_dependency(matplotlib, "matplotlib", "tracker plotting")
        options = PlotOptions(figsize=figsize, dpi=dpi, log_y=log_y, eps=eps)
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=options.figsize, dpi=options.dpi)
        ax = fig.gca()
        Tracker.plot_single(self, ax, log_y=options.log_y, eps=options.eps)
        plt.savefig(path)
        plt.close("all")

    @classmethod
    def plot_single(
        cls,
        tracker: "Tracker",
        ax: Any,
        log_y: bool = False,
        eps: float = 1e-10,
        label: str | None = None,
    ) -> Any:
        x_values, y_values = tracker_points(tracker, log_y=log_y, eps=eps)
        (line,) = ax.plot(x_values, y_values, label=label)
        return line

    def get_last_mean_yvals(self, period: int) -> float | None:
        if len(self.y_values) < period:
            return None
        return float(np.mean(self.y_values[-period:]))


def get_trackerkeys(plot_types: dict[str, list[str]]) -> list[str]:
    return list(plot_types.keys())


def make_trackers(trackerkeys: list[str]) -> dict[str, Tracker]:
    return {trackerkey: Tracker() for trackerkey in trackerkeys}


def plot_trackers(
    plot_types: dict[str, list[str]],
    plot_dir: str,
    trackers: dict[str, Tracker],
) -> None:
    require_optional_dependency(matplotlib, "matplotlib", "tracker plotting")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for trackerkey, types in plot_types.items():
        if trackerkey in trackers.keys():
            if "normal" in types:
                plot_path = os.path.join(plot_dir, trackerkey + ".png")
                trackers[trackerkey].plot(plot_path, log_y=False)
            if "log" in types:
                plot_path = os.path.join(plot_dir, "log_" + trackerkey + ".png")
                trackers[trackerkey].plot(plot_path, log_y=True)
    plt.close("all")


def update_trackers_from_snapshot(
    trackers: dict[str, Tracker],
    snapshot: MetricSnapshot,
    metric_names: Iterable[str],
) -> None:
    """Record one snapshot's metrics into ``trackers`` for ``metric_names``.

    A metric is recorded only when the snapshot carries it (mirrors the legacy
    behaviour where unmatched tracker keys were silently skipped). ``None`` /
    ``"None"`` values are recorded as ``None`` so plotting drops them later.
    """
    values = snapshot.values()
    for name in metric_names:
        if name in values:
            trackers[name].add(snapshot.iteration, as_optional_float(values[name]))


def update_trackers(
    iteration: int,
    trackers: dict[str, Tracker],
    loss_dict: dict[str, Any],
    pred_dict: dict[str, Any],
    last_lr: float,
    min_grad_norm: float,
    max_grad_norm: float,
    mean_grad_norm: float,
    plot_types: dict[str, list[str]],
) -> None:
    """Backward-compatible wrapper around ``update_trackers_from_snapshot``.

    Kept because callers (the worker loop in ``base.py`` and the tracker tests)
    still pass the flat positional argument list.
    """
    snapshot = MetricSnapshot(
        iteration=iteration,
        loss=loss_dict,
        pred=pred_dict,
        learning_rate=last_lr,
        grad_norms=GradNorms(
            min=min_grad_norm, max=max_grad_norm, mean=mean_grad_norm
        ),
    )
    update_trackers_from_snapshot(trackers, snapshot, get_trackerkeys(plot_types))


def render_trackers_subprocess(
    plot_types: dict[str, list[str]],
    plot_dir: str,
    trackers: dict[str, Tracker],
    timeout: float = 120.0,
) -> None:
    """Render tracker plots in a fresh ``spawn`` subprocess.

    matplotlib deadlocks if it is first used inside a process created with
    ``os.fork()``: the child inherits lock state from the parent that imported
    matplotlib, but none of the threads that would release those locks. The
    integration training benchmark forks a worker child (see
    ``vetta.integration.training_pipeline.supervise_child``), so calling
    ``plot_trackers`` directly there hangs forever on the first ``savefig``.

    Running the rendering in a ``spawn`` context starts a clean interpreter with
    no inherited locks, which is safe regardless of how the calling process was
    created. ``plot_trackers`` and its arguments are all picklable, so they can
    cross the process boundary unchanged.
    """
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=plot_trackers,
        args=(plot_types, plot_dir, trackers),
    )
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join(5.0)
        raise TimeoutError(
            f"tracker plot rendering exceeded {timeout:.1f}s and was terminated"
        )
    if process.exitcode != 0:
        raise RuntimeError(
            f"tracker plot rendering subprocess failed with exit code {process.exitcode}"
        )


def save_trackers(
    directory: str,
    trackers: dict[str, Tracker],
    plot_types: dict[str, list[str]],
    do_plot: bool = True,
    plot_timeout: float = 120.0,
) -> None:
    data_dir = mkdir_custom(os.path.join(directory, "data"))
    for trackerkey in get_trackerkeys(plot_types):
        trackers[trackerkey].save(os.path.join(data_dir, trackerkey + ".json"))
    if do_plot:
        trackers_plot_dir = mkdir_custom(os.path.join(directory, "plot"))
        render_trackers_subprocess(
            plot_types, trackers_plot_dir, trackers, timeout=plot_timeout
        )


def cleanup_trackers(directory: str, iteration_counter: int, expiry_period: int) -> None:
    if expiry_period <= 0:
        return
    iter_dirs = listdir_fullpath(directory, True)
    if iter_dirs is not None:
        for iter_dir in iter_dirs:
            counter_i = int(os.path.split(iter_dir)[-1].split("_")[-1])
            if counter_i < iteration_counter - expiry_period:
                import shutil

                shutil.rmtree(iter_dir)
