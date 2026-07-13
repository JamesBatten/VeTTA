from __future__ import annotations

import dataclasses
import os
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Callable


@dataclass(frozen=True)
class TrainerRunOptions:
    """Checkpoint-resume / override behaviour for a training run."""

    load_checkpoint: str | None = None
    reset_optimizer: bool = False
    reset_scheduler: bool = False
    reset_trackers: bool = False
    override_config: bool = False
    override_lr: float | None = None
    instance_id: str = field(default_factory=lambda: str(int(time.time() * 1000)))


@dataclass(frozen=True)
class WorkerLaunch:
    """Per-process launch parameters (distributed rank, ports, output dir)."""

    rank: int
    world_size: int
    master_port: int
    output_dir: str
    envvars_path: str
    n_iters: int
    start_iter: int = 0


@dataclass(frozen=True)
class TrainingComponents:
    """The pluggable training callables and the plot-type map."""

    make_model: Callable
    compute_pred: Callable[..., dict]
    compute_loss: Callable
    get_paths: Callable
    plot_types: dict[str, list[str]] | None = None
    cnoise: Any | None = None
    extras_dict: Callable | None = None
    check_model_state: Callable | None = None
    update_weights: Callable | None = None
    save_batch: Callable | None = None
    save_pred: Callable | None = None


@dataclass(frozen=True)
class CheckpointComponents:
    """Optional model/cnoise save hooks for checkpointing."""

    save_model: Callable | None = None
    save_cnoise: Callable | None = None


@dataclass(frozen=True)
class WorkerSpec:
    """Everything one ``worker`` invocation needs, bundled into one object."""

    launch: WorkerLaunch
    run: TrainerRunOptions
    dataset_cls_dict: dict[str, type]
    config: dict[str, Any]
    components: TrainingComponents
    checkpoint: CheckpointComponents = CheckpointComponents()
    verbose: bool = False
    min_lr: float | None = None


def worker_spawn_entry(rank: int, spec: WorkerSpec, worker_fn: Callable) -> None:
    """``torch.multiprocessing.spawn`` adapter: inject ``rank`` into the spec.

    ``spawn`` calls ``fn(i, *args)`` with the process index first, so this
    replaces ``spec.launch.rank`` with the spawned rank before running the
    worker. ``worker_fn`` is passed explicitly to avoid a circular import with
    ``base``.
    """
    launch = dataclasses.replace(spec.launch, rank=rank)
    worker_fn(dataclasses.replace(spec, launch=launch))


@dataclass(frozen=True)
class CheckpointPaths:
    """Typed checkpoint path set for one iteration directory."""

    root: str
    model_dir: str
    model: str
    optimizer: str
    scheduler: str
    trackers: str
    config: str
    metadata: str

    @classmethod
    def from_dir(cls, checkpoint_dir: str, model_filename: str = "model") -> "CheckpointPaths":
        model_dir = os.path.join(checkpoint_dir, "model")
        return cls(
            root=checkpoint_dir,
            model_dir=model_dir,
            model=os.path.join(model_dir, model_filename),
            optimizer=os.path.join(checkpoint_dir, "optimizer"),
            scheduler=os.path.join(checkpoint_dir, "scheduler"),
            trackers=os.path.join(checkpoint_dir, "trackers"),
            config=os.path.join(checkpoint_dir, "config.json"),
            metadata=os.path.join(checkpoint_dir, "metadata.json"),
        )

    def as_default_dict(self) -> dict[str, str]:
        """Return the legacy ``get_paths_default`` dict.

        In the default scheme ``"model"`` is a flat file at ``<dir>/model`` (here
        ``model_dir``); there is no separate model subdirectory.
        """
        return {
            "model": self.model_dir,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "trackers": self.trackers,
            "config": self.config,
            "metadata": self.metadata,
        }
