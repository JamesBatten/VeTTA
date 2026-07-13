from __future__ import annotations

import time
from typing import Any

from pydantic import Field

from vetta.model import VesselTreeAutoencoder
from vetta.settings.base import VettaSettings


class TrainerOptimizerSettings(VettaSettings):
    optimizer_type: str = "AdamW"
    optimizer_lr: float = 1e-5
    override_lr: float | None = None
    backwards_per_iteration: int = 1
    weight_decay: float = 0.1
    max_grad_norm: float = 300.0
    do_optimize: bool = True
    weight_reconst: float = 0.999
    weight_kl: float = 0.001


class TrainerSchedulerSettings(VettaSettings):
    scheduler_type: str = "Custom"
    scheduler_warmup: int = 1000
    scheduler_gamma: float = 0.1
    scheduler_period: int = 10000
    freeze_duration: int | None = None


class TrainerCostFactorSettings(VettaSettings):
    pos: float = 0.9
    rad: float = 0.1
    topology: float = 1.0
    skip_vessel: float = 0.1
    enc_v: float = 0.005


class TrainerVesselTreeAutoencoderSettings(VettaSettings):
    dataset_configs_dict: dict[str, Any] | None = None
    dataset_name: str = "tree_dataset"
    data_mode: str = "pipeline"
    batch_sizes_dict: dict[str, Any] = Field(default_factory=lambda: {"tree_dataset": 10})
    checkpoint_period: int = 1000
    tracker_period: int = 10
    tracker_expiry_period: int = 100
    optimizer_config: TrainerOptimizerSettings = Field(default_factory=TrainerOptimizerSettings)
    scheduler_config: TrainerSchedulerSettings = Field(default_factory=TrainerSchedulerSettings)
    cost_factors: TrainerCostFactorSettings = Field(default_factory=TrainerCostFactorSettings)
    multi_gpu: bool = False
    num_workers: int = 1
    prefetch_factor: int = 10
    plot_trackers: bool = True
    start_time: float = Field(default_factory=time.time)
    save_batches: bool = False
    save_pred: bool = False
    load_batches_path: str | None = None
    start_batch_idx: int | None = None
    max_load_batches: int | None = None
    track_throughput: bool = True
    visualise_pred: bool = False
    visualise_period: int = 20
    throughput_delay: float = 5.0
    master_port: int = 12355
    matching_k: int = 3
    to_cuda: bool = True
    verbose: bool = False
    use_vae: bool = False


def default_trainer_config() -> dict[str, Any]:
    """Return the trainer config dict, including ``model_config``."""

    config = TrainerVesselTreeAutoencoderSettings().to_dict()
    config["model_config"] = VesselTreeAutoencoder.default_config()
    return config

