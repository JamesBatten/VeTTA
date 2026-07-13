from __future__ import annotations

import dataclasses
from typing import Any

import torch
from torch.multiprocessing import spawn

from vetta.settings.training import default_trainer_config
from vetta.training.base import Trainer
from vetta.training.base import default_save_batch_fn
from vetta.training.base import default_save_pred_fn
from vetta.training.base import default_update_weights_fn
from vetta.training.base import worker
from vetta.training.environment import EnvironmentVariables
from vetta.training.runtime import CheckpointComponents
from vetta.training.runtime import TrainingComponents
from vetta.training.runtime import WorkerLaunch
from vetta.training.runtime import WorkerSpec
from vetta.training.runtime import worker_spawn_entry
from vetta.training.autoencoder import check_model_state
from vetta.training.autoencoder import compute_loss
from vetta.training.autoencoder import compute_pred
from vetta.training.autoencoder import dataset_name_to_cls
from vetta.training.autoencoder import get_extras_dict
from vetta.training.autoencoder import get_paths
from vetta.training.autoencoder import make_model
from vetta.training.autoencoder import plot_types
from vetta.training.autoencoder import save_model


class TrainerVesselTreeAutoEncoder(Trainer):
    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return default_trainer_config()

    def _worker_spec(self, trainer_dir: str, n_iterations: int) -> WorkerSpec:
        dataset_cls_dict = {
            self.config["dataset_name"]: dataset_name_to_cls(self.config["dataset_name"])
        }
        return WorkerSpec(
            launch=WorkerLaunch(
                rank=0,
                world_size=1,
                master_port=self.config["master_port"],
                output_dir=trainer_dir,
                envvars_path=EnvironmentVariables.get_path(),
                n_iters=n_iterations,
            ),
            run=self.run_options,
            dataset_cls_dict=dataset_cls_dict,
            config=self.config,
            components=TrainingComponents(
                make_model=make_model,
                compute_pred=compute_pred,
                compute_loss=compute_loss,
                get_paths=get_paths,
                plot_types=plot_types(self.config),
                extras_dict=get_extras_dict,
                check_model_state=check_model_state,
                update_weights=default_update_weights_fn,
                save_batch=default_save_batch_fn,
                save_pred=default_save_pred_fn,
            ),
            checkpoint=CheckpointComponents(save_model=save_model),
            verbose=self.config["verbose"],
        )

    def train(self, trainer_dir: str, n_iterations: int) -> None:
        spec = self._worker_spec(trainer_dir, n_iterations)

        if self.config["data_mode"] == "pipeline":
            world_size = 1
            if self.config["multi_gpu"] and torch.cuda.device_count() > 1:
                world_size = torch.cuda.device_count()
            spec = dataclasses.replace(
                spec, launch=dataclasses.replace(spec.launch, world_size=world_size)
            )
            spawn(
                worker_spawn_entry,
                args=(spec, worker),
                nprocs=world_size,
                join=True,
            )
            return

        if self.config["data_mode"] == "load_batches":
            worker(spec)
            return

        raise ValueError(f"unknown data_mode {self.config['data_mode']!r}")
