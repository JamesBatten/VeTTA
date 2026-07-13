from __future__ import annotations

import abc
import copy
import datetime
import inspect
import os
from typing import Any
from typing import Callable

import torch
import torch.distributed
import torch.nn as nn
import torch.utils.data
from torch.nn.utils import clip_grad_norm_

from vetta.training.custom_scheduler import make_custom_scheduler
from vetta.training.environment import EnvironmentVariables
from vetta.training.runtime import CheckpointPaths
from vetta.training.runtime import TrainerRunOptions
from vetta.training.runtime import WorkerSpec
from vetta.training.tracker import Tracker
from vetta.training.tracker import cleanup_trackers
from vetta.training.tracker import get_trackerkeys
from vetta.training.tracker import make_trackers
from vetta.training.tracker import save_trackers
from vetta.training.tracker import update_trackers
from vetta.utils import listdir_fullpath
from vetta.utils import load_json
from vetta.utils import make_json_serializable
from vetta.utils import mkdir_custom
from vetta.utils import printf
from vetta.utils import remove_custom
from vetta.utils import save_json


SHARING_STRATEGY = "file_system"
METADATA_ROOT = "/tmp/exchanger_dev_metadata"


def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


class Trainer(metaclass=abc.ABCMeta):
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        run_options: TrainerRunOptions | None = None,
        **run_option_overrides: Any,
    ):
        self.config = self.default_config() if config is None else config
        if run_options is None:
            run_options = TrainerRunOptions(**run_option_overrides)
        elif run_option_overrides:
            raise TypeError(
                "pass either run_options or individual run-option keyword arguments, not both"
            )
        self.run_options = run_options

    # The individual run-option attributes are exposed as read-only properties so
    # existing subclasses and callers keep working while new code uses
    # ``self.run_options`` directly.
    @property
    def load_checkpoint(self) -> str | None:
        return self.run_options.load_checkpoint

    @property
    def reset_optimizer(self) -> bool:
        return self.run_options.reset_optimizer

    @property
    def reset_scheduler(self) -> bool:
        return self.run_options.reset_scheduler

    @property
    def reset_trackers(self) -> bool:
        return self.run_options.reset_trackers

    @property
    def override_config(self) -> bool:
        return self.run_options.override_config

    @property
    def override_lr(self) -> float | None:
        return self.run_options.override_lr

    @property
    def instance_id(self) -> str:
        return self.run_options.instance_id

    @classmethod
    @abc.abstractmethod
    def default_config(cls) -> dict[str, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, trainer_dir: str, n_iterations: int) -> None:
        raise NotImplementedError


def make_optimizer(
    model: nn.parallel.DistributedDataParallel | nn.Module,
    config: dict,
    override_lr: float | None = None,
) -> torch.optim.Optimizer:
    if isinstance(model, nn.parallel.DistributedDataParallel):
        params = model.module.parameters()
    elif isinstance(model, nn.Module):
        params = model.parameters()
    else:
        raise Exception("unknown model type " + str(type(model)))
    learning_rate = config["optimizer_lr"]
    if override_lr is not None:
        learning_rate = override_lr
    optimizer_type = config["optimizer_type"]
    optimizer_classes = {
        "Adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "SGD": torch.optim.SGD,
        "Adagrad": torch.optim.Adagrad,
        "Adadelta": torch.optim.Adadelta,
    }
    if optimizer_type not in optimizer_classes:
        raise Exception("unknown optimizer type " + str(optimizer_type))
    return optimizer_classes[optimizer_type](
        [{"params": params}],
        lr=learning_rate,
        weight_decay=config["weight_decay"],
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict,
    scheduler_path: str | None = None,
    fresh_scheduler: bool = False,
    last_epoch: int | None = None,
    override_lr: float | None = None,
    min_mult: float | None = None,
) -> torch.optim.lr_scheduler._LRScheduler:
    if override_lr is not None:
        return torch.optim.lr_scheduler.ConstantLR(optimizer, override_lr)

    scheduler_type = config["scheduler_type"]
    if scheduler_type == "ExponentialLR":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer, gamma=config["scheduler_gamma"]
        )
    elif scheduler_type == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=config["scheduler_period"],
            gamma=config["scheduler_gamma"],
            last_epoch=-1,
        )
    elif scheduler_type == "Custom":
        scheduler = make_custom_scheduler(
            optimizer=optimizer,
            warmup_steps=config["scheduler_warmup"],
            gamma_factor=config["scheduler_gamma"],
            gamma_period=config["scheduler_period"],
            min_mult=min_mult,
        )
    elif scheduler_type == "pcomposite":
        # The pcomposite schedule (and its PComposite dependency) is unused by
        # the vessel-tree autoencoder trainer and is not ported.
        raise NotImplementedError(
            "the 'pcomposite' scheduler is not ported to vetta.training"
        )
    else:
        raise Exception("unhandled scheduler type " + str(scheduler_type))

    if scheduler_path is not None and not fresh_scheduler:
        scheduler.load_state_dict(torch.load(scheduler_path))
        if last_epoch is not None:
            for _ in range(0, last_epoch):
                scheduler.step()
    return scheduler


def get_paths_default(checkpoint_dir: str) -> dict[str, str]:
    return CheckpointPaths.from_dir(checkpoint_dir).as_default_dict()


def save_checkpoint(
    config: dict,
    checkpoints_dir: str,
    iteration_counter: int,
    model: nn.parallel.DistributedDataParallel,
    cnoise: Any | None,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    trackers: dict[str, Tracker] | None,
    plot_types: dict[str, list[str]] | None,
    save_model_fn: Callable | None,
    save_cnoise_fn: Callable | None,
    get_paths_fn: Callable,
) -> None:
    iter_dir = mkdir_custom(
        os.path.join(checkpoints_dir, "iter_" + str(iteration_counter).zfill(7))
    )
    paths = get_paths_fn(iter_dir)
    save_json(paths["metadata"], {"iteration_counter": iteration_counter})
    md = model
    if config["data_mode"] == "pipeline":
        md = model.module
    if save_model_fn is None:
        torch.save(md.state_dict(), paths["model"])
    else:
        save_model_fn(md, paths)
    if save_cnoise_fn is not None:
        save_cnoise_fn(cnoise, paths)
    torch.save(optimizer.state_dict(), paths["optimizer"])
    torch.save(scheduler.state_dict(), paths["scheduler"])
    save_json(paths["config"], make_json_serializable(config))
    if config["plot_trackers"]:
        trackers_dir = mkdir_custom(os.path.join(iter_dir, "trackers"))
        if trackers is not None and plot_types is not None:
            save_trackers(trackers_dir, trackers, plot_types, True)


def default_check_model_state(
    model: nn.parallel.DistributedDataParallel,
    trainer_config: dict,
    iteration_counter: int,
) -> nn.parallel.DistributedDataParallel:
    """Default model-state hook; intentionally returns the model unchanged."""
    return model


def default_update_weights_fn(
    optimizer: torch.optim.Optimizer,
    model: nn.parallel.DistributedDataParallel,
    trainer_config: dict,
    iteration_counter: int,
) -> None:
    """Default optimizer hook used by the ported trainer runtime."""
    if trainer_config["optimizer_config"]["do_optimize"]:
        optimizer.step()


def default_save_batch_fn(
    train_batch_dict: dict,
    directory: str,
    iteration_counter: int,
) -> None:
    """Optional batch-export hook; intentionally disabled by default."""
    return None


def default_save_pred_fn(
    train_batch_dict: dict,
    pred_dict: dict,
    loss_dict: dict,
    directory: str,
    iteration_counter: int,
    config: dict,
) -> None:
    """Optional prediction-export hook; intentionally disabled by default."""
    return None


def write_metadata(iteration_counter: int, instance_id: str) -> None:
    metadata_dir = mkdir_custom(
        os.path.join(METADATA_ROOT, instance_id), "return"
    )
    current_files = listdir_fullpath(metadata_dir, True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    newfile = os.path.join(metadata_dir, timestamp + ".json")
    save_json(newfile, {"iteration_counter": iteration_counter})
    if current_files is not None and len(current_files) > 10:
        remove_custom(current_files[0])


def erase_metadata(instance_id: str) -> None:
    metadata_dir = os.path.join(METADATA_ROOT, instance_id)
    if os.path.exists(metadata_dir):
        import shutil

        shutil.rmtree(metadata_dir)


class BatchIterator:
    def __init__(
        self,
        dataset_cls,
        bdir,
        start_batch_idx,
        max_load_batches,
        batch_size,
        to_cuda=True,
    ):
        self.dataset_cls = dataset_cls
        batches_paths = listdir_fullpath(bdir)
        s0 = start_batch_idx if start_batch_idx is not None else 0
        if max_load_batches is None:
            s1 = len(batches_paths) - 1
        else:
            s1 = min(len(batches_paths), s0 + max_load_batches)
        self.batches_paths = batches_paths[s0:s1]
        self.batch_size = batch_size
        self.to_cuda = to_cuda

    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        path = self.batches_paths[self.counter]
        self.counter += 1
        if self.counter >= len(self.batches_paths):
            self.counter = 0
        batch_dict = self.dataset_cls.load_batch(
            path, return_tensor=True, return_cuda=self.to_cuda
        )
        t_keys = list(batch_dict.keys())
        b = batch_dict[t_keys[0]].shape[0]
        if b < self.batch_size:
            raise Exception(
                "loaded batch size " + str(b) + " is smaller than specified "
                "batch size " + str(self.batch_size)
            )
        for ky in batch_dict.keys():
            batch_dict[ky] = batch_dict[ky][0 : self.batch_size]
        return batch_dict


def worker(spec: WorkerSpec) -> None:
    # Unpack the spec into the local names the loop below uses. This keeps the
    # (large, ported) loop body unchanged while the call surface is one object.
    launch = spec.launch
    run = spec.run
    components = spec.components
    checkpoint = spec.checkpoint

    rank = launch.rank
    world_size = launch.world_size
    master_port = launch.master_port
    instance_id = run.instance_id
    output_dir = launch.output_dir
    envvars_path = launch.envvars_path
    n_iters = launch.n_iters
    dataset_cls_dict = spec.dataset_cls_dict
    config = spec.config
    make_model_fn = components.make_model
    compute_pred_fn = components.compute_pred
    compute_loss_fn = components.compute_loss
    get_paths_fn = components.get_paths
    start_iter = launch.start_iter
    load_checkpoint = run.load_checkpoint
    reset_optimizer = run.reset_optimizer
    reset_scheduler = run.reset_scheduler
    reset_trackers = run.reset_trackers
    override_config = run.override_config
    override_lr = run.override_lr
    plot_types = components.plot_types
    cnoise = components.cnoise
    save_model_fn = checkpoint.save_model
    save_cnoise_fn = checkpoint.save_cnoise
    extras_dict_fn = components.extras_dict
    check_model_state_fn = components.check_model_state or default_check_model_state
    update_weights_fn = components.update_weights or default_update_weights_fn
    save_batch_fn = components.save_batch or default_save_batch_fn
    save_pred_fn = components.save_pred or default_save_pred_fn
    verbose = spec.verbose
    min_lr = spec.min_lr

    EnvironmentVariables.load(envvars_path)
    gpus = EnvironmentVariables()["gpus"]
    if rank >= len(gpus):
        raise Exception("rank is greater than the number of gpus")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus[rank])

    if rank == 0:
        erase_metadata(instance_id)

    if "to_cuda" not in config.keys():
        raise Exception("config does not contain the key: to_cuda")
    device_str = "cuda" if config["to_cuda"] else "cpu"

    paths = None
    if load_checkpoint is not None:
        if get_paths_fn is not None:
            paths = get_paths_fn(load_checkpoint)
        else:
            paths = get_paths_default(load_checkpoint)
        if override_config:
            config = load_json(paths["config"])

    assert config["dataset_configs_dict"] is not None

    train_iters_dict = {}
    if config["data_mode"] == "pipeline":
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(master_port)
        torch.distributed.init_process_group("gloo", rank=rank, world_size=world_size)
        torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)

        for dataset_key in dataset_cls_dict.keys():
            dataset_config = copy.copy(config["dataset_configs_dict"][dataset_key])
            dataset_config["rank"] = rank
            dataset = dataset_cls_dict[dataset_key](config=dataset_config)
            dataset.candidate_noise = cnoise
            dataloader = torch.utils.data.DataLoader(
                dataset=dataset,
                batch_size=config["batch_sizes_dict"][dataset_key],
                shuffle=False,
                num_workers=config["num_workers"],
                pin_memory=False,
                prefetch_factor=config["prefetch_factor"],
                worker_init_fn=set_worker_sharing_strategy,
            )
            train_iters_dict[dataset_key] = iter(dataloader)
    elif config["data_mode"] == "load_batches":
        for dataset_key in dataset_cls_dict.keys():
            dataloader = BatchIterator(
                dataset_cls_dict[dataset_key],
                config["load_batches_path"],
                start_batch_idx=config["start_batch_idx"],
                max_load_batches=config["max_load_batches"],
                batch_size=config["batch_sizes_dict"][dataset_key],
                to_cuda=config["to_cuda"],
            )
            train_iters_dict[dataset_key] = iter(dataloader)

    extras_dict = None
    if extras_dict_fn is not None:
        extras_dict = extras_dict_fn(config)

    sig = inspect.signature(make_model_fn)
    if "to_cuda" in sig.parameters:
        model = make_model_fn(
            config["model_config"], load_checkpoint, paths, device_str, extras_dict,
            to_cuda=config["to_cuda"],
        )
    else:
        model = make_model_fn(
            config["model_config"], load_checkpoint, paths, device_str, extras_dict,
        )
    model = check_model_state_fn(model, config, start_iter + 1)
    if config["data_mode"] == "pipeline":
        model = nn.parallel.DistributedDataParallel(model, device_ids=[0])

    optimizer = make_optimizer(
        model, config["optimizer_config"], override_lr=override_lr
    )

    min_mult = None
    if min_lr is not None:
        start_lr = config["optimizer_config"]["optimizer_lr"]
        min_mult = min_lr / start_lr
    scheduler = make_scheduler(
        optimizer, config["scheduler_config"], override_lr=override_lr,
        min_mult=min_mult,
    )

    trackers = None
    if plot_types is not None:
        trackers = make_trackers(get_trackerkeys(plot_types))

    start_iter = 0
    if load_checkpoint is not None and paths is not None:
        metadata_json = load_json(paths["metadata"])
        start_iter = int(metadata_json["iteration_counter"])
        if not reset_optimizer:
            optimizer.load_state_dict(torch.load(paths["optimizer"]))
        if not reset_scheduler:
            scheduler.load_state_dict(torch.load(paths["scheduler"]))
        if not reset_trackers:
            trackers_data_dir = os.path.join(load_checkpoint, "trackers", "data")
            if plot_types is not None and trackers is not None:
                for trackerkey in get_trackerkeys(plot_types):
                    trackers[trackerkey] = Tracker.load(
                        os.path.join(trackers_data_dir, trackerkey + ".json")
                    )
    if override_lr is not None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = override_lr

    iteration_counter = start_iter + 1
    backwards_per_iteration = config["optimizer_config"].get(
        "backwards_per_iteration", 1
    )

    finished = False
    n_batches = 0
    while not finished:
        if rank == 0:
            if verbose:
                printf()
                printf("step " + str(iteration_counter))
            write_metadata(iteration_counter, instance_id)

        optimizer.zero_grad()

        culm_loss_dict = None
        pred_dict = None
        for _ in range(0, backwards_per_iteration):
            train_batch_dict = {}
            for key_str in train_iters_dict.keys():
                train_batch_dict[key_str] = next(train_iters_dict[key_str])

            if config["save_batches"]:
                save_batch_fn(
                    train_batch_dict,
                    os.path.join(output_dir, "batches"),
                    iteration_counter,
                )

            pred_dict = compute_pred_fn(
                train_batch_dict, model, config, iteration_counter
            )
            loss_dict = compute_loss_fn(
                pred_dict, train_batch_dict, iteration_counter, config
            )

            if config["save_pred"]:
                save_pred_fn(
                    train_batch_dict, pred_dict, loss_dict,
                    os.path.join(output_dir, "prediction"),
                    iteration_counter, config,
                )

            for key_str in loss_dict.keys():
                if loss_dict[key_str] is not None:
                    loss_dict[key_str] *= 1.0 / float(backwards_per_iteration)

            if culm_loss_dict is None:
                culm_loss_dict = loss_dict
            else:
                for key_str in culm_loss_dict.keys():
                    if loss_dict[key_str] is not None:
                        culm_loss_dict[key_str] += loss_dict[key_str]
                    else:
                        culm_loss_dict[key_str] = None

            loss_dict["loss"].backward()
            n_batches += 1

        loss_dict = culm_loss_dict
        assert loss_dict is not None
        if verbose:
            printf(
                "train_loss replica " + str(rank) + " : "
                + str(float(loss_dict["loss"]))
            )

        max_grad_norm_cfg = config["optimizer_config"].get("max_grad_norm")
        if max_grad_norm_cfg is not None:
            clip_grad_norm_(model.parameters(), max_grad_norm_cfg)

        grad_norms = [
            p.grad.data.norm() for p in model.parameters() if p.grad is not None
        ]
        min_grad_norm = min(grad_norms)
        max_grad_norm = max(grad_norms)
        mean_grad_norm = float(sum(grad_norms)) / float(len(grad_norms))

        skip_weight_update = False
        if config.get("skip_bad_batches"):
            sb_thres = config["skip_batch_threshold"]
            sb_delay = config["skip_batch_delay"]
            sb_period = config["skip_batch_period"]
            if loss_dict["loss"] is not None:
                loss_val = float(loss_dict["loss"])
                if n_batches >= sb_delay and trackers is not None:
                    if len(trackers["loss"].y_values) > sb_period:
                        last_mean = trackers["loss"].get_last_mean_yvals(sb_period)
                        if last_mean is not None and last_mean > 0.0:
                            if loss_val / last_mean > sb_thres:
                                skip_weight_update = True

        if not skip_weight_update:
            update_weights_fn(optimizer, model, config, iteration_counter)

        if trackers is not None and plot_types is not None:
            update_trackers(
                iteration_counter, trackers, loss_dict, pred_dict,
                float(scheduler.get_last_lr()[-1]), min_grad_norm,
                max_grad_norm, mean_grad_norm, plot_types,
            )

        scheduler.step()

        if iteration_counter >= start_iter + n_iters:
            finished = True

        if config["data_mode"] == "pipeline":
            torch.distributed.barrier()

        if rank == 0:
            do_save_trackers = False
            if iteration_counter > 0 and (iteration_counter - start_iter) % config["tracker_period"] == 0:
                if iteration_counter > start_iter + 1:
                    do_save_trackers = True
            if finished:
                do_save_trackers = True
            if trackers is not None and plot_types is not None and do_save_trackers:
                trackers_dir = mkdir_custom(
                    os.path.join(output_dir, "trackers"), "return"
                )
                trackers_dir_i = mkdir_custom(
                    os.path.join(trackers_dir, "iter_" + str(iteration_counter).zfill(7))
                )
                save_trackers(trackers_dir_i, trackers, plot_types, False)
                cleanup_trackers(
                    trackers_dir, iteration_counter, config["tracker_expiry_period"]
                )

            do_save_checkpoint = False
            if iteration_counter > 0 and (iteration_counter - start_iter) % config["checkpoint_period"] == 0:
                if iteration_counter > start_iter + 1:
                    do_save_checkpoint = True
            if finished:
                do_save_checkpoint = True
            if do_save_checkpoint:
                checkpoints_dir = mkdir_custom(
                    os.path.join(output_dir, "checkpoints"), "return"
                )
                save_checkpoint(
                    config, checkpoints_dir, iteration_counter, model,
                    cnoise, optimizer, scheduler, trackers, plot_types,
                    save_model_fn, save_cnoise_fn, get_paths_fn,
                )
                if verbose:
                    printf("checkpoint saved")

        iteration_counter += 1

        if "freeze_duration" in config["scheduler_config"].keys():
            if iteration_counter == config["scheduler_config"]["freeze_duration"]:
                model.module = check_model_state_fn(model.module, config, iteration_counter)

    if config["data_mode"] == "pipeline":
        torch.distributed.destroy_process_group()
