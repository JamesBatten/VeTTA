from __future__ import annotations

import argparse
import os
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel

from vetta.data import TreeDataset
from vetta.model import AutoencoderBatch
from vetta.model import VesselTreeAutoencoder
from vetta.model import VesselTreeAutoencoderConfig
from vetta.model import add_octaves
from vetta.settings.data import TreeDatasetSettings
from vetta.settings.data import TreePipelineSettings
from vetta.settings.model import VesselTreeAutoencoderSettings
from vetta.settings.training import TrainerVesselTreeAutoencoderSettings
from vetta.training.common import normalize_domain
from vetta.training.matching import batch_topk_matching_fast
from vetta.utils import mkdir_custom
from vetta.utils import to_tensor


# Profiles

@dataclass(frozen=True)
class DatasetProfile:
    """Static description of one vessel-tree dataset variant."""

    cli_name: str
    dataset_name: str
    chunk_dataset: str
    data_size: int
    posdims: int = 2
    include_rad: bool = False
    load_radius: bool = False
    dataset_cls: type[TreeDataset] = field(default=TreeDataset)


DATASET_PROFILES: dict[str, DatasetProfile] = {
    "ssa": DatasetProfile(
        cli_name="ssa",
        dataset_name="tree_dataset",
        chunk_dataset="ssa",
        data_size=54,
    ),
    "ssa_rad": DatasetProfile(
        cli_name="ssa_rad",
        dataset_name="tree_dataset_rad",
        chunk_dataset="ssa",
        data_size=56,
        include_rad=True,
        load_radius=True,
    ),
}

_PROFILES_BY_DATASET_NAME: dict[str, DatasetProfile] = {
    profile.dataset_name: profile for profile in DATASET_PROFILES.values()
}


def get_dataset_profile(cli_name: str) -> DatasetProfile:
    """Look up a profile by its CLI ``--dataset`` name (``ssa`` / ``ssa_rad``)."""
    try:
        return DATASET_PROFILES[cli_name]
    except KeyError:
        raise ValueError(f"unknown dataset {cli_name!r}") from None


def dataset_profile_for_name(dataset_name: str) -> DatasetProfile:
    """Look up a profile by its internal dataset name (``tree_dataset`` etc.)."""
    try:
        return _PROFILES_BY_DATASET_NAME[dataset_name]
    except KeyError:
        raise ValueError(f"unknown dataset name {dataset_name!r}") from None


# Math

def log_safe(x: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Natural log with a tiny epsilon, dispatching on tensor vs ndarray.

    Reproduces the legacy ``_log_like`` behaviour shared by the prediction and
    loss modules: ``log(x + 1e-10)`` for both ``torch.Tensor`` and ``np.ndarray``
    inputs. Group 5 parity of the ``backwards_compatibility_paper`` umbrella (see
    docs/public_legacy_surfaces.md): a comment-only parity claim, not a toggle.
    """
    if torch.is_tensor(x):
        return torch.log(x + 1e-10)
    return np.log(x + 1e-10)


# Config

def apply_nested_overrides(
    config: dict[str, Any],
    overrides: Mapping[tuple[str, ...], Any],
) -> None:
    """Apply ``{(key, ...): value}`` overrides into a nested config dict in place."""
    for path, value in overrides.items():
        target = config
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value


@dataclass(frozen=True)
class VesselAutoencoderTrainConfigBuilder:
    """Builds the pipeline/dataset/model/trainer config dicts from CLI args.

    ``flags`` carries the parsed boolean values; ``profile`` is the single source
    of truth for the per-dataset differences (data size, radius handling).
    """

    args: argparse.Namespace
    flags: Any
    profile: DatasetProfile

    @property
    def dataset_name(self) -> str:
        return self.profile.dataset_name

    def pipeline_config(self) -> TreePipelineSettings:
        overrides = {
            "n_servers": self.args.n_servers,
            "start_port": self.args.start_port,
            "target_queue_length": self.args.target_queue_length,
            "verbose": self.flags.pipeline_verbose,
        }
        return TreePipelineSettings.from_defaults(overrides)

    def dataset_config(self) -> dict[str, Any]:
        overrides: dict[str, Any] = {
            "n_servers": self.args.n_servers,
            "start_port": self.args.start_port,
            "add_splits": True,
            "add_semi": True,
            "min_pivot_pos": self.args.min_pivot_pos,
            "max_pivot_pos": self.args.max_pivot_pos,
            "order_seed": self.args.order_seed,
            "allow_augmentation": self.flags.allow_augmentation,
            "allow_jitter": self.flags.allow_jitter,
            "filter_non_proximal": self.flags.filter_non_proximal,
        }
        if self.profile.load_radius:
            overrides["load_radius"] = True

        dataset_config = TreeDatasetSettings.from_defaults(overrides).to_dict()
        dataset_config["aug_config"]["probs"]["zoom"] = self.args.zoom_prob
        dataset_config["aug_config"]["probs"]["translate"] = self.args.translate_prob
        dataset_config["aug_config"]["probs"]["pos_jitter"] = self.args.pos_jitter_prob
        dataset_config["aug_config"]["probs"]["rad_jitter"] = self.args.rad_jitter_prob
        return dataset_config

    def model_config(self) -> dict[str, Any]:
        overrides: dict[str, Any] = {
            "n_heads": self.args.n_heads,
            "n_encoder_layers": self.args.encoder_layers,
            "n_decoder_layers": self.args.decoder_layers,
            "z_dim": self.args.z_dim,
            "n_slots": self.args.num_slots,
            "mlp_nonlinearity": self.args.mlp_nonlinearity,
            "include_lhs_enc": self.flags.include_lhs_enc,
            "include_lhs_rad": self.flags.include_lhs_rad,
            "dim_feedforward_transformer": self.args.dim_feedforward_transformer,
            "use_vae": self.flags.use_vae,
            "data_size": self.profile.data_size,
            "posdims": self.profile.posdims,
            "backwards_compatibility_paper": self.flags.backwards_compatibility_paper,
        }
        if self.profile.include_rad:
            overrides["include_rad"] = True
        return VesselTreeAutoencoderSettings.from_defaults(overrides).to_dict()

    def trainer_config(self, dataset_config: dict[str, Any]) -> dict[str, Any]:
        args = self.args
        overrides = {
            "dataset_name": self.dataset_name,
            "save_batches": self.flags.save_batches,
            "matching_k": args.matching_k,
            "checkpoint_period": args.checkpoint_period,
            "use_vae": self.flags.use_vae,
            "verbose": self.flags.verbose,
            "master_port": args.master_port,
            "dataset_configs_dict": {self.dataset_name: dataset_config},
        }
        trainer_config = TrainerVesselTreeAutoencoderSettings.from_defaults(
            overrides
        ).to_dict()
        trainer_config["model_config"] = self.model_config()
        apply_nested_overrides(
            trainer_config,
            {
                ("cost_factors", "pos"): args.pos_cost,
                ("cost_factors", "topology"): args.topology_cost,
                ("cost_factors", "skip_vessel"): args.skip_vessel_cost,
                ("cost_factors", "enc_v"): args.enc_v_cost,
                ("cost_factors", "rad"): args.rad_cost,
                ("scheduler_config", "scheduler_warmup"): args.scheduler_warmup,
                ("scheduler_config", "scheduler_period"): args.scheduler_period,
                ("optimizer_config", "optimizer_lr"): args.optimizer_lr,
                ("optimizer_config", "weight_decay"): args.weight_decay,
                ("optimizer_config", "backwards_per_iteration"): args.backwards_per_iteration,
                ("optimizer_config", "weight_reconst"): args.weight_reconst,
                ("optimizer_config", "weight_kl"): args.weight_kl,
                ("batch_sizes_dict", self.dataset_name): args.batch_size,
            },
        )
        return trainer_config


# Dataset

def dataset_name_to_cls(
    dataset_name: str,
) -> type[TreeDataset]:
    return dataset_profile_for_name(dataset_name).dataset_cls


# Checkpoint

def get_paths(checkpoint_dir: str) -> dict[str, str]:
    model_dir = os.path.join(checkpoint_dir, "model")
    return {
        "model": model_dir,
        "vt_autoencoder": os.path.join(model_dir, "vt_autoencoder"),
        "optimizer": os.path.join(checkpoint_dir, "optimizer"),
        "scheduler": os.path.join(checkpoint_dir, "scheduler"),
        "trackers": os.path.join(checkpoint_dir, "trackers"),
        "config": os.path.join(checkpoint_dir, "config.json"),
        "metadata": os.path.join(checkpoint_dir, "metadata.json"),
    }


def save_model(model: torch.nn.Module, paths: dict[str, str]) -> None:
    mkdir_custom(paths["model"])
    torch.save(model.state_dict(), paths["vt_autoencoder"])


# Model

def make_model(
    model_config: dict[str, Any],
    load_checkpoint: str | None,
    paths: dict[str, str],
    device_str: str,
    extras_dict: dict[str, Any],
) -> torch.nn.Module:
    del extras_dict
    model = VesselTreeAutoencoder(VesselTreeAutoencoderConfig.from_mapping(model_config))
    if load_checkpoint:
        model.load_state_dict(torch.load(paths["vt_autoencoder"]))
    return model.to(device_str)


def check_model_state(
    model: torch.nn.Module,
    trainer_config: dict[str, Any],
    iteration: int,
) -> torch.nn.Module:
    del trainer_config
    del iteration
    return model


def get_extras_dict(trainer_config: dict[str, Any]) -> dict[str, Any]:
    del trainer_config
    return {}


# Plotting

NORMAL_AND_LOG = ["normal", "log"]
LOG_ONLY = ["log"]


BASE_PLOT_TYPES: dict[str, list[str]] = {
    "learning_rate": NORMAL_AND_LOG,
    "loss": NORMAL_AND_LOG,
    "reconst_loss_lhs": NORMAL_AND_LOG,
    "reconst_loss_rhs": NORMAL_AND_LOG,
    "pos_loss": LOG_ONLY,
    "unweighted_pos_loss": LOG_ONLY,
    "top_loss": LOG_ONLY,
    "unweighted_top_loss": LOG_ONLY,
    "skip_vessel_loss": LOG_ONLY,
    "unweighted_skip_vessel_loss": LOG_ONLY,
    "enc_loss": LOG_ONLY,
    "unweighted_enc_loss": LOG_ONLY,
    "lograd_loss": LOG_ONLY,
    "unweighted_lograd_loss": LOG_ONLY,
    "min_grad_norm": NORMAL_AND_LOG,
    "max_grad_norm": NORMAL_AND_LOG,
    "mean_grad_norm": NORMAL_AND_LOG,
}

VAE_PLOT_TYPES: dict[str, list[str]] = {
    "kl_loss": NORMAL_AND_LOG,
    "reconst_loss": NORMAL_AND_LOG,
    "unweighted_reconst_loss": NORMAL_AND_LOG,
    "unweighted_kl_loss": NORMAL_AND_LOG,
}


def plot_types(config: dict[str, Any]) -> dict[str, list[str]]:
    # Fresh copies of the mode lists so callers can't mutate the shared
    # constants through the returned map.
    ret = {key: list(modes) for key, modes in BASE_PLOT_TYPES.items()}
    if config["use_vae"]:
        ret.update({key: list(modes) for key, modes in VAE_PLOT_TYPES.items()})
    return ret


# Prediction

def make_autoencoder_batch(
    bdict: dict[str, Any],
    *,
    include_radius: bool,
) -> AutoencoderBatch:
    """Build an ``AutoencoderBatch`` from a tree dataset batch dict.

    Radius (``lograd`` / ``lograd_lhs``) is the only dataset-dependent part: it
    is added only when ``include_radius`` is set, centralising the
    "radius means log-radius fields" rule.
    """
    kwargs: dict[str, Any] = {
        "pos": bdict["global_pos"],
        "depth": bdict["depth"],
        "edges": bdict["edges"],
        "edges_mask": bdict["edges_mask"],
        "pos_lhs": bdict["global_pos_lhs"],
        "depth_lhs": bdict["depth_lhs"],
        "edges_lhs": bdict["edges_lhs"],
        "edges_mask_lhs": bdict["edges_mask_lhs"],
        "topology": bdict["topology"],
        "topology_lhs": bdict["topology_lhs"],
        "query_idx": bdict["query_idx"],
        "qeidx": bdict["qeidx"],
    }
    if include_radius:
        kwargs["lograd"] = log_safe(bdict["radius"])
        kwargs["lograd_lhs"] = log_safe(bdict["radius_lhs"])
    return AutoencoderBatch.from_tree_tensors(**kwargs)


def compute_pred(
    train_batch_dict: dict[str, dict[str, Any]],
    model: VesselTreeAutoencoder | DistributedDataParallel,
    config: dict[str, Any],
    iteration_counter: int,
) -> dict[str, torch.Tensor]:
    del iteration_counter
    dname = config["dataset_name"]
    profile = dataset_profile_for_name(dname)
    batch = make_autoencoder_batch(
        train_batch_dict[dname], include_radius=profile.include_rad
    )
    # The distributed train.py path wraps the model in DistributedDataParallel,
    # which only intercepts forward(); the custom forward_batch lives on the
    # underlying module. Unwrap (mirroring base.py's model.module usage) so both
    # the wrapped pipeline path and the unwrapped integration path work.
    core = model.module if isinstance(model, DistributedDataParallel) else model
    return core.forward_batch(batch, to_cuda=config["to_cuda"])


# Loss

# Loss metrics that are intentionally always None today because skip_vessel and
# encoder heads are inactive in this trainer.
INACTIVE_LOSS_METRICS = (
    "skip_vessel_loss",
    "unweighted_skip_vessel_loss",
    "enc_loss",
    "unweighted_enc_loss",
)


@dataclass(frozen=True)
class Matching:
    """Left/right soft-assignment masks from the topk matcher."""

    lhs: torch.Tensor
    rhs: torch.Tensor


@dataclass(frozen=True)
class CostMatrices:
    """Per-component cost matrices plus their weighted total."""

    total: torch.Tensor
    components: dict[str, torch.Tensor]


def compute_loss_kl(
    z_mu: torch.Tensor,
    z_logvar: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    del config
    kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mu.pow(2) - z_logvar.exp())
    kl_loss /= z_mu.size(0)
    return kl_loss


def compute_unweighted_loss_aux(
    cm: torch.Tensor,
    ml: torch.Tensor,
    mr: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_l = torch.sum(cm * ml) / torch.sum(ml)
    loss_r = torch.sum(cm * mr) / torch.sum(mr)
    return loss_l + loss_r, loss_l, loss_r


def compute_weighted_loss_aux(
    cm: torch.Tensor,
    ml: torch.Tensor,
    mr: torch.Tensor,
    c: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_l = torch.sum(cm * ml) / torch.sum(ml) * c
    loss_r = torch.sum(cm * mr) / torch.sum(mr) * c
    return loss_l + loss_r, loss_l, loss_r


def _compute_cost_matrices(
    pred_dict: dict[str, torch.Tensor],
    bdict: dict[str, torch.Tensor],
    include_rad: bool,
    config: dict[str, Any],
) -> CostMatrices:
    """Build the per-component cost matrices and the weighted total.

    Mirrors the legacy inline construction: position and topology always,
    log-radius only for the radius dataset when the model emits ``lograd``.
    """
    pred_pos = pred_dict["pos"].unsqueeze(2)
    pred_top = pred_dict["topology"].unsqueeze(2)
    tgt_top = bdict["topology"].unsqueeze(1)
    if config["to_cuda"]:
        tgt_top = tgt_top.cuda()

    tgt_pos = bdict["global_pos"].unsqueeze(1)
    if config["to_cuda"]:
        tgt_pos = tgt_pos.cuda()

    tgt_lograd = None
    pred_lograd = None
    if include_rad:
        tgt_lograd = log_safe(bdict["radius"]).unsqueeze(1)
        if config["to_cuda"]:
            tgt_lograd = tgt_lograd.cuda()
        pred_lograd = pred_dict.get("lograd")

    if config["model_config"]["out_mode"] == "octaves":
        tgt_pos, _ = normalize_domain(tgt_pos, None, config["model_config"]["domain"])
        channels = list(range(config["model_config"]["posdims"]))
        tgt_pos = add_octaves(
            tgt_pos,
            octaves=config["model_config"]["pos_octaves"],
            dim=-1,
            channels=channels,
            ret_type="torch",
            include_base=False,
        )
        if config["to_cuda"]:
            tgt_pos = tgt_pos.cuda()

    pos_cost_mat = torch.sum((pred_pos - tgt_pos) ** 2.0, dim=-1)
    pos_cost_mat /= float(pred_pos.shape[-1])

    top_cost_mat = torch.sum((pred_top - tgt_top) ** 2.0, dim=-1)
    top_cost_mat /= float(pred_top.shape[-1])

    cf = config["cost_factors"]
    cost_mat = cf["pos"] * pos_cost_mat + cf["topology"] * top_cost_mat

    components: dict[str, torch.Tensor] = {
        "pos": pos_cost_mat,
        "topology": top_cost_mat,
    }
    if include_rad and pred_lograd is not None:
        lograd_cost_mat = (pred_lograd - tgt_lograd) ** 2.0
        cost_mat = cost_mat + cf["rad"] * lograd_cost_mat
        components["lograd"] = lograd_cost_mat

    return CostMatrices(total=cost_mat, components=components)


def _compute_matching(
    cost_total: torch.Tensor,
    bdict: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> Matching:
    qc = bdict["query_children"]
    if config["to_cuda"]:
        qc = qc.cuda()
    mlhs_np, mrhs_np = batch_topk_matching_fast(cost_total, qc, config["matching_k"])
    mlhs = to_tensor(mlhs_np)
    rlhs = to_tensor(mrhs_np)
    if config["to_cuda"]:
        mlhs = mlhs.cuda()
        rlhs = rlhs.cuda()
    return Matching(lhs=mlhs, rhs=rlhs)


def _add_vae_losses(
    ret: dict[str, torch.Tensor | None],
    pred_dict: dict[str, torch.Tensor],
    unweighted_reconst_loss: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    """Populate VAE loss keys and return the final ``loss`` value.

    When VAE is disabled the loss is just the unweighted reconstruction term and
    no extra keys are added (matching the legacy behaviour).
    """
    if not config["use_vae"]:
        return unweighted_reconst_loss

    unweighted_kl_loss = compute_loss_kl(
        pred_dict["z_mu"], pred_dict["z_logvar"], config
    )
    reconst_loss = config["optimizer_config"]["weight_reconst"] * unweighted_reconst_loss
    kl_loss = config["optimizer_config"]["weight_kl"] * unweighted_kl_loss
    ret["kl_loss"] = kl_loss
    ret["reconst_loss"] = reconst_loss
    ret["unweighted_reconst_loss"] = unweighted_reconst_loss
    ret["unweighted_kl_loss"] = unweighted_kl_loss
    return reconst_loss + kl_loss


def compute_loss(
    pred_dict: dict[str, torch.Tensor],
    train_batch_dict: dict[str, dict[str, Any]],
    iteration_counter: int,
    config: dict[str, Any],
) -> dict[str, torch.Tensor | None]:
    del iteration_counter
    profile = dataset_profile_for_name(config["dataset_name"])
    bdict = train_batch_dict[config["dataset_name"]]

    cost_matrices = _compute_cost_matrices(pred_dict, bdict, profile.include_rad, config)
    matching = _compute_matching(cost_matrices.total, bdict, config)

    unweighted_reconst_loss, reconst_loss_lhs, reconst_loss_rhs = (
        compute_unweighted_loss_aux(cost_matrices.total, matching.lhs, matching.rhs)
    )

    cf = config["cost_factors"]
    pos_cost_mat = cost_matrices.components["pos"]
    top_cost_mat = cost_matrices.components["topology"]
    pos_loss, _, _ = compute_weighted_loss_aux(pos_cost_mat, matching.lhs, matching.rhs, cf["pos"])
    unweighted_pos_loss, _, _ = compute_unweighted_loss_aux(pos_cost_mat, matching.lhs, matching.rhs)
    top_loss, _, _ = compute_weighted_loss_aux(top_cost_mat, matching.lhs, matching.rhs, cf["topology"])
    unweighted_top_loss, _, _ = compute_unweighted_loss_aux(top_cost_mat, matching.lhs, matching.rhs)

    lograd_cost_mat = cost_matrices.components.get("lograd")
    if lograd_cost_mat is not None:
        lograd_loss, _, _ = compute_weighted_loss_aux(
            lograd_cost_mat, matching.lhs, matching.rhs, cf["rad"]
        )
        unweighted_lograd_loss, _, _ = compute_unweighted_loss_aux(
            lograd_cost_mat, matching.lhs, matching.rhs
        )
    else:
        lograd_loss = None
        unweighted_lograd_loss = None

    ret_dict: dict[str, torch.Tensor | None] = {
        "loss": None,  # filled by _add_vae_losses below
        "reconst_loss_lhs": reconst_loss_lhs,
        "reconst_loss_rhs": reconst_loss_rhs,
        "pos_loss": pos_loss,
        "unweighted_pos_loss": unweighted_pos_loss,
        "top_loss": top_loss,
        "unweighted_top_loss": unweighted_top_loss,
        "skip_vessel_loss": None,
        "unweighted_skip_vessel_loss": None,
        "enc_loss": None,
        "unweighted_enc_loss": None,
        "lograd_loss": lograd_loss,
        "unweighted_lograd_loss": unweighted_lograd_loss,
    }
    for name in INACTIVE_LOSS_METRICS:
        ret_dict[name] = None

    ret_dict["loss"] = _add_vae_losses(
        ret_dict, pred_dict, unweighted_reconst_loss, config
    )
    return ret_dict
