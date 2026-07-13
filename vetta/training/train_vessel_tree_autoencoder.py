from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Any
from typing import Sequence

from vetta.data import TreePipeline
from vetta.training.common import str_to_bool
from vetta.training.environment import get_chunk_paths
from vetta.training.experiment import BaseExperiment
from vetta.training.trainer_vessel_tree_autoencoder import TrainerVesselTreeAutoEncoder
from vetta.training.autoencoder import VesselAutoencoderTrainConfigBuilder
from vetta.training.autoencoder import dataset_profile_for_name
from vetta.training.autoencoder import get_dataset_profile


def build_train_vessel_tree_autoencoder_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ssa")
    parser.add_argument("--chunks", type=str, default="chunks_all")
    parser.add_argument("--n_servers", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=3)
    parser.add_argument("--start_port", type=int, default=5555)
    parser.add_argument("--master_port", type=int, default=12355)
    parser.add_argument("--target_queue_length", type=int, default=100)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--encoder_layers", type=int, default=2)
    parser.add_argument("--decoder_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward_transformer", type=int, default=2048)
    parser.add_argument("--z_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--min_pivot_pos", type=int, default=None)
    parser.add_argument("--max_pivot_pos", type=int, default=None)
    parser.add_argument("--matching_k", type=int, default=3)
    parser.add_argument("--num_slots", type=int, default=32)
    parser.add_argument("--mlp_nonlinearity", type=str, default="gelu")
    parser.add_argument("--pos_cost", type=float, default=0.9)
    parser.add_argument("--topology_cost", type=float, default=1.0)
    parser.add_argument("--skip_vessel_cost", type=float, default=0.1)
    parser.add_argument("--rad_cost", type=float, default=0.1)
    parser.add_argument("--enc_v_cost", type=float, default=0.005)
    parser.add_argument("--order_seed", type=int, default=None)
    parser.add_argument("--optimizer_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--backwards_per_iteration", type=int, default=1)
    parser.add_argument("--checkpoint_period", type=int, default=1000)
    parser.add_argument("--scheduler_warmup", type=int, default=100)
    parser.add_argument("--scheduler_period", type=int, default=300)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument(
        "--output_root",
        "--output-root",
        dest="output_root",
        type=str,
        default=None,
        help=(
            "Base directory for experiment output. Defaults to "
            "<repo>/experiments_data/output_data. The experiment name and "
            "experiment_NNNNNN subdirectories are appended under this root, so "
            "point it at a large volume (e.g. /workspace/...) for long runs."
        ),
    )
    parser.add_argument("--n_steps", type=int, default=1_000_000)
    parser.add_argument("--zoom_prob", type=float, default=0.5)
    parser.add_argument("--translate_prob", type=float, default=0.5)
    parser.add_argument("--pos_jitter_prob", type=float, default=0.5)
    parser.add_argument("--rad_jitter_prob", type=float, default=0.0)
    parser.add_argument("--verbose", type=str, default="False")
    parser.add_argument("--pipeline_verbose", type=str, default="False")
    parser.add_argument("--save_batches", type=str, default="False")
    parser.add_argument("--allow_augmentation", type=str, default="False")
    parser.add_argument("--allow_jitter", type=str, default="False")
    parser.add_argument("--filter_non_proximal", type=str, default="False")
    parser.add_argument("--include_lhs_enc", type=str, default="False")
    parser.add_argument("--include_lhs_rad", type=str, default="False")
    parser.add_argument("--use_vae", type=str, default="False")
    parser.add_argument("--weight_reconst", type=float, default=0.999999)
    parser.add_argument("--weight_kl", type=float, default=0.000001)
    parser.add_argument(
        "--backwards-compatibility-paper",
        dest="backwards_compatibility_paper",
        action="store_true",
        default=False,
        help=(
            "Activate the paper/legacy behaviours needed to reproduce the "
            "original proposed checkpoints: MLP2 ignores norm_1 (hard-coded "
            "groupnorm) and VesselEdgesEncoder gathers topology_b from indices_a. "
            "See docs/public_legacy_surfaces.md. Off by default (the fixed "
            "behaviours); the value is recorded in the checkpoint config.json."
        ),
    )
    return parser


@dataclass(frozen=True)
class TrainVesselTreeAutoencoderFlags:
    verbose: bool
    pipeline_verbose: bool
    save_batches: bool
    allow_augmentation: bool
    allow_jitter: bool
    filter_non_proximal: bool
    include_lhs_enc: bool
    include_lhs_rad: bool
    use_vae: bool
    backwards_compatibility_paper: bool


def parse_train_vessel_tree_autoencoder_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, TrainVesselTreeAutoencoderFlags]:
    parser = build_train_vessel_tree_autoencoder_parser()
    args = parser.parse_args(argv)
    flags = TrainVesselTreeAutoencoderFlags(
        verbose=str_to_bool(args.verbose),
        pipeline_verbose=str_to_bool(args.pipeline_verbose),
        save_batches=str_to_bool(args.save_batches),
        allow_augmentation=str_to_bool(args.allow_augmentation),
        allow_jitter=str_to_bool(args.allow_jitter),
        filter_non_proximal=str_to_bool(args.filter_non_proximal),
        include_lhs_enc=str_to_bool(args.include_lhs_enc),
        include_lhs_rad=str_to_bool(args.include_lhs_rad),
        use_vae=str_to_bool(args.use_vae),
        backwards_compatibility_paper=args.backwards_compatibility_paper,
    )
    return args, flags


def resolve_train_vessel_tree_dataset_name(dataset: str) -> str:
    return get_dataset_profile(dataset).dataset_name


def _config_builder(
    args: argparse.Namespace,
    flags: TrainVesselTreeAutoencoderFlags,
    dataset_name: str,
) -> VesselAutoencoderTrainConfigBuilder:
    return VesselAutoencoderTrainConfigBuilder(
        args=args,
        flags=flags,
        profile=dataset_profile_for_name(dataset_name),
    )


def start_pipeline(
    args: argparse.Namespace,
    flags: TrainVesselTreeAutoencoderFlags,
    dataset_name: str,
) -> TreePipeline:
    builder = _config_builder(args, flags, dataset_name)
    chunk_paths = get_chunk_paths(builder.profile.chunk_dataset, "train", args.chunks)
    pipeline = TreePipeline(chunk_paths, builder.pipeline_config())
    pipeline.start()
    time.sleep(1.0)
    return pipeline


def make_dataset_config(
    args: argparse.Namespace,
    flags: TrainVesselTreeAutoencoderFlags,
    dataset_name: str,
) -> dict[str, Any]:
    return _config_builder(args, flags, dataset_name).dataset_config()


def make_model_config(
    args: argparse.Namespace,
    flags: TrainVesselTreeAutoencoderFlags,
    dataset_name: str,
) -> dict[str, Any]:
    return _config_builder(args, flags, dataset_name).model_config()


def make_trainer_config(
    args: argparse.Namespace,
    flags: TrainVesselTreeAutoencoderFlags,
    dataset_config: dict[str, Any],
    dataset_name: str,
) -> dict[str, Any]:
    return _config_builder(args, flags, dataset_name).trainer_config(dataset_config)


class TrainVesselTreeAutoencoderExperiment(BaseExperiment):
    def __init__(
        self,
        args: argparse.Namespace,
        flags: TrainVesselTreeAutoencoderFlags,
    ):
        self.args = args
        self.flags = flags

    def name(self) -> str:
        return "train_vessel_tree_autoencoder"

    def output_root(self) -> str:
        if self.args.output_root is not None:
            return os.path.expanduser(self.args.output_root)
        return super().output_root()

    def execute(self, output_dir: str) -> None:
        dataset_name = resolve_train_vessel_tree_dataset_name(self.args.dataset)
        pipeline = start_pipeline(self.args, self.flags, dataset_name)
        dataset_config = make_dataset_config(self.args, self.flags, dataset_name)
        trainer_config = make_trainer_config(
            self.args,
            self.flags,
            dataset_config,
            dataset_name,
        )
        trainer = TrainerVesselTreeAutoEncoder(
            config=trainer_config,
            load_checkpoint=self.args.load_checkpoint,
        )
        try:
            trainer.train(output_dir, self.args.n_steps)
        finally:
            pipeline.stop()


def run_cli(argv: Sequence[str] | None = None) -> None:
    args, flags = parse_train_vessel_tree_autoencoder_args(argv)
    experiment = TrainVesselTreeAutoencoderExperiment(args, flags)
    experiment.run()
