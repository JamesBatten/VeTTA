"""Typed data settings for buffers, tree datasets, and transforms."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from vetta.settings.base import VettaSettings


def _default_aug_config() -> dict[str, Any]:
    return {
        "pos_noise": 0.005,
        "probs": {
            "zoom": 0.5,
            "rotate": 0.5,
            "translate": 0.5,
            "jitter": 0.5,
        },
        "max_degrees": 45.0,
        "min_zoom": 0.75,
        "max_zoom": 1.5,
        "max_delta": 0.0,
        "margin_w": 0.02,
    }


def _default_cand_stats() -> dict[str, Any]:
    return {
        "ncand_gen": [8, 16, 32],
        "mdist_range": [0.05, 0.002],
        "uniform_rnd_bounds": [-0.1, 1.1],
        "std_fctr_range": [0.25, 0.01],
        "offset_fctr_0": 2.0,
        "offset_trunc_0": 5.0,
        "n_cells": 16,
        "k_noise": 16,
        "gnoise_size": 2.0,
        "gnoise_n_std": 3,
        "n_timesteps": 11,
        "zoom_base": 32.0,
        "mdist": 0.1,
    }


class TreeDatasetSettings(VettaSettings):
    mode: str = "train"
    seed: int | None = None
    rank: int | None = None
    start_port: int | None = None
    n_servers: int | None = None
    target_queue_length: int = 100
    allow_augmentation: bool = True
    allow_jitter: bool = False
    allow_zoom: bool = False
    allow_rotate: bool = True
    allow_translate: bool = True
    allow_rnd_add: bool = False
    load_segmentation: bool = False
    load_radius: bool = False
    normalise: bool = True
    pad_size: int = 0
    pad_val: float = 0.0
    pad_before_aug: bool = False
    tgt_seg_size: int = 250
    do_shuffle: bool = True
    rotate_mode: str = "pad_rotate_crop"
    sigma: float = 3.5
    aug_config: dict[str, Any] = Field(default_factory=_default_aug_config)
    norm_domain: list[float] = Field(default_factory=lambda: [-0.25, 0.25])
    wrap_domain: list[float] = Field(default_factory=lambda: [-0.5, 0.5])
    concat_posgrid: bool = True
    receive_mode: str = "tree"
    poll_time: float = 0.001
    tpn_timeout: float = 1.0
    show_warnings: bool = True
    add_splits: bool = False
    add_semi: bool = False
    filter_non_proximal: bool = True
    add_candidates: bool = False
    min_pivot_pos: Any = None
    max_pivot_pos: Any = None
    fix_cand_mode: str | None = None
    ignore_aug_bounds: bool = False
    static_cand_seed: int | None = None
    dynamic_cand_seed: int | None = None
    order_seed: int | None = None
    filter_no_children: bool = True
    n_candidates: int = 32
    max_targets: int = 2
    cand_time_range: list[float] = Field(default_factory=lambda: [0.0, 1.0])
    cand_stats: dict[str, Any] = Field(default_factory=_default_cand_stats)
    init_prob: float = 0.1
    posdims: int = 2
    verbose: bool = False
    target_mode: str = "multi"


class ChunkTreeBufferSettings(VettaSettings):
    target_queue_length: int = 100
    seed: int = 1
    do_shuffle: bool = True
    name: str = "chunk_tree_buffer"
    grayscale: bool = True
    load_segmentation: bool = False


class TreePipelineServerSettings(VettaSettings):
    mode: str = "train"
    seed: int = 1
    seed_prime: int = 11
    do_shuffle: bool = True
    n_chunk_tree_buffers: int = 2
    verbose: bool = False
    target_queue_length: int = 100
    poll_timeout: int = 100  # in milliseconds
    sleep_time: float = 1e-5
    tcp_port: int | None = None
    name: str = "tree_pipeline_server"
    send_mode: str = "tree"
    join_timeout: float = 5.0
    report_stats: bool = False
    load_segmentation: bool = False
    report_delay: float = 5.0


class TreePipelineSettings(VettaSettings):
    mode: str = "train"
    seed: int = 1
    seed_prime: int = 7
    do_shuffle: bool = True
    n_servers: int = 3
    n_chunk_tree_buffers: int = 2  # per server
    verbose: bool = False
    target_queue_length: int = 100
    poll_time: float = 0.001
    start_port: int = 5555
    send_mode: str = "tree"
    report_stats: bool = False
    load_segmentation: bool = False
    join_timeout: float = 5.0


class TreeRotateSettings(VettaSettings):
    max_degrees: float = 45.0
    rot_center: tuple[float, float] = (0.0, 0.0)
    fill: float = 0.0
    prob: float = 0.5
    wrap_domain: tuple[float, float] = (-0.5, 0.5)
    ignore_aug_bounds: bool = False


class TreeTranslateSettings(VettaSettings):
    max_delta: float = 0.0
    prob: float = 0.5
    wrap_domain: tuple[float, float] = (-0.5, 0.5)
    ignore_aug_bounds: bool = False


class TreeZoomSettings(VettaSettings):
    min_zoom: float = 0.75
    max_zoom: float = 1.5
    fill: float = 0.0
    zoom_center: tuple[float, float] = (0.5, 0.5)
    prob: float = 0.5
    wrap_domain: tuple[float, float] = (-0.5, 0.5)
    ignore_aug_bounds: bool = False
