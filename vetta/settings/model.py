"""Typed model settings for the private package."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from vetta.settings.base import VettaSettings


def _default_vtenc_config() -> dict[str, Any]:
    # Imported lazily so that importing the settings package does not pull in the
    # model package at module load time (settings.model <-> model would
    # otherwise form an import cycle when settings is imported first).
    from vetta.model import VesselEdgesEncoder

    return VesselEdgesEncoder.default_config()


class VesselTreeAutoencoderSettings(VettaSettings):
    vtenc_full_config: dict[str, Any] = Field(default_factory=_default_vtenc_config)
    vtenc_partial_config: dict[str, Any] = Field(default_factory=_default_vtenc_config)
    n_slots: int = 32
    z_dim: int = 64
    data_size: int = 70
    enc_size: int = 64
    mlp_hidden_dims: int = 2048
    n_heads: int = 12
    vdim: int = 64
    mlp_nonlinearity: str = "gelu"
    dropout: float = 0.0
    n_encoder_layers: int = 6
    n_decoder_layers: int = 6
    dim_feedforward_transformer: int = 2048
    activation_transformer: str = "relu"
    pos_octaves: list[int] = Field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    wrap_domain: tuple[float, float] = (-0.5, 0.5)
    # Octave-target position domain for out_mode="octaves": the octaves loss maps
    # normalised positions through this domain to [0, 1] before octave encoding,
    # and inference maps the decoded fraction back through it. The trained
    # proposed checkpoints use [-3, 3]; the model/encoder itself only reads
    # wrap_domain, but the octaves loss needs this. Train and infer must share it.
    domain: tuple[float, float] = (-3.0, 3.0)
    posdims: int = 2
    topology_size: int = 3
    out_mode: str = "octaves"
    input_include_base: bool = False
    include_skip_vessel: bool = False
    include_edges_enc_v: bool = False
    include_rad: bool = False
    include_lhs_enc: bool = False
    include_lhs_rad: bool = False
    use_vae: bool = False
    # Umbrella switch for the paper/legacy behaviours needed to load the existing
    # proposed checkpoints (see docs/public_legacy_surfaces.md). When True the
    # model reproduces two baked-in quirks: MLP2 ignores norm_1 (hard-coded
    # groupnorm) and VesselEdgesEncoder gathers topology_b from indices_a. Defaults
    # to False (the fixed behaviours).
    backwards_compatibility_paper: bool = False


VesselTreeAutoencoderConfig = VesselTreeAutoencoderSettings
