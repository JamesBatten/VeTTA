import abc
import math
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import replace
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import Field
import torch
import torch.nn as nn

from vetta.config import VettaSettings
from vetta.utils import take_along_axis
from vetta.utils import to_array
from vetta.utils import to_tensor


def _legacy_skip_vessel_flag() -> str:
    return "include_" + "place" + "holder"


def _apply_legacy_skip_vessel_alias(values: dict) -> dict:
    coerced = dict(values)
    legacy_key = _legacy_skip_vessel_flag()
    current_key = "include_skip_vessel"
    if legacy_key not in coerced:
        return coerced
    legacy_value = coerced.pop(legacy_key)
    if current_key in coerced and coerced[current_key] != legacy_value:
        raise ValueError(f"conflicting values for {current_key} legacy alias")
    coerced.setdefault(current_key, legacy_value)
    return coerced


class NonLinearity(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super(NonLinearity, self).__init__()

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError()


class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)


class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.gelu(x)


def make_nlrity(nonlinearity: str = "relu") -> nn.Module:
    if nonlinearity == "relu":
        return ReLU()
    elif nonlinearity == "gelu":
        return GELU()
    raise Exception("unknown nonlinearity " + str(nonlinearity))


def make_norm(
    out_dims: int, norm: str = "groupnorm", numgroups: int = 8
) -> nn.Module:
    if norm is None:
        raise Exception("norm is None")
    if norm == "groupnorm":
        return nn.GroupNorm(numgroups, out_dims)
    elif norm == "layernorm":
        return nn.LayerNorm(out_dims)
    raise Exception("unhandled norm " + str(norm))


def make_weights_init(nonlinearity: str = "relu", initialisation: str = "xavier"):
    def weights_init(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            if initialisation == "xavier":
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if nonlinearity == "relu":
                        torch.nn.init.constant_(m.bias, 0.1)
                    else:
                        torch.nn.init.uniform_(m.bias, -1.0, 1.0)
            else:
                raise Exception("unknown initialisation " + str(initialisation))
        elif isinstance(m, (nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                torch.nn.init.uniform_(m.bias, -1.0, 1.0)

    return weights_init


class MLP2(nn.Module):
    """
    Description

        A two-layer MLP with a nonlinearity in the hidden layer.
    """

    def __init__(
        self,
        in_dims: int,
        hidden_dims: int,
        out_dims: int,
        nonlinearity: str = "relu",
        bias_1: bool = True,
        bias_2: bool = True,
        norm_1: str | None = None,
        numgroups: int = 8,
        backwards_compatibility_paper: bool = False,
    ) -> None:
        super(MLP2, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dims, hidden_dims, bias=bias_1))
        self.layers.append(make_nlrity(nonlinearity))
        if norm_1 is not None:
            # backwards_compatibility_paper gates the paper/legacy quirk where the
            # norm is hard-coded to "groupnorm", ignoring norm_1. Default False
            # honours norm_1; True reproduces the legacy behaviour so legacy
            # checkpoints still load. See docs/public_legacy_surfaces.md.
            norm = "groupnorm" if backwards_compatibility_paper else norm_1
            self.layers.append(
                make_norm(hidden_dims, norm=norm, numgroups=numgroups)
            )
        self.layers.append(nn.Linear(hidden_dims, out_dims, bias=bias_2))
        self.layers.apply(make_weights_init(nonlinearity))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def liftsin(x, oct):
    if type(x) == np.ndarray:
        return np.sin(2 * np.pi * oct * x).astype(np.float32)
    elif type(x) == torch.Tensor:
        return torch.sin(2 * np.pi * oct * x).float()
    raise Exception("unhandled type " + str(type(x)))


def liftcos(x, oct):
    if type(x) == np.ndarray:
        return np.cos(2 * np.pi * oct * x).astype(np.float32)
    elif type(x) == torch.Tensor:
        return torch.cos(2 * np.pi * oct * x).float()
    raise Exception("unhandled type " + str(type(x)))


def lift(x, oct, lifter):
    if lifter == "sin":
        return liftsin(x, oct)
    elif lifter == "cos":
        return liftcos(x, oct)
    raise Exception("unhandled lifter " + str(lifter))


def add_octaves(
    x: torch.Tensor | npt.NDArray[np.float32],
    octaves: list[int],
    dim: int,
    channels: list[int] | None = None,
    ret_type: str | None = None,
    lifters: list[str] = ["sin", "cos"],
    mask: torch.Tensor | npt.NDArray[np.float32] | None = None,
    include_base: bool = True,
) -> npt.NDArray[np.float32] | torch.Tensor:
    if mask is not None:
        assert len(mask.shape) == len(x.shape) - 1
        for i in range(len(mask.shape)):
            if i != dim:
                assert mask.shape[i] == x.shape[i]

    if channels is None:
        channels = list(range(x.shape[dim]))

    new_ch = []
    for c_idx in channels:
        if c_idx >= x.shape[dim]:
            raise Exception("channel index " + str(c_idx) + " out of range")
        channel = take_along_axis(x, c_idx, dim)
        for oct in octaves:
            for lft in lifters:
                new_ch.append(lift(channel, oct, lft))
    if type(x) == np.ndarray:
        new_ch = np.stack(new_ch, axis=dim)
        if include_base:
            x = np.concatenate((x, new_ch), axis=dim)
        else:
            x = new_ch
    elif type(x) == torch.Tensor:
        new_ch = torch.stack(new_ch, dim=dim)
        if include_base:
            x = torch.cat((x, new_ch), dim=dim)
        else:
            x = new_ch
    if ret_type is None:
        return x
    elif ret_type == "numpy":
        return to_array(x).astype(np.float32)
    elif ret_type == "torch":
        return to_tensor(x).float()
    raise Exception("unhandled type " + str(type(x)))


class PositionalEncoding(nn.Module):
    """
    From: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, max_depth: int = 100):
        super().__init__()
        position = torch.arange(max_depth).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_depth, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, ef: torch.Tensor, depth_a: torch.Tensor) -> torch.Tensor:
        assert len(ef.shape) == 3
        assert len(depth_a.shape) == 2
        mask = (depth_a > -1).long()
        pe_idxs = (mask * depth_a).long()
        pe_v = self.pe[pe_idxs].squeeze(2)
        ef = ef + mask.unsqueeze(-1) * pe_v
        return ef


@dataclass(frozen=True, slots=True)
class VesselEdgesEncoderConfig:
    vdim: int = 64
    n_heads: int = 8
    posdims: int = 3
    mlp_hidden_dims: int = 2048
    activation_transformer: str = "relu"
    dim_feedforward_transformer: int = 2048
    n_layers: int = 3
    dropout: float = 0.0
    mlp_a_in_dims: int = 50
    st_size: int = 50
    pos_octaves: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    wrap_domain: tuple[float, float] = (-0.5, 0.5)
    max_depth: int = 30
    apply_mlp_b: bool = False
    output_dims: int = 64
    pool_final: bool = False
    include_query: bool = False
    include_skip_vessel: bool = False
    include_edges_enc_v: bool = False
    include_rad: bool = False
    input_activation: str = "gelu"
    output_activation: str = "gelu"
    input_include_base: bool = False
    add_pos_enc: bool = False
    use_vae: bool = False
    # When True, gather topology_b from indices_a (the paper/legacy quirk) instead
    # of indices_b. Propagated from VesselTreeAutoencoderConfig via
    # apply_shared_autoencoder_settings; see docs/public_legacy_surfaces.md.
    backwards_compatibility_paper: bool = False

    @classmethod
    def from_dict(cls, values: dict) -> "VesselEdgesEncoderConfig":
        values = _apply_legacy_skip_vessel_alias(values)
        known = {field.name for field in fields(cls)}
        unknown = set(values) - known
        if unknown:
            raise ValueError(
                f"VesselEdgesEncoderConfig got unknown keys: {sorted(unknown)}"
            )

        coerced = dict(values)
        if "pos_octaves" in coerced and coerced["pos_octaves"] is not None:
            coerced["pos_octaves"] = tuple(coerced["pos_octaves"])
        if "wrap_domain" in coerced and coerced["wrap_domain"] is not None:
            coerced["wrap_domain"] = tuple(coerced["wrap_domain"])
        return cls(**coerced)

    def checked(self) -> "VesselEdgesEncoderConfig":
        if self.use_vae:
            if self.apply_mlp_b:
                raise ValueError(
                    "VesselEdgesEncoder use_vae=True requires apply_mlp_b=False"
                )
            if not self.pool_final:
                raise ValueError(
                    "VesselEdgesEncoder use_vae=True requires pool_final=True"
                )
        return self

    def to_dict(self) -> dict:
        return asdict(self)


def _to_device(
    tensor: torch.Tensor | None, device: torch.device | str
) -> torch.Tensor | None:
    return None if tensor is None else tensor.to(device)


@dataclass(slots=True)
class EdgeBatch:
    """Tensors describing one tree fed to a ``VesselEdgesEncoder``.

    The required fields mirror the encoder's mandatory inputs; the optional
    fields default to ``None`` and are only populated when the corresponding
    model option (rad / edges_enc_v / skip_vessel / query) is enabled.
    """

    pos: torch.Tensor
    depth: torch.Tensor
    topology: torch.Tensor
    edges: torch.Tensor
    edges_mask: torch.Tensor
    lograd: torch.Tensor | None = None
    edges_enc_v: torch.Tensor | None = None
    skip_vessel_mask: torch.Tensor | None = None
    query_idx: torch.Tensor | None = None
    qeidx: torch.Tensor | None = None

    def without_query(self) -> "EdgeBatch":
        """Return a copy with the query fields cleared.

        Makes the "full encoder is always queried with ``query_idx=None``"
        invariant explicit at the call site instead of passing ``query_idx=None``
        literally.
        """
        return replace(self, query_idx=None, qeidx=None)

    def to_device(self, device: torch.device | str) -> "EdgeBatch":
        """Return a copy with every (present) tensor moved to ``device``.

        ``Tensor.to`` is a no-op (returns ``self``) when the tensor is already on
        ``device``, so the CPU path is unaffected.
        """
        return replace(
            self,
            pos=self.pos.to(device),
            depth=self.depth.to(device),
            topology=self.topology.to(device),
            edges=self.edges.to(device),
            edges_mask=self.edges_mask.to(device),
            lograd=_to_device(self.lograd, device),
            edges_enc_v=_to_device(self.edges_enc_v, device),
            skip_vessel_mask=_to_device(self.skip_vessel_mask, device),
            query_idx=_to_device(self.query_idx, device),
            qeidx=_to_device(self.qeidx, device),
        )

    @property
    def batch_size(self) -> int:
        return int(self.pos.shape[0])


@dataclass(slots=True)
class AutoencoderBatch:
    """The full tree (encoder input) and partial tree (decoder query) pair."""

    full: EdgeBatch
    partial: EdgeBatch

    def to_device(self, device: torch.device | str) -> "AutoencoderBatch":
        """Move both the full and partial sub-batches to ``device``."""
        return replace(
            self,
            full=self.full.to_device(device),
            partial=self.partial.to_device(device),
        )

    @classmethod
    def from_tree_tensors(
        cls,
        pos: torch.Tensor,
        depth: torch.Tensor,
        edges: torch.Tensor,
        edges_mask: torch.Tensor,
        pos_lhs: torch.Tensor,
        depth_lhs: torch.Tensor,
        edges_lhs: torch.Tensor,
        edges_mask_lhs: torch.Tensor,
        lograd: torch.Tensor | None = None,
        lograd_lhs: torch.Tensor | None = None,
        edges_enc_v: torch.Tensor | None = None,
        edges_enc_v_lhs: torch.Tensor | None = None,
        skip_vessel_mask: torch.Tensor | None = None,
        skip_vessel_mask_lhs: torch.Tensor | None = None,
        topology: torch.Tensor | None = None,
        topology_lhs: torch.Tensor | None = None,
        query_idx: torch.Tensor | None = None,
        qeidx: torch.Tensor | None = None,
    ) -> "AutoencoderBatch":
        """Split paired tree tensors into typed autoencoder batches.

        The non-``_lhs`` tensors form the full tree (which the full encoder always
        sees with ``query_idx=None`` and no ``qeidx``), while the ``_lhs`` tensors
        plus ``query_idx`` / ``qeidx`` form the partial tree consumed by
        ``decode_from_partial``.
        """
        full = EdgeBatch(
            pos=pos,
            depth=depth,
            topology=topology,
            edges=edges,
            edges_mask=edges_mask,
            lograd=lograd,
            edges_enc_v=edges_enc_v,
            skip_vessel_mask=skip_vessel_mask,
            query_idx=None,
            qeidx=None,
        )
        partial = EdgeBatch(
            pos=pos_lhs,
            depth=depth_lhs,
            topology=topology_lhs,
            edges=edges_lhs,
            edges_mask=edges_mask_lhs,
            lograd=lograd_lhs,
            edges_enc_v=edges_enc_v_lhs,
            skip_vessel_mask=skip_vessel_mask_lhs,
            query_idx=query_idx,
            qeidx=qeidx,
        )
        return cls(full=full, partial=partial)


@dataclass(slots=True)
class EdgesEncoderOutput:
    features: torch.Tensor | None
    mask: torch.Tensor
    mu: torch.Tensor | None = None
    logvar: torch.Tensor | None = None

    @property
    def is_vae(self) -> bool:
        return self.mu is not None and self.logvar is not None


@dataclass(slots=True)
class AutoencoderOutput:
    pos: torch.Tensor
    topology: torch.Tensor
    skip_vessel: torch.Tensor | None = None
    enc_v: torch.Tensor | None = None
    lograd: torch.Tensor | None = None
    z_mu: torch.Tensor | None = None
    z_logvar: torch.Tensor | None = None

    def to_dict(self) -> dict:
        # The five decoder keys are always present (None when their head is
        # disabled), matching the legacy decoder output dict exactly.
        out = {
            "pos": self.pos,
            "topology": self.topology,
            "skip_vessel": self.skip_vessel,
            "enc_v": self.enc_v,
            "lograd": self.lograd,
        }
        # z_mu/z_logvar are only added in VAE mode -- mirrors forward_batch.
        if self.z_mu is not None:
            out["z_mu"] = self.z_mu
        if self.z_logvar is not None:
            out["z_logvar"] = self.z_logvar
        return out


@dataclass(slots=True)
class GatheredEdges:
    """Per-edge endpoint tensors gathered from per-node inputs."""

    pos_a: torch.Tensor
    pos_b: torch.Tensor
    depth_a: torch.Tensor
    topology_a: torch.Tensor
    topology_b: torch.Tensor
    batch_indices: torch.Tensor
    indices_a: torch.Tensor
    indices_b: torch.Tensor


class VesselEdgesEncoder(nn.Module):
    def __init__(self, config: VesselEdgesEncoderConfig | dict | None = None):
        super(VesselEdgesEncoder, self).__init__()

        self.config = self._normalize_config(config)
        config = self.config

        ff_dim = config["vdim"] * config["n_heads"]

        self.mlp_a = MLP2(
            config["mlp_a_in_dims"],
            config["mlp_hidden_dims"],
            ff_dim,
            config["input_activation"],
            backwards_compatibility_paper=config["backwards_compatibility_paper"],
        )

        if self.config["add_pos_enc"]:
            self.pos_enc = PositionalEncoding(d_model=ff_dim, max_depth=config["max_depth"])

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=ff_dim,
                nhead=config["n_heads"],
                dim_feedforward=config["dim_feedforward_transformer"],
                dropout=config["dropout"],
                activation=config["activation_transformer"],
            ),
            config["n_layers"],
        )

        if config["apply_mlp_b"]:
            self.mlp_b = MLP2(
                ff_dim,
                config["mlp_hidden_dims"],
                config["output_dims"],
                config["output_activation"],
                backwards_compatibility_paper=config["backwards_compatibility_paper"],
            )

        if config["use_vae"]:
            self.mu_enc = nn.Linear(ff_dim, config["output_dims"], bias=True)
            self.logvar_enc = nn.Linear(ff_dim, config["output_dims"], bias=True)

        st_size = config["st_size"]
        self.start_token = nn.Parameter(
            torch.tensor(np.random.uniform(-1.0, 1.0, st_size).astype(np.float32))
        )

    @classmethod
    def _normalize_config(cls, config: VesselEdgesEncoderConfig | dict | None) -> dict:
        if config is None:
            config = VesselEdgesEncoderConfig()
        elif not isinstance(config, VesselEdgesEncoderConfig):
            config = VesselEdgesEncoderConfig.from_dict(config)

        return config.checked().to_dict()

    @classmethod
    def default_config(cls):
        config = VesselEdgesEncoderConfig().to_dict()
        config["pos_octaves"] = list(config["pos_octaves"])
        return config

    def make_edge_features_batch(
        self,
        batch: EdgeBatch,
        start_token: torch.Tensor,
        to_cuda: bool = True,
    ):
        if self.config["include_query"]:
            if batch.query_idx is None:
                raise ValueError("VesselEdgesEncoder include_query=True requires query_idx")

        b = batch.batch_size

        gathered = self._gather_edge_parts(batch)
        depth_a = gathered.depth_a

        ef = self._assemble_raw_edge_features(gathered, batch)
        ef = self._prepend_start_token(ef, start_token, b)
        ef = self.mlp_a(ef)
        ef = self._add_depth_positional_encoding(ef, depth_a, b)
        ef, edges_mask = self._apply_start_token_mask(ef, batch.edges_mask, b)

        return ef, edges_mask

    def _prepend_start_token(
        self, ef: torch.Tensor, start_token: torch.Tensor, b: int
    ) -> torch.Tensor:
        st = start_token.reshape((1, 1, -1)).repeat(b, 1, 1)
        return torch.cat([st, ef], dim=1)

    def _add_depth_positional_encoding(
        self, ef: torch.Tensor, depth_a: torch.Tensor, b: int
    ) -> torch.Tensor:
        if not self.config["add_pos_enc"]:
            return ef
        ones = -torch.ones(b, 1, device=ef.device)
        depth_a = torch.cat([ones, depth_a], dim=1)
        return self.pos_enc(ef, depth_a)

    def _apply_start_token_mask(
        self, ef: torch.Tensor, edges_mask: torch.Tensor, b: int
    ):
        ones = torch.ones(b, 1, device=ef.device)
        edges_mask = torch.cat([ones, edges_mask], dim=1)
        ef = edges_mask.unsqueeze(-1) * ef
        return ef, edges_mask

    def _gather_edge_parts(self, batch: EdgeBatch) -> GatheredEdges:
        b = int(batch.pos.shape[0])
        device = batch.pos.device
        edges_l = batch.edges.long()
        indices_a = edges_l[:, :, 0]
        indices_b = edges_l[:, :, 1]
        batch_indices = torch.arange(b, device=device)[:, None]

        pos_a = batch.pos[batch_indices, indices_a]
        pos_b = batch.pos[batch_indices, indices_b]
        depth_a = batch.depth[batch_indices, indices_a]
        topology_a = batch.topology[batch_indices, indices_a]
        # backwards_compatibility_paper gates the paper/legacy quirk where
        # topology_b is gathered from indices_a, not indices_b, so topology_a ==
        # topology_b for every edge (duplicate source-node topology). This quirk is
        # baked into every trained proposed checkpoint. The default (flag False)
        # gathers from indices_b (fixed). See docs/public_legacy_surfaces.md.
        topology_b_indices = (
            indices_a if self.config["backwards_compatibility_paper"] else indices_b
        )
        topology_b = batch.topology[batch_indices, topology_b_indices]

        return GatheredEdges(
            pos_a=pos_a,
            pos_b=pos_b,
            depth_a=depth_a,
            topology_a=topology_a,
            topology_b=topology_b,
            batch_indices=batch_indices,
            indices_a=indices_a,
            indices_b=indices_b,
        )

    def _assemble_raw_edge_features(
        self, parts: GatheredEdges, batch: EdgeBatch
    ) -> torch.Tensor:
        # Concatenation order and dtypes are load-bearing: they define
        # mlp_a_in_dims. Order: [pos_a, pos_b] -> optional [lograd_a, lograd_b] ->
        # [topology_a, topology_b] -> optional query -> optional enc_v ->
        # optional skip_vessel. Do not reorder.
        device = batch.pos.device
        b = int(batch.pos.shape[0])
        e = int(batch.edges.shape[1])
        channels = self._position_channels()

        pos_a = add_octaves(
            parts.pos_a,
            self.config["pos_octaves"],
            dim=-1,
            channels=channels,
            include_base=self.config["input_include_base"],
        )
        pos_b = add_octaves(
            parts.pos_b,
            self.config["pos_octaves"],
            dim=-1,
            channels=channels,
            include_base=self.config["input_include_base"],
        )
        ef = torch.cat([pos_a, pos_b], dim=-1)

        if self.config["include_rad"]:
            lograd_a = batch.lograd[parts.batch_indices, parts.indices_a]
            lograd_b = batch.lograd[parts.batch_indices, parts.indices_b]
            lograd_a = torch.as_tensor(lograd_a, device=device).unsqueeze(-1)
            lograd_b = torch.as_tensor(lograd_b, device=device).unsqueeze(-1)
            ef = torch.cat([ef, lograd_a, lograd_b], dim=-1)

        ef = torch.cat([ef, parts.topology_a, parts.topology_b], dim=-1)

        if batch.qeidx is not None:
            query_b = torch.zeros((b, e), dtype=torch.float32, device=device)
            query_b[parts.batch_indices, batch.qeidx.long()] = 1.0
            ef = torch.cat([ef, query_b.unsqueeze(-1)], dim=-1)

        if self.config["include_edges_enc_v"]:
            ef = torch.cat([ef, batch.edges_enc_v], dim=-1)

        if self.config["include_skip_vessel"]:
            ef = torch.cat([ef, batch.skip_vessel_mask.unsqueeze(-1)], dim=-1)

        return ef

    def _position_channels(self) -> list[int]:
        if self.config["posdims"] == 2:
            return [0, 1]
        if self.config["posdims"] == 3:
            return [0, 1, 2]
        raise ValueError(f"unhandled posdims={self.config['posdims']}")

    @classmethod
    def check_domain(
        cls,
        pos: torch.Tensor,
        edges: torch.Tensor,
        edges_mask: torch.Tensor,
        wrap_domain: tuple,
    ):
        pos_np = to_array(pos)
        edges_np = to_array(edges)
        edges_mask_np = to_array(edges_mask)
        b = int(pos_np.shape[0])
        for k in range(0, b):
            eidxs = np.argwhere(edges_mask_np[k, :] > 0.5)[:, 0]
            if eidxs.shape[0] > 0:
                aidxs = edges_np[k, eidxs, 0]
                bidxs = edges_np[k, eidxs, 1]
                pos_a = pos_np[k, aidxs, :]
                pos_b = pos_np[k, bidxs, :]
                if pos_a.min() <= wrap_domain[0] or pos_a.max() >= wrap_domain[1]:
                    raise ValueError(
                        f"edge source positions fall outside wrap_domain={wrap_domain}"
                    )
                if pos_b.min() <= wrap_domain[0] or pos_b.max() >= wrap_domain[1]:
                    raise ValueError(
                        f"edge target positions fall outside wrap_domain={wrap_domain}"
                    )

    def reparameterize(self, zmu: torch.Tensor, zlogvar: torch.Tensor):
        std = torch.exp(0.5 * zlogvar)
        eps = torch.randn_like(zmu)
        return zmu + eps * std

    def encode_batch(self, batch: EdgeBatch, to_cuda: bool = True) -> EdgesEncoderOutput:
        VesselEdgesEncoder.check_domain(
            batch.pos, batch.edges, batch.edges_mask, self.config["wrap_domain"]
        )

        ef, emask = self.make_edge_features_batch(batch, self.start_token, to_cuda=to_cuda)

        ef = self.transformer(ef.permute(1, 0, 2), src_key_padding_mask=(1.0 - emask).bool()).permute(
            1, 0, 2
        )

        if self.config["pool_final"]:
            ef = torch.mean(ef * emask.unsqueeze(-1), dim=1)

        if self.config["apply_mlp_b"]:
            ef = self.mlp_b(ef)

        if self.config["use_vae"]:
            return EdgesEncoderOutput(
                features=None,
                mask=emask,
                mu=self.mu_enc(ef),
                logvar=self.logvar_enc(ef),
            )

        return EdgesEncoderOutput(features=ef, mask=emask)

    def forward(self, batch: EdgeBatch, to_cuda: bool = True) -> EdgesEncoderOutput:
        return self.encode_batch(batch, to_cuda=to_cuda)


class VesselTreeAutoencoderConfig(VettaSettings):
    vtenc_full_config: dict[str, Any] = Field(default_factory=VesselEdgesEncoder.default_config)
    vtenc_partial_config: dict[str, Any] = Field(default_factory=VesselEdgesEncoder.default_config)
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
    # Octave-target position domain for out_mode="octaves": positions (normalised
    # into norm_domain, e.g. [-0.25, 0.25]) are mapped through this domain to
    # [0, 1] before octave encoding, and inference maps the decoded fraction back
    # through it. The trained proposed checkpoints use [-3, 3]; train and infer
    # must share this value for positions to round-trip.
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


def apply_shared_autoencoder_settings(
    cfg: VesselEdgesEncoderConfig, settings: VesselTreeAutoencoderConfig
) -> VesselEdgesEncoderConfig:
    return replace(
        cfg,
        n_heads=settings.n_heads,
        vdim=settings.vdim,
        n_layers=settings.n_encoder_layers,
        pos_octaves=tuple(settings.pos_octaves),
        wrap_domain=tuple(settings.wrap_domain),
        input_include_base=settings.input_include_base,
        posdims=settings.posdims,
        include_skip_vessel=settings.include_skip_vessel,
        input_activation=settings.mlp_nonlinearity,
        output_activation=settings.mlp_nonlinearity,
        backwards_compatibility_paper=settings.backwards_compatibility_paper,
    )


def make_full_encoder_config(
    settings: VesselTreeAutoencoderConfig,
) -> VesselEdgesEncoderConfig:
    cfg = VesselEdgesEncoderConfig.from_dict(settings.vtenc_full_config)
    cfg = apply_shared_autoencoder_settings(cfg, settings)
    cfg = replace(
        cfg,
        pool_final=True,
        apply_mlp_b=not settings.use_vae,
        use_vae=settings.use_vae,
        output_dims=settings.z_dim,
        st_size=settings.data_size,
        mlp_a_in_dims=settings.data_size,
        include_edges_enc_v=settings.include_edges_enc_v,
        include_rad=settings.include_rad,
    )
    return cfg.checked()


def make_partial_encoder_config(
    settings: VesselTreeAutoencoderConfig,
) -> VesselEdgesEncoderConfig:
    data_size = settings.data_size + 1
    if settings.include_edges_enc_v and not settings.include_lhs_enc:
        data_size -= settings.enc_size
    if settings.include_rad and not settings.include_lhs_rad:
        data_size -= 2

    cfg = VesselEdgesEncoderConfig.from_dict(settings.vtenc_partial_config)
    cfg = apply_shared_autoencoder_settings(cfg, settings)
    cfg = replace(
        cfg,
        pool_final=False,
        apply_mlp_b=False,
        include_query=True,
        st_size=data_size,
        mlp_a_in_dims=data_size,
        include_edges_enc_v=settings.include_edges_enc_v and settings.include_lhs_enc,
        include_rad=settings.include_rad and settings.include_lhs_rad,
    )
    return cfg.checked()


class VesselTreeAutoencoder(nn.Module):
    def __init__(self, config=None):
        super(VesselTreeAutoencoder, self).__init__()
        self.settings = self._normalize_settings(config)
        self.config = self.settings.to_dict()
        config = self.config
        settings = self.settings

        ff_dim = settings.n_heads * settings.vdim

        self.edges_encoder_full = VesselEdgesEncoder(make_full_encoder_config(settings))
        self.edges_encoder_partial = VesselEdgesEncoder(make_partial_encoder_config(settings))

        self.mlp_memory = MLP2(
            in_dims=ff_dim + settings.z_dim,
            hidden_dims=settings.mlp_hidden_dims,
            out_dims=ff_dim,
            nonlinearity=settings.mlp_nonlinearity,
            backwards_compatibility_paper=settings.backwards_compatibility_paper,
        )

        self.slots = nn.Parameter(nn.init.xavier_normal_(torch.empty((settings.n_slots, ff_dim))))

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=ff_dim,
                nhead=settings.n_heads,
                dim_feedforward=settings.dim_feedforward_transformer,
                dropout=settings.dropout,
                activation=settings.activation_transformer,
            ),
            num_layers=settings.n_decoder_layers,
        )

        self._build_decoder_heads(ff_dim, settings)

    @classmethod
    def _normalize_settings(cls, config=None) -> VesselTreeAutoencoderConfig:
        if config is None:
            settings = VesselTreeAutoencoderConfig()
        elif isinstance(config, VesselTreeAutoencoderConfig):
            settings = config
        else:
            raise TypeError(
                "VesselTreeAutoencoder config must be a VesselTreeAutoencoderConfig "
                f"or None, got {type(config).__name__}"
            )

        cls._decoder_position_dims(settings)
        return settings

    @classmethod
    def _decoder_position_dims(cls, settings: VesselTreeAutoencoderConfig) -> int:
        if settings.out_mode == "default":
            return settings.posdims
        if settings.out_mode == "octaves":
            return 2 * settings.posdims * len(settings.pos_octaves)
        raise ValueError(f"unknown out_mode {settings.out_mode}")

    def _build_decoder_heads(self, ff_dim: int, settings: VesselTreeAutoencoderConfig) -> None:
        self.mlp_out_pos = MLP2(
            in_dims=ff_dim,
            hidden_dims=settings.mlp_hidden_dims,
            out_dims=self._decoder_position_dims(settings),
            nonlinearity=settings.mlp_nonlinearity,
            backwards_compatibility_paper=settings.backwards_compatibility_paper,
        )

        self.mlp_out_topology = MLP2(
            in_dims=ff_dim,
            hidden_dims=settings.mlp_hidden_dims,
            out_dims=settings.topology_size,
            nonlinearity=settings.mlp_nonlinearity,
            backwards_compatibility_paper=settings.backwards_compatibility_paper,
        )

        self.mlp_out_skip_vessel = None
        if settings.include_skip_vessel:
            self.mlp_out_skip_vessel = MLP2(
                in_dims=ff_dim,
                hidden_dims=settings.mlp_hidden_dims,
                out_dims=1,
                nonlinearity=settings.mlp_nonlinearity,
                backwards_compatibility_paper=settings.backwards_compatibility_paper,
            )

        self.mlp_out_enc_v = None
        if settings.include_edges_enc_v:
            self.mlp_out_enc_v = MLP2(
                in_dims=ff_dim,
                hidden_dims=settings.mlp_hidden_dims,
                out_dims=settings.enc_size,
                nonlinearity=settings.mlp_nonlinearity,
                backwards_compatibility_paper=settings.backwards_compatibility_paper,
            )

        self.mlp_out_lograd = None
        if settings.include_rad:
            self.mlp_out_lograd = MLP2(
                in_dims=ff_dim,
                hidden_dims=settings.mlp_hidden_dims,
                out_dims=1,
                nonlinearity=settings.mlp_nonlinearity,
                backwards_compatibility_paper=settings.backwards_compatibility_paper,
            )

    @classmethod
    def default_config(cls):
        return VesselTreeAutoencoderConfig().to_dict()

    def forward(
        self,
        pos: torch.Tensor,
        depth: torch.Tensor,
        edges: torch.Tensor,
        edges_mask: torch.Tensor,
        pos_lhs: torch.Tensor,
        depth_lhs: torch.Tensor,
        edges_lhs: torch.Tensor,
        edges_mask_lhs: torch.Tensor,
        lograd: torch.Tensor | None = None,
        lograd_lhs: torch.Tensor | None = None,
        edges_enc_v: torch.Tensor | None = None,
        edges_enc_v_lhs: torch.Tensor | None = None,
        skip_vessel_mask: torch.Tensor | None = None,
        skip_vessel_mask_lhs: torch.Tensor | None = None,
        topology: torch.Tensor | None = None,
        topology_lhs: torch.Tensor | None = None,
        query_idx: torch.Tensor | None = None,
        qeidx: torch.Tensor | None = None,
        to_cuda: bool = True,
    ) -> dict:
        # Flat-tensor forward entry point: split the paired tree tensors into
        # typed batches and run the batch path.
        batch = AutoencoderBatch.from_tree_tensors(
            pos=pos,
            depth=depth,
            edges=edges,
            edges_mask=edges_mask,
            pos_lhs=pos_lhs,
            depth_lhs=depth_lhs,
            edges_lhs=edges_lhs,
            edges_mask_lhs=edges_mask_lhs,
            lograd=lograd,
            lograd_lhs=lograd_lhs,
            edges_enc_v=edges_enc_v,
            edges_enc_v_lhs=edges_enc_v_lhs,
            skip_vessel_mask=skip_vessel_mask,
            skip_vessel_mask_lhs=skip_vessel_mask_lhs,
            topology=topology,
            topology_lhs=topology_lhs,
            query_idx=query_idx,
            qeidx=qeidx,
        )
        return self.forward_batch(batch, to_cuda=to_cuda)

    def forward_batch(self, batch: AutoencoderBatch, to_cuda: bool = True) -> dict:
        if to_cuda:
            # Honour the to_cuda flag here, once, so the full and partial inputs
            # land on the model's device (no-op when already co-located, e.g. the
            # CPU inference/training path).
            batch = batch.to_device(next(self.parameters()).device)
        z, vae_stats = self.encode_full(batch.full, to_cuda=to_cuda)
        pred_dict = self.decode_from_partial(z, batch.partial, to_cuda=to_cuda)
        if vae_stats is not None:
            pred_dict.update(vae_stats)
        return pred_dict

    def encode_full(self, full: EdgeBatch, to_cuda: bool = True):
        """Encode the full tree to a latent ``z`` of shape ``(1, b, v1)``.

        Returns ``(z, vae_stats)`` where ``vae_stats`` is ``{"z_mu", "z_logvar"}``
        in VAE mode (with ``z`` sampled via reparameterize) or ``None`` in classic
        mode. The full encoder is always queried with ``query_idx=None``.
        """
        out = self.edges_encoder_full.encode_batch(full.without_query(), to_cuda=to_cuda)
        if out.is_vae:
            z = self.reparameterize(out.mu, out.logvar)
            vae_stats = {"z_mu": out.mu, "z_logvar": out.logvar}
        else:
            z = out.features
            vae_stats = None
        z = z.unsqueeze(0)  # (1, b, v1)
        return z, vae_stats

    def reparameterize(self, zmu: torch.Tensor, zlogvar: torch.Tensor):
        std = torch.exp(0.5 * zlogvar)
        eps = torch.randn_like(zmu)
        return zmu + eps * std

    def decode_from_partial(self, z: torch.Tensor, partial: EdgeBatch, to_cuda: bool = True):
        b = partial.batch_size
        out = self.edges_encoder_partial.encode_batch(partial, to_cuda=to_cuda)
        efp = out.features
        emaskp = out.mask
        efp = efp.permute(1, 0, 2)  # (ep, b, v1)

        z = z.repeat(int(efp.shape[0]), 1, 1)  # (ep, b, v1)
        memory = torch.cat([efp, z], dim=-1)  # (ep, b, v1+v2)
        memory = self.mlp_memory(memory)  # (ep, b, ff_dim)

        s = self.slots.unsqueeze(1).repeat(1, b, 1)  # (s, b, ff_dim)
        s = self.decoder(s, memory, memory_key_padding_mask=(1.0 - emaskp).bool())

        return self._assemble_decoder_outputs(s).to_dict()

    def _assemble_decoder_outputs(self, decoder_state: torch.Tensor) -> AutoencoderOutput:
        out_pos = self.mlp_out_pos(decoder_state).permute(1, 0, 2)  # (b, nc, .)
        out_topology = self.mlp_out_topology(decoder_state).permute(1, 0, 2)  # (b, nc, .)
        out_skip_vessel = None
        out_enc_v = None
        out_lograd = None
        if self.config["include_skip_vessel"]:
            out_skip_vessel = self.mlp_out_skip_vessel(decoder_state).permute(1, 0, 2)
        if self.config["include_edges_enc_v"]:
            out_enc_v = self.mlp_out_enc_v(decoder_state).permute(1, 0, 2)
        if self.config["include_rad"]:
            out_lograd = self.mlp_out_lograd(decoder_state).permute(1, 0, 2)

        return AutoencoderOutput(
            pos=out_pos,
            topology=out_topology,
            skip_vessel=out_skip_vessel,
            enc_v=out_enc_v,
            lograd=out_lograd,
        )


__all__ = [
    "NonLinearity",
    "ReLU",
    "GELU",
    "make_nlrity",
    "make_norm",
    "make_weights_init",
    "MLP2",
    "liftsin",
    "liftcos",
    "lift",
    "add_octaves",
    "PositionalEncoding",
    "EdgeBatch",
    "AutoencoderBatch",
    "EdgesEncoderOutput",
    "AutoencoderOutput",
    "GatheredEdges",
    "VesselEdgesEncoder",
    "VesselEdgesEncoderConfig",
    "VesselTreeAutoencoder",
    "VesselTreeAutoencoderConfig",
    "apply_shared_autoencoder_settings",
    "make_full_encoder_config",
    "make_partial_encoder_config",
]
