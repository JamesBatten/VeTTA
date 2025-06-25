# --- FILE: vetta/model/vetta.py ---
"""
Implements the Vessel Tree Transformer Autoencoder (VETTA), a model for learning latent
representations of vessel tree structures and performing conditional generation.
"""
import torch
import torch.nn as nn
import copy
from typing import Dict, Any, Optional, Tuple

# Corrected import paths to reflect the project structure
from vetta.model.vessel_edges_encoder import VesselEdgesEncoder
from vetta.model.mlp2 import MLP2


class Vetta(nn.Module):
    """
    Implements a Vessel Tree Transformer Autoencoder (VETTA).

    This model learns a latent representation of a vessel tree structure. It consists
    of two main parts:
    1. An Encoder: Processes a full vessel tree graph, represented by node
       positions, depths, topology, and edges, into a fixed-size latent
       vector `z`. It supports both a standard autoencoder and a Variational
       Autoencoder (VAE) mode.
    2. A Decoder: Takes the latent vector `z` and a partially-observed
       vessel tree as input, and predicts the properties (position, topology, etc.)
       of the missing parts of the tree. It uses a Transformer-based architecture
       to conditionally generate the output.

    The model is highly configurable through a dictionary passed at initialization.
    """

    # --- Initialization and Configuration ---

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initializes the VETTA model components based on the provided configuration."""
        super().__init__()

        self.config = config if config is not None else self.default_config()

        ff_dim = self.config['n_heads'] * self.config['vdim']

        # Decompose initialization into logical, modular methods
        self.edges_encoder_full, self.edges_encoder_partial = self._create_encoders()
        self.mlp_memory, self.slots, self.decoder = self._create_decoder_components(ff_dim)

        (
            self.mlp_out_pos,
            self.mlp_out_topology,
            self.mlp_out_placeholder,
            self.mlp_out_enc_v,
            self.mlp_out_lograd,
        ) = self._create_output_heads(ff_dim)

    def _create_encoders(self) -> Tuple[VesselEdgesEncoder, VesselEdgesEncoder]:
        """Creates the full and partial vessel tree encoders based on the config."""

        # --- Full Encoder Configuration ---
        eef_config = copy.deepcopy(self.config['vtenc_full_config'])
        eef_config.update({
            'pool_final': True,
            'apply_mlp_b': not self.config['use_vae'],
            'use_vae': self.config['use_vae'],
            'output_dims': self.config['z_dim'],
            'n_heads': self.config['n_heads'],
            'vdim': self.config['vdim'],
            'st_size': self.config['data_size'],
            'mlp_a_in_dims': self.config['data_size'],
            'n_layers': self.config['n_encoder_layers'],
            'pos_octaves': self.config['pos_octaves'],
            'wrap_domain': self.config['wrap_domain'],
            'input_include_base': self.config['input_include_base'],
            'posdims': self.config['posdims'],
            'include_edges_enc_v': self.config['include_edges_enc_v'],
            'include_placeholder': self.config['include_placeholder'],
            'include_rad': self.config['include_rad'],
            'input_activation': self.config['mlp_nonlinearity'],
            'output_activation': self.config['mlp_nonlinearity'],
        })
        full_encoder = VesselEdgesEncoder(eef_config)

        # --- Partial Encoder Configuration ---
        eep_config = copy.deepcopy(self.config['vtenc_partial_config'])

        # The input size for the partial encoder depends on what features from the
        # Left-Hand-Side (LHS) context are included.
        dsize = self.config['data_size'] + 1  # Base size + query flag
        if self.config['include_edges_enc_v'] and not self.config['include_lhs_enc']:
            dsize -= self.config['enc_size']
        if self.config['include_rad'] and not self.config['include_lhs_rad']:
            dsize -= 2  # Radius from node 'a' and 'b'

        # Determine which features to pass to the partial encoder based on config
        include_edges_enc_v = self.config['include_edges_enc_v'] and self.config['include_lhs_enc']
        include_rad = self.config['include_rad'] and self.config['include_lhs_rad']

        eep_config.update({
            'pool_final': False,
            'apply_mlp_b': False,
            'n_heads': self.config['n_heads'],
            'vdim': self.config['vdim'],
            'st_size': dsize,
            'mlp_a_in_dims': dsize,
            'n_layers': self.config['n_encoder_layers'],
            'pos_octaves': self.config['pos_octaves'],
            'wrap_domain': self.config['wrap_domain'],
            'include_query': True,
            'input_include_base': self.config['input_include_base'],
            'posdims': self.config['posdims'],
            'include_edges_enc_v': include_edges_enc_v,
            'include_placeholder': self.config['include_placeholder'],
            'include_rad': include_rad,
            'input_activation': self.config['mlp_nonlinearity'],
            'output_activation': self.config['mlp_nonlinearity'],
        })
        partial_encoder = VesselEdgesEncoder(eep_config)

        return full_encoder, partial_encoder

    def _create_decoder_components(self, ff_dim: int) -> Tuple[MLP2, nn.Parameter, nn.TransformerDecoder]:
        """Creates the core components of the decoder branch."""

        # MLP to process the concatenated partial encoding and global latent vector
        mlp_memory = MLP2(
            in_dims=ff_dim + self.config['z_dim'],
            hidden_dims=self.config['mlp_hidden_dims'],
            out_dims=ff_dim,
            nonlinearity=self.config['mlp_nonlinearity']
        )

        # Learnable "slots" that act as queries for the decoder
        slots = nn.Parameter(torch.empty((self.config['n_slots'], ff_dim)))
        nn.init.xavier_normal_(slots)

        # Standard Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=ff_dim,
            nhead=self.config['n_heads'],
            dim_feedforward=self.config['dim_feedforward_transformer'],
            dropout=self.config['dropout'],
            activation=self.config['activation_transformer']
        )
        decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=self.config['n_decoder_layers']
        )

        return mlp_memory, slots, decoder

    def _create_output_heads(self, ff_dim: int) -> Tuple[nn.Module, ...]:
        """Creates the final MLP heads for predicting different output properties."""

        def make_head(out_dims: int) -> MLP2:
            """Factory for creating a standard output MLP head."""
            return MLP2(
                in_dims=ff_dim,
                hidden_dims=self.config['mlp_hidden_dims'],
                out_dims=out_dims,
                nonlinearity=self.config['mlp_nonlinearity']
            )

        # --- Position Head ---
        out_mode = self.config['out_mode']
        if out_mode == 'default':
            pos_out_dims = self.config['posdims']
        elif out_mode == 'octaves':
            pos_out_dims = 2 * self.config['posdims'] * len(self.config['pos_octaves'])
        else:
            raise ValueError(f"Unknown out_mode '{out_mode}'. Must be 'default' or 'octaves'.")
        mlp_out_pos = make_head(pos_out_dims)

        # --- Topology Head ---
        mlp_out_topology = make_head(self.config['topology_size'])

        # --- Optional Heads ---
        mlp_out_placeholder = make_head(1) if self.config['include_placeholder'] else None
        mlp_out_enc_v = make_head(self.config['enc_size']) if self.config['include_edges_enc_v'] else None
        mlp_out_lograd = make_head(1) if self.config['include_rad'] else None

        return mlp_out_pos, mlp_out_topology, mlp_out_placeholder, mlp_out_enc_v, mlp_out_lograd

    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        """Provides a default configuration dictionary for the model."""
        return {
            'vtenc_full_config': VesselEdgesEncoder.default_config(),
            'vtenc_partial_config': VesselEdgesEncoder.default_config(),
            'n_slots': 32,
            'z_dim': 64,
            'data_size': 70,
            'enc_size': 64,
            'mlp_hidden_dims': 2048,
            'n_heads': 12,
            'vdim': 64,
            'mlp_nonlinearity': 'gelu',
            'dropout': 0.0,
            'n_encoder_layers': 6,
            'n_decoder_layers': 6,
            'dim_feedforward_transformer': 2048,
            'activation_transformer': 'relu',
            'pos_octaves': [1, 2, 4, 8, 16, 32],
            'wrap_domain': (-0.5, 0.5),
            'posdims': 2,
            'topology_size': 3,
            'out_mode': 'octaves',  # 'default' or 'octaves'
            'input_include_base': False,
            'include_placeholder': False,
            'include_edges_enc_v': False,
            'include_rad': False,
            'include_lhs_enc': False,  # Whether to include `edges_enc_v` in the partial encoder input
            'include_lhs_rad': False,  # Whether to include `lograd` in the partial encoder input
            'use_vae': False
        }

    # --- Core Logic ---

    @staticmethod
    def reparameterize(z_mu: torch.Tensor, z_logvar: torch.Tensor) -> torch.Tensor:
        """
        Applies the reparameterization trick for VAEs.

        Args:
            z_mu: The predicted mean of the latent distribution.
            z_logvar: The predicted log-variance of the latent distribution.

        Returns:
            A sample from the latent distribution.
        """
        std = torch.exp(0.5 * z_logvar)
        epsilon = torch.randn_like(z_mu)
        return z_mu + epsilon * std

    def forward(self,
                pos: torch.Tensor,
                depth: torch.Tensor,
                edges: torch.Tensor,
                edges_mask: torch.Tensor,
                pos_lhs: torch.Tensor,
                depth_lhs: torch.Tensor,
                edges_lhs: torch.Tensor,
                edges_mask_lhs: torch.Tensor,
                lograd: Optional[torch.Tensor] = None,
                lograd_lhs: Optional[torch.Tensor] = None,
                edges_enc_v: Optional[torch.Tensor] = None,
                edges_enc_v_lhs: Optional[torch.Tensor] = None,
                placeholder_mask: Optional[torch.Tensor] = None,
                placeholder_mask_lhs: Optional[torch.Tensor] = None,
                topology: Optional[torch.Tensor] = None,
                topology_lhs: Optional[torch.Tensor] = None,
                query_idx: Optional[torch.Tensor] = None,
                qeidx: Optional[torch.Tensor] = None
                ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Performs the full autoencoding process.

        Note on device placement: This model expects input tensors to be on the
        same device as the model parameters. The `to_cuda` argument has been
        removed to follow standard PyTorch practice. Move data to the correct
        device before calling `forward` (e.g., `data.to(device)`).

        Args:
            pos: Full graph node positions.
            depth: Full graph node depths.
            edges: Full graph edge definitions.
            edges_mask: Full graph active edge mask.
            pos_lhs: Partial (Left-Hand Side) graph node positions.
            depth_lhs: Partial graph node depths.
            edges_lhs: Partial graph edge definitions.
            edges_mask_lhs: Partial graph active edge mask.
            lograd: Full graph node log-radius.
            lograd_lhs: Partial graph node log-radius.
            edges_enc_v: Full graph pre-computed edge features.
            edges_enc_v_lhs: Partial graph pre-computed edge features.
            placeholder_mask: Full graph placeholder mask.
            placeholder_mask_lhs: Partial graph placeholder mask.
            topology: Full graph node topology features.
            topology_lhs: Partial graph node topology features.
            query_idx: Query node indices for the partial graph.
            qeidx: Query edge indices for the partial graph.

        Returns:
            A dictionary containing the predicted output tensors ('pos', 'topology',
            etc.) and, if in VAE mode, the latent distribution parameters
            ('z_mu', 'z_logvar').
        """
        # 1. Encode the full vessel graph into a latent vector 'z'.
        if self.config['use_vae']:
            z_mu, z_logvar, _ = self.edges_encoder_full(
                pos=pos, lograd=lograd, depth=depth, edges=edges,
                edges_mask=edges_mask, topology=topology,
                placeholder_mask=placeholder_mask, edges_enc_v=edges_enc_v
            )
            z = self.reparameterize(z_mu, z_logvar)
        else:
            z, _ = self.edges_encoder_full(
                pos=pos, lograd=lograd, depth=depth, edges=edges,
                edges_mask=edges_mask, topology=topology,
                placeholder_mask=placeholder_mask, edges_enc_v=edges_enc_v
            )

        # Add a singleton dimension to 'z' for compatibility with the decoder,
        # treating it as a sequence of length 1. Shape: (1, B, z_dim)
        z = z.unsqueeze(0)

        # 2. Decode 'z' conditioned on the partial (LHS) graph to predict outputs.
        pred_dict = self.decoder_branch(
            z=z, pos_lhs=pos_lhs, lograd_lhs=lograd_lhs, depth_lhs=depth_lhs,
            edges_lhs=edges_lhs, edges_mask_lhs=edges_mask_lhs,
            topology_lhs=topology_lhs, query_idx=query_idx, qeidx=qeidx,
            placeholder_mask_lhs=placeholder_mask_lhs,
            edges_enc_v_lhs=edges_enc_v_lhs
        )

        # 3. If in VAE mode, include latent variables in the output for loss calculation.
        if self.config['use_vae']:
            pred_dict['z_mu'] = z_mu
            pred_dict['z_logvar'] = z_logvar

        return pred_dict

    def decoder_branch(self,
                       z: torch.Tensor,
                       pos_lhs: torch.Tensor,
                       lograd_lhs: Optional[torch.Tensor],
                       depth_lhs: torch.Tensor,
                       edges_lhs: torch.Tensor,
                       edges_mask_lhs: torch.Tensor,
                       topology_lhs: torch.Tensor,
                       query_idx: torch.Tensor,
                       qeidx: torch.Tensor,
                       placeholder_mask_lhs: Optional[torch.Tensor],
                       edges_enc_v_lhs: Optional[torch.Tensor]
                       ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Runs the decoder part of the model.

        Args:
            z: The latent vector from the encoder, shape (1, B, z_dim).
            ...lhs tensors: The partial graph data used for conditioning.

        Returns:
            A dictionary of predicted output tensors.
        """
        batch_size = pos_lhs.shape[0]

        # 1. Get a contextualized encoding of the partial (LHS) graph.
        # The partial encoder does not pool, so output is per-edge.
        efp, emaskp = self.edges_encoder_partial(
            pos=pos_lhs, lograd=lograd_lhs, depth=depth_lhs,
            edges=edges_lhs, edges_mask=edges_mask_lhs, topology=topology_lhs,
            query_idx=query_idx, qeidx=qeidx, placeholder_mask=placeholder_mask_lhs,
            edges_enc_v=edges_enc_v_lhs
        )
        # Reshape for Transformer: (B, Seq, Feat) -> (Seq, B, Feat)
        efp = efp.permute(1, 0, 2)

        # 2. Create the decoder memory by combining the partial encoding with the global latent vector.
        # Repeat 'z' to match the number of partial edges.
        z_repeated = z.repeat(efp.shape[0], 1, 1)  # (Seq, B, z_dim)
        memory_input = torch.cat([efp, z_repeated], dim=-1)  # (Seq, B, ff_dim + z_dim)
        memory = self.mlp_memory(memory_input)  # (Seq, B, ff_dim)

        # 3. Use the Transformer decoder to refine the learnable query slots.
        # The slots act as the query (tgt), and the memory is the context.
        s = self.slots.unsqueeze(1).repeat(1, batch_size, 1)  # (n_slots, B, ff_dim)

        # The padding mask ensures the decoder doesn't attend to padded edges in the memory.
        padding_mask = (1.0 - emaskp).bool()

        s = self.decoder(
            tgt=s,
            memory=memory,
            memory_key_padding_mask=padding_mask
        )

        # 4. Project the decoder's output slots into the final predictions using output heads.
        # Permute from (Seq, B, Feat) -> (B, Seq, Feat) for conventional output shape.
        out_pos = self.mlp_out_pos(s).permute(1, 0, 2)
        out_topology = self.mlp_out_topology(s).permute(1, 0, 2)

        out_placeholder = self.mlp_out_placeholder(s).permute(1, 0, 2) if self.mlp_out_placeholder else None
        out_enc_v = self.mlp_out_enc_v(s).permute(1, 0, 2) if self.mlp_out_enc_v else None
        out_lograd = self.mlp_out_lograd(s).permute(1, 0, 2) if self.mlp_out_lograd else None

        return {
            'pos': out_pos,
            'topology': out_topology,
            'placeholder': out_placeholder,
            'enc_v': out_enc_v,
            'lograd': out_lograd,
        }