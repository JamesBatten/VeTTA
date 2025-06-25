# --- FILE: vetta/model/vessel_edges_encoder.py ---
"""
Encodes vessel tree structures represented by nodes and edges into a feature representation.

This module contains two main classes:
1. PositionalEncoding: A standard Transformer-style positional encoding module that adds
   information about the depth of a node in the tree.
2. VesselEdgesEncoder: The main encoder module. It processes a batch of vessel trees by:
   a. Assembling features for each edge from the properties of its connected nodes (position,
      radius, topology, etc.).
   b. Applying positional octaves (sinusoidal embeddings) to coordinate data.
   c. Using a TransformerEncoder to process the sequence of edge features, allowing
      the model to learn relationships between all edges in a graph.
   d. Optionally pooling the final features and supporting a VAE-style output with
      mean and log-variance heads.
"""
import torch
import torch.nn as nn
import math
import numpy as np
from typing import Optional, List, Dict, Any, Tuple

from vetta.model.mlp2 import MLP2
from vetta.common.utils import add_octaves, to_array


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding based on node depth.
    From: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, max_depth: int = 100):
        super().__init__()
        position = torch.arange(max_depth).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_depth, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, ef: torch.Tensor, depth_a: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to edge features based on the depth of the 'a' node.

        Args:
            ef (torch.Tensor): Edge features of shape (B, E, D_model).
            depth_a (torch.Tensor): Depth of the 'a' node for each edge, shape (B, E).
                                    Inactive nodes should have depth < 0.
        Returns:
            torch.Tensor: Edge features with added positional encoding, shape (B, E, D_model).
        """
        assert len(ef.shape) == 3
        assert len(depth_a.shape) == 2
        
        mask = (depth_a > -1).long()
        # Replace invalid depths (-1) with 0 for safe indexing
        pe_idxs = (mask * depth_a).long()
        pe_v = self.pe[pe_idxs].squeeze(2)  # (B, E, D_model)
        
        # Only add positional encoding to active nodes
        ef = ef + mask.unsqueeze(-1) * pe_v
        return ef


class VesselEdgesEncoder(nn.Module):
    """
    Encodes a vessel graph by processing its edge features with a Transformer.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super(VesselEdgesEncoder, self).__init__()

        if config is None:
            config = VesselEdgesEncoder.default_config()
        self.config = config

        if config['use_vae']:
            if config['apply_mlp_b']:
                raise ValueError("`apply_mlp_b` must be False when `use_vae` is True.")
            if not config['pool_final']:
                raise ValueError("`pool_final` must be True when `use_vae` is True.")

        ff_dim = config['vdim'] * config['n_heads']

        self.mlp_a = MLP2(
            config['mlp_a_in_dims'], config['mlp_hidden_dims'],
            ff_dim, config['input_activation']
        )

        if self.config['add_pos_enc']:
            self.pos_enc = PositionalEncoding(
                d_model=ff_dim, max_depth=config['max_depth']
            )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=ff_dim, nhead=config['n_heads'],
                dim_feedforward=config['dim_feedforward_transformer'],
                dropout=config['dropout'], activation=config['activation_transformer']
            ), num_layers=config['n_layers']
        )

        self.mlp_b = None
        if config['apply_mlp_b']:
            self.mlp_b = MLP2(
                ff_dim, config['mlp_hidden_dims'],
                config['output_dims'], config['output_activation']
            )

        self.mu_enc = None
        self.logvar_enc = None
        if config['use_vae']:
            self.mu_enc = nn.Linear(ff_dim, config['output_dims'], bias=True)
            self.logvar_enc = nn.Linear(ff_dim, config['output_dims'], bias=True)

        self.start_token = nn.Parameter(torch.empty(config['st_size']))
        nn.init.uniform_(self.start_token, -1.0, 1.0)


    @classmethod
    def default_config(cls) -> Dict[str, Any]:
        return {
            'vdim': 64,
            'n_heads': 8,
            'posdims': 3,
            'mlp_hidden_dims': 2048,
            'activation_transformer': 'relu',
            'dim_feedforward_transformer': 2048,
            'n_layers': 3,
            'dropout': 0.0,
            'mlp_a_in_dims': 50,
            'st_size': 50,
            'pos_octaves': [1, 2, 4, 8, 16, 32],
            'wrap_domain': (-0.5, 0.5),
            'max_depth': 30,
            'apply_mlp_b': False,
            'output_dims': 64,
            'pool_final': False,
            'include_query': False,
            'include_placeholder': False,
            'include_edges_enc_v': False,
            'include_rad': False,
            'input_activation': 'gelu',
            'output_activation': 'gelu',
            'input_include_base': False,
            'add_pos_enc': False,
            'use_vae': False
        }


    def _assemble_edge_features(
        self,
        pos: torch.Tensor,
        depth: torch.Tensor,
        topology: torch.Tensor,
        edges: torch.Tensor,
        lograd: Optional[torch.Tensor] = None,
        qeidx: Optional[torch.Tensor] = None,
        placeholder_mask: Optional[torch.Tensor] = None,
        edges_enc_v: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gathers and concatenates node features to form raw edge features."""
        B, E, _ = edges.shape
        device = pos.device
        edges_l = edges.long()

        # Gather features for nodes 'a' and 'b' of each edge
        indices_a = edges_l[:, :, 0]
        indices_b = edges_l[:, :, 1]
        
        # Efficiently gather features using advanced indexing
        batch_idx = torch.arange(B, device=device)[:, None]
        pos_a = pos[batch_idx, indices_a]
        pos_b = pos[batch_idx, indices_b]
        depth_a = depth[batch_idx, indices_a]
        topology_a = topology[batch_idx, indices_a]
        topology_b = topology[batch_idx, indices_b] # Note: original used indices_a, assumed typo. Corrected to indices_b.

        # Apply positional encoding (octaves) to coordinate data
        channels = list(range(self.config['posdims']))
        pos_a_octaves = add_octaves(
            pos_a, self.config['pos_octaves'], dim=-1, channels=channels,
            include_base=self.config['input_include_base']
        )
        pos_b_octaves = add_octaves(
            pos_b, self.config['pos_octaves'], dim=-1, channels=channels,
            include_base=self.config['input_include_base']
        )

        # Build a list of all feature components to be concatenated
        feature_components = [pos_a_octaves, pos_b_octaves]

        if self.config['include_rad']:
            if lograd is None:
                raise ValueError("`lograd` must be provided when `include_rad` is True.")
            lograd_a = lograd[batch_idx, indices_a].unsqueeze(-1)
            lograd_b = lograd[batch_idx, indices_b].unsqueeze(-1)
            feature_components.extend([lograd_a, lograd_b])
        
        feature_components.extend([topology_a, topology_b])

        if qeidx is not None:
            query_b = torch.zeros((B, E, 1), dtype=torch.float32, device=device)
            query_b[batch_idx, qeidx.long(), :] = 1.0
            feature_components.append(query_b)

        if self.config['include_edges_enc_v']:
            if edges_enc_v is None:
                raise ValueError("`edges_enc_v` must be provided when `include_edges_enc_v` is True.")
            feature_components.append(edges_enc_v)

        if self.config['include_placeholder']:
            if placeholder_mask is None:
                raise ValueError("`placeholder_mask` must be provided when `include_placeholder` is True.")
            feature_components.append(placeholder_mask.unsqueeze(-1))
            
        ef = torch.cat(feature_components, dim=-1)
        
        return ef, depth_a


    @staticmethod
    def check_domain(
        pos: torch.Tensor,
        edges: torch.Tensor,
        edges_mask: torch.Tensor,
        wrap_domain: tuple
    ) -> None:
        """
        Debug utility to check if node positions are within the expected domain.
        Note: This is a non-differentiable operation and should not be used during training.
        """
        pos_np = to_array(pos)
        edges_np = to_array(edges)
        edges_mask_np = to_array(edges_mask)
        for k in range(pos_np.shape[0]):
            active_edge_idxs = np.where(edges_mask_np[k] > 0.5)[0]
            if active_edge_idxs.size > 0:
                node_idxs = edges_np[k, active_edge_idxs].flatten()
                active_pos = pos_np[k, node_idxs]
                if not (active_pos.min() > wrap_domain[0] and active_pos.max() < wrap_domain[1]):
                    raise ValueError(f"Batch {k}: Positions are outside the wrap domain {wrap_domain}.")


    @staticmethod
    def reparameterize(zmu: torch.Tensor, zlogvar: torch.Tensor) -> torch.Tensor:
        """Standard VAE reparameterization trick."""
        std = torch.exp(0.5 * zlogvar)
        eps = torch.randn_like(zmu)
        return zmu + eps * std


    def forward(
        self,
        pos: torch.Tensor,
        depth: torch.Tensor,
        topology: torch.Tensor,
        edges: torch.Tensor,
        edges_mask: torch.Tensor,
        lograd: Optional[torch.Tensor] = None,
        edges_enc_v: Optional[torch.Tensor] = None,
        placeholder_mask: Optional[torch.Tensor] = None,
        query_idx: Optional[torch.Tensor] = None, # Note: Not used directly, but qeidx is.
        qeidx: Optional[torch.Tensor] = None,
        **kwargs # Absorb unused arguments like to_cuda
    ) -> Tuple[torch.Tensor, ...]:
        """
        Main forward pass for the VesselEdgesEncoder.

        Args:
            pos (torch.Tensor): Node positions, shape (B, N, P_dims).
            depth (torch.Tensor): Node depths, shape (B, N).
            topology (torch.Tensor): Node topology features, shape (B, N, T_dims).
            edges (torch.Tensor): Edge definitions (node indices), shape (B, E, 2).
            edges_mask (torch.Tensor): Mask for active edges, shape (B, E).
            ... and other optional feature tensors.

        Returns:
            - If not VAE: (ef, emask), features and mask.
            - If VAE: (emu, elogvar, emask), mean, log-variance, and mask.
        """
        if self.config['include_query'] and query_idx is None:
            raise ValueError("`query_idx` must be provided when `include_query` is True.")

        B = pos.shape[0]
        device = pos.device

        # 1. Assemble raw edge features from node properties
        ef, depth_a = self._assemble_edge_features(
            pos=pos, lograd=lograd, depth=depth, topology=topology, edges=edges,
            edges_enc_v=edges_enc_v, placeholder_mask=placeholder_mask, qeidx=qeidx
        )

        # 2. Prepend the learnable start token
        st = self.start_token.reshape(1, 1, -1).repeat(B, 1, 1)
        ef = torch.cat([st, ef], dim=1)

        # 3. Lift features with the first MLP
        ef = self.mlp_a(ef)

        # 4. (Optional) Add depth-based positional encoding
        if self.config['add_pos_enc']:
            start_token_depth = -torch.ones(B, 1, device=device)
            depth_a_padded = torch.cat([start_token_depth, depth_a], dim=1)
            ef = self.pos_enc(ef, depth_a_padded)

        # 5. Prepare the attention mask, adding a mask for the start token
        emask = torch.cat([torch.ones(B, 1, device=device), edges_mask], dim=1)
        # Apply mask to features (for safety, although padding_mask handles this)
        ef = ef * emask.unsqueeze(-1)
        
        # 6. Pass through the Transformer Encoder
        # PyTorch Transformer expects (Seq, Batch, Feature)
        src_key_padding_mask = (1.0 - emask).bool()
        ef = self.transformer(
            ef.permute(1, 0, 2), src_key_padding_mask=src_key_padding_mask
        ).permute(1, 0, 2)

        # 7. (Optional) Pool features across the sequence dimension
        if self.config['pool_final']:
            # Masked average pooling
            masked_ef = ef * emask.unsqueeze(-1)
            ef = masked_ef.sum(dim=1) / emask.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # 8. (Optional) Apply final MLP
        if self.config['apply_mlp_b']:
            ef = self.mlp_b(ef)

        # 9. (Optional) VAE output
        if self.config['use_vae']:
            emu = self.mu_enc(ef)
            elogvar = self.logvar_enc(ef)
            return emu, elogvar, emask

        return ef, emask