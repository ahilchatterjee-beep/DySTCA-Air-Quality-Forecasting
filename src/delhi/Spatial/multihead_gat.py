import torch
import torch.nn as nn

from .gat_layer import GATLayer


class MultiHeadGAT(nn.Module):
    """
    Multi-Head Graph Attention Network — Concat+Project aggregation.

    Responsibilities
    ----------------
    1. Execute num_heads GAT heads in parallel, each producing
       (output_dim // num_heads) features.
    2. Concatenate head outputs → output_dim.
    3. Add input residual (projected if input_dim ≠ output_dim).
    4. Apply LayerNorm.

    WHY CONCAT INSTEAD OF MEAN
    ---------------------------
    Averaging head outputs forces every head to predict in the
    same subspace.  Concatenation gives each head an independent
    (output_dim // num_heads)-dimensional subspace, preserving
    the full expressive capacity of multi-head attention.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.num_heads = num_heads
        self.output_dim = output_dim

        # Each head produces head_dim features.
        # Concatenated over num_heads → output_dim.
        assert output_dim % num_heads == 0, (
            f"output_dim ({output_dim}) must be divisible by "
            f"num_heads ({num_heads})"
        )
        self.head_dim = output_dim // num_heads

        self.gat_heads = nn.ModuleList(
            [
                GATLayer(
                    input_dim,
                    self.head_dim     # each head: [*, N, head_dim]
                )
                for _ in range(num_heads)
            ]
        )

        # Project concatenated heads back to output_dim
        # (identity when output_dim == num_heads * head_dim, but
        # kept explicit for clarity and future flexibility)
        self.out_proj = nn.Linear(
            num_heads * self.head_dim,
            output_dim,
            bias=False
        )

        if input_dim != output_dim:
            self.res_proj = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.res_proj = nn.Identity()

        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(p=dropout)

    # ---------------------------------------------------------
    # Forward Pass
    # ---------------------------------------------------------

    def forward(
        self,
        x,
        edge_index=None,
        adj=None
    ):
        """
        Forward propagation.
        Supports both batched adjacency [..., N, N] and edge_index.

        x          : [..., N, input_dim]
        edge_index : [2, num_edges]   (optional)
        adj        : [..., N, N]      (optional)
        returns    : [..., N, output_dim]
        """

        head_outputs = []

        for gat_head in self.gat_heads:
            head_output = gat_head(
                x,
                edge_index=edge_index,
                adj=adj
            )                         # [..., N, head_dim]
            head_outputs.append(head_output)

        # Concatenate along the feature dimension
        # [..., N, num_heads * head_dim]
        concatenated = torch.cat(head_outputs, dim=-1)

        # Project to output_dim
        output = self.out_proj(concatenated)   # [..., N, output_dim]

        output = self.dropout(output)

        # Residual connection from input features + LayerNorm
        residual = self.res_proj(x)
        output = self.layer_norm(output + residual)

        return output