# ==========================================================
# Temporal/new_TE.py
# TCN TEMPORAL ENCODER WITH POSITIONAL ENCODING
#
# Input:
#     [B, T, N, F]
#
# Output:
#     [B, T, N, E]
#
# ==========================================================

import math
import torch
import torch.nn as nn

from .tcn import TCN


class TemporalEncoder(nn.Module):
    """
    Temporal Encoder backed by the upgraded Temporal Convolutional Network (TCN).
    Incorporates dynamic sinusoidal positional encodings and residual connections.
    """

    def __init__(
        self,
        input_dim,
        embedding_dim,
        kernel_size=3,
        dropout=0.2,
        num_blocks=6,
        dilations=None
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # ------------------------------------------------------
        # INPUT PROJECTION
        # [B*N, F, T] -> [B*N, E, T]
        # ------------------------------------------------------
        self.input_projection = nn.Conv1d(
            in_channels=input_dim,
            out_channels=embedding_dim,
            kernel_size=1
        )

        # ------------------------------------------------------
        # TCN BACKBONE
        # [B*N, E, T] -> [B*N, E, T]
        # ------------------------------------------------------
        self.tcn = TCN(
            channels=embedding_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            num_blocks=num_blocks,
            dilations=dilations
        )

        # ------------------------------------------------------
        # LAYER NORMALIZATION
        # ------------------------------------------------------
        self.layer_norm = nn.LayerNorm(embedding_dim)

    # ==========================================================
    # INPUT VALIDATION
    # ==========================================================

    def _check_input(self, x):
        if not isinstance(x, torch.Tensor):
            raise TypeError("TemporalEncoder expected a torch.Tensor.")

        if x.dim() != 4:
            raise ValueError(
                f"TemporalEncoder expected [B,T,N,F]. Received {tuple(x.shape)}."
            )

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"TemporalEncoder expected {self.input_dim} features, got {x.shape[-1]}."
            )

        if torch.isnan(x).any():
            raise ValueError("TemporalEncoder input contains NaN values.")

        if torch.isinf(x).any():
            raise ValueError("TemporalEncoder input contains Inf values.")

    # ==========================================================
    # POSITIONAL ENCODING GENERATION
    # ==========================================================

    def _get_positional_encoding(self, sequence_length, device, dtype):
        """
        Dynamically generates sinusoidal positional encoding of shape [1, E, T].
        Supports any sequence length T without hardcoding.
        """
        pe = torch.zeros(sequence_length, self.embedding_dim, device=device, dtype=dtype)
        position = torch.arange(0, sequence_length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / self.embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe.transpose(0, 1).unsqueeze(0)

    # ==========================================================
    # FORWARD
    # ==========================================================

    def forward(self, x):
        self._check_input(x)

        batch_size, sequence_length, num_nodes, num_features = x.shape

        # [B, T, N, F] -> [B, N, T, F] -> [B*N, T, F] -> [B*N, F, T]
        x_flat = x.permute(0, 2, 1, 3).reshape(
            batch_size * num_nodes, sequence_length, num_features
        ).permute(0, 2, 1)

        # 1. Project input features: [B*N, F, T] -> [B*N, E, T]
        proj = self.input_projection(x_flat)

        # 2. Add dynamic positional encoding: [1, E, T]
        pos_enc = self._get_positional_encoding(
            sequence_length, device=proj.device, dtype=proj.dtype
        )
        tcn_in = proj + pos_enc

        # 3. Apply TCN backbone
        tcn_out = self.tcn(tcn_in)

        # 4. Residual connection from projected input
        out = tcn_out + proj

        # 5. [B*N, E, T] -> [B*N, T, E] -> LayerNorm
        out = out.permute(0, 2, 1)
        out = self.layer_norm(out)

        # 6. Restore original dimensions: [B, N, T, E] -> [B, T, N, E]
        out = out.reshape(batch_size, num_nodes, sequence_length, self.embedding_dim)
        out = out.permute(0, 2, 1, 3)

        expected_shape = (batch_size, sequence_length, num_nodes, self.embedding_dim)
        if out.shape != expected_shape:
            raise RuntimeError(
                f"TemporalEncoder output shape mismatch. Expected {expected_shape}, got {tuple(out.shape)}."
            )

        return out