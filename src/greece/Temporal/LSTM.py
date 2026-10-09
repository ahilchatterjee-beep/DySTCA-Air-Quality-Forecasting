# ==========================================================
# lstm.py
# LSTM Temporal Encoder
#
# Input:
#     [B*N, T, E]
#
# Output:
#     [B*N, T, E]
#
# This module replaces the TCN inside TemporalEncoder.
# The existing tcn.py remains untouched.
# ==========================================================

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    """
    LSTM-based temporal encoder.

    Input:
        [B*N, T, E]

    Output:
        [B*N, T, E]

    where:
        B = batch size
        N = number of stations
        T = historical sequence length
        E = embedding dimension
    """

    def __init__(
        self,
        embedding_dim,
        hidden_dim=None,
        num_layers=2,
        dropout=0.2,
        bidirectional=False
    ):
        super().__init__()

        # ------------------------------------------------------
        # Configuration
        # ------------------------------------------------------

        self.embedding_dim = embedding_dim

        if hidden_dim is None:
            hidden_dim = embedding_dim

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM dropout is only active when num_layers > 1.
        lstm_dropout = (
            dropout
            if num_layers > 1
            else 0.0
        )

        # ------------------------------------------------------
        # LSTM
        #
        # Input:
        #     [B*N, T, E]
        #
        # Output:
        #     [B*N, T, hidden_dim * directions]
        # ------------------------------------------------------

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=bidirectional
        )

        # ------------------------------------------------------
        # Output dimension
        # ------------------------------------------------------

        self.output_dim = (
            hidden_dim * 2
            if bidirectional
            else hidden_dim
        )

        # ------------------------------------------------------
        # Project back to embedding dimension
        #
        # This guarantees that the rest of the architecture
        # always receives E-dimensional temporal embeddings.
        # ------------------------------------------------------

        if self.output_dim != embedding_dim:

            self.output_projection = nn.Linear(
                self.output_dim,
                embedding_dim
            )

        else:

            self.output_projection = nn.Identity()

        # ------------------------------------------------------
        # Normalization
        #
        # Helps keep the temporal representation stable before
        # it is passed into the fusion layer.
        # ------------------------------------------------------

        self.layer_norm = nn.LayerNorm(
            embedding_dim
        )

    # ==========================================================
    # FORWARD
    # ==========================================================

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape [B*N, T, E]

        Returns
        -------
        torch.Tensor
            Shape [B*N, T, E]
        """

        # ------------------------------------------------------
        # Input validation
        # ------------------------------------------------------

        if x.dim() != 3:

            raise ValueError(
                "LSTMEncoder expects input with shape "
                "[B*N, T, E]. "
                f"Got {tuple(x.shape)}."
            )

        if x.shape[-1] != self.embedding_dim:

            raise ValueError(
                "LSTM input embedding dimension mismatch. "
                f"Expected {self.embedding_dim}, "
                f"got {x.shape[-1]}."
            )

        if torch.isnan(x).any():

            raise ValueError(
                "LSTM input contains NaN values."
            )

        if torch.isinf(x).any():

            raise ValueError(
                "LSTM input contains Inf values."
            )

        # ------------------------------------------------------
        # LSTM
        # ------------------------------------------------------

        output, _ = self.lstm(x)

        # ------------------------------------------------------
        # Restore required embedding dimension
        # ------------------------------------------------------

        output = self.output_projection(
            output
        )

        # ------------------------------------------------------
        # Layer normalization
        # ------------------------------------------------------

        output = self.layer_norm(
            output
        )

        # ------------------------------------------------------
        # Output validation
        # ------------------------------------------------------

        if output.shape != x.shape:

            raise ValueError(
                "LSTM output shape mismatch. "
                f"Expected {tuple(x.shape)}, "
                f"got {tuple(output.shape)}."
            )

        return output