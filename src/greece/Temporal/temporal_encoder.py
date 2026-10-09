import torch.nn as nn

from .tcn import TCN


class TemporalEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        embedding_dim,
        kernel_size,
        dropout,
        num_blocks
    ):
        super().__init__()

        # ---------------------------------------------------------
        # Input Projection
        # ---------------------------------------------------------
        #
        # [B*N, input_dim, T]
        #          ↓
        # [B*N, embedding_dim, T]
        #
        # Example:
        # 14 → 64
        #

        self.input_projection = nn.Conv1d(
            in_channels=input_dim,
            out_channels=embedding_dim,
            kernel_size=1
        )

        # ---------------------------------------------------------
        # Temporal Convolutional Network
        # ---------------------------------------------------------

        self.tcn = TCN(
            channels=embedding_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            num_blocks=num_blocks
        )

    # ---------------------------------------------------------
    # Forward Pass
    # ---------------------------------------------------------

    def forward(self, x):

        # -----------------------------------------------------
        # Input:
        #
        # [B, T, N, F]
        #
        # Example:
        # [B, 24, 10, 14]
        # -----------------------------------------------------

        batch_size, sequence_length, num_nodes, num_features = x.shape

        # -----------------------------------------------------
        # Move nodes next to batch
        #
        # [B, T, N, F]
        #        ↓
        # [B, N, T, F]
        # -----------------------------------------------------

        x = x.permute(
            0,
            2,
            1,
            3
        )

        # -----------------------------------------------------
        # Combine batch and nodes
        #
        # [B, N, T, F]
        #        ↓
        # [B*N, T, F]
        # -----------------------------------------------------

        x = x.reshape(
            batch_size * num_nodes,
            sequence_length,
            num_features
        )

        # -----------------------------------------------------
        # Prepare for Conv1D
        #
        # [B*N, T, F]
        #        ↓
        # [B*N, F, T]
        # -----------------------------------------------------

        x = x.permute(
            0,
            2,
            1
        )

        # -----------------------------------------------------
        # Input Projection
        #
        # [B*N, 14, T]
        #        ↓
        # [B*N, 64, T]
        # -----------------------------------------------------

        x = self.input_projection(x)

        # -----------------------------------------------------
        # Temporal Convolutional Network
        #
        # [B*N, 64, T]
        #        ↓
        # [B*N, 64, T]
        # -----------------------------------------------------

        x = self.tcn(x)

        # -----------------------------------------------------
        # Restore temporal dimension order
        #
        # [B*N, 64, T]
        #        ↓
        # [B*N, T, 64]
        # -----------------------------------------------------

        x = x.permute(
            0,
            2,
            1
        )

        # -----------------------------------------------------
        # Restore node dimension
        #
        # [B*N, T, 64]
        #        ↓
        # [B, N, T, 64]
        # -----------------------------------------------------

        x = x.reshape(
            batch_size,
            num_nodes,
            sequence_length,
            -1
        )

        # -----------------------------------------------------
        # Final output
        #
        # [B, N, T, 64]
        #        ↓
        # [B, T, N, 64]
        #
        # Temporal embedding
        # -----------------------------------------------------

        x = x.permute(
            0,
            2,
            1,
            3
        )

        return x