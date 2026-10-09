# ==========================================================
# temporal_only_model.py
# Temporal-Only Baseline Model
# ==========================================================

import torch
import torch.nn as nn

from Temporal.temporal_encoder import TemporalEncoder
from Prediction.prediction_layer import PredictionLayer


class TemporalOnlyModel(nn.Module):

    def __init__(
        self,
        num_nodes,
        input_dim=14,
        embedding_dim=128,
        sequence_length=24
    ):

        super().__init__()

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.sequence_length = sequence_length

        # --------------------------------------------------
        # TEMPORAL ENCODER
        # --------------------------------------------------

        self.temporal_encoder = TemporalEncoder(
            input_dim=self.input_dim,
            embedding_dim=self.embedding_dim,
            kernel_size=3,
            dropout=0.2,
            num_blocks=4
        )

        # --------------------------------------------------
        # PREDICTION
        # --------------------------------------------------

        self.prediction_layer = PredictionLayer()

    # ======================================================
    # INPUT CHECK
    # ======================================================

    def check_input(self, x):

        if x.dim() != 4:
            raise ValueError(
                f"Expected input [B,T,N,F], got {x.shape}"
            )

        batch_size, sequence_length, num_nodes, num_features = x.shape

        if sequence_length != self.sequence_length:
            raise ValueError(
                f"Expected sequence length "
                f"{self.sequence_length}, got {sequence_length}"
            )

        if num_nodes != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} nodes, got {num_nodes}"
            )

        if num_features != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features, got {num_features}"
            )

        if torch.isnan(x).any():
            raise ValueError("Input contains NaN values.")

        if torch.isinf(x).any():
            raise ValueError("Input contains Inf values.")

    # ======================================================
    # FORWARD
    # ======================================================

    def forward(self, x):

        self.check_input(x)

        # --------------------------------------------------
        # TEMPORAL ONLY
        # --------------------------------------------------

        temporal_embeddings = self.temporal_encoder(x)

        expected_shape = (
            x.shape[0],
            self.sequence_length,
            self.num_nodes,
            self.embedding_dim
        )

        if temporal_embeddings.shape != expected_shape:
            raise ValueError(
                f"Expected temporal embeddings "
                f"{expected_shape}, "
                f"got {temporal_embeddings.shape}"
            )

        # --------------------------------------------------
        # DIRECTLY PREDICT FROM TEMPORAL EMBEDDINGS
        # --------------------------------------------------

        predictions = self.prediction_layer(
            temporal_embeddings
        )

        expected_output_shape = (
            x.shape[0],
            self.num_nodes,
            1
        )

        if predictions.shape != expected_output_shape:
            raise ValueError(
                f"Expected prediction shape "
                f"{expected_output_shape}, "
                f"got {predictions.shape}"
            )

        return predictions