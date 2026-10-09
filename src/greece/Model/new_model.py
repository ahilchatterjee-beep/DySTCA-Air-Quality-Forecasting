# ==========================================================
# model.py
# Spatio-Temporal Air Quality Prediction Model
# Hyper-Heuristic Integrated
# ==========================================================


import torch
import torch.nn as nn

from config import (
    EMBEDDING_DIM,
    NUM_ATTENTION_HEADS,
    SEQUENCE_LENGTH,
    PREDICTION_HORIZON,
    NUM_TCN_BLOCKS,
    NUM_STATIONS,
    INPUT_DIM
)

from Temporal.new_TE import TemporalEncoder

from Spatial.new_spatial_encoder import SpatialEncoder

from Fusion.fusion_layer import FusionLayer

from Prediction.prediction_layer import PredictionLayer

from Spatial.hyper_heuristic import HyperHeuristic

from .revin import RevIN


class SpatioTemporalModel(nn.Module):

    def __init__(
        self,
        num_nodes=NUM_STATIONS,
        input_dim=INPUT_DIM,
        embedding_dim=EMBEDDING_DIM,
        sequence_length=SEQUENCE_LENGTH,
        prediction_horizon=PREDICTION_HORIZON,
        # --------------------------------------------------
        # Initial graph parameters
        # --------------------------------------------------

        decay_constant=0.5,

        similarity_threshold=0.2,

        num_heads=NUM_ATTENTION_HEADS,

        # --------------------------------------------------
        # Hyper-Heuristic
        # --------------------------------------------------

        use_hyper_heuristic=True,

        decay_range=(0.01, 1.0),

        similarity_threshold_range=(
            0.01,
            1.0
        ),

        num_candidates=10,

        exploration_probability=0.30,

        exploitation_fraction=0.15
    ):

        super().__init__()


        # ==================================================
        # BASIC CONFIGURATION
        # ==================================================

        self.num_nodes = num_nodes

        self.input_dim = input_dim

        self.embedding_dim = embedding_dim

        self.sequence_length = sequence_length

        self.prediction_horizon = prediction_horizon

        # ==================================================
        # GRAPH PARAMETERS
        # ==================================================

        # These are GLOBAL parameters.
        #
        # They are shared across:
        #
        #   - all stations
        #   - all station pairs
        #   - all timesteps
        #
        # The graph itself can still change with time because
        # the similarity score depends on time-varying data.

        self.decay_constant = float(
            decay_constant
        )

        self.similarity_threshold = float(
            similarity_threshold
        )


        # ==================================================
        # HYPER-HEURISTIC
        # ==================================================

        self.use_hyper_heuristic = (
            use_hyper_heuristic
        )

        self.hyper_heuristic = None


        if self.use_hyper_heuristic:

            self.hyper_heuristic = HyperHeuristic(

                decay_range=(
                    decay_range
                ),

                similarity_threshold_range=(
                    similarity_threshold_range
                ),

                num_candidates=(
                    num_candidates
                ),

                exploration_probability=(
                    exploration_probability
                ),

                exploitation_fraction=(
                    exploitation_fraction
                )
            )


        # ==================================================
        # TEMPORAL ENCODER
        # ==================================================

        self.temporal_encoder = TemporalEncoder(

            input_dim=self.input_dim,

            embedding_dim=self.embedding_dim,

            kernel_size=3,

            dropout=0.25,

            num_blocks=NUM_TCN_BLOCKS
        )


        # ==================================================
        # SPATIAL ENCODER
        # ==================================================

        self.spatial_encoder = SpatialEncoder(

            input_dim=self.input_dim,

            output_dim=self.embedding_dim,

            decay_constant=(
                self.decay_constant
            ),

            similarity_threshold=(
                self.similarity_threshold
            ),

            num_heads=num_heads
        )


        # ==================================================
        # FUSION
        # ==================================================

        # Explicitly pass the embedding dimension so that
        # FusionLayer cannot silently use a different default.

        self.fusion_layer = FusionLayer(

            embedding_dim=self.embedding_dim,
            dropout=0.2
        )


        # ==================================================
        # PREDICTION
        # ==================================================

        self.prediction_layer = PredictionLayer(
            embedding_dim=self.embedding_dim,
            output_dim=self.prediction_horizon,
            dropout=0.25
        )


        # ==================================================
        # REVERSIBLE INSTANCE NORMALIZATION (RevIN)
        # ==================================================

        self.revin = RevIN(
            num_nodes=self.num_nodes,
            eps=1e-5,
            affine=False
        )


        # ==================================================
        # STATUS
        # ==================================================

        self.hyper_heuristic_optimized = False


    # ======================================================
    # INPUT CHECK
    # ======================================================

    def check_input(
        self,
        x
    ):

        if x.dim() != 4:

            raise ValueError(

                f"Expected input [B,T,N,F], "
                f"got {x.shape}"
            )


        (
            batch_size,
            sequence_length,
            num_nodes,
            num_features
        ) = x.shape


        if sequence_length != (
            self.sequence_length
        ):

            raise ValueError(

                f"Expected sequence length "
                f"{self.sequence_length}, "
                f"got {sequence_length}"
            )


        if num_nodes != self.num_nodes:

            raise ValueError(

                f"Expected {self.num_nodes} nodes, "
                f"got {num_nodes}"
            )


        if num_features != self.input_dim:

            raise ValueError(

                f"Expected {self.input_dim} features, "
                f"got {num_features}"
            )


        if torch.isnan(x).any():

            raise ValueError(
                "Input contains NaN values."
            )


        if torch.isinf(x).any():

            raise ValueError(
                "Input contains Inf values."
            )


    # ======================================================
    # HYPER-HEURISTIC PARAMETER UPDATE
    # ======================================================

    def set_graph_parameters(
        self,
        decay_constant,
        similarity_threshold
    ):

        """
        Update the GLOBAL graph parameters selected
        by the Hyper-Heuristic.

        These values remain constant until another
        HH candidate is explicitly evaluated/applied.

        SpatialEncoder owns the graph cache and is
        responsible for invalidating it.
        """

        self.decay_constant = float(
            decay_constant
        )

        self.similarity_threshold = float(
            similarity_threshold
        )


        # --------------------------------------------------
        # SpatialEncoder owns the graph.
        # --------------------------------------------------

        self.spatial_encoder.set_graph_parameters(

            decay_constant=(
                self.decay_constant
            ),

            similarity_threshold=(
                self.similarity_threshold
            )
        )


        self.hyper_heuristic_optimized = True


       


    # ======================================================
    # APPLY HYPER-HEURISTIC CANDIDATE
    # ======================================================

    def apply_hyper_heuristic_candidate(
        self,
        candidate
    ):

        if candidate is None:

            raise ValueError(
                "Hyper-heuristic candidate is None."
            )


        if "decay_constant" not in candidate:

            raise KeyError(
                "Candidate missing "
                "'decay_constant'."
            )


        if "similarity_threshold" not in candidate:

            raise KeyError(
                "Candidate missing "
                "'similarity_threshold'."
            )


        self.set_graph_parameters(

            decay_constant=(
                candidate[
                    "decay_constant"
                ]
            ),

            similarity_threshold=(
                candidate[
                    "similarity_threshold"
                ]
            )
        )


    # ======================================================
    # RUN HYPER-HEURISTIC
    # ======================================================

    def optimize_graph_parameters(
        self,
        evaluation_function
    ):

        """
        Run the Hyper-Heuristic OUTSIDE forward().

        The Hyper-Heuristic searches for one GLOBAL pair:

            k
            threshold

        The evaluation function is responsible for:

            1. Applying candidate parameters.
            2. Training the candidate model.
            3. Evaluating validation predictions.
            4. Returning validation R².

        Higher validation R² is better.

        After optimization, the best candidate is applied
        and remains fixed for subsequent training.
        """


        # --------------------------------------------------
        # HH DISABLED
        # --------------------------------------------------

        if not self.use_hyper_heuristic:

            return (

                {
                    "decay_constant":
                        self.decay_constant,

                    "similarity_threshold":
                        self.similarity_threshold
                },

                None
            )


        # --------------------------------------------------
        # CHECK EVALUATION FUNCTION
        # --------------------------------------------------

        if evaluation_function is None:

            raise ValueError(
                "evaluation_function is required."
            )


        print(
            "\n" + "=" * 70
        )

        print(
            "STARTING HYPER-HEURISTIC OPTIMIZATION"
        )

        print(
            "=" * 70
        )


        # --------------------------------------------------
        # RUN HH
        # --------------------------------------------------

        (
            best_candidate,
            best_score
        ) = self.hyper_heuristic.optimize(

            evaluation_function
        )


        # --------------------------------------------------
        # CHECK RESULT
        # --------------------------------------------------

        if best_candidate is None:

            raise RuntimeError(
                "Hyper-Heuristic returned no candidate."
            )


        # --------------------------------------------------
        # APPLY BEST PARAMETERS
        # --------------------------------------------------

        self.apply_hyper_heuristic_candidate(
            best_candidate
        )


        # --------------------------------------------------
        # FINAL HH RESULT
        # --------------------------------------------------

        print(
            "\n" + "=" * 70
        )

        print(
            "HYPER-HEURISTIC COMPLETE"
        )

        print(
            "=" * 70
        )

        print(
            "Best decay constant       :",
            best_candidate[
                "decay_constant"
            ]
        )

        print(
            "Best similarity threshold :",
            best_candidate[
                "similarity_threshold"
            ]
        )

        print(
            "Best validation R²        :",
            best_score
        )

        print(
            "=" * 70
        )


        return (

            best_candidate,

            best_score
        )


    # ======================================================
    # TEMPORAL ENCODING
    # ======================================================

    def encode_temporal(
        self,
        x
    ):

        temporal_embeddings = (

            self.temporal_encoder(x)
        )


        expected_shape = (

            x.shape[0],

            self.sequence_length,

            self.num_nodes,

            self.embedding_dim
        )


        if (
            temporal_embeddings.shape
            !=
            expected_shape
        ):

            raise ValueError(

                f"Expected temporal embeddings "
                f"{expected_shape}, "
                f"got {temporal_embeddings.shape}"
            )


        return temporal_embeddings


    # ======================================================
    # SPATIAL ENCODING
    # ======================================================

    def encode_spatial(
        self,
        station_records,
        scaled_node_features
    ):

        """
        Input:

            station_records:
                [B][T][N]

            scaled_node_features:
                [B,T,N,F]

        Output:

            [B,T,N,E]

        IMPORTANT:

        Every timestep receives its own station
        records.

        Therefore the graph can update according
        to the current timestep's environmental
        conditions.

        The GLOBAL k and threshold remain fixed.
        """


        # ==================================================
        # INPUT TYPE
        # ==================================================

        if not isinstance(
            scaled_node_features,
            torch.Tensor
        ):

            raise TypeError(
                "scaled_node_features "
                "must be a Tensor."
            )


        # ==================================================
        # INPUT DIMENSIONS
        # ==================================================

        if scaled_node_features.dim() != 4:

            raise ValueError(

                "Expected [B,T,N,F], "
                f"got {scaled_node_features.shape}"
            )


        (
            batch_size,
            sequence_length,
            num_nodes,
            num_features
        ) = scaled_node_features.shape


        # ==================================================
        # VALIDATE SHAPE
        # ==================================================

        if sequence_length != (
            self.sequence_length
        ):

            raise ValueError(

                f"Expected "
                f"{self.sequence_length} timesteps, "
                f"got {sequence_length}"
            )


        if num_nodes != self.num_nodes:

            raise ValueError(

                f"Expected "
                f"{self.num_nodes} nodes, "
                f"got {num_nodes}"
            )


        if num_features != self.input_dim:

            raise ValueError(

                f"Expected "
                f"{self.input_dim} features, "
                f"got {num_features}"
            )


        if len(station_records) != batch_size:

            raise ValueError(

                f"Expected {batch_size} "
                f"station-record sequences, "
                f"got {len(station_records)}"
            )


        # ==================================================
        # VECTORIZED SPATIAL ENCODING (ALL B*T AT ONCE)
        # ==================================================

        spatial_embeddings = self.spatial_encoder(
            station_records,
            scaled_node_features
        )

        expected_shape = (
            batch_size,
            self.sequence_length,
            self.num_nodes,
            self.embedding_dim
        )

        if spatial_embeddings.shape != expected_shape:
            raise ValueError(
                f"Expected spatial embeddings {expected_shape}, "
                f"got {spatial_embeddings.shape}"
            )

        return spatial_embeddings


    # ======================================================
    # FORWARD
    # ======================================================

    def forward(
        self,
        x,
        station_records
    ):

        """
        Complete model:

            Input
              │
              ├──────────────┐
              │              │
              ▼              ▼
          Temporal        Spatial
              │              │
              │          Graph(t)
              │              │
              │             GAT
              │              │
              └──────┬───────┘
                     ▼
                   Fusion
                     │
                     ▼
                 Prediction

        Hyper-Heuristic is NOT executed inside forward().

        The HH is executed externally before final training
        and the selected global graph parameters are then
        kept fixed.
        """


        # ==================================================
        # INPUT
        # ==================================================

        self.check_input(x)


        # ==================================================
        # TEMPORAL
        # ==================================================

        temporal_embeddings = (

            self.encode_temporal(x)
        )


        # ==================================================
        # SPATIAL
        # ==================================================

        spatial_embeddings = (

            self.encode_spatial(

                station_records,

                x
            )
        )


        expected_embedding_shape = (

            x.shape[0],

            self.sequence_length,

            self.num_nodes,

            self.embedding_dim
        )


        if (
            temporal_embeddings.shape
            !=
            expected_embedding_shape
        ):

            raise ValueError(

                f"Temporal embedding mismatch: "
                f"{temporal_embeddings.shape}"
            )


        if (
            spatial_embeddings.shape
            !=
            expected_embedding_shape
        ):

            raise ValueError(

                f"Spatial embedding mismatch: "
                f"{spatial_embeddings.shape}"
            )


        # ==================================================
        # FUSION
        # ==================================================

        fused_embeddings = (

            self.fusion_layer(

                temporal_embeddings,

                spatial_embeddings
            )
        )


        if (
            fused_embeddings.shape
            !=
            expected_embedding_shape
        ):

            raise ValueError(

                f"Fusion output mismatch: "
                f"{fused_embeddings.shape}"
            )


        # ==================================================
        # PREDICTION WITH REVIN ADAPTIVE NORMALIZATION
        # ==================================================

        # x[:, :, :, 0] is the historical PM2.5 sequence [B, T, N]
        pm25_raw = x[:, :, :, 0]
        pm25_norm, pm25_mean, pm25_std = self.revin.normalize(pm25_raw)

        # Predict normalized future trajectory [B, N, H]
        norm_predictions = self.prediction_layer(
            fused_embeddings,
            historical_pm25=pm25_norm
        )

        # Denormalize predictions back to input scale via RevIN
        predictions = self.revin.denormalize(
            norm_predictions,
            pm25_mean,
            pm25_std
        )


        expected_output_shape = (
            x.shape[0],
            self.num_nodes,
            self.prediction_horizon
        )


        if (
            predictions.shape
            !=
            expected_output_shape
        ):

            raise ValueError(

                f"Expected prediction shape "
                f"{expected_output_shape}, "
                f"got {predictions.shape}"
            )


        return predictions