# ==========================================================
# model.py
# Spatio-Temporal Air Quality Prediction Model
# Hyper-Heuristic Integrated
# ==========================================================

import torch
import torch.nn as nn

from Temporal.temporal_encoder import TemporalEncoder
from Spatial.spatial_encoder import SpatialEncoder
from Fusion.fusion_layer import FusionLayer
from Prediction.prediction_layer import PredictionLayer

from Spatial.hyper_heuristic import HyperHeuristic


class SpatioTemporalModel(nn.Module):

    def __init__(
        self,
        num_nodes,
        input_dim=14,
        embedding_dim=128,
        sequence_length=24,

        # --------------------------------------------------
        # Initial graph parameters
        # --------------------------------------------------

        decay_constant=0.5,
        similarity_threshold=0.2,

        num_heads=8,

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

            dropout=0.2,

            num_blocks=4
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

        self.fusion_layer = FusionLayer()


        # ==================================================
        # PREDICTION
        # ==================================================

        self.prediction_layer = PredictionLayer()


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
        # IMPORTANT
        # --------------------------------------------------
        #
        # Do NOT manually modify GraphBuilder here.
        #
        # SpatialEncoder owns the graph cache.
        #
        # Updating the SpatialEncoder parameters causes
        # the graph cache to be invalidated.
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


        print(
            "\n" + "=" * 70
        )

        print(
            "GRAPH PARAMETERS UPDATED"
        )

        print(
            "=" * 70
        )

        print(
            "Decay constant       :",
            self.decay_constant
        )

        print(
            "Similarity threshold :",
            self.similarity_threshold
        )

        print(
            "Spatial graph cache  : INVALIDATED"
        )

        print(
            "=" * 70
        )


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
        # OUTPUT STORAGE
        # ==================================================

        batch_spatial_embeddings = []


        # ==================================================
        # BATCH
        # ==================================================

        for batch_index in range(
            batch_size
        ):

            sample_records = (
                station_records[
                    batch_index
                ]
            )


            if len(sample_records) != (
                self.sequence_length
            ):

                raise ValueError(
                    f"Batch {batch_index} contains "
                    f"{len(sample_records)} timesteps."
                )


            timestep_embeddings = []


            # ==================================================
            # TIME
            # ==================================================

            for time_index in range(
                self.sequence_length
            ):

                stations = (
                    sample_records[
                        time_index
                    ]
                )


                if len(stations) != (
                    self.num_nodes
                ):

                    raise ValueError(
                        f"Batch {batch_index}, "
                        f"timestep {time_index}: "
                        f"expected "
                        f"{self.num_nodes} stations, "
                        f"got {len(stations)}."
                    )


                # --------------------------------------------------
                # Same feature tensor used by temporal branch.
                # --------------------------------------------------

                scaled_features = (
                    scaled_node_features[
                        batch_index,
                        time_index
                    ]
                )


                # --------------------------------------------------
                # CRITICAL:
                #
                # This timestep gets THIS timestep's station
                # conditions.
                #
                # Therefore:
                #
                #     k = constant
                #     threshold = constant
                #
                # but the similarity scores can change because
                # wind/environmental conditions change.
                #
                # Therefore Graph(t) can change.
                # --------------------------------------------------

                spatial_embedding = (
                    self.spatial_encoder(

                        stations,

                        scaled_features
                    )
                )


                expected_shape = (

                    self.num_nodes,

                    self.embedding_dim
                )


                if (
                    spatial_embedding.shape
                    !=
                    expected_shape
                ):

                    raise ValueError(
                        f"Expected spatial embedding "
                        f"{expected_shape}, "
                        f"got "
                        f"{spatial_embedding.shape}"
                    )


                timestep_embeddings.append(
                    spatial_embedding
                )


            # ==================================================
            # [T,N,E]
            # ==================================================

            sample_embedding = torch.stack(

                timestep_embeddings,

                dim=0
            )


            batch_spatial_embeddings.append(
                sample_embedding
            )


        # ==================================================
        # [B,T,N,E]
        # ==================================================

        spatial_embeddings = torch.stack(

            batch_spatial_embeddings,

            dim=0
        )


        expected_shape = (

            batch_size,

            self.sequence_length,

            self.num_nodes,

            self.embedding_dim
        )


        if (
            spatial_embeddings.shape
            !=
            expected_shape
        ):

            raise ValueError(
                f"Expected spatial embeddings "
                f"{expected_shape}, "
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
        # PREDICTION
        # ==================================================

        predictions = (
            self.prediction_layer(
                fused_embeddings
            )
        )


        from config import PREDICTION_HORIZON
        expected_output_shape = (x.shape[0], self.num_nodes, PREDICTION_HORIZON)


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
