import torch
import torch.nn as nn

from .new_graph_builder import GraphBuilder
from .multihead_gat import MultiHeadGAT
from config import NUM_ATTENTION_HEADS

class SpatialEncoder(nn.Module):

    """
    Spatial Encoder

    Responsibilities
    ----------------
    1. Build the spatial graph from station information.
    2. Receive already-scaled node features.
    3. Apply Multi-Head GAT.
    4. Return spatial embeddings.

    Graph behavior
    --------------

    Station locations remain fixed.

    The GLOBAL graph parameters:

        k
        similarity_threshold

    remain fixed until explicitly changed by the
    Hyper-Heuristic.

    However, wind speed may change at every timestep.

    Therefore:

        k                = constant
        threshold        = constant
        coordinates      = constant
        wind speed       = dynamic
        graph             = dynamic

    For the new Delhi dataset the GraphBuilder uses:

        Sij = Vi * exp(-k * dij)

    where Vi is the current wind speed at source
    station i.

    Therefore the graph can change from timestep to
    timestep even though k and threshold remain fixed.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        decay_constant,
        similarity_threshold,
        num_heads=NUM_ATTENTION_HEADS
    ):

        super().__init__()


        # ==================================================
        # GRAPH PARAMETERS
        # ==================================================

        self.decay_constant = float(
            decay_constant
        )

        self.similarity_threshold = float(
            similarity_threshold
        )


        # ==================================================
        # GRAPH BUILDER
        # ==================================================

        self.graph_builder = GraphBuilder(

            decay_constant=self.decay_constant,

            similarity_threshold=(
                self.similarity_threshold
            )
        )


        # ==================================================
        # MULTI-HEAD GAT
        # ==================================================

        self.multihead_gat = MultiHeadGAT(

            input_dim=input_dim,

            output_dim=output_dim,

            num_heads=num_heads
        )


        # ==================================================
        # GRAPH CACHE
        # ==================================================

        self._cached_edge_index = None

        self._cached_station_signature = None

        self._cached_distance_matrix = None


        # ==================================================
        # DIAGNOSTICS
        # ==================================================

        self._diagnostic_printed = False

        self._feature_diagnostic_printed = False

        self._gat_diagnostic_printed = False


    # ======================================================
    # STATION SIGNATURE
    # ======================================================

    @staticmethod
    def _make_station_signature(
        stations
    ):

        """
        Creates a signature containing the information
        that can affect the graph.

        Fixed:
            Station_ID
            Latitude
            Longitude

        Dynamic:
            Wind_Speed

        IMPORTANT:

        k and threshold are NOT included because they
        are model-level global parameters and are handled
        through set_graph_parameters().
        """

        signature = []


        for station in stations:

            # --------------------------------------------------
            # Wind speed
            # --------------------------------------------------

            if "Wind_Speed" in station:

                wind_speed = float(
                    station["Wind_Speed"]
                )

            elif "windspeed_kph" in station:

                wind_speed = float(
                    station["windspeed_kph"]
                )

            # --------------------------------------------------
            # Backward compatibility
            # --------------------------------------------------

            elif (
                "Wind_U" in station
                and
                "Wind_V" in station
            ):

                wind_u = float(
                    station["Wind_U"]
                )

                wind_v = float(
                    station["Wind_V"]
                )

                wind_speed = (

                    wind_u ** 2

                    +

                    wind_v ** 2
                ) ** 0.5

            else:

                raise KeyError(

                    "Station record must contain "
                    "'Wind_Speed' or "
                    "'windspeed_kph'."
                )


            signature.append(

                (
                    station["Station_ID"],

                    float(
                        station["Latitude"]
                    ),

                    float(
                        station["Longitude"]
                    ),

                    wind_speed
                )
            )


        return tuple(
            signature
        )


    # ======================================================
    # UPDATE GRAPH PARAMETERS
    # ======================================================

    def set_graph_parameters(
        self,
        decay_constant=None,
        similarity_threshold=None
    ):

        """
        Update GLOBAL graph parameters.

        These values are constant across:

            - all stations
            - all station pairs
            - all timesteps

        until the Hyper-Heuristic explicitly changes them.

        Changing either parameter invalidates the graph
        cache because the current edge structure may no
        longer be valid.
        """

        if decay_constant is not None:

            self.decay_constant = float(
                decay_constant
            )


        if similarity_threshold is not None:

            self.similarity_threshold = float(
                similarity_threshold
            )


        # --------------------------------------------------
        # Update GraphBuilder
        # --------------------------------------------------

        self.graph_builder.k = (
            self.decay_constant
        )

        self.graph_builder.similarity_threshold = (
            self.similarity_threshold
        )


        # --------------------------------------------------
        # Graph must be rebuilt
        # --------------------------------------------------

        self._cached_edge_index = None

        self._cached_station_signature = None


        # --------------------------------------------------
        # Diagnostics can run again
        # --------------------------------------------------

        self._diagnostic_printed = False


    # ======================================================
    # GRAPH
    # ======================================================

    def _get_graph(
        self,
        stations,
        device
    ):

        """
        Return the current dynamic graph.

        First timestep:
            build_graph()

        Later timestep:

            if station conditions unchanged:
                reuse graph

            if wind speed changed:
                update_graph()

        Therefore:

            fixed coordinates
            +
            changing wind speed
            =
            dynamic graph
        """

        # ==================================================
        # CURRENT STATION SIGNATURE
        # ==================================================

        current_signature = (

            self._make_station_signature(
                stations
            )
        )


        # ==================================================
        # FIRST GRAPH
        # ==================================================

        if (
            self._cached_edge_index
            is None
        ):

            edge_index = (

                self.graph_builder.build_graph(
                    stations
                )
            )


            if not isinstance(
                edge_index,
                torch.Tensor
            ):

                edge_index = torch.tensor(

                    edge_index,

                    dtype=torch.long
                )


            edge_index = (
                edge_index.long()
            )


            # --------------------------------------------------
            # Cache graph
            # --------------------------------------------------

            self._cached_edge_index = (

                edge_index.detach().cpu()
            )


            self._cached_station_signature = (
                current_signature
            )


            return edge_index.to(
                device
            )


        # ==================================================
        # CHECK WHETHER CONDITIONS CHANGED
        # ==================================================

        if (

            self._cached_station_signature

            ==

            current_signature

        ):

            # --------------------------------------------------
            # Nothing affecting the graph changed.
            # --------------------------------------------------

            return self._cached_edge_index.to(
                device
            )


        # ==================================================
        # DYNAMIC GRAPH UPDATE
        # ==================================================

        previous_edge_index = (

            self._cached_edge_index
        )


        updated_edge_index = (

            self.graph_builder.update_graph(

                previous_edge_index,

                stations
            )
        )


        if not isinstance(
            updated_edge_index,
            torch.Tensor
        ):

            updated_edge_index = torch.tensor(

                updated_edge_index,

                dtype=torch.long
            )


        updated_edge_index = (
            updated_edge_index.long()
        )


        # ==================================================
        # DIAGNOSTIC
        # ==================================================

        previous_edges = (
            previous_edge_index.shape[1]
        )

        updated_edges = (
            updated_edge_index.shape[1]
        )


        


        # ==================================================
        # UPDATE CACHE
        # ==================================================

        self._cached_edge_index = (

            updated_edge_index.detach().cpu()
        )


        self._cached_station_signature = (
            current_signature
        )


        return updated_edge_index.to(
            device
        )


    # ======================================================
    # GRAPH DIAGNOSTICS
    # ======================================================

    def diagnose_graph(
        self,
        stations,
        edge_index
    ):

        num_nodes = len(
            stations
        )

        num_edges = (
            edge_index.shape[1]
        )


        print("\n")

        print("=" * 70)

        print(
            "SPATIAL GRAPH DIAGNOSTIC"
        )

        print("=" * 70)


        print(
            "Decay constant       :",
            self.decay_constant
        )

        print(
            "Similarity threshold :",
            self.similarity_threshold
        )

        print(
            "Number of nodes      :",
            num_nodes
        )

        print(
            "Number of edges      :",
            num_edges
        )


        if num_edges == 0:

            print(
                "\nWARNING: GRAPH HAS ZERO EDGES!"
            )

            print(
                "=" * 70
            )

            return


        # ==================================================
        # INCOMING
        # ==================================================

        incoming_edges = torch.bincount(

            edge_index[1],

            minlength=num_nodes
        )


        print(
            "\nIncoming edges per station:"
        )

        print(
            incoming_edges
        )


        zero_incoming = (
            incoming_edges == 0
        )


        print(
            "\nStations with zero incoming edges:",
            zero_incoming.sum().item()
        )


        if zero_incoming.any():

            print(
                "Zero-incoming station indices:"
            )

            print(
                torch.where(
                    zero_incoming
                )[0]
            )


        # ==================================================
        # OUTGOING
        # ==================================================

        outgoing_edges = torch.bincount(

            edge_index[0],

            minlength=num_nodes
        )


        print(
            "\nOutgoing edges per station:"
        )

        print(
            outgoing_edges
        )


        zero_outgoing = (
            outgoing_edges == 0
        )


        print(
            "\nStations with zero outgoing edges:",
            zero_outgoing.sum().item()
        )


        if zero_outgoing.any():

            print(
                "Zero-outgoing station indices:"
            )

            print(
                torch.where(
                    zero_outgoing
                )[0]
            )


        # ==================================================
        # SELF LOOPS
        # ==================================================

        source_nodes = (
            edge_index[0]
        )

        target_nodes = (
            edge_index[1]
        )


        self_loops = (

            source_nodes

            ==

            target_nodes
        )


        num_self_loops = (
            self_loops.sum().item()
        )


        print(
            "\nSelf-loops:"
        )

        print(
            "Number of self-loops:",
            num_self_loops
        )


        if num_self_loops > 0:

            print(
                "Self-loop node indices:"
            )

            print(
                source_nodes[
                    self_loops
                ]
            )

        else:

            print(
                "WARNING: No self-loops found."
            )


        print(
            "\nEdge index:"
        )

        print(
            edge_index
        )


        print(
            "\n" + "=" * 70
        )


    # ======================================================
    # NODE FEATURE DIAGNOSTICS
    # ======================================================

    def diagnose_node_features(
        self,
        scaled_features
    ):

        print("\n")

        print("=" * 70)

        print(
            "SPATIAL GAT INPUT DIAGNOSTIC"
        )

        print("=" * 70)


        print(
            "Shape:",
            scaled_features.shape
        )

        print(
            "Mean :",
            scaled_features.mean().item()
        )

        print(
            "Std  :",
            scaled_features.std().item()
        )

        print(
            "Min  :",
            scaled_features.min().item()
        )

        print(
            "Max  :",
            scaled_features.max().item()
        )

        print(
            "NaNs :",
            torch.isnan(
                scaled_features
            ).sum().item()
        )

        print(
            "Infs :",
            torch.isinf(
                scaled_features
            ).sum().item()
        )

        print(
            "=" * 70
        )


    # ======================================================
    # GAT OUTPUT DIAGNOSTICS
    # ======================================================

    def diagnose_gat_output(
        self,
        spatial_embeddings
    ):

        print("\n")

        print("=" * 70)

        print(
            "GAT OUTPUT DIAGNOSTIC"
        )

        print("=" * 70)


        print(
            "Shape:",
            spatial_embeddings.shape
        )

        print(
            "Mean :",
            spatial_embeddings.mean().item()
        )

        print(
            "Std  :",
            spatial_embeddings.std().item()
        )

        print(
            "Min  :",
            spatial_embeddings.min().item()
        )

        print(
            "Max  :",
            spatial_embeddings.max().item()
        )

        print(
            "NaNs :",
            torch.isnan(
                spatial_embeddings
            ).sum().item()
        )

        print(
            "Infs :",
            torch.isinf(
                spatial_embeddings
            ).sum().item()
        )

        print(
            "=" * 70
        )


    # ======================================================
    # FORWARD
    # ======================================================

    def forward(
        self,
        stations,
        scaled_features
    ):

        # ==================================================
        # INPUT CHECK
        # ==================================================

        if not isinstance(
            scaled_features,
            torch.Tensor
        ):

            raise TypeError(
                "scaled_features must be "
                "a torch.Tensor."
            )


        if scaled_features.dim() == 4:
            # ==================================================
            # VECTORIZED BATCH PATH: [B, T, N, F]
            # Computes dynamic graph and GAT across all B*T in parallel
            # ==================================================
            B, T, N, F = scaled_features.shape

            # 1. Distance matrix between N stations
            if self._cached_distance_matrix is None or self._cached_distance_matrix.size(0) != N:
                sample_stations = stations[0][0] if isinstance(stations[0], (list, tuple)) else stations[0]
                self._cached_distance_matrix = self.graph_builder.compute_distance_matrix(sample_stations)

            # 2. Extract wind speeds [B, T, N]
            wind_speeds = torch.tensor(
                [[[float(s.get("Wind_Speed", s.get("windspeed_kph", 0.0))) for s in t_step] for t_step in b_sample] for b_sample in stations],
                dtype=scaled_features.dtype,
                device=scaled_features.device
            )

            # 3. Vectorized dynamic adjacency matrix: [B, T, N, N]
            adj = self.graph_builder.compute_batched_adjacency(
                self._cached_distance_matrix,
                wind_speeds
            )

            # 4. Multi-Head GAT over [B*T, N, F]
            features_flat = scaled_features.reshape(B * T, N, F)
            adj_flat = adj.reshape(B * T, N, N)

            spatial_embeddings_flat = self.multihead_gat(features_flat, adj=adj_flat)
            spatial_embeddings = spatial_embeddings_flat.reshape(B, T, N, -1)

            return spatial_embeddings

        if scaled_features.dim() != 2:

            raise ValueError(

                "Expected scaled_features "
                "with shape [N, F] or [B, T, N, F], "

                f"but got "
                f"{scaled_features.shape}"
            )


        if (
            scaled_features.shape[0]
            !=
            len(stations)
        ):

            raise ValueError(

                f"Number of nodes in "
                f"scaled_features "
                f"({scaled_features.shape[0]}) "
                f"does not match number "
                f"of stations "
                f"({len(stations)})."
            )


        # ==================================================
        # NUMERICAL CHECK
        # ==================================================

        if torch.isnan(
            scaled_features
        ).any():

            raise ValueError(
                "scaled_features contains "
                "NaN values."
            )


        if torch.isinf(
            scaled_features
        ).any():

            raise ValueError(
                "scaled_features contains "
                "Inf values."
            )


        # ==================================================
        # GRAPH
        # ==================================================

        edge_index = self._get_graph(

            stations,

            scaled_features.device
        )


        # ==================================================
        # GRAPH DIAGNOSTIC
        # ==================================================

        if not self._diagnostic_printed:

            self.diagnose_graph(

                stations,

                edge_index
            )

            self._diagnostic_printed = True


        # ==================================================
        # FEATURE DIAGNOSTIC
        # ==================================================

        if not self._feature_diagnostic_printed:

            self.diagnose_node_features(

                scaled_features
            )

            self._feature_diagnostic_printed = True


        # ==================================================
        # GAT
        # ==================================================

        spatial_embeddings = (

            self.multihead_gat(

                scaled_features,

                edge_index
            )
        )


        # ==================================================
        # OUTPUT DIAGNOSTIC
        # ==================================================

        if not self._gat_diagnostic_printed:

            self.diagnose_gat_output(

                spatial_embeddings
            )

            self._gat_diagnostic_printed = True


        # ==================================================
        # RETURN
        # ==================================================

        return spatial_embeddings